import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
from utils import initialize_charset
import numpy as np
import torch.nn.utils.parametrize as parametrize
# from memory_profiler import profile


class HebbianLinear(nn.Linear):
    def __init__(self, in_features, out_features, charset, bias=True, normalize=True, clip_weights=False, updater='dfa', requires_grad=False, is_last_layer=False, plast_clip=1, batch_size=1, forget_rate=0.7, plast_proportion=0.2):
        super(HebbianLinear, self).__init__(in_features, out_features, bias)

        # Set requires_grad for the base class parameters
        self.weight.requires_grad = False # Base weights are not trained directly
        if bias:
            # For backprop/bptt, bias needs requires_grad=True so PyTorch computes bias.grad
            # For DFA, we set it to False since we handle bias manually
            self.bias.requires_grad = (updater in ['backprop', 'bptt'])

        self.normalize = normalize
        self.clip_weights = clip_weights
        self.updater = updater
        self.is_last_layer = is_last_layer
        self.in_traces = nn.Parameter(torch.zeros(batch_size, in_features), requires_grad=requires_grad)
        self.out_traces = nn.Parameter(torch.zeros(batch_size, out_features), requires_grad=requires_grad)

        self.last_high_plast_update_norm = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.last_low_plast_update_norm = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.register_buffer('t', torch.tensor(1.0))
        self.batch_size = batch_size


        # Initialize weights with the adjusted gain
        self.feedback_weights = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(len(charset), out_features)), requires_grad=requires_grad)
        # self.weight.data = torch.nn.init.xavier_uniform_(torch.empty(out_features, in_features), gain=gain)

        # Candidate weights are per-sample and store the "fast" or plastic weights.
        # They require gradients only if we are using the backprop or bptt updater.
        self.candidate_weights = nn.Parameter(torch.zeros(self.batch_size, out_features, in_features), requires_grad=(updater in ['backprop', 'bptt']))
        distribution = torch.ones_like(self.weight)
        rand_vals = torch.rand_like(self.weight)
        self.mask = nn.Parameter((rand_vals < plast_proportion).bool(), requires_grad=False)
        distribution[self.mask] = plast_clip

        mask_tier_two = rand_vals < 0.01
        forget_dist = torch.zeros_like(self.weight)
        forget_dist[self.mask] = forget_rate
        forget_dist[mask_tier_two] = forget_rate
        self.forgetting_factor = nn.Parameter(forget_dist, requires_grad=False)

        # Initialize plasticity parameters with the generated values
        if self.is_last_layer == False:
            self.plasticity = nn.Parameter(distribution, requires_grad=requires_grad)
        else:
            self.plasticity = nn.Parameter(torch.ones_like(self.weight), requires_grad=requires_grad)
        print(f"Number of non-zero values in self.plasticity: {torch.count_nonzero(self.plasticity).item()}")

        self.plasticity_feedback_weights = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(len(charset), out_features)), requires_grad=requires_grad)

    def wipe(self):
        # Suppose candidate_weights is of shape [B, out_features, in_features]
        # Aggregate across the batch (e.g., average) to get a unified copy:
        aggregated = self.candidate_weights.mean(dim=0, keepdim=True)
        # Then set every candidate weight in the batch to this aggregated value:
        self.candidate_weights.data.copy_(aggregated.repeat(self.batch_size, 1, 1))
        
        # Ensure the mask is broadcastable or repeated along the batch dimension
        batch_mask = self.mask.unsqueeze(0).repeat(self.batch_size, 1, 1)  # Shape: [B, out_features, in_features]

        # Apply the mask
        self.weight[self.mask] = 0
        # Use .data to modify the tensor in-place without interfering with autograd
        self.candidate_weights.data[batch_mask] = 0
        # Reset the time counter at the start of the sequence
        self.t.fill_(0.0)

    def forward(self, input):
        batch_size = input.size(0)
        
        # Perform a batched matrix multiplication.
        # input: [B, in_features] -> reshape to [B, in_features, 1]
        input_unsq = input.unsqueeze(2)

        # The output will be [B, out_features, 1] and then we can squeeze the last dimension.
        output = torch.bmm(self.candidate_weights, input_unsq).squeeze(2)
        
        # Optionally add a bias if needed.
        if self.bias is not None:
            output = output + self.bias
        self.update_imprints(input, output)
        return output

    def update_imprints(self, input, output):
        self.in_traces.data = input
        self.out_traces.data = output

    def populate_dfa_gradients(self, error_signal):
        """Populate gradients using DFA feedback weights for gradient-based update."""
        input = self.in_traces.data
        input_expanded = input.unsqueeze(1)  # Shape: [batch_size, 1, in_features]
        
        # Project error signal using feedback weights (DFA-specific)
        if self.is_last_layer:
            projected_error = error_signal
        else:
            # For non-last layers in DFA, project the error signal
            # The feedback weights should match: [vocab_size, out_features]
            # error_signal: [batch_size, vocab_size] -> projected_error: [batch_size, out_features]
            projected_error = error_signal @ self.feedback_weights
        
        # Store projected error for bias updates
        self._last_projected_error = projected_error
        
        # Compute per-batch gradient using outer product
        # projected_error: [batch_size, out_features]
        # input_expanded: [batch_size, 1, in_features]
        out = projected_error.unsqueeze(2)  # [batch_size, out_features, 1]
        gradient = out * input_expanded  # [batch_size, out_features, in_features]
        
        # Populate candidate_weights.grad
        if self.candidate_weights.grad is None:
            self.candidate_weights.grad = gradient.clone()
        else:
            self.candidate_weights.grad.copy_(gradient)

    def apply_unified_updates(self, learning_rate, grad_clip, state):
        """Unified update mechanism for both DFA and backprop.
        
        Args:
            learning_rate: Learning rate for updates
            grad_clip: Gradient clipping value
            state: Training state dictionary for logging
        """
        if self.candidate_weights.grad is None:
            return
            
        # Get the gradient (already populated by either DFA or backprop)
        update = -self.candidate_weights.grad.clone()
        
        # Apply plasticity scaling and masking (same for both methods)
        if not self.is_last_layer:
            plasticity_expanded = self.plasticity.unsqueeze(0)  # [1, out_features, in_features]
            mask_expanded = self.mask.unsqueeze(0)  # [1, out_features, in_features]
            
            # Scale by plasticity and mask
            update = update * plasticity_expanded
            # update = update * mask_expanded
            
            # Apply gradient clipping (element-wise for consistency)
            if grad_clip > 0:
                update = torch.where(mask_expanded, 
                                    torch.clamp(update, -grad_clip, grad_clip), 
                                    update)
        
        self.candidate_weights.data = self.candidate_weights.data + learning_rate * update
        
        # Log norms if requested
        if state.get("log_norms_now", False):
            self._log_update_norms(update)
        
        # Update bias using the gradient if this is DFA
        self._update_bias_from_grad(learning_rate)
        # Apply normalization and weight clipping if enabled
        self._apply_regularization()

    def _update_bias_from_grad(self, learning_rate):
        """Helper method to update bias using stored gradients."""
        if hasattr(self, 'bias') and self.bias is not None:
            if self.updater == 'dfa':
                # For DFA, manually update bias using projected error
                if hasattr(self, '_last_projected_error'):
                    bias_update = -learning_rate * self._last_projected_error.mean(dim=0)
                    if len(bias_update.shape) > 1:
                        bias_update = bias_update.mean(dim=0)
                    self.bias.data += bias_update
            elif self.updater in ['backprop', 'bptt']:
                # For backprop/bptt, manually update bias using the computed bias gradient
                if self.bias.grad is not None:
                    bias_update = -learning_rate * self.bias.grad
                    self.bias.data += bias_update

    
    def _log_update_norms(self, update):
        """Helper method to log update norms."""
        with torch.no_grad():
            mask_expanded = self.mask.unsqueeze(0).expand_as(update)
            
            high_plast_update = update[mask_expanded]
            low_plast_update = update[~mask_expanded]
            
            high_norm = torch.norm(high_plast_update).item() if high_plast_update.numel() > 0 else 0.0
            self.last_high_plast_update_norm.data.fill_(high_norm)
            
            low_norm = torch.norm(low_plast_update).item() if low_plast_update.numel() > 0 else 0.0
            self.last_low_plast_update_norm.data.fill_(low_norm)
    
    def _update_bias(self, projected_error, learning_rate):
        """Helper method to update bias consistently."""
        if hasattr(self, 'bias') and self.bias is not None and self.updater == 'dfa':
            # For DFA, manually update bias. For backprop, optimizer handles it.
            bias_update = learning_rate * projected_error.mean(dim=0)
            if len(bias_update.shape) > 1:
                bias_update = bias_update.mean(dim=0)
            self.bias.data += bias_update
    
    def _apply_regularization(self):
        """Helper method to apply normalization and weight clipping."""
        if self.normalize:
            for p in self.parameters():
                # Skip boolean tensors (like masks) and only normalize float tensors
                if p.dtype.is_floating_point:
                    p.data = p.data / (p.data.norm(2) + 1e-6)
        
        if self.clip_weights != 0:
            self.candidate_weights.data.clamp_(-self.clip_weights, self.clip_weights)


    def apply_forget_step(self):
        """Applies forgetting factor to high-plasticity weights.
        This is done with no_grad to prevent interference with backprop."""
        with torch.no_grad():
            # Use non-inplace multiplication to avoid RuntimeError during backprop.
            # The original `mul_` was an inplace operation that corrupted the
            # computation graph needed by autograd for the backward pass.
            self.candidate_weights.data = self.candidate_weights.data * (1 - self.forgetting_factor)

    def scale_gradients(self, plast_clip):
        """Scales gradients for high-plasticity weights before optimizer step."""
        if self.candidate_weights.grad is None:
            return

        # Do not scale gradients for the final layer, mirroring the DFA update rule.
        if self.is_last_layer:
            return

        with torch.no_grad():
            # Create a scaling tensor based on the plasticity mask.
            # The scaling factor is `plast_clip`, making the effective learning rate
            # for high-plasticity weights `learning_rate * plast_clip`, which
            # mirrors the logic in the DFA updater.
            lr_scale = plast_clip
            # self.mask is [out, in], grad is [B, out, in]
            scaling_factor = torch.ones_like(self.mask, dtype=torch.float)
            scaling_factor[self.mask] = lr_scale
            
            # Apply scaling
            self.candidate_weights.grad *= scaling_factor.unsqueeze(0)

    def get_norms(self):
        """Calculates and returns weight and last update norms."""
        with torch.no_grad():
            weights = self.candidate_weights.data
            # Ensure mask is broadcastable for indexing
            mask_expanded = self.mask.unsqueeze(0).expand_as(weights)

            combined_weight_norm = torch.norm(weights).item()

            # Check if any high plasticity weights exist before calculating norm
            high_plast_weights = weights[mask_expanded]
            high_plast_norm = torch.norm(high_plast_weights).item() if high_plast_weights.numel() > 0 else 0.0

            # Check if any low plasticity weights exist
            low_plast_weights = weights[~mask_expanded]
            low_plast_norm = torch.norm(low_plast_weights).item() if low_plast_weights.numel() > 0 else 0.0

            # update_norm = self.last_update_norm.item()

        return {
            'weight_norm': combined_weight_norm,
            'high_plast_weight_norm': high_plast_norm,
            'low_plast_weight_norm': low_plast_norm,
            'high_plast_update_norm': self.last_high_plast_update_norm.item(),
            'low_plast_update_norm': self.last_low_plast_update_norm.item(),
        }

    def store_grad_norms(self):
        """Calculates the norm of the current gradient and stores it."""
        if self.candidate_weights.grad is None:
            self.last_high_plast_update_norm.data.fill_(0.0)
            self.last_low_plast_update_norm.data.fill_(0.0)
            return

        with torch.no_grad():
            grad = self.candidate_weights.grad
            mask_expanded = self.mask.unsqueeze(0).expand_as(grad)

            high_plast_grad = grad[mask_expanded]
            low_plast_grad = grad[~mask_expanded]

            high_norm = torch.norm(high_plast_grad).item() if high_plast_grad.numel() > 0 else 0.0
            self.last_high_plast_update_norm.data.fill_(high_norm)

            low_norm = torch.norm(low_plast_grad).item() if low_plast_grad.numel() > 0 else 0.0
            self.last_low_plast_update_norm.data.fill_(low_norm)

class EtherealRNN(torch.nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, num_layers, charset,
        dropout_rate=0, residual_connection=False, init_type='zero',
        normalize=True, clip_weights=False, updater='dfa',
        plast_clip=1, batch_size=1, forget_rate=0.7, plast_proportion=0.2,
        enable_recurrence=True
    ):
        super(EtherealRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.init_type = init_type
        inner_size = input_size + hidden_size
        self.residual_connection = residual_connection
        self.batch_size = batch_size
        self.forget_rate = forget_rate
        self.enable_recurrence = enable_recurrence

        # Using HebbianLinear instead of Linear
        self.linear_layers = torch.nn.ModuleList([
            HebbianLinear(
                inner_size, inner_size, charset,
                normalize=normalize, clip_weights=clip_weights,
                updater=updater, plast_clip=plast_clip,
                batch_size=batch_size, forget_rate=forget_rate,
                plast_proportion=plast_proportion
            )
        ])
        for _ in range(1, num_layers):
            self.linear_layers.append(HebbianLinear(
                inner_size, inner_size, charset,
                normalize=normalize, clip_weights=clip_weights,
                updater=updater, plast_clip=plast_clip,
                batch_size=batch_size, forget_rate=forget_rate,
                plast_proportion=plast_proportion
            ))

        # Dropout layers
        self.dropout = nn.Dropout(dropout_rate)

        # Final layers for hidden and output, also using HebbianLinear
        self.i2h = HebbianLinear(
            inner_size, hidden_size, charset,
            normalize=normalize, clip_weights=clip_weights,
            updater=updater, plast_clip=plast_clip,
            batch_size=batch_size, forget_rate=forget_rate,
            plast_proportion=plast_proportion
        )
        self.i2o = HebbianLinear(
            inner_size, output_size, charset,
            normalize=normalize, clip_weights=clip_weights,
            updater=updater, requires_grad=False, is_last_layer=True,
            plast_clip=plast_clip, batch_size=batch_size, forget_rate=forget_rate,
            plast_proportion=plast_proportion
        )
        self.self_grad = HebbianLinear(
            inner_size, output_size, charset,
            normalize=normalize, clip_weights=clip_weights,
            updater=updater, requires_grad=False, is_last_layer=True,
            plast_clip=plast_clip, batch_size=batch_size, forget_rate=forget_rate,
            plast_proportion=plast_proportion
        )
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # print(f"input shape: {input.shape}, hidden shape: {hidden.shape}")
        combined = torch.cat((input, hidden), dim=1)
        if self.residual_connection:
            residual = combined.clone()  # Store the original combined tensor for residual connection

        # Pass through the Hebbian linear layers with ReLU and Dropout
        for layer in self.linear_layers:
            combined = layer(combined)
            combined = F.gelu(combined)
            # combined = self.dropout(combined)

        # Add the residual (original combined tensor) to the output of the layers
        # print(f"residual_shape: {residual.shape}, combined shape: {combined.shape}")
        if self.residual_connection:
            combined += residual

        # Split into hidden and output
        if self.enable_recurrence:
            hidden = self.i2h(combined)
        else:
            hidden = torch.zeros_like(hidden)  # disable hidden connection
        output = self.i2o(combined)
        self_grad = self.self_grad(combined)
        hidden = torch.tanh(hidden)  # Apply tanh function to keep hidden from blowing up after many recurrences

        # output.requires_grad = True # This is now handled in the training loop for DFA.
        # output = self.dropout(output)  # Apply dropout to the output before softmax
        # output = self.softmax(output)
        return output, hidden, self_grad

    def initHidden(self, batch_size):
        device = next(self.parameters()).device
        return torch.zeros(batch_size, self.hidden_size, device=device, requires_grad=False)

    def apply_forget_step(self):
        """Calls apply_forget_step on all HebbianLinear layers."""
        for layer in self.linear_layers:
            layer.apply_forget_step()
        self.i2h.apply_forget_step()
        self.i2o.apply_forget_step()
        self.self_grad.apply_forget_step()

    def scale_gradients(self, plast_clip):
        """Calls scale_gradients on all HebbianLinear layers."""
        for layer in self.linear_layers:
            layer.scale_gradients(plast_clip)
        self.i2h.scale_gradients(plast_clip)
        self.i2o.scale_gradients(plast_clip)
        # self_grad is not trained with backprop, so no gradients to scale
        # self.self_grad.scale_gradients(plast_clip)

    
    def get_all_norms(self):
        """Aggregates norms from all HebbianLinear layers."""
        all_norms = {}
        
        def _collect_norms(layers_list, prefix):
            for i, layer in enumerate(layers_list):
                if isinstance(layer, HebbianLinear):
                    layer_norms = layer.get_norms()
                    for key, value in layer_norms.items():
                        all_norms[f'{prefix}_{i}_{key}'] = value

        _collect_norms(self.linear_layers, 'linear')
        _collect_norms([self.i2h], 'i2h')
        _collect_norms([self.i2o], 'i2o')
        # You might want to log self_grad norms too if it's important
        _collect_norms([self.self_grad], 'self_grad') 

        return all_norms

    def store_all_grad_norms(self):
        """Calls store_grad_norms on all HebbianLinear layers that are trained."""
        for layer in self.linear_layers:
            layer.store_grad_norms()
        self.i2h.store_grad_norms()
        self.i2o.store_grad_norms()
        # self_grad is not trained with backprop, so its grad will be None.
        # self.self_grad.store_grad_norms()

    def wipe(self):
        for layer in self.linear_layers:
            layer.wipe()
        self.i2h.wipe()
        self.i2o.wipe()
        self.self_grad.wipe()


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_rate=0.1, init_type='zero', enable_recurrence=True):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.init_type = init_type
        self.enable_recurrence = enable_recurrence

        # Replace HebbianLinear with standard Linear layers
        self.linear_layers = nn.ModuleList([nn.Linear(input_size + hidden_size, hidden_size)])
        for _ in range(1, num_layers):
            self.linear_layers.append(nn.Linear(hidden_size, hidden_size))

        # Dropout layers
        self.dropout = nn.Dropout(dropout_rate)

        # Final layers for hidden and output
        self.i2h = nn.Linear(hidden_size, hidden_size)
        self.i2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # print(f"input shape: {input.shape}, hidden shape: {hidden.shape}")
        combined = torch.cat((input, hidden), dim=1)

        # Pass through the linear layers with ReLU and Dropout
        for layer in self.linear_layers:
            combined = layer(combined)
            combined = F.relu(combined)
            # combined = self.dropout(combined)

        # Split into hidden and output
        if self.enable_recurrence:
            hidden = self.i2h(combined)
        else:
            hidden = torch.zeros_like(hidden)  # disable hidden connection
        output = self.i2o(combined)
        hidden = torch.tanh(hidden)
        # output = self.dropout(output)
        # output = self.softmax(output)
        return output, hidden, None

    def get_all_norms(self):
        """Calculates weight and gradient norms for SimpleRNN."""
        all_norms = {}
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.requires_grad:
                    all_norms[f'{name}_weight_norm'] = torch.norm(param.data).item()
                    if param.grad is not None:
                        all_norms[f'{name}_grad_norm'] = torch.norm(param.grad).item()
                    else:
                        all_norms[f'{name}_grad_norm'] = 0.0 # No grad yet/available
        return all_norms

    def initHidden(self, batch_size):
        device = next(self.parameters()).device
        return torch.zeros(batch_size, self.hidden_size, device=device)
