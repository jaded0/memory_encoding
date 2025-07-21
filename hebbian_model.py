import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
from utils import initialize_charset
import numpy as np
import torch.nn.utils.parametrize as parametrize
# from memory_profiler import profile


class PlasticityNorm(nn.Module):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr

    def forward(self, plasticity):
        norm = plasticity.norm(p=2, dim=None, keepdim=True)
        return self.lr * plasticity / norm

class HebbianLinear(nn.Linear):
    def __init__(self, in_features, out_features, charset, bias=True, normalize=True, clip_weights=False, updater='dfa', requires_grad=False, is_last_layer=False, plast_clip=1, batch_size=1, forget_rate=0.7, plast_proportion=0.2):
        super(HebbianLinear, self).__init__(in_features, out_features, bias)

        # Set requires_grad for the base class parameters
        self.weight.requires_grad = False # Base weights are not trained directly
        if bias:
            # Bias is trained with backprop if the updater is backprop
            self.bias.requires_grad = (updater == 'backprop')

        self.imprints = nn.Parameter(torch.zeros_like(self.weight), requires_grad=requires_grad)
        self.normalize = normalize
        self.clip_weights = clip_weights
        self.updater = updater
        self.is_last_layer = is_last_layer
        self.in_traces = nn.Parameter(torch.zeros(batch_size, in_features), requires_grad=requires_grad)
        self.out_traces = nn.Parameter(torch.zeros(batch_size, out_features), requires_grad=requires_grad)

        self.last_high_plast_update_norm = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.last_low_plast_update_norm = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.register_buffer('t', torch.tensor(1.0))
        self.learning_rate = 1
        self.batch_size = batch_size


        # Initialize weights with the adjusted gain
        self.feedback_weights = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(len(charset), out_features)), requires_grad=requires_grad)
        # self.weight.data = torch.nn.init.xavier_uniform_(torch.empty(out_features, in_features), gain=gain)

        # Candidate weights are per-sample and store the "fast" or plastic weights.
        # They require gradients only if we are using the backprop updater.
        self.candidate_weights = nn.Parameter(torch.zeros(self.batch_size, out_features, in_features), requires_grad=(updater == 'backprop'))
        self.plasticity_candidate_weights = nn.Parameter(torch.zeros_like(self.weight), requires_grad=requires_grad)
        distribution = torch.ones_like(self.weight)
        rand_vals = torch.rand_like(self.weight)
        self.mask = nn.Parameter((rand_vals < plast_proportion).bool(), requires_grad=False)
        distribution[self.mask] = plast_clip

        mask_tier_two = rand_vals < 0.01
        forget_dist = torch.zeros_like(self.weight)
        forget_dist[self.mask] = forget_rate
        forget_dist[mask_tier_two] = forget_rate
        self.forgetting_factor = nn.Parameter(forget_dist, requires_grad=False)

        uniform = torch.empty_like(self.weight).uniform_(0.1, 2.0) * math.pi
        self.phase_shift = nn.Parameter(torch.zeros_like(self.weight), requires_grad=False)
        self.phase_shift[(rand_vals < 0.1).bool()] = math.pi/2
        uniform[~self.mask] = 0 # only wave on our high-plast values here.
        # Initialize plasticity parameters with the generated values
        if self.is_last_layer == False:
            self.plasticity = nn.Parameter(distribution, requires_grad=requires_grad)
            self.frequency = nn.Parameter(uniform, requires_grad=requires_grad)  # Initialize plasticity parameters
        else:
            self.plasticity = nn.Parameter(torch.ones_like(self.weight), requires_grad=requires_grad)  # Initialize plasticity parameters
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
        self.candidate_weights[batch_mask] = 0
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

    def apply_imprints(self, reward, learning_rate, plast_learning_rate, plast_clip, imprint_rate, grad_clip, state):
        input = self.in_traces.data
        output = self.out_traces.data
        # reward = reward.detach()
        # input = self.in_traces.detach()
        # output = self.out_traces.detach()
        
        # Reshape input and output for broadcasting
        input_expanded = input.unsqueeze(1)  # Shape: [batch_size, 1, in_features]
        output_expanded = output.unsqueeze(2)  # Shape: [batch_size, out_features, 1]
        global_error = reward

        # Project the global error using the fixed random matrix
        # I think in backprop these two values actually come from the next layer's 
        # error vector and weights. chain ruleee
        # print(self.feedback_weights.shape)
        if self.is_last_layer==True: 
            projected_error = global_error
        else:
            projected_error = global_error @ self.feedback_weights  # assuming matrix multiplication
            # I am currently completely ignoring the effect of dropout and relu. 
            # it oughta look like the derivative of the output of the two with respect to the inputs. 
            # prolly either one or zero. It'll just zero out some values in the error signal bc they were
            # zeroed out in the actual output.

            # Apply ReLU derivative
            # relu_derivative = (output.squeeze() > 0).float()  # 1 for activated neurons, 0 otherwise
            # projected_error *= relu_derivative

        # This is the DFA update rule, previously 'nocycle'. It's the only non-backprop rule.
        out = projected_error.unsqueeze(2)

        candidate_update = out * input.unsqueeze(1)#(input.unsqueeze(1) - out_weights_product)
        
        # Reset or decay candidate_weights
        batch_agg_candidate_update = candidate_update#.mean(dim=0)
        update = batch_agg_candidate_update


        if self.is_last_layer:
            self.candidate_weights.add_(update, alpha=learning_rate)

        else:
            self.candidate_weights.mul_(1 - self.forgetting_factor)      # forget

            update.mul_(learning_rate * self.plasticity)             # scale
            update.mul_(self.mask)                                   # gate

            if grad_clip > 0:
                update.masked_clamp_(self.mask, -grad_clip, grad_clip)

            self.candidate_weights.add_(update)

    
        if state["log_norms_now"] == True:
            # Log the norms of the weights and updates
            with torch.no_grad():
                mask_expanded = self.mask.unsqueeze(0).expand_as(update)

                high_plast_update = update[mask_expanded]
                low_plast_update = update[~mask_expanded]

                # Calculate and store high plasticity update norm
                high_norm = torch.norm(high_plast_update).item() if high_plast_update.numel() > 0 else 0.0
                self.last_high_plast_update_norm.data.fill_(high_norm)

                # Calculate and store low plasticity update norm
                low_norm = torch.norm(low_plast_update).item() if low_plast_update.numel() > 0 else 0.0
                self.last_low_plast_update_norm.data.fill_(low_norm)


        # self.weight.data += update

        # Update bias if applicable
        if hasattr(self, 'bias') and self.bias is not None:
            # print("has")
            # Bias is updated based on the mean error across the batch
            # print(f'shape of bias:{self.bias.data.shape}, shape of mean: {projected_error.mean(dim=0).shape}')
            bias_update = learning_rate * projected_error.mean(dim=0).mean(dim=0)
            self.bias.data += bias_update

        if self.normalize:
            # Normalize the weights to prevent them from exploding
            for p in self.parameters():
                p.data = p.data / (p.data.norm(2) + 1e-6)
        
        if self.clip_weights != 0:
            max_weight_value = self.clip_weights
            # for p in self.weight:
            self.weight.data.clamp_(-max_weight_value, max_weight_value)

    def apply_forget_step(self):
        """Applies forgetting factor to high-plasticity weights.
        This is done with no_grad to prevent interference with backprop."""
        with torch.no_grad():
            # Use non-inplace multiplication to avoid RuntimeError during backprop.
            # The original `mul_` was an inplace operation that corrupted the
            # computation graph needed by autograd for the backward pass.
            self.candidate_weights.data = self.candidate_weights.data * (1 - self.forgetting_factor)

    def scale_gradients(self, plast_learning_rate, learning_rate):
        """Scales gradients for high-plasticity weights before optimizer step."""
        if self.candidate_weights.grad is None:
            return

        with torch.no_grad():
            if learning_rate > 0:
                # Create a scaling tensor based on the plasticity mask
                # Default scale is 1, high plasticity scale is plast_lr / base_lr
                lr_scale = (plast_learning_rate / learning_rate)
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

class HebbyRNN(torch.nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, num_layers, charset,
        dropout_rate=0, residual_connection=False, init_type='zero',
        normalize=True, clip_weights=False, updater='dfa',
        plast_clip=1, batch_size=1, forget_rate=0.7, plast_proportion=0.2
    ):
        super(HebbyRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.init_type = init_type
        inner_size = input_size + hidden_size
        self.residual_connection = residual_connection
        self.batch_size = batch_size
        self.forget_rate = forget_rate

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
        # hidden = self.i2h(combined) # This call is expensive and its result is immediately discarded.
        hidden = torch.zeros_like(hidden) # disable hidden connection
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

    def scale_gradients(self, plast_learning_rate, learning_rate):
        """Calls scale_gradients on all HebbianLinear layers."""
        for layer in self.linear_layers:
            layer.scale_gradients(plast_learning_rate, learning_rate)
        self.i2h.scale_gradients(plast_learning_rate, learning_rate)
        self.i2o.scale_gradients(plast_learning_rate, learning_rate)
        # self_grad is not trained with backprop, so no gradients to scale
        # self.self_grad.scale_gradients(plast_learning_rate, learning_rate)

    def apply_imprints(self, reward, learning_rate, plast_learning_rate, plast_clip, imprint_rate, grad_clip, state):
        # Apply imprints for all HebbianLinear layers
        for layer in self.linear_layers:
            layer.apply_imprints(reward, learning_rate, plast_learning_rate, plast_clip, imprint_rate, grad_clip, state)
        # self.i2h.apply_imprints(reward, learning_rate, plast_learning_rate, plast_clip, imprint_rate)
        self.i2o.apply_imprints(reward, learning_rate, plast_learning_rate, plast_clip, imprint_rate, grad_clip, state)
        self.self_grad.apply_imprints(reward, learning_rate, plast_learning_rate, plast_clip, imprint_rate, grad_clip, state)
    
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

    def wipe(self):
        for layer in self.linear_layers:
            layer.wipe()
        


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_rate=0.1, init_type='zero'):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.init_type = init_type

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
        hidden = self.i2h(combined)
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
