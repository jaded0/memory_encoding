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
    def __init__(self, in_features, out_features, charset, bias=True, normalize=True, clip_weights=False, update_rule='damage', requires_grad=False, is_last_layer=False, plast_clip=1, batch_size=1, forget_rate=0.7):
        super(HebbianLinear, self).__init__(in_features, out_features, bias)

        # Set requires_grad for the base class parameters
        self.weight.requires_grad = requires_grad
        if bias:
            self.bias.requires_grad = requires_grad

        self.imprints = nn.Parameter(torch.zeros_like(self.weight), requires_grad=requires_grad)
        self.normalize = normalize
        self.clip_weights = clip_weights
        self.update_rule = update_rule
        self.is_last_layer = is_last_layer
        self.in_traces = nn.Parameter(torch.zeros(in_features), requires_grad=requires_grad)
        self.out_traces = nn.Parameter(torch.zeros(out_features), requires_grad=requires_grad)
        self.register_buffer('t', torch.tensor(1.0))
        self.learning_rate = 1
        self.batch_size = batch_size


        # Initialize weights with the adjusted gain
        self.feedback_weights = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(len(charset), out_features)), requires_grad=requires_grad)
        # self.weight.data = torch.nn.init.xavier_uniform_(torch.empty(out_features, in_features), gain=gain)

        if update_rule == 'plastic_candidate' or update_rule == 'static_plastic_candidate':
            self.candidate_weights = nn.Parameter(torch.zeros(self.batch_size, out_features, in_features), requires_grad=False)
            self.plasticity_candidate_weights = nn.Parameter(torch.zeros_like(self.weight), requires_grad=requires_grad)
            distribution = torch.ones_like(self.weight)
            rand_vals = torch.rand_like(self.weight)
            self.mask = nn.Parameter((rand_vals < 0.2).bool(), requires_grad=False)
            distribution[self.mask] = plast_clip

            mask_tier_two = rand_vals < 0.01
            forget_dist = torch.ones_like(self.weight)
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

    def apply_imprints(self, reward, learning_rate, plast_learning_rate, plast_clip, imprint_rate, grad_clip):
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

        if self.update_rule == 'static_plastic_candidate':
            out = projected_error.unsqueeze(2)

            candidate_update = out * input.unsqueeze(1)#(input.unsqueeze(1) - out_weights_product)
            
            # Reset or decay candidate_weights
            batch_agg_candidate_update = candidate_update#.mean(dim=0)
            update = batch_agg_candidate_update

            if self.is_last_layer:
                update = update * learning_rate #* 0.1
                # self.weight.data += update
                self.candidate_weights.data += update
            else:
                plastic_mask = (((torch.cos(self.t * self.frequency.data + self.phase_shift) + 1) * 0.5) > 0.5)
                # plastic_mask = self.mask
                self.candidate_weights *= torch.max(~self.mask, (1 - plastic_mask * (self.forgetting_factor)))
                update = update * (learning_rate * self.plasticity.data) * plastic_mask
                self.t += 1
                # self.weight.data += update
                update[self.mask.unsqueeze(0).expand_as(update)].clamp_(-grad_clip, grad_clip)
                self.candidate_weights.data += update
                # print(self.weight.norm(p=2))
        else:
            raise ValueError(f'Invalid update rule: {self.update_rule}')


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

class HebbyRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, charset, dropout_rate=0, residual_connection=False, init_type='zero', normalize=True, clip_weights=False, update_rule='damage', plast_clip=1, batch_size=1, forget_rate=0.7):
        super(HebbyRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.init_type = init_type
        inner_size = input_size+hidden_size
        self.residual_connection = residual_connection
        self.batch_size = batch_size
        self.forget_rate = forget_rate

        # Using HebbianLinear instead of Linear
        self.linear_layers = torch.nn.ModuleList([HebbianLinear(inner_size, inner_size, charset, normalize=normalize, clip_weights=clip_weights, update_rule=update_rule, plast_clip=plast_clip, batch_size=batch_size, forget_rate = forget_rate)])
        for _ in range(1, num_layers):
            self.linear_layers.append(HebbianLinear(inner_size, inner_size, charset, normalize=normalize, clip_weights=clip_weights, update_rule=update_rule, plast_clip=plast_clip, batch_size=batch_size, forget_rate = forget_rate))

        # Dropout layers
        self.dropout = nn.Dropout(dropout_rate)

        # Final layers for hidden and output, also using HebbianLinear
        self.i2h = HebbianLinear(inner_size, hidden_size, charset, normalize=normalize, clip_weights=clip_weights, update_rule=update_rule, plast_clip=plast_clip, batch_size=batch_size, forget_rate = forget_rate)
        self.i2o = HebbianLinear(inner_size, output_size, charset, normalize=normalize, clip_weights=clip_weights, update_rule=update_rule, requires_grad=False, is_last_layer=True, plast_clip=plast_clip, batch_size=batch_size, forget_rate = forget_rate)
        self.self_grad = HebbianLinear(inner_size, output_size, charset, normalize=normalize, clip_weights=clip_weights, update_rule=update_rule, requires_grad=False, is_last_layer=True, plast_clip=plast_clip, batch_size=batch_size, forget_rate = forget_rate)
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
        hidden = self.i2h(combined)
        hidden = torch.zeros_like(hidden) # disable hidden connection
        output = self.i2o(combined)
        self_grad = self.self_grad(combined)
        hidden = torch.tanh(hidden)  # Apply tanh function to keep hidden from blowing up after many recurrences

        output.requires_grad = True
        # output = self.dropout(output)  # Apply dropout to the output before softmax
        # output = self.softmax(output)
        return output, hidden, self_grad

    def initHidden(self, batch_size):
        device = next(self.parameters()).device
        return torch.zeros(batch_size, self.hidden_size, device=device, requires_grad=False)

    def apply_imprints(self, reward, learning_rate, plast_learning_rate, plast_clip, imprint_rate, grad_clip):
        # Apply imprints for all HebbianLinear layers
        for layer in self.linear_layers:
            layer.apply_imprints(reward, learning_rate, plast_learning_rate, plast_clip, imprint_rate, grad_clip)
        # self.i2h.apply_imprints(reward, learning_rate, plast_learning_rate, plast_clip, imprint_rate)
        self.i2o.apply_imprints(reward, learning_rate, plast_learning_rate, plast_clip, imprint_rate, grad_clip)
        self.self_grad.apply_imprints(reward, learning_rate, plast_learning_rate, plast_clip, imprint_rate, grad_clip)
    
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
        return output, hidden

    def initHidden(self, batch_size):
        device = next(self.parameters()).device
        return torch.zeros(batch_size, self.hidden_size, device=device)
