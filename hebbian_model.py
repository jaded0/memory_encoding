import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
from utils import initialize_charset
import numpy as np
import torch.nn.utils.parametrize as parametrize

class PlasticityNorm(nn.Module):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr

    def forward(self, plasticity):
        norm = plasticity.norm(p=2, dim=None, keepdim=True)
        return self.lr * plasticity / norm

class HebbianLinear(nn.Linear):
    def __init__(self, in_features, out_features, charset, bias=True, normalize=True, clip_weights=False, update_rule='damage', alpha=0.5, requires_grad=False, is_last_layer=False, candecay=0.9, plast_candecay=0.5, plast_clip=1):
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
        self.candecay = candecay
        self.plast_candecay = plast_candecay
        self.t = 1
        self.learning_rate = 1

        # Calculate the adjusted gain for [-1, 1] range
        # gain = 1 / math.sqrt(6 / (in_features + out_features))

        # Initialize weights with the adjusted gain
        self.feedback_weights = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(len(charset), out_features)), requires_grad=requires_grad)
        # self.weight.data = torch.nn.init.xavier_uniform_(torch.empty(out_features, in_features), gain=gain)

        if update_rule == 'covariance':
            self.alpha = alpha  # Decay factor for the exponential moving average
        
        if update_rule == 'candidate':
            self.candidate_weights = nn.Parameter(torch.zeros_like(self.weight), requires_grad=requires_grad)
        if update_rule == 'plastic_candidate' or update_rule == 'static_plastic_candidate':
            self.candidate_weights = nn.Parameter(torch.zeros_like(self.weight), requires_grad=requires_grad)
            self.plasticity_candidate_weights = nn.Parameter(torch.zeros_like(self.weight), requires_grad=requires_grad)
            # Generate random values with a log-uniform distribution between 1e-2 and 1e2
            log_uniform = torch.empty_like(self.weight).normal_(0, plast_clip)#.clamp_(1e-1,1e5)
            uniform = torch.empty_like(self.weight).uniform_(0.1, 3.141)
            # print(uniform)
            # Initialize plasticity parameters with the generated values
            if self.is_last_layer == False:
                self.plasticity = nn.Parameter(log_uniform, requires_grad=requires_grad)
                self.frequency = nn.Parameter(uniform, requires_grad=requires_grad)  # Initialize plasticity parameters
            else:
                self.plasticity = nn.Parameter(torch.ones_like(self.weight), requires_grad=requires_grad)  # Initialize plasticity parameters
            # self.plasticity.data = self.plasticity.data / (torch.norm(self.plasticity.data, p=1) + 1e-8)
            # keep the effective learning rate to be the learning rate, on average
            # parametrize.register_parametrization(self, 'plasticity', PlasticityNorm(self.learning_rate))
            # self.plasticity = nn.Parameter(torch.nn.init.xavier_normal_(torch.ones_like(self.weight)), requires_grad=requires_grad)
            # nn.init.kaiming_uniform_(self.plasticity, a=math.sqrt(5))
            self.plasticity_feedback_weights = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(len(charset), out_features)), requires_grad=requires_grad)


    def forward(self, input):
        # print(f"in forward. input requires grad? {input.requires_grad}")
        output = super(HebbianLinear, self).forward(input)
        # print(f"in forward. output requires grad? {output.requires_grad}")
        self.update_imprints(input, output)
        # print(output)
        # max_activation = 1
        # output.clamp_(-max_activation, max_activation)
        return output

    def update_imprints(self, input, output):
        # print("input shape:", input.shape)
        # print("output shape:", output.shape)
    

        self.in_traces.data = input
        self.out_traces.data = output

    def apply_imprints(self, reward, learning_rate, plast_learning_rate, plast_clip, imprint_rate, stochasticity):
        input = self.in_traces.data
        output = self.out_traces.data    
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

        if self.update_rule == 'damage':
            # Element-wise multiplication with broadcasting
            # Results in a [batch_size, out_features, in_features] tensor
            imprint_update = projected_error.unsqueeze(2) * input_expanded

            # for p in imprint_update:
            #     if torch.isnan(p.data).any():
            #         print("the nan is in the initial imprint_update in the learning rule")
            #         sys.exit(1)
            
            # Compute the difference and square it
            diff_squared = (projected_error.unsqueeze(2) - input_expanded) ** 2
            # for p in diff_squared:
            #     if torch.isnan(p.data).any():
            #         print("the nan is in the diff_squared in the learning rule")
            #         sys.exit(1)
            
            # Update the imprint using the new rule: oa*ia - (oa-ia)^2
            imprint_update = imprint_update - diff_squared
            # for p in imprint_update:
            #     if torch.isnan(p.data).any():
            #         print("the nan is in the imprint_update in the learning rule, at the end")
            #         sys.exit(1)
        elif self.update_rule == 'oja':
            imprint_update = output_expanded*(input_expanded - output_expanded * self.weight.data)
        elif self.update_rule == 'competitive':
            imprint_update = output_expanded*(input_expanded - self.weight.data)
        elif self.update_rule =='covariance':
            # Covariance Rule: Δw = η * (y - θ_y) * (x - θ_x)
            imprint_update = (projected_error - self.out_traces).unsqueeze(2) * (input - self.in_traces).unsqueeze(1)
            self.imprints.data += imprint_update.sum(dim=0)

            # Update the running averages (traces) for inputs and projected_errors
            self.in_traces.data = self.alpha * self.in_traces.data + (1 - self.alpha) * input.mean(dim=0)
            self.out_traces.data = self.alpha * self.out_traces.data + (1 - self.alpha) * projected_error.mean(dim=0)
        elif self.update_rule == 'hpca':
            # hpca - Hebbian Principal Component Analysis (HPCA) rule: Updates weights based on the input and the output, 
            # subtracting the reconstructed input from all previous neurons
            # (y_i * (x - Σ(y_j * w_j) for j=1 to i)).
            out = output
            # Outer product of output and weights for all neurons
            # print(projected_error.unsqueeze(2).shape, self.imprints.unsqueeze(0).shape)
            outer_product = out.unsqueeze(2) * self.weight.data.unsqueeze(0)
            
            # Compute cumulative sum for the reconstruction term
            reconstruction = torch.cumsum(outer_product, dim=1)

            # Compute the weight update for all neurons
            imprint_update = (input.unsqueeze(1) - reconstruction) * projected_error.unsqueeze(2)
            # imprint_update += imprint_update.sum(dim=0)
            # print(imprint_update.shape)
            imprint_update = imprint_update.squeeze()
            update = learning_rate * imprint_update * projected_error.T
        elif self.update_rule == 'candidate':
            # only evaluate past weight updates with the current reward signal


            # dfa = projected_error.unsqueeze(2) * input.unsqueeze(1)
            # If dropout was applied during forward pass, apply the same mask here
            # if self.dropout_mask is not None:
            #     projected_error *= self.dropout_mask
            # out = projected_error.unsqueeze(2)
            out = output.unsqueeze(2)
            out_weights_product = out * self.weight.data
            # Now, calculate the candidate_update
            # Note that element-wise multiplication (*) is broadcasted over the batch dimension
            candidate_update = out * (input.unsqueeze(1) - out_weights_product)

            # Reset or decay candidate_weights
            self.candidate_weights.data *= self.candecay  # Example: decay by half is 0.5
            batch_agg_candidate_update = candidate_update.mean(dim=0)
            self.candidate_weights.data += batch_agg_candidate_update*(1-self.candecay)

            sign = torch.sign(batch_agg_candidate_update)
            product = batch_agg_candidate_update*self.candidate_weights.data
            imprint_update = sign * torch.abs(product)**0.5
            update = learning_rate * imprint_update
        elif self.update_rule == 'plastic_candidate':
            out = projected_error.unsqueeze(2)
            # out_plasticity = (global_error @ self.feedback_weights).unsqueeze(2)
            out_plasticity = (global_error @ self.plasticity_feedback_weights).unsqueeze(2)
            out_weights_product = out * self.weight.data
            # plasticity_out_weights_product = out_plasticity * self.plasticity.data

            candidate_update = out * (input.unsqueeze(1) - out_weights_product)
            plasticity_candidate_update = out_plasticity * self.candidate_weights.data #(self.candidate_weights.data - plasticity_out_weights_product)
            
            # Reset or decay candidate_weights
            self.candidate_weights.data *= self.candecay  # Example: decay by half is 0.5
            self.plasticity_candidate_weights.data *= self.plast_candecay  # Example: decay by half is 0.5

            # self.candidate_weights.data += candidate_update.mean(dim=0)
            # self.plasticity_candidate_weights.data += plasticity_candidate_update.mean(dim=0)
            batch_agg_candidate_update = candidate_update.mean(dim=0)
            batch_agg_plasticity_candidate_update = plasticity_candidate_update.mean(dim=0)
            self.candidate_weights.data += batch_agg_candidate_update*(1-self.candecay)
            self.plasticity_candidate_weights.data += batch_agg_plasticity_candidate_update*(1-self.plast_candecay)
            # self.candidate_weights.data *= 1/(1-self.candecay**self.t)
            # self.plasticity_candidate_weights.data *= 1/(1-0.999**self.t)
            # self.t += 1


            # sign = torch.sign(batch_agg_candidate_update)
            # product = batch_agg_candidate_update*self.candidate_weights.data
            # imprint_update = sign * torch.abs(product)**0.5
            sign = torch.sign(self.plasticity_candidate_weights.data)
            product = batch_agg_plasticity_candidate_update*self.plasticity_candidate_weights.data
            plasticity_imprint_update = sign * torch.abs(product)**0.5

            update = batch_agg_candidate_update
            # imprint_update = self.candidate_weights.data
            # plasticity_imprint_update = self.plasticity_candidate_weights.data
            # print(f"are imprint_update and self.candidate_weights different now? {imprint_update - self.candidate_weights.data}")
            # Scale and shift the plasticity values
            # shift, scale = 1,1e8
            # shifted_plasticity = self.plasticity.data + shift
            # scaled_plasticity = scale / (1 + torch.exp(shifted_plasticity) + 1e-40)
            # update = update * scaled_plasticity
            
            # print(update)
            # print(torch.isnan(update).any())
            # update = update * torch.relu(0.01-torch.abs(self.candidate_weights.data))
            # print(update)
            # print(torch.isnan(update).any())

            update = learning_rate * update

            # fluctuate with sine wave
            if not self.is_last_layer:
                # update = update * torch.sin(0.0001*self.t*self.plasticity.data)
                update = update * self.plasticity.data #((torch.sin(self.t*self.frequency.data)+1)/2)
                self.plasticity.data += (plast_learning_rate * plasticity_imprint_update)#.clamp(-plast_clip,plast_clip)
            else:
                update *= 1e-2
            self.t += 1

            # print(torch.isnan(update).any())

            # self.plasticity.data = self.plasticity.data / (torch.norm(self.plasticity.data, p=1) + 1e-8)

            # update.clamp_(-1e2,1e2)
            # update.clamp_(-plast_clip,plast_clip)
            # update = update * self.plasticity.data
            # self.plasticity.data += plast_learning_rate * plasticity_imprint_update
            # self.plasticity.data.clamp_(-2,plast_clip)
            # self.plasticity.data.clamp_(-plast_clip,plast_clip)
            # print(plast_learning_rate * plasticity_imprint_update)

            # self.weight.data -= self.weight.data*(learning_rate/100)*self.plasticity.data

        elif self.update_rule == 'static_plastic_candidate':
            out = projected_error.unsqueeze(2)
            # out_weights_product = out * self.weight.data

            candidate_update = out * input.unsqueeze(1)#(input.unsqueeze(1) - out_weights_product)
            
            # Reset or decay candidate_weights
            # self.candidate_weights.data *= self.candecay  # Example: decay by half is 0.5
            batch_agg_candidate_update = candidate_update.mean(dim=0)
            # self.candidate_weights.data += batch_agg_candidate_update*(1-self.candecay)
            update = learning_rate * batch_agg_candidate_update
            if self.is_last_layer:
                update = update * 0.1
                self.weight.data += update
            else:
                update = update * self.plasticity.data #* torch.sin(self.t*self.frequency.data)
                self.weight.data += update
                # print(self.weight.norm(p=2))
                # self.weight.data *= 1/(0.1*torch.abs(self.plasticity.data))
                self.weight.data -= 0.1*self.plasticity.data
            # update.clamp_(-plast_clip, plast_clip)
        else:
            update = reward.T  * learning_rate * imprint_update + reward.T  * imprint_rate * imprint_update


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

        # Apply stochastic noise to the weights
        # for p in self.parameters():
        #     noise = stochasticity * torch.randn_like(p.data)
        #     p.data += noise

class HebbyRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, charset, dropout_rate=0.1, residual_connection=False, init_type='zero', normalize=True, clip_weights=False, update_rule='damage', candecay=0.9, plast_candecay=0.5, plast_clip=1):
        super(HebbyRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.init_type = init_type
        inner_size = input_size+hidden_size
        self.residual_connection = residual_connection
        self.plast_candecay = plast_candecay

        # # some peculiar new layers for outer product-ing
        # self.linear1 = HebbianLinear(inner_size, inner_size, charset, normalize=normalize, clip_weights=clip_weights, update_rule=update_rule, candecay=candecay)
        # self.linear2 = HebbianLinear(inner_size, inner_size, charset, normalize=normalize, clip_weights=clip_weights, update_rule=update_rule, candecay=candecay)

        # Using HebbianLinear instead of Linear
        self.linear_layers = torch.nn.ModuleList([HebbianLinear(inner_size, inner_size, charset, normalize=normalize, clip_weights=clip_weights, update_rule=update_rule, candecay=candecay, plast_candecay=plast_candecay, plast_clip=plast_clip)])
        for _ in range(1, num_layers):
            self.linear_layers.append(HebbianLinear(inner_size, inner_size, charset, normalize=normalize, clip_weights=clip_weights, update_rule=update_rule, candecay=candecay, plast_candecay=plast_candecay, plast_clip=plast_clip))

        # Dropout layers
        self.dropout = nn.Dropout(dropout_rate)

        # Final layers for hidden and output, also using HebbianLinear
        self.i2h = HebbianLinear(inner_size, hidden_size, charset, normalize=normalize, clip_weights=clip_weights, update_rule=update_rule, candecay=candecay, plast_candecay=plast_candecay, plast_clip=plast_clip)
        self.i2o = HebbianLinear(inner_size, output_size, charset, normalize=normalize, clip_weights=clip_weights, update_rule=update_rule, requires_grad=False, is_last_layer=True, candecay=candecay, plast_candecay=plast_candecay, plast_clip=plast_clip)
        self.softmax = torch.nn.LogSoftmax(dim=1)
    #     # Initialize weights
    #     self.init_weights()

    # def init_weights(self):
    #     for layer in self.linear_layers:
    #         self._init_weight(layer)

    #     self._init_weight(self.i2h)
    #     self._init_weight(self.i2o)

    # def _init_weight(self, layer):
    #     if self.init_type == 'zero':
    #         nn.init.zeros_(layer.weight)
    #         if layer.bias is not None:
    #             nn.init.zeros_(layer.bias)
    #     elif self.init_type == 'orthogonal':
    #         nn.init.orthogonal_(layer.weight)
    #         if layer.bias is not None:
    #             nn.init.zeros_(layer.bias)

    def forward(self, input, hidden):
        # print(f"input shape: {input.shape}, hidden shape: {hidden.shape}")
        combined = torch.cat((input, hidden), dim=1)
        if self.residual_connection:
            residual = combined.clone()  # Store the original combined tensor for residual connection

        # # try out some sort of inner activation matmul inspired by attention
        # a1 = self.linear1(combined)  # Shape: [batch_size, input_dim]
        # a1 = F.relu(a1)
        # a2 = self.linear2(combined)  # Shape: [batch_size, output_dim]
        # a2 = F.relu(a2)
        
        # # Compute the outer product of the activations
        # # Note: we need to reshape tensors to make the outer product
        # outer_product = torch.bmm(a1.unsqueeze(2), a2.unsqueeze(1))
        # # Shape: [batch_size, input_dim, output_dim]

        # # Now, use the outer product as the dynamic weight matrix
        # # Apply the dynamic weight matrix to the input
        # combined = torch.bmm(combined.unsqueeze(1), outer_product).squeeze(1)
        # # combined = F.relu(combined)

        # Pass through the Hebbian linear layers with ReLU and Dropout
        for layer in self.linear_layers:
            combined = layer(combined)
            combined = F.sigmoid(combined)
            # combined = self.dropout(combined)

        # Add the residual (original combined tensor) to the output of the layers
        # print(f"residual_shape: {residual.shape}, combined shape: {combined.shape}")
        if self.residual_connection:
            combined += residual

        # Split into hidden and output
        hidden = self.i2h(combined)
        hidden = torch.zeros_like(hidden)
        output = self.i2o(combined)
        # hidden = (1.0/hidden.shape[1])*torch.tanh(hidden)  # Apply tanh function to keep hidden from blowing up after many recurrences, as well as control for magnitude of recurrent connection
        hidden = torch.tanh(hidden)  # Apply tanh function to keep hidden from blowing up after many recurrences

        output.requires_grad = True
        # output = self.dropout(output)  # Apply dropout to the output before softmax
        # output = self.softmax(output)
        return output, hidden

    def initHidden(self, batch_size):
        device = next(self.parameters()).device
        return torch.zeros(batch_size, self.hidden_size, device=device, requires_grad=False)

    def apply_imprints(self, reward, learning_rate, plast_learning_rate, plast_clip, imprint_rate, stochasticity):
        # Apply imprints for all HebbianLinear layers
        # self.linear1.apply_imprints(reward, learning_rate, plast_learning_rate, plast_clip, imprint_rate, stochasticity)
        # self.linear2.apply_imprints(reward, learning_rate, plast_learning_rate, plast_clip, imprint_rate, stochasticity)
        for layer in self.linear_layers:
            layer.apply_imprints(reward, learning_rate, plast_learning_rate, plast_clip, imprint_rate, stochasticity)
        # self.i2h.apply_imprints(reward, learning_rate, plast_learning_rate, plast_clip, imprint_rate, stochasticity)
        self.i2o.apply_imprints(reward, learning_rate, plast_learning_rate, plast_clip, imprint_rate, stochasticity)


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

    #     # Initialize weights
    #     self.init_weights()

    # def init_weights(self):
    #     for layer in self.linear_layers:
    #         self._init_weight(layer)

    #     self._init_weight(self.i2h)
    #     self._init_weight(self.i2o)

    # def _init_weight(self, layer):
    #     if self.init_type == 'zero':
    #         nn.init.zeros_(layer.weight)
    #         if layer.bias is not None:
    #             nn.init.zeros_(layer.bias)
    #     elif self.init_type == 'orthogonal':
    #         nn.init.orthogonal_(layer.weight)
    #         if layer.bias is not None:
    #             nn.init.zeros_(layer.bias)

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

    # def initHidden(self):
    #     return torch.zeros(1, self.hidden_size)

    def initHidden(self, batch_size):
        device = next(self.parameters()).device
        return torch.zeros(batch_size, self.hidden_size, device=device)


if __name__ == "__main__":
    # Example instantiation of HebbianLinear
    layer = HebbianLinear(in_features=10, out_features=5)

    # Checking if the shapes are the same
    print("Shape of weights:", layer.weight.shape)
    print("Shape of imprints:", layer.imprints.shape)
    print("Are the shapes identical?", layer.weight.shape == layer.imprints.shape)

    # Generate random data
    input_data = torch.randn(3, 10)  # Batch size of 3, input features 10

    # Pass data through the HebbianLinear layer
    output = layer(input_data)

    print("Weights:\n ", layer.weight)
    layer.apply_imprints(reward=0.5, learning_rate=0.1, plast_learning_rate=0.01, plast_clip=2,  imprint_rate=0.1)
    print("Weights after imprint:\n ", layer.weight)

    # Ensure the input size matches the number of features for each input
    input_size = 70
    output_size = 70
    n_hidden = 128
    rnn = HebbyRNN(input_size, n_hidden, output_size,3)

    # Define the loss function (criterion) and optimizer
    criterion = torch.nn.NLLLoss()
    # optimizer = torch.optim.Adam(rnn.parameters(), lr=0.005)





    # In your training loop, after the weight update step
    # clip_weights(rnn, max_norm=0.5)  # Choose an appropriate max_norm value