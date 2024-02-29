import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys

# Apply Clipping
def clip_weights(model, max_norm):
    with torch.no_grad():
        for param in model.parameters():
            param.data.clamp_(-max_norm, max_norm)

class HebbianLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, normalize=True, clip_weights=False, update_rule='damage', alpha=0.5):
        super(HebbianLinear, self).__init__(in_features, out_features, bias)
        self.imprints = nn.Parameter(torch.zeros_like(self.weight), requires_grad=False)
        self.normalize = normalize
        self.clip_weights = clip_weights
        self.update_rule = update_rule

        if update_rule == 'covariance':
            self.alpha = alpha  # Decay factor for the exponential moving average
            self.in_traces = nn.Parameter(torch.zeros(in_features), requires_grad=False)
            self.out_traces = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        
        if update_rule == 'candidate':
            self.candidate_weights = nn.Parameter(torch.zeros_like(self.weight), requires_grad=False)

    def forward(self, input):
        # print(input)
        output = super(HebbianLinear, self).forward(input)
        self.update_imprints(input, output)
        # print(output)
        return output

    def update_imprints(self, input, output):
        # print("input shape:", input.shape)
        # print("output shape:", output.shape)
    
        # Hebbian update rule: imprint = input * output
        # Adjusting to compute the required [5, 10] imprint matrix for each batch
        # Reshape input and output for broadcasting
        input_expanded = input.unsqueeze(1)  # Shape: [batch_size, 1, in_features]
        output_expanded = output.unsqueeze(2)  # Shape: [batch_size, out_features, 1]
        # if torch.isnan(input_expanded).any():
        #     print("the nan is in the input_expanded")
        #     sys.exit(1)

        if self.update_rule == 'damage':
            # Element-wise multiplication with broadcasting
            # Results in a [batch_size, out_features, in_features] tensor
            imprint_update = output_expanded * input_expanded

            # for p in imprint_update:
            #     if torch.isnan(p.data).any():
            #         print("the nan is in the initial imprint_update in the learning rule")
            #         sys.exit(1)
            
            # Compute the difference and square it
            diff_squared = (output_expanded - input_expanded) ** 2
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
            imprint_update = (output - self.out_traces).unsqueeze(2) * (input - self.in_traces).unsqueeze(1)
            self.imprints.data += imprint_update.sum(dim=0)

            # Update the running averages (traces) for inputs and outputs
            self.in_traces.data = self.alpha * self.in_traces.data + (1 - self.alpha) * input.mean(dim=0)
            self.out_traces.data = self.alpha * self.out_traces.data + (1 - self.alpha) * output.mean(dim=0)
        elif self.update_rule == 'hpca':
            # hpca - Hebbian Principal Component Analysis (HPCA) rule: Updates weights based on the input and the output, 
            # subtracting the reconstructed input from all previous neurons
            # (y_i * (x - Σ(y_j * w_j) for j=1 to i)).

            # Outer product of output and weights for all neurons
            outer_product = output.unsqueeze(2) * self.imprints.unsqueeze(0)
            
            # Compute cumulative sum for the reconstruction term
            reconstruction = torch.cumsum(outer_product, dim=1)

            # Compute the weight update for all neurons
            imprint_update = (input.unsqueeze(1) - reconstruction) * output.unsqueeze(2)
            # self.imprints.data += imprint_update.sum(dim=0)
        elif self.update_rule == 'candidate':
            # only evaluate past weight updates with the current reward signal
            imprint_update = self.candidate_weights.data

            # Reset or decay candidate_weights
            self.candidate_weights.data *= 0.9  # Example: decay by half is 0.5

            candidate_update = output_expanded*(input_expanded - output_expanded * self.weight.data) # oja's rule, reused
            self.candidate_weights.data += candidate_update.sum(dim=0)

        # Sum over the batch dimension to get the final imprint update
        self.imprints.data = imprint_update.sum(dim=0)

    def apply_imprints(self, reward, learning_rate, imprint_rate, stochasticity):
        # Check if reward is NaN
        # if math.isnan(reward):
        #     print("Warning: Reward is NaN")
        #     sys.exit(1)
        imprint_update = self.imprints.data
        # ltd_threshold=0.6
        # # Calculate LTD adjustments
        # # For imprints less than the threshold, subtract their absolute value from the threshold
        # ltd_adjustment = torch.where(imprint_update < ltd_threshold, 
        #                             ltd_threshold - imprint_update.abs(), 
        #                             imprint_update)

        # # Reapply the original sign of the imprint update
        # update = torch.where(imprint_update < 0, 
        #                                 -ltd_adjustment, 
        #                                 ltd_adjustment)

        # this actually clips the imprint updates, not the weights themselves
        # if self.clip_weights:
        #     max_weight_value = 0.1
        #     for p in self.imprints:
        #         # if torch.isnan(p.data).any():
        #         #     print("the nan is in the imprint")
        #         #     sys.exit(1)
        #         p.data.clamp_(-max_weight_value, max_weight_value)

        update = reward * learning_rate * imprint_update + reward * imprint_rate * imprint_update
        # for p in update:
        #     if torch.isnan(p.data).any():
        #         print(f"the nan is in the update, here's the reward: {reward} and here's the imprint_update:\n{imprint_update}")
        #         sys.exit(1)
        # clip the update
        # update = torch.clamp(update, -0.1, 0.1)
        # print("update:", update)
        self.weight.data += update

        # for p in self.weight:
        #     if torch.isnan(p.data).any():
        #         print("the nan is in the weights, post-update")
        #         sys.exit(1)

        if self.normalize:
            # Normalize the weights to prevent them from exploding
            for p in self.parameters():
                p.data = p.data / (p.data.norm(2) + 1e-6)
        
        if self.clip_weights:
            max_weight_value = 0.1
            for p in self.parameters():
                p.data.clamp_(-max_weight_value, max_weight_value)

        # Apply stochastic noise to the weights
        for p in self.parameters():
            noise = stochasticity * torch.randn_like(p.data)
            p.data += noise

class HebbyRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_rate=0.1, init_type='zero', normalize=True, clip_weights=False, update_rule='damage'):
        super(HebbyRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.init_type = init_type

        # Using HebbianLinear instead of Linear
        self.linear_layers = torch.nn.ModuleList([HebbianLinear(input_size + hidden_size, hidden_size, normalize=normalize, clip_weights=clip_weights, update_rule=update_rule)])
        for _ in range(1, num_layers):
            self.linear_layers.append(HebbianLinear(hidden_size, hidden_size, normalize=normalize, clip_weights=clip_weights, update_rule=update_rule))

        # Dropout layers
        self.dropout = nn.Dropout(dropout_rate)

        # Final layers for hidden and output, also using HebbianLinear
        self.i2h = HebbianLinear(hidden_size, hidden_size, normalize=normalize, clip_weights=clip_weights, update_rule=update_rule)
        self.i2o = HebbianLinear(hidden_size, output_size, normalize=normalize, clip_weights=clip_weights, update_rule=update_rule)
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
        combined = torch.cat((input, hidden), dim=1)

        # Pass through the Hebbian linear layers with ReLU and Dropout
        for layer in self.linear_layers:
            combined = layer(combined)
            combined = F.relu(combined)
            combined = self.dropout(combined)

        # Split into hidden and output
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        hidden = torch.tanh(hidden)  # Apply tanh function to keep hidden from blowing up after many recurrences
        output = self.dropout(output)  # Apply dropout to the output before softmax
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

    def apply_imprints(self, reward, learning_rate, imprint_rate, stochasticity):
        # Apply imprints for all HebbianLinear layers
        for layer in self.linear_layers:
            layer.apply_imprints(reward, learning_rate, imprint_rate, stochasticity)
        self.i2h.apply_imprints(reward, learning_rate, imprint_rate, stochasticity)
        self.i2o.apply_imprints(reward, learning_rate, imprint_rate, stochasticity)


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
        combined = torch.cat((input, hidden), dim=1)

        # Pass through the linear layers with ReLU and Dropout
        for layer in self.linear_layers:
            combined = layer(combined)
            combined = F.relu(combined)
            combined = self.dropout(combined)

        # Split into hidden and output
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        hidden = torch.tanh(hidden)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)



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
    layer.apply_imprints(reward=0.5, learning_rate=0.1, imprint_rate=0.1)
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
    clip_weights(rnn, max_norm=0.5)  # Choose an appropriate max_norm value