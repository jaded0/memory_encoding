import torch
import torch.nn as nn
import torch.nn.functional as F


# Apply Clipping
def clip_weights(model, max_norm):
    with torch.no_grad():
        for param in model.parameters():
            param.data.clamp_(-max_norm, max_norm)

class HebbianLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, normalize=True, update_rule='damage'):
        super(HebbianLinear, self).__init__(in_features, out_features, bias)
        self.imprints = nn.Parameter(torch.zeros_like(self.weight))
        self.normalize = normalize
        self.update_rule = update_rule

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

        if self.update_rule == 'damage':
            # Element-wise multiplication with broadcasting
            # Results in a [batch_size, out_features, in_features] tensor
            imprint_update = output_expanded * input_expanded

            # Compute the difference and square it
            diff_squared = (output_expanded - input_expanded) ** 2

            # Update the imprint using the new rule: oa*ia - (oa-ia)^2
            imprint_update = imprint_update - diff_squared
        elif self.update_rule == 'oja':
            imprint_update = output_expanded*(input_expanded - output_expanded * self.imprints.data)
        # Sum over the batch dimension to get the final imprint update
        self.imprints.data = imprint_update.sum(dim=0)

    def apply_imprints(self, reward, learning_rate, imprint_rate, stochasticity):

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

        update = reward * learning_rate * imprint_update + reward * imprint_rate * imprint_update
        # clip the update
        # update = torch.clamp(update, -0.1, 0.1)
        # print("update:", update)
        self.weight.data += update
        if self.normalize:
            # Normalize the weights to prevent them from exploding
            for p in self.parameters():
                p.data = p.data / (p.data.norm(2) + 1e-6)

        # Apply stochastic noise to the weights
        for p in self.parameters():
            noise = stochasticity * torch.randn_like(p.data)
            p.data += noise

class SimpleRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_rate=0.1, init_type='zero', normalize=True, update_rule='damage'):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.init_type = init_type

        # Using HebbianLinear instead of Linear
        self.linear_layers = torch.nn.ModuleList([HebbianLinear(input_size + hidden_size, hidden_size, normalize=normalize, update_rule=update_rule)])
        for _ in range(1, num_layers):
            self.linear_layers.append(HebbianLinear(hidden_size, hidden_size, normalize=normalize, update_rule=update_rule))

        # Dropout layers
        self.dropout = nn.Dropout(dropout_rate)

        # Final layers for hidden and output, also using HebbianLinear
        self.i2h = HebbianLinear(hidden_size, hidden_size, normalize=normalize, update_rule=update_rule)
        self.i2o = HebbianLinear(hidden_size, output_size, normalize=normalize, update_rule=update_rule)
        self.softmax = torch.nn.LogSoftmax(dim=1)
        # Initialize weights
        self.init_weights()

    def init_weights(self):
        for layer in self.linear_layers:
            self._init_weight(layer)

        self._init_weight(self.i2h)
        self._init_weight(self.i2o)

    def _init_weight(self, layer):
        if self.init_type == 'zero':
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif self.init_type == 'orthogonal':
            nn.init.orthogonal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

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
    rnn = SimpleRNN(input_size, n_hidden, output_size,3)

    # Define the loss function (criterion) and optimizer
    criterion = torch.nn.NLLLoss()
    # optimizer = torch.optim.Adam(rnn.parameters(), lr=0.005)





    # In your training loop, after the weight update step
    clip_weights(rnn, max_norm=0.5)  # Choose an appropriate max_norm value