import torch
import torch.nn as nn
import torch.nn.functional as F


# Apply Clipping
def clip_weights(model, max_norm):
    with torch.no_grad():
        for param in model.parameters():
            param.data.clamp_(-max_norm, max_norm)

class HebbianLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(HebbianLinear, self).__init__(in_features, out_features, bias)
        self.imprints = nn.Parameter(torch.zeros_like(self.weight))

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

        # Element-wise multiplication with broadcasting
        # Results in a [batch_size, out_features, in_features] tensor
        imprint_update = output_expanded * input_expanded

        # Sum over the batch dimension to get the final imprint update
        self.imprints.data = imprint_update.sum(dim=0)

    def apply_imprints(self, reward, learning_rate, imprint_rate):
        # Apply the imprints to the weights
        # self.weight.data += reward * learning_rate * self.imprints
        imprint_update = self.imprints.data
        # print("norm_imprint_update:", norm_imprint_update)

        # Apply the normalized imprints
        # The reward can be positive (for LTP) or negative (for LTD)
        self.weight.data += reward * learning_rate * imprint_update + reward * imprint_rate * imprint_update

class SimpleRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Using HebbianLinear instead of Linear
        self.linear_layers = torch.nn.ModuleList([HebbianLinear(input_size + hidden_size, hidden_size)])
        for _ in range(1, num_layers):
            self.linear_layers.append(HebbianLinear(hidden_size, hidden_size))

        # Final layers for hidden and output, also using HebbianLinear
        self.i2h = HebbianLinear(hidden_size, hidden_size)
        self.i2o = HebbianLinear(hidden_size, output_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), dim=1)

        # Pass through the Hebbian linear layers with ReLU
        for layer in self.linear_layers:
            combined = layer(combined)
            combined = F.relu(combined)

        # Split into hidden and output
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        # print(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

    def apply_imprints(self, reward, learning_rate, imprint_rate):
        # Apply imprints for all HebbianLinear layers
        for layer in self.linear_layers:
            layer.apply_imprints(reward, learning_rate, imprint_rate)
        self.i2h.apply_imprints(reward, learning_rate, imprint_rate)
        self.i2o.apply_imprints(reward, learning_rate, imprint_rate)




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