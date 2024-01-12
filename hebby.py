# %%
from dataset_creation import TextDataset
from torch.utils.data import DataLoader
import numpy as np

# Instantiate the dataset
text_dataset = TextDataset(directory='data/SPGC-tokens-2018-07-18/', sequence_length=100)
print(f"Dataset created with {len(text_dataset)} sequences.")

# Create a DataLoader without a sampler
dataloader = DataLoader(text_dataset, batch_size=1)

# Iterate over a few batches and print their contents
for i, (sequences, inputs) in enumerate(dataloader):
    if i >= 2:  # Adjust this value to see more/less batches
        break

    print(f"\nBatch {i+1}")
    print(f"Inputs shape: {inputs.shape}")

    # Optionally print the actual sequences (comment out if too verbose)
    sequence = ''.join([text_dataset.idx_to_char[int(idx)] for idx in inputs[0]])
    # target = text_dataset.idx_to_char[int(targets[0])]
    print(f"Sequence: {sequence}")


# %%
import torch

# Define chars using keys of char_to_idx
chars = list(text_dataset.char_to_idx.keys())

n_characters = len(chars)  # Number of unique characters
print(f"Number of unique characters: {n_characters}")
print(f"Characters: {chars}")

# %%
import torch.nn as nn

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



    def apply_imprints(self, reward, learning_rate):
        # Apply the imprints to the weights
        # self.weight.data += reward * learning_rate * self.imprints
        imprint_update = self.imprints.data
        # Normalize the imprint updates
        norm_imprint_update = imprint_update - imprint_update.mean()
        # print("norm_imprint_update:", norm_imprint_update)

        # Apply the normalized imprints
        # The reward can be positive (for LTP) or negative (for LTD)
        self.weight.data += reward * learning_rate * norm_imprint_update


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
layer.apply_imprints(reward=0.5, learning_rate=0.1)
print("Weights after imprint:\n ", layer.weight)

# %%
import torch.nn.functional as F

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
        print(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

    def apply_imprints(self, reward, learning_rate):
        # Apply imprints for all HebbianLinear layers
        for layer in self.linear_layers:
            layer.apply_imprints(reward, learning_rate)
        self.i2h.apply_imprints(reward, learning_rate)
        self.i2o.apply_imprints(reward, learning_rate)


# Ensure the input size matches the number of features for each input
input_size = n_characters
output_size = n_characters
n_hidden = 128
rnn = SimpleRNN(input_size, n_hidden, output_size,3)

# Define the loss function (criterion) and optimizer
criterion = torch.nn.NLLLoss()
# optimizer = torch.optim.Adam(rnn.parameters(), lr=0.005)



# Apply Clipping
def clip_weights(model, max_norm):
    with torch.no_grad():
        for param in model.parameters():
            param.data.clamp_(-max_norm, max_norm)

# In your training loop, after the weight update step
clip_weights(rnn, max_norm=0.5)  # Choose an appropriate max_norm value


# %%
import torch

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return text_dataset.char_to_idx[letter]

# Just for demonstration, turn a letter into a <1 x n_characters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_characters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_characters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_characters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

print(letterToTensor('J'))

print(lineToTensor('Jones').size())

# %%
text_dataset[3]

# %%
def randomTrainingExample():
    """Generate a random training example from the dataset"""
    sequence, line_tensor = text_dataset[np.random.randint(len(text_dataset))]
    return sequence, line_tensor

randomTrainingExample()

# %%
learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

def train(line_tensor):
    hidden = rnn.initHidden()
    rnn.zero_grad()

    for i in range(line_tensor.size()[0] - 1):
        hot_input_char_tensor = torch.nn.functional.one_hot(line_tensor[i], num_classes=n_characters).type(torch.float).unsqueeze(0)
        output, hidden = rnn(hot_input_char_tensor, hidden)

    # print("output shape:", output.shape)
    # print("line_tensor shape:", line_tensor.shape)
    # print(output)
    # print(line_tensor[-1].unsqueeze(0))
    loss = criterion(output, line_tensor[-1].unsqueeze(0))
    # print(loss)

    # Convert loss to a reward signal
    reward = 1 / (1 + loss.item())  # Example conversion, assuming loss is non-negative
    # print(reward)
    
    clip_weights(rnn, max_norm=0.5)  # Choose an appropriate max_norm value

    # Apply Hebbian updates
    rnn.apply_imprints(reward, learning_rate)

    # Perform backward pass and optimizer step if using gradient descent for other parameters
    # loss.backward()
    # optimizer.step()

    return output, loss.item()


# %%
import time
import math

n_iters = 100000
print_every = 50
plot_every = 10



# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for iter in range(1, n_iters + 1):
    sequence, line_tensor = randomTrainingExample()
    output, loss = train(line_tensor)
    current_loss += loss

    # Print ``iter`` number, loss, name and guess
    if iter % print_every == 0:
        # Use the output to generate a character prediction
        topv, topi = output.topk(1, dim=1)  # Change dim to 1
        predicted_char = text_dataset.idx_to_char[topi[0, 0].item()]
        target_char = sequence[-1]
        correct = '✓' if predicted_char == target_char else '✗ (%s)' % target_char
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, sequence, predicted_char, correct))

        # also print some weights:
        # print("i2h weights:", rnn.i2h.weight)
        # print("i2o weights:", rnn.i2o.weight)

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

# %%
import matplotlib.pyplot as plt

plt.figure()
plt.plot(all_losses)

# %%



