from dataset_creation import TextDataset
from torch.utils.data import DataLoader
import torch

# Instantiate the dataset
text_dataset = TextDataset(directory='data/SPGC-tokens-2018-07-18/', sequence_length=100)
print(f"Dataset created with {len(text_dataset)} sequences.")

# Create a DataLoader without a sampler
dataloader = DataLoader(text_dataset, batch_size=1)

# Define chars using keys of char_to_idx
chars = list(text_dataset.char_to_idx.keys())

n_characters = len(chars)  # Number of unique characters
print(f"Number of unique characters: {n_characters}")
print(f"Characters: {chars}")

# Define the RNN model
class SimpleRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = torch.nn.Linear(input_size + hidden_size, output_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)  # Update dim to 1 for batch processing

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), dim=1)  # Change dimension to 1

        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

# Ensure the input size matches the number of features for each input
input_size = n_characters
output_size = n_characters
n_hidden = 128
rnn = SimpleRNN(input_size, n_hidden, output_size)

# Define the loss function (criterion) and optimizer
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.005)

# Apply Gradient Clipping
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.005)
torch.nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=1.0)  # Clip gradients during training

def train(input_line_tensor, target_char_tensor, hidden):
    rnn.zero_grad()

    # Process the entire input sequence
    output, hidden = rnn(input_line_tensor, hidden)  # No need for loop here

    # Modify the target tensor shape
    target_char_tensor = target_char_tensor.view(-1)

    # Compute the loss
    loss = criterion(output.view(-1, len(chars)), target_char_tensor)
    loss.backward()
    optimizer.step()

    return output, hidden, loss.item()

# Training loop
for epoch in range(1, 101):
    for batch_idx, (inputs) in enumerate(dataloader):
        # Initialize variables to store the history and predicted characters for each batch
        history = []
        predicted_chars = []

        input_line_tensor = inputs[0]  # Get the first character
        hidden = rnn.initHidden()  # Initialize hidden state once per sequence
        
        for char_idx in range(input_line_tensor.shape[0] - 1):
            # Convert to one-hot encoding for each character
            hot_input_char_tensor = torch.nn.functional.one_hot(input_line_tensor[char_idx], num_classes=n_characters).type(torch.float)
            
            # Train and calculate loss for each character
            target_char = input_line_tensor[char_idx + 1].unsqueeze(0)
            output, hidden, loss = train(hot_input_char_tensor.unsqueeze(0), target_char, hidden)
           
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Char {char_idx} Loss: {loss}')

            if batch_idx % 1000 == 0:
                # Use the output to generate a character prediction
                topv, topi = output.topk(1, dim=1)  # Change dim to 1
                predicted_char = text_dataset.idx_to_char[topi[0, 0].item()]
                target_char = text_dataset.idx_to_char[target_char.item()]

                # Append the current character and prediction to their respective lists
                history.append(target_char)
                predicted_chars.append(predicted_char)

                # Display the summarized history
                history_str = ''.join(history)
                predicted_str = ''.join(predicted_chars)
                print(f'Epoch {epoch}, Batch {batch_idx}, Char {char_idx}\ntarget: {target_char} predicted: {predicted_char}\nHistory: "{history_str}", Predicted: "{predicted_str}"')
