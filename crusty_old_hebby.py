# %%
import torch
from datasets import load_dataset
import numpy as np

# Load dataset
dataset = load_dataset("roneneldan/TinyStories")
dataset = dataset['train'].select(range(1000))

# Your charset
charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,.;:'\"?!\n- "
char_to_idx = {char: idx for idx, char in enumerate(charset)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}
n_characters = len(charset)

# Define processing functions
def filter_text(examples):
    """Filter out characters not in the charset for a batch of texts."""
    return {'text': [''.join([char for char in text if char in charset]) for text in examples['text']]}

# Define processing function to convert text to indices
def text_to_indices(examples):
    tensors = [torch.tensor([char_to_idx[char] for char in text], dtype=torch.long) for text in examples['text']]
    return {'tensor': tensors}


# Apply the processing functions using .map
print("mapping the filter")
dataset = dataset.map(filter_text,batched=True)
print("mapping text to indices")
dataset = dataset.map(text_to_indices, batched=True)
print('preprocessed') 

# Define a simple collate_fn
def collate_fn(batch):
    # Extract the single item in the batch
    item = batch[0]
    text = item['text']
    tensor = item['tensor']
    tensor = torch.tensor(tensor)
    return text, tensor

# Function to get a random training example from the DataLoader
def randomTrainingExample(dataloader):
    for text, tensor in dataloader:
        return text, tensor



# Create a DataLoader
dataloader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=1, 
    shuffle=True, 
    collate_fn=collate_fn
)

# Get a random training example
random_sequence, random_tensor = randomTrainingExample(dataloader)
print("Random Sequence:", random_sequence)
print("Random Tensor:", random_tensor)
print("random tensor type: ", type(random_tensor[0]))

# %%
from hebbian_model import HebbianLinear, SimpleRNN, clip_weights
# Import the functions for saving model data
from vis import register_hooks, save_model_data

input_size = n_characters
output_size = n_characters
n_hidden = 128
rnn = SimpleRNN(input_size, n_hidden, output_size, 1)

# Zero Initialization
for p in rnn.parameters():
    torch.nn.init.zeros_(p)

# Orthogonal Initialization (particularly good for RNNs)
# for p in rnn.parameters():
#     if p.dim() > 1:
#         nn.init.orthogonal_(p)

# Register hooks to capture activations
activations = register_hooks(rnn)

# Initialize a counter for training instances
training_instance = 0

# Define the loss function (criterion)
criterion = torch.nn.NLLLoss()

# %%
learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn
imprint_rate = 0.00
last_n_rewards = [0]
last_n_reward_avg = 0
n_rewards = 1000
stochasticity = 0.0001
save_frequency = 1000000  # Save data every 10k characters
rnn.train()
def train(line_tensor, l2_lambda=0.000):
    global training_instance, last_n_rewards, last_n_reward_avg
    hidden = rnn.initHidden()  # Initialize the hidden state of the RNN

    losses = []  # To store losses for each step in the sequence
    og_losses = []  # To store the original losses for each step in the sequence
    reg_losses = []  # To store the L2 regularization losses for each step in the sequence
    l2_reg = None  # For accumulating the L2 regularization term
    output = None  # To store the output of the RNN

    for i in range(line_tensor.size()[0] - 1):
        l2_reg = None  # Reset L2 regularization term for each character
        # Convert the current character to a one-hot encoded tensor
        hot_input_char_tensor = torch.nn.functional.one_hot(line_tensor[i], num_classes=n_characters).type(torch.float).unsqueeze(0)

        with torch.no_grad():  # Disable gradient calculations
            # Forward pass through the RNN
            output, hidden = rnn(hot_input_char_tensor, hidden)

            # Compute the loss for this step
            loss = criterion(output, line_tensor[-1].unsqueeze(0))

            # Compute the L2 regularization term
            for param in rnn.parameters():
                if l2_reg is None:
                    l2_reg = param.norm(2)  # L2 norm of the parameter
                else:
                    l2_reg = l2_reg + param.norm(2)
            og_losses.append(loss.item())  # Store the original loss for this step
            reg_losses.append(l2_lambda*l2_reg.item())
            loss = loss + l2_lambda * l2_reg  # Add the L2 regularization term to the loss
            losses.append(loss.item())  # Store the loss for this step

            # Convert loss to a reward signal for Hebbian updates
            reward = -loss.item()
            last_n_rewards.append(reward)
            if len(last_n_rewards) > n_rewards:
                last_n_rewards.pop(0)
            last_n_reward_avg = sum(last_n_rewards) / len(last_n_rewards)
            reward_update = reward - last_n_reward_avg

            # Apply Hebbian updates to the network
            rnn.apply_imprints(reward_update, learning_rate, imprint_rate, stochasticity)

        if (training_instance % save_frequency == 0):
            # Save the model and activations periodically
            save_model_data(rnn, activations, training_instance)
        training_instance += 1

    # Calculate the average loss for the sequence
    loss_avg = sum(losses) / len(losses)
    og_loss_avg = sum(og_losses) / len(og_losses)
    reg_loss_avg = sum(reg_losses) / len(reg_losses)
    return output, loss_avg, og_loss_avg, reg_loss_avg



# %%
import wandb
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="hebby",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": learning_rate,
    "architecture": "crazy hebbian thing",
    "dataset": "wikidump",
    "epochs": 1,
    "stochasticity": stochasticity,
    "imprint_rate": imprint_rate,
    "n_rewards": n_rewards,
    }
)

# %%
import time
import math

n_iters = 10000
print_every = 50
plot_every = 20

# Keep track of losses for plotting
current_loss = 0
current_og_loss = 0
current_reg_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for iter in range(1, n_iters + 1):
    sequence, line_tensor = randomTrainingExample(dataloader)
    output, loss, og_loss, reg_loss = train(line_tensor)
    current_loss += loss
    current_og_loss += og_loss
    current_reg_loss += reg_loss
    # Check if loss is NaN
    if math.isnan(loss):
        print("Loss is NaN, breaking...")
        parameters = list(rnn.parameters())
        for p in parameters:
            print(p.data)
        break
    # Print ``iter`` number, loss, name and guess
    if iter % print_every == 0:
        # Use the output to generate a character prediction
        topv, topi = output.topk(1, dim=1)  # Change dim to 1
        predicted_char = idx_to_char[topi[0, 0].item()]
        target_char = sequence[-1]
        correct = '✓' if predicted_char == target_char else '✗ (%s)' % target_char
        wandb.log({"correct": correct, "predicted_char": predicted_char, "target_char": target_char, "sequence": sequence})
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, sequence, predicted_char, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        # Use the output to generate a character prediction
        topv, topi = output.topk(1, dim=1)  # Change dim to 1
        predicted_char = idx_to_char[topi[0, 0].item()]
        target_char = sequence[-1]
        correct = '✓' if predicted_char == target_char else '✗ (%s)' % target_char
        wandb.log({"loss": current_loss / plot_every, "og_loss": current_og_loss / plot_every, "reg_loss": current_reg_loss / plot_every,"correct": correct, "predicted_char": predicted_char, "target_char": target_char, "sequence": sequence})
        all_losses.append(current_loss / plot_every)
        current_loss = 0
        current_og_loss = 0
        current_reg_loss = 0

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()

# %%
import matplotlib.pyplot as plt

plt.figure()
plt.plot(all_losses)

# %%
from vis import visualize_model_data
# Now visualize the data for a specific layer and instance
# Replace 'layer_name' with the actual name of the layer you want to visualize
visualize_model_data('i2h', 0)


# %%
from vis import visualize_all_layers_and_save

visualize_all_layers_and_save(rnn, 0, "jusone.png")

# %%
from vis import create_animation_from_visualizations

# create_animation_from_visualizations(rnn, 'model_data', 'model_evolution.gif', format='gif')

# %%
create_animation_from_visualizations(rnn, 'model_data', 'model_evolution.mp4', format='mp4')


