import torch
from hebbian_model import HebbianLinear, HebbyRNN, SimpleRNN
from vis import register_hooks, save_model_data, visualize_model_data, visualize_all_layers_and_save, create_animation_from_visualizations
import wandb
import matplotlib.pyplot as plt
from preprocess import load_and_preprocess_data
from utils import randomTrainingExample, timeSince, str2bool, idx_to_char, n_characters
import time
import math
import argparse
import time
import sys
import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(x.dtype)

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Mean Squared Error loss function and its derivative
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred).numpy() ** 2)

def mse_loss_derivative(y_true, y_pred):
    # return 2 * (y_pred - y_true.numpy()) / y_true.numpy().size
    global_error = 2.0 / y_true.numpy().size * (y_pred - y_true.numpy())
    return global_error

# Neural Network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros(output_size)
        self.random_feedback = np.random.randn(self.W2.shape[1], self.W1.shape[1])

    def forward(self, X):
        X.numpy()
        # Forward pass
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        return torch.from_numpy(self.Z2).float()

    def backpropagation(self, X, error, learning_rate):
        # Backpropagation
        # error = mse_loss_derivative(y, y_pred)
        dW2 = np.dot(self.A1.T, error)
        db2 = np.sum(error, axis=0)

        error_hidden = np.dot(error, self.W2.T) * relu_derivative(self.A1)
        dW1 = np.dot(X.T, error_hidden)
        db1 = np.sum(error_hidden, axis=0)

        # Update weights and biases
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def direct_feedback_alignment(self, X, y, y_pred, error, learning_rate, reroll=True):
        # Direct Feedback Alignment
        # error = mse_loss_derivative(y, y_pred)
        if reroll==True:
          self.random_feedback = np.random.randn(self.W2.shape[1], self.W1.shape[1])
        error_hidden = np.dot(error, self.random_feedback) * relu_derivative(self.A1)

        # Calculate weight updates
        dW1 = np.dot(X.T, error_hidden)
        db1 = np.sum(error_hidden, axis=0)
        print(f"A1 shape: {self.A1.T.shape}, error shape: {error.shape}")
        dW2 = np.dot(self.A1.T, error)
        print(f"dW2 shape: {dW2.shape}, W2 shape: {self.W2.shape}")
        db2 = np.sum(error, axis=0)

        # Update weights and biases
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2


def train_backprop(line_tensor, onehot_line_tensor, rnn, config, optimizer):
    criterion = config['criterion']

    hidden = rnn.initHidden()
    rnn.zero_grad()
    loss = 0

    for i in range(onehot_line_tensor.size()[0] - 1): 
        hot_input_char_tensor = onehot_line_tensor[i].unsqueeze(0)
        output, nhidden = rnn(hot_input_char_tensor, hidden)
        final_char = onehot_line_tensor[i+1].unsqueeze(0).float()
        # loss += criterion(output, final_char)
        loss = criterion(output, final_char)

        loss.backward()
        # optimizer.step()
        for p in rnn.parameters():
            if p.grad is not None:
                p.data.add_(p.grad.data, alpha=-config['learning_rate'])

    return output, loss.item()/onehot_line_tensor.size()[0], 0, 0

def train_hebby(line_tensor, onehot_line_tensor, rnn, config, state):
    hidden = rnn.initHidden()
    losses, og_losses, reg_losses = [], [], []
    l2_reg, output = None, None

    for i in range(onehot_line_tensor.size()[0] - 1):
        l2_reg = None  # Reset L2 regularization term for each character

        hot_input_char_tensor = onehot_line_tensor[i].unsqueeze(0)

        # with torch.no_grad():  # Disable gradient calculations
        # Forward pass through the RNN
        output, hidden = rnn(hot_input_char_tensor, hidden)
        # output = rnn.forward(hot_input_char_tensor)

        # Compute the loss for this step
        final_char = onehot_line_tensor[i+1].squeeze(0)
        # print(final_char)
        noutput = output.squeeze(0)
        # if isinstance(output, np.ndarray):
        #     noutput = torch.from_numpy(noutput).float() 
        loss = config['criterion'](final_char, noutput)
        # loss = mse_loss(final_char, output)
        # print(f"old loss: {old_loss}, new loss: {loss}")
        # if math.isnan(loss):
        #     print("Warning: Loss is NaN")
        #     sys.exit(1)
        # Compute the L2 regularization term
        # for param in rnn.parameters():
        #     if l2_reg is None:
        #         l2_reg = param.norm(2)  # L2 norm of the parameter
        #     else:
        #         l2_reg = l2_reg + param.norm(2)
        # og_losses.append(loss.item())  # Store the original loss for this step
        # reg_losses.append(config['l2_lambda']*l2_reg.item())
        # loss = loss + config['l2_lambda'] * l2_reg  # Add the L2 regularization term to the loss
        losses.append(loss.item())  # Store the loss for this step

        # Convert loss to a reward signal for Hebbian updates
        if config["delta_rewards"]:
            reward = -loss.item()
            state['last_n_rewards'].append(reward)
            if len(state['last_n_rewards']) > config['len_reward_history']:
                state['last_n_rewards'].pop(0)
            last_n_reward_avg = sum(state['last_n_rewards']) / len(state['last_n_rewards'])
            reward_update = reward - last_n_reward_avg
        else:
            # global_error = torch.autograd.grad(loss, output, retain_graph=True)
            # global_error = global_error[0][0]
            # print(global_error)
            # global_error = 2.0 / final_char.numpy().size * (output.numpy() - final_char.numpy())
            global_error = 2.0 / final_char.size()[-1] * (output - final_char)
            # global_error = 2.0 / onehot_line_tensor.size()[1] * (output - final_char)
            reward_update = -global_error
        
        # Apply Hebbian updates to the network
        rnn.apply_imprints(reward_update, config["learning_rate"], config["imprint_rate"], config["stochasticity"])
        # rnn.direct_feedback_alignment(hot_input_char_tensor.numpy(), final_char, output, global_error.numpy(), config['learning_rate'])
        # rnn.direct_feedback_alignment(hot_input_char_tensor.numpy(), final_char, output, global_error.numpy(), config['learning_rate'], reroll=False)
        # rnn.backpropagation(hot_input_char_tensor.numpy(), final_char, output, config['learning_rate'])

        # if (state["training_instance"] % config["save_frequency"] == 0 and state['training_instance'] != 0):
        #     # Save the model and activations periodically
        #     save_model_data(rnn, state["activations"], state["training_instance"], config['track'])

        state['training_instance'] += 1

    # Calculate the average loss for the sequence
    loss_avg = sum(losses) / onehot_line_tensor.size()[0]
    # og_loss_avg = sum(og_losses) / len(og_losses)
    # reg_loss_avg = sum(reg_losses) / len(reg_losses)
    og_loss_avg, reg_loss_avg = 0,0 # i don wanna refactor rn
    return output, loss_avg, og_loss_avg, reg_loss_avg

def train(line_tensor, onehot_line_tensor, rnn, config, state, optimizer=None):
    if config['update_rule'] == "backprop":
        return train_backprop(line_tensor, onehot_line_tensor, rnn, config, optimizer)
    else:
        return train_hebby(line_tensor, onehot_line_tensor, rnn, config, state)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a model with specified hyperparameters.')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate for the optimizer')
    parser.add_argument('--imprint_rate', type=float, default=0.00, help='Imprint rate for Hebbian updates')
    parser.add_argument('--stochasticity', type=float, default=0.0001, help='Stochasticity in Hebbian updates')
    parser.add_argument('--len_reward_history', type=int, default=1000, help='Number of rewards to track')
    parser.add_argument('--save_frequency', type=int, default=100000, help='How often to save and display model weights.')
    parser.add_argument('--hidden_size', type=int, default=128, help='Size of hidden layers in RNN')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers in RNN')
    parser.add_argument('--n_iters', type=int, default=10000, help='Number of training iterations')
    parser.add_argument('--print_freq', type=int, default=50, help='Frequency of printing training progress')
    parser.add_argument('--plot_freq', type=int, default=20, help='Frequency of plotting training loss')
    parser.add_argument('--update_rule', type=str, default='damage', help='How to update weights.')
    parser.add_argument('--normalize', type=str2bool, nargs='?', const=True, default=True, help='Whether to normalize the weights.')
    parser.add_argument('--clip_weights', type=float, default=1, help='Whether to clip the weights.')
    parser.add_argument('--track', type=str2bool, nargs='?', const=True, default=True, help='Whether to track progress online.')
    parser.add_argument('--delta_rewards', type=str2bool, nargs='?', const=True, default=True, help='Whether to calculate rewards by change in reward instead.')
    parser.add_argument('--dataset', type=str, default='roneneldan/tinystories', help='The dataset used for training.')
    
    # Add other parameters as needed
    args = parser.parse_args()
    config = {
        "learning_rate": args.learning_rate,
        "imprint_rate": args.imprint_rate,
        "stochasticity": args.stochasticity,
        "len_reward_history": args.len_reward_history,
        "save_frequency": args.save_frequency,
        "criterion": torch.nn.MSELoss(),
        # "criterion": torch.nn.CrossEntropyLoss(),
        "l2_lambda": 0.000,  # Example static hyperparameter
        "n_hidden": args.hidden_size,
        "n_layers": args.num_layers,
        "track": args.track,
        "dataset": args.dataset,
        "update_rule": args.update_rule,
        "delta_rewards": args.delta_rewards,
    }
    print(args.track)
    if args.track:
        # wandb initialization
        wandb.init(project="hebby", config={
            "learning_rate": args.learning_rate,
            "architecture": "crazy hebbian thing",
            "dataset": args.dataset,
            "epochs": 1,
            "stochasticity": args.stochasticity,
            "imprint_rate": args.imprint_rate,
            "len_reward_history": args.len_reward_history,
            "update_rule": args.update_rule,
            "normalize": args.normalize,
            "clip_weights": args.clip_weights,
            "delta_rewards": args.delta_rewards,
        })

    # Load data
    dataloader = load_and_preprocess_data(args.dataset)

    # Model Initialization
    input_size = n_characters
    output_size = n_characters
    
    optimizer = None

    if args.update_rule == "backprop":
        rnn = SimpleRNN(input_size, config["n_hidden"], output_size, config["n_layers"])
        optimizer = torch.optim.Adam(rnn.parameters(), lr=config['learning_rate'])
    else:
        rnn = HebbyRNN(input_size, config["n_hidden"], output_size, config["n_layers"], normalize=args.normalize, clip_weights=args.clip_weights, update_rule=args.update_rule)
        # rnn = NeuralNetwork(input_size=n_characters, hidden_size=128, output_size=n_characters)


    # rnn.train()
    state = {
        "training_instance": 0,
        "last_n_rewards": [0],
        "last_n_reward_avg": 0,
        # "activations": register_hooks(rnn)
    }

    # Training Loop
    current_loss = 0
    all_losses = []
    start = time.time()

    for iter in range(1, args.n_iters + 1):
        sequence, line_tensor, onehot_line_tensor = randomTrainingExample(dataloader)
        # print(sequence)
        output, loss, og_loss, reg_loss = train(line_tensor, onehot_line_tensor, rnn, config, state, optimizer)
        # Loss tracking
        current_loss += loss
        if iter % args.plot_freq == 0:
            # Print training progress
            topv, topi = output.topk(1, dim=1)
            predicted_char = idx_to_char[topi[0, 0].item()]
            target_char = sequence[-1]
            correct = '✓' if predicted_char == target_char else '✗ (%s)' % target_char
            sequence = sequence[-50:]
            if args.track:
                wandb.log({"loss": loss, "avg_loss": current_loss / args.plot_freq, "correct": correct, "predicted_char": predicted_char, "target_char": target_char, "sequence": sequence})
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / args.n_iters * 100, timeSince(start), loss, sequence, predicted_char, correct))
            all_losses.append(current_loss / args.plot_freq)
            current_loss = 0

    if args.track:
        wandb.finish()

    # Plotting the Training Loss
    # plt.figure()
    # plt.plot(all_losses)
    # plt.title("Training Loss Over Time")
    # plt.xlabel("Iterations")
    # plt.ylabel("Loss")
    # plt.show()

    # # Visualization and Animations
    # visualize_model_data('i2h', 0)
    # visualize_all_layers_and_save(rnn, 0, "jusone.png")
    create_animation_from_visualizations(rnn, 'model_data', 'model_evolution.mp4', format='mp4')

if __name__ == '__main__':
    main()