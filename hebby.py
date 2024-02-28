import torch
from hebbian_model import HebbianLinear, SimpleRNN, clip_weights
from vis import register_hooks, save_model_data, visualize_model_data, visualize_all_layers_and_save, create_animation_from_visualizations
import wandb
import matplotlib.pyplot as plt
from preprocess import load_and_preprocess_data
from utils import randomTrainingExample, timeSince, str2bool, idx_to_char, n_characters
import time
import math
import argparse

def train(line_tensor, onehot_line_tensor, rnn, config, state):
    hidden = rnn.initHidden()
    losses, og_losses, reg_losses = [], [], []
    l2_reg, output = None, None

    for i in range(onehot_line_tensor.size()[0] - 1):
        l2_reg = None  # Reset L2 regularization term for each character

        hot_input_char_tensor = onehot_line_tensor[i].unsqueeze(0)

        with torch.no_grad():  # Disable gradient calculations
            # Forward pass through the RNN
            output, hidden = rnn(hot_input_char_tensor, hidden)

            # Compute the loss for this step
            final_char = line_tensor[i+1].unsqueeze(0)
            loss = config['criterion'](output, final_char)

            # Compute the L2 regularization term
            for param in rnn.parameters():
                if l2_reg is None:
                    l2_reg = param.norm(2)  # L2 norm of the parameter
                else:
                    l2_reg = l2_reg + param.norm(2)
            og_losses.append(loss.item())  # Store the original loss for this step
            reg_losses.append(config['l2_lambda']*l2_reg.item())
            loss = loss + config['l2_lambda'] * l2_reg  # Add the L2 regularization term to the loss
            losses.append(loss.item())  # Store the loss for this step

            # Convert loss to a reward signal for Hebbian updates
            reward = -loss.item()
            state['last_n_rewards'].append(reward)
            if len(state['last_n_rewards']) > config['len_reward_history']:
                state['last_n_rewards'].pop(0)
            last_n_reward_avg = sum(state['last_n_rewards']) / len(state['last_n_rewards'])
            reward_update = reward - last_n_reward_avg

            # Apply Hebbian updates to the network
            rnn.apply_imprints(reward_update, config["learning_rate"], config["imprint_rate"], config["stochasticity"])

        if (state["training_instance"] % config["save_frequency"] == 0 and state['training_instance'] != 0):
            # Save the model and activations periodically
            save_model_data(rnn, state["activations"], state["training_instance"], config['track'])

        state['training_instance'] += 1

    # Calculate the average loss for the sequence
    loss_avg = sum(losses) / len(losses)
    og_loss_avg = sum(og_losses) / len(og_losses)
    reg_loss_avg = sum(reg_losses) / len(reg_losses)
    return output, loss_avg, og_loss_avg, reg_loss_avg

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
    parser.add_argument('--track', type=str2bool, nargs='?', const=True, default=True, help='Whether to track progress online.')
    
    # Add other parameters as needed
    args = parser.parse_args()
    config = {
        "learning_rate": args.learning_rate,
        "imprint_rate": args.imprint_rate,
        "stochasticity": args.stochasticity,
        "len_reward_history": args.len_reward_history,
        "save_frequency": args.save_frequency,
        "criterion": torch.nn.NLLLoss(),
        "l2_lambda": 0.000,  # Example static hyperparameter
        "n_hidden": args.hidden_size,
        "n_layers": args.num_layers,
        "track": args.track,
    }
    print(args.track)
    if args.track:
        # wandb initialization
        wandb.init(project="hebby", config={
            "learning_rate": args.learning_rate,
            "architecture": "crazy hebbian thing",
            "dataset": "TinyStories",
            "epochs": 1,
            "stochasticity": args.stochasticity,
            "imprint_rate": args.imprint_rate,
            "len_reward_history": args.len_reward_history,
            "update_rule": args.update_rule,
            "normalize": args.normalize
        })

    # Load data
    dataloader = load_and_preprocess_data()

    # Model Initialization
    input_size = n_characters
    output_size = n_characters
    rnn = SimpleRNN(input_size, config["n_hidden"], output_size, config["n_layers"], normalize=args.normalize, update_rule=args.update_rule)

    rnn.train()
    state = {
        "training_instance": 0,
        "last_n_rewards": [0],
        "last_n_reward_avg": 0,
        "activations": register_hooks(rnn)
    }

    # Training Loop
    current_loss = 0
    all_losses = []
    start = time.time()

    for iter in range(1, args.n_iters + 1):
        sequence, line_tensor, onehot_line_tensor = randomTrainingExample(dataloader)
        output, loss, og_loss, reg_loss = train(line_tensor, onehot_line_tensor, rnn, config, state)
        # Loss tracking
        current_loss += loss
        if iter % args.print_freq == 0:
            # Print training progress
            topv, topi = output.topk(1, dim=1)
            predicted_char = idx_to_char[topi[0, 0].item()]
            target_char = sequence[-1]
            correct = '✓' if predicted_char == target_char else '✗ (%s)' % target_char
            sequence = sequence[-50:]
            if args.track:
                wandb.log({"loss": loss, "correct": correct, "predicted_char": predicted_char, "target_char": target_char, "sequence": sequence})
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / args.n_iters * 100, timeSince(start), loss, sequence, predicted_char, correct))
        if iter % args.plot_freq == 0:
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