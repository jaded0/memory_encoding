import torch
from hebbian_model import HebbianLinear, HebbyRNN, SimpleRNN
# from vis import register_hooks, save_model_data, visualize_model_data, visualize_all_layers_and_save, create_animation_from_visualizations
import wandb
import matplotlib.pyplot as plt
from preprocess import load_and_preprocess_data
from utils import randomTrainingExample, timeSince, str2bool, initialize_charset
import time
import math
import argparse
import sys
import numpy as np
import itertools
import os
import psutil
from wandb_osh.hooks import TriggerWandbSyncHook  # <-- New!

trigger_sync = TriggerWandbSyncHook()  # <--- New!

def print_graph(g, level=0):
    if g is None: return
    print(' ' * level * 4, g)
    for sub_g in g.next_functions:
        if sub_g[0] is not None:
            print_graph(sub_g[0], level+1)

def train_backprop(line_tensor, onehot_line_tensor, rnn, config, optimizer, log_outputs=False):
    criterion = config['criterion']

    batch_size = onehot_line_tensor.shape[0]
    hidden = rnn.initHidden(batch_size=batch_size)
    rnn.zero_grad()
    loss_total = 0
    # losses = []

    all_outputs = []
    all_labels = []

    for i in range(onehot_line_tensor.size()[1] - 1): 
        hot_input_char_tensor = onehot_line_tensor[:, i, :] # overcomplicated only bc batching
        pe_matrix = config["pe_matrix"]
        if pe_matrix is not None:
            # position i might exceed MAX_SEQ_LEN, so clamp or wrap as needed
            pe_vec = pe_matrix[min(i, pe_matrix.size(0)-1)]  # shape [pos_dim]
            # expand to [batch_size, pos_dim]
            pe_vec = pe_vec.unsqueeze(0).expand(batch_size, -1)
            # concat onto the onehot
            hot_input_char_tensor = torch.cat([hot_input_char_tensor, pe_vec], dim=1)

        output, hidden = rnn(hot_input_char_tensor, hidden)
        final_char = onehot_line_tensor[:, i+1, :]
        loss_total += criterion(output, final_char)
        # loss = criterion(output, final_char)
        # losses.append(loss.item())

        if log_outputs:
            all_outputs.append(output[0])
            all_labels.append(final_char[0])

    loss_total.mean().backward()
    # optimizer.step()
    for p in rnn.parameters():
        if p.grad is not None:
            p.data.add_(p.grad.data, alpha=-config['learning_rate'])

    loss_sum = loss_total.mean().item()/onehot_line_tensor.shape[1]
    # loss_sum = sum(losses) / onehot_line_tensor.size()[0]
    return output, loss_sum, 0, 0, all_outputs, all_labels

def train_hebby(line_tensor, onehot_line_tensor, rnn, config, state, log_outputs=False):
    batch_size = onehot_line_tensor.shape[0]
    # print(f"batch size: {batch_size}")
    hidden = rnn.initHidden(batch_size=batch_size)
    losses, og_losses, reg_losses = [], [], []
    l2_reg, output = None, None
    # print(f"full tensor shape: {onehot_line_tensor.shape}")
    all_outputs = []
    all_labels = []

    rnn.wipe()

    for i in range(onehot_line_tensor.shape[1] - 1):
        l2_reg = None  # Reset L2 regularization term for each character

        hot_input_char_tensor = onehot_line_tensor[:, i, :] # overcomplicated only bc batching
        hot_input_char_tensor.requires_grad = False

        pe_matrix = config["pe_matrix"]
        if pe_matrix is not None:
            # position i might exceed MAX_SEQ_LEN, so clamp or wrap as needed
            pe_vec = pe_matrix[min(i, pe_matrix.size(0)-1)]  # shape [pos_dim]
            # regularize to onehot magnitude
            # pe_vec *= 1/onehot_line_tensor.shape[1]
            # expand to [batch_size, pos_dim]
            pe_vec = pe_vec.unsqueeze(0).expand(batch_size, -1)
            # concat onto the onehot
            hot_input_char_tensor = torch.cat([hot_input_char_tensor, pe_vec], dim=1)

        # with torch.no_grad():  # Disable gradient calculations
        # Forward pass through the RNN
        # print(hot_input_char_tensor.shape, hidden.shape)
        # print(f"require_grad. hot_input_char_tensor: {hot_input_char_tensor.requires_grad}, hidden: {hidden.requires_grad}")
        output, hidden, self_grad = rnn(hot_input_char_tensor, hidden)
        # print(f"does output require grad? {output.requires_grad}. hidden? {hidden.requires_grad}")
        # output.requires_grad=True

        # Compute the loss for this step
        final_char = onehot_line_tensor[:, i+1, :]
        # print(final_char)
        # print(f"shapes. final char: {final_char.shape}, noutput: {noutput.shape}")
        # if isinstance(output, np.ndarray):
        #     noutput = torch.from_numpy(noutput).float() 
        # if log_outputs:
        #     charset, char_to_idx, idx_to_char, n_characters = initialize_charset("jbrazzy/baby_names")
        #     print(f"so I guess that's {idx_to_char.get(torch.argmax(output[0]).item())}")
        #     print(f"versus {idx_to_char.get(torch.argmax(final_char[0]).item())}")
        #     print(f"what's given was {idx_to_char.get(torch.argmax(hot_input_char_tensor[0]).item())}")
        loss = config['criterion'](output, final_char)
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
        losses.append(loss[0].item())  # Store the loss for this step

        # Convert loss to a reward signal for Hebbian updates
        if config["delta_rewards"]:
            reward = -loss.item()
            state['last_n_rewards'].append(reward)
            if len(state['last_n_rewards']) > config['len_reward_history']:
                state['last_n_rewards'].pop(0)
            last_n_reward_avg = sum(state['last_n_rewards']) / len(state['last_n_rewards'])
            reward_update = reward - last_n_reward_avg
        else:
            global_error = torch.autograd.grad(loss, output, grad_outputs=torch.ones_like(loss), retain_graph=False)[0]
            # global_error = gglobal_error[0]
            reward_update = -global_error
            rnn.zero_grad()
        
        # Apply Hebbian updates to the network
        # threshold = 1000000
        # # lr = 1e-3 if state["training_instance"] < threshold else config["learning_rate"]
        # lr = 1e-3 if loss[0].item() < 1.7 else config["learning_rate"]
        # # plast_clip = 1e3 if state["training_instance"] < threshold else config["plast_clip"]
        # if state["training_instance"] == threshold:
        #     print(f"reached threshold at {threshold}")
        lr = config["learning_rate"]
        plast_clip = config["plast_clip"]
        reward_update += torch.clamp(self_grad, min=-config["self_grad"], max=config["self_grad"])
        rnn.apply_imprints(reward_update, lr, config["plast_learning_rate"], plast_clip, config["imprint_rate"], config["stochasticity"], config["grad_clip"])

        # if (state["training_instance"] % config["save_frequency"] == 0 and state['training_instance'] != 0):
        #     # Save the model and activations periodically
        #     print("---\nsaving activations\n---")
        #     save_model_data(rnn, state["activations"], state["training_instance"], config['track'])

        state['training_instance'] += 1

        if log_outputs:
            all_outputs.append(output[0])
            all_labels.append(final_char[0])

    # Calculate the average loss for the sequence
    loss_avg = sum(losses) / onehot_line_tensor.shape[1]
    # og_loss_avg = sum(og_losses) / len(og_losses)
    # reg_loss_avg = sum(reg_losses) / len(reg_losses)
    og_loss_avg, reg_loss_avg = 0,0 # i don wanna refactor rn
    return output, loss_avg, og_loss_avg, reg_loss_avg, all_outputs, all_labels

def train(line_tensor, onehot_line_tensor, rnn, config, state, optimizer=None, log_outputs=False):
    if config['update_rule'] == "backprop":
        return train_backprop(line_tensor, onehot_line_tensor, rnn, config, optimizer, log_outputs)
    else:
        return train_hebby(line_tensor, onehot_line_tensor, rnn, config, state, log_outputs)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a model with specified hyperparameters.')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate for the optimizer')
    parser.add_argument('--plast_learning_rate', type=float, default=0.005, help='Learning rate for the plasticity')
    parser.add_argument('--plast_clip', type=float, default=0.005, help='How high the plasticity can go.')
    parser.add_argument('--imprint_rate', type=float, default=0.00, help='Imprint rate for Hebbian updates')
    parser.add_argument('--forget_rate', type=float, default=0.00, help='Forget rate, forgetting factor, prevents explosion.')
    parser.add_argument('--stochasticity', type=float, default=0.0001, help='Stochasticity in Hebbian updates')
    parser.add_argument('--len_reward_history', type=int, default=1000, help='Number of rewards to track')
    parser.add_argument('--save_frequency', type=int, default=100000, help='How often to save and display model weights.')
    parser.add_argument('--residual_connection', type=str2bool, nargs='?', const=True, default=True, help='whether to have a skip connection')
    parser.add_argument('--grad_clip', type=float, default=1e-1, help='Clip gradients to this value.') 
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
    parser.add_argument('--notes', type=str, default='nothing to say', help='talk about this run')
    parser.add_argument('--candecay', type=float, default=0.9, help='Decay of the candidate weights, each step.')
    parser.add_argument('--plast_candecay', type=float, default=0.5, help='Decay rate for plastic candidate weights')
    parser.add_argument('--group', type=str, default="nothing_in_particular", help='Description of what sort of experiment is being run, here.')
    parser.add_argument('--batch_size', type=int, default=4, help='how much to stuff in at once')
    parser.add_argument('--positional_encoding_dim', type=int, default=0,
                        help='Dimension for optional positional encoding (0 means off).')
    parser.add_argument('--self_grad', type=float, default=0.0, help='Scale of self_grad. grad based replacement for recurrence.')

    # grab slurm jobid if it exists.
    job_id = os.environ.get("SLURM_JOB_ID") if os.environ.get("SLURM_JOB_ID") else "no_SLURM"
    print("SLURM Job ID:", job_id)

    args = parser.parse_args()
    config = {
        "learning_rate": args.learning_rate,
        "plast_learning_rate": args.plast_learning_rate,
        "plast_clip": args.plast_clip,
        "imprint_rate": args.imprint_rate,
        "forget_rate": args.forget_rate,
        "stochasticity": args.stochasticity,
        "len_reward_history": args.len_reward_history,
        "save_frequency": args.save_frequency,
        # "criterion": torch.nn.MSELoss(),
        "criterion": torch.nn.CrossEntropyLoss(reduction='none'),
        "l2_lambda": 0.000,  # Example static hyperparameter
        "residual_connection": args.residual_connection,
        "grad_clip": args.grad_clip,
        "n_hidden": args.hidden_size,
        "n_layers": args.num_layers,
        "track": args.track,
        "dataset": args.dataset,
        "update_rule": args.update_rule,
        "delta_rewards": args.delta_rewards,
        "candecay": args.candecay,
        "plast_candecay": args.plast_candecay,
        "batch_size": args.batch_size,
        "self_grad": args.self_grad,
    }
    print(args.track)
    if args.track:
        # wandb initialization
        wandb.init(project="hebby", group=args.group, notes=args.notes, config={
            "learning_rate": args.learning_rate,
            "plast_learning_rate": args.plast_learning_rate,
            "plast_clip": args.plast_clip,
            "architecture": "crazy hebbian thing",
            "update_rule": args.update_rule,
            "residual_connection": args.residual_connection,
            "grad_clip": args.grad_clip,
            "n_hidden": args.hidden_size,
            "n_layers": args.num_layers,
            "dataset": args.dataset,
            "epochs": 1,
            "stochasticity": args.stochasticity,
            "imprint_rate": args.imprint_rate,
            "forget_rate": args.forget_rate,
            "len_reward_history": args.len_reward_history,
            "normalize": args.normalize,
            "clip_weights": args.clip_weights,
            "delta_rewards": args.delta_rewards,
            "candecay": args.candecay,
            "plast_candecay": args.plast_candecay,
            "plot_frequency": args.plot_freq,
            "batch_size": args.batch_size,
            "slurm_id": job_id,
            "positional_encoding_dim": args.positional_encoding_dim,
            "save_frequency": args.save_frequency,
            "self_grad": args.self_grad,
        })

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    charset, char_to_idx, idx_to_char, n_characters = initialize_charset(args.dataset)
    # Ensure that idx_to_char contains all necessary mappings
    print(f"Character set size: {n_characters}")
    dataloader = load_and_preprocess_data(args.dataset, args.batch_size)
    
    # Decide a max sequence length to support
    MAX_SEQ_LEN = 2000  # or any upper bound you expect
    pos_dim = args.positional_encoding_dim
    # print(f'pos_dim: {1/pos_dim}')
    # print(f'alt: {(30/pos_dim)}')
    if pos_dim > 0:
        # Precompute a [MAX_SEQ_LEN, pos_dim] matrix
        pe_matrix = torch.zeros(MAX_SEQ_LEN, pos_dim)
        for pos in range(MAX_SEQ_LEN):
            for dim in range(pos_dim):
                # A typical sinusoidal formula:
                #   angle = pos / (1e4 ** (2*(dim//2)/pos_dim))
                # We'll do something simpler for brevity:
                pe_matrix[pos, dim] = math.sin(pos + dim*0.1) * (8/pos_dim)

        # Move it to GPU if needed
        pe_matrix = pe_matrix.to(device)
        config["pe_matrix"] = pe_matrix
    else:
        config["pe_matrix"] = None

    # Model Initialization
    input_size = n_characters + pos_dim
    output_size = n_characters
    
    optimizer = None

    if args.update_rule == "backprop":
        rnn = SimpleRNN(input_size, config["n_hidden"], output_size, config["n_layers"])
        optimizer = torch.optim.Adam(rnn.parameters(), lr=config['learning_rate'])
    else:
        rnn = HebbyRNN(input_size, config["n_hidden"], output_size, config["n_layers"], charset, normalize=args.normalize, residual_connection=args.residual_connection, clip_weights=args.clip_weights, update_rule=args.update_rule, candecay=config["candecay"], plast_candecay=config["plast_candecay"], plast_clip=config["plast_clip"], batch_size=config["batch_size"], forget_rate=config["forget_rate"])

    if torch.cuda.is_available():
        print("cuda available!")
        rnn = rnn.to(device)

    # rnn.train()
    state = {
        "training_instance": 0,
        "last_n_rewards": [0],
        "last_n_reward_avg": 0,
        # "activations": register_hooks(rnn)
    }

    # Training Loop
    current_loss = 0
    current_correct = 0
    all_losses = []
    start = time.time()
    step_start = time.time()
    try:
        for iter, (sequence, line_tensor, onehot_line_tensor) in enumerate(itertools.cycle(dataloader), 1):
            if iter > args.n_iters:
                break

            line_tensor = line_tensor.to(device)
            onehot_line_tensor = onehot_line_tensor.to(device)

            log_outputs = (iter % (args.plot_freq * 5) == 0)
            output, loss, og_loss, reg_loss, all_outputs, all_labels = train(line_tensor, onehot_line_tensor, rnn, config, state, optimizer, log_outputs)
            output = output.detach()
            topv, topi = output.topk(2, dim=1)  # Get the top 2 predictions
            correct_answer = line_tensor[0][-1]  # Get the correct answer

            # Check if the correct answer is either the top-1 or top-2 prediction
            if topi[0][0] == correct_answer:
                corr = 1  # Correct prediction as the top pick
            elif topi[0][1] == correct_answer:
                corr = 0.5  # Correct prediction as the second highest
            else:
                corr = 0  # Incorrect prediction

            # Loss tracking
            current_loss += loss
            current_correct += corr
            if iter % (args.plot_freq * 5) == 0:  # Logging less frequently
                topv, topi = output.topk(1, dim=1)
                predicted_char = idx_to_char[topi[0, 0].item()]
                target_char = sequence[0][-1]
                iscorrect = predicted_char == target_char
                correct = '✓' if iscorrect else '✗ (%s)' % target_char
                sequence = sequence[0]

                # Check for invalid indices before logging
                valid_outputs = [idx_to_char.get(torch.argmax(o).item(), None) for o in all_outputs]
                valid_outputs_2 = [idx_to_char.get(torch.topk(o, 2).indices[1].item(), None) for o in all_outputs]
                valid_labels = [idx_to_char.get(torch.argmax(l).item(), None) for l in all_labels]

                # Filter out None values
                valid_outputs = [char for char in valid_outputs if char is not None]

                # if args.track:
                #     log_data = {
                #         "loss": loss.item() if hasattr(loss, "item") else loss,
                #         "avg_loss": float(current_loss) / (args.plot_freq * 5),
                #         "correct": float(current_correct) / (args.plot_freq * 5),
                #         "predicted_char": str(predicted_char),
                #         "target_char": str(target_char),
                #         "sequence": str(sequence),
                #         "was_correct": float(current_correct) / (args.plot_freq * 5),
                #     }
                #     wandb.log(log_data, commit=True)
                #     # Optionally flush wandb's internal buffers (mostly useful in offline mode):
                #     if hasattr(wandb.run, "_flush"):
                #         wandb.run._flush()
                if args.track:
                    wandb.log({
                        "loss": loss, 
                        "avg_loss": current_loss / (args.plot_freq * 5), 
                        "correct": current_correct / (args.plot_freq * 5), 
                        "predicted_char": predicted_char, 
                        "target_char": target_char, 
                        "sequence": sequence,
                        "was_correct": current_correct / (args.plot_freq * 5),
                        # "all_outputs": valid_outputs,
                        # "all_labels": valid_labels
                    })

                # Print the progression of the entire sequence
                progression = ''.join(valid_outputs)  # Convert list of characters to a string
                progression2 = ''.join(valid_outputs_2)
                print(f'{iter} {iter / args.n_iters * 100:.2f}% ({timeSince(start)}) {loss:.4f} avg: {(current_loss/(args.plot_freq*5)):.5f} Sequence: \n\033[91m{sequence}\033[0m\n \033[93m{progression}\033[0m {correct}\n \033[35m{progression2}\033[0m')
                # print(f"Memory usage: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")
                all_losses.append(current_loss / (args.plot_freq * 5))
                current_loss = 0
                current_correct = 0
                # all_outputs.clear()
                # all_labels.clear()
                # gc.collect()
                # torch.cuda.empty_cache()
            if iter % (args.print_freq * 5) == 0:
                trigger_sync()


    except KeyboardInterrupt:
        print("\nok, finishing up..")

    if args.track:
        wandb.finish()
    # create_animation_from_visualizations(rnn, 'model_data', 'model_evolution.mp4', format='mp4')


if __name__ == '__main__':
    main()
