# hebby.py
import torch
from hebbian_model import HebbianLinear, HebbyRNN, SimpleRNN
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
# from memory_profiler import profile

trigger_sync = TriggerWandbSyncHook()  # <--- New!

def train_backprop(line_tensor, onehot_line_tensor, rnn, config, optimizer, log_outputs=False):
    criterion = config['criterion']

    batch_size = onehot_line_tensor.shape[0]
    hidden = rnn.initHidden(batch_size=batch_size)
    rnn.zero_grad()
    loss_total = 0

    all_outputs = []
    all_labels = []
    n_characters = onehot_line_tensor.shape[2] # Get n_characters from the tensor shape

    for i in range(onehot_line_tensor.size()[1] - 1):
        # Get current character's one-hot vector
        current_char_tensor = onehot_line_tensor[:, i, :]

        # --- Start Modification: Conditional Input Construction ---
        if config['input_mode'] == 'last_two':
            # Get previous character's one-hot vector (or zeros if i=0)
            if i == 0:
                previous_char_tensor = torch.zeros_like(current_char_tensor)
            else:
                previous_char_tensor = onehot_line_tensor[:, i-1, :]
            # Concatenate current and previous character tensors
            combined_char_tensor = torch.cat([current_char_tensor, previous_char_tensor], dim=1)
        elif config['input_mode'] == 'last_one':
            # Use only the current character
            combined_char_tensor = current_char_tensor
        else:
            raise ValueError(f"Invalid input_mode: {config['input_mode']}")
        # --- End Modification ---

        # Handle positional encoding (concatenated onto the combined_char_tensor)
        pe_matrix = config["pe_matrix"]
        if pe_matrix is not None:
            # position i might exceed MAX_SEQ_LEN, so clamp or wrap as needed
            pe_vec = pe_matrix[min(i, pe_matrix.size(0)-1)]  # shape [pos_dim]
            # expand to [batch_size, pos_dim]
            pe_vec = pe_vec.unsqueeze(0).expand(batch_size, -1)
            # concat positional encoding onto the character tensor
            hot_input_char_tensor = torch.cat([combined_char_tensor, pe_vec], dim=1)
        else:
            # If no positional encoding, the input is just the combined characters
            hot_input_char_tensor = combined_char_tensor

        output, hidden = rnn(hot_input_char_tensor, hidden)
        final_char = onehot_line_tensor[:, i+1, :]
        loss_total += criterion(output, final_char)

        if log_outputs:
            all_outputs.append(output[0]) # Note: Only logs the first item in the batch
            all_labels.append(final_char[0]) # Note: Only logs the first item in the batch

    loss_total = loss_total.mean() # Average loss across batch and sequence length
    loss_total.backward()
    # optimizer.step() # Using manual update below
    for p in rnn.parameters():
        if p.grad is not None:
            # Apply gradient clipping before adding
            torch.nn.utils.clip_grad_norm_(p, config['grad_clip']) # Clip individual param gradients
            p.data.add_(p.grad.data, alpha=-config['learning_rate'])

    loss_avg = loss_total.item() # Already averaged
    # The original code returned loss per character, let's keep consistency if needed, though averaged loss is more standard
    # loss_sum = loss_total.mean().item() / onehot_line_tensor.shape[1] # This seems redundant if loss is already averaged

    return output, loss_avg, 0, 0, all_outputs, all_labels # Return averaged loss

def train_hebby(line_tensor, onehot_line_tensor, rnn, config, state, log_outputs=False):
    batch_size = onehot_line_tensor.shape[0]
    hidden = rnn.initHidden(batch_size=batch_size)
    losses = []
    output = None
    all_outputs = []
    all_labels = []

    rnn.wipe()
    n_characters = onehot_line_tensor.shape[2] # Get n_characters from the tensor shape

    for i in range(onehot_line_tensor.shape[1] - 1):
        l2_reg = None  # Reset L2 regularization term for each character

        # Get current character's one-hot vector
        current_char_tensor = onehot_line_tensor[:, i, :]
        current_char_tensor.requires_grad = False

        # --- Start Modification: Conditional Input Construction ---
        if config['input_mode'] == 'last_two':
            # Get previous character's one-hot vector (or zeros if i=0)
            if i == 0:
                previous_char_tensor = torch.zeros_like(current_char_tensor)
            else:
                previous_char_tensor = onehot_line_tensor[:, i-1, :]
            previous_char_tensor.requires_grad = False
            # Concatenate current and previous character tensors
            combined_char_tensor = torch.cat([current_char_tensor, previous_char_tensor], dim=1)
        elif config['input_mode'] == 'last_one':
             # Use only the current character
            combined_char_tensor = current_char_tensor
        else:
            raise ValueError(f"Invalid input_mode: {config['input_mode']}")
        # --- End Modification ---


        # Handle positional encoding (concatenated onto the combined_char_tensor)
        pe_matrix = config["pe_matrix"]
        if pe_matrix is not None:
            # position i might exceed MAX_SEQ_LEN, so clamp or wrap as needed
            pe_vec = pe_matrix[min(i, pe_matrix.size(0)-1)]  # shape [pos_dim]
            # regularize to onehot magnitude
            # pe_vec *= 1/onehot_line_tensor.shape[1]
            # expand to [batch_size, pos_dim]
            pe_vec = pe_vec.unsqueeze(0).expand(batch_size, -1)
            # concat positional encoding onto the character tensor
            hot_input_char_tensor = torch.cat([combined_char_tensor, pe_vec], dim=1)
        else:
            # If no positional encoding, the input is just the combined characters
            hot_input_char_tensor = combined_char_tensor

        output, hidden, self_grad = rnn(hot_input_char_tensor, hidden)

        # Compute the loss for this step
        final_char = onehot_line_tensor[:, i+1, :]
        # Make sure criterion reduction is 'none' to get per-batch-item loss
        loss = config['criterion'](output, final_char)
        losses.append(loss.detach()) # Store the loss tensor [batch_size] for this step

        # Convert loss to a reward signal for Hebbian updates
        # Ensure grad_outputs matches the shape of loss
        global_error = torch.autograd.grad(loss, output, grad_outputs=torch.ones_like(loss), retain_graph=False)[0]
        reward_update = -global_error
        rnn.zero_grad()

        lr = config["learning_rate"]
        plast_clip = config["plast_clip"]
        reward_update += torch.clamp(self_grad, min=-config["self_grad"], max=config["self_grad"])
        rnn.apply_imprints(reward_update, lr, config["plast_learning_rate"], plast_clip, config["imprint_rate"], config["grad_clip"])

        state['training_instance'] += 1

        if log_outputs:
            all_outputs.append(output[0].detach()) # Note: Only logs the first item in the batch
            all_labels.append(final_char[0].detach()) # Note: Only logs the first item in the batch


    # Calculate the average loss for the sequence (average over batch and time)
    if losses:
        # Stack losses: list of [B] tensors -> tensor [T, B]
        stacked_losses = torch.stack(losses)
        # Average over time (dim 0) and batch (dim 1)
        loss_avg = stacked_losses.mean().item()
    else:
        loss_avg = 0.0 # Handle edge case of very short sequences

    og_loss_avg, reg_loss_avg = 0,0 # i don wanna refactor rn
    return output, loss_avg, og_loss_avg, reg_loss_avg, all_outputs, all_labels


def train(line_tensor, onehot_line_tensor, rnn, config, state, optimizer=None, log_outputs=False):
    if config['update_rule'] == "backprop":
        return train_backprop(line_tensor, onehot_line_tensor, rnn, config, optimizer, log_outputs)
    else:
        # Ensure criterion reduction is 'none' for Hebby calculation of per-example gradient
        if config['criterion'].reduction != 'none':
             print("Warning: Overriding criterion reduction to 'none' for Hebby training.")
             config['criterion'] = type(config['criterion'])(reduction='none') # Recreate criterion with 'none'
        return train_hebby(line_tensor, onehot_line_tensor, rnn, config, state, log_outputs)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a model with specified hyperparameters.')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate for the optimizer')
    parser.add_argument('--plast_learning_rate', type=float, default=0.005, help='Learning rate for the plasticity')
    parser.add_argument('--plast_clip', type=float, default=0.005, help='How high the plasticity can go.')
    parser.add_argument('--imprint_rate', type=float, default=0.00, help='Imprint rate for Hebbian updates')
    parser.add_argument('--forget_rate', type=float, default=0.00, help='Forget rate, forgetting factor, prevents explosion.')
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
    parser.add_argument('--dataset', type=str, default='roneneldan/tinystories', help='The dataset used for training.')
    parser.add_argument('--notes', type=str, default='nothing to say', help='talk about this run')
    parser.add_argument('--group', type=str, default="nothing_in_particular", help='Description of what sort of experiment is being run, here.')
    parser.add_argument('--batch_size', type=int, default=4, help='how much to stuff in at once')
    parser.add_argument('--positional_encoding_dim', type=int, default=0,
                        help='Dimension for optional positional encoding (0 means off).')
    parser.add_argument('--self_grad', type=float, default=0.0, help='Scale of self_grad. grad based replacement for recurrence.')
    # --- Start Modification: Add input_mode argument ---
    parser.add_argument('--input_mode', type=str, default='last_two', choices=['last_one', 'last_two'],
                        help='Input mode: use last one or last two characters.')
    # --- End Modification ---

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
        "save_frequency": args.save_frequency,
        # Use 'mean' for backprop, will be overridden to 'none' in train() for Hebby
        "criterion": torch.nn.CrossEntropyLoss(reduction='mean'),
        "residual_connection": args.residual_connection,
        "grad_clip": args.grad_clip,
        "n_hidden": args.hidden_size,
        "n_layers": args.num_layers,
        "track": args.track,
        "dataset": args.dataset,
        "update_rule": args.update_rule,
        "batch_size": args.batch_size,
        "self_grad": args.self_grad,
        # --- Start Modification: Add input_mode to config ---
        "input_mode": args.input_mode,
        # --- End Modification ---
    }
    print(f"Input mode selected: {args.input_mode}") # Inform user
    if args.track:
        # wandb initialization
        wandb_config = {
            "learning_rate": args.learning_rate,
            "plast_learning_rate": args.plast_learning_rate,
            "plast_clip": args.plast_clip,
            "architecture": "crazy hebbian thing" if args.update_rule != "backprop" else "SimpleRNN", # Adjusted architecture name
            "update_rule": args.update_rule,
            "residual_connection": args.residual_connection,
            "grad_clip": args.grad_clip,
            "n_hidden": args.hidden_size,
            "n_layers": args.num_layers,
            "dataset": args.dataset,
            "epochs": 1, # This seems fixed, maybe adjust?
            "imprint_rate": args.imprint_rate,
            "forget_rate": args.forget_rate,
            "normalize": args.normalize,
            "clip_weights": args.clip_weights,
            "plot_frequency": args.plot_freq,
            "batch_size": args.batch_size,
            "slurm_id": job_id,
            "positional_encoding_dim": args.positional_encoding_dim,
            "save_frequency": args.save_frequency,
            "self_grad": args.self_grad,
            # --- Start Modification: Add input_mode to wandb config ---
            "input_mode": args.input_mode,
            # --- End Modification ---
        }
        wandb.init(project="hebby", group=args.group, notes=args.notes, config=wandb_config)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    charset, char_to_idx, idx_to_char, n_characters = initialize_charset(args.dataset)
    # Ensure that idx_to_char contains all necessary mappings
    print(f"Character set size: {n_characters}")
    # Use drop_last=True if batch size doesn't divide dataset size evenly
    dataloader = load_and_preprocess_data(args.dataset, args.batch_size, drop_last=True)

    # Decide a max sequence length to support
    MAX_SEQ_LEN = 2000  # or any upper bound you expect
    pos_dim = args.positional_encoding_dim
    if pos_dim > 0:
        # Precompute a [MAX_SEQ_LEN, pos_dim] matrix
        pe_matrix = torch.zeros(MAX_SEQ_LEN, pos_dim)
        position = torch.arange(0, MAX_SEQ_LEN, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, pos_dim, 2).float() * (-math.log(10000.0) / pos_dim))
        pe_matrix[:, 0::2] = torch.sin(position * div_term)
        pe_matrix[:, 1::2] = torch.cos(position * div_term)
        # Normalize PE matrix? Optional, could help.
        # pe_matrix = F.normalize(pe_matrix, p=2, dim=1)

        # Move it to GPU if needed
        pe_matrix = pe_matrix.to(device)
        config["pe_matrix"] = pe_matrix
    else:
        config["pe_matrix"] = None

    # Model Initialization
    # --- Start Modification: Calculate input_size based on input_mode ---
    if args.input_mode == 'last_two':
        char_input_dim = n_characters * 2
    elif args.input_mode == 'last_one':
        char_input_dim = n_characters
    else:
        # This case should be prevented by argparse choices, but adding for safety
        raise ValueError(f"Invalid input_mode: {args.input_mode}")

    input_size = char_input_dim + pos_dim # Base input size from characters + positional encoding
    # --- End Modification ---

    output_size = n_characters
    print(f"Model Input Size: {input_size}, Hidden Size: {config['n_hidden']}, Output Size: {output_size}") # Log calculated size

    optimizer = None

    if args.update_rule == "backprop":
        # Note: SimpleRNN's internal structure assumes input -> hidden, not input -> combined
        # Let's adjust SimpleRNN or how input_size is used if needed.
        # Assuming SimpleRNN's first layer takes input_size + hidden_size
        # We need to adjust the SimpleRNN definition or how we call it if input_size logic differs.
        # --> Sticking to the provided SimpleRNN structure for now. It seems it *always* concatenates input+hidden.
        # --> We need to pass the 'base' input size (chars + PE) to SimpleRNN, not the already-combined size.
        base_input_size = input_size # The size calculated above (chars + PE)
        rnn = SimpleRNN(base_input_size, config["n_hidden"], output_size, config["n_layers"], dropout_rate=0) # Pass base_input_size
        optimizer = torch.optim.Adam(rnn.parameters(), lr=config['learning_rate'])
    else:
        # HebbyRNN takes the combined size (input+hidden) in its layers.
        # The calculated 'input_size' above is just the 'input' part fed at each step.
        # HebbyRNN internally combines it with hidden_size.
        base_input_size = input_size # The size calculated above (chars + PE)
        rnn = HebbyRNN(base_input_size, config["n_hidden"], output_size, config["n_layers"], charset,
                       normalize=args.normalize, residual_connection=args.residual_connection,
                       clip_weights=args.clip_weights, update_rule=args.update_rule,
                       plast_clip=config["plast_clip"], batch_size=config["batch_size"],
                       forget_rate=config["forget_rate"])

    if torch.cuda.is_available():
        print("cuda available!")
        rnn = rnn.to(device)

    state = {
        "training_instance": 0,
        "last_n_rewards": [0],
        "last_n_reward_avg": 0,
    }

    # Training Loop
    current_loss = 0
    current_correct = 0 # Using top-1 accuracy here for simplicity in aggregation
    all_losses = []
    start = time.time()

    def infinite_dataloader(dataloader):
        while True:
            for batch in dataloader:
                yield batch

    data_iterator = infinite_dataloader(dataloader)

    try:
        # Initialize accumulators for the less frequent PLOT interval (averaging)
        current_loss_plot_interval = 0.0
        current_correct_plot_interval = 0
        current_step_acc_t1_plot_interval = 0.0
        current_step_acc_t2_plot_interval = 0.0
        num_detailed_calcs_in_plot_interval = 0
        all_outputs = [] # Initialize lists outside loop, clear after use
        all_labels = []

        for iter in range(1, args.n_iters + 1):
            # Fetch next batch
            try:
                 sequence, line_tensor, onehot_line_tensor = next(data_iterator)
            except StopIteration: # Should not happen with infinite_dataloader, but good practice
                 print("DataLoader exhausted and restarted.") # Info message
                 data_iterator = infinite_dataloader(dataloader)
                 sequence, line_tensor, onehot_line_tensor = next(data_iterator)


            line_tensor = line_tensor.to(device)
            onehot_line_tensor = onehot_line_tensor.to(device)

            # Ensure batch size matches model expectation if using HebbyRNN with fixed batch size param
            if args.update_rule != "backprop" and hasattr(rnn, 'batch_size') and onehot_line_tensor.shape[0] != rnn.batch_size:
                 print(f"Warning: Batch size mismatch ({onehot_line_tensor.shape[0]} vs {rnn.batch_size}). Skipping batch.")
                 continue # Skip this batch

            # Determine if detailed outputs are needed (for the frequent PRINT interval)
            log_outputs_for_train = (iter % args.print_freq == 0)

            # --- Train Step ---
            output, loss, og_loss, reg_loss, current_all_outputs, current_all_labels = train(
                line_tensor, onehot_line_tensor, rnn, config, state, optimizer, log_outputs=log_outputs_for_train
            )
            # Store detailed outputs if they were generated
            if log_outputs_for_train:
                all_outputs = current_all_outputs
                all_labels = current_all_labels

            # --- Accumulate Stats for the PLOT (Averaging) Interval ---
            # Use instantaneous loss (sequence average loss from train)
            current_loss_plot_interval += loss

            # Calculate correctness based on the *last* character prediction
            if output is not None and line_tensor.numel() > 0:
                 output_detached = output.detach()
                 last_output = output_detached[0] # First item in batch, last time step output
                 last_target_idx = line_tensor[0, -1].item() # Target index for the last char
                 predicted_idx = torch.argmax(last_output).item()
                 is_correct = (predicted_idx == last_target_idx)
                 current_correct_plot_interval += 1 if is_correct else 0

            # ==============================================================
            # --- Frequent Detailed Console Logging Period (print_freq) ---
            # ==============================================================
            if iter % args.print_freq == 0:
                # Print iteration status and INSTANTANEOUS loss for this sequence
                print(f'{iter} {iter / args.n_iters * 100:.2f}% ({timeSince(start)}) InstLoss: {loss:.4f}')

                # Check if detailed outputs were generated and stored
                if all_outputs and all_labels:
                    # --- Calculate Step-by-Step Accuracy ---
                    num_steps = len(all_outputs)
                    num_correct_steps = 0
                    num_top2_correct_steps = 0
                    if num_steps > 0:
                        with torch.no_grad():
                             # (Calculation code for step_accuracy_top1/top2 remains the same)
                            for i in range(num_steps):
                                step_output = all_outputs[i]
                                step_label_onehot = all_labels[i]
                                target_idx = torch.argmax(step_label_onehot).item()
                                top_val, top_idx = torch.topk(step_output, 2)
                                predicted_idx_top1 = top_idx[0].item()
                                if predicted_idx_top1 == target_idx:
                                    num_correct_steps += 1
                                    num_top2_correct_steps += 1
                                else:
                                    if len(top_idx) > 1:
                                        predicted_idx_top2 = top_idx[1].item()
                                        if predicted_idx_top2 == target_idx:
                                            num_top2_correct_steps += 1

                        step_accuracy_top1 = (num_correct_steps / num_steps) if num_steps > 0 else 0.0
                        step_accuracy_top2 = (num_top2_correct_steps / num_steps) if num_steps > 0 else 0.0
                    else:
                        step_accuracy_top1 = 0.0
                        step_accuracy_top2 = 0.0

                    # --- Print Detailed Step Acc & Sequences ---
                    current_step_acc_t1_plot_interval += step_accuracy_top1
                    current_step_acc_t2_plot_interval += step_accuracy_top2
                    num_detailed_calcs_in_plot_interval += 1
                    # *** END ADDED LINES ***

                    # --- Print Detailed Step Acc & Sequences ---
                    print(f'  Detailed @ {iter}: StepAccT1: {step_accuracy_top1:.4f} StepAccT2: {step_accuracy_top2:.4f}')
                    try:
                         # (Code to get sequence strings remains the same)
                         target_sequence_str = sequence[0]
                         predicted_chars = [idx_to_char.get(torch.argmax(o).item(), '?') for o in all_outputs]
                         predicted_chars_top2 = [idx_to_char.get(torch.topk(o, 2).indices[1].item(), '?') for o in all_outputs]
                         progression_top1 = ''.join(predicted_chars)
                         progression_top2 = ''.join(predicted_chars_top2)
                         last_predicted_char = predicted_chars[-1] if predicted_chars else '?'
                         last_target_char = target_sequence_str[-1] if target_sequence_str else '?'
                         correct_marker = '✓' if last_predicted_char == last_target_char else f'✗ ({last_target_char})'

                         print(f'  Target: \033[91m{target_sequence_str}\033[0m')
                         print(f'  Pred T1: \033[93m{progression_top1}\033[0m {correct_marker}')
                         print(f'  Pred T2: \033[35m{progression_top2}\033[0m')
                    except Exception as e:
                        print(f"  Error preparing detailed print strings: {e}")

                    # Clear the detailed output lists after use
                    all_outputs.clear()
                    all_labels.clear()
                # No WandB logging in this frequent block

            # ==============================================================
            # --- Averaged Console Print & WandB Log Period (plot_freq) ---
            # ==============================================================
            # Check plot_freq > 0 to avoid division by zero
            if args.plot_freq > 0 and iter % args.plot_freq == 0:
                # Calculate averages over the plot interval
                avg_loss_plot = current_loss_plot_interval / args.plot_freq
                avg_acc_plot = current_correct_plot_interval / args.plot_freq

                # Print the calculated averages for this interval
                print(f'--- Avg Interval {iter // args.plot_freq} (ending @ {iter}) ---')
                print(f'  Avg Loss (last {args.plot_freq} iters): {avg_loss_plot:.4f}')
                print(f'  Avg Acc (last {args.plot_freq} iters): {avg_acc_plot:.4f}')
                print(f'-------------------------------------')


                # --- Log Averages to WandB ---
                if args.track:
                    # Calculate average step accuracies for the interval
                    avg_step_acc_t1_plot = (current_step_acc_t1_plot_interval / num_detailed_calcs_in_plot_interval) if num_detailed_calcs_in_plot_interval > 0 else 0.0
                    avg_step_acc_t2_plot = (current_step_acc_t2_plot_interval / num_detailed_calcs_in_plot_interval) if num_detailed_calcs_in_plot_interval > 0 else 0.0

                    wandb.log({
                        "iter": iter,
                        "avg_loss": avg_loss_plot,      # Log the averaged loss
                        "avg_accuracy": avg_acc_plot,    # Log the averaged accuracy
                        "avg_step_acc_t1": avg_step_acc_t1_plot, # Log averaged step acc T1
                        "avg_step_acc_t2": avg_step_acc_t2_plot  # Log averaged step acc T2
                    })

                # Reset ALL accumulators for the next plot interval
                current_loss_plot_interval = 0.0
                current_correct_plot_interval = 0
                current_step_acc_t1_plot_interval = 0.0
                current_step_acc_t2_plot_interval = 0.0
                num_detailed_calcs_in_plot_interval = 0


            # ==============================================================
            # --- W&B Offline Sync Trigger ---
            # ==============================================================
            # Trigger sync less often than WandB logging frequency (plot_freq)
            is_offline = os.getenv("WANDB_MODE") == "offline"
            if args.plot_freq > 0 and iter % (args.plot_freq * 10) == 0 and args.track and is_offline:
                print("Triggering W&B sync...")
                trigger_sync()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Finishing up...")
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback


    finally: # Ensure wandb finishes even on error/interrupt
        if args.track and wandb.run is not None:
            print("Finishing W&B run...")
            wandb.finish()
            print("W&B run finished.")




if __name__ == '__main__':
    main()