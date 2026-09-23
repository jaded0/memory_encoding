# train.py
import torch
from ephemeral_model import EphemeralRNN, SimpleRNN
import wandb
import matplotlib.pyplot as plt
from preprocess import load_and_preprocess_data
from reproducibility import DataStream, capture_rng_state, record_seed_in_slurm, resolve_seed, seed_everything
from metrics import IntervalMetrics, recall_chance
from utils import randomTrainingExample, timeSince, str2bool, initialize_charset, save_checkpoint, load_checkpoint, read_checkpoint
import time
import math
import argparse
import sys
import itertools
import os
import psutil
import signal
try:
    from wandb_osh.hooks import TriggerWandbSyncHook  # <-- New!
except ImportError:
    TriggerWandbSyncHook = None

class TermColors:
    GREEN = '\033[92m'
    PURPLE = '\033[95m' # Magenta, often used for Purple
    WHITE = '\033[97m'
    RESET = '\033[0m'

def plot_ascii_bar_graph(data, title, max_width=40):
    if not data:
        return

    print(f"--- {title} ---")
    
    # Check if there's anything to plot
    valid_values = [v for v in data.values() if v > 0]
    if not valid_values:
        for label, value in sorted(data.items()):
            print(f"  {label}: {value:.6f}")
        print("  (all values are zero or negative)")
        return

    max_val = max(valid_values)
    max_label_len = max(len(label) for label in data.keys())

    for label, value in sorted(data.items()):
        bar_len = int((value / max_val) * max_width) if value > 0 else 0
        bar = '█' * bar_len
        print(f"  {label.ljust(max_label_len)} | {bar} ({value:.6f})")

# from memory_profiler import profile

trigger_sync = TriggerWandbSyncHook() if TriggerWandbSyncHook else None  # <--- New!

# --- W&B end-of-run markers ---
def wb_mark_end(reason: str, tags=None, exit_code: int | None = None):
    """Record an end reason in both tags and summary. Safe if tracking is off."""
    if not (wandb.run and getattr(wandb.run, "summary", None) is not None):
        return
    # structured summary
    wandb.run.summary["end_reason"] = reason
    wandb.run.summary[f"end_is_{reason}"] = True
    if exit_code is not None:
        wandb.run.summary["end_exit_code_suggested"] = int(exit_code)  # read later at finish()

    # tags (filter-friendly)
    if tags:
        current = set(getattr(wandb.run, "tags", []))
        wandb.run.tags = list(current.union(set(tags)))

def train_batch(line_tensor, onehot_line_tensor, rnn, config, state, optimizer=None, log_outputs=False):
    """Trains on one batch of sequences with DFA, backprop or BPTT."""
    updater = config['updater']
    criterion = config['criterion']
    batch_size = onehot_line_tensor.shape[0]
    hidden = rnn.initHidden(batch_size=batch_size)

    # For EphemeralRNN, reset the ephemeral weights at the start of the sequence
    if isinstance(rnn, EphemeralRNN):
        rnn.start_sequence_wipe()

    loss_total = 0.0
    losses = []  # For DFA (per-batch losses)
    step_preds, step_losses = [], []  # [T-1] x [B], for per-interval metrics
    num_steps = 0
    all_outputs = []
    all_labels = []

    for i in range(onehot_line_tensor.size()[1] - 1):
        # For BPTT, we keep gradients flowing through time by NOT detaching hidden state
        if updater != 'bptt':
            hidden = hidden.detach()

        # Get current character's one-hot vector
        current_char_tensor = onehot_line_tensor[:, i, :]
        if updater == 'dfa':
            current_char_tensor.requires_grad = False

        # Conditional Input Construction
        if config['input_mode'] == 'last_two':
            if i == 0:
                previous_char_tensor = torch.zeros_like(current_char_tensor)
            else:
                previous_char_tensor = onehot_line_tensor[:, i-1, :]
            if updater == 'dfa':
                previous_char_tensor.requires_grad = False
            combined_char_tensor = torch.cat([current_char_tensor, previous_char_tensor], dim=1)
        elif config['input_mode'] == 'last_one':
            combined_char_tensor = current_char_tensor
        else:
            raise ValueError(f"Invalid input_mode: {config['input_mode']}")

        # Handle positional encoding
        pe_matrix = config["pe_matrix"]
        if pe_matrix is not None:
            pe_vec = pe_matrix[min(i, pe_matrix.size(0)-1)]
            pe_vec = pe_vec.unsqueeze(0).expand(batch_size, -1)
            hot_input_char_tensor = torch.cat([combined_char_tensor, pe_vec], dim=1)
        else:
            hot_input_char_tensor = combined_char_tensor

        # Forward pass
        output, hidden, self_grad = rnn(hot_input_char_tensor, hidden)
        final_char = onehot_line_tensor[:, i+1, :]
        
        # Compute loss and update weights based on updater type
        if updater == 'dfa':
            # DFA-specific processing
            output.requires_grad_(True)
            loss = criterion(output, final_char)
            losses.append(loss.detach())
            
            # Convert loss to reward signal
            global_error = torch.autograd.grad(loss, output, grad_outputs=torch.ones_like(loss), retain_graph=False)[0]
            reward_update = global_error
            rnn.zero_grad()
            
            # Add self_grad if configured
            if config.get("self_grad", 0) > 0:
                reward_update += torch.clamp(self_grad, min=-config["self_grad"], max=config["self_grad"])
            
            # Apply DFA updates
            if isinstance(rnn, EphemeralRNN):
                # Populate gradients using DFA feedback weights
                for layer in rnn.linear_layers:
                    layer.populate_dfa_gradients(reward_update)
                rnn.i2o.populate_dfa_gradients(reward_update)
                rnn.self_grad.populate_dfa_gradients(reward_update)
                
                # Apply the updates using the DFA-populated gradients
                for layer in rnn.linear_layers:
                    layer.apply_update(config["learning_rate"], config["grad_clip"], state)
                rnn.i2o.apply_update(config["learning_rate"], config["grad_clip"], state)
                rnn.self_grad.apply_update(config["learning_rate"], config["grad_clip"], state)
                
                # Forget after the whole update (incl. clamp/normalize), as in the paper
                rnn.apply_forget_step()
                
                # Clear gradients after the updates
                rnn.zero_grad()
            
            state['training_instance'] += 1
            loss_total += loss.mean().item()  # Convert to scalar for consistency
            
        elif updater == 'backprop':
            # Backprop-specific processing
            if isinstance(rnn, EphemeralRNN):
                # EphemeralRNN with TRUE backprop - compute gradients through the network
                step_loss = criterion(output, final_char)
                losses.append(step_loss.detach())
                
                # Zero gradients before backward pass
                rnn.zero_grad()
                
                # Compute gradients through the entire network (true backprop)
                total_loss = step_loss.mean() if step_loss.dim() > 0 else step_loss
                total_loss.backward(retain_graph=False)
                
                # Scale the ephemeral weights' gradients
                rnn.scale_ephemeral_grads(config["plast_clip"])
                
                # Store gradient norms for logging
                if state.get('log_norms_now', False):
                    rnn.store_all_grad_norms()
                
                # # Store projected error for bias updates (backprop case)
                # # For backprop, we can extract the bias gradient that was already computed
                # # The bias gradient is equivalent to the sum of output gradients across batch
                # if hasattr(rnn.i2o, 'bias') and rnn.i2o.bias is not None and rnn.i2o.bias.grad is not None:
                #     # Use the bias gradient as the projected error (it's already summed across batch)
                #     bias_grad = rnn.i2o.bias.grad.clone()
                #     for layer in rnn.linear_layers:
                #         layer._last_projected_error = bias_grad.unsqueeze(0)  # Add batch dim for consistency
                #     rnn.i2h._last_projected_error = bias_grad.unsqueeze(0)
                #     rnn.i2o._last_projected_error = bias_grad.unsqueeze(0)
                
                # Apply the updates using the gradients computed by backprop
                for layer in rnn.linear_layers:
                    layer.apply_update(config["learning_rate"], config["grad_clip"], state)
                rnn.i2h.apply_update(config["learning_rate"], config["grad_clip"], state)
                rnn.i2o.apply_update(config["learning_rate"], config["grad_clip"], state)
                
                # Forget after the whole update (incl. clamp/normalize), as in the paper
                rnn.apply_forget_step()
                
                # Clear gradients after the updates
                rnn.zero_grad()
                
                state['training_instance'] += 1
                loss_total += step_loss.mean().item() if step_loss.dim() > 0 else step_loss.item()
            else:
                # Standard SimpleRNN with backprop
                optimizer.zero_grad()
                step_loss = criterion(output, final_char)
                # With 'none' reduction, we get per-example losses, so take mean for backward
                total_loss = step_loss.mean() if step_loss.dim() > 0 else step_loss
                total_loss.backward()
                
                if config['grad_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(rnn.parameters(), config['grad_clip'])
                
                optimizer.step()
                loss_total += step_loss.mean().item() if step_loss.dim() > 0 else step_loss.item()
            
        elif updater == 'bptt':
            # BPTT-specific processing - accumulate loss across sequence
            step_loss = criterion(output, final_char)
            
            # For BPTT, accumulate losses across the entire sequence
            if i == 0:
                # Initialize accumulated loss on first step
                accumulated_loss = step_loss
            else:
                # Add to accumulated loss (this maintains the computation graph)
                accumulated_loss = accumulated_loss + step_loss
            
            loss_total += step_loss.mean().item() if step_loss.dim() > 0 else step_loss.item()
            
            # Only backward and update on the last step to get full sequence gradients
            if i == onehot_line_tensor.size()[1] - 2:  # Last step
                if isinstance(rnn, EphemeralRNN):
                    # EphemeralRNN with BPTT - backward through entire accumulated loss
                    total_loss = accumulated_loss.mean() if accumulated_loss.dim() > 0 else accumulated_loss
                    total_loss.backward(retain_graph=False)
                    
                    # Scale the ephemeral weights' gradients
                    rnn.scale_ephemeral_grads(config["plast_clip"])
                    
                    # Store gradient norms for logging
                    if state.get('log_norms_now', False):
                        rnn.store_all_grad_norms()
                    
                    # Manual optimizer step for EphemeralRNN
                    with torch.no_grad():
                        for param in rnn.parameters():
                            if param.grad is not None:
                                param.data -= config["learning_rate"] * param.grad
                                param.grad.zero_()

                    # Forget after the update (only once here), as in the paper
                    rnn.apply_forget_step()
                else:
                    # Standard SimpleRNN with BPTT
                    optimizer.zero_grad()
                    total_loss = accumulated_loss.mean() if accumulated_loss.dim() > 0 else accumulated_loss
                    total_loss.backward()
                    
                    if config['grad_clip'] > 0:
                        torch.nn.utils.clip_grad_norm_(rnn.parameters(), config['grad_clip'])
                    
                    optimizer.step()

        num_steps += 1
        step_preds.append(output.detach().argmax(dim=1))
        step_losses.append((loss if updater == 'dfa' else step_loss).detach())

        if log_outputs:
            if updater == 'dfa':
                all_outputs.append(output[0].detach())
                all_labels.append(final_char[0].detach())
            else:
                all_outputs.append(output[0])
                all_labels.append(final_char[0])

    # Calculate final loss
    if updater == 'dfa' and losses:
        stacked_losses = torch.stack(losses)
        loss_avg = stacked_losses.mean().item()
    else:
        loss_avg = loss_total / num_steps if num_steps > 0 else 0.0

    return output, loss_avg, torch.stack(step_preds), torch.stack(step_losses), all_outputs, all_labels


def train(line_tensor, onehot_line_tensor, rnn, config, state, optimizer=None, log_outputs=False):
    """Main training function that sets up criterion and calls train_batch."""
    # For ALL updaters, use 'none' reduction to preserve per-example gradients
    # This allows independent weight updates per sequence in the batch
    # This is critical for ephemeral weights to adapt independently per sequence
    if config['criterion'].reduction != 'none':
        print(f"Warning: Overriding criterion reduction to 'none' for {config['updater']} training.")
        config['criterion'] = type(config['criterion'])(reduction='none')
    
    return train_batch(line_tensor, onehot_line_tensor, rnn, config, state, optimizer, log_outputs)

def main():
    # Parse command-line arguments
    # Defaults match the configuration the run scripts actually use; the model hyperparameters
    # (lr, plast_clip, forget_rate, hidden_size, plast_proportion, dataset) are the bench_sweep point
    # that solves 3-char palindromes without recurrence.
    parser = argparse.ArgumentParser(description='Train a model with specified hyperparameters.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--plast_learning_rate', type=float, default=0.005, help='Learning rate for the plasticity')
    parser.add_argument('--plast_clip', type=float, default=1e5, help='Plasticity (learning-rate multiplier) of the ephemeral weights, alpha.')
    parser.add_argument('--imprint_rate', type=float, default=0.00, help='Imprint rate (unused)')
    parser.add_argument('--forget_rate', type=float, default=0.01, help='Fraction of each ephemeral weight removed per step (w <- (1 - forget_rate) w).')
    parser.add_argument('--checkpoint_save_freq', type=int, default=10000,
                        help='How often to save a checkpoint (in iterations).')
    parser.add_argument('--residual_connection', type=str2bool, nargs='?', const=True, default=False, help='whether to have a skip connection')
    parser.add_argument('--grad_clip', type=float, default=0, help='Element-wise clip on ephemeral-weight updates (0 = off).')
    parser.add_argument('--hidden_size', type=int, default=1024, help='Size of hidden layers in RNN')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers in RNN')
    parser.add_argument('--n_iters', type=int, default=10000, help='Number of training iterations')
    parser.add_argument('--print_freq', type=int, default=50, help='Frequency of printing training progress')
    parser.add_argument('--model_type', type=str, default='ephemeral', choices=['rnn', 'ephemeral'], help='Model architecture to use.')
    parser.add_argument('--updater', type=str, default='dfa', choices=['dfa', 'backprop', 'bptt'], help='Weight update algorithm to use.')
    parser.add_argument('--normalize', type=str2bool, nargs='?', const=True, default=False, help='Rescale each layer\'s candidate weights to unit norm after each update.')
    parser.add_argument('--clip_weights', type=float, default=0, help='Clamp candidate weights to [-clip_weights, clip_weights] (0 = off).')
    parser.add_argument('--track', type=str2bool, nargs='?', const=True, default=True, help='Whether to track progress online.')
    parser.add_argument('--dataset', type=str, default='3_palindrome_dataset_vary_length', help='The dataset used for training.')
    parser.add_argument('--notes', type=str, default='nothing to say', help='talk about this run')
    parser.add_argument('--wandb_project', type=str, default='ephemeral-weights', help='W&B project to log to.')
    parser.add_argument('--group', type=str, default="nothing_in_particular", help='Description of what sort of experiment is being run, here.')
    parser.add_argument('--tags', nargs='*', default=[], help="List of tags for WandB")
    parser.add_argument('--batch_size', type=int, default=16, help='how much to stuff in at once')
    parser.add_argument('--positional_encoding_dim', type=int, default=0,
                        help='Dimension for optional positional encoding (0 means off).')
    parser.add_argument('--self_grad', type=float, default=0.0, help='Scale of self_grad. grad based replacement for recurrence.')
    parser.add_argument('--input_mode', type=str, default='last_one', choices=['last_one', 'last_two'],
                        help='Input mode: use last one or last two characters.')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints.')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Resume from this checkpoint (always resumes; errors if missing).')
    parser.add_argument('--plast_proportion', type=float, default=0.1, help='Proportion of weights that are ephemeral in each layer.')
    parser.add_argument('--enable_recurrence', type=str2bool, nargs='?', const=True, default=False, help='Whether to enable recurrent hidden state connections')
    parser.add_argument('--log_freq', type=int, default=None, help='Frequency for W&B sync triggers (overrides LOG_FREQ environment variable)')
    parser.add_argument('--resume', type=str2bool, nargs='?', const=True, default=False, help='Resume from <checkpoint_dir>/latest_checkpoint.pth if it exists.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed Python, NumPy, Torch, and data loading (unset = drawn from the OS on a fresh run, '
                             'read from the checkpoint on resume).')
    parser.add_argument('--deterministic', type=str2bool, nargs='?', const=True, default=None,
                        help='Require deterministic Torch operations (unset = off on a fresh run, '
                             'the checkpoint\'s value on resume).')

    # grab slurm jobid if it exists.
    job_id = os.environ.get("SLURM_JOB_ID") if os.environ.get("SLURM_JOB_ID") else "no_SLURM"
    print("SLURM Job ID:", job_id)
    
    args = parser.parse_args()

    # Define the path to the latest checkpoint
    latest_checkpoint_path = os.path.join(args.checkpoint_dir, "latest_checkpoint.pth")

    # An explicit --resume_checkpoint always resumes; --resume picks up latest_checkpoint.pth if present.
    checkpoint_to_load = None
    if args.resume_checkpoint:
        if not os.path.isfile(args.resume_checkpoint):
            raise FileNotFoundError(f"Explicit resume checkpoint not found: {args.resume_checkpoint}")
        checkpoint_to_load = args.resume_checkpoint
        print(f"Attempting to resume from explicit checkpoint: {checkpoint_to_load}")
    elif args.resume and os.path.isfile(latest_checkpoint_path):
        checkpoint_to_load = latest_checkpoint_path
        print(f"Found latest checkpoint. Attempting to resume from: {checkpoint_to_load}")
    elif args.resume:
        print(f"--resume given but no checkpoint at {latest_checkpoint_path}. Starting from scratch.")
    else:
        print("Starting from scratch (pass --resume or --resume_checkpoint to resume).")
    # Read the checkpoint before anything consumes randomness: it decides the seed.
    checkpoint = read_checkpoint(checkpoint_to_load) if checkpoint_to_load else None

    try:
        seed, deterministic, seed_source = resolve_seed(args.seed, args.deterministic, checkpoint)
        seed_everything(seed, deterministic=deterministic)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Seed: {seed} ({seed_source}), deterministic: {deterministic}")
    if seed is None:
        print("WARNING: resumed legacy unseeded run: no seed is set; the checkpoint's RNG states are "
              "restored, but data order does not continue exactly.")
    record_seed_in_slurm(seed)
    
    # Set log_freq: command line arg takes precedence over environment variable
    if args.log_freq is not None:
        log_freq = args.log_freq
        print(f"Log frequency set from command line: {log_freq}")
    else:
        log_freq = int(os.getenv("LOG_FREQ", "5000"))
        print(f"Log frequency set from environment (LOG_FREQ): {log_freq}")

    config = {
        "learning_rate": args.learning_rate,
        "plast_learning_rate": args.plast_learning_rate,
        "plast_clip": args.plast_clip,
        "imprint_rate": args.imprint_rate,
        "forget_rate": args.forget_rate,
        "checkpoint_save_freq": args.checkpoint_save_freq,
        # Use 'mean' for backprop, will be overridden to 'none' in train() for the ephemeral model
        "criterion": torch.nn.CrossEntropyLoss(reduction='mean'),
        "residual_connection": args.residual_connection,
        "grad_clip": args.grad_clip,
        "n_hidden": args.hidden_size,
        "n_layers": args.num_layers,
        "track": args.track,
        "dataset": args.dataset,
        "model_type": args.model_type,
        "updater": args.updater,
        "batch_size": args.batch_size,
        "self_grad": args.self_grad,
        "input_mode": args.input_mode,
        "plast_proportion": args.plast_proportion,
        "enable_recurrence": args.enable_recurrence,
        "seed": seed,
        "deterministic": deterministic,
    }
    # Record every other CLI argument too, so checkpoints carry it and a resume can diff it.
    config.update({key: value for key, value in vars(args).items() if key not in config})
    print(f"Input mode selected: {args.input_mode}") # Inform user

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir, exist_ok=True) # exist_ok=True for robustness

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    charset, char_to_idx, idx_to_char, n_characters = initialize_charset(args.dataset)
    config["charset_size"] = n_characters         # ← add this line
    print(f"Character set size: {n_characters}")

    # Use drop_last=True if batch size doesn't divide dataset size evenly
    dataloader = load_and_preprocess_data(
        args.dataset, args.batch_size, drop_last=True, seed=seed
    )

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
    if args.input_mode == 'last_two':
        char_input_dim = n_characters * 2
    elif args.input_mode == 'last_one':
        char_input_dim = n_characters
    else:
        # This case should be prevented by argparse choices, but adding for safety
        raise ValueError(f"Invalid input_mode: {args.input_mode}")

    input_size = char_input_dim + pos_dim # Base input size from characters + positional encoding

    output_size = n_characters
    print(f"Model Input Size: {input_size}, Hidden Size: {config['n_hidden']}, Output Size: {output_size}") # Log calculated size

    optimizer = None
    start_iter = 1
    base_input_size = input_size # The size calculated above (chars + PE)

    if args.model_type == 'rnn':
        print(f"Initializing SimpleRNN model with '{args.updater}' updater.")
        rnn = SimpleRNN(base_input_size, config["n_hidden"], output_size, config["n_layers"], 
                       dropout_rate=0, enable_recurrence=args.enable_recurrence)
    elif args.model_type == 'ephemeral':
        print(f"Initializing EphemeralRNN model with '{args.updater}' updater.")
        rnn = EphemeralRNN(
            base_input_size, config["n_hidden"], output_size, config["n_layers"], charset,
            normalize=args.normalize, residual_connection=args.residual_connection,
            clip_weights=args.clip_weights, updater=args.updater,
            plast_clip=config["plast_clip"], batch_size=config["batch_size"],
            forget_rate=config["forget_rate"], plast_proportion=config["plast_proportion"],
            enable_recurrence=args.enable_recurrence
        )
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    # Optimizer is only needed for backprop and bptt (regardless of model type)
    if args.updater in ['backprop', 'bptt']:
        optimizer = torch.optim.SGD(rnn.parameters(), lr=config['learning_rate'])

    state = {
        "training_instance": 0,
        "last_n_rewards": [0],
        "last_n_reward_avg": 0,
        "wandb_step": 0,  # Initialize wandb_step
        "log_norms_now": False,
    }

    # --- Resume from Checkpoint ---
    # The model and data stream are built (consuming seeded randomness exactly as the original
    # run did) before load_checkpoint overwrites the weights and restores the RNG states.
    data_stream = DataStream(dataloader)
    if checkpoint_to_load:
        # Any load failure aborts the run: silently restarting from scratch hides
        # the failure and mixes fresh weights into a "resumed" experiment.
        rnn, optimizer, start_iter, loaded_main_state, loaded_config = load_checkpoint(
            checkpoint_to_load, rnn, config, optimizer=optimizer, device=device, checkpoint=checkpoint
        )
        if not data_stream.load_state_dict(checkpoint.get("data_stream_state")) and seed is not None:
            print("WARNING: checkpoint has no data-stream position; data order restarts from the seed's first epoch.")
        checkpoint = None  # everything needed has been copied out; free the CPU copy
        state.update(loaded_main_state) # Update your main program state
        print(f"resumed, starting from iter: {start_iter}")

        # Check if plast_clip has changed and update plasticity parameters if needed
        if isinstance(rnn, EphemeralRNN):
            # The mask comes from the state dict. Checkpoints saved before last layers lost
            # their ephemeral entries keep them; left as saved so the run continues unchanged.
            stale = [name for name in ('i2o', 'self_grad') if getattr(rnn, name).ephemeral_mask.any()]
            if stale:
                print(f"WARNING: checkpoint predates empty last-layer masks: {', '.join(stale)} keep "
                      "their saved ephemeral entries (decayed and wiped). Start fresh for the current behaviour.")
            loaded_plast_clip = loaded_config.get('plast_clip', 1.0)
            current_plast_clip = config.get('plast_clip', 1.0)

            if loaded_plast_clip != current_plast_clip:
                print(f"Plasticity changed from {loaded_plast_clip} to {current_plast_clip}")
                print("Updating plasticity parameters in all layers...")
                rnn.set_plasticity(current_plast_clip)
                print("Plasticity parameters updated successfully!")
            else:
                print(f"Plasticity unchanged: {current_plast_clip}")

    elif torch.cuda.is_available(): # No checkpoint_to_load specified AT ALL, and cuda is available
        print("No checkpoint specified for loading. Moving model to GPU.")
        rnn = rnn.to(device)
    # else: model remains on CPU if no checkpoint and no CUDA

    if args.track:
        # wandb initialization
        wandb_config = {
            "learning_rate": args.learning_rate,
            "plast_learning_rate": args.plast_learning_rate,
            "plast_clip": args.plast_clip,
            "nominal_mean_lr": args.learning_rate * (1-args.plast_proportion + args.plast_proportion * args.plast_clip),
            "nominal_ephemeral_lr": args.learning_rate * args.plast_clip,
            "architecture": args.model_type,
            "updater": args.updater,
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
            "log_freq": log_freq,
            "batch_size": args.batch_size,
            "slurm_id": job_id,
            "positional_encoding_dim": args.positional_encoding_dim,
            "checkpoint_save_freq": args.checkpoint_save_freq,
            "self_grad": args.self_grad,
            "input_mode": args.input_mode,
            "plast_proportion": args.plast_proportion,
            "enable_recurrence": args.enable_recurrence,
            "seed": seed,
            "seed_source": seed_source,
            "deterministic": deterministic,
            "recall_chance": recall_chance(args.dataset),
        }
        # A resumed checkpoint always starts a new W&B run; record where it came from.
        if checkpoint_to_load:
            wandb_config["resumed_from_checkpoint"] = checkpoint_to_load
            wandb_config["resumed_at_iter"] = start_iter
            print("Resuming a checkpoint: starting a new W&B run (W&B run resumption is not supported).")
        print(f"tags given to wandb: {args.tags}")
        wandb.init(project=args.wandb_project,
                group=args.group,
                notes=args.notes,
                tags=args.tags,
                config=wandb_config,
                )
        print(f"Initialized WandB with Run ID: {wandb.run.id}")


    # Training Loop
    start = time.time()
    
    # Flag to track if NaN has been detected
    nan_detected = False
    
    # Early stopping for high loss values - sliding window tracking
    EARLY_STOP_LOSS_THRESHOLD = 5.0
    EARLY_STOP_WINDOW_SIZE = 10
    loss_window = []  # Sliding window of average losses from print_freq intervals
    high_loss_count = 0  # Count of consecutive intervals with loss > threshold
    early_stopped = False

    # SLURM sends SIGUSR1 shortly before the wall-time limit (#SBATCH --signal=B:USR1@600, forwarded by
    # forward_signals.sh) and SIGTERM at the limit. Stop cleanly at the next iteration boundary.
    stop_signals = []
    def request_stop(signum, frame):
        stop_signals.append(signum)
    previous_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGUSR1, signal.SIGTERM)}
    stopped_by_signal = None

    def checkpoint_state(next_iter):
        return {
            'iter': next_iter,
            'model_state_dict': rnn.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'main_program_state': state,
            'config': config,
            'data_stream_state': data_stream.state_dict(),
            **capture_rng_state(),
        }

    try:
        # Metrics accumulated over each print_freq interval (whole batch, every step)
        interval = IntervalMetrics(args.dataset)
        interval_start = time.time()
        
        # These will be populated per iteration if log_outputs_for_train is true
        # and then cleared after print_freq, as per your original logic.
        all_outputs_for_print_freq = [] 
        all_labels_for_print_freq = []

        for iter in range(start_iter, args.n_iters + 1):
            if stop_signals:
                stopped_by_signal = signal.Signals(stop_signals[0])
                print(f"Received {stopped_by_signal.name}: stopping before iteration {iter}.")
                if args.checkpoint_save_freq > 0:
                    save_checkpoint(checkpoint_state(iter), args.checkpoint_dir, "latest_checkpoint.pth")
                else:
                    print("Checkpointing disabled (checkpoint_save_freq=0). No checkpoint saved.")
                if stopped_by_signal == signal.SIGUSR1:
                    wb_mark_end("time_limit", tags=["end:time_limit"], exit_code=124)
                else:
                    wb_mark_end("terminated", tags=["end:terminated"], exit_code=143)
                break

            # Fetch next batch (the stream is endless and tracks its own position)
            sequence, line_tensor, onehot_line_tensor = next(data_stream)

            line_tensor = line_tensor.to(device)
            onehot_line_tensor = onehot_line_tensor.to(device)

            # Ensure batch size matches model expectation if using EphemeralRNN with fixed batch size param
            if args.updater != "backprop" and hasattr(rnn, 'batch_size') and onehot_line_tensor.shape[0] != rnn.batch_size:
                 print(f"Warning: Batch size mismatch ({onehot_line_tensor.shape[0]} vs {rnn.batch_size}). Skipping batch.")
                 continue # Skip this batch

            # Determine if detailed outputs are needed (for the frequent PRINT interval)
            log_outputs_for_train = (iter % args.print_freq == 0)

            # --- Train Step ---
            state["log_norms_now"] = (iter % args.print_freq == 0)
            # The train function returns step-by-step outputs for the first batch item if log_outputs=True
            output, loss, step_preds, step_losses, current_iter_all_outputs, current_iter_all_labels = train(
                line_tensor, onehot_line_tensor, rnn, config, state, optimizer, log_outputs=log_outputs_for_train
            )
            
            # Terminate on a non-finite loss (NaN or inf; isnan alone misses inf)
            if not math.isfinite(loss):
                print(f"Non-finite loss ({loss}) detected. Terminating training.")
                nan_detected = True
                wb_mark_end("nan_detected", tags=["end:nan", "NaN"], exit_code=1)
                break  # Exit the training loop
            
            # Store detailed outputs if they were generated FOR THIS ITERATION for print_freq
            if log_outputs_for_train:
                all_outputs_for_print_freq = current_iter_all_outputs
                all_labels_for_print_freq = current_iter_all_labels

            interval.update(sequence, onehot_line_tensor, step_preds, step_losses)

            # ==============================================================
            # --- Frequent Detailed Console Logging Period (print_freq) ---
            # ==============================================================
            if iter % args.print_freq == 0:
                print(f'{iter} {iter / args.n_iters * 100:.2f}% ({timeSince(start)}) InstLoss: {loss:.4f}')

                # Check if detailed outputs were generated for this iteration
                if all_outputs_for_print_freq and all_labels_for_print_freq:
                    source_char_display_list = []
                    target_display_list = []
                    pred_t1_display_list = []
                    pred_t2_display_list = []

                    num_prediction_steps = len(all_outputs_for_print_freq)
                    num_correct_t1_for_seq = 0
                    num_correct_top2_for_seq = 0 # Top-2 inclusive (T1 or T2 correct)

                    for i in range(num_prediction_steps):
                        # Input character that led to this prediction (from the first item in batch)
                        source_char_idx = line_tensor[0, i].item()
                        source_char = idx_to_char.get(source_char_idx, '?') # Ensure idx_to_char is in scope
                        source_char_display_list.append(source_char)

                        # Target character for this step
                        step_target_onehot = all_labels_for_print_freq[i]
                        actual_target_idx = torch.argmax(step_target_onehot).item()
                        actual_target_char = idx_to_char.get(actual_target_idx, '?')

                        # Predictions for this step
                        step_output_logits = all_outputs_for_print_freq[i]
                        top_val, top_idx = torch.topk(step_output_logits, 2)
                        
                        predicted_idx_t1 = top_idx[0].item()
                        predicted_char_t1 = idx_to_char.get(predicted_idx_t1, '?')

                        predicted_idx_t2 = -1
                        predicted_char_t2 = " " 
                        if len(top_idx) > 1:
                            predicted_idx_t2 = top_idx[1].item()
                            predicted_char_t2 = idx_to_char.get(predicted_idx_t2, '?')
                        
                        # 1. Target Character ("Original" in your request)
                        if actual_target_idx == predicted_idx_t1:
                            # If T1 prediction matches the actual target
                            target_display_list.append(f"{TermColors.GREEN}{actual_target_char}{TermColors.RESET}")
                        elif len(top_idx) > 1 and predicted_idx_t2 == actual_target_idx:
                            # Else, if T1 did NOT match, but T2 exists and T2 matches the actual target
                            target_display_list.append(f"{TermColors.PURPLE}{actual_target_char}{TermColors.RESET}")
                        else:
                            # Else (T1 didn't match, AND (T2 didn't exist OR T2 also didn't match))
                            target_display_list.append(f"{TermColors.WHITE}{actual_target_char}{TermColors.RESET}")
                        

                        # 2. Pred T1 Character
                        if predicted_idx_t1 == actual_target_idx:
                            pred_t1_display_list.append(f"{TermColors.GREEN}{predicted_char_t1}{TermColors.RESET}")
                            num_correct_t1_for_seq += 1
                            num_correct_top2_for_seq += 1 # If T1 is correct, Top2 is correct
                        else:
                            pred_t1_display_list.append(f"{TermColors.RESET}{predicted_char_t1}{TermColors.RESET}")
                            # Check if T2 was correct for Top2 accuracy
                            if len(top_idx) > 1 and predicted_idx_t2 == actual_target_idx:
                                num_correct_top2_for_seq += 1
                        
                        # 3. Pred T2 Character
                        if len(top_idx) > 1: # If a T2 prediction exists
                            if predicted_idx_t2 == actual_target_idx:
                                pred_t2_display_list.append(f"{TermColors.GREEN}{predicted_char_t2}{TermColors.RESET}")
                            else: # T2 exists and is incorrect
                                pred_t2_display_list.append(f"{TermColors.WHITE}{predicted_char_t2}{TermColors.RESET}")
                        else: # No distinct T2 prediction
                            pred_t2_display_list.append(f"{TermColors.PURPLE} {TermColors.RESET}") # Purple space

                    # Print the assembled strings for the first sequence in the batch
                    # print(f"  Src : {''.join(source_char_display_list)}")
                    print(f"  Trg :  {''.join(target_display_list)}")
                    # print(f"  PrT1:  {''.join(pred_t1_display_list)}")
                    # print(f"  PrT2:  {''.join(pred_t2_display_list)}")
                    
                    # Calculate and print step-wise accuracy for THIS specific displayed sequence
                    seq_step_accuracy_t1 = (num_correct_t1_for_seq / num_prediction_steps) if num_prediction_steps > 0 else 0.0
                    seq_step_accuracy_top2 = (num_correct_top2_for_seq / num_prediction_steps) if num_prediction_steps > 0 else 0.0
                    print(f'  Seq Acc: T1 {seq_step_accuracy_t1:.4f}, Top2 {seq_step_accuracy_top2:.4f}')
                    print("-" * 40) # Separator

                    # Clear the detailed output lists after use for this print_freq iteration
                    all_outputs_for_print_freq.clear()
                    all_labels_for_print_freq.clear()
                # No WandB logging directly in this very frequent print_freq block

            # ==============================================================
            # --- Averaged Console Print & WandB Log Period (print_freq) ---
            # ==============================================================
            if args.print_freq > 0 and iter % args.print_freq == 0:
                metrics = interval.summary()
                metrics["iters_per_sec"] = interval.iterations / (time.time() - interval_start)
                avg_loss_plot = metrics.get("loss", float("nan"))

                print(f'--- Interval metrics (ending @ iter {iter}, whole batch) ---')
                for key, value in metrics.items():
                    print(f'  {key}: {value:.4f}')
                print(f'-------------------------------------------')

                if args.track:
                    # Calculate and gather norms
                    model_norms = rnn.get_all_norms()

                    # --- ASCII Bar Graph for Gradient/Update Norms ---
                    ephemeral_update_norms = {
                        k.replace('_ephemeral_update_norm', ''): v
                        for k, v in model_norms.items()
                        if 'ephemeral_update_norm' in k
                    }
                    slow_update_norms = {
                        k.replace('_slow_update_norm', ''): v
                        for k, v in model_norms.items()
                        if 'slow_update_norm' in k
                    }
                    plot_ascii_bar_graph(ephemeral_update_norms, "Ephemeral Update/Grad Norms")
                    plot_ascii_bar_graph(slow_update_norms, "Slow Update/Grad Norms")
                    
                    # Calculate averages for each type
                    ephemeral_weights = [v for k, v in model_norms.items() if 'ephemeral_weight_norm' in k]
                    slow_weights = [v for k, v in model_norms.items() if 'slow_weight_norm' in k]
                    ephemeral_updates = [v for k, v in model_norms.items() if 'ephemeral_update_norm' in k]
                    slow_updates = [v for k, v in model_norms.items() if 'slow_update_norm' in k]
                    # Also keep track of backprop norms if needed (check if 'grad_norm' exists)
                    grad_norms = [v for k, v in model_norms.items() if 'grad_norm' in k]
                    all_weights = [v for k, v in model_norms.items() if 'weight_norm' in k] # For backprop or combined ephemeral

                    avg_ephemeral_w_norm = sum(ephemeral_weights) / len(ephemeral_weights) if ephemeral_weights else 0.0
                    avg_slow_w_norm = sum(slow_weights) / len(slow_weights) if slow_weights else 0.0
                    avg_ephemeral_u_norm = sum(ephemeral_updates) / len(ephemeral_updates) if ephemeral_updates else 0.0
                    avg_slow_u_norm = sum(slow_updates) / len(slow_updates) if slow_updates else 0.0
                    avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0 # For backprop
                    avg_weight_norm = sum(all_weights) / len(all_weights) if all_weights else 0.0 # For backprop/combined

                    log_data = {
                        "iter": iter,
                        **metrics,
                        "avg_weight_norm": avg_weight_norm, # Combined / Backprop
                        "avg_grad_update_norm": avg_grad_norm or (avg_ephemeral_u_norm + avg_slow_u_norm), # Backprop, or ephemeral + slow
                        "avg_ephemeral_weight_norm": avg_ephemeral_w_norm,
                        "avg_slow_weight_norm": avg_slow_w_norm,
                        "avg_ephemeral_update_norm": avg_ephemeral_u_norm,
                        "avg_slow_update_norm": avg_slow_u_norm,
                        # **model_norms # Log all individual norms too
                    }
                    wandb.log(log_data, step=state["wandb_step"], commit=True)
                    # print some norm data too
                    print(f'  Avg Weight Norm: {avg_weight_norm:.4f}')
                    print(f'  Avg Grad/Update Norm: {avg_grad_norm:.4f}')
                
                # Early stopping logic - check loss after logging to WandB
                loss_window.append(avg_loss_plot)
                if len(loss_window) > EARLY_STOP_WINDOW_SIZE:
                    loss_window.pop(0)  # Keep only the last EARLY_STOP_WINDOW_SIZE values
                
                # Check if current loss exceeds threshold
                if avg_loss_plot > EARLY_STOP_LOSS_THRESHOLD:
                    high_loss_count += 1
                    print(f"  High loss detected ({avg_loss_plot:.4f} > {EARLY_STOP_LOSS_THRESHOLD}). Count: {high_loss_count}/{EARLY_STOP_WINDOW_SIZE}")
                else:
                    high_loss_count = 0  # Reset counter if loss drops below threshold
                
                # Early stop if we've had high loss for the required number of intervals
                if high_loss_count >= EARLY_STOP_WINDOW_SIZE:
                    print(f"Early stopping: Loss has been > {EARLY_STOP_LOSS_THRESHOLD} for {EARLY_STOP_WINDOW_SIZE} consecutive intervals.")
                    early_stopped = True
                    wb_mark_end("high_loss_early_stop", tags=["end:high_loss", "early_stop"], exit_code=1)
                    break  # Exit the training loop
                
                state["wandb_step"] += 1
                interval.reset()
                interval_start = time.time()

            # ==============================================================
            # --- W&B Offline Sync Trigger ---
            # ==============================================================
            is_offline = os.getenv("WANDB_MODE") == "offline"
            if args.print_freq > 0 and iter % (log_freq) == 0 and args.track and is_offline: # Trigger less often
                print("Triggering W&B sync...")
                # the following code is pointless because environment variables don't change for a running process. I'll want to do it with signal handler or file based checks. 
                # whatever I do can't slow anything down. Logging needs to be hyper lightweight.
                # if int(os.getenv("LOG_FREQ", "5000")) != log_freq:
                #     print(f"Log frequency has been updated to {log_freq} in the environment variable.")
                #     log_freq = int(os.getenv("LOG_FREQ", "5000")) # Update log_freq if changed in env
                try:
                    if trigger_sync:
                        trigger_sync()
                except Exception as e:
                    print(f"Error during W&B sync: {e}")

            # ==============================================================
            # --- Checkpointing ---
            # ==============================================================
            if args.checkpoint_save_freq > 0 and iter % args.checkpoint_save_freq == 0:
                save_checkpoint(checkpoint_state(iter + 1), args.checkpoint_dir, "latest_checkpoint.pth") # Overwrites latest

        # End of training loop - mark normal completion if no early stopping occurred
        if args.track and wandb.run and not nan_detected and not early_stopped and not stopped_by_signal:
            wb_mark_end("completed", tags=["end:completed"], exit_code=0)

        # If NaN was detected, we should not log the normal completion
        if nan_detected:
            print("Training terminated due to NaN loss.")
            sys.exit(1)
        elif early_stopped:
            print("Training terminated due to high loss early stopping.")
            sys.exit(1)
        elif stopped_by_signal:
            # 124 = timeout convention (SLURM's pre-limit warning); 143 = 128 + SIGTERM
            sys.exit(124 if stopped_by_signal == signal.SIGUSR1 else 143)


    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Attempting to save final checkpoint...")
        wb_mark_end("user_interrupt", tags=["end:user_interrupt"], exit_code=130)
        # Optionally save a final checkpoint on interrupt
        if args.checkpoint_dir and args.checkpoint_save_freq > 0: # Ensure dir is specified and checkpointing is enabled
            final_checkpoint_state = checkpoint_state(iter + 1 if 'iter' in locals() else start_iter)
            save_checkpoint(final_checkpoint_state, args.checkpoint_dir, "interrupt_checkpoint.pth")
            save_checkpoint(final_checkpoint_state, args.checkpoint_dir, "latest_checkpoint.pth") # also update latest
        elif args.checkpoint_save_freq == 0:
            print("Checkpointing disabled (checkpoint_save_freq=0). No checkpoint saved on interrupt.")
        print("Finishing up...")
        sys.exit(130)  # conventional exit code for SIGINT, so wrappers don't count this as success
    except Exception:
        # Re-raise so the process exits non-zero and wrapper scripts see the failure.
        wb_mark_end("crashed", tags=["end:crashed"], exit_code=1)
        raise


    finally: # Ensure wandb finishes even on error/interrupt
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        if args.track and wandb.run is not None:
            print("Finishing W&B run...")
            wandb.finish()
            print("W&B run finished.")




if __name__ == '__main__':
    main()
