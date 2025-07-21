# hebby.py
import torch
from hebbian_model import HebbianLinear, HebbyRNN, SimpleRNN
import wandb
import matplotlib.pyplot as plt
from preprocess import load_and_preprocess_data
from utils import randomTrainingExample, timeSince, str2bool, initialize_charset, save_checkpoint, load_checkpoint
import time
import math
import argparse
import sys
import numpy as np
import itertools
import os
import psutil
from wandb_osh.hooks import TriggerWandbSyncHook  # <-- New!

class TermColors:
    GREEN = '\033[92m'
    PURPLE = '\033[95m' # Magenta, often used for Purple
    WHITE = '\033[97m'
    RESET = '\033[0m'

# from memory_profiler import profile

trigger_sync = TriggerWandbSyncHook()  # <--- New!

def train_backprop(line_tensor, onehot_line_tensor, rnn, config, optimizer, log_outputs=False):
    criterion = config['criterion']

    batch_size = onehot_line_tensor.shape[0]
    hidden = rnn.initHidden(batch_size=batch_size)
    optimizer.zero_grad()

    loss_total = 0
    num_steps = 0 # Keep track of the number of steps for averaging

    all_outputs = []
    all_labels = []
    n_characters = onehot_line_tensor.shape[2] # Get n_characters from the tensor shape

    for i in range(onehot_line_tensor.size()[1] - 1):
        # For HebbyRNN with backprop, apply per-step weight decay (forgetting)
        if isinstance(rnn, HebbyRNN):
            rnn.apply_forget_step()

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
        # criterion with reduction='mean' gives the average loss for this step's batch
        step_loss = criterion(output, final_char)
        loss_total += step_loss
        num_steps += 1

        if log_outputs:
            all_outputs.append(output[0]) # Note: Only logs the first item in the batch
            all_labels.append(final_char[0]) # Note: Only logs the first item in the batch

    if num_steps > 0:
        loss_avg_per_step = loss_total / num_steps # Calculate the average loss over all steps
        loss_avg_per_step.backward() # Compute gradients based on the average loss

        # For HebbyRNN, scale gradients for differential plasticity before clipping/stepping
        if isinstance(rnn, HebbyRNN):
            rnn.scale_gradients(config['plast_learning_rate'], config['learning_rate'])

        # Apply gradient clipping only if grad_clip is positive
        if config['grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(rnn.parameters(), config['grad_clip'])

        optimizer.step() # Update parameters using the optimizer
        loss_avg_item = loss_avg_per_step.item()
    else:
        loss_avg_item = 0.0 # Handle case with no steps

    return output, loss_avg_item, 0, 0, all_outputs, all_labels # Return averaged loss item

def train_dfa(line_tensor, onehot_line_tensor, rnn, config, state, log_outputs=False):
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
        rnn.apply_imprints(reward_update, lr, config["plast_learning_rate"], plast_clip, config["imprint_rate"], config["grad_clip"], state)

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
    if config['updater'] == "backprop":
        # Ensure criterion reduction is 'mean' for backprop
        if config['criterion'].reduction != 'mean':
             print("Warning: Overriding criterion reduction to 'mean' for backprop training.")
             config['criterion'] = type(config['criterion'])(reduction='mean')
        return train_backprop(line_tensor, onehot_line_tensor, rnn, config, optimizer, log_outputs)
    elif config['updater'] == "dfa":
        # Ensure criterion reduction is 'none' for DFA calculation of per-example gradient
        if config['criterion'].reduction != 'none':
             print("Warning: Overriding criterion reduction to 'none' for DFA training.")
             config['criterion'] = type(config['criterion'])(reduction='none') # Recreate criterion with 'none'
        return train_dfa(line_tensor, onehot_line_tensor, rnn, config, state, log_outputs)
    else:
        raise ValueError(f"Unknown updater: {config['updater']}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a model with specified hyperparameters.')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate for the optimizer')
    parser.add_argument('--plast_learning_rate', type=float, default=0.005, help='Learning rate for the plasticity')
    parser.add_argument('--plast_clip', type=float, default=0.005, help='How high the plasticity can go.')
    parser.add_argument('--imprint_rate', type=float, default=0.00, help='Imprint rate for Hebbian updates')
    parser.add_argument('--forget_rate', type=float, default=0.00, help='Forget rate, forgetting factor, prevents explosion.')
    parser.add_argument('--checkpoint_save_freq', type=int, default=10000,
                        help='How often to save a checkpoint (in iterations).')
    parser.add_argument('--residual_connection', type=str2bool, nargs='?', const=True, default=True, help='whether to have a skip connection')
    parser.add_argument('--grad_clip', type=float, default=1e-1, help='Clip gradients to this value.')
    parser.add_argument('--hidden_size', type=int, default=128, help='Size of hidden layers in RNN')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers in RNN')
    parser.add_argument('--n_iters', type=int, default=10000, help='Number of training iterations')
    parser.add_argument('--print_freq', type=int, default=50, help='Frequency of printing training progress')
    parser.add_argument('--model_type', type=str, default='hebby', choices=['rnn', 'hebby'], help='Model architecture to use.')
    parser.add_argument('--updater', type=str, default='dfa', choices=['dfa', 'backprop'], help='Weight update algorithm to use.')
    parser.add_argument('--normalize', type=str2bool, nargs='?', const=True, default=True, help='Whether to normalize the weights.')
    parser.add_argument('--clip_weights', type=float, default=1, help='Whether to clip the weights.')
    parser.add_argument('--track', type=str2bool, nargs='?', const=True, default=True, help='Whether to track progress online.')
    parser.add_argument('--dataset', type=str, default='roneneldan/tinystories', help='The dataset used for training.')
    parser.add_argument('--notes', type=str, default='nothing to say', help='talk about this run')
    parser.add_argument('--group', type=str, default="nothing_in_particular", help='Description of what sort of experiment is being run, here.')
    parser.add_argument('--tags', nargs='*', default=[], help="List of tags for WandB")
    parser.add_argument('--batch_size', type=int, default=4, help='how much to stuff in at once')
    parser.add_argument('--positional_encoding_dim', type=int, default=0,
                        help='Dimension for optional positional encoding (0 means off).')
    parser.add_argument('--self_grad', type=float, default=0.0, help='Scale of self_grad. grad based replacement for recurrence.')
    parser.add_argument('--input_mode', type=str, default='last_two', choices=['last_one', 'last_two'],
                        help='Input mode: use last one or last two characters.')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints.')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume training from (e.g., checkpoints/latest_checkpoint.pth).')
    parser.add_argument('--plast_proportion', type=float, default=0.2, help='Proportion of weights that are plastic in Hebbian layers.')  # <-- Add this line

    # grab slurm jobid if it exists.
    job_id = os.environ.get("SLURM_JOB_ID") if os.environ.get("SLURM_JOB_ID") else "no_SLURM"
    print("SLURM Job ID:", job_id)
    log_freq = int(os.getenv("LOG_FREQ", "5000"))
    print("Log frequency set to:", log_freq)

    args = parser.parse_args()
    config = {
        "learning_rate": args.learning_rate,
        "plast_learning_rate": args.plast_learning_rate,
        "plast_clip": args.plast_clip,
        "imprint_rate": args.imprint_rate,
        "forget_rate": args.forget_rate,
        "checkpoint_save_freq": args.checkpoint_save_freq,
        # Use 'mean' for backprop, will be overridden to 'none' in train() for Hebby
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
    }
    print(f"Input mode selected: {args.input_mode}") # Inform user

    # Define the path to the latest checkpoint
    latest_checkpoint_path = os.path.join(args.checkpoint_dir, "latest_checkpoint.pth")

# --- WandB Run ID Management for Resumption ---
    wandb_run_id = None
    is_new_id = False # Flag to track if a new ID is generated
    if args.track: # Only manage run ID if tracking is enabled
        wandb_run_id_file_path = os.path.join(args.checkpoint_dir, "wandb_run_id.txt")

        if args.resume_checkpoint or os.path.isfile(latest_checkpoint_path): # If we are potentially resuming
            if os.path.exists(wandb_run_id_file_path):
                # with open(wandb_run_id_file_path, "r") as f:
                #     wandb_run_id = f.read().strip()
                print(f"Found existing WandB run ID: {wandb_run_id}")
            else:
                # This case is tricky: we are resuming a model checkpoint,
                # but no WandB ID was saved. This could happen if:
                # 1. Tracking was off during the run that created the checkpoint.
                # 2. The wandb_run_id.txt file was accidentally deleted.
                # We'll generate a new ID and save it, effectively starting a "new" WandB run
                # that continues the model's progress.
                print(f"Warning: Resuming checkpoint but no WandB run ID file found in {args.checkpoint_dir}.")
                print("A new WandB run will be started for this resumed session.")
                is_new_id = True
                # Fall through to generate new ID if wandb_run_id is still None
        
        if not wandb_run_id: # If it's the first run or ID was not found for a resume scenario
            wandb_run_id = wandb.util.generate_id()
            is_new_id = True # Set flag to indicate a new ID was generated
            try:
                with open(wandb_run_id_file_path, "w") as f:
                    f.write(wandb_run_id)
                print(f"Generated and saved new WandB run ID: {wandb_run_id}")
            except IOError as e:
                print(f"Warning: Could not write WandB run ID to {wandb_run_id_file_path}: {e}")
                print("Resuming WandB run might not work correctly across restarts.")







    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir, exist_ok=True) # exist_ok=True for robustness

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    charset, char_to_idx, idx_to_char, n_characters = initialize_charset(args.dataset)
    config["charset_size"] = n_characters         # ← add this line
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
        print("Initializing SimpleRNN model.")
        if args.updater != 'backprop':
            raise ValueError("SimpleRNN model only supports the 'backprop' updater.")
        rnn = SimpleRNN(base_input_size, config["n_hidden"], output_size, config["n_layers"], dropout_rate=0)
    elif args.model_type == 'hebby':
        print(f"Initializing HebbyRNN model with '{args.updater}' updater.")
        rnn = HebbyRNN(
            base_input_size, config["n_hidden"], output_size, config["n_layers"], charset,
            normalize=args.normalize, residual_connection=args.residual_connection,
            clip_weights=args.clip_weights, updater=args.updater,
            plast_clip=config["plast_clip"], batch_size=config["batch_size"],
            forget_rate=config["forget_rate"], plast_proportion=config["plast_proportion"]
        )
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    # Optimizer is only needed for backprop
    if args.updater == 'backprop':
        optimizer = torch.optim.Adam(rnn.parameters(), lr=config['learning_rate'])

    state = {
        "training_instance": 0,
        "last_n_rewards": [0],
        "last_n_reward_avg": 0,
        "wandb_step": 0,  # Initialize wandb_step
        "log_norms_now": False,
    }

    # --- Resume from Checkpoint ---

    # Attempt to resume if --resume_checkpoint is given OR if latest_checkpoint.pth exists
    # Prioritize explicit --resume_checkpoint if provided
    checkpoint_to_load = None
    if args.resume_checkpoint:
        if os.path.isfile(args.resume_checkpoint):
            checkpoint_to_load = args.resume_checkpoint
            print(f"Attempting to resume from explicit checkpoint: {checkpoint_to_load}")
        else:
            print(f"Warning: Explicit resume checkpoint {args.resume_checkpoint} not found.")
            # Fallback to latest if explicit one is missing but latest exists
            if os.path.isfile(latest_checkpoint_path):
                print(f"Falling back to latest checkpoint: {latest_checkpoint_path}")
                checkpoint_to_load = latest_checkpoint_path
            else:
                print("No checkpoint found to resume from. Starting from scratch.")

    elif os.path.isfile(latest_checkpoint_path): # No explicit resume, but latest exists
        checkpoint_to_load = latest_checkpoint_path
        print(f"Found latest checkpoint. Attempting to resume from: {checkpoint_to_load}")
    else:
        print("No checkpoint specified and no latest_checkpoint.pth found. Starting from scratch.")


    if checkpoint_to_load:
        try:
            # Pass device to load_checkpoint
            rnn, optimizer, start_iter, loaded_main_state, loaded_config = load_checkpoint(
                checkpoint_to_load, rnn, config, optimizer=optimizer, device=device
            )
            state.update(loaded_main_state) # Update your main program state
            print(f"resumed, starting from iter: {start_iter}")




        except FileNotFoundError: # Should be rare due to os.path.isfile checks, but good for safety
            print(f"Warning: Checkpoint file {checkpoint_to_load} not found during load attempt. Starting from scratch.")
            start_iter = 1 # Reset start_iter
            state = { # Reset state
                "training_instance": 0,
                "last_n_rewards": [0],
                "last_n_reward_avg": 0,
                "wandb_step": 0,  # Initialize wandb_step
                "log_norms_now": False,
            }
        except Exception as e:
            if isinstance(e, RuntimeError) and "configuration mismatch" in str(e):
                raise  # abort run, no fallbacks
            print(f"Error loading checkpoint {checkpoint_to_load}: {e}. Starting from scratch.")
            import traceback
            traceback.print_exc()
            start_iter = 1 # Reset start_iter
            state = { # Reset state
                "training_instance": 0,
                "last_n_rewards": [0],
                "last_n_reward_avg": 0,
                "wandb_step": 0,  # Initialize wandb_step
                "log_norms_now": False,
            }
            if torch.cuda.is_available(): # If loading failed but cuda is an option
                print("Fallback: Moving freshly initialized model to GPU after failed checkpoint load.")
                rnn = rnn.to(device)
                if optimizer: # If backprop and optimizer exists
                    # Move optimizer states to device if it was re-initialized
                    # This is more for general robustness if optimizer was somehow re-created
                    # For your Hebby case, optimizer is None, so this part is less critical here.
                    for state_val in optimizer.state.values():
                        for k, v in state_val.items():
                            if isinstance(v, torch.Tensor):
                                state_val[k] = v.to(device)
            else:
                print("Fallback: Model remains on CPU after failed checkpoint load (no CUDA).")

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
            "effective_lr": args.learning_rate * (1-args.plast_proportion + args.plast_proportion * args.plast_clip),
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
        }
        # Key change here: use the determined wandb_run_id and resume="allow"
        print(f"tags given to wandb: {args.tags}")
        if state['wandb_step'] == 0 or is_new_id: # If starting fresh or new ID generated
            wandb.init(project="hebby",
                    group=args.group,
                    notes=args.notes,
                    tags=args.tags,
                    config=wandb_config,
                    id=wandb_run_id,  # Use the persistent ID
                    )
        else:
            print(f"WandB run ID: {wandb_run_id}, attempting to resume from iteration {start_iter}")
            resume_from_string = f"{wandb_run_id}?_step={state['wandb_step'] -1}" if wandb_run_id else None
            print(f"Resuming from: {resume_from_string}")
            wandb.init(project="hebby",
                    group=args.group,
                    notes=args.notes,
                    tags=args.tags,
                    config=wandb_config,
                    #    id=wandb_run_id, # Use the persistent ID
                    #    resume="must",
                    resume_from=resume_from_string)  # Allow resuming the run if ID exists on WandB server

        print(f"Initialized WandB with Run ID: {wandb.run.id if wandb.run else 'None'}")
        if wandb.run and wandb.run.resumed:
            print(f"Successfully resumed WandB run: {wandb.run.id}")


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
        current_correct_plot_interval = 0 # For last character accuracy
        current_step_acc_t1_plot_interval = 0.0 # For step-wise T1 accuracy
        current_step_acc_t2_plot_interval = 0.0 # For step-wise Top-2 accuracy
        num_detailed_calcs_in_plot_interval = 0
        
        # These will be populated per iteration if log_outputs_for_train is true
        # and then cleared after print_freq, as per your original logic.
        all_outputs_for_print_freq = [] 
        all_labels_for_print_freq = []

        for iter in range(start_iter, args.n_iters + 1):
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
            state["log_norms_now"] = (iter % args.print_freq == 0)
            # The train function returns step-by-step outputs for the first batch item if log_outputs=True
            output, loss, og_loss, reg_loss, current_iter_all_outputs, current_iter_all_labels = train(
                line_tensor, onehot_line_tensor, rnn, config, state, optimizer, log_outputs=log_outputs_for_train
            )
            
            # Store detailed outputs if they were generated FOR THIS ITERATION for print_freq
            if log_outputs_for_train:
                all_outputs_for_print_freq = current_iter_all_outputs
                all_labels_for_print_freq = current_iter_all_labels

            # --- Accumulate Stats for the PLOT (Averaging) Interval ---
            current_loss_plot_interval += loss # Use instantaneous loss (sequence average loss from train)

            # Calculate correctness based on the *last* character prediction for print_freq
            if output is not None and line_tensor.numel() > 0:
                 output_detached = output.detach()
                 last_output_first_item = output_detached[0] 
                 last_target_idx_first_item = line_tensor[0, -1].item() 
                 predicted_idx_last_char = torch.argmax(last_output_first_item).item()
                 is_correct_last_char = (predicted_idx_last_char == last_target_idx_first_item)
                 current_correct_plot_interval += 1 if is_correct_last_char else 0

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
                    
                    # Accumulate these sequence-specific accuracies for the print_freq interval
                    current_step_acc_t1_plot_interval += seq_step_accuracy_t1
                    current_step_acc_t2_plot_interval += seq_step_accuracy_top2
                    num_detailed_calcs_in_plot_interval += 1

                    # Clear the detailed output lists after use for this print_freq iteration
                    all_outputs_for_print_freq.clear()
                    all_labels_for_print_freq.clear()
                # No WandB logging directly in this very frequent print_freq block

            # ==============================================================
            # --- Averaged Console Print & WandB Log Period (print_freq) ---
            # ==============================================================
            if args.print_freq > 0 and iter % args.print_freq == 0:
                avg_loss_plot = current_loss_plot_interval / args.print_freq
                avg_acc_last_char_plot = current_correct_plot_interval / args.print_freq

                print(f'--- Avg Interval Data (ending @ iter {iter}) ---')
                print(f'  Avg Loss ({args.print_freq} iters): {avg_loss_plot:.4f}')
                print(f'  Avg Acc (last char, {args.print_freq} iters): {avg_acc_last_char_plot:.4f}')

                avg_step_acc_t1_plot = (current_step_acc_t1_plot_interval / num_detailed_calcs_in_plot_interval) if num_detailed_calcs_in_plot_interval > 0 else 0.0
                avg_step_acc_t2_plot = (current_step_acc_t2_plot_interval / num_detailed_calcs_in_plot_interval) if num_detailed_calcs_in_plot_interval > 0 else 0.0
                print(f'  Avg Step Acc T1 ({num_detailed_calcs_in_plot_interval} seqs): {avg_step_acc_t1_plot:.4f}')
                print(f'  Avg Step Acc Top2 ({num_detailed_calcs_in_plot_interval} seqs): {avg_step_acc_t2_plot:.4f}')
                print(f'-------------------------------------------')

                if args.track:
                    # Calculate and gather norms
                    model_norms = rnn.get_all_norms()
                    
                    # Calculate averages for each type
                    high_plast_weights = [v for k, v in model_norms.items() if 'high_plast_weight_norm' in k]
                    low_plast_weights = [v for k, v in model_norms.items() if 'low_plast_weight_norm' in k]
                    high_plast_updates = [v for k, v in model_norms.items() if 'high_plast_update_norm' in k]
                    low_plast_updates = [v for k, v in model_norms.items() if 'low_plast_update_norm' in k]
                    # Also keep track of backprop norms if needed (check if 'grad_norm' exists)
                    grad_norms = [v for k, v in model_norms.items() if 'grad_norm' in k]
                    all_weights = [v for k, v in model_norms.items() if 'weight_norm' in k] # For backprop or combined hebby

                    avg_hp_w_norm = sum(high_plast_weights) / len(high_plast_weights) if high_plast_weights else 0.0
                    avg_lp_w_norm = sum(low_plast_weights) / len(low_plast_weights) if low_plast_weights else 0.0
                    avg_hp_u_norm = sum(high_plast_updates) / len(high_plast_updates) if high_plast_updates else 0.0
                    avg_lp_u_norm = sum(low_plast_updates) / len(low_plast_updates) if low_plast_updates else 0.0
                    avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0 # For backprop
                    avg_weight_norm = sum(all_weights) / len(all_weights) if all_weights else 0.0 # For backprop/combined

                    log_data = {
                        "iter": iter,
                        "avg_loss": avg_loss_plot,
                        "avg_accuracy": avg_acc_last_char_plot,
                        "avg_step_acc_t1": avg_step_acc_t1_plot,
                        "avg_step_acc_t2": avg_step_acc_t2_plot,
                        "avg_weight_norm": avg_weight_norm, # Combined / Backprop
                        "avg_grad_update_norm": avg_grad_norm or (avg_hp_u_norm + avg_lp_u_norm), # Backprop or sum of Hebby
                        "avg_high_plast_weight_norm": avg_hp_w_norm,
                        "avg_low_plast_weight_norm": avg_lp_w_norm,
                        "avg_high_plast_update_norm": avg_hp_u_norm,
                        "avg_low_plast_update_norm": avg_lp_u_norm,
                        # **model_norms # Log all individual norms too
                    }
                    wandb.log(log_data, step=state["wandb_step"], commit=True)
                    # print some norm data too
                    print(f'  Avg Weight Norm: {avg_weight_norm:.4f}')
                    print(f'  Avg Grad/Update Norm: {avg_grad_norm:.4f}')
                
                state["wandb_step"] += 1
                current_loss_plot_interval = 0.0
                current_correct_plot_interval = 0
                current_step_acc_t1_plot_interval = 0.0
                current_step_acc_t2_plot_interval = 0.0
                num_detailed_calcs_in_plot_interval = 0

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
                    trigger_sync()
                except Exception as e:
                    print(f"Error during W&B sync: {e}")

            # ==============================================================
            # --- Checkpointing ---
            # ==============================================================
            if iter % args.checkpoint_save_freq == 0:
                checkpoint_state = {
                    'iter': iter + 1,
                    'model_state_dict': rnn.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                    'main_program_state': state,
                    'config': config,
                    'torch_rng_state': torch.get_rng_state(),
                    'numpy_rng_state': np.random.get_state(),
                }
                save_checkpoint(checkpoint_state, args.checkpoint_dir, "latest_checkpoint.pth") # Overwrites latest

        # End of training loop
        if args.track and wandb.run:
            print("Final W&B sync and finish run...")
            trigger_sync() 
            wandb.finish()


    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Attempting to save final checkpoint...")
        # Optionally save a final checkpoint on interrupt
        if args.checkpoint_dir: # Ensure dir is specified
            final_checkpoint_state = {
                'iter': iter + 1 if 'iter' in locals() else start_iter,
                'model_state_dict': rnn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                'main_program_state': state,
                'config': config,
                'torch_rng_state': torch.get_rng_state(),
                'numpy_rng_state': np.random.get_state(),
            }
            save_checkpoint(final_checkpoint_state, args.checkpoint_dir, "interrupt_checkpoint.pth")
            save_checkpoint(final_checkpoint_state, args.checkpoint_dir, "latest_checkpoint.pth") # also update latest
        print("Finishing up...")
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
