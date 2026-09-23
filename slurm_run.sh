#!/bin/bash --login
# ==============================================================================
# slurm_run.sh - SLURM submission script for a single long train.py run
#
#   sbatch slurm_run.sh               # submit from the repo root (on the login node)
#   SMOKE=1 ./slurm_run.sh            # login node: cache HF datasets + check env, then exit
#
# The job name (--job-name below) is the experiment's identity: it names the
# checkpoint dir, and the run always passes --resume, so a requeue, a preemption
# or sweeps/bulk_restart.sh continues from checkpoints/<job-name>/latest_checkpoint.pth.
# Give each new experiment a new job name, or it will continue the old one.
#
# SMOKE=1 runs this exact config for a few CPU iterations with W&B disabled and
# Hugging Face online, into checkpoints/_smoke, so the dataset is cached for the
# offline compute nodes and a broken environment fails fast. Never set it as the
# default here. Local runs: run_training.sh. Sweeps: sweeps/.
# ==============================================================================

# --- SLURM Directives ---
#SBATCH --time=16-00:00:00        # Max walltime (HH:MM:SS)
#SBATCH --signal=B:USR1@600    # warn training 10 min before the limit (see forward_signals.sh)
#SBATCH --ntasks=10            # Number of CPU cores requested
#SBATCH --nodes=1              # Number of nodes requested
#SBATCH --gpus=1               # Number of GPUs requested
#SBATCH --mem-per-cpu=8000M    # Memory per CPU core (e.g., 8GB)
#SBATCH --mail-type=BEGIN,END,FAIL # Email notifications
#SBATCH --job-name=text_scale_5_3 # Job name in queue
#SBATCH --output=hebby_train_%j.out # Standard output file (%j = job ID)
#SBATCH --mail-user=jaden.lorenc@gmail.com # Your email address
#SBATCH --qos=standby      # Make it preemptable
#SBATCH --requeue          # Requeue on preemption or failure

# Resolve repo root regardless of how this script was launched: sbatch runs a
# spooled copy of this file, so $0 won't point here under sbatch, but
# SLURM_SUBMIT_DIR (the directory sbatch was invoked from) does.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" || exit 1
SMOKE=${SMOKE:-0}

# ======================== Environment Setup ===================================
echo "--- Setting up Environment ---"
# Load Conda environment
# source /path/to/your/miniconda3/etc/profile.d/conda.sh # Adjust path if needed
conda activate hebby || { [[ $SMOKE == 1 ]] && { echo "conda activate hebby failed"; exit 1; }; }
echo "Activated Conda environment: $CONDA_DEFAULT_ENV"

# Configure W&B and HuggingFace for offline use (if needed)
export WANDB_MODE=offline
export WANDB_EXECUTABLE=$CONDA_PREFIX/bin/python # Ensure W&B uses the conda python
if [[ $SMOKE == 1 ]]; then
    export WANDB_MODE=disabled   # login node has internet: let HF download and cache
else
    export HF_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    echo "HF Offline mode enabled."
fi
echo "WANDB_MODE set to: $WANDB_MODE"

# Optional: Check GPU status
# nvidia-smi

echo "--- Environment Setup Complete ---"

# ======================== Experiment Identification ===========================

# --- Checkpointing ---
# The checkpoint dir is keyed by job name so every requeue (same job ID) and
# every bulk_restart.sh resubmission (new job ID, same name) finds it.
EXPERIMENT_NAME="$SLURM_JOB_NAME"
CHECKPOINT_DIR="./checkpoints/${EXPERIMENT_NAME}" # Persistent directory for this experiment
RESUME=true                  # Continue latest_checkpoint.pth if present (see header)

if [[ $SMOKE != 1 ]]; then
    # --- add to run list (for bulk_restart.sh) and refuse duplicates ---
    RUN_LIST="current_runs.txt"
    grep -qxF "$EXPERIMENT_NAME" "$RUN_LIST" 2>/dev/null || echo "$EXPERIMENT_NAME" >> "$RUN_LIST"
    if squeue -h -n "$EXPERIMENT_NAME" -o "%A" \
           | grep -v "^${SLURM_JOB_ID}$" \
           | grep -q .; then
      echo "⏩  $EXPERIMENT_NAME already RUNNING or PENDING – aborting."; exit 0
    fi
fi

# --- Experiment Identification (W&B) ---
# Each resume starts a new W&B run that records resumed_from_checkpoint and
# resumed_at_iter (W&B run resumption is not supported); group by experiment.
GROUP=$EXPERIMENT_NAME
NOTES="an attempt to scale prematurely"
TAGS=(mega big_scale)

CHECKPOINT_SAVE_FREQ=500

# ======================== Core Training Parameters ============================
# --- Training Strategy ---
# MODEL_TYPE: 'ephemeral' for the plastic model, 'rnn' for a standard SimpleRNN.
# UPDATER: 'dfa' for Direct Feedback Alignment, 'backprop' for standard backpropagation, 'bptt' for backpropagation through time.
#
# To run EphemeralRNN with backprop: MODEL_TYPE='ephemeral', UPDATER='backprop', LEARNING_RATE=1e-5 (example)
# To run SimpleRNN with backprop: MODEL_TYPE='rnn', UPDATER='backprop', LEARNING_RATE=1e-3 (example)
# To run EphemeralRNN with BPTT: MODEL_TYPE='ephemeral', UPDATER='bptt', LEARNING_RATE=1e-5 (example)
#
MODEL_TYPE='ephemeral'           # ephemeral | rnn
UPDATER='dfa'                # dfa | backprop | bptt
INPUT_MODE='last_one'        # last_one | last_two

# --- Learning Rates & Clipping ---
LEARNING_RATE=1e-5           # Base learning rate
PLAST_CLIP=1e3               # Plasticity (learning-rate multiplier) of ephemeral weights, alpha
GRAD_CLIP=0                  # Element-wise clip on ephemeral-weight updates (0 = off)

# --- Ephemeral Weights (ignored by the rnn baseline) ---
FORGET_RATE=0.1              # Fraction of each ephemeral weight removed per step: w <- (1 - FORGET_RATE) w
SELF_GRAD=0                  # Experimental recurrent replacement
PLAST_PROPORTION=0.1         # Proportion of weights that are plastic in ephemeral layers  # <-- Add this line
ENABLE_RECURRENCE=false       # Whether to enable recurrent hidden state connections

# --- Regularization & Stability ---
NORMALIZE=false              # Normalize weights post-update (true/false)
CLIP_WEIGHTS=0               # Max absolute weight value (0=off)

# ======================== Model Architecture ==================================
HIDDEN_SIZE=1024              # RNN hidden state units
NUM_LAYERS=3                 # Number of RNN layers
RESIDUAL_CONNECTION=false     # Use skip connections (true/false)
POS_ENCODING=0             # Positional encoding dimension (0=off)

# ======================== Data & Training Loop ================================
# --- Dataset ---
DATASET='roneneldan/tinystories' # palindrome_dataset | roneneldan/tinystories | 4_palindrome_dataset_vary_length | 2_resequence | long_range_memory_dataset
BATCH_SIZE=16                 # Sequences per batch

# --- Loop Control & Logging ---
N_ITERS=10000000           # Total training steps (iterations)
PRINT_FREQ=50                # Console print basic avg loss/acc frequency
LOG_FREQ=500              # W&B sync frequency for offline mode

# ======================== Smoke Mode ==========================================
TRACK=true
if [[ $SMOKE == 1 ]]; then
    TRACK=false
    EXPERIMENT_NAME="_smoke"
    GROUP="_smoke"
    CHECKPOINT_DIR="./checkpoints/_smoke"
    RESUME=false
    N_ITERS=20
    PRINT_FREQ=10
    CHECKPOINT_SAVE_FREQ=$N_ITERS
fi

# ======================== Execution ===========================================
echo "--- Starting Training ---"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Job Name: $SLURM_JOB_NAME"
[[ $SMOKE == 1 ]] && echo "  SMOKE mode: $N_ITERS iterations, W&B disabled, HF online"
echo "  Group: $GROUP | Model: $MODEL_TYPE | Updater: $UPDATER | Input: $INPUT_MODE | LR: $LEARNING_RATE"
echo "  Dataset: $DATASET | Batch: $BATCH_SIZE | Hidden: $HIDDEN_SIZE | PosEnc: $POS_ENCODING"
echo "  Checkpoint Dir: $CHECKPOINT_DIR (resume: $RESUME, save every $CHECKPOINT_SAVE_FREQ)"

# Create checkpoint directory if it doesn't exist
mkdir -p "$CHECKPOINT_DIR"

# Save a copy of this script for reproducibility (bulk_restart.sh resubmits it)
[[ $SMOKE == 1 ]] || cp "$0" "$CHECKPOINT_DIR/run_used.sh"

source ./sweeps/forward_signals.sh
forward_signals python -u train.py \
    --model_type $MODEL_TYPE \
    --updater $UPDATER \
    --input_mode $INPUT_MODE \
    --learning_rate $LEARNING_RATE \
    --plast_clip $PLAST_CLIP \
    --grad_clip $GRAD_CLIP \
    --forget_rate $FORGET_RATE \
    --self_grad $SELF_GRAD \
    --normalize $NORMALIZE \
    --clip_weights $CLIP_WEIGHTS \
    --hidden_size $HIDDEN_SIZE \
    --num_layers $NUM_LAYERS \
    --residual_connection $RESIDUAL_CONNECTION \
    --positional_encoding_dim $POS_ENCODING \
    --dataset $DATASET \
    --batch_size $BATCH_SIZE \
    --n_iters $N_ITERS \
    --print_freq $PRINT_FREQ \
    --log_freq $LOG_FREQ \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --checkpoint_save_freq $CHECKPOINT_SAVE_FREQ \
    --resume $RESUME \
    --track $TRACK \
    --group "$GROUP" \
    --tags "${TAGS[@]}" \
    --notes "$NOTES" \
    --plast_proportion $PLAST_PROPORTION \
    --enable_recurrence $ENABLE_RECURRENCE

status=$?
echo "--- Training Finished (exit $status) ---"
# Smoke runs must fail loudly; normal runs keep the original always-0 exit.
[[ $SMOKE == 1 ]] && exit $status

# ==============================================================================
