#!/bin/bash --login
# ==============================================================================
# whole_run.sh - SLURM submission script for running hebby.py
# ==============================================================================

# --- SLURM Directives ---
#SBATCH --time=72:00:00        # Max walltime (HH:MM:SS)
#SBATCH --ntasks=10            # Number of CPU cores requested
#SBATCH --nodes=1              # Number of nodes requested
#SBATCH --gpus=1               # Number of GPUs requested
#SBATCH --mem-per-cpu=8000M    # Memory per CPU core (e.g., 8GB)
#SBATCH --mail-type=BEGIN,END,FAIL # Email notifications
#SBATCH --job-name=hebby_train # Job name in queue
#SBATCH --output=hebby_train_%j.out # Standard output file (%j = job ID)
#SBATCH --mail-user=jaden.lorenc@gmail.com # Your email address

# ======================== Environment Setup ===================================
echo "--- Setting up Environment ---"
# Load Conda environment
# source /path/to/your/miniconda3/etc/profile.d/conda.sh # Adjust path if needed
conda activate hebby
echo "Activated Conda environment: $CONDA_DEFAULT_ENV"

# Configure W&B and HuggingFace for offline use (if needed)
export WANDB_MODE=offline
export WANDB_EXECUTABLE=$CONDA_PREFIX/bin/python # Ensure W&B uses the conda python
export HF_OFFLINE=1
export HF_DATASETS_OFFLINE=1
echo "WANDB_MODE set to: $WANDB_MODE"
echo "HF Offline mode enabled."

# Optional: Check GPU status
# nvidia-smi

echo "--- Environment Setup Complete ---"

# ======================== Experiment Identification ===========================
# --- W&B Logging ---
# --- Experiment Identification (W&B) ---
GROUP='check_phenomenon'
NOTES="very very small, otherwise identical to chocolate-bee."

# ======================== Core Training Parameters ============================
# --- Training Strategy ---
UPDATE_RULE='static_plastic_candidate'       # backprop | static_plastic_candidate | dfa | etc.
INPUT_MODE='last_one'        # last_one | last_two

# --- Learning Rates & Clipping ---
LEARNING_RATE=1e-5           # Base learning rate
PLAST_LEARNING_RATE=1e-10    # Plasticity LR (for specific rules)
PLAST_CLIP=1e4               # Plasticity max value (for specific rules)
GRAD_CLIP=0                  # Max gradient norm

# --- Hebbian / Plasticity Specifics (ignored by backprop) ---
IMPRINT_RATE=0.3             # Hebbian imprint strength
FORGET_RATE=0.01              # Weight decay/forgetting factor
SELF_GRAD=0                  # Experimental recurrent replacement

# --- Regularization & Stability ---
NORMALIZE=false              # Normalize weights post-update (true/false)
CLIP_WEIGHTS=0               # Max absolute weight value (0=off)

# ======================== Model Architecture ==================================
HIDDEN_SIZE=128              # RNN hidden state units
NUM_LAYERS=3                 # Number of RNN layers
RESIDUAL_CONNECTION=false    # Use skip connections (true/false)
POS_ENCODING=64             # Positional encoding dimension (0=off)

# ======================== Data & Training Loop ================================
# --- Dataset ---
DATASET='palindrome_dataset_vary_length' # palindrome_dataset | roneneldan/tinystories | palindrome_dataset_vary_length | 2_resequence | long_range_memory_dataset
BATCH_SIZE=4                 # Sequences per batch

# --- Loop Control & Logging ---
N_ITERS=1000000000           # Total training steps (iterations)
PRINT_FREQ=2500                # Console print basic avg loss/acc frequency
PLOT_FREQ=2500                # WandB log freq + Detailed console print freq
SAVE_FREQUENCY=10000000      # Save model frequency (iters, if implemented)

# ======================== Execution ===========================================
echo "--- Starting Training ---"
echo "  Group: $GROUP | Rule: $UPDATE_RULE | Input: $INPUT_MODE | LR: $LEARNING_RATE"
echo "  Dataset: $DATASET | Batch: $BATCH_SIZE | Hidden: $HIDDEN_SIZE | PosEnc: $POS_ENCODING"

# Run the python script with unbuffered output (-u)
python -u hebby.py \
    --update_rule $UPDATE_RULE \
    --input_mode $INPUT_MODE \
    --learning_rate $LEARNING_RATE \
    --plast_learning_rate $PLAST_LEARNING_RATE \
    --plast_clip $PLAST_CLIP \
    --grad_clip $GRAD_CLIP \
    --imprint_rate $IMPRINT_RATE \
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
    --plot_freq $PLOT_FREQ \
    --save_frequency $SAVE_FREQUENCY \
    --track true \
    --group "$GROUP" \
    --notes "$NOTES"

echo "--- Training Finished ---"

# ======================== Post-Run (Optional) =================================
# Generate memory profile plot if mprof was used during the python run
# mprof plot --output=memory_profile_${SLURM_JOB_ID}.png

# Clean up model data directory
echo "Cleaning up model data..."
rm -f model_data/* # Use -f to force remove without prompts
echo "Cleanup complete."

# ==============================================================================