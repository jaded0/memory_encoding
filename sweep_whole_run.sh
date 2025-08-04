#!/bin/bash --login
# ==============================================================================
# sweep_whole_run.sh - SLURM submission script for running hebby.py sweeps
#                      using job arrays.
# ==============================================================================

# --- SLURM Directives (MUST BE NEAR THE TOP) ---
#SBATCH --time=3:00:00        # Max walltime per task
#SBATCH --nodes=1              # Request 1 node per task
#SBATCH --ntasks=10             # Request 1 task per job array instance
#SBATCH --cpus-per-task=1      # Explicitly request 1 CPU core for that task
#SBATCH --gpus=1               # Request 1 GPU per task
#SBATCH --mem-per-cpu=8000M    # Memory per CPU core (e.g., 8GB)
#SBATCH --mail-type=BEGIN,END,FAIL # Email notifications
#SBATCH --job-name=hebby_sweep # Base job name
#SBATCH --array=0-11%15        # CALCULATE AND UPDATE THIS RANGE (see below) - Example: Run max 10 of 12 jobs
#SBATCH --output=slurm_logs/hebby_sweep_%A_%a.out # Ensure slurm_logs directory exists! %A=jobID, %a=taskID
#SBATCH --mail-user=jaden.lorenc@gmail.com # Your email address


# ======================== Parameter Definitions & Calculation ================
echo "--- Preparing Sweep Parameters ---"
# --- Define Hyperparameter Options for Architecture Sweep ---
declare -a model_types=('ethereal' 'rnn')
declare -a updaters=('backprop' 'dfa' 'bptt')
declare -a enable_recurrence=('true' 'false')

# --- Calculate Total Number of Jobs ---
num_model_types=${#model_types[@]}
num_updaters=${#updaters[@]}
num_enable_recurrence=${#enable_recurrence[@]}
total_jobs=$((num_model_types * num_updaters * num_enable_recurrence))
last_job_index=$((total_jobs - 1)) # SLURM array indices are 0-based

# *** IMPORTANT: Update the --array directive above with the calculated last_job_index ***
#     You might need to run this calculation part once manually first, or
#     submit a preliminary job that just calculates and echoes the range,
#     or accept that the #SBATCH line might have a placeholder range initially.
#     For now, I'll leave the example range (0-11), assuming 2*3*2=12 jobs.
echo "Sweep Info: Total Jobs = $total_jobs (Indices 0-$last_job_index)"
echo "Note: Ensure #SBATCH --array line matches 0-$last_job_index"


# ======================== Environment Setup ===================================
echo "--- Task ID: $SLURM_ARRAY_TASK_ID / Job ID: $SLURM_ARRAY_JOB_ID ---"
echo "--- Setting up Environment ---"
# Load Conda environment
# source /path/to/your/miniconda3/etc/profile.d/conda.sh # Adjust path if needed
conda activate hebby
echo "Activated Conda environment: $CONDA_DEFAULT_ENV"

# Configure W&B and HuggingFace for offline use
export WANDB_MODE=offline
export WANDB_EXECUTABLE=$CONDA_PREFIX/bin/python
export HF_OFFLINE=1
export HF_DATASETS_OFFLINE=1
echo "WANDB_MODE set to: $WANDB_MODE"
echo "HF Offline mode enabled."
echo "--- Environment Setup Complete ---"

# ======================== Parameter Calculation for this Task =================
# Map the SLURM_ARRAY_TASK_ID to specific hyperparameter indices
# Order: enable_recurrence -> updater -> model_type
idx_er=$((SLURM_ARRAY_TASK_ID % num_enable_recurrence))
idx_up=$(((SLURM_ARRAY_TASK_ID / num_enable_recurrence) % num_updaters))
idx_mt=$(((SLURM_ARRAY_TASK_ID / (num_enable_recurrence * num_updaters)) % num_model_types))

# Get the actual parameter values
MODEL_TYPE=${model_types[$idx_mt]}
UPDATER=${updaters[$idx_up]}
ENABLE_RECURRENCE=${enable_recurrence[$idx_er]}

# ======================== Experiment Identification (Dynamic) =================
GROUP='arch_sweep'
RUN_NOTES="Model=${MODEL_TYPE}_Updater=${UPDATER}_Recurrence=${ENABLE_RECURRENCE}_TaskID=${SLURM_ARRAY_TASK_ID}"
TAGS=(arch_sweep sweep)

# ======================== Fixed Parameters (Not Swept) - Based on whole_run.sh ========================
INPUT_MODE='last_one'        # From whole_run.sh
LEARNING_RATE=1e-4           # From whole_run.sh
PLAST_LEARNING_RATE=1e-10    # From whole_run.sh
PLAST_CLIP=5e3               # From whole_run.sh (5e3 instead of 1e4)
GRAD_CLIP=0                  # From whole_run.sh
IMPRINT_RATE=0.3             # From whole_run.sh
FORGET_RATE=0.01             # From whole_run.sh (0.01 instead of variable)
SELF_GRAD=0                  # From whole_run.sh
NORMALIZE=false              # From whole_run.sh
CLIP_WEIGHTS=0               # From whole_run.sh
HIDDEN_SIZE=256              # From whole_run.sh
NUM_LAYERS=3                 # From whole_run.sh
RESIDUAL_CONNECTION=false    # From whole_run.sh
POS_ENCODING=128             # From whole_run.sh
DATASET='2_small_palindrome_dataset_vary_length' # From whole_run.sh
BATCH_SIZE=16                # From whole_run.sh (16 instead of 4)
N_ITERS=1000000000           # From whole_run.sh
PRINT_FREQ=5000              # From whole_run.sh (5000 instead of 2500)
CHECKPOINT_SAVE_FREQ=500000  # From whole_run.sh
PLAST_PROPORTION=0.2         # From whole_run.sh

# ======================== Execution ===========================================
echo "--- Starting Training Task $SLURM_ARRAY_TASK_ID ---"
echo "  Parameters for this task:"
echo "    Model Type: $MODEL_TYPE"
echo "    Updater: $UPDATER"
echo "    Enable Recurrence: $ENABLE_RECURRENCE"
echo "    Input Mode: $INPUT_MODE"
echo "    Learning Rate: $LEARNING_RATE"
echo "    Plast Clip: $PLAST_CLIP"
echo "    Forget Rate: $FORGET_RATE"
echo "    Grad Clip: $GRAD_CLIP"
echo "    Plast Proportion: $PLAST_PROPORTION"
echo "  W&B Group: $GROUP"
echo "  W&B Notes/Run Name: $RUN_NOTES"
echo "  Output File: slurm_logs/hebby_sweep_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
echo "---"

# Run the python script
python -u hebby.py \
    --model_type $MODEL_TYPE \
    --updater $UPDATER \
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
    --checkpoint_save_freq $CHECKPOINT_SAVE_FREQ \
    --track true \
    --group "$GROUP" \
    --tags "${TAGS[@]}" \
    --notes "$RUN_NOTES" \
    --plast_proportion $PLAST_PROPORTION \
    --enable_recurrence $ENABLE_RECURRENCE

echo "--- Task $SLURM_ARRAY_TASK_ID Finished ---"

# Optional Post-Run Cleanup can remain here
# ==============================================================================
