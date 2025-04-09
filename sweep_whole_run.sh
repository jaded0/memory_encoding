#!/bin/bash --login
# ==============================================================================
# sweep_whole_run.sh - SLURM submission script for running hebby.py sweeps
#                      using job arrays.
# ==============================================================================

# --- SLURM Directives (MUST BE NEAR THE TOP) ---
#SBATCH --time=2:00:00        # Max walltime per task
#SBATCH --nodes=1              # Request 1 node per task
#SBATCH --ntasks=1             # Request 1 task per job array instance
#SBATCH --cpus-per-task=1      # Explicitly request 1 CPU core for that task
#SBATCH --gpus=1               # Request 1 GPU per task
#SBATCH --mem=8G               # Request 8 Gigabytes total memory FOR THE TASK
#SBATCH --mail-type=BEGIN,END,FAIL # Email notifications
#SBATCH --job-name=hebby_sweep # Base job name
#SBATCH --array=0-23%10        # CALCULATE AND UPDATE THIS RANGE (see below) - Example: Run max 10 of 24 jobs
#SBATCH --output=slurm_logs/hebby_sweep_%A_%a.out # Ensure slurm_logs directory exists! %A=jobID, %a=taskID
#SBATCH --mail-user=jaden.lorenc@gmail.com # Your email address


# ======================== Parameter Definitions & Calculation ================
echo "--- Preparing Sweep Parameters ---"
# --- Define Hyperparameter Options ---
declare -a update_rules=('backprop' 'static_plastic_candidate')
declare -a input_modes=('last_one' 'last_two')
declare -a learning_rates=('1e-3' '1e-4' '1e-5')
declare -a plast_clips=('1e3' '1e4')

# --- Calculate Total Number of Jobs ---
num_update_rules=${#update_rules[@]}
num_input_modes=${#input_modes[@]}
num_learning_rates=${#learning_rates[@]}
num_plast_clips=${#plast_clips[@]}
total_jobs=$((num_update_rules * num_input_modes * num_learning_rates * num_plast_clips))
last_job_index=$((total_jobs - 1)) # SLURM array indices are 0-based

# *** IMPORTANT: Update the --array directive above with the calculated last_job_index ***
#     You might need to run this calculation part once manually first, or
#     submit a preliminary job that just calculates and echoes the range,
#     or accept that the #SBATCH line might have a placeholder range initially.
#     For now, I'll leave the example range (0-23), assuming 2*2*3*2=24 jobs.
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
idx_pc=$((SLURM_ARRAY_TASK_ID % num_plast_clips))
idx_lr=$(((SLURM_ARRAY_TASK_ID / num_plast_clips) % num_learning_rates))
idx_im=$(((SLURM_ARRAY_TASK_ID / (num_plast_clips * num_learning_rates)) % num_input_modes))
idx_ur=$(((SLURM_ARRAY_TASK_ID / (num_plast_clips * num_learning_rates * num_input_modes)) % num_update_rules))

# Get the actual parameter values
UPDATE_RULE=${update_rules[$idx_ur]}
INPUT_MODE=${input_modes[$idx_im]}
LEARNING_RATE=${learning_rates[$idx_lr]}
PLAST_CLIP=${plast_clips[$idx_pc]}

# ======================== Experiment Identification (Dynamic) =================
GROUP='Sweep_UR-IM-LR-PC'
RUN_NOTES="Rule=${UPDATE_RULE}_Mode=${INPUT_MODE}_LR=${LEARNING_RATE}_PC=${PLAST_CLIP}_TaskID=${SLURM_ARRAY_TASK_ID}"

# ======================== Fixed Parameters (Not Swept) ========================
PLAST_LEARNING_RATE=1e-10
GRAD_CLIP=0
IMPRINT_RATE=0.3
FORGET_RATE=0.3
SELF_GRAD=0
NORMALIZE=false
CLIP_WEIGHTS=0
HIDDEN_SIZE=256
NUM_LAYERS=3
RESIDUAL_CONNECTION=false
POS_ENCODING=128
DATASET='palindrome_dataset'
BATCH_SIZE=32
N_ITERS=10000000000000
PRINT_FREQ=2500
PLOT_FREQ=25000
SAVE_FREQUENCY=10000000

# ======================== Execution ===========================================
echo "--- Starting Training Task $SLURM_ARRAY_TASK_ID ---"
echo "  Parameters for this task:"
echo "    Update Rule: $UPDATE_RULE"
echo "    Input Mode: $INPUT_MODE"
echo "    Learning Rate: $LEARNING_RATE"
echo "    Plast Clip: $PLAST_CLIP"
echo "  W&B Group: $GROUP"
echo "  W&B Notes/Run Name: $RUN_NOTES"
echo "  Output File: slurm_logs/hebby_sweep_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
echo "---"

# Run the python script
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
    --notes "$RUN_NOTES"

echo "--- Task $SLURM_ARRAY_TASK_ID Finished ---"

# Optional Post-Run Cleanup can remain here
# ==============================================================================