#!/bin/bash --login
# ==============================================================================
# bench_sweep.sh - SLURM submission script for running hebby.py sweeps
#                   using job arrays for benchmarking sweep.
# ==============================================================================

# --- SLURM Directives (MUST NEAR AT THE TOP) ---
#SBATCH --time=48:00:00        # Max walltime per task
#SBATCH --nodes=1              # Request 1 node per task
#SBATCH --ntasks=10             # Request 1 task per job array instance
#SBATCH --cpus-per-task=1      # Explicitly request 1 CPU core for that task
#SBATCH --gpus=1               # Request 1 GPU per task
#SBATCH --mem-per-cpu=8000M    # Memory per CPU core (e.g., 8GB)
#SBATCH --mail-type=BEGIN,END,FAIL # Email notifications
#SBATCH --job-name=bench_sweep # Base job name
#SBATCH --array=0-287%20       # 288 total jobs (1*2*2*6*3*2*2), max 15 concurrent
#SBATCH --output=slurm_logs/bench_sweep_%A_%a.out # Ensure slurm_logs directory exists! %A=jobID, %a=taskID
#SBATCH --mail-user=jaden.lorenc@gmail.com # Your email address


# ======================== Parameter Definitions & Calculation ================
echo "--- Preparing Bench Sweep Parameters ---"
# --- Define Hyperparameter Options for Bench Sweep ---
declare -a model_types=('ethereal')
declare -a updaters=('backprop' 'dfa')
declare -a enable_recurrence=('true' 'false')
declare -a clip_weights=('0' '1' '5' '10' '1e-1' '1e-2')
declare -a plast_clip=('1e4' '1e3' '1e2')
declare -a grad_clip=('0' '1')
declare -a residual_connections=('true' 'false')

# --- Calculate Total Number of Jobs ---
num_model_types=${#model_types[@]}
num_updaters=${#updaters[@]}
num_enable_recurrence=${#enable_recurrence[@]}
num_clip_weights=${#clip_weights[@]}
num_plast_clip=${#plast_clip[@]}
num_grad_clip=${#grad_clip[@]}
num_residual_connections=${#residual_connections[@]}
total_jobs=$((num_model_types * num_updaters * num_enable_recurrence * num_clip_weights * num_plast_clip * num_grad_clip * num_residual_connections))
last_job_index=$((total_jobs - 1)) # SLURM array indices are 0-based

echo "Bench Sweep Info: Total Jobs = $total_jobs (Indices 0-$last_job_index)"
echo "Parameter combinations:"
echo "  model_types: ${num_model_types} (${model_types[@]})"
echo "  updaters: ${num_updaters} (${updaters[@]})"
echo "  enable_recurrence: ${num_enable_recurrence} (${enable_recurrence[@]})"
echo "  clip_weights: ${num_clip_weights} (${clip_weights[@]})"
echo "  plast_clip: ${num_plast_clip} (${plast_clip[@]})"
echo "  grad_clip: ${num_grad_clip} (${grad_clip[@]})"
echo "  residual_connections: ${num_residual_connections} (${residual_connections[@]})"


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
# Order: grad_clip -> plast_clip -> clip_weights -> enable_recurrence -> updater -> model_type -> residual_connection
idx_gc=$((SLURM_ARRAY_TASK_ID % num_grad_clip))
idx_pc=$(((SLURM_ARRAY_TASK_ID / num_grad_clip) % num_plast_clip))
idx_cw=$(((SLURM_ARRAY_TASK_ID / (num_grad_clip * num_plast_clip)) % num_clip_weights))
idx_er=$(((SLURM_ARRAY_TASK_ID / (num_grad_clip * num_plast_clip * num_clip_weights)) % num_enable_recurrence))
idx_up=$(((SLURM_ARRAY_TASK_ID / (num_grad_clip * num_plast_clip * num_clip_weights * num_enable_recurrence)) % num_updaters))
idx_mt=$(((SLURM_ARRAY_TASK_ID / (num_grad_clip * num_plast_clip * num_clip_weights * num_enable_recurrence * num_updaters)) % num_model_types))
idx_rc=$(((SLURM_ARRAY_TASK_ID / (num_grad_clip * num_plast_clip * num_clip_weights * num_enable_recurrence * num_updaters * num_model_types)) % num_residual_connections))

# Get the actual parameter values
MODEL_TYPE=${model_types[$idx_mt]}
UPDATER=${updaters[$idx_up]}
ENABLE_RECURRENCE=${enable_recurrence[$idx_er]}
CLIP_WEIGHTS=${clip_weights[$idx_cw]}
PLAST_CLIP=${plast_clip[$idx_pc]}
GRAD_CLIP=${grad_clip[$idx_gc]}
RESIDUAL_CONNECTION=${residual_connections[$idx_rc]}

# ======================== Experiment Identification (Dynamic) =================
GROUP='bench_sweep'
RUN_NOTES="Model=${MODEL_TYPE}_Updater=${UPDATER}_Recurrence=${ENABLE_RECURRENCE}_ClipWeights=${CLIP_WEIGHTS}_PlastClip=${PLAST_CLIP}_GradClip=${GRAD_CLIP}_ResidualConnection=${RESIDUAL_CONNECTION}_TaskID=${SLURM_ARRAY_TASK_ID}"
TAGS=(bench_sweep sweep long_sweep)

# ======================== Fixed Parameters (Not Swept) - Based on whole_run.sh ========================
INPUT_MODE='last_one'        # From whole_run.sh
LEARNING_RATE=1e-3           # From whole_run.sh
PLAST_LEARNING_RATE=1e-10    # From whole_run.sh
IMPRINT_RATE=0.3             # From whole_run.sh
FORGET_RATE=0.01             # From whole_run.sh (0.01 instead of variable)
SELF_GRAD=0                  # From whole_run.sh
NORMALIZE=false              # From whole_run.sh
HIDDEN_SIZE=256              # From whole_run.sh
NUM_LAYERS=3                 # From whole_run.sh
POS_ENCODING=0             # From whole_run.sh
DATASET='2_small_palindrome_dataset_vary_length' # From whole_run.sh
BATCH_SIZE=16                # From whole_run.sh (16 instead of 4)
N_ITERS=8000000           # From whole_run.sh
PRINT_FREQ=5000              # From whole_run.sh (5000 instead of 2500)
CHECKPOINT_SAVE_FREQ=500000  # From whole_run.sh
PLAST_PROPORTION=0.2         # From whole_run.sh

# ======================== Execution ===========================================
echo "--- Starting Bench Sweep Training Task $SLURM_ARRAY_TASK_ID ---"
echo "  Parameters for this task:"
echo "    Model Type: $MODEL_TYPE"
echo "    Updater: $UPDATER"
echo "    Enable Recurrence: $ENABLE_RECURRENCE"
echo "    Clip Weights: $CLIP_WEIGHTS"
echo "    Plast Clip: $PLAST_CLIP"
echo "    Grad Clip: $GRAD_CLIP"
echo "    Residual Connection: $RESIDUAL_CONNECTION"
echo "    Input Mode: $INPUT_MODE"
echo "    Learning Rate: $LEARNING_RATE"
echo "    Forget Rate: $FORGET_RATE"
echo "    Plast Proportion: $PLAST_PROPORTION"
echo "  W&B Group: $GROUP"
echo "  W&B Notes/Run Name: $RUN_NOTES"
echo "  Output File: slurm_logs/bench_sweep_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
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

echo "--- Bench Sweep Task $SLURM_ARRAY_TASK_ID Finished ---"

# Optional Post-Run Cleanup can remain here
# ==============================================================================
