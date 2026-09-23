#!/bin/bash
# ==============================================================================
# run_training.sh - Run a single train.py experiment (local or interactive).
#
#   bash run_training.sh              # normal run (W&B online)
#   SMOKE=1 bash run_training.sh      # login node: cache HF datasets + check env, then exit
#
# SMOKE=1 is for the cluster login node (internet, no GPU): it runs a few CPU
# iterations with W&B disabled into checkpoints/_smoke, which downloads and
# caches the dataset and fails fast on a broken environment. Never set it as
# the default here. Cluster runs: slurm_run.sh. Sweeps: sweeps/.
# Flag semantics and defaults: `python train.py --help`.
# ==============================================================================

cd "$(dirname "$0")" || exit 1
SMOKE=${SMOKE:-0}

# --- W&B Tracking ---
export WANDB_MODE=online # online | offline | disabled

# ======================== Experiment Identification ===========================
# The checkpoint dir is keyed by experiment name, so reruns of the same name
# share it. A fresh run (RESUME=false) overwrites latest_checkpoint.pth there;
# RESUME=true picks it up. Each resume starts a new W&B run that records
# resumed_from_checkpoint / resumed_at_iter (W&B run resumption is not supported).
EXPERIMENT_NAME="botb"
CHECKPOINT_DIR="./checkpoints/${EXPERIMENT_NAME}"
RESUME=false                 # true | false
CHECKPOINT_SAVE_FREQ=1000000 # Iterations between checkpoints

GROUP=$EXPERIMENT_NAME
NOTES="trying the best of the sweep"
TAGS=()                      # e.g. TAGS=(bptt long)

# ======================== Core Training Parameters ============================
# --- Training Strategy ---
# MODEL_TYPE: 'ephemeral' (fast weights + decay) or 'rnn' (SimpleRNN baseline).
# UPDATER: 'dfa' (Direct Feedback Alignment), 'backprop' (per-step, hidden
# detached), or 'bptt' (backprop through time). See README "Updaters" for which
# combinations train which layers; rnn + dfa changes no parameters.
MODEL_TYPE='ephemeral'       # ephemeral | rnn
UPDATER='dfa'                # dfa | backprop | bptt
INPUT_MODE='last_one'        # last_one | last_two

# --- Learning Rates & Clipping ---
LEARNING_RATE=1e-3           # Base learning rate
PLAST_CLIP=1e3               # Plasticity (learning-rate multiplier) of ephemeral weights, alpha
GRAD_CLIP=0                  # Element-wise clip on ephemeral-weight updates (0 = off)

# --- Ephemeral Weights (ignored by the rnn baseline) ---
FORGET_RATE=0.01             # Fraction of each ephemeral weight removed per step: w <- (1 - FORGET_RATE) w
PLAST_PROPORTION=0.2         # Fraction of each layer's weights that are ephemeral
SELF_GRAD=0                  # Experimental gradient-based replacement for recurrence
ENABLE_RECURRENCE=false      # Feed the hidden state back into the next step

# --- Regularization & Stability ---
NORMALIZE=false              # Rescale each layer's parameters to unit norm after each update
CLIP_WEIGHTS=1               # Clamp ephemeral weights to [-CLIP_WEIGHTS, CLIP_WEIGHTS] (0 = off)

# ======================== Model Architecture ==================================
HIDDEN_SIZE=1024             # RNN hidden state units
NUM_LAYERS=3                 # Number of RNN layers
RESIDUAL_CONNECTION=false    # Skip connections between layers
POS_ENCODING=0               # Positional encoding dimension (0 = off)

# ======================== Data & Training Loop ================================
# Names containing palindrome_dataset, long_range_memory_dataset or resequence
# load from synth_datasets/ on disk; anything else (e.g. roneneldan/tinystories)
# is downloaded from Hugging Face, so only those need a SMOKE run to cache.
DATASET='4_palindrome_dataset_vary_length'
BATCH_SIZE=16                # Sequences per batch
N_ITERS=12000000             # Total training iterations
PRINT_FREQ=5000              # Console progress frequency

# ======================== Smoke Mode ==========================================
TRACK=true
if [[ $SMOKE == 1 ]]; then
    export WANDB_MODE=disabled
    TRACK=false
    EXPERIMENT_NAME="_smoke"
    CHECKPOINT_DIR="./checkpoints/_smoke"
    RESUME=false
    N_ITERS=20
    PRINT_FREQ=10
    CHECKPOINT_SAVE_FREQ=$N_ITERS
fi

# ======================== Execution ===========================================
echo "--- Starting Training ---"
[[ $SMOKE == 1 ]] && echo "  SMOKE mode: $N_ITERS iterations, W&B disabled"
echo "  Group: $GROUP | Model: $MODEL_TYPE | Updater: $UPDATER | Input: $INPUT_MODE | LR: $LEARNING_RATE"
echo "  Dataset: $DATASET | Batch: $BATCH_SIZE | Hidden: $HIDDEN_SIZE | PosEnc: $POS_ENCODING"
echo "  Checkpoint Dir: $CHECKPOINT_DIR (resume: $RESUME, save every $CHECKPOINT_SAVE_FREQ)"

mkdir -p "$CHECKPOINT_DIR"

# Save a copy of this script for reproducibility
[[ $SMOKE == 1 ]] || cp "$0" "$CHECKPOINT_DIR/run_used.sh"

python -u train.py \
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
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --checkpoint_save_freq $CHECKPOINT_SAVE_FREQ \
    --resume $RESUME \
    --track $TRACK \
    --group "$GROUP" \
    --tags "${TAGS[@]}" \
    --notes "$NOTES" \
    --plast_proportion $PLAST_PROPORTION \
    --enable_recurrence $ENABLE_RECURRENCE

echo "--- Training Finished ---"
# ==============================================================================
