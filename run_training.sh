#!/bin/bash
# ==============================================================================
# run_training.sh - Run a single train.py experiment (local or interactive).
#
#   bash run_training.sh              # normal run (W&B online)
#
# Cluster runs: slurm_run.sh (first-time setup: setup_cluster/). Sweeps: sweeps/.
# Flag semantics and defaults: `python train.py --help`.
# ==============================================================================

cd "$(dirname "$0")" || exit 1

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
# combinations train which layers; rnn + dfa is the ephemeral model's DFA
# without ephemeral weights (every layer, i2h included, learns every step).
MODEL_TYPE='ephemeral'       # ephemeral | rnn
UPDATER='dfa'                # dfa | backprop | bptt
INPUT_MODE='last_one'        # last_one | last_two

# --- Learning Rates & Clipping ---
LEARNING_RATE=1e-3           # Base learning rate
PLAST_CLIP=1e3               # Plasticity (learning-rate multiplier) of ephemeral weights, alpha
GRAD_CLIP=0                  # ephemeral: element-wise clamp on ephemeral-weight updates; rnn: grad-norm clip, also under dfa (0 = off)
GRAD_CLIP_FLAG=$([[ $MODEL_TYPE == rnn ]] && echo --grad_norm_clip || echo --ephemeral_update_clamp)

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
# load from synth_datasets/ on disk; Hugging Face ones (e.g. roneneldan/tinystories)
# must be prepared once first: python preprocess.py <name> (cluster: setup_cluster/).
DATASET='4_palindrome_dataset_vary_length'
BATCH_SIZE=16                # Sequences per batch
N_ITERS=12000000             # Total training iterations
PRINT_FREQ=5000              # Console progress frequency

# ======================== Execution ===========================================
echo "--- Starting Training ---"
echo "  Group: $GROUP | Model: $MODEL_TYPE | Updater: $UPDATER | Input: $INPUT_MODE | LR: $LEARNING_RATE"
echo "  Dataset: $DATASET | Batch: $BATCH_SIZE | Hidden: $HIDDEN_SIZE | PosEnc: $POS_ENCODING"
echo "  Checkpoint Dir: $CHECKPOINT_DIR (resume: $RESUME, save every $CHECKPOINT_SAVE_FREQ)"

mkdir -p "$CHECKPOINT_DIR"

# Save a copy of this script for reproducibility
cp "$0" "$CHECKPOINT_DIR/run_used.sh"

python -u train.py \
    --model_type $MODEL_TYPE \
    --updater $UPDATER \
    --input_mode $INPUT_MODE \
    --learning_rate $LEARNING_RATE \
    --plasticity $PLAST_CLIP \
    $GRAD_CLIP_FLAG $GRAD_CLIP \
    --forget_rate $FORGET_RATE \
    --self_grad $SELF_GRAD \
    --unit_norm_weights $NORMALIZE \
    --weight_clamp $CLIP_WEIGHTS \
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
    --track true \
    --group "$GROUP" \
    --tags "${TAGS[@]}" \
    --notes "$NOTES" \
    --ephemeral_fraction $PLAST_PROPORTION \
    --enable_recurrence $ENABLE_RECURRENCE

echo "--- Training Finished ---"
# ==============================================================================
