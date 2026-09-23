#!/bin/bash
# ==============================================================================
# run_training.sh - Runs train.py with specified hyperparameters.
# ==============================================================================

# Resolve repo root regardless of how this script was launched.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1

# --- W&B Tracking ---
export WANDB_MODE=online # online | offline | disabled

# ======================== Experiment Identification ===========================

# --- Checkpointing ---
# CHECKPOINT_DIR needs to be persistent and accessible by all requeued jobs.
# Using SLURM_JOB_NAME or a fixed experiment name can be better than SLURM_JOB_ID if you want
# the *same* checkpoint directory to be used across requeues of the *same conceptual experiment*.
# Let's assume you have a base experiment name.
EXPERIMENT_NAME="test_backprop_fixed"
CHECKPOINT_DIR="./checkpoints/${EXPERIMENT_NAME}" # Persistent directory for this experiment

# --- Experiment Identification (W&B) ---
# It's good practice to include SLURM_JOB_ID in group/notes if you want to trace requeues in W&B
# However, a requeued job gets a NEW SLURM_JOB_ID.
# To maintain a single WandB run across preemptions, you'd need to:
# 1. Generate a unique run ID *once* (e.g., on the first submission).
# 2. Save this ID to a file in the CHECKPOINT_DIR.
# 3. On subsequent (requeued) runs, read this ID and use wandb.init(resume="allow", id=...)
# This is more advanced, for now, each requeue might start a new WandB run unless you handle this.
# For simplicity with --requeue, you might let WandB create new runs and correlate them manually by group/notes.

GROUP=$EXPERIMENT_NAME
NOTES="Testing fixed backprop with unified updates"
# TAGS=(bptt)


# RESUME_FROM is NOT set here for automatic requeue. Python script will find "latest_checkpoint.pth".
# RESUME_FROM=""
CHECKPOINT_SAVE_FREQ=100000

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
UPDATER='backprop'                # dfa | backprop | bptt
INPUT_MODE='last_one'        # last_one | last_two

# --- Learning Rates & Clipping ---
LEARNING_RATE=1e-4           # Base learning rate (lower for stability)
PLAST_CLIP=1e1               # Plasticity max value (much lower for stability)
GRAD_CLIP=1e0                  # Max gradient norm (enable clipping)

# --- Plasticity Specifics (ignored by backprop) ---
FORGET_RATE=0.01             # Weight decay/forgetting factor
SELF_GRAD=0                  # Experimental recurrent replacement
PLAST_PROPORTION=0.2         # Proportion of weights that are plastic in ephemeral layers  # <-- Add this line
ENABLE_RECURRENCE=true       # Whether to enable recurrent hidden state connections

# --- Regularization & Stability ---
NORMALIZE=false              # Normalize weights post-update (true/false)
CLIP_WEIGHTS=1e0               # Max absolute weight value

# ======================== Model Architecture ==================================
HIDDEN_SIZE=256              # RNN hidden state units (smaller for testing)
NUM_LAYERS=2                 # Number of RNN layers (fewer for testing)
RESIDUAL_CONNECTION=true     # Use skip connections (true/false)
POS_ENCODING=0             # Positional encoding dimension (0=off)

# ======================== Data & Training Loop ================================
# --- Dataset ---
DATASET='2_small_palindrome_dataset_vary_length' # palindrome_dataset | roneneldan/tinystories | palindrome_dataset_vary_length | 2_resequence | long_range_memory_dataset
BATCH_SIZE=8                 # Sequences per batch (smaller for testing)

# --- Loop Control & Logging ---
N_ITERS=50           # Total training steps (iterations)
PRINT_FREQ=10                # Console print basic avg loss/acc frequency

# ======================== Execution ===========================================
echo "--- Starting Training ---"
echo "  Group: $GROUP | Rule: $UPDATE_RULE | Input: $INPUT_MODE | LR: $LEARNING_RATE"
echo "  Dataset: $DATASET | Batch: $BATCH_SIZE | Hidden: $HIDDEN_SIZE | PosEnc: $POS_ENCODING"
echo "  Checkpoint Dir: $CHECKPOINT_DIR"
echo "  Checkpoint Save Freq: $CHECKPOINT_SAVE_FREQ"

# Create checkpoint directory if it doesn't exist
mkdir -p "$CHECKPOINT_DIR"

# Save a copy of this script for reproducibility
cp "$0" "$CHECKPOINT_DIR/run_used.sh"


python -u train.py \
    --model_type $MODEL_TYPE \
    --updater $UPDATER \
    --input_mode $INPUT_MODE \
    --learning_rate $LEARNING_RATE \
    --plasticity $PLAST_CLIP \
    --ephemeral_update_clamp $GRAD_CLIP \
    --grad_norm_clip $GRAD_CLIP \
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
    --track true \
    --group "$GROUP" \
    --tags "${TAGS[@]}" \
    --notes "$NOTES" \
    --ephemeral_fraction $PLAST_PROPORTION \
    --enable_recurrence $ENABLE_RECURRENCE

echo "--- Training Finished ---"
# ==============================================================================
