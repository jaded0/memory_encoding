#!/bin/bash
# ==============================================================================
# run_training.sh - Runs hebby.py with specified hyperparameters.
# ==============================================================================

# --- W&B Tracking ---
export WANDB_MODE=online # online | offline | disabled

# ======================== Experiment Identification ===========================

# --- Checkpointing ---
# CHECKPOINT_DIR needs to be persistent and accessible by all requeued jobs.
# Using SLURM_JOB_NAME or a fixed experiment name can be better than SLURM_JOB_ID if you want
# the *same* checkpoint directory to be used across requeues of the *same conceptual experiment*.
# Let's assume you have a base experiment name.
EXPERIMENT_NAME="abomination_updated"
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
NOTES="trying no cycles"
TAGS=(mega cycle_test)


# RESUME_FROM is NOT set here for automatic requeue. Python script will find "latest_checkpoint.pth".
# RESUME_FROM=""
CHECKPOINT_SAVE_FREQ=100000

# ======================== Core Training Parameters ============================
# --- Training Strategy ---
UPDATE_RULE='nocycle'       # backprop | static_plastic_candidate | dfa | etc.
INPUT_MODE='last_one'        # last_one | last_two

# --- Learning Rates & Clipping ---
LEARNING_RATE=1e-4           # Base learning rate
PLAST_LEARNING_RATE=1e-10    # Plasticity LR (for specific rules)
PLAST_CLIP=1e4               # Plasticity max value (for specific rules)
GRAD_CLIP=0                  # Max gradient norm

# --- Hebbian / Plasticity Specifics (ignored by backprop) ---
IMPRINT_RATE=0.3             # Hebbian imprint strength
FORGET_RATE=0.01             # Weight decay/forgetting factor
SELF_GRAD=0                  # Experimental recurrent replacement
PLAST_PROPORTION=0.1         # Proportion of weights that are plastic in Hebbian layers  # <-- Add this line

# --- Regularization & Stability ---
NORMALIZE=false              # Normalize weights post-update (true/false)
CLIP_WEIGHTS=0               # Max absolute weight value (0=off)

# ======================== Model Architecture ==================================
HIDDEN_SIZE=256              # RNN hidden state units
NUM_LAYERS=3                 # Number of RNN layers
RESIDUAL_CONNECTION=false    # Use skip connections (true/false)
POS_ENCODING=128             # Positional encoding dimension (0=off)

# ======================== Data & Training Loop ================================
# --- Dataset ---
DATASET='2_small_palindrome_dataset_vary_length' # palindrome_dataset | roneneldan/tinystories | palindrome_dataset_vary_length | 2_resequence | long_range_memory_dataset
BATCH_SIZE=16                 # Sequences per batch

# --- Loop Control & Logging ---
N_ITERS=1000000000           # Total training steps (iterations)
PRINT_FREQ=5000                # Console print basic avg loss/acc frequency

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
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --checkpoint_save_freq $CHECKPOINT_SAVE_FREQ \
    --track true \
    --group "$GROUP" \
    --tags "${TAGS[@]}" \
    --notes "$NOTES" \
    --plast_proportion $PLAST_PROPORTION   # <-- Pass plast_proportion

echo "--- Training Finished ---"
# ==============================================================================