#!/bin/bash
# Test script for backprop with EtherealRNN - stable parameters

# --- Experiment Identification ---
EXPERIMENT_NAME="backprop_stable_test"
CHECKPOINT_DIR="./checkpoints/${EXPERIMENT_NAME}"
GROUP=$EXPERIMENT_NAME
NOTES="Testing stable backprop with EtherealRNN"

# --- Training Parameters ---
MODEL_TYPE='ethereal'           # ethereal | rnn
UPDATER='backprop'              # dfa | backprop | bptt
INPUT_MODE='last_one'           # last_one | last_two

# --- Learning Rates & Clipping ---
LEARNING_RATE=1e-4              # Base learning rate - was causing NaN with 1e-3
PLAST_LEARNING_RATE=1e-10       # Plasticity LR (for specific rules)
PLAST_CLIP=1e3                  # Plasticity max value - was causing NaN with 1e4
GRAD_CLIP=0                     # Max gradient norm

# --- Hebbian / Plasticity Specifics (ignored by backprop) ---
IMPRINT_RATE=0.3                # Hebbian imprint strength
FORGET_RATE=0.01                # Weight decay/forgetting factor - was causing NaN with 0.1
SELF_GRAD=0                     # Experimental recurrent replacement
PLAST_PROPORTION=0.2            # Proportion of weights that are plastic
ENABLE_RECURRENCE=true          # Whether to enable recurrent hidden state connections

# --- Regularization & Stability ---
NORMALIZE=false                 # Normalize weights post-update
CLIP_WEIGHTS=0                  # Max absolute weight value (0=off)

# --- Model Architecture ---
HIDDEN_SIZE=256                 # RNN hidden state units (smaller for testing)
NUM_LAYERS=2                    # Number of RNN layers (fewer for testing)
RESIDUAL_CONNECTION=true        # Use skip connections
POS_ENCODING=0                  # Positional encoding dimension (0=off)

# --- Data & Training Loop ---
DATASET='2_small_palindrome_dataset_vary_length'  # Small dataset for quick testing
BATCH_SIZE=4                    # Small batch size for testing
N_ITERS=1000                    # Short test run
PRINT_FREQ=100                  # Frequent logging
CHECKPOINT_SAVE_FREQ=500        # Save checkpoints

echo "--- Starting Backprop Stability Test ---"
echo "  Group: $GROUP | Model: $MODEL_TYPE | Updater: $UPDATER"
echo "  LR: $LEARNING_RATE | PLAST_CLIP: $PLAST_CLIP | FORGET_RATE: $FORGET_RATE"
echo "  Dataset: $DATASET | Batch: $BATCH_SIZE | Hidden: $HIDDEN_SIZE"
echo "  Checkpoint Dir: $CHECKPOINT_DIR"

# Create checkpoint directory if it doesn't exist
mkdir -p "$CHECKPOINT_DIR"

# Save a copy of this script for reproducibility
cp "$0" "$CHECKPOINT_DIR/run_used.sh"

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
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --checkpoint_save_freq $CHECKPOINT_SAVE_FREQ \
    --track false \
    --group "$GROUP" \
    --notes "$NOTES" \
    --plast_proportion $PLAST_PROPORTION \
    --enable_recurrence $ENABLE_RECURRENCE

echo "--- Backprop Stability Test Finished ---"
