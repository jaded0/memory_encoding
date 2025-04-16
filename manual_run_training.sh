#!/bin/bash
# ==============================================================================
# run_training.sh - Runs hebby.py with specified hyperparameters.
# ==============================================================================

# --- W&B Tracking ---
export WANDB_MODE=online # online | offline | disabled

# --- Experiment Identification (W&B) ---
GROUP='finding_bug'
NOTES="back to og config"

# ======================== Core Training Parameters ============================
# --- Training Strategy ---
UPDATE_RULE='static_plastic_candidate'       # backprop | static_plastic_candidate | dfa | etc.
INPUT_MODE='last_two'        # last_one | last_two

# --- Learning Rates & Clipping ---
LEARNING_RATE=1e-4           # Base learning rate
PLAST_LEARNING_RATE=1e-10    # Plasticity LR (for specific rules)
PLAST_CLIP=1e4               # Plasticity max value (for specific rules)
GRAD_CLIP=0                  # Max gradient norm

# --- Hebbian / Plasticity Specifics (ignored by backprop) ---
IMPRINT_RATE=0.3             # Hebbian imprint strength
FORGET_RATE=0.3              # Weight decay/forgetting factor
SELF_GRAD=0                  # Experimental recurrent replacement

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
DATASET='palindrome_dataset_vary_length' # palindrome_dataset | roneneldan/tinystories | palindrome_dataset_vary_length | 2_resequence
BATCH_SIZE=32                 # Sequences per batch

# --- Loop Control & Logging ---
N_ITERS=1000000000           # Total training steps (iterations)
PRINT_FREQ=2500                # Console print basic avg loss/acc frequency
PLOT_FREQ=2500                # WandB log freq + Detailed console print freq
SAVE_FREQUENCY=10000000      # Save model frequency (iters, if implemented)

# ======================== Execution ===========================================
echo "--- Starting Training ---"
echo "  Group: $GROUP | Rule: $UPDATE_RULE | Input: $INPUT_MODE | LR: $LEARNING_RATE"
echo "  Dataset: $DATASET | Batch: $BATCH_SIZE | Hidden: $HIDDEN_SIZE | PosEnc: $POS_ENCODING"

python hebby.py \
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
# ==============================================================================