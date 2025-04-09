#!/bin/bash

# ==============================================================================
# run_training.sh
#
# Shell script to run the training script `hebby.py` with specified
# hyperparameters.
# ==============================================================================

# --- Configuration ---

# Set WANDB_MODE to 'online' for real-time logging, 'offline' for local logging,
# or 'disabled' to turn off Weights & Biases tracking.
export WANDB_MODE=online

# Clean previous model data (optional, uncomment if needed)
# echo "Removing previous model data..."
# rm -f model_data/*

# --- Experiment Setup ---

# Descriptive name for the experiment group in Weights & Biases.
GROUP='PalindromeBenchmark'

# Short description/notes for this specific run in Weights & Biases.
NOTES="Benchmarking backprop on palindromes with last_two input."

# --- Core Model & Training Parameters ---

# Update Rule: Determines how model weights are adjusted.
# Options: backprop, static_plastic_candidate, dfa, damage, oja, competitive,
#          covariance, hpca, candidate, plastic_candidate.
# See hebby.py for details on each rule.
UPDATE_RULE='backprop'

# Input Mode: Determines how many previous characters are fed into the model.
# Options:
#   last_one: Uses only the most recent character.
#   last_two: Uses the two most recent characters (current + previous).
INPUT_MODE='last_two' # Choose 'last_one' or 'last_two'

# Learning Rate: Step size for weight updates (backprop or Hebby rule base rate).
# Smaller values -> slower, potentially more stable.
# Larger values -> faster, potentially unstable.
LEARNING_RATE=1e-3

# Plasticity Learning Rate: Specific learning rate for plasticity parameters
# used in 'plastic_candidate' rules. (Ignored by 'backprop').
PLAST_LEARNING_RATE=1e-10 # Relevant only for specific update rules

# Plasticity Clip: Maximum value for plasticity parameters. (Ignored by 'backprop').
PLAST_CLIP=1e3 # Relevant only for specific update rules

# Imprint Rate: Strength of imprinting in Hebbian rules. (Ignored by 'backprop').
IMPRINT_RATE=0.3 # Relevant only for specific update rules

# Forget Rate: Decay factor for weights in some Hebbian rules to prevent explosion.
# (Ignored by 'backprop').
FORGET_RATE=0.3 # Relevant only for specific update rules

# Self Gradient: Scale of a gradient-based term replacing recurrent connection.
# Experimental feature. (Likely ignored by 'backprop').
SELF_GRAD=0

# Normalize Weights: Whether to normalize weight magnitudes after updates.
# Can help prevent exploding weights. (Boolean: true/false).
NORMALIZE=false

# Clip Weights: Maximum absolute value for weights (0 means no clipping).
CLIP_WEIGHTS=0

# Gradient Clipping: Maximum norm for gradients before updates (prevents explosions).
GRAD_CLIP=1e-3

# --- Architecture Parameters ---

# Hidden Size: Number of units in the RNN's hidden state.
HIDDEN_SIZE=256

# Number of Layers: Stacked layers in the RNN.
NUM_LAYERS=3

# Residual Connection: Add skip connections in the RNN layers. (Boolean: true/false).
RESIDUAL_CONNECTION=false

# Positional Encoding Dimension: Size of the vector added to input to indicate position.
# Set to 0 to disable positional encoding.
POS_ENCODING=128

# --- Dataset & Training Loop Parameters ---

# Dataset Identifier: Name of the dataset to use.
# Examples: palindrome_dataset, roneneldan/tinystories, jbrazzy/baby_names,
#           brucewlee1/htest-palindrome, long_range_memory_dataset,
#           palindrome_dataset_vary_length, 4_resequence
DATASET='palindrome_dataset'

# Batch Size: Number of sequences processed in parallel.
BATCH_SIZE=8

# Number of Iterations: Total training steps.
N_ITERS=1000000000 # Effectively runs indefinitely until stopped

# Print Frequency: How often (in iterations) to print progress updates.
PRINT_FREQ=500

# Plot Frequency: How often (in iterations) to log detailed metrics (like step accuracy)
# to Weights & Biases and potentially print sequence examples.
PLOT_FREQ=500 # Should be <= PRINT_FREQ

# Save Frequency: How often (in iterations) to save model checkpoints (if implemented).
SAVE_FREQUENCY=10000000 # Set high if saving is not implemented/needed often

# --- Execution ---

echo "Starting training with the following settings:"
echo "  Update Rule: $UPDATE_RULE"
echo "  Input Mode: $INPUT_MODE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Hidden Size: $HIDDEN_SIZE"
echo "  Num Layers: $NUM_LAYERS"
echo "  Dataset: $DATASET"
echo "  Batch Size: $BATCH_SIZE"
echo "  Positional Encoding Dim: $POS_ENCODING"
echo "  WandB Group: $GROUP"
echo "  WandB Notes: $NOTES"
echo "---"

python hebby.py --learning_rate $LEARNING_RATE \
                --plast_learning_rate $PLAST_LEARNING_RATE \
                --plast_clip $PLAST_CLIP \
                --imprint_rate $IMPRINT_RATE \
                --forget_rate $FORGET_RATE \
                --save_frequency $SAVE_FREQUENCY \
                --hidden_size $HIDDEN_SIZE \
                --num_layers $NUM_LAYERS \
                --n_iters $N_ITERS \
                --print_freq $PRINT_FREQ \
                --plot_freq $PLOT_FREQ \
                --update_rule $UPDATE_RULE \
                --input_mode $INPUT_MODE \
                --normalize $NORMALIZE \
                --clip_weights $CLIP_WEIGHTS \
                --track $TRACK \
                --dataset $DATASET \
                --batch_size $BATCH_SIZE \
                --residual_connection $RESIDUAL_CONNECTION \
                --grad_clip $GRAD_CLIP \
                --positional_encoding_dim $POS_ENCODING \
                --self_grad $SELF_GRAD \
                --group "$GROUP" \
                --notes "$NOTES"

echo "---"
echo "Training finished or interrupted."
# ==============================================================================