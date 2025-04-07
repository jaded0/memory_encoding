#!/bin/bash

# run_training.sh
# Shell script to run the training script with custom hyperparameters.
rm model_data/*


export WANDB_MODE=online

# How to update weights.
# Options are:
# damage - Damage rule: Increases weights based on product of output and input, then decreases based on squared difference (oa*ia - (oa-ia)^2).
# oja - Oja's rule: Modifies weights based on output and the difference between input and weighted output (y * (x - y * w)).
# competitive - Competitive learning rule: Similar to Oja's rule but without scaling by output in subtraction term (y * (x - w)).
# covariance - Covariance rule: Adjusts weights based on the covariance between deviations of output and input from their running averages ((y - mean(y)) * (x - mean(x))).
# hpca - Hebbian Principal Component Analysis (HPCA) rule: Updates weights based on the input and the output, subtracting the reconstructed input from all previous neurons (y_i * (x - Σ(y_j * w_j) for j=1 to i)).
# candidate - Custom reward-based update: Introduces candidate weight changes that are temporarily applied and evaluated. Permanent updates to weights are made based on a reward signal, modulating the efficacy of the changes.
# backprop - Standard backpropagation: Computes gradients using chain rule and updates weights based on gradient descent. Relies on global error propagated from the output layer.
# dfa - Direct Feedback Alignment: Updates weights based on a direct projection of the output error to each layer using fixed, random feedback connections. Enables more local and parallel weight updates compared to backpropagation.
# plastic_candidate
# static_plastic_candidate
UPDATE_RULE='backprop'

GROUP='whatever'

NOTES="benchmark w backprop again."

# A gradient-based replacement to the recurrent connection. 
# Is this metalearning?
SELF_GRAD=0 #1e-6

# Whether to normalize the weights at each update.
# Doing so seems to prevent the runaway exploding weights effect.
# true or false
NORMALIZE=false

CLIP_WEIGHTS=0

# Learning rate for the optimizer
# Lower values mean slower but more stable training, higher values mean faster but potentially unstable training.
LEARNING_RATE=1e-4
PLAST_LEARNING_RATE=1e-10
PLAST_CLIP=1e4
RESIDUAL_CONNECTION=false

# gradient clip
GRAD_CLIP=1e-3

# Imprint rate for Hebbian updates
# Affects the strength of imprinting in Hebbian learning. Set to 0 for no imprinting.
IMPRINT_RATE=0.3

# Controls the gradient growth, preventing explosion.
FORGET_RATE=0.3

# Size of hidden layers in RNN
# Larger sizes create a more complex model but require more computational resources.
HIDDEN_SIZE=256

# Number of layers in RNN
NUM_LAYERS=3

# Frequency of saving and displaying model weights
# Lower values save more frequently but may slow down training.
SAVE_FREQUENCY=10000000

# Number of training iterations, like 1000000000
N_ITERS=1000000000

# Frequency of printing training progress
# Lower values provide more frequent updates.
PRINT_FREQ=500

# Frequency of plotting training loss
# Lower values plot more frequently.
PLOT_FREQ=500

# true or false
TRACK=true

# roneneldan/tinystories
# jbrazzy/baby_names
# brucewlee1/htest-palindrome
# long_range_memory_dataset
# palindrome_dataset
# palindrome_dataset_vary_length
# 4_resequence
DATASET=palindrome_dataset_vary_length
BATCH_SIZE=32
POS_ENCODING=128

# Running the training script with the specified hyperparameters
python hebby.py --learning_rate $LEARNING_RATE \
                       --group $GROUP \
                       --imprint_rate $IMPRINT_RATE \
                       --forget_rate $FORGET_RATE \
                       --plast_learning_rate $PLAST_LEARNING_RATE \
                       --plast_clip $PLAST_CLIP \
                       --save_frequency $SAVE_FREQUENCY \
                       --hidden_size $HIDDEN_SIZE \
                       --num_layers $NUM_LAYERS \
                       --n_iters $N_ITERS \
                       --print_freq $PRINT_FREQ \
                       --plot_freq $PLOT_FREQ  \
                       --update_rule $UPDATE_RULE \
                       --normalize $NORMALIZE \
                       --clip_weights $CLIP_WEIGHTS \
                       --track $TRACK \
                       --dataset $DATASET \
                       --batch_size $BATCH_SIZE \
                       --residual_connection $RESIDUAL_CONNECTION \
                       --grad_clip $GRAD_CLIP \
                       --notes "$NOTES" \
                       --positional_encoding_dim $POS_ENCODING \
                       --self_grad $SELF_GRAD