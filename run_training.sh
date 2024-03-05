#!/bin/bash

# run_training.sh
# Shell script to run the training script with custom hyperparameters.
rm model_data/*

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
UPDATE_RULE='dfa'



# Whether to normalize the weights at each update.
# Doing so seems to prevent the runaway exploding weights effect.
# true or false
NORMALIZE=false

CLIP_WEIGHTS=0

# Learning rate for the optimizer
# Lower values mean slower but more stable training, higher values mean faster but potentially unstable training.
LEARNING_RATE=0.01

# Imprint rate for Hebbian updates
# Affects the strength of imprinting in Hebbian learning. Set to 0 for no imprinting.
IMPRINT_RATE=0.0

# Stochasticity in Hebbian updates
# Controls the amount of random noise added in updates. Higher values increase randomness.
STOCHASTICITY=0.0

# Number of rewards to track for averaging
# Higher values smooth out the reward signal over more steps.
LEN_REWARD_HISTORY=10
DELTA_REWARDS=false

# Size of hidden layers in RNN
# Larger sizes create a more complex model but require more computational resources.
HIDDEN_SIZE=128

# Number of layers in RNN
NUM_LAYERS=0

# Frequency of saving and displaying model weights
# Lower values save more frequently but may slow down training.
SAVE_FREQUENCY=10000

# Number of training iterations, like 100000
N_ITERS=30000

# Frequency of printing training progress
# Lower values provide more frequent updates.
PRINT_FREQ=300

# Frequency of plotting training loss
# Lower values plot more frequently.
PLOT_FREQ=300

# true or false
TRACK=true

DATASET=jbrazzy/baby_names

# Running the training script with the specified hyperparameters
python hebby.py --learning_rate $LEARNING_RATE \
                       --imprint_rate $IMPRINT_RATE \
                       --stochasticity $STOCHASTICITY \
                       --len_reward_history $LEN_REWARD_HISTORY \
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
                       --delta_rewards $DELTA_REWARDS