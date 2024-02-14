#!/bin/bash

# run_training.sh
# Shell script to run the training script with custom hyperparameters.

# Learning rate for the optimizer
# Lower values mean slower but more stable training, higher values mean faster but potentially unstable training.
LEARNING_RATE=0.005

# Imprint rate for Hebbian updates
# Affects the strength of imprinting in Hebbian learning. Set to 0 for no imprinting.
IMPRINT_RATE=0.0

# Stochasticity in Hebbian updates
# Controls the amount of random noise added in updates. Higher values increase randomness.
STOCHASTICITY=0.0001

# Number of rewards to track for averaging
# Higher values smooth out the reward signal over more steps.
LEN_REWARD_HISTORY=1000

# Frequency of saving and displaying model weights
# Lower values save more frequently but may slow down training.
SAVE_FREQUENCY=500000

# Size of hidden layers in RNN
# Larger sizes create a more complex model but require more computational resources.
HIDDEN_SIZE=128

# Number of layers in RNN
NUM_LAYERS=3

# Number of training iterations
N_ITERS=10000

# Frequency of printing training progress
# Lower values provide more frequent updates.
PRINT_FREQ=100  # Example alternative: 25

# Frequency of plotting training loss
# Lower values plot more frequently.
PLOT_FREQ=10  # Example alternative: 50

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
                       --plot_freq $PLOT_FREQ
