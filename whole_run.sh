#!/bin/bash --login
# The above line is a directive to run the script in the login shell.

#SBATCH --time=72:00:00   # walltime.  hours:minutes:seconds
#SBATCH --ntasks=10   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=8000M   # 8G memory per CPU core
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --job-name=memory_encoding
#BATCH --output ./memory_encoding.out
#SBATCH --mail-user jaden.lorenc@gmail.com

# some helpful debugging options
# set -e
# set -u

# Limit virtual memory to 30 GB (30*1024*1024 KB)
# ulimit -v $((15 * 1024 * 1024))

# nvidia-smi

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
# source /apps/miniconda3/latest/etc/profile.d/conda.sh
conda activate hebby
export WANDB_EXECUTABLE=$CONDA_PREFIX/bin/python
export WANDB_MODE=offline

export HF_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo its working
# run_training.sh
# Shell script to run the training script with custom hyperparameters.

# How to update weights.
# Options are:
# backprop - Standard backpropagation: Computes gradients using chain rule and updates weights based on gradient descent. Relies on global error propagated from the output layer.
# static_plastic_candidate
UPDATE_RULE='static_plastic_candidate'

GROUP='vary_pals'

NOTES="vary the size of the palindromes, maybe it helps with stability"

# A gradient-based replacement to the recurrent connection. 
# Is this metalearning?
SELF_GRAD=0

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
GRAD_CLIP=0

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
N_ITERS=10000000000000

# Frequency of printing training progress
# Lower values provide more frequent updates.
PRINT_FREQ=5000

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
# python synth_datasets.py
# Running the training script with the specified hyperparameters
python -u hebby.py --learning_rate $LEARNING_RATE \
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

# mprof plot --output=memory_profile.png

rm model_data/*
