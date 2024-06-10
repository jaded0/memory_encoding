#!/bin/bash

# run_training.sh

rm model_data/*

UPDATE_RULE='candidate'
NORMALIZE=false
CLIP_WEIGHTS=0
LEARNING_RATE=1e-4
IMPRINT_RATE=5e-7
STOCHASTICITY=5e-12
LEN_REWARD_HISTORY=10
DELTA_REWARDS=false
HIDDEN_SIZE=256
NUM_LAYERS=5
SAVE_FREQUENCY=990001
N_ITERS=1000000
PRINT_FREQ=300
PLOT_FREQ=300
TRACK=false
DATASET=jbrazzy/baby_names
CANDECAY=0.1

python hebby.py --learning_rate $LEARNING_RATE \
--imprint_rate $IMPRINT_RATE \
--stochasticity $STOCHASTICITY \
--len_reward_history $LEN_REWARD_HISTORY \
--save_frequency $SAVE_FREQUENCY \
--hidden_size $HIDDEN_SIZE \
--num_layers $NUM_LAYERS \
--n_iters $N_ITERS \
--print_freq $PRINT_FREQ \
--plot_freq $PLOT_FREQ \
--update_rule $UPDATE_RULE \
--normalize $NORMALIZE \
--clip_weights $CLIP_WEIGHTS \
--track $TRACK \
--dataset $DATASET \
--delta_rewards $DELTA_REWARDS \
--candecay $CANDECAY