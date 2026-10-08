#!/bin/bash
# usage: train.sh SEED
source ~/miniforge3/etc/profile.d/conda.sh; conda activate hebby
export CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCHINDUCTOR_CACHE_DIR=$(mktemp -d /tmp/inductor_tr_XXXXXX)
cd ~/lowrank_deep/transition/code
S=$1
mkdir -p ~/lowrank_deep/transition/ckpt_s$S
python -u train.py --model_type ephemeral --updater dfa --enable_recurrence false --fused_update true --dataset long_range_memory_dataset --input_mode last_one --learning_rate 1e-4 --plasticity 3e4 --forget_rate 0.01 --ephemeral_fraction 0.1 --ephemeral_update_clamp 0 --weight_clamp 0 --slow_weight_decay 0 --hidden_size 1024 --num_layers 3 --residual_connection false --positional_encoding_dim 0 --batch_size 16 --n_iters 60000 --print_freq 1000 --checkpoint_dir ~/lowrank_deep/transition/ckpt_s$S --checkpoint_save_freq 2500 --track false --seed $S --deterministic true
