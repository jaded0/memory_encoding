#!/bin/bash
# usage (on deckard): deckard_launch.sh <gpu> <name> <dataset> <n_iters> [extra train.py args...]
source ~/miniforge3/etc/profile.d/conda.sh; conda activate hebby
cd ~/bptt_debug
gpu=$1 name=$2 dataset=$3 n_iters=$4; shift 4
mkdir -p runs/logs
export CUDA_VISIBLE_DEVICES=$gpu WANDB_MODE=disabled HF_DATASETS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
nohup python -u train.py --model_type rnn --updater bptt --enable_recurrence true \
    --dataset "$dataset" --input_mode last_one --hidden_size 1024 --num_layers 3 --batch_size 16 \
    --residual_connection false --positional_encoding_dim 0 --n_iters "$n_iters" --print_freq 5000 \
    --checkpoint_save_freq 0 --checkpoint_dir "runs/ckpt_$name" --track false --seed 2718 "$@" \
    > "runs/logs/$name.log" 2>&1 &
echo "$name pid $!"
