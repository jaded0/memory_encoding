#!/bin/bash
# usage: run_arm.sh NAME HIDDEN SEED N_ITERS GPU [extra flags]
source ~/miniforge3/etc/profile.d/conda.sh; conda activate hebby
name=$1 hidden=$2 seed=$3 iters=$4 gpu=$5; shift 5
root=$HOME/lowrank_deep/margin
mkdir -p $root/runs/$name
cd $root/code
export CUDA_VISIBLE_DEVICES=$gpu WANDB_MODE=offline HF_DATASETS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d $root/runs/$name/inductor_XXXXXX)
exec python -u train.py --model_type ephemeral --updater dfa --enable_recurrence false --fused_update true \
  --dataset long_range_memory_dataset --input_mode last_one --learning_rate 1e-4 \
  --plasticity 3e4 --forget_rate 0.01 --ephemeral_fraction 0.1 --ephemeral_update_clamp 0 \
  --weight_clamp 0 --slow_weight_decay 0 --hidden_size $hidden --num_layers 3 --residual_connection false \
  --positional_encoding_dim 0 --batch_size 16 --n_iters $iters --print_freq 1000 \
  --checkpoint_dir $root/runs/$name/ckpt --checkpoint_save_freq 0 --track false \
  --resume false --seed $seed --deterministic false --alignment_log_every 5000 --log_margin true "$@"
