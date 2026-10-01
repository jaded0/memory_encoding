#!/bin/bash
# usage: run_arm.sh NAME STRUCTURE ALPHA SEED N_ITERS [extra flags]
source ~/miniforge3/etc/profile.d/conda.sh; conda activate hebby
name=$1 structure=$2 alpha=$3 seed=$4 iters=$5; shift 5
root=$HOME/lowrank_deep/three_arm
mkdir -p $root/runs/$name
cd $root/code
export CUDA_VISIBLE_DEVICES=1 WANDB_MODE=offline HF_DATASETS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d $root/runs/$name/inductor_XXXXXX)
exec python -u train.py --model_type ephemeral --updater dfa --enable_recurrence false \
  --dataset long_range_memory_dataset --input_mode last_one --learning_rate 1e-4 \
  --plasticity $alpha --forget_rate 0.01 --ephemeral_fraction 0.1 --ephemeral_update_clamp 0 \
  --weight_clamp 0 --slow_weight_decay 0 --hidden_size 256 --num_layers 3 --residual_connection false \
  --positional_encoding_dim 0 --batch_size 16 --n_iters $iters --print_freq 1000 \
  --checkpoint_dir $root/runs/$name/ckpt --checkpoint_save_freq 1000 --track false \
  --resume true --seed $seed --deterministic false --fast_structure $structure "$@"
