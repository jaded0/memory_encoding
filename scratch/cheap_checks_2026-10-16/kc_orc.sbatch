#!/bin/bash
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus=1
#SBATCH --mem=30G
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --exclude=cs-1-5
#SBATCH --job-name=kc_nofast
#SBATCH --output=/home/jaded79/cheap_checks_2026-10-16/logs/%x_%j.out
# Cheap check KC (PREREG 2026-10-16): KRC1 recipe, fast weights off (--plasticity 0), seed 3141. Code: main 6b8710f.
source ~/.bashrc >/dev/null 2>&1
conda activate hebby
cd ~/cheap_checks_2026-10-16/code
export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d /tmp/inductor_kc_XXXX) WANDB_MODE=offline CUBLAS_WORKSPACE_CONFIG=:4096:8
nvidia-smi --query-gpu=name --format=csv,noheader
d=~/cheap_checks_2026-10-16/runs/KRC1-nofast-s3141; mkdir -p $d
python -u train.py --dataset long_range_memory_dataset --input_mode last_one --forget_rate 0.01 --ephemeral_update_clamp 0 \
  --weight_clamp 0 --slow_weight_decay 0 --hidden_size 1024 --num_layers 3 \
  --residual_connection false --positional_encoding_dim 0 --batch_size 16 --track false --deterministic true \
  --resume true --model_type ephemeral --updater dfa --enable_recurrence false --fused_update true \
  --learning_rate 1e-4 --plasticity 0 --ephemeral_fraction 0.1 --optimizer sgd \
  --n_iters 100000 --print_freq 500 --early_stop_window 0 --heldout_eval_every 50000 --heldout_batches 8 \
  --checkpoint_save_freq 5000 --checkpoint_keep_every 10000 --seed 3141 --checkpoint_dir $d > $d/train.log 2>&1
python -u heldout.py --checkpoint $d/latest_checkpoint.pth --protocols observed strict no_fast --batches 128 --json $d/final_eval.json > $d/final_eval.log 2>&1
