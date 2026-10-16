#!/bin/bash
# KC: key-recall KRC1 recipe with fast weights off (--plasticity 0: masks kept, fast entries never written).
# usage: kc_train.sh SEED GPU [N_ITERS] [FUSED]   (run from the repo root)
source ~/miniforge3/etc/profile.d/conda.sh; conda activate hebby
export CUDA_VISIBLE_DEVICES=$2 WANDB_MODE=offline CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCHINDUCTOR_CACHE_DIR=$(mktemp -d /tmp/inductor_kc_XXXX)
seed=$1; n=${3:-100000}; fused=${4:-true}
d=${KC_OUT:-$HOME/cheap_checks_2026-10-16/runs}/KRC1-nofast-s$seed; mkdir -p $d
python -u train.py --dataset long_range_memory_dataset --input_mode last_one --forget_rate 0.01 --ephemeral_update_clamp 0 \
  --weight_clamp 0 --slow_weight_decay 0 --hidden_size 1024 --num_layers 3 \
  --residual_connection false --positional_encoding_dim 0 --batch_size 16 --track false --deterministic true \
  --resume true --model_type ephemeral --updater dfa --enable_recurrence false --fused_update $fused \
  --learning_rate 1e-4 --plasticity 0 --ephemeral_fraction 0.1 --optimizer sgd \
  --n_iters $n --print_freq 500 --early_stop_window 0 --heldout_eval_every 50000 --heldout_batches 8 \
  --checkpoint_save_freq 5000 --checkpoint_keep_every 10000 --seed $seed --checkpoint_dir $d > $d/train.log 2>&1
python -u heldout.py --checkpoint $d/latest_checkpoint.pth --protocols observed strict no_fast --batches 128 --json $d/final_eval.json > $d/final_eval.log 2>&1
