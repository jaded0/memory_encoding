#!/bin/bash
# Readout-only slow-lr test from B_late-s3141 150k (drift theory checks §5 follow-up 1).
# run.sh NAME GPU   -- NAME in CTRL RO3 TR3. Code: immutable snapshot of f03ea4b in $O/code.
O=$HOME/readout_lr_2026-10-07
CKPT=$HOME/overnight_2026-10-01/runs/B_late-s3141/checkpoint_00150000.pth
NAME=$1; GPU=$2
RECIPE=(--model_type ephemeral --updater dfa --enable_recurrence false --fused_update true
  --dataset 3_palindrome_dataset_vary_length --input_mode last_one --learning_rate 1e-3
  --plasticity 1e4 --forget_rate 0.01 --ephemeral_fraction 0.2 --weight_clamp 1
  --hidden_size 1024 --num_layers 3 --residual_connection false --positional_encoding_dim 0
  --batch_size 16 --track false --deterministic true --seed 3141 --resume true
  --print_freq 500 --trace_loop_every 500 --checkpoint_save_freq 10000
  --checkpoint_keep_every 50000 --checkpoint_keep_max 0 --early_stop_window 0)
case $NAME in
  CTRL) EXTRA=(--n_iters 210000) ;;
  RO3) EXTRA=(--n_iters 450000 --readout_slow_lr_scale 0.3) ;;
  TR3) EXTRA=(--n_iters 450000 --trunk_slow_lr_scale 0.3) ;;
  *) echo "unknown job $NAME"; exit 2 ;;
esac
D=$O/runs/$NAME; mkdir -p "$D"
[ -f "$D/latest_checkpoint.pth" ] || EXTRA+=(--resume_checkpoint "$CKPT")
source ~/miniforge3/etc/profile.d/conda.sh; conda activate hebby
cd "$O/code"
export CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled HF_DATASETS_OFFLINE=1 HF_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d "${TMPDIR:-/tmp}/inductor_rolr_XXXXXX")
echo "=== $NAME start $(date) gpu $GPU host $(hostname)" >> "$D/train.log"
echo "=== flags: ${RECIPE[*]} ${EXTRA[*]}" >> "$D/train.log"
python -u train.py "${RECIPE[@]}" "${EXTRA[@]}" --checkpoint_dir "$D" >> "$D/train.log" 2>&1
rc=$?; rm -rf "$TORCHINDUCTOR_CACHE_DIR"
echo "=== exit $rc $(date)" >> "$D/train.log"; exit $rc
