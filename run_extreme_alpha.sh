#!/bin/bash
# Find the instability boundary at extreme alpha values
set -e

DATASET="long_range_memory_small"
N_ITERS=2000
BATCH_SIZE=4
LOG_DIR="proper_baseline_logs"

for GAMMA in 0.01 0.7; do
  for ALPHA in 100000 500000 1000000 5000000; do
    RUN_NAME="fixed_g${GAMMA}_a${ALPHA}_s42"
    echo "=== $RUN_NAME ==="
    
    PYTHONHASHSEED=42 python hebby.py \
      --dataset "$DATASET" \
      --n_iters "$N_ITERS" \
      --print_freq 200 \
      --track False \
      --model_type ethereal \
      --updater dfa \
      --enable_recurrence False \
      --normalize False \
      --grad_clip 0 \
      --plast_clip "$ALPHA" \
      --forget_rate "$GAMMA" \
      --batch_size "$BATCH_SIZE" \
      --no_resume True \
      --controller_mode fixed \
      --control_log_dir "$LOG_DIR" \
      --control_run_name "$RUN_NAME" \
    || echo "Run $RUN_NAME failed (exploded)"
  done
done

echo "=== Extreme alpha sweep complete ==="
