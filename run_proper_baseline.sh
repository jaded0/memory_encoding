#!/bin/bash
# Phase 2: Baseline alpha sweep with enable_recurrence=False on long_range_memory_small
set -e

DATASET="long_range_memory_small"
N_ITERS=5000
BATCH_SIZE=4
LOG_DIR="proper_baseline_logs"

mkdir -p "$LOG_DIR"

for GAMMA in 0.01 0.7; do
  for ALPHA in 100 500 1000 5000 10000 50000; do
    for SEED in 42 123 456; do
      RUN_NAME="fixed_g${GAMMA}_a${ALPHA}_s${SEED}"
      echo ""
      echo "=== $RUN_NAME ==="
      echo "$(date): Starting $RUN_NAME"
      
      PYTHONHASHSEED=$SEED python hebby.py \
        --dataset "$DATASET" \
        --n_iters "$N_ITERS" \
        --print_freq 500 \
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
      || echo "Run $RUN_NAME failed (possibly exploded)"
      
      echo "$(date): Finished $RUN_NAME"
    done
  done
done

echo ""
echo "=== Baseline sweep complete ==="
echo "CSV files:"
ls -la "$LOG_DIR"/*.csv 2>/dev/null | wc -l
