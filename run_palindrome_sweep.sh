#!/bin/bash
# Sweep alpha on palindrome tasks (harder memory task) with enable_recurrence=False
# Tests both easy (long_range_memory) and hard (palindrome) tasks
set -e

BATCH_SIZE=4
LOG_DIR="palindrome_baseline_logs"
mkdir -p "$LOG_DIR"

# Use 4_palindrome_dataset_vary_length (matches prior wandb runs)
DATASET="4_palindrome_dataset_vary_length"
N_ITERS=5000

# Test with gamma=0.01 (what prior runs used) at a range of alpha values
GAMMA=0.01
for ALPHA in 100 1000 10000 50000 100000 500000; do
  for SEED in 42 123 456; do
    RUN_NAME="fixed_g${GAMMA}_a${ALPHA}_s${SEED}"
    echo ""
    echo "=== $RUN_NAME ==="
    echo "$(date): Starting"
    
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
    || echo "Run $RUN_NAME failed (exploded)"
    
    echo "$(date): Finished"
  done
done

# Also test gamma=0.7 at high alpha values where instability is more likely
GAMMA=0.7
for ALPHA in 1000 10000 100000 500000; do
  for SEED in 42 123; do
    RUN_NAME="fixed_g${GAMMA}_a${ALPHA}_s${SEED}"
    echo ""
    echo "=== $RUN_NAME ==="
    echo "$(date): Starting"
    
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
    || echo "Run $RUN_NAME failed (exploded)"
    
    echo "$(date): Finished"
  done
done

echo ""
echo "=== Palindrome sweep complete ==="
ls -la "$LOG_DIR"/*.csv 2>/dev/null | wc -l
