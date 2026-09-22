#!/bin/bash
# Compare fixed vs adaptive controller on palindrome task
# Focus on alpha values where fixed has high loss (10000-100000 range)
set -e

DATASET="4_palindrome_dataset_vary_length"
N_ITERS=10000
BATCH_SIZE=4
GAMMA=0.01
LOG_DIR="palindrome_comparison_logs"
mkdir -p "$LOG_DIR"

# Test at alpha values spanning the loss-explosion boundary
for ALPHA in 1000 5000 10000 50000 100000; do
  for SEED in 42 123 456 789 1024; do
    for MODE in fixed adaptive; do
      RUN_NAME="${MODE}_a${ALPHA}_s${SEED}"
      echo ""
      echo "=== $RUN_NAME ==="
      echo "$(date): Starting"
      
      PYTHONHASHSEED=$SEED python hebby.py \
        --dataset "$DATASET" \
        --n_iters "$N_ITERS" \
        --print_freq 1000 \
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
        --controller_mode "$MODE" \
        --control_log_dir "$LOG_DIR" \
        --control_run_name "$RUN_NAME" \
        --alpha_min 1 \
        --alpha_max 500000 \
      || echo "Run $RUN_NAME failed"
      
      echo "$(date): Finished"
    done
  done
done

echo ""
echo "=== Comparison complete ==="
ls "$LOG_DIR"/*.csv 2>/dev/null | wc -l
