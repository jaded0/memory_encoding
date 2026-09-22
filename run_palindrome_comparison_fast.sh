#!/bin/bash
# Lean comparison: fixed vs adaptive on palindrome
# Focus on alpha values where loss explodes for fixed
set -e

DATASET="4_palindrome_dataset_vary_length"
N_ITERS=5000
BATCH_SIZE=4
GAMMA=0.01
LOG_DIR="palindrome_comparison_logs"
mkdir -p "$LOG_DIR"

for ALPHA in 1000 10000 100000; do
  for SEED in 42 123 456; do
    for MODE in fixed adaptive; do
      RUN_NAME="${MODE}_a${ALPHA}_s${SEED}"
      # Skip if already exists
      if [ -f "$LOG_DIR/${RUN_NAME}.csv" ]; then
        LINES=$(wc -l < "$LOG_DIR/${RUN_NAME}.csv")
        if [ "$LINES" -gt 4000 ]; then
          echo "Skipping $RUN_NAME (already done, $LINES lines)"
          continue
        fi
      fi
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
