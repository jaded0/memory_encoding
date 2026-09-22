#!/bin/bash
# 4-way controller comparison on palindrome task: fixed, lqr, hinf, adaptive
#
# Three phases:
#   1. Collect sysid data (fixed-alpha runs on palindrome)
#   2. Fit plant model + design LQR/Hinf controllers
#   3. Run all four controllers across alpha values
set -e

# Activate the hebby conda environment
eval "$(conda shell.bash hook)"
conda activate hebby

DATASET="4_palindrome_dataset_vary_length"
GAMMA=0.01
BATCH_SIZE=4

SYSID_LOG_DIR="palindrome_sysid_logs"
SYSID_RESULTS_DIR="palindrome_sysid_results"
COMPARISON_LOG_DIR="palindrome_4way_logs"

N_SYSID_ITERS=5000
N_COMPARISON_ITERS=5000

# ============================================================
# Phase 1: System ID data collection
# ============================================================
echo "=== Phase 1: Collecting sysid data ==="
mkdir -p "$SYSID_LOG_DIR"

for ALPHA in 1000 5000 10000; do
  RUN_NAME="sysid_alpha_${ALPHA}"
  if [ -f "$SYSID_LOG_DIR/${RUN_NAME}.csv" ]; then
    LINES=$(wc -l < "$SYSID_LOG_DIR/${RUN_NAME}.csv")
    if [ "$LINES" -gt 4000 ]; then
      echo "Skipping sysid α=$ALPHA (already done, $LINES lines)"
      continue
    fi
  fi
  echo ""
  echo "--- Sysid α=$ALPHA ---"
  python hebby.py \
    --dataset "$DATASET" \
    --n_iters "$N_SYSID_ITERS" \
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
    --control_log_dir "$SYSID_LOG_DIR" \
    --control_run_name "$RUN_NAME" \
  || echo "Sysid α=$ALPHA failed (possibly exploded) — expected for unstable runs"
done

echo ""
echo "=== Sysid data collected ==="
ls "$SYSID_LOG_DIR"/*.csv 2>/dev/null

# ============================================================
# Phase 2: Fit plant model + design controllers
# ============================================================
echo ""
echo "=== Phase 2: Fitting plant model ==="

python analyze_sysid.py \
  --log_dir "$SYSID_LOG_DIR" \
  --alpha0 1000 \
  --gamma0 "$GAMMA" \
  --output_dir "$SYSID_RESULTS_DIR" \
  --Q 1.0 \
  --R 0.001 \
  --alpha_min 1 \
  --alpha_max 500000

echo ""
echo "Sysid results:"
cat "$SYSID_RESULTS_DIR/sysid_results.json"

# ============================================================
# Phase 3: 4-way comparison
# ============================================================
echo ""
echo "=== Phase 3: Running 4-way comparison ==="
mkdir -p "$COMPARISON_LOG_DIR"

SYSID_JSON="$SYSID_RESULTS_DIR/sysid_results.json"

for ALPHA in 1000 10000 100000; do
  for SEED in 42 123 456; do
    for MODE in fixed lqr hinf adaptive; do
      RUN_NAME="${MODE}_a${ALPHA}_s${SEED}"
      # Skip if already done
      if [ -f "$COMPARISON_LOG_DIR/${RUN_NAME}.csv" ]; then
        LINES=$(wc -l < "$COMPARISON_LOG_DIR/${RUN_NAME}.csv")
        if [ "$LINES" -gt 4000 ]; then
          echo "Skipping $RUN_NAME (already done, $LINES lines)"
          continue
        fi
      fi
      echo ""
      echo "=== $RUN_NAME ==="
      echo "$(date): Starting"

      # LQR and Hinf need sysid results
      SYSID_ARG=""
      if [ "$MODE" = "lqr" ] || [ "$MODE" = "hinf" ]; then
        SYSID_ARG="--sysid_results $SYSID_JSON"
      fi

      PYTHONHASHSEED=$SEED python hebby.py \
        --dataset "$DATASET" \
        --n_iters "$N_COMPARISON_ITERS" \
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
        --control_log_dir "$COMPARISON_LOG_DIR" \
        --control_run_name "$RUN_NAME" \
        --alpha_min 1 \
        --alpha_max 500000 \
        $SYSID_ARG \
      || echo "Run $RUN_NAME failed"

      echo "$(date): Finished"
    done
  done
done

echo ""
echo "=== 4-way comparison complete ==="
ls "$COMPARISON_LOG_DIR"/*.csv 2>/dev/null | wc -l
echo "files in $COMPARISON_LOG_DIR/"
