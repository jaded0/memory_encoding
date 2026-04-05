#!/bin/bash
# Run system ID experiments: sweep α values spanning stable/unstable boundary
# Each run collects per-step CSV data for plant model fitting

set -e

DATASET="long_range_memory_dataset"
N_ITERS=5000
FORGET_RATE=0.7
BATCH_SIZE=4
LOG_DIR="control_logs"

mkdir -p "$LOG_DIR"

# Sweep α (plast_clip) values across the stable/unstable boundary
for ALPHA in 100 500 1000 5000 10000; do
    echo "=== Running system ID with α=${ALPHA} ==="
    python hebby.py \
        --dataset "$DATASET" \
        --n_iters "$N_ITERS" \
        --print_freq 50 \
        --track False \
        --model_type ethereal \
        --updater dfa \
        --plast_clip "$ALPHA" \
        --forget_rate "$FORGET_RATE" \
        --batch_size "$BATCH_SIZE" \
        --no_resume True \
        --controller_mode fixed \
        --control_log_dir "$LOG_DIR" \
        --control_run_name "sysid_alpha_${ALPHA}" \
    || echo "Run with α=${ALPHA} failed (possibly exploded) — expected for unstable runs"
done

echo ""
echo "=== System ID data collection complete ==="
echo "CSV files in: $LOG_DIR/"
ls -la "$LOG_DIR"/*.csv
