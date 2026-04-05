#!/bin/bash
# Run comparison experiments: fixed vs LQR vs H-inf controllers
# Tests across multiple seeds and α₀ values

set -e

DATASET="long_range_memory_dataset"
N_ITERS=10000
FORGET_RATE=0.7
BATCH_SIZE=4
LOG_DIR="comparison_logs"
SYSID_RESULTS="sysid_results/sysid_results.json"

mkdir -p "$LOG_DIR"

# Check that system ID results exist
if [ ! -f "$SYSID_RESULTS" ]; then
    echo "ERROR: System ID results not found at $SYSID_RESULTS"
    echo "Run analyze_sysid.py first!"
    exit 1
fi

# Test at multiple α₀ values, including ones that are unstable with fixed control
for ALPHA in 1000 5000 10000; do
    for SEED in 42 123 456 789 1024; do
        for MODE in fixed lqr hinf; do
            RUN_NAME="${MODE}_alpha${ALPHA}_seed${SEED}"
            echo "=== ${RUN_NAME} ==="

            # Set random seed via PYTHONHASHSEED and torch manual seed
            PYTHONHASHSEED=$SEED python hebby.py \
                --dataset "$DATASET" \
                --n_iters "$N_ITERS" \
                --print_freq 100 \
                --track False \
                --model_type ethereal \
                --updater dfa \
                --plast_clip "$ALPHA" \
                --forget_rate "$FORGET_RATE" \
                --batch_size "$BATCH_SIZE" \
                --no_resume True \
                --controller_mode "$MODE" \
                --control_log_dir "$LOG_DIR" \
                --control_run_name "$RUN_NAME" \
                --sysid_results "$SYSID_RESULTS" \
            || echo "Run ${RUN_NAME} failed"
        done
    done
done

echo ""
echo "=== Comparison experiments complete ==="
echo "CSV files in: $LOG_DIR/"
echo "Run plot_results.py to generate comparison plots."
