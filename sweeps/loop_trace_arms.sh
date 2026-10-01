#!/bin/bash
# Collapse-versus-stabilizer arms for the feedback-loop traces (Deckard, one GPU per run).
#   sweeps/loop_trace_arms.sh ARM GPU [SEED] [N_ITERS]
# The base recipe is the stabilizer pilot's control (benchmarks/stabilizer_pilot.md): 3-char
# reversal, lr 1e-3, plasticity 1e4 (lr*alpha = 10), forget 0.01, 20% fast, weight clamp 1, three
# 1024-wide layers, batch 16, recurrence off, no LayerNorm, fused update. On ORC it collapsed at 215k
# and 330k iterations; with print_freq 500 here it blows up within the first ~5k iterations (loss
# 10+; train.py's early stop then ends it after ten intervals), which makes it cheap to trace.
# Each arm changes one setting; the default 30k iterations is enough to separate them. Loop traces every TRACE_EVERY iterations and a
# rolling window of numbered checkpoints (the newest KEEP_MAX, every KEEP_EVERY iterations) for
# trace_replay.py. Output: $ROOT/runs/ARM-sSEED/{train.log,traces/,checkpoint_*.pth}.
set -eo pipefail
ARM=${1:?arm: control tanh layer_norm clamp0.3 clip30}
GPU=${2:?gpu index}
SEED=${3:-3141}
N_ITERS=${4:-30000}
ROOT=${ROOT:-$HOME/loop_trace_2026-09-30}
PRINT_FREQ=${PRINT_FREQ:-500}
TRACE_EVERY=${TRACE_EVERY:-500}
KEEP_EVERY=${KEEP_EVERY:-1000}
KEEP_MAX=${KEEP_MAX:-30}
LR=${LR:-1e-3}

weight_clamp=1 extra=()
case $ARM in
    control) ;;
    tanh) extra=(--output_tanh true) ;;
    layer_norm) extra=(--layer_norm true) ;;
    clamp0.3) weight_clamp=0.3 ;;
    clip30) extra=(--grad_norm_clip 30) ;;
    *) echo "unknown arm $ARM"; exit 1 ;;
esac

source ~/miniforge3/etc/profile.d/conda.sh
conda activate hebby
cd "$ROOT/code"
run_dir=$ROOT/runs/$ARM-s$SEED
mkdir -p "$run_dir"
export CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled HF_DATASETS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d "${TMPDIR:-/tmp}/inductor_loop_trace_XXXXXX")
python -u train.py --model_type ephemeral --updater dfa --enable_recurrence false --fused_update true \
    --dataset 3_palindrome_dataset_vary_length --input_mode last_one --learning_rate "$LR" \
    --plasticity 1e4 --forget_rate 0.01 --ephemeral_fraction 0.2 --weight_clamp "$weight_clamp" \
    --hidden_size 1024 --num_layers 3 --residual_connection false --positional_encoding_dim 0 \
    --batch_size 16 --n_iters "$N_ITERS" --print_freq "$PRINT_FREQ" --track false \
    --seed "$SEED" --deterministic true \
    --checkpoint_dir "$run_dir" --checkpoint_save_freq 10000 \
    --checkpoint_keep_every "$KEEP_EVERY" --checkpoint_keep_max "$KEEP_MAX" \
    --trace_loop_every "$TRACE_EVERY" "${extra[@]}" > "$run_dir/train.log" 2>&1
