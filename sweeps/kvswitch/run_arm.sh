#!/bin/bash
# One kvswitch arm on one GPU, then stream_eval.py on every kept checkpoint.
# usage: run_arm.sh NAME DATASET SEED FORGET PLASTICITY ITERS [WIPE_EVERY [FAST_CLAMP]]
#   WIPE_EVERY defaults to the stream length (1024): fast weights carry through each stream.
#   FAST_CLAMP defaults to 1 (--fast_weight_clamp; 0 = off, the clamp-erasure control).
# Env: ROOT (default ~/kvswitch_2026-10-02), CODE (default $ROOT/code), GPU (default 0).
# Recipe: the wipe_forget key-recall recipe (hidden 256, 3 layers, DFA, recurrence off, lr 1e-4,
# 10% fast, fast clamp 1, which made carry-over trainable there), early stop off (the early
# blowup is a transient).
set -euo pipefail
NAME=$1 DATASET=$2 SEED=$3 F=$4 ALPHA=$5 ITERS=$6 WIPE=${7:-1024} CLAMP=${8:-1}
ROOT=${ROOT:-$HOME/kvswitch_2026-10-02}
CODE=${CODE:-$ROOT/code}
KEEP=${KEEP_EVERY:-50000}
source ~/miniforge3/etc/profile.d/conda.sh
conda activate hebby
export CUDA_VISIBLE_DEVICES=${GPU:-0} WANDB_MODE=offline HF_DATASETS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d /tmp/inductor_kvswitch_XXXXXX)
RUN=$ROOT/runs/$NAME
mkdir -p "$RUN" "$ROOT/logs" "$ROOT/evals"
cd "$CODE"
python -u train.py --model_type ephemeral --updater dfa --enable_recurrence false --fused_update true \
    --dataset "$DATASET" --input_mode last_one --learning_rate 1e-4 --plasticity "$ALPHA" \
    --forget_rate "$F" --ephemeral_fraction 0.1 --ephemeral_update_clamp 0 --weight_clamp 0 \
    --fast_weight_clamp "$CLAMP" --slow_weight_decay 0 --hidden_size 256 --num_layers 3 \
    --residual_connection false --positional_encoding_dim 0 --batch_size 16 --n_iters "$ITERS" \
    --print_freq 2000 --checkpoint_dir "$RUN" --checkpoint_save_freq 10000 \
    --checkpoint_keep_every "$KEEP" --early_stop_window 0 --track false --seed "$SEED" \
    --deterministic true --wipe_every "$WIPE" --heldout_eval_every 50000 --heldout_batches 4 \
    > "$ROOT/logs/$NAME.log" 2>&1
for ckpt in "$RUN"/checkpoint_*.pth "$RUN"/latest_checkpoint.pth; do
    [[ -f $ckpt ]] || continue
    tag=$(basename "$ckpt" .pth)
    python -u stream_eval.py --checkpoint "$ckpt" --split validation \
        --json "$ROOT/evals/${NAME}__${tag}.json" > "$ROOT/evals/${NAME}__${tag}.txt" 2>&1
done
rm -rf "$TORCHINDUCTOR_CACHE_DIR"
echo "done $NAME"
