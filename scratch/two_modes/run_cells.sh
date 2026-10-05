#!/bin/bash
# usage: CODE=... CKPT=... RUNS=... PY=... run_cells.sh cells_file
# Each cell: resume ckpt at stage with plasticity = m*1e4, same seed/data stream, early stop off, tracer on.
set -u
CELLS=$1
while read -r name ck n_iters m pf; do
  D=$RUNS/$name; mkdir -p "$D"
  if grep -q "^=== exit 0" "$D/train.log" 2>/dev/null; then echo "skip $name"; continue; fi
  alpha=$(python3 -c "print($m*1e4)")
  FLAGS=(--model_type ephemeral --updater dfa --enable_recurrence false --fused_update true
    --dataset 3_palindrome_dataset_vary_length --input_mode last_one --learning_rate 1e-3
    --plasticity $alpha --forget_rate 0.01 --ephemeral_fraction 0.2 --weight_clamp 1
    --hidden_size 1024 --num_layers 3 --residual_connection false --positional_encoding_dim 0
    --batch_size 16 --track false --deterministic true --resume true
    --n_iters $n_iters --print_freq $pf --trace_loop_every $pf --checkpoint_save_freq 5000
    --checkpoint_keep_every 0 --checkpoint_keep_max 2 --early_stop_window 0 --seed 3141)
  [[ -f $D/latest_checkpoint.pth ]] || FLAGS+=(--resume_checkpoint "$CKPT/$ck.pth")
  echo "=== cell $name start $(date) host $(hostname) m=$m ckpt=$ck" >> "$D/train.log"
  echo "=== flags: ${FLAGS[*]}" >> "$D/train.log"
  (cd "$CODE" && WANDB_MODE=disabled HF_DATASETS_OFFLINE=1 HF_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    TORCHINDUCTOR_CACHE_DIR=$(mktemp -d "${TMPDIR:-/tmp}/ind_XXXXXX") $PY -u train.py "${FLAGS[@]}" --checkpoint_dir "$D" >> "$D/train.log" 2>&1)
  echo "=== exit $? $(date)" >> "$D/train.log"
  # keep traces and the final checkpoint only
done < "$CELLS"
