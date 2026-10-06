#!/bin/bash
# usage: GPU=0 ROOT=~/stab_margin_2026-10-06 ARMRUNS=~/stab_trace_2026-10-02/runs run_cells_st.sh cells_file
# cell line: name arm stage n_iters m pf [reseed]   (alpha = m*1e4; resume ARMRUNS/ARM/checkpoint_<stage>.pth)
set -u
source ~/miniforge3/etc/profile.d/conda.sh; conda activate hebby
CODE=$ROOT/code; RUNS=$ROOT/runs
while read -r name arm stage n_iters m pf rs; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  D=$RUNS/$name; mkdir -p "$D"
  if grep -q "^=== exit 0" "$D/train.log" 2>/dev/null; then echo "skip $name"; continue; fi
  case $arm in ST1) X="--layer_norm true --weight_clamp 1";; ST2) X="--weight_clamp 0.3";; ST3) X="--grad_norm_clip 30 --weight_clamp 1";; ST4) X="--output_tanh true --weight_clamp 1";; esac
  alpha=$(python3 -c "print($m*1e4)")
  FLAGS=(--model_type ephemeral --updater dfa --enable_recurrence false --fused_update true
    --dataset 3_palindrome_dataset_vary_length --input_mode last_one --learning_rate 1e-3
    --plasticity $alpha --forget_rate 0.01 --ephemeral_fraction 0.2 $X
    --hidden_size 1024 --num_layers 3 --residual_connection false --positional_encoding_dim 0
    --batch_size 16 --track false --deterministic true --resume true
    --n_iters $n_iters --print_freq $pf --trace_loop_every $pf --checkpoint_save_freq 5000
    --checkpoint_keep_every 0 --checkpoint_keep_max 2 --early_stop_window 0 --seed 3141)
  [[ -f $D/latest_checkpoint.pth ]] || FLAGS+=(--resume_checkpoint "$ARMRUNS/$arm/checkpoint_$(printf %08d $stage).pth")
  [[ -n "${rs:-}" ]] && FLAGS+=(--resume_reseed "$rs")
  echo "=== cell $name start $(date) host $(hostname) m=$m arm=$arm stage=$stage" >> "$D/train.log"
  echo "=== flags: ${FLAGS[*]}" >> "$D/train.log"
  (cd "$CODE" && CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled HF_DATASETS_OFFLINE=1 HF_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    TORCHINDUCTOR_CACHE_DIR=$(mktemp -d "${TMPDIR:-/tmp}/ind_stab_XXXXXX") python -u train.py "${FLAGS[@]}" --checkpoint_dir "$D" >> "$D/train.log" 2>&1)
  echo "=== exit $? $(date)" >> "$D/train.log"
done < "$1"
