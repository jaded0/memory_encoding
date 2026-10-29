#!/usr/bin/env bash
# Bounded timing only, not a pilot/confirmation run. Does not alter datasets/code.
set -euo pipefail
code=${1:?clean code checkout}; root=${2:?new timing root}; arm=${3:?par2 or spar}
cd "$code"
[[ -z $(git status --porcelain) ]] || exit 2
case "$arm" in
  par2) flags=(--order2_layers 0,1,2);;
  spar) flags=(--order2_layers 0,1,2 --order2_map sparse --order2_sparse_m 273 --order2_sparse_k 17);;
  *) exit 2;;
esac
mkdir -p "$root"
[[ ! -e "$root/$arm.log" ]] || exit 2
export WANDB_MODE=offline CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCHINDUCTOR_CACHE_DIR="$root/inductor-$arm"
git rev-parse HEAD > "$root/$arm.commit"
start=$(date +%s)
set +e
timeout --signal=TERM --kill-after=5s 575s python -u train.py \
  --dataset mqar2_16_q4 --input_mode last_two --forget_rate .01 \
  --ephemeral_update_clamp 0 --weight_clamp 0 --slow_weight_decay 0 \
  --hidden_size 1024 --num_layers 3 --residual_connection false --positional_encoding_dim 0 \
  --batch_size 16 --track false --deterministic true --resume false --model_type ephemeral \
  --updater dfa --enable_recurrence false --fused_update true --learning_rate 1e-4 \
  --order2_temperature 4 --order2_dim 16 --plasticity 3e4 --ephemeral_fraction .1 \
  --optimizer sgd --n_iters 1000 --print_freq 100 --early_stop_window 0 \
  --heldout_eval_every 0 --checkpoint_save_freq 1000 --seed 2718 \
  --checkpoint_dir "$root/$arm" "${flags[@]}" > "$root/$arm.log" 2>&1
rc=$?
set -e
printf 'TIMING EXIT %s ELAPSED %s REQUESTED_ITERS 1000\n' "$rc" "$(( $(date +%s) - start ))" | tee "$root/$arm.status"
exit "$rc"
