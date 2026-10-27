#!/usr/bin/env bash
# One explicitly approved run, then the original O2 scoring pipeline. No submission here.
set -euo pipefail
data=${1:?dataset}; n=${2:?iterations}; arm=${3:?arm}; seed=${4:?seed}; root=${5:?outroot}
: "${K13_GATE_MANIFEST:?Set path to approved JSON manifest after regime verification and log reread}"
python - "$K13_GATE_MANIFEST" "$data" "$n" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
assert m.get('regime_verified') is True and m.get('launch_approved') is True
assert m['dataset'] == sys.argv[2] and m['n_iters'] == int(sys.argv[3])
assert m.get('approval_reference') and m.get('pilot_evidence')
PY
case "$arm" in
  lin) flags=();;
  par2) flags=(--order2_layers 1,2);;
  par1) flags=(--order2_layers 1,2 --order2_map order1);;
  rep2) flags=(--order2_layers 1,2 --order2_mode replace);;
  spar) flags=(--order2_layers 1,2 --order2_map sparse --order2_sparse_m 273 --order2_sparse_k 17);;
  *) echo "unknown arm: $arm" >&2; exit 2;;
esac
d="$root/$data-$arm-s$seed"
mkdir -p "$d"
commit=$(git rev-parse HEAD)
if [[ -e "$d/code_commit" && $(cat "$d/code_commit") != "$commit" ]]; then
  echo 'Refusing mixed-commit resume' >&2; exit 2
fi
printf '%s\n' "$commit" > "$d/code_commit"
cp "$K13_GATE_MANIFEST" "$d/launch_gate.json"
if [[ ! -f "$d/final_eval.json" ]]; then
  python -u train.py --dataset "$data" --input_mode last_one --forget_rate .01 \
    --ephemeral_update_clamp 0 --weight_clamp 0 --slow_weight_decay 0 \
    --hidden_size 1024 --num_layers 3 --residual_connection false --positional_encoding_dim 0 \
    --batch_size 16 --track false --deterministic true --resume true --model_type ephemeral \
    --updater dfa --enable_recurrence false --fused_update true --learning_rate 1e-4 \
    --plasticity 3e4 --ephemeral_fraction .1 --optimizer sgd --n_iters "$n" --print_freq 500 \
    --early_stop_window 0 --heldout_eval_every 25000 --heldout_batches 16 \
    --checkpoint_save_freq 25000 --checkpoint_keep_every 50000 --seed "$seed" \
    --checkpoint_dir "$d" "${flags[@]}" >> "$d/train.log" 2>&1
  echo 'TRAIN EXIT 0' >> "$d/train.log"
  python -u heldout.py --checkpoint "$d/latest_checkpoint.pth" --protocols observed strict no_fast \
    --batches 128 --json "$d/final_eval.json" > "$d/final_eval.log" 2>&1
  echo 'EVAL EXIT 0' >> "$d/train.log"
fi
python -u scratch/order2_2026-10-22/o2_keys.py "$d/latest_checkpoint.pth" \
  --out "$d/keys" --batches 64 --temps 1,4,16 > "$d/keys.log" 2>&1
echo 'KEYS EXIT 0' >> "$d/train.log"
