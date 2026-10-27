#!/usr/bin/env bash
# Explicit pilot exception only; all confirmation calls require a verified manifest.
set -euo pipefail
if [[ ${1:-} == --pilot ]]; then export K13B_PILOT=1; shift; fi
data=${1:?dataset}; n=${2:?iterations}; arm=${3:?arm}; seed=${4:?seed}; root=${5:?outroot}
: "${K13B_CODE:?clean checkout}" "${K13B_COMMIT:?expected commit}"
cd "$K13B_CODE"
[[ $(git rev-parse HEAD) == "$K13B_COMMIT" ]]
[[ -z $(git status --porcelain) ]] || { echo 'Refusing dirty checkout' >&2; exit 2; }
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi A100 || {
  echo 'Expected A100; refusing different hardware' >&2; exit 2;
}
export WANDB_MODE=offline CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$HOME/.cache/inductor_k13b"
export TORCHINDUCTOR_CACHE_DIR
TORCHINDUCTOR_CACHE_DIR=$(mktemp -d "$HOME/.cache/inductor_k13b/XXXXXX")
pilot=${K13B_PILOT:-0}
if [[ $pilot == 1 ]]; then
  [[ $arm == lin && $seed == 2718 && $n == 200000 ]] || exit 2
  case "$data" in mqar2_4_q4|mqar2_8_q4|mqar2_16_q4|mqar2_32_q4) ;; *) exit 2;; esac
elif [[ $pilot == 0 ]]; then
  : "${K13B_GATE_MANIFEST:?verified and explicitly approved gate manifest}"
  python scratch/k13_bigkey_2026-10-27/gate.py validate "$K13B_GATE_MANIFEST" \
    --dataset "$data" --iterations "$n" --arm "$arm" --seed "$seed"
  read -r train_timeout score_timeout < <(python - "$K13B_GATE_MANIFEST" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print(m['training_timeout_seconds'], m['scoring_timeout_seconds'])
PY
)
else
  echo 'K13B_PILOT must be 0 or 1' >&2; exit 2
fi
case "$arm" in
  lin) flags=();;
  par2) flags=(--order2_layers 0,1,2);;
  par1) flags=(--order2_layers 0,1,2 --order2_map order1);;
  rep2) flags=(--order2_layers 0,1,2 --order2_mode replace);;
  spar) flags=(--order2_layers 0,1,2 --order2_map sparse --order2_sparse_m 273 --order2_sparse_k 17);;
  *) exit 2;;
esac
d="$root/$data-$arm-s$seed"
# Pilot reruns would silently spend more than the approved four allocations.
[[ ! -e "$d" ]] || { echo "Refusing existing run directory: $d" >&2; exit 2; }
mkdir -p "$d"
printf '%s\n' "$K13B_COMMIT" > "$d/code_commit"
if [[ $pilot == 0 ]]; then cp "$K13B_GATE_MANIFEST" "$d/launch_gate.json"; fi
start=$(date +%s)
prefix=()
if [[ $pilot == 1 ]]; then train_timeout=4500; fi
prefix=(timeout --signal=TERM --kill-after=10s "${train_timeout}s")
set +e
"${prefix[@]}" python -u train.py --dataset "$data" --input_mode last_two --forget_rate .01 \
  --ephemeral_update_clamp 0 --weight_clamp 0 --slow_weight_decay 0 \
  --hidden_size 1024 --num_layers 3 --residual_connection false --positional_encoding_dim 0 \
  --batch_size 16 --track false --deterministic true --resume false --model_type ephemeral \
  --updater dfa --enable_recurrence false --fused_update true --learning_rate 1e-4 \
  --order2_temperature 4 --order2_dim 16 \
  --plasticity 3e4 --ephemeral_fraction .1 --optimizer sgd --n_iters "$n" --print_freq 500 \
  --early_stop_window 0 --heldout_eval_every 5000 --heldout_batches 8 \
  --checkpoint_save_freq 5000 --checkpoint_keep_every 5000 --seed "$seed" \
  --checkpoint_dir "$d" "${flags[@]}" >> "$d/train.log" 2>&1
rc=$?
set -e
elapsed=$(( $(date +%s) - start ))
# A timeout is NEVER marked as completed.
printf 'TRAIN EXIT %s\n' "$rc" >> "$d/train.log"
python - "$d" "$data" "$n" "$arm" "$seed" "$rc" "$elapsed" "$K13B_COMMIT" <<'PY'
import importlib.util, json, pathlib, re, sys
spec = importlib.util.spec_from_file_location('gate', 'scratch/k13_bigkey_2026-10-27/gate.py')
gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)
d, data, n, arm, seed, rc, elapsed, commit = sys.argv[1:]
rc = int(rc); d = pathlib.Path(d)
run = dict(dataset=data, arm=arm, seed=int(seed), requested_iterations=int(n), train_exit=rc,
           status='completed' if rc == 0 else 'timeout' if rc == 124 else 'failed',
           training_seconds=int(elapsed), code_commit=commit, recipe=gate.RECIPE)
(d/'pilot_run.json').write_text(json.dumps(run, indent=2)+'\n')
points = []; iteration = None
for line in (d/'train.log').read_text().splitlines():
    match = re.search(r'Interval metrics \(ending @ iter (\d+)', line)
    if match: iteration = int(match[1])
    match = re.match(r'\s+heldout_strict/recall_acc:\s*([\d.eE+-]+)', line)
    if match and iteration and iteration % 5000 == 0:
        points.append(dict(iteration=iteration, strict_recall=float(match[1])))
(d/'learning_curve.json').write_text(json.dumps(dict(split='validation', points=points), indent=2)+'\n')
PY
if [[ $rc != 0 && ! ( $pilot == 1 && $rc == 124 ) ]]; then exit "$rc"; fi
[[ -f "$d/latest_checkpoint.pth" ]] || { echo 'No complete checkpoint to analyze' >&2; exit 2; }
if [[ $pilot == 1 ]]; then
  # analyze.py accepts a POSITIONAL checkpoint; 8 batches x B16 = 128 episodes.
  # Leave at least 15 allocation minutes for same-input suppression/geometry.
  timeout --signal=TERM --kill-after=10s 600s python -u scratch/k13_bigkey_2026-10-27/analyze.py "$d/latest_checkpoint.pth" \
    --datasets "$data" --batches 8 --split validation --raw-records --out "$d/analysis.json" \
    > "$d/analysis.log" 2>&1
else
  score_start=$(date +%s)
  timeout --signal=TERM --kill-after=10s "${score_timeout}s" python -u heldout.py --checkpoint "$d/latest_checkpoint.pth" --split test \
    --protocols observed strict no_fast --batches 128 --json "$d/final_eval.json" \
    > "$d/final_eval.log" 2>&1
  score_remaining=$(( score_timeout - $(date +%s) + score_start ))
  [[ $score_remaining -gt 0 ]] || { echo 'Scoring budget exhausted' >&2; exit 2; }
  timeout --signal=TERM --kill-after=10s "${score_remaining}s" python -u \
    scratch/k13_bigkey_2026-10-27/analyze.py "$d/latest_checkpoint.pth" \
    --datasets "$data" --batches 8 --split test --raw-records --out "$d/analysis.json" \
    > "$d/analysis.log" 2>&1
fi
echo 'EVAL EXIT 0' >> "$d/train.log"
