#!/usr/bin/env bash
# K15 one-run launcher. No submission or implicit approval. Resume at the exact same commit.
set -euo pipefail
phase=${1:?phase}; arm=${2:?arm}; seed=${3:?seed}
root=${K15_ROOT:-$HOME/k15_2026-11-05}
cd "$root/code"
scripts=$PWD/scratch/k15_2026-11-05
python "$scripts/gate.py" "$root/manifest.json"
read -r commit train_timeout score_timeout krx < <(python - "$root/manifest.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print(m['commit'], m['training_timeout_seconds'], m['scoring_timeout_seconds'], int(m['krx_supported']))
PY
)
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi A6000 || exit 2
flags=(--order2_layers 0,1,2)
case "$phase:$arm" in
  bigkey:par2-t4) ;; # Only k15_queue.py may append this, on K13 par2 GIVEUP.
  bigkey:par2-t16) flags+=(--order2_temperature 16);;
  bigkey:sum2-t4) flags+=(--order2_read_norm sum);;
  bigkey:sum2-t16) flags+=(--order2_read_norm sum --order2_temperature 16);;
  bigkey:sink2-t4) flags+=(--order2_read_norm sum --order2_sink fixed);;
  krx:rep2|krx:sum2-rep)
    [[ $krx == 1 ]] || { echo 'KRX dropped: replace does not cover every fast layer' >&2; exit 2; }
    flags+=(--order2_mode replace)
    [[ $arm == rep2 ]] || flags+=(--order2_read_norm sum)
    ;;
  *) exit 2;;
esac
case $phase in
  bigkey)
    [[ " 2718 3141 1618 1414 1732 " == *" $seed "* ]] || exit 2
    data=mqar2_16_q4; n=195000
    recipe=(--input_mode last_two --learning_rate 1e-4 --plasticity 3e4 --ephemeral_fraction .1
      --ephemeral_update_clamp 0 --weight_clamp 0 --slow_weight_decay 0 --print_freq 500
      --heldout_eval_every 5000 --heldout_batches 8)
    ;;
  krx)
    [[ " 2718 3141 1618 1414 5772 " == *" $seed "* ]] || exit 2
    data=long_range_memory_dataset; n=30000
    recipe=(--input_mode last_one --learning_rate .001 --plasticity 10000 --ephemeral_fraction .2
      --weight_clamp 1 --layer_norm false --grad_norm_clip 0 --print_freq 100 --heldout_eval_every 0)
    ;;
esac
d=$root/runs/$data-$arm-s$seed
resume=(); train_done=0; used=0
if [[ -e $d ]]; then
  [[ $(cat "$d/code_commit") == "$commit" ]] || { echo 'Mixed commit resume refused' >&2; exit 2; }
  cmp -s "$root/manifest.json" "$d/launch_gate.json" || { echo 'Changed manifest resume refused' >&2; exit 2; }
  grep -q '^EVAL EXIT 0$' "$d/train.log" 2>/dev/null && exit 0
  used=$(cat "$d/train_seconds" 2>/dev/null || echo 0)
  if grep -q '^TRAIN EXIT 0$' "$d/train.log" 2>/dev/null; then train_done=1
  elif [[ -f $d/latest_checkpoint.pth ]]; then resume=(--resume_checkpoint "$d/latest_checkpoint.pth")
  fi
  echo "RESUME $(date -Is) used_seconds=$used checkpoint=${resume[1]:-none} train_done=$train_done" | tee -a "$d/resumes.txt" >> "$d/train.log"
fi
mkdir -p "$d"
printf '%s\n' "$commit" > "$d/code_commit"
cp "$root/manifest.json" "$d/launch_gate.json"
export WANDB_MODE=offline CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$HOME/.cache/inductor_k15"
export TORCHINDUCTOR_CACHE_DIR
TORCHINDUCTOR_CACHE_DIR=$(mktemp -d "$HOME/.cache/inductor_k15/XXXXXX")
trap 'rm -rf -- "$TORCHINDUCTOR_CACHE_DIR"' EXIT
export DISK_WAIT_NOTIFY="echo \"DISKWAIT $data-$arm-s$seed \$(date -Is)\" >> '$root/status.txt'"
start=$(date +%s); left=$((train_timeout - used)); rc=124
set +e
if (( train_done )); then rc=0
elif (( left > 0 )); then
  timeout --signal=TERM --kill-after=30s "${left}s" python -u train.py --dataset "$data" \
    --forget_rate .01 --hidden_size 1024 --num_layers 3 --residual_connection false \
    --positional_encoding_dim 0 --batch_size 16 --track false --deterministic true --resume false \
    --model_type ephemeral --updater dfa --enable_recurrence false --fused_update true \
    --order2_temperature 4 --order2_dim 16 --optimizer sgd --n_iters "$n" --early_stop_window 0 \
    --checkpoint_save_freq 5000 --checkpoint_keep_every 25000 --seed "$seed" --min_free_disk_gb 15 \
    "${recipe[@]}" "${resume[@]}" --checkpoint_dir "$d" "${flags[@]}" >> "$d/train.log" 2>&1
  rc=$?
fi
set -e
elapsed=$((used + $(date +%s) - start))
printf '%s\n' "$elapsed" > "$d/train_seconds"
printf 'TRAIN EXIT %s\n' "$rc" >> "$d/train.log"
python - "$d" "$data" "$n" "$arm" "$seed" "$rc" "$elapsed" "$commit" <<'PY'
import json, pathlib, re, sys
d, data, n, arm, seed, rc, elapsed, commit = sys.argv[1:]
d = pathlib.Path(d)
record = dict(dataset=data, arm=arm, seed=int(seed), requested_iterations=int(n), train_exit=int(rc),
              training_seconds=int(elapsed), code_commit=commit,
              resumes=(d/'resumes.txt').read_text().count('\n') if (d/'resumes.txt').exists() else 0,
              status='completed' if rc == '0' else 'timeout' if rc == '124' else 'failed')
(d/'run.json').write_text(json.dumps(record, indent=2)+'\n')
curve = {}; iteration = None
for line in (d/'train.log').read_text(errors='replace').splitlines():
    m = re.search(r'Interval metrics \(ending @ iter (\d+)', line)
    if m: iteration = int(m[1])
    m = re.match(r'\s+heldout_strict/recall_acc:\s*([\d.eE+-]+)', line)
    if m and iteration and iteration % 5000 == 0: curve[iteration] = float(m[1])
(d/'learning_curve.json').write_text(json.dumps(dict(split='validation', points=[
    dict(iteration=i, strict_recall=v) for i, v in sorted(curve.items())]), indent=2)+'\n')
PY
(( rc == 0 )) || exit "$rc"
[[ -f $d/latest_checkpoint.pth ]] || exit 2
score_start=$(date +%s)
timeout --signal=TERM --kill-after=30s "${score_timeout}s" python -u heldout.py \
  --checkpoint "$d/latest_checkpoint.pth" --split test --protocols observed strict no_fast \
  --batches 128 --json "$d/final_eval.json" > "$d/final_eval.log" 2>&1
remaining() { echo $((score_timeout - $(date +%s) + score_start)); }
diagnostic_flags=()
if [[ $phase == bigkey ]]; then
  left=$(remaining); (( left > 0 )) || exit 124
  timeout --signal=TERM --kill-after=30s "${left}s" python -u scratch/k13_bigkey_2026-10-27/analyze.py \
    "$d/latest_checkpoint.pth" --datasets "$data" --batches 8 --split test --raw-records \
    --out "$d/analysis.json" > "$d/analysis.log" 2>&1
  diagnostic_flags+=(--analysis "$d/analysis.json")
fi
left=$(remaining); (( left > 0 )) || exit 124
timeout --signal=TERM --kill-after=30s "${left}s" python -u "$scripts/diagnostics.py" \
  "$d/latest_checkpoint.pth" --device cuda --batches 2 --out "$d/weight_diagnostics.json" \
  "${diagnostic_flags[@]}" > "$d/weight_diagnostics.log" 2>&1
echo 'EVAL EXIT 0' >> "$d/train.log"
