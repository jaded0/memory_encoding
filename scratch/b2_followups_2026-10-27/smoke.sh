#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$(git -C "$HERE" rev-parse --show-toplevel)"
source "$HERE/arms.sh"
ROOT=${B2FU_SMOKE_ROOT:?outside-repo smoke output directory}
PY=${PYTHON:-$HOME/miniforge3/envs/hebby/bin/python}
export B2FU_SMOKE_TRAIN=$ROOT/train.json
export OMP_NUM_THREADS=3
export TORCHINDUCTOR_CACHE_DIR=$HOME/.cache/inductor_b2fu_smoke
mkdir -p "$ROOT/logs" "$ROOT/runs" "$ROOT/eval" "$TORCHINDUCTOR_CACHE_DIR"
# Each arm (training AND evaluation) has an independent 600s hard wall limit.
if [[ ${1:-} != --one ]]; then
  nvidia-smi
  for arm in "${ARMS_A[@]}" "${ARMS_B[@]}"; do
    timeout --signal=TERM --kill-after=10s 590s bash "$0" --one "$arm" > "$ROOT/logs/$arm.log" 2>&1
  done
  exit
fi
arm=$2; arm_args "$arm"
git rev-parse HEAD
date -Is
"$PY" -u "$HERE/smoke_train.py" "${FLAGS[@]}" --seed 7901 --n_iters "${B2FU_SMOKE_ITERS:-100}" \
  --print_freq 10 --resume false --checkpoint_save_freq 100 --checkpoint_keep_every 100 \
  --checkpoint_dir "$ROOT/runs/$arm"
EVAL=scratch/b2_tinystories_2026-10-22/b2_eval.py
"$PY" -u "$EVAL" tiny --checkpoint "$ROOT/runs/$arm/latest_checkpoint.pth" \
  --val_file "$ROOT/val.json" --stories 16 --length 64 --out "$ROOT/eval/$arm"
"$PY" -u "$EVAL" horizon --checkpoint "$ROOT/runs/$arm/latest_checkpoint.pth" \
  --val_file "$ROOT/val.json" --stories 16 --length 64 --wipe_every 1,16,64 --out "$ROOT/eval/$arm-horizon"
echo "SMOKE_OK $arm"; date -Is
