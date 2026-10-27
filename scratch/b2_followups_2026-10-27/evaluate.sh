#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$(git -C "$HERE" rev-parse --show-toplevel)"
ROOT=${B2FU_ROOT:-$HOME/b2_followups_2026-10-27}; PY=${PYTHON:-python}
arm=${1:?arm}; seed=${2:?seed}; d=$ROOT/runs/$arm-s$seed
EVAL=scratch/b2_tinystories_2026-10-22/b2_eval.py
mkdir -p "$ROOT/eval_out/"{final,horizon,curves}
COMMON=(--val_file "$ROOT/inputs/val8192.json" --length 512)
"$PY" -u "$EVAL" tiny --checkpoint "$d/latest_checkpoint.pth" "${COMMON[@]}" --names "$ROOT/inputs/names.json" \
  --stories 4096 --out "$ROOT/eval_out/final/$arm-s$seed"
"$PY" -u "$EVAL" horizon --checkpoint "$d/latest_checkpoint.pth" "${COMMON[@]}" \
  --stories 1024 --wipe_every 2,4,8,16,32,64,128,256 --out "$ROOT/eval_out/horizon/$arm-s$seed"
shopt -s nullglob
for ck in "$d"/checkpoint_*.pth; do
  it=$(basename "$ck" .pth); it=${it#checkpoint_}; it=$((10#$it))
  ((it % 2500 == 0)) || continue
  "$PY" -u "$EVAL" curve --checkpoint "$ck" "${COMMON[@]}" --stories 256 \
    --out "$ROOT/eval_out/curves/$arm-s$seed-it$it"
done
