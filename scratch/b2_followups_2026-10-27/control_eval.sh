#!/usr/bin/env bash
# Evaluation only; stage original B2 checkpoints, never retrain these controls.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$(git -C "$HERE" rev-parse --show-toplevel)"
source "$HERE/arms.sh"
[[ ${B2FU_APPROVED:-} == A ]] || { echo 'Control eval is launch gated too' >&2; exit 2; }
ROOT=${B2FU_ROOT:-$HOME/b2_followups_2026-10-27}; PY=${PYTHON:-python}
EVAL=scratch/b2_tinystories_2026-10-22/b2_eval.py
mkdir -p "$ROOT/controls/"{final,horizon,horizon4096}
(cd "$ROOT/inputs" && sha256sum --strict -c SHA256SUMS)
(cd "$ROOT/inputs/original" && sha256sum --strict -c SHA256SUMS)
git rev-parse HEAD | tee "$ROOT/controls/code_commit.txt"
git diff --exit-code; git diff --cached --exit-code
sha256sum "$ROOT/inputs/"{val8192,names}.json > "$ROOT/controls/input_sha256.txt"
: > "$ROOT/controls/checkpoint_sha256.txt"
for seed in "${SEEDS[@]}"; do
  for arm in NOFAST DEEP; do
    ck=$ROOT/inputs/original/$arm-s$seed.pth
    test -s "$ck"
    sha256sum "$ck" >> "$ROOT/controls/checkpoint_sha256.txt"
    "$PY" -u "$EVAL" tiny --checkpoint "$ck" --names "$ROOT/inputs/names.json" \
      --val_file "$ROOT/inputs/val8192.json" --stories 4096 --length 512 \
      --out "$ROOT/controls/final/$arm-s$seed"
    "$PY" -u "$EVAL" horizon --checkpoint "$ck" --val_file "$ROOT/inputs/val8192.json" \
      --stories 1024 --length 512 --wipe_every 2,4,8,16,32,64,128,256 \
      --out "$ROOT/controls/horizon/$arm-s$seed"
    if [[ $arm == DEEP ]]; then
      "$PY" -u "$EVAL" horizon --checkpoint "$ck" --val_file "$ROOT/inputs/val8192.json" \
        --stories 4096 --length 512 --wipe_every 16 --out "$ROOT/controls/horizon4096/$arm-s$seed"
    fi
  done
done
