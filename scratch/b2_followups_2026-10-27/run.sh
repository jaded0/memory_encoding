#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CODE=$(git -C "$HERE" rev-parse --show-toplevel)
source "$HERE/arms.sh"
arm=${1:?arm}; seed=${2:?seed}
arm_args "$arm"
case "$arm" in REC-*|RNN-*) part=B;; *) part=A;; esac
[[ ${B2FU_APPROVED:-} == "$part" ]] || { echo "Launch gated: need B2FU_APPROVED=$part after Jaden's yes" >&2; exit 2; }
ROOT=${B2FU_ROOT:-$HOME/b2_followups_2026-10-27}
PY=${PYTHON:-python}
export EPHEMERAL_DATA_DIR=${EPHEMERAL_DATA_DIR:-$HOME/memory_encoding/processed_datasets}
mkdir -p "$ROOT/runs/$arm-s$seed" "$ROOT/eval_out/"{final,horizon,curves} "$ROOT/logs" "$HOME/.cache/inductor_b2fu"
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$(mktemp -d "$HOME/.cache/inductor_b2fu/XXXXXX")}
export TMPDIR=${TMPDIR:-$HOME/.cache/tmp_b2fu}; mkdir -p "$TMPDIR"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-3}
cd "$CODE"
# Fail before spending GPU time if exact old-B2 inputs are not staged.
test -s "$ROOT/inputs/val8192.json"; test -s "$ROOT/inputs/names.json"
# Manifest must come from the original B2 source, not be invented from new files.
(cd "$ROOT/inputs" && sha256sum --strict -c SHA256SUMS)
git rev-parse HEAD | tee "$ROOT/runs/$arm-s$seed/code_commit.txt"
git diff --exit-code; git diff --cached --exit-code
sha256sum "$ROOT/inputs/"{val8192,names}.json > "$ROOT/runs/$arm-s$seed/input_sha256.txt"
nvidia-smi --query-gpu=name,uuid --format=csv
"$PY" -u train.py "${FLAGS[@]}" --seed "$seed" --checkpoint_dir "$ROOT/runs/$arm-s$seed" 2>&1 | tee -a "$ROOT/runs/$arm-s$seed/train.log"
bash "$HERE/evaluate.sh" "$arm" "$seed"
