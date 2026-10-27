#!/usr/bin/env bash
# Frozen-weight screen only. At most 580 seconds per point; no training or submissions.
set -euo pipefail
ck=${1:?round-1 lin checkpoint}; out=${2:?output directory}
py=${PYTHON:-$HOME/miniforge3/envs/hebby/bin/python}
gpu=${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2n | head -1 | cut -d, -f1)}
export CUDA_VISIBLE_DEVICES=$gpu
mkdir -p "$out"
for d in 0 4 8 16 32; do
  timeout 580 "$py" scratch/k13_crosstalk_2026-10-27/pilot.py "$ck" \
    --mode load --load "$d" --out "$out/load$d.json" > "$out/load$d.log" 2>&1
done
for d in 4 8 16 32; do
  timeout 580 "$py" scratch/k13_crosstalk_2026-10-27/pilot.py "$ck" \
    --mode gap --load "$d" --out "$out/gap$d.json" > "$out/gap$d.log" 2>&1
  timeout 580 "$py" scratch/k13_crosstalk_2026-10-27/pilot.py "$ck" \
    --mode load --load "$d" --selective-writes --out "$out/selective$d.json" > "$out/selective$d.log" 2>&1
done
"$py" scratch/k13_crosstalk_2026-10-27/screen.py "$out"
