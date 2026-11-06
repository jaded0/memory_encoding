#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
base=$HOME/p6_look_2026-11-05
script=$HOME/memory_encoding/scratch/p6_look_2026-11-05/measure.py
py=$HOME/miniforge3/envs/hebby/bin/python
cd "$base/code_bfr"
for arm in plain_rms plain_ln DC2_ln; do
  seeds="2718 3141 1618"
  if [[ $arm == plain_ln ]]; then seeds="2718 1414 1618"; fi
  for seed in $seeds; do
    name=kr_${arm}_s${seed}
    if [[ $arm == plain_ln ]]; then root=runs; else root=orc_ckpt; fi
    "$py" "$script" --code "$PWD" --checkpoint "$HOME/bfr_kr_2026-10-22/$root/$name/checkpoint_00030000.pth" --task kr --out "$base/results/$name.json"
  done
done
cd "$base/code_k13"
for k in 4 8 16 32; do
  "$py" "$script" --code "$PWD" --checkpoint "$base/checkpoints/K$k/latest_checkpoint.pth" --task mqar --out "$base/results/mqar_K$k.json"
done
# Retain the initially measured, prior-unhealthy seed as a flagged sensitivity only.
cd "$base/code_bfr"
"$py" "$script" --code "$PWD" --checkpoint "$HOME/bfr_kr_2026-10-22/runs/kr_plain_ln_s3141/checkpoint_00030000.pth" --task kr --out "$base/results/kr_plain_ln_s3141.json"
cd "$base/code_k13"
for k in 4 8 16 32; do
  "$py" "$HOME/memory_encoding/scratch/p6_look_2026-11-05/probe_layers.py" --code "$PWD" --checkpoint "$base/checkpoints/K$k/latest_checkpoint.pth" --out "$base/results/probe_K$k.json"
done
echo ANALYSIS_DONE
