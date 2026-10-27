#!/bin/bash
# Direct evaluation only on the ORC login GPU, per S12 launch approval.
# Ratios first; then both null-eval tasks. A nonzero exit creates FAILED, never DONE.
source ~/miniforge3/etc/profile.d/conda.sh
conda activate hebby
set -eo pipefail
BASE=${BASE:-$HOME/s12_2026-10-27}
export BASE CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
S=$BASE/code/scratch/s12_2026-10-27
trap 'touch "$BASE/short_FAILED"' ERR
bash "$S/s12_ratio_r2.sbatch"
python "$BASE/code/scratch/bfr_2026-10-22/ratio_summary.py" "$HOME/bfr_2026-10-22/ratios_r2" > "$BASE/ratios_r2_summary.txt"
touch "$BASE/ratios_DONE"
SLURM_ARRAY_TASK_ID=0 bash "$S/s12_null.sbatch"
SLURM_ARRAY_TASK_ID=1 bash "$S/s12_null.sbatch"
python "$S/s12_score.py" null "$BASE/null" > "$BASE/s12_null.jsonl"
touch "$BASE/short_DONE"
