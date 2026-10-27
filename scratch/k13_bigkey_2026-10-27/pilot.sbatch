#!/usr/bin/env bash
# Four approved pilots only: max 4 x 90 minutes = 6 A100-hours including analysis.
#SBATCH --time=0-01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus=1
#SBATCH --mem=30G
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --exclude=cs-1-5,cs-1-3
#SBATCH --job-name=k13b-pilot
#SBATCH --array=0-3%4
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
source "$HOME/.bashrc" >/dev/null 2>&1
conda activate hebby
: "${K13B_CODE:?clean checkout}" "${K13B_COMMIT:?expected commit}"
ks=(4 8 16 32)
t=${SLURM_ARRAY_TASK_ID:?}
[[ $t =~ ^[0-3]$ ]]
cd "$K13B_CODE"
bash scratch/k13_bigkey_2026-10-27/train.sh --pilot "mqar2_${ks[$t]}_q4" 200000 lin 2718 \
  "${K13B_OUTROOT:-$HOME/nobackup/archive/k13_bigkey_2026-10-27/pilot}"
