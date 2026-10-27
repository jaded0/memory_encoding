#!/usr/bin/env bash
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus=1
#SBATCH --mem=30G
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --exclude=cs-1-5,cs-1-3
#SBATCH --job-name=k13r2
#SBATCH --array=0-24%4
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
source "$HOME/.bashrc" >/dev/null 2>&1
conda activate hebby
: "${K13_CODE:?path to clean branch checkout}"
: "${K13_COMMIT:?expected code commit}"
: "${K13_DATASET:?selected verified dataset}"
: "${K13_ITERS:?pilot-selected iterations}"
: "${K13_GATE_MANIFEST:?approved gate JSON}"
export WANDB_MODE=offline CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$HOME/.cache/inductor_k13r2"
export TORCHINDUCTOR_CACHE_DIR
TORCHINDUCTOR_CACHE_DIR=$(mktemp -d "$HOME/.cache/inductor_k13r2/XXXXXX")
seeds=(2718 3141 1618 1414 1732); arms=(lin par2 par1 rep2 spar)
t=${SLURM_ARRAY_TASK_ID:?}
[[ $t -ge 0 && $t -lt 25 ]]
cd "$K13_CODE"
[[ $(git rev-parse HEAD) == "$K13_COMMIT" ]]
[[ -z $(git status --porcelain --untracked-files=no) ]]
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi A100 || {
  echo 'Expected ORC A100 placement; refusing different hardware' >&2; exit 2;
}
bash scratch/k13_crosstalk_2026-10-27/train.sh "$K13_DATASET" "$K13_ITERS" \
  "${arms[$((t / 5))]}" "${seeds[$((t % 5))]}" \
  "${K13_OUTROOT:-$HOME/nobackup/archive/k13_crosstalk_2026-10-27/runs}"
