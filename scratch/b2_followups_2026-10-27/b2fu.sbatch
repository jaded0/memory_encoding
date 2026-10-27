#!/bin/bash
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gpus=a100:1
#SBATCH --mem=30G
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --exclude=cs-1-5
#SBATCH --job-name=b2fu
#SBATCH --array=0-14%3
#SBATCH --output=/home/jaded79/b2_followups_2026-10-27/logs/%x_%A_%a.out
# Submit with --export=ALL,B2FU_PART=A,B2FU_APPROVED=A (B is separately gated).
set -eo pipefail
source ~/.bashrc >/dev/null 2>&1
conda activate hebby
set -u
CODE=${B2FU_CODE:-$HOME/me_b2fu}
HERE=$CODE/scratch/b2_followups_2026-10-27
source "$HERE/arms.sh"
case "${B2FU_PART:?A or B}" in A) arms=("${ARMS_A[@]}");; B) arms=("${ARMS_B[@]}");; *) exit 2;; esac
t=${SLURM_ARRAY_TASK_ID:?}; ((t >= 0 && t < 15)) || exit 2
bash "$HERE/run.sh" "${arms[$((t % 3))]}" "${SEEDS[$((t / 3))]}"
