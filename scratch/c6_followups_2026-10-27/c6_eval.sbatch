#!/bin/bash
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus=1
#SBATCH --mem=30G
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --exclude=cs-1-5,cs-1-3
#SBATCH --job-name=c6_eval
#SBATCH --output=/home/jaded79/c6fu_2026-10-27/logs/%x_%A_%a.out
# C6 follow-ups 2 and 6: load_eval.py on every kept 50k checkpoint of task SLURM_ARRAY_TASK_ID (after its last training chunk; submit.sh).
source ~/.bashrc >/dev/null 2>&1
conda activate hebby
cd ~/c6fu_2026-10-27/code
echo "task $SLURM_ARRAY_TASK_ID, code $(cat COMMIT)"; nvidia-smi --query-gpu=name --format=csv,noheader
mkdir -p $HOME/.cache/inductor_c6; export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d $HOME/.cache/inductor_c6/XXXX)
bash scratch/c6_followups_2026-10-27/run_c6.sh eval $SLURM_ARRAY_TASK_ID
