#!/bin/bash
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus=1
#SBATCH --mem=30G
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --exclude=cs-1-5,cs-1-3
#SBATCH --job-name=s7d
#SBATCH --array=0-2
#SBATCH --output=/home/jaded79/f9s7_2026-10-27/logs/%x_%A_%a.out
# Batch S7 part b (queue 2026-10-27; PREREG_fastlayers.md "Batch S7"): slow weight decay from scratch in plain at R-break.
# The plain R-break command (V2 flags, no arm flag; = battery A1 plain) plus --slow_weight_decay 3e-6 (R4's value),
# seeds 620101 620203 620307, 600k, on main. A resubmission resumes each run from runs/<name>/latest_checkpoint.pth.
source ~/.bashrc >/dev/null 2>&1
conda activate hebby
t=$SLURM_ARRAY_TASK_ID
seeds=(620101 620203 620307)
seed=${seeds[$t]}
echo "task $t -> Rbreak-S7D-s$seed"; nvidia-smi --query-gpu=name --format=csv,noheader
cd ~/f9s7_2026-10-27/code
mkdir -p $HOME/.cache/inductor_f9s7; export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d $HOME/.cache/inductor_f9s7/XXXX)
bash scratch/f9_s7_2026-10-27/run_arm.sh S7D $seed 600000 ../runs
