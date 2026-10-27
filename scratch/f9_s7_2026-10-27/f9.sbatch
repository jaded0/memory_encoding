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
#SBATCH --job-name=f9
#SBATCH --array=0-2
#SBATCH --output=/home/jaded79/f9s7_2026-10-27/logs/%x_%A_%a.out
# Batch F9 (queue 2026-10-27; PREREG_fastlayers.md "Batch F9"): DC2 with the element-wise weight clamp off.
# V2's DC2 R-break command (v2.sbatch tasks 5-7) with --weight_clamp 0, seeds 620101 620203 620307, 600k, on main.
# A resubmission (after a timeout) resumes each run from runs/<name>/latest_checkpoint.pth.
source ~/.bashrc >/dev/null 2>&1
conda activate hebby
t=$SLURM_ARRAY_TASK_ID
seeds=(620101 620203 620307)
seed=${seeds[$t]}
echo "task $t -> Rbreak-F9-s$seed"; nvidia-smi --query-gpu=name --format=csv,noheader
cd ~/f9s7_2026-10-27/code
mkdir -p $HOME/.cache/inductor_f9s7; export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d $HOME/.cache/inductor_f9s7/XXXX)
bash scratch/f9_s7_2026-10-27/run_arm.sh F9 $seed 600000 ../runs
