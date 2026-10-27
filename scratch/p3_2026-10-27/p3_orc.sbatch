#!/bin/bash
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus=1
#SBATCH --mem=30G
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --exclude=cs-1-5,cs-1-3
#SBATCH --job-name=p3
#SBATCH --array=0-11
#SBATCH --output=/home/jaded79/p3_2026-10-27/logs/%x_%A_%a.out
# P3, the expansive write-gain test (queue 2026-10-27; prereg draft P3.md). Tasks 0-11: {ctl, g1, g2, g4} x seeds {2718, 3141, 1618},
# KRC1 key-recall recipe, 100k. Optional tasks 12-14: g2w64 x 3 seeds (median-window sensitivity), only if Jaden adds them (--array=12-14).
# Code: ~/p3_2026-10-27/code = a clone of branch p3-write-gain (each run records its commit in code_commit).
source ~/.bashrc >/dev/null 2>&1
conda activate hebby
t=$SLURM_ARRAY_TASK_ID
seeds=(2718 3141 1618); arms=(ctl g1 g2 g4 g2w64)
arm=${arms[$((t / 3))]}; seed=${seeds[$((t % 3))]}
echo "task $t -> $arm s$seed"; nvidia-smi --query-gpu=name --format=csv,noheader
cd ~/p3_2026-10-27/code
mkdir -p $HOME/.cache/inductor_p3; export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d $HOME/.cache/inductor_p3/XXXX)
bash scratch/p3_2026-10-27/p3_train.sh $arm $seed $HOME/p3_2026-10-27/runs 100000
