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
#SBATCH --job-name=fs_C2
#SBATCH --array=0-4
#SBATCH --output=/home/jaded79/fslt_2026-10-22/logs/%x_%A_%a.out
# Batch FS ROUND 2, C6b (finalize agent B4, 2026-10-24; PREREG_fastlayers.md "Batch FS ROUND 2"): DR with the fast clamp
# scaled to dense, fc = phi * 1 = 0.2. Runs on deckard GPU 0 via deckard_queue.sh (CONCURRENCY=3 ... fsC2.sbatch 0 1 2 3 4).
# 0-4 DRc additive_dense lr 2e-4 alpha 1e4, wc 1 on S, fc 0.2 on F.
source ~/.bashrc >/dev/null 2>&1
command -v conda >/dev/null || source ~/miniforge3/etc/profile.d/conda.sh
conda activate hebby
t=$SLURM_ARRAY_TASK_ID
seeds=(620101 620203 620307 620411 620509)
seed=${seeds[$t]}
arm=DRc; lr=0.0002; st=additive_dense; alpha=10000; clamps="--weight_clamp 1 --fast_weight_clamp 0.2"
name=C-$arm-s$seed
echo "task $t -> $name ($st lr $lr alpha $alpha $clamps)"; nvidia-smi --query-gpu=name --format=csv,noheader
cd ~/fslt_2026-10-22/code
mkdir -p $HOME/.cache/inductor_fs; export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d $HOME/.cache/inductor_fs/XXXX)
bash scratch/fast_structure_2026-10-22/run_C.sh $name $st $lr $alpha $seed ../runs_C "$clamps"
