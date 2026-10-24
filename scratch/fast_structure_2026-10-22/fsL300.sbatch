#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus=1
#SBATCH --mem=30G
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --exclude=cs-1-5,cs-1-3
#SBATCH --job-name=fs_L300
#SBATCH --array=0-15
#SBATCH --output=/home/jaded79/fslt_2026-10-22/logs/%x_%A_%a.out
# Batch FS part L extension to 300k (finalize agent B4, 2026-10-24; PREREG_fastlayers.md "Batch FS EXTENSION").
# 0-4 A3 exclusive 3e4 | 5-9 D3 additive_dense 3e3 | 10-12 D10 additive_dense 1e4 | 13-15 M3 additive_masked 3e4. (A10, D30: diverged, not resumed.)
source ~/.bashrc >/dev/null 2>&1
conda activate hebby
t=$SLURM_ARRAY_TASK_ID
seeds=(2718 3141 1618 4241 8128)
if   [ $t -lt 5 ];  then arm=A3;  st=exclusive;       alpha=30000; seed=${seeds[$t]}
elif [ $t -lt 10 ]; then arm=D3;  st=additive_dense;  alpha=3000;  seed=${seeds[$((t-5))]}
elif [ $t -lt 13 ]; then arm=D10; st=additive_dense;  alpha=10000; seed=${seeds[$((t-10))]}
else                     arm=M3;  st=additive_masked; alpha=30000; seed=${seeds[$((t-13))]}
fi
name=L-$arm-s$seed
echo "task $t -> $name ($st alpha $alpha) to 300k"; nvidia-smi --query-gpu=name --format=csv,noheader
cd ~/fslt_2026-10-22/code
mkdir -p $HOME/.cache/inductor_fs; export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d $HOME/.cache/inductor_fs/XXXX)
bash scratch/fast_structure_2026-10-22/run_L300.sh $name $st $alpha $seed ../runs_L
