#!/bin/bash --login
# two modes of collapse: alpha-kick cells. sbatch --array=0-5%4 orc_cells.sbatch
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus=1
#SBATCH --mem=30G
#SBATCH --partition=cs,cs2
#SBATCH --qos=cs
#SBATCH --exclude=cs-1-5
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --job-name=twomodes
#SBATCH --output=/home/jaded79/memory_encoding_benchmarks/two_modes_2026-10-03/logs/%x_%A_%a.out
set -eo pipefail
ROOT=/home/jaded79/memory_encoding_benchmarks/two_modes_2026-10-03
BUCKETS=(D1 D2 O2 O3 O1 O4)
b=${BUCKETS[$SLURM_ARRAY_TASK_ID]}
conda activate hebby
set -u
nvidia-smi --query-gpu=name --format=csv,noheader; hostname
export CODE=$ROOT/code CKPT=$ROOT/ckpt RUNS=$ROOT/runs PY=$CONDA_PREFIX/bin/python
bash $ROOT/run_cells.sh $ROOT/cells_$b.txt
