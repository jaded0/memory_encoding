#!/bin/bash
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gpus=1
#SBATCH --mem=30G
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --job-name=b2_main
#SBATCH --array=0-15
#SBATCH --output=/home/jaded79/b2_2026-10-22/logs/%x_%A_%a.out
# B2 main batch on ORC: seeds 7302 7403 7504 7605 x arms NOFAST L0 DEEP DEEPS (seed 7201 runs on deckard GPU 1).
source ~/.bashrc >/dev/null 2>&1
conda activate hebby
export EPHEMERAL_DATA_DIR=$HOME/memory_encoding/processed_datasets
mkdir -p $HOME/.cache/inductor_b2 $HOME/.cache/tmp_b2
export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d $HOME/.cache/inductor_b2/XXXX); export TMPDIR=$(mktemp -d $HOME/.cache/tmp_b2/XXXX)
cd ~/b2_2026-10-22/code
source ../b2/main_common.sh
arms=(NOFAST L0 DEEP DEEPS); seeds=(7302 7403 7504 7605)
t=$SLURM_ARRAY_TASK_ID; arm=${arms[$((t % 4))]}; seed=${seeds[$((t / 4))]}
echo "task $t -> $arm-s$seed"; nvidia-smi --query-gpu=name --format=csv,noheader
run_arm $arm $seed python
