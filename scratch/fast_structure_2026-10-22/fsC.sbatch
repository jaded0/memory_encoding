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
#SBATCH --job-name=fs_C
#SBATCH --array=0-29
#SBATCH --output=/home/jaded79/fslt_2026-10-22/logs/%x_%A_%a.out
# Batch FS part C (C6; Claude agent B, 2026-10-22; pre-registered in PREREG_fastlayers.md "Batch FS"): exclusive vs additive_dense on 3pal, 600k.
# R-reason (lr 2e-4): 0-4 AR exclusive a5e4 wc1 | 5-9 DR dense a1e4 wc1 fc1 | 10-14 AR0 exclusive no clamps | 15-19 DR0 dense a1e4 no clamps
# R-break (lr 1e-3): 20-24 AB exclusive a1e4 wc1 | 25-29 DB dense a2e3 wc1 fc1.   (DRx dense a5e4 x3 runs on deckard GPU 0.)
source ~/.bashrc >/dev/null 2>&1
conda activate hebby
t=$SLURM_ARRAY_TASK_ID
seeds=(620101 620203 620307 620411 620509)
seed=${seeds[$((t % 5))]}
case $((t / 5)) in
  0) arm=AR;  lr=0.0002; st=exclusive;      alpha=50000; clamps="--weight_clamp 1 --fast_weight_clamp 0" ;;
  1) arm=DR;  lr=0.0002; st=additive_dense; alpha=10000; clamps="--weight_clamp 1 --fast_weight_clamp 1" ;;
  2) arm=AR0; lr=0.0002; st=exclusive;      alpha=50000; clamps="--weight_clamp 0 --fast_weight_clamp 0" ;;
  3) arm=DR0; lr=0.0002; st=additive_dense; alpha=10000; clamps="--weight_clamp 0 --fast_weight_clamp 0" ;;
  4) arm=AB;  lr=0.001;  st=exclusive;      alpha=10000; clamps="--weight_clamp 1 --fast_weight_clamp 0" ;;
  5) arm=DB;  lr=0.001;  st=additive_dense; alpha=2000;  clamps="--weight_clamp 1 --fast_weight_clamp 1" ;;
esac
name=C-$arm-s$seed
echo "task $t -> $name ($st lr $lr alpha $alpha $clamps)"; nvidia-smi --query-gpu=name --format=csv,noheader
cd ~/fslt_2026-10-22/code
mkdir -p $HOME/.cache/inductor_fs; export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d $HOME/.cache/inductor_fs/XXXX)
bash scratch/fast_structure_2026-10-22/run_C.sh $name $st $lr $alpha $seed ../runs_C "$clamps"
