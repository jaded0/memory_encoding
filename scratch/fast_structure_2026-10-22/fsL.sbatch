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
#SBATCH --job-name=fs_L
#SBATCH --array=0-15
#SBATCH --output=/home/jaded79/fslt_2026-10-22/logs/%x_%A_%a.out
# Batch FS part L (Claude agent B, 2026-10-22; pre-registered in PREREG_fastlayers.md "Batch FS"): the mask load test.
# Train on mqarw_r1_52_q8 (K ~ U{1..52}), K1/bench-panel MQAR recipe, clamp-free, 150k; then load_eval.py at 50k/100k/150k.
# 0-4 A3 exclusive alpha 3e4 | 5-9 D3 additive_dense alpha 3e3 (matched, c_eff 0.3) | 10-12 A10 exclusive 1e5 | 13-15 D10 dense 1e4 (c_eff 1.0)
# (M3 additive_masked 3e4 x3 and D30 dense 3e4 x2 run on deckard GPU 0: fsL_deckard.sh.)
source ~/.bashrc >/dev/null 2>&1
conda activate hebby
t=$SLURM_ARRAY_TASK_ID
seeds=(2718 3141 1618 4241 8128)
if   [ $t -lt 5 ];  then arm=A3;  st=exclusive;      alpha=30000;  seed=${seeds[$t]}
elif [ $t -lt 10 ]; then arm=D3;  st=additive_dense; alpha=3000;   seed=${seeds[$((t-5))]}
elif [ $t -lt 13 ]; then arm=A10; st=exclusive;      alpha=100000; seed=${seeds[$((t-10))]}
else                     arm=D10; st=additive_dense; alpha=10000;  seed=${seeds[$((t-13))]}
fi
name=L-$arm-s$seed
echo "task $t -> $name ($st alpha $alpha)"; nvidia-smi --query-gpu=name --format=csv,noheader
cd ~/fslt_2026-10-22/code
mkdir -p $HOME/.cache/inductor_fs; export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d $HOME/.cache/inductor_fs/XXXX)
bash scratch/fast_structure_2026-10-22/run_L.sh $name $st $alpha $seed ../runs_L
