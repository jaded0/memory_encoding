#!/bin/bash
#SBATCH --time=0-03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus=1
#SBATCH --mem=30G
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --exclude=cs-1-5,cs-1-3
#SBATCH --job-name=bfr_ratio
#SBATCH --output=/home/jaded79/bfr_2026-10-22/logs/%x_%j.out
# Batch BFR: bfr_ratio.py (observed held-out replay, 16 batches) on the ORC KR checkpoints (10k, 30k), and on the 3pal LC checkpoints at
# 150k and 600k for ln, absrun (this batch) and the Batch Q F7 rms runs, for comparison. Skips outputs that exist; skips missing checkpoints.
source ~/.bashrc >/dev/null 2>&1
conda activate hebby
S=~/bfr_2026-10-22/scripts; O=~/bfr_2026-10-22/ratios; mkdir -p $O
$S/ratio_all.sh ~/bfr_2026-10-22/code_kr ~/bfr_2026-10-22/runs_kr $O
cd ~/bfr_2026-10-22/code_main
for d in ~/bfr_2026-10-22/runs/BFR-LC-* ~/batchQ_2026-10-17/runs/Q-F7-RMS-s*; do
  for k in 00150000 00600000; do
    c=$d/checkpoint_$k.pth; o=$O/ratio_$(basename $d)_${k:2:3}k.json
    [ -f $c ] && [ ! -s $o ] && python $S/bfr_ratio.py --checkpoint $c --batches 16 --json $o > /dev/null 2>&1 || true
  done
done
echo RATIO JOB DONE
