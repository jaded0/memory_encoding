#!/bin/bash
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus=1
#SBATCH --mem=30G
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --exclude=cs-1-5,cs-1-3
#SBATCH --job-name=s12_ratio
#SBATCH --output=/home/jaded79/s12_2026-10-27/logs/%x_%j.out
# S12 follow-up 4 (prep 2026-10-27; no prediction): Batch BFR's bfr_ratio.py (trunk fast/slow read ratios, observed held-out replay,
# 16 batches) on the BFR2 round-2 final checkpoints: LC ln 1.2M (runs2/) and KR plain/DC2 ln 100k (runs2_kr/). Eval only.
# Output in $BFR/ratios_r2/ (ratio_<run>_<iter>k.json, the round-1 naming); read with
#   python scratch/bfr_2026-10-22/ratio_summary.py ~/bfr_2026-10-22/ratios_r2   (round-1 values: same on ~/bfr_2026-10-22/ratios)
BASE=${BASE:-$HOME/s12_2026-10-27}; B=${BFR:-$HOME/bfr_2026-10-22}
source ~/.bashrc >/dev/null 2>&1; source ~/miniforge3/etc/profile.d/conda.sh 2>/dev/null
conda activate hebby
O=$B/ratios_r2; mkdir -p $O
one() {  # code dir, checkpoint, output name
  [ -f "$2" ] || { echo "MISSING $2"; return; }
  [ -s $O/$3.json ] && return
  (cd $1 && python scratch/bfr_2026-10-22/bfr_ratio.py --checkpoint $2 --batches 16 --json $O/$3.json > /dev/null)
  echo "$3 EXIT $?"
}
for s in 620101 620203 620307 620411 620509; do
  one $BASE/code $B/runs2/BFR-LC-ln-s$s/checkpoint_01200000.pth ratio_BFR-LC-ln-s${s}_1200k
done
for arm in plain DC2; do for s in 2718 3141 1618 1414 5772; do
  one $BASE/code_kr $B/runs2_kr/kr_${arm}_ln_s$s/checkpoint_00100000.pth ratio_kr_${arm}_ln_s${s}_100k
done; done
echo "RATIO R2 DONE"
