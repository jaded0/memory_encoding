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
#SBATCH --job-name=s12_null
#SBATCH --array=0-1
#SBATCH --output=/home/jaded79/s12_2026-10-27/logs/%x_%A_%a.out
# S12 follow-up 2 (prep 2026-10-27; PREREG "Batch S12F", N1-N4): ln_null_eval.py (strict, no_fast, swap, swap_raw, noise; 128 held-out
# batches) on the existing Batch BFR / BFR2 ln checkpoints on ORC. Eval only, no training. Skips outputs that exist; skips missing checkpoints.
# 0 = KR (code_kr, RN's KR data): plain and DC2 ln x 5 seeds at 30k (round 1, runs_kr/), 60k and 100k (round 2, runs2_kr/),
#     and DC2 at 10k (plain's 10k/20k checkpoints stayed on manticore: ~/bfr_kr_2026-10-22/runs/, see run_null_manticore.sh).
# 1 = LC (code, 3pal data): BFR-LC-ln x 5 seeds at 150k, 600k (runs/) and 1.2M (runs2/).
# Output: $BASE/null/null_<run>_<iter>k.json; collect with `python s12_score.py null $BASE/null > s12_null.jsonl`.
BASE=${BASE:-$HOME/s12_2026-10-27}; B=${BFR:-$HOME/bfr_2026-10-22}
source ~/.bashrc >/dev/null 2>&1; source ~/miniforge3/etc/profile.d/conda.sh 2>/dev/null
conda activate hebby
set -eo pipefail
t=${SLURM_ARRAY_TASK_ID:-0}
O=$BASE/null; mkdir -p $O
nvidia-smi --query-gpu=name --format=csv,noheader
one() {  # checkpoint, output name
  [ -f "$1" ] || { echo "MISSING $1"; return 1; }
  [ -s $O/$2.json ] && { echo "skip $2"; return; }
  python -u scratch/s12_2026-10-27/ln_null_eval.py --checkpoint $1 --batches ${BATCHES:-128} --json $O/$2.json > $O/$2.log 2>&1
  echo "$2 EXIT $?"
}
if [ $t -eq 0 ]; then
  cd $BASE/code_kr || exit 1
  for arm in plain DC2; do for s in 2718 3141 1618 1414 5772; do
    n=kr_${arm}_ln_s$s
    [ $arm = DC2 ] && one $B/runs_kr/$n/checkpoint_00010000.pth null_${n}_10k
    one $B/runs_kr/$n/checkpoint_00030000.pth null_${n}_30k
    one $B/runs2_kr/$n/checkpoint_00060000.pth null_${n}_60k
    one $B/runs2_kr/$n/checkpoint_00100000.pth null_${n}_100k
  done; done
else
  cd $BASE/code || exit 1
  for s in 620101 620203 620307 620411 620509; do
    n=BFR-LC-ln-s$s
    one $B/runs/$n/checkpoint_00150000.pth null_${n}_150k
    one $B/runs/$n/checkpoint_00600000.pth null_${n}_600k
    one $B/runs2/$n/checkpoint_01200000.pth null_${n}_1200k
  done
fi
echo "NULL TASK $t DONE"
