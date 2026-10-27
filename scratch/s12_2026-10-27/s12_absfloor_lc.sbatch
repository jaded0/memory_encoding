#!/bin/bash
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus=1
#SBATCH --mem=30G
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --exclude=cs-1-5,cs-1-3
#SBATCH --job-name=s12_afl
#SBATCH --array=0-5
#SBATCH --output=/home/jaded79/s12_2026-10-27/logs/%x_%A_%a.out
# S12 follow-up 3 (prep 2026-10-27; PREREG "Batch S12F", A1-A4): absrun with a weaker divisor on the 3pal late collapse.
# = Batch BFR's LC absrun arm (batchBFR.sbatch tasks 5-9) + --absrun_floor c, c in {10, 100} x seeds 620101 620203 620307, 300k.
# 0-2: c = 10, 3-5: c = 100. Pairs with BFR-LC-absrun-s<seed> (ORC ~/bfr_2026-10-22/runs/; 150k checkpoints there, ratios in ratios/).
# Resubmitting resumes unfinished tasks (--resume true). After training: heldout.py, then bfr_ratio.py (trunk eff/slow) at 150k and the end.
N_ITERS=${N_ITERS:-300000}
BASE=${BASE:-$HOME/s12_2026-10-27}
source ~/.bashrc >/dev/null 2>&1; source ~/miniforge3/etc/profile.d/conda.sh 2>/dev/null
conda activate hebby
t=${SLURM_ARRAY_TASK_ID:-0}
seeds3=(620101 620203 620307); floors=(10 100)
c=${floors[$((t/3))]}; seed=${seeds3[$((t%3))]}; name=S12-LC-absrunC$c-s$seed
common="--dataset 3_palindrome_dataset_vary_length --model_type ephemeral --updater dfa --hidden_size 1024 --batch_size 16 --input_mode last_one
  --learning_rate 0.001 --plasticity 10000 --ephemeral_fraction .2 --forget_rate .01 --weight_clamp 1 --enable_recurrence False
  --residual_connection False --layer_norm False --positional_encoding_dim 0 --track false --deterministic true --fused_update True
  --optimizer sgd --print_freq 500 --early_stop_window 0 --resume true --grad_norm_clip 0"
cd $BASE/code || exit 1
echo "task $t -> $name ($N_ITERS its, absrun, absrun_floor $c), code $(cat COMMIT 2>/dev/null)"; nvidia-smi --query-gpu=name --format=csv,noheader
mkdir -p $HOME/.cache/inductor_s12; export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d $HOME/.cache/inductor_s12/XXXX)
d=$BASE/runs/$name; mkdir -p $d
python -u train.py $common --num_layers 3 --seed $seed --n_iters $N_ITERS --checkpoint_dir $d \
  --fast_read_norm absrun --absrun_floor $c \
  --checkpoint_save_freq 25000 --checkpoint_keep_every 50000 --heldout_eval_every 50000 --heldout_batches 8
rc=$?; echo "TRAIN EXIT $rc"
[ $rc -eq 0 ] || exit $rc
python -u heldout.py --checkpoint $d/latest_checkpoint.pth --protocols observed strict no_fast --batches 128 --json $d/final_eval.json
echo "EVAL EXIT $?"
mkdir -p $BASE/ratios
for k in 150000 $N_ITERS; do
  ck=$d/checkpoint_$(printf %08d $k).pth; [ -f $ck ] || continue
  python scratch/bfr_2026-10-22/bfr_ratio.py --checkpoint $ck --batches 16 --json $BASE/ratios/ratio_${name}_$((k / 1000))k.json > /dev/null
  echo "RATIO $k EXIT $?"
done
