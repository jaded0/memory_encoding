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
#SBATCH --job-name=s12_aflkr
#SBATCH --array=0-5
#SBATCH --output=/home/jaded79/s12_2026-10-27/logs/%x_%A_%a.out
# S12 follow-up 3, KR part (prep 2026-10-27; PREREG "Batch S12F", A5-A6): key recall at c-phi = 2 (the RN recipe, RN's regenerated
# long_range_memory_dataset in $BASE/code_kr), plain absrun + --absrun_floor c, c in {10, 100} x seeds 2718 3141 1618, 30k.
# 0-2: c = 10, 3-5: c = 100. Same flags as bfr_kr.sbatch's plain_absrun arm (Batch BFR), which it pairs with.
N_ITERS=${N_ITERS:-30000}
BASE=${BASE:-$HOME/s12_2026-10-27}
CKPT_EVERY=${CKPT_EVERY:-10000}  # smoke tests only: N_ITERS must be a multiple (latest_checkpoint.pth is saved on these)
source ~/.bashrc >/dev/null 2>&1; source ~/miniforge3/etc/profile.d/conda.sh 2>/dev/null
conda activate hebby
set -eo pipefail
t=${SLURM_ARRAY_TASK_ID:-0}
seeds3=(2718 3141 1618); floors=(10 100)
c=${floors[$((t/3))]}; seed=${seeds3[$((t%3))]}; name=kr_plain_absrunC${c}_s$seed
cd $BASE/code_kr || exit 1
echo "task $t -> $name ($N_ITERS its), code $(cat COMMIT 2>/dev/null)"; nvidia-smi --query-gpu=name --format=csv,noheader
mkdir -p $HOME/.cache/inductor_s12; export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d $HOME/.cache/inductor_s12/XXXX)
d=$BASE/runs_kr/$name; mkdir -p $d
python -u train.py --dataset long_range_memory_dataset --model_type ephemeral --updater dfa \
  --hidden_size 1024 --num_layers 3 --batch_size 16 --input_mode last_one --learning_rate 0.001 \
  --plasticity 10000 --ephemeral_fraction .2 --forget_rate .01 --weight_clamp 1 --enable_recurrence False \
  --residual_connection False --layer_norm False --positional_encoding_dim 0 --track false \
  --deterministic true --fused_update True --optimizer sgd --grad_norm_clip 0 --print_freq 100 \
  --early_stop_window 0 --checkpoint_save_freq $CKPT_EVERY --checkpoint_keep_every 10000 --seed $seed --n_iters $N_ITERS \
  --heldout_eval_every 0 --checkpoint_dir $d --resume true --fast_read_norm absrun --absrun_floor $c >> $d/train.log 2>&1
rc=$?; echo "EXIT $rc" >> $d/train.log
python heldout.py --checkpoint $d/latest_checkpoint.pth --protocols observed strict no_fast --batches 128 \
  --json $d/heldout.json > $d/heldout.log 2>&1
echo "EVAL EXIT $?"
mkdir -p $BASE/ratios
for k in 10000 $N_ITERS; do
  ck=$d/checkpoint_$(printf %08d $k).pth
  [ -f $ck ] || [ $k != $N_ITERS ] || ck=$d/latest_checkpoint.pth  # the json records the true iteration
  [ -f $ck ] || continue
  python scratch/bfr_2026-10-22/bfr_ratio.py --checkpoint $ck --batches 16 --json $BASE/ratios/ratio_${name}_$((k / 1000))k.json > /dev/null
  echo "RATIO $k EXIT $?"
done
