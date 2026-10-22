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
#SBATCH --job-name=bfr_kr
#SBATCH --array=0-29
#SBATCH --output=/home/jaded79/bfr_2026-10-22/logs/%x_%A_%a.out
# Batch BFR, KR part on ORC (agent A, 2026-10-22; PREREG_fastlayers.md "Batch BFR"). Same flags as run_kr.sh on manticore (the RN recipe), main b26e88d,
# RN's regenerated long_range_memory_dataset (md5-identical on ORC and manticore). manticore runs plain_ln (2080 Ti) and plain_off (1080 Ti); these 30 here.
source ~/.bashrc >/dev/null 2>&1
conda activate hebby
t=$SLURM_ARRAY_TASK_ID
seeds=(2718 3141 1618 1414 5772)
specs=("DC2_ln|--decouple_fast_feedback 2 --fast_read_norm ln" "plain_absrun|--fast_read_norm absrun"
       "DC2_absrun|--decouple_fast_feedback 2 --fast_read_norm absrun" "plain_rms|--fast_read_norm rms"
       "DC2_rms|--decouple_fast_feedback 2 --fast_read_norm rms" "DC2_off|--decouple_fast_feedback 2")
spec=${specs[$((t/5))]}; seed=${seeds[$((t%5))]}
name=kr_${spec%%|*}_s$seed; extra=${spec#*|}; n=30000
case $name in *_off_*) n=10000;; esac
echo "task $t -> $name ($n its)"; nvidia-smi --query-gpu=name --format=csv,noheader
cd ~/bfr_2026-10-22/code_kr
mkdir -p $HOME/.cache/inductor_bfr; export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d $HOME/.cache/inductor_bfr/XXXX)
d=../runs_kr/$name; mkdir -p $d
python -u train.py --dataset long_range_memory_dataset --model_type ephemeral --updater dfa \
  --hidden_size 1024 --num_layers 3 --batch_size 16 --input_mode last_one --learning_rate 0.001 \
  --plasticity 10000 --ephemeral_fraction .2 --forget_rate .01 --weight_clamp 1 --enable_recurrence False \
  --residual_connection False --layer_norm False --positional_encoding_dim 0 --track false \
  --deterministic true --fused_update True --optimizer sgd --grad_norm_clip 0 --print_freq 100 \
  --early_stop_window 0 --checkpoint_save_freq 10000 --checkpoint_keep_every 10000 --seed $seed --n_iters $n \
  --heldout_eval_every 0 --checkpoint_dir $d --resume true $extra > $d/train.log 2>&1
echo "EXIT $?" >> $d/train.log
python heldout.py --checkpoint $d/latest_checkpoint.pth --protocols observed strict no_fast --batches 128 \
  --json $d/heldout.json > $d/heldout.log 2>&1
echo "EVAL EXIT $?"
