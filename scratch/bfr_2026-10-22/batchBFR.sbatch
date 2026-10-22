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
#SBATCH --job-name=bfr
#SBATCH --array=0-15
#SBATCH --output=/home/jaded79/bfr_2026-10-22/logs/%x_%A_%a.out
# Batch BFR (agent A, 2026-10-22, pre-registered in PREREG_fastlayers.md "Batch BFR"). main b26e88d (code_main = git archive).
# 0-4   LC: 3pal R-break (= Batch Q F7 recipe) + --fast_read_norm ln, seeds 620101..620509, 600k
# 5-9   LC: same with --fast_read_norm absrun
# 10-15 SIX: 6-layer, 5k, ln (10-12) / absrun (13-15), seeds 620101 620203 620307
source ~/.bashrc >/dev/null 2>&1
conda activate hebby
t=$SLURM_ARRAY_TASK_ID
seeds5=(620101 620203 620307 620411 620509)
common="--dataset 3_palindrome_dataset_vary_length --model_type ephemeral --updater dfa --hidden_size 1024 --batch_size 16 --input_mode last_one
  --learning_rate 0.001 --plasticity 10000 --ephemeral_fraction .2 --forget_rate .01 --weight_clamp 1 --enable_recurrence False
  --residual_connection False --layer_norm False --positional_encoding_dim 0 --track false --deterministic true --fused_update True
  --optimizer sgd --print_freq 500 --early_stop_window 0 --resume true --grad_norm_clip 0"
if [ $t -lt 10 ]; then
  modes=(ln absrun); mode=${modes[$((t/5))]}; seed=${seeds5[$((t%5))]}
  name=BFR-LC-$mode-s$seed; iters=600000; layers=3
  extra="--checkpoint_save_freq 25000 --checkpoint_keep_every 50000 --heldout_eval_every 50000 --heldout_batches 8"
else
  i=$((t-10)); modes=(ln absrun); mode=${modes[$((i/3))]}; seed=${seeds5[$((i%3))]}
  name=BFR-SIX-$mode-s$seed; iters=5000; layers=6
  extra="--checkpoint_save_freq 5000 --heldout_eval_every 0"
fi
echo "task $t -> $name ($layers layers, $iters its, fast_read_norm $mode)"; nvidia-smi --query-gpu=name --format=csv,noheader
cd ~/bfr_2026-10-22/code_main
mkdir -p $HOME/.cache/inductor_bfr; export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d $HOME/.cache/inductor_bfr/XXXX)
d=../runs/$name; mkdir -p $d
python -u train.py $common --num_layers $layers --seed $seed --n_iters $iters --checkpoint_dir $d --fast_read_norm $mode $extra
rc=$?; echo "TRAIN EXIT $rc"
python -u heldout.py --checkpoint $d/latest_checkpoint.pth --protocols observed strict no_fast --batches 128 --json $d/final_eval.json
echo "EVAL EXIT $?"
