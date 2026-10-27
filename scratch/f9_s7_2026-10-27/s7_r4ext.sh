#!/bin/bash
# Batch S7 part a (queue 2026-10-27; PREREG_fastlayers.md "Batch S7"): extend R4 (slow weight decay 3e-6 from
# B_late-s3141's 150k checkpoint) from 300k to 600k on deckard, an exact continuation of INT2_R4.
# Usage (on deckard, from the code directory ~/f9s7_2026-10-27/code):
#   setsid nohup bash scratch/f9_s7_2026-10-27/s7_r4ext.sh GPU > ../logs/s7_r4ext.out 2>&1 &
# Source: ~/overnight_2026-10-01/runs/INT2_R4/latest_checkpoint.pth (iter 300000, code_version 12, the only copy of
# R4 at 300k; this script copies it into the run directory first and never writes to the source directory).
# Flags: R4's own (jobdefs.sh CTRL + int_jobdefs.sh INT_COMMON + --slow_weight_decay 3e-6, seed 3141), with these
# observation-only changes: --print_freq 500 (R4 printed every 2500; 500 gives V2's onset windows; print steps run
# the unfused step, a rounding-level difference), numbered checkpoints every 50k (all kept), held-out every 50k on
# 8 batches, and the V2/F9 divergence guard (--early_stop_window 100; R4 used the default 10 at print 2500).
# SRC, RUNDIR and ITERS override the source, run directory and 600000 (smoke tests only).
# A second invocation continues from the run's own latest_checkpoint.pth.
GPU=$1
src=${SRC:-$HOME/overnight_2026-10-01/runs/INT2_R4/latest_checkpoint.pth}
d=${RUNDIR:-$HOME/f9s7_2026-10-27/runs/R4ext-s3141}; mkdir -p $d ../logs
source ~/miniforge3/etc/profile.d/conda.sh; conda activate hebby
set -u   # after conda activate (its MKL hook reads unset variables)
export CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled HF_DATASETS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d "${TMPDIR:-/tmp}/inductor_r4ext_XXXXXX")
[ -f $d/source_INT2_R4_300k.pth ] || cp -p $src $d/source_INT2_R4_300k.pth
if [ -f $d/latest_checkpoint.pth ]; then res="--resume true"; else res="--resume_checkpoint $d/source_INT2_R4_300k.pth"; fi
echo "=== $(date) host $(hostname) gpu $GPU code $(cat CODE_COMMIT 2>/dev/null || echo unknown) resume: $res" >> $d/train.log
if grep -qE "^Early stopping:|^Non-finite loss" $d/train.log; then
  echo "run already stopped by the divergence guard; not resuming" | tee -a $d/train.log
else
  python -u train.py --model_type ephemeral --updater dfa --enable_recurrence false --fused_update true \
    --dataset 3_palindrome_dataset_vary_length --input_mode last_one --learning_rate 1e-3 \
    --plasticity 1e4 --forget_rate 0.01 --ephemeral_fraction 0.2 --weight_clamp 1 \
    --hidden_size 1024 --num_layers 3 --residual_connection false --positional_encoding_dim 0 \
    --batch_size 16 --track false --deterministic true --resume true \
    --slow_weight_decay 3e-6 --seed 3141 --n_iters ${ITERS:-600000} \
    --print_freq 500 --trace_loop_every 2500 --checkpoint_save_freq 5000 \
    --checkpoint_keep_every 50000 --checkpoint_keep_max 0 --early_stop_window 100 \
    --heldout_eval_every 50000 --heldout_batches 8 \
    --checkpoint_dir $d $res >> $d/train.log 2>&1
  echo "TRAIN EXIT $?" | tee -a $d/train.log
fi
python -u heldout.py --checkpoint $d/latest_checkpoint.pth --protocols observed strict no_fast --batches 128 \
  --json $d/final_eval.json > $d/final_eval.log 2>&1
echo "EVAL EXIT $?" | tee -a $d/train.log
rm -rf "$TORCHINDUCTOR_CACHE_DIR"
