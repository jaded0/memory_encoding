#!/bin/bash
# One from-scratch run of Batch F9 or S7b plus its final held-out evaluation (queue 2026-10-27).
# Usage (from the code directory): bash scratch/f9_s7_2026-10-27/run_arm.sh ARM SEED ITERS RUNS_DIR
#   F9    = the V2 DC2 R-break recipe (Batch V2 v2.sbatch, tasks 5-9) with --weight_clamp 0 instead of 1.
#   S7D   = the plain R-break recipe (V2's flags without an arm flag = battery A1 plain) plus --slow_weight_decay 3e-6.
#   DC2c1 / plain = the unchanged V2 DC2 / plain recipes (smoke-test references only, not part of either batch).
# Everything else is V2's command verbatim (lr 1e-3, alpha 1e4, frac .2, forget .01, print 500, checkpoints every
# 25k, numbered every 50k, held-out every 50k on 8 batches), except the divergence guard --early_stop_window 100:
# stop after 100 consecutive 500-iteration windows with loss > 5 (50k iterations). V2 used 0 (never stop). The
# longest excursion that recovered in this recipe is ~29 windows (battery A1 plain s620411, 5k-19k), and V2's onset
# rule needs 5 windows, so the guard cannot stop a run before its onset is recorded or end a recoverable excursion.
# A NaN/inf loss always stops train.py. The code commit is read from CODE_COMMIT (written by stage_code.sh).
# SAVE_FREQ overrides the 25k checkpoint interval (smoke tests only).
# Resumable: re-running continues from <run>/latest_checkpoint.pth unless the guard already stopped the run.
set -u
arm=$1; seed=$2; iters=$3; runs=$4
case $arm in
  F9)    clamp=0; extra="--decouple_fast_feedback 2" ;;
  S7D)   clamp=1; extra="--slow_weight_decay 3e-6" ;;
  DC2c1) clamp=1; extra="--decouple_fast_feedback 2" ;;
  plain) clamp=1; extra="" ;;
  *) echo "unknown arm $arm"; exit 2 ;;
esac
name=Rbreak-$arm-s$seed; d=$runs/$name; mkdir -p $d
echo "=== $(date) host $(hostname) arm $arm seed $seed iters $iters code $(cat CODE_COMMIT 2>/dev/null || echo unknown)" >> $d/train.log
if grep -qE "^Early stopping:|^Non-finite loss" $d/train.log; then
  echo "run already stopped by the divergence guard; not resuming" | tee -a $d/train.log
else
  python -u train.py --dataset 3_palindrome_dataset_vary_length --model_type ephemeral --updater dfa \
    --hidden_size 1024 --num_layers 3 --batch_size 16 --input_mode last_one --learning_rate 0.001 \
    --plasticity 10000 --ephemeral_fraction .2 --forget_rate .01 --weight_clamp $clamp --enable_recurrence False \
    --residual_connection False --layer_norm False --positional_encoding_dim 0 --track false \
    --deterministic true --fused_update True --optimizer sgd --grad_norm_clip 0 --print_freq 500 \
    --early_stop_window 100 --checkpoint_save_freq ${SAVE_FREQ:-25000} --checkpoint_keep_every 50000 --seed $seed --resume true \
    --n_iters $iters --heldout_eval_every 50000 --heldout_batches 8 \
    --checkpoint_dir $d $extra >> $d/train.log 2>&1
  echo "TRAIN EXIT $?" | tee -a $d/train.log
fi
python -u heldout.py --checkpoint $d/latest_checkpoint.pth --protocols observed strict no_fast --batches 128 \
  --json $d/final_eval.json > $d/final_eval.log 2>&1
echo "EVAL EXIT $?" | tee -a $d/train.log
