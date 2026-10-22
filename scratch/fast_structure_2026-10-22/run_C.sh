#!/bin/bash
# One C6 run on 3pal (Batch FS part C), V2 / Batch Q flags, and its final held-out evaluation.
# Usage: run_C.sh NAME STRUCTURE LR ALPHA SEED RUNS_DIR "CLAMP FLAGS". Run from the code directory; resumable.
name=$1; st=$2; lr=$3; alpha=$4; seed=$5; runs=$6; clamps=$7
d=$runs/$name; mkdir -p $d
python -u train.py --dataset 3_palindrome_dataset_vary_length --model_type ephemeral --updater dfa --fast_structure $st \
  --learning_rate $lr --plasticity $alpha --ephemeral_fraction .2 --forget_rate .01 $clamps --ephemeral_update_clamp 0 \
  --hidden_size 1024 --num_layers 3 --batch_size 16 --input_mode last_one --enable_recurrence False \
  --residual_connection False --layer_norm False --positional_encoding_dim 0 --track false --deterministic true \
  --fused_update True --optimizer sgd --grad_norm_clip 0 --print_freq 500 --early_stop_window 10 \
  --checkpoint_save_freq 25000 --checkpoint_keep_every 50000 --heldout_eval_every 50000 --heldout_batches 8 \
  --seed $seed --n_iters 600000 --checkpoint_dir $d --resume true >> $d/train.log 2>&1
rc=$?; echo "TRAIN EXIT $rc"
[ -f $d/final_eval.json ] || python -u heldout.py --checkpoint $d/latest_checkpoint.pth --protocols observed strict no_fast \
  --batches 128 --json $d/final_eval.json >> $d/eval.log 2>&1
echo "EVAL EXIT $?"
