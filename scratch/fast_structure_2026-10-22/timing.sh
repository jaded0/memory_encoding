#!/bin/bash
# Speed of the three structures at hidden 1024, one run alone on the GPU (CUDA_VISIBLE_DEVICES set by
# the caller), fused/compiled and unfused, on 3pal (R-reason flags) and on the mqarw mixture (load-test
# flags). Writes timing_<task>_<structure>_<fused>.log; summarize with timing_summary.py.
# Usage (from the code directory): bash scratch/fast_structure_2026-10-22/timing.sh OUTDIR
out=${1:?outdir}; mkdir -p $out
for task in 3pal mqarw; do
  for structure in exclusive additive_masked additive_dense; do
    for fused in True False; do
      if [ $task = 3pal ]; then
        alpha=50000; [ $structure = additive_dense ] && alpha=10000
        flags="--dataset 3_palindrome_dataset_vary_length --learning_rate 0.0002 --plasticity $alpha --ephemeral_fraction .2
               --weight_clamp 1 --n_iters 3000 --print_freq 500"
        [ $structure != exclusive ] && flags="$flags --fast_weight_clamp 1"
      else
        alpha=30000; [ $structure = additive_dense ] && alpha=3000
        flags="--dataset mqarw_r1_52_q8 --learning_rate 1e-4 --plasticity $alpha --ephemeral_fraction 0.1
               --weight_clamp 0 --n_iters 600 --print_freq 100"
      fi
      python -u train.py $flags --fast_structure $structure --model_type ephemeral --updater dfa \
        --hidden_size 1024 --num_layers 3 --batch_size 16 --input_mode last_one --forget_rate 0.01 \
        --enable_recurrence False --residual_connection False --positional_encoding_dim 0 --track false \
        --seed 1 --deterministic true --fused_update $fused --early_stop_window 0 \
        --checkpoint_save_freq 0 --checkpoint_dir $out/ck_${task}_${structure}_$fused \
        > $out/timing_${task}_${structure}_$fused.log 2>&1
    done
  done
done
