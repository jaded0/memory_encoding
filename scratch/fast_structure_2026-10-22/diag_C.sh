#!/bin/bash
# Diagnostic, not pre-registered (Batch FS part C): replay C-DR-s620101 and C-AR-s620101 (deterministic, same flags as
# run_C.sh) to 50k with loop traces every 500 iterations, to see which layer's fast read runs away before DR's ~43k
# collapse. Run from the code directory on deckard GPU 0.
export CUDA_VISIBLE_DEVICES=0
common="--dataset 3_palindrome_dataset_vary_length --model_type ephemeral --updater dfa --learning_rate 0.0002
  --ephemeral_fraction .2 --forget_rate .01 --weight_clamp 1 --ephemeral_update_clamp 0 --hidden_size 1024 --num_layers 3
  --batch_size 16 --input_mode last_one --enable_recurrence False --residual_connection False --layer_norm False
  --positional_encoding_dim 0 --track false --deterministic true --fused_update True --optimizer sgd --grad_norm_clip 0
  --print_freq 500 --early_stop_window 0 --checkpoint_save_freq 0 --heldout_eval_every 50000 --heldout_batches 8
  --seed 620101 --n_iters 50000 --trace_loop_every 500"
mkdir -p ../runs_Cdiag
python -u train.py $common --fast_structure additive_dense --plasticity 10000 --fast_weight_clamp 1 \
  --checkpoint_dir ../runs_Cdiag/DR-s620101 > ../runs_Cdiag/DR-s620101.log 2>&1 &
python -u train.py $common --fast_structure exclusive --plasticity 50000 --fast_weight_clamp 0 \
  --checkpoint_dir ../runs_Cdiag/AR-s620101 > ../runs_Cdiag/AR-s620101.log 2>&1 &
wait
