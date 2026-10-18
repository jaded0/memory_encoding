#!/bin/bash
# Re-forward drift test, warm-start arm (PREREG ADDENDUM 1). usage: rf_warm.sh K SEED GPU
# Waits for runs/K1-s<SEED>/checkpoint_00050000.pth, converts it to K passes at alpha 1e4/K (warmstart.py), resumes to 350k.
source ~/miniforge3/etc/profile.d/conda.sh; conda activate hebby
export CUDA_VISIBLE_DEVICES=$3 OMP_NUM_THREADS=4 TORCHINDUCTOR_CACHE_DIR=$(mktemp -d /tmp/inductor_rfw_XXXX)
R=~/reforward_2026-10-16; K=$1; seed=$2; cd $R/code
src=$R/runs/K1-s$seed/checkpoint_00050000.pth
until [ -e $src ] && [ $(( $(date +%s) - $(stat -c %Y $src) )) -gt 60 ]; do sleep 60; done
alpha=$(python -c "print(int(10000/$K))")
j=W${K}-s$seed; d=$R/runs/$j; mkdir -p $d
[ -e $d/latest_checkpoint.pth ] && start=(--resume true) || { python $R/warmstart.py $src $K $d/start_50k.pth > $d/warmstart.log 2>&1; start=(--resume_checkpoint $d/start_50k.pth); }
F=(--dataset 3_palindrome_dataset_vary_length --model_type ephemeral --updater dfa
   --hidden_size 1024 --num_layers 3 --batch_size 16 --input_mode last_one --learning_rate 0.001
   --plasticity $alpha --ephemeral_fraction .2 --forget_rate .01 --weight_clamp 1 --enable_recurrence False
   --residual_connection False --layer_norm False --positional_encoding_dim 0 --track false
   --deterministic true --fused_update True --optimizer sgd --grad_norm_clip 0 --print_freq 500
   --early_stop_window 0 --checkpoint_save_freq 25000 --checkpoint_keep_every 25000
   --n_iters 350000 --heldout_eval_every 50000 --heldout_batches 8 --fast_backward_per_forward $K)
python -u train.py "${F[@]}" --seed $seed "${start[@]}" --checkpoint_dir $d >> $d/train.log 2>&1; rc=$?
python -u heldout.py --checkpoint $d/latest_checkpoint.pth --protocols observed strict no_fast --batches 128 --json $d/final_eval.json > $d/final_eval.log 2>&1
echo "$(date +%F\ %T) EXIT $rc $j" >> $R/runs/status.txt
