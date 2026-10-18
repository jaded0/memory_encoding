#!/bin/bash
# Re-forward drift test (PREREG_fastlayers.md, "Re-forward drift test (2026-10-16)"). usage: rf_run.sh K SEED GPU [N_ITERS] [OUTDIR]
# 3-char R-break (lr 1e-3, recurrence off, fraction .2, forget .01, weight clamp 1), plain (no fast-layer flags),
# --fast_backward_per_forward K at matched lr*alpha*K: alpha = 1e4 / K. Code: main bf2cb03 (git archive).
source ~/miniforge3/etc/profile.d/conda.sh; conda activate hebby
export CUDA_VISIBLE_DEVICES=$3 OMP_NUM_THREADS=4 TORCHINDUCTOR_CACHE_DIR=$(mktemp -d /tmp/inductor_rf_XXXX)
R=~/reforward_2026-10-16; K=$1; seed=$2; N=${4:-350000}; OUT=${5:-$R/runs}; cd $R/code
alpha=$(python -c "print(int(10000/$K))")
F=(--dataset 3_palindrome_dataset_vary_length --model_type ephemeral --updater dfa
   --hidden_size 1024 --num_layers 3 --batch_size 16 --input_mode last_one --learning_rate 0.001
   --plasticity $alpha --ephemeral_fraction .2 --forget_rate .01 --weight_clamp 1 --enable_recurrence False
   --residual_connection False --layer_norm False --positional_encoding_dim 0 --track false
   --deterministic true --fused_update ${FUSED:-True} --optimizer sgd --grad_norm_clip 0 --print_freq 500
   --early_stop_window 0 --checkpoint_save_freq 25000 --checkpoint_keep_every 25000
   --n_iters $N --heldout_eval_every 50000 --heldout_batches 8 --fast_backward_per_forward $K)
j=K${K}-s$seed; d=$OUT/$j; mkdir -p $d
python -u train.py "${F[@]}" --seed $seed --resume true --checkpoint_dir $d > $d/train.log 2>&1; rc=$?
python -u heldout.py --checkpoint $d/latest_checkpoint.pth --protocols observed strict no_fast --batches 128 --json $d/final_eval.json > $d/final_eval.log 2>&1
echo "$(date +%F\ %T) EXIT $rc $j" >> $OUT/status.txt
