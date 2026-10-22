#!/bin/bash
# Batch O2 (order-two fast read, video experiment 5; PREREG_fastlayers.md "Batch O2"). One run + its measurements.
# usage (from the repo root): o2_train.sh DATASET N_ITERS ARM SEED OUTROOT
#   DATASET long_range_memory_dataset (KR, 100k) or mqar_K (200k). Recipe = KRC1 / Batch K bench recipe:
#   lr 1e-4, alpha 3e4, frac .1, forget .01, no clamps, 3x1024, B16, last_one, recurrence off, fused, deterministic.
#   ARM: lin par2 par1 rep2 rep1, and t-variants rep2t1 rep2t16 par2t16 (--order2_temperature).
# Afterwards: heldout.py (observed/strict/no_fast, 128 batches) and o2_keys.py (64 batches, t = 1,4,16) on 50k and the final checkpoint.
data=$1; n=$2; arm=$3; seed=$4; root=$5
case $arm in
  lin) af="";;
  par2) af="--order2_layers 1,2";;
  par1) af="--order2_layers 1,2 --order2_map order1";;
  rep2) af="--order2_layers 1,2 --order2_mode replace";;
  rep1) af="--order2_layers 1,2 --order2_mode replace --order2_map order1";;
  rep2t1) af="--order2_layers 1,2 --order2_mode replace --order2_temperature 1";;
  rep2t16) af="--order2_layers 1,2 --order2_mode replace --order2_temperature 16";;
  par2t16) af="--order2_layers 1,2 --order2_temperature 16";;
  *) echo "unknown arm $arm"; exit 2;;
esac
tag=$([ $data = long_range_memory_dataset ] && echo KR || echo $data)
d=$root/$tag-$arm-s$seed; mkdir -p $d
git rev-parse HEAD > $d/code_commit
if [ ! -f $d/final_eval.json ]; then
python -u train.py --dataset $data --input_mode last_one --forget_rate 0.01 --ephemeral_update_clamp 0 \
  --weight_clamp 0 --slow_weight_decay 0 --hidden_size 1024 --num_layers 3 \
  --residual_connection false --positional_encoding_dim 0 --batch_size 16 --track false --deterministic true \
  --resume true --model_type ephemeral --updater dfa --enable_recurrence false --fused_update true \
  --learning_rate 1e-4 --plasticity 3e4 --ephemeral_fraction 0.1 --optimizer sgd \
  --n_iters $n --print_freq 500 --early_stop_window 0 --heldout_eval_every 25000 --heldout_batches 16 \
  --checkpoint_save_freq 25000 --checkpoint_keep_every 50000 --seed $seed --checkpoint_dir $d $af >> $d/train.log 2>&1
echo "TRAIN EXIT $?" >> $d/train.log
python -u heldout.py --checkpoint $d/latest_checkpoint.pth --protocols observed strict no_fast --batches 128 --json $d/final_eval.json > $d/final_eval.log 2>&1
echo "EVAL EXIT $?" >> $d/train.log
fi
ck=""; for it in 50000 $n; do f=$d/checkpoint_$(printf %08d $it).pth; [ -f $f ] && ck="$ck $f"; done
[ -z "$ck" ] && ck=$d/latest_checkpoint.pth
python -u scratch/order2_2026-10-22/o2_keys.py $ck --out $d/keys --batches 64 --temps 1,4,16 > $d/keys.log 2>&1
echo "KEYS EXIT $?" >> $d/train.log
