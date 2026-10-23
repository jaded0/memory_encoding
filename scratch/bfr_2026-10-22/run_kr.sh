#!/bin/bash
# Batch BFR, KR arm: key recall at c-phi = 2 (the RN recipe, RN's regenerated dataset). Usage: run_kr.sh NAME GPU SEED NITERS "extra args"
# Runs in ~/bfr_kr_2026-10-22 on manticore; code = git archive of main b26e88d.
source ~/miniforge3/etc/profile.d/conda.sh; conda activate hebby
name=$1; gpu=$2; seed=$3; n=$4; extra=$5
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu TORCHINDUCTOR_CACHE_DIR=$(mktemp -d /tmp/inductor_bfr_XXXX)
cd ~/bfr_kr_2026-10-22/code; d=~/bfr_kr_2026-10-22/runs/$name
if grep -q "^EXIT 0" $d/train.log 2>/dev/null && [ -f $d/heldout.json ]; then echo "skip $name"; exit 0; fi
rm -rf $d; mkdir -p $d
echo "start $name gpu $gpu $(date)"
python -u train.py --dataset long_range_memory_dataset --model_type ephemeral --updater dfa \
  --hidden_size 1024 --num_layers 3 --batch_size 16 --input_mode last_one --learning_rate 0.001 \
  --plasticity 10000 --ephemeral_fraction .2 --forget_rate .01 --weight_clamp 1 --enable_recurrence False \
  --residual_connection False --layer_norm False --positional_encoding_dim 0 --track false \
  --deterministic true --fused_update True --optimizer sgd --grad_norm_clip 0 --print_freq 100 \
  --early_stop_window 0 --checkpoint_save_freq 10000 --checkpoint_keep_every 10000 --seed $seed --n_iters $n \
  --heldout_eval_every 0 --checkpoint_dir $d $extra > $d/train.log 2>&1
echo "EXIT $?" >> $d/train.log
python heldout.py --checkpoint $d/latest_checkpoint.pth --protocols observed strict no_fast --batches 128 \
  --json $d/heldout.json > $d/heldout.log 2>&1
echo "done $name $(date)"
