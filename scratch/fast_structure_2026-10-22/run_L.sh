#!/bin/bash
# One load-test run (Batch FS part L) and its evaluations. Usage: run_L.sh NAME STRUCTURE ALPHA SEED RUNS_DIR
# Run from the code directory. Resumable: re-running continues training and redoes missing evaluations.
name=$1; st=$2; alpha=$3; seed=$4; runs=$5
d=$runs/$name; mkdir -p $d
python -u train.py --dataset mqarw_r1_52_q8 --model_type ephemeral --updater dfa --fast_structure $st \
  --plasticity $alpha --learning_rate 1e-4 --ephemeral_fraction 0.1 --forget_rate 0.01 \
  --weight_clamp 0 --ephemeral_update_clamp 0 --fast_weight_clamp 0 --slow_weight_decay 0 \
  --hidden_size 1024 --num_layers 3 --batch_size 16 --input_mode last_one --positional_encoding_dim 0 \
  --enable_recurrence False --residual_connection False --layer_norm False --fused_update True \
  --deterministic true --track false --seed $seed --n_iters 150000 --print_freq 2500 \
  --checkpoint_dir $d --checkpoint_save_freq 5000 --checkpoint_keep_every 50000 \
  --trace_loop_every 25000 --resume true >> $d/train.log 2>&1
rc=$?; echo "TRAIN EXIT $rc"
for it in 50000 100000 150000; do
  ck=$d/checkpoint_$(printf %08d $it).pth
  [ -f $ck ] || continue
  out=$d/load_$((it / 1000))k.json
  [ -f $out ] && continue
  python -u scratch/fast_structure_2026-10-22/load_eval.py --checkpoint $ck --json $out >> $d/eval.log 2>&1
  echo "EVAL $it EXIT $?"
done
# A run stopped early (divergence) has no numbered checkpoint at 150k: evaluate its last state.
if [ ! -f $d/checkpoint_00150000.pth ] && [ -f $d/latest_checkpoint.pth ] && [ ! -f $d/load_final.json ]; then
  python -u scratch/fast_structure_2026-10-22/load_eval.py --checkpoint $d/latest_checkpoint.pth \
    --json $d/load_final.json --ablations all >> $d/eval.log 2>&1
  echo "EVAL final EXIT $?"
fi
