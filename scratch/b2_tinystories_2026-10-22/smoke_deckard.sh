#!/bin/bash
# B2 recipe smoke on deckard GPU 1 (same flags as smoke.sbatch). Usage: smoke_deckard.sh TASK [TASK...]; runs them concurrently on GPU 1.
export CUDA_VISIBLE_DEVICES=1
export EPHEMERAL_DATA_DIR=$HOME/b2_2026-10-22/processed_datasets
PY=$HOME/miniforge3/envs/hebby/bin/python
common="--dataset roneneldan/tinystories --model_type ephemeral --updater dfa --hidden_size 1024 --num_layers 3 --batch_size 16
  --input_mode last_one --ephemeral_fraction .2 --forget_rate .01 --weight_clamp 1 --enable_recurrence False
  --residual_connection False --positional_encoding_dim 0 --track false --deterministic true --fused_update True
  --optimizer sgd --print_freq 100 --early_stop_window 0 --checkpoint_save_freq 1000 --resume true
  --max_seq_len 512 --n_iters 3000 --seed 7101"
RB="--learning_rate 0.001 --plasticity 10000"
WK="--learning_rate 0.0001 --plasticity 15000"
WH="--learning_rate 0.001 --plasticity 1500"
names=(rb-LN rb-rms rb-LN-rms rb-L01 rb-DC2 rb-L0 wk-plain wk-LN wk-rms wh-plain wh-LN nofast-1e-3 nofast-1e-4 wh-L0)
flags=("$RB --layer_norm True" "$RB --fast_read_norm rms" "$RB --layer_norm True --fast_read_norm rms" "$RB --fast_layers 0,1"
       "$RB --decouple_fast_feedback 2" "$RB --fast_layers 0" "$WK" "$WK --layer_norm True" "$WK --fast_read_norm rms"
       "$WH" "$WH --layer_norm True" "--learning_rate 0.001 --plasticity 0" "--learning_rate 0.0001 --plasticity 0" "$WH --fast_layers 0")
cd $HOME/b2_2026-10-22/code
for t in "$@"; do
  name=smoke-${names[$t]}; d=../runs/$name; mkdir -p $d
  echo "task $t -> $name: ${flags[$t]}" > ../logs/$name.out
  nohup $PY -u train.py $common ${flags[$t]} --checkpoint_dir $d >> ../logs/$name.out 2>&1 &
done
wait
