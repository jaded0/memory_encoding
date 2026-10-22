#!/bin/bash
# B2 main batch, seed 7201 of all four arms, concurrently on deckard GPU 1.
export CUDA_VISIBLE_DEVICES=1
export EPHEMERAL_DATA_DIR=$HOME/b2_2026-10-22/processed_datasets
cd $HOME/b2_2026-10-22/code
source ../b2/main_common.sh
for arm in NOFAST L0 DEEP DEEPS; do
  run_arm $arm 7201 $HOME/miniforge3/envs/hebby/bin/python > ../logs/main-$arm-s7201.out 2>&1 &
done
wait
