#!/bin/bash
# Batch FS part C on deckard GPU 0: DRx (dense alpha 5e4 at R-reason, equal alpha = 5x the matched gain), seeds 620101 620203 620307.
export CUDA_VISIBLE_DEVICES=0
seeds=(620101 620203 620307)
for i in 0 1 2; do
  bash scratch/fast_structure_2026-10-22/run_C.sh C-DRx-s${seeds[$i]} additive_dense 0.0002 50000 ${seeds[$i]} ../runs_C \
    "--weight_clamp 1 --fast_weight_clamp 1" > ../runs_C_DRx_$i.out 2>&1 &
done
wait
