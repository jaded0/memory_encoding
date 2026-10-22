#!/bin/bash
# Batch FS part L arms on deckard GPU 0 (pre-registered; GPU 1 belongs to another agent): M3 additive_masked alpha 3e4 x3,
# D30 additive_dense alpha 3e4 x2, all concurrently on GPU 0. Run from the code directory with nohup.
export CUDA_VISIBLE_DEVICES=0
seeds=(2718 3141 1618)
for i in 0 1 2; do
  bash scratch/fast_structure_2026-10-22/run_L.sh L-M3-s${seeds[$i]} additive_masked 30000 ${seeds[$i]} ../runs_L > ../runs_L_M3_$i.out 2>&1 &
done
for i in 0 1; do
  bash scratch/fast_structure_2026-10-22/run_L.sh L-D30-s${seeds[$i]} additive_dense 30000 ${seeds[$i]} ../runs_L > ../runs_L_D30_$i.out 2>&1 &
done
wait
