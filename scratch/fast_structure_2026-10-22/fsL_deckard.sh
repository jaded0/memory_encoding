#!/bin/bash
# Batch FS part L arms on deckard GPU 0 (GPU 1 belongs to another agent): D30 additive_dense alpha 3e4 x2, concurrently. Run from the code directory with nohup.
export CUDA_VISIBLE_DEVICES=0
seeds=(2718 3141 1618)
# (M3 moved to ORC, fsLm.sbatch: three 150k mqarw runs would take ~80 h on this one GPU.)
for i in 0 1; do
  bash scratch/fast_structure_2026-10-22/run_L.sh L-D30-s${seeds[$i]} additive_dense 30000 ${seeds[$i]} ../runs_L > ../runs_L_D30_$i.out 2>&1 &
done
wait
