#!/bin/bash
# Run Batch FS array tasks on deckard GPU 0 that were cancelled on ORC (scancel <job>_<task> first, from omen
# via manticore), CONCURRENCY at a time. Usage (from the code directory):
#   CONCURRENCY=2 setsid nohup bash scratch/fast_structure_2026-10-22/deckard_queue.sh fsC.sbatch 20 21 22 ... &
# The sbatch file's body runs as is with SLURM_ARRAY_TASK_ID set (same paths on deckard and ORC).
export CUDA_VISIBLE_DEVICES=0
script=scratch/fast_structure_2026-10-22/$1; shift
mkdir -p ../logs
for t in "$@"; do
  while [ $(jobs -rp | wc -l) -ge ${CONCURRENCY:-2} ]; do sleep 30; done
  SLURM_ARRAY_TASK_ID=$t bash $script > ../logs/$(basename $script .sbatch)_deckard_$t.out 2>&1 &
  sleep 5
done
wait
