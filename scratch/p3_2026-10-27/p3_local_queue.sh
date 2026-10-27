#!/bin/bash
# P3 on a local box (deckard / omen / manticore): run "ARM:SEED" jobs on one GPU, CONCURRENCY at a time.
# usage (from the code directory, a checkout of p3-write-gain; hebby env):
#   GPU=0 CONCURRENCY=3 OUT=$HOME/p3_2026-10-27/runs setsid nohup bash scratch/p3_2026-10-27/p3_local_queue.sh \
#       ctl:2718 g1:2718 g2:2718 g4:2718 ctl:3141 g1:3141 > ../p3_queue_gpu0.log 2>&1 &
# Each job is p3_train.sh ARM SEED $OUT ${N:-100000}; logs go to $OUT/logs/.
source ~/miniforge3/etc/profile.d/conda.sh; conda activate hebby
export CUDA_VISIBLE_DEVICES=${GPU:-0}
out=${OUT:-$HOME/p3_2026-10-27/runs}; mkdir -p $out/logs
for job in "$@"; do
  while [ $(jobs -rp | wc -l) -ge ${CONCURRENCY:-2} ]; do sleep 30; done
  arm=${job%%:*}; seed=${job##*:}
  echo "$(date -Is) start $arm s$seed on GPU $CUDA_VISIBLE_DEVICES"
  export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d /tmp/inductor_p3_XXXX)
  bash scratch/p3_2026-10-27/p3_train.sh $arm $seed $out ${N:-100000} > $out/logs/KR-$arm-s$seed.out 2>&1 &
  sleep 5
done
wait
echo "$(date -Is) queue done"
