#!/bin/bash
# Extra workers on GPU 0 (1080 Ti, unfused step) pulling from the same queue as launch_kr.sh (after the KR split: manticore keeps plain_ln
# on the 2080 Ti and plain_off here; the other 30 KR runs went to ORC, bfr_kr.sbatch).
cd ~/bfr_kr_2026-10-22/scripts
P0=${P0:-5}
take() { flock queue.lock bash -c 'head -n1 queue.txt; sed -i 1d queue.txt'; }
worker() { gpu=$1
  while true; do
    job=$(take); [ -z "$job" ] && break
    IFS="|" read -r name seed n extra <<< "$job"
    ./run_kr.sh "$name" $gpu "$seed" "$n" "$extra"
  done; }
for k in $(seq $P0); do worker 0 >> worker0.log 2>&1 & sleep 5; done
wait
echo GPU0 DONE $(date)
