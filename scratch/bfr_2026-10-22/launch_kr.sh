#!/bin/bash
# Batch BFR KR: 40 runs from one shared queue; P1 workers on GPU 1 (2080 Ti), P0 on GPU 0 (1080 Ti). Each worker takes the next
# job under a lock, so the faster GPU takes more. Normalized arms first, the known-failing off arms last.
cd ~/bfr_kr_2026-10-22/scripts
P1=${P1:-5}; P0=${P0:-4}
: > queue.txt
for spec in "plain_ln|--fast_read_norm ln" "DC2_ln|--decouple_fast_feedback 2 --fast_read_norm ln" \
            "plain_absrun|--fast_read_norm absrun" "DC2_absrun|--decouple_fast_feedback 2 --fast_read_norm absrun" \
            "plain_rms|--fast_read_norm rms" "DC2_rms|--decouple_fast_feedback 2 --fast_read_norm rms" \
            "plain_off|" "DC2_off|--decouple_fast_feedback 2"; do
  for seed in 2718 3141 1618 1414 5772; do
    name=kr_${spec%%|*}_s$seed; n=30000
    case $name in *_off_*) n=10000;; esac
    echo "$name|$seed|$n|${spec#*|}" >> queue.txt
  done
done
take() { flock queue.lock bash -c 'head -n1 queue.txt; sed -i 1d queue.txt'; }
worker() { gpu=$1
  while true; do
    job=$(take); [ -z "$job" ] && break
    IFS="|" read -r name seed n extra <<< "$job"
    ./run_kr.sh "$name" $gpu "$seed" "$n" "$extra"
  done; }
for k in $(seq $P1); do worker 1 >> worker1.log 2>&1 & sleep 5; done
for k in $(seq $P0); do worker 0 >> worker0.log 2>&1 & sleep 5; done
wait
echo ALL DONE $(date)
