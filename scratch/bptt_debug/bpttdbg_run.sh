#!/bin/bash
# usage: bpttdbg_run.sh <script.py> "<args1>" "<args2>" ...   (runs each config in parallel, prints the recall lines)
cd "$(dirname "$0")/.."
script=$1; shift
i=0
for args in "$@"; do
    ( out=$(CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 ~/miniforge3/envs/hebby/bin/python "scratch/$script" $args 2>&1 | tr '\r' '\n' | grep -E "^[0-9]+ loss|Error|error")
      printf '== %s\n%s\n' "$args" "$out" ) > "/tmp/bpttdbg_$$_$i.log" &
    i=$((i+1))
done
wait
cat /tmp/bpttdbg_$$_*.log
rm -f /tmp/bpttdbg_$$_*.log
