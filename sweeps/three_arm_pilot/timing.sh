#!/bin/bash
cd ~/lowrank_deep/three_arm
for s in exclusive additive_masked additive_dense; do for f in false true; do
  rm -rf runs/timing_${s}_$f
  ./run_arm.sh timing_${s}_$f $s 1e4 7 5000 --fused_update $f > timing_${s}_$f.log 2>&1
  echo "$s fused=$f: $(grep iters_per_sec timing_${s}_$f.log | awk '{print $2}' | tr '\n' ' ')" >> timing_summary.txt
done; done
echo DONE >> timing_summary.txt
