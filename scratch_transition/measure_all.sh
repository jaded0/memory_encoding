#!/bin/bash
# usage: measure_all.sh SEED   Watcher: measure every numbered snapshot as it appears. Stops when ckpt_060001 is measured.
source ~/miniforge3/etc/profile.d/conda.sh; conda activate hebby
export CUDA_VISIBLE_DEVICES=0
cd ~/lowrank_deep/transition/code
T=~/lowrank_deep/transition
mkdir -p $T/meas2
while true; do
  did=0
  for S in $1; do
    for f in $T/ckpt_s$S/ckpt_*.pth; do
      [ -e "$f" ] || continue
      b=$(basename $f .pth); out=$T/meas2/s${S}_$b.json
      [ -e $out ] && continue
      [ -e $T/init_s$S/latest_checkpoint.pth ] || continue
      python scratch_transition/measure_snapshot.py --checkpoint $f --init $T/init_s$S/latest_checkpoint.pth --out $out --batches 32 > $T/meas2/s${S}_$b.log 2>&1
      did=1
    done
  done
  [ -e $T/meas2/s$1_ckpt_060001.json ] && break
  [ $did = 0 ] && sleep 30
done
