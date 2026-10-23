#!/bin/bash
# Copy ORC checkpoints to manticore for the ratio measurement (the ORC queue was hours deep). Runs on manticore.
dst=~/bfr_kr_2026-10-22/orc_ckpt
pull() { mkdir -p $dst/$2; [ -s $dst/$2/$3 ] || ssh orc "cat $1/$3" > $dst/$2/$3 || rm -f $dst/$2/$3; }
for r in $(ssh orc "ls ~/bfr_2026-10-22/runs_kr"); do
  for k in checkpoint_00010000.pth checkpoint_00030000.pth; do
    ssh orc "test -f ~/bfr_2026-10-22/runs_kr/$r/$k" && pull "~/bfr_2026-10-22/runs_kr/$r" $r $k
  done
done
for r in BFR-LC-ln-s620101 BFR-LC-ln-s620307 BFR-LC-ln-s620411 BFR-LC-ln-s620203 BFR-LC-ln-s620509 \
         BFR-LC-absrun-s620101 BFR-LC-absrun-s620203 BFR-LC-absrun-s620307 BFR-LC-absrun-s620411 BFR-LC-absrun-s620509; do
  for k in checkpoint_00150000.pth checkpoint_00600000.pth; do
    ssh orc "test -f ~/bfr_2026-10-22/runs/$r/$k" && pull "~/bfr_2026-10-22/runs/$r" $r $k
  done
done
for s in 620101 620203 620307; do
  for k in checkpoint_00150000.pth checkpoint_00600000.pth; do pull "~/batchQ_2026-10-17/runs/Q-F7-RMS-s$s" Q-F7-RMS-s$s $k; done
done
echo PULL DONE
