#!/bin/bash
# After pull_ckpts.sh: ratios for the ORC KR runs (KR data, code/) and the 3pal LC + F7 rms checkpoints (ORC's 3pal data, code3/). Runs on manticore.
source ~/miniforge3/etc/profile.d/conda.sh; conda activate hebby
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-0}
B=~/bfr_kr_2026-10-22; S=$B/scripts; O=$B/ratios
until grep -q "PULL DONE" $S/pull.log; do sleep 30; done
$S/ratio_all.sh $B/code $B/orc_ckpt $O
cd $B/code3
for d in $B/orc_ckpt/BFR-LC-* $B/orc_ckpt/Q-F7-RMS-*; do
  for k in 00150000 00600000; do
    c=$d/checkpoint_$k.pth; o=$O/ratio_$(basename $d)_${k:2:3}k.json
    if [ -f $c ] && [ ! -s $o ]; then python $S/bfr_ratio.py --checkpoint $c --batches 16 --json $o > /dev/null 2>&1 || echo "FAIL $c"; fi
  done
done
echo MANTICORE RATIOS DONE
