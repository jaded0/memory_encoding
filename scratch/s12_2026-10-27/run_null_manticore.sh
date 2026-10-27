#!/bin/bash
# S12 follow-up 2, the part on manticore: ln_null_eval.py on round-1 plain-ln KR checkpoints that exist only here (10k, 20k; 30k too,
# as a cross-check against ORC's copy). Usage: GPU=1 bash run_null_manticore.sh   (~3 min per checkpoint on the 2080 Ti at 128 batches)
here=$(cd $(dirname $0) && pwd)
source ~/miniforge3/etc/profile.d/conda.sh; conda activate hebby
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-1}
O=${OUT:-~/s12_null}; mkdir -p $O
cd ~/bfr_kr_2026-10-22/code   # git archive b26e88d + RN's KR data; ln_null_eval.py needs only main >= b26e88d
for s in 2718 3141 1618 1414 5772; do
  n=kr_plain_ln_s$s
  for k in 10 20 30; do
    c=~/bfr_kr_2026-10-22/runs/$n/checkpoint_000${k}000.pth; o=$O/null_${n}_${k}k
    [ -f $c ] || { echo "MISSING $c"; continue; }
    [ -s $o.json ] && continue
    python -u $here/ln_null_eval.py --checkpoint $c --batches ${BATCHES:-128} --json $o.json > $o.log 2>&1
    echo "$n ${k}k EXIT $?"
  done
done
echo MANTICORE NULL DONE
