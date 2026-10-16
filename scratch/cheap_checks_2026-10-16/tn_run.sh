#!/bin/bash
source ~/miniforge3/etc/profile.d/conda.sh; conda activate hebby
O=~/fastlayers_2026-10-13; E=$O/evals/tn; cd $O
one() { f=$1; r=$(basename $(dirname $f)); it=$(basename $f .pth | sed s/checkpoint_//)
  cd $O/code_m && TN_THREADS=3 TN_BATCHES=4 python ../tn_norms.py $f > $E/${r}_$it.jsonl 2> $E/${r}_$it.err; }
export -f one; export O E
ls runs_m/*/checkpoint_*.pth runs_v2/RGs0.1-s620101/checkpoint_*.pth runs_v2/RGs0.03-s620101/checkpoint_*.pth v2ck/Rbreak-*/checkpoint_*.pth | sed "s#^#$O/#" | xargs -P 16 -I{} bash -c "one {}"
cat $E/*.jsonl | grep "^{" > $O/evals/tn_all.jsonl; echo DONE > $E/done.txt
