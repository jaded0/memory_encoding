#!/bin/bash
# bfr_ratio.py on every KR run's numbered checkpoints (10k, and 30k for 30k runs), observed protocol, 16 batches. Usage: ratio_all.sh CODE_DIR RUNS_DIR OUT_DIR
here=$(cd $(dirname $0) && pwd); code=$1; runs=$2; out=$3; mkdir -p $out
cd $code
for d in $runs/kr_*; do
  n=$(basename $d)
  for c in $d/checkpoint_000[13]0000.pth; do
    [ -f "$c" ] || continue
    it=$(basename $c .pth | sed 's/checkpoint_000//; s/0000$/0k/; s/^0//')
    o=$out/ratio_${n}_${it}.json
    [ -s $o ] && continue
    python $here/bfr_ratio.py --checkpoint $c --batches 16 --json $o > /dev/null 2>&1 || echo "FAIL $n $it"
  done
done
echo RATIOS DONE
