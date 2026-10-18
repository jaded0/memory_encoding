#!/bin/bash
# Re-forward drift test analysis on deckard (PREREG "Re-forward drift test (2026-10-16)"). usage: rf_analyze.sh
# collect (onsets, held-out traces, final evals), drift_extra (slow top SV every 25k), stepgain2 (readout share in range(B_2)) at
# 50k/100k/150k, LV (K-aware) at 50k/100k/150k and at the last 25k checkpoint before each late onset.
source ~/miniforge3/etc/profile.d/conda.sh; conda activate hebby
R=~/reforward_2026-10-16; O=~/fastlayers_2026-10-13; E=$R/evals; mkdir -p $E/sg $E/lv; cd $R/code
python $R/rf_collect.py $R/runs/[KW]*/ > $E/rf_collect.jsonl
DX_THREADS=8 python $O/drift_extra.py $(ls -d $R/runs/[KW]*/) > $E/rf_dx.jsonl 2> $E/rf_dx.err
for d in $R/runs/[KW]*/; do r=$(basename $d)
  for it in 00050000 00100000 00150000; do f=$d/checkpoint_$it.pth; [ -e $f ] && echo $f; done
done > $E/ck_main.txt
python - $E/rf_collect.jsonl >> $E/ck_main.txt <<'PY'
import json, sys, os
for line in open(sys.argv[1]):
    r = json.loads(line)
    if r["onset"]:
        it = (r["onset"] - 1) // 25000 * 25000
        p = os.path.expanduser(f"~/reforward_2026-10-16/runs/{r['run']}/checkpoint_{it:08d}.pth")
        if it > 0 and os.path.exists(p): print(p)
PY
sort -u $E/ck_main.txt -o $E/ck_main.txt
one() { f=$1; r=$(basename $(dirname $f)); it=$(basename $f .pth | sed s/checkpoint_//)
  SG_THREADS=4 python $O/stepgain2.py $f > $E/sg/${r}_$it.jsonl 2> $E/sg/${r}_$it.err; }
export -f one; export O E
cat $E/ck_main.txt | xargs -P 6 -I{} bash -c "one {}"
cat $E/sg/*.jsonl | grep "^{" > $E/rf_sg.jsonl
CUDA_VISIBLE_DEVICES=0 LV_BATCHES=4 python $R/lv_measure_k.py $(sed -n '1~2p' $E/ck_main.txt) > $E/lv/lv_a.jsonl 2> $E/lv/err_a.log &
CUDA_VISIBLE_DEVICES=1 LV_BATCHES=4 python $R/lv_measure_k.py $(sed -n '2~2p' $E/ck_main.txt) > $E/lv/lv_b.jsonl 2> $E/lv/err_b.log &
wait; cat $E/lv/lv_a.jsonl $E/lv/lv_b.jsonl > $E/rf_lv.jsonl
echo "$(date +%F\ %T) DONE" > $E/analyze_done.txt
