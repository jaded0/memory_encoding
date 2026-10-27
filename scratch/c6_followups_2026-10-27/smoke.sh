#!/bin/bash
# Smoke tests for the C6 follow-up launchers (omen, RTX 4090; each step a few hundred iterations, < 2 GB). Not a result.
# Usage (from the code directory, hebby active): bash scratch/c6_followups_2026-10-27/smoke.sh STEP OUT_DIR [OLD_CODE_DIR]
#   equivM, equivA  M3 (task 17) / A3 (task 9 from scratch), 150 its with this code and with OLD_CODE_DIR (Batch FS's ORC code, a195879):
#          the printed losses must be identical, so the 600k runs continue Batch FS's runs on code that trains them the same way.
#   fu2    tasks 0, 3, 6 (A3f0, D3f0, M3f0) to 100 its, then the eval stage (50 and 100) on 2 loads x 2 batches, then collect_L.py.
#   ext    the follow-up 6 extension path: task 17 resumed from equivM's checkpoint at 50 (--resume_checkpoint, new dir) to 150 must print
#          the same losses as the uninterrupted run.
#   chunk  a chunked run: USR1 at ~iteration 75 through both forward_signals layers, a second chunk finishes at 150 with the same losses (from 125 on),
#          a spare third chunk trains nothing.
here=scratch/c6_followups_2026-10-27
step=$1; out=$2; old=$3
small="C6_KEEP=50 C6_SAVE=25 C6_PRINT=25 C6_TRACE=50"
losses() {  # "iteration:loss" for every print of a run (logparse: a resumed run's repeated print, the later wins), from FROM on
  python -c "import sys; sys.path.insert(0, '$here/../fast_structure_2026-10-22'); from logparse import intervals
print(' '.join(f'{i}:{m[\"loss\"]!r}' for i, m in intervals(sys.argv[1]) if i >= int(sys.argv[2])))" $1/train.log ${2:-0}
}
case $step in
equivM|equivA)
  [ $step = equivM ] && { t=17; n=L-M3-s4241; } || { t=9; n=L-A3-s2718; }
  env $small C6_TARGET=150 C6_INIT=none C6_RUNS=$out/new bash $here/run_c6.sh train $t
  (cd $old && cp -r $OLDPWD/$here scratch/ 2>/dev/null; env $small C6_TARGET=150 C6_INIT=none C6_RUNS=$out/old bash $here/run_c6.sh train $t)
  for n in $n; do
    echo "$n new: $(losses $out/new/$n)"; echo "$n old: $(losses $out/old/$n)"
    cmp <(losses $out/new/$n) <(losses $out/old/$n) && echo "$n IDENTICAL" || echo "$n DIFFER"
  done ;;
fu2)
  for t in 0 3 6; do env $small C6_TARGET=100 C6_RUNS=$out/L0 bash $here/run_c6.sh train $t; done
  for t in 0 3 6; do env $small C6_TARGET=100 C6_RUNS=$out/L0 C6_EVAL_ARGS="--loads 4 52 --batches 2" bash $here/run_c6.sh eval $t; done
  python $here/../fast_structure_2026-10-22/collect_L.py $out/L0 $out/L0/collect.jsonl
  python - $out/L0/collect.jsonl <<'PY'
import json, sys
for line in open(sys.argv[1]):
    r = json.loads(line)
    if r.get("strict") is None: continue
    eq = all(abs(r["read_only0"][k] - r["read_all"][k]) < 1e-12 for k in r["read_all"])
    print(r["run"], r["checkpoint"], "read_all == strict:", r["read_all_matches_strict"], "| only0 == all (deep layers slow):", eq,
          "| drive ratio per layer (L0, L1, L2):", r["fast_slow_drive"])
PY
  ;;
ext)
  src=$out/new/L-M3-s4241
  env $small C6_TARGET=150 C6_INIT=$src/checkpoint_00000050.pth C6_RUNS=$out/ext bash $here/run_c6.sh train 17
  echo "uninterrupted: $(losses $src 75)"; echo "resumed:       $(losses $out/ext/L-M3-s4241 75)"
  [ "$(losses $src 75)" = "$(losses $out/ext/L-M3-s4241 75)" ] && echo "EXTENSION IDENTICAL" || echo "EXTENSION DIFFERS" ;;
chunk)
  src=$out/new/L-M3-s4241
  source sweeps/forward_signals.sh
  (export $small C6_TARGET=150 C6_INIT=none C6_RUNS=$out/chunk; forward_signals bash $here/run_c6.sh train 17) &
  wrapper=$!
  until grep -q "ending @ iter 75," $out/chunk/L-M3-s4241/train.log 2>/dev/null; do sleep 2; done
  kill -USR1 $wrapper; wait $wrapper
  tail -2 $out/chunk/L-M3-s4241/c6_stages.log; grep "stopping before" $out/chunk/L-M3-s4241/train.log
  env $small C6_TARGET=150 C6_INIT=none C6_RUNS=$out/chunk bash $here/run_c6.sh train 17
  env $small C6_TARGET=150 C6_INIT=none C6_RUNS=$out/chunk bash $here/run_c6.sh train 17   # a spare chunk: must not train
  tail -3 $out/chunk/L-M3-s4241/c6_stages.log
  # the print after the stop averages a partial interval, so compare from the first whole interval of chunk 2 on
  echo "uninterrupted: $(losses $src 125)"; echo "chunked:       $(losses $out/chunk/L-M3-s4241 125)"
  [ "$(losses $src 125)" = "$(losses $out/chunk/L-M3-s4241 125)" ] && echo "CHUNKED IDENTICAL" || echo "CHUNKED DIFFERS" ;;
esac
