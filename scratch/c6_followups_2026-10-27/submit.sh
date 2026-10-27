#!/bin/bash
# Submit C6 follow-up tasks on ORC as CHUNKS chained one-day training arrays plus one evaluation array.
# Usage (on ORC, from ~/c6fu_2026-10-27/code): bash scratch/c6_followups_2026-10-27/submit.sh TASKS CHUNKS [--test-only]
#   follow-up 2: submit.sh 0-8 3      (300k from scratch: ~21-26 A100-h per run, so 2 chunks plus a spare)
#   follow-up 6: submit.sh 9-18 4     (9-16 extend 300k -> 600k in 1-2 chunks; 17-18 are 600k from scratch, ~50-55 h: 3 chunks plus a spare)
# A chunk whose run is already finished exits within a minute. --dependency=aftercorr: task i of chunk k+1 starts when task i of
# chunk k ends OK (the train script always exits 0 unless the node fails or the hard limit kills it; then resubmit with this script:
# every run resumes from its own latest_checkpoint.pth). --test-only: sbatch --test-only on the first chunk only, nothing queued.
set -e
tasks=$1; chunks=${2:-3}; here=scratch/c6_followups_2026-10-27
[ -n "$tasks" ] || { echo "usage: submit.sh TASKS CHUNKS [--test-only]"; exit 2; }
[ -f COMMIT ] || { echo "run from ~/c6fu_2026-10-27/code (setup_orc.sh writes COMMIT)"; exit 2; }
mkdir -p ../logs
if [ "$3" = --test-only ]; then
  sbatch --test-only --array=$tasks $here/c6_train.sbatch
  sbatch --test-only --array=$tasks $here/c6_eval.sbatch
  exit 0
fi
jid=$(sbatch --parsable --array=$tasks $here/c6_train.sbatch); echo "chunk 1: $jid"; all=$jid
for ((c = 2; c <= chunks; c++)); do
  jid=$(sbatch --parsable --array=$tasks --dependency=aftercorr:$jid $here/c6_train.sbatch); echo "chunk $c: $jid"; all="$all $jid"
done
ejid=$(sbatch --parsable --array=$tasks --dependency=aftercorr:$jid $here/c6_eval.sbatch); echo "eval: $ejid"
echo "$(date -Is) tasks $tasks code $(cat COMMIT) train $all eval $ejid" >> ../submitted.log
