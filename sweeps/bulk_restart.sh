#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1  # current_runs.txt and checkpoints/ live at the repo root

FILE="current_runs.txt"
[[ -f $FILE ]] || { echo "❌ $FILE not found"; exit 1; }

while IFS= read -r run || [[ -n $run ]]; do
  [[ -z $run ]] && continue                      # skip blanks
  if squeue -h -n "$run" | grep -q .; then
    echo "⏩  $run already in queue – skipped"
  else
    jobscript="./checkpoints/$run/run_used.sh"
    if [[ -x $jobscript ]]; then
      echo "➜  sbatch $jobscript"
      sbatch "$jobscript"
      sleep 10  # stagger submissions
    else
      echo "⚠️  $jobscript missing or not executable"
    fi
  fi
done < "$FILE"
