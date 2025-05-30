#!/usr/bin/env bash
set -euo pipefail

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
      sleep 1  # Add 3-second delay between submissions
    else
      echo "⚠️  $jobscript missing or not executable"
    fi
  fi
done < "$FILE"
