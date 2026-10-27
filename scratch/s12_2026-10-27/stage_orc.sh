#!/bin/bash
# Stage the S12 follow-up code on ORC (run on manticore from a checkout of memory_encoding). Usage:
#   bash scratch/s12_2026-10-27/stage_orc.sh [COMMIT]     (default: origin/s12-followups)
# Makes ~/s12_2026-10-27/{code,code_kr} on ORC = git archive COMMIT (COMMIT file inside), with synth_datasets linked as Batch BFR's:
#   code    -> ~/memory_encoding_data/current/synth_datasets   (3pal; = ~/bfr_2026-10-22/code_main)
#   code_kr -> ~/readnorm_2026-10-16/repo/synth_datasets       (RN's regenerated KR data; = ~/bfr_2026-10-22/code_kr)
# Refuses to overwrite an existing stage (runs must keep the code they started with) unless FORCE=1. Stages nothing else; launches nothing.
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
commit=$(git -C "$repo" rev-parse "${1:-origin/s12-followups}")
R='~/s12_2026-10-27'
if ssh orc "test -e $R/code" && [ "${FORCE:-0}" != 1 ]; then
  echo "ORC $R/code exists (COMMIT $(ssh orc "cat $R/code/COMMIT")); FORCE=1 to replace"; exit 1
fi
ssh orc "rm -rf $R/code $R/code_kr && mkdir -p $R/code $R/logs $R/runs $R/runs_kr $R/null $R/ratios"
git -C "$repo" archive "$commit" | ssh orc "tar -x -C $R/code"
ssh orc "cd $R && echo $commit > code/COMMIT && cp -a code code_kr \
  && ln -s ~/memory_encoding_data/current/synth_datasets code/synth_datasets \
  && ln -s ~/readnorm_2026-10-16/repo/synth_datasets code_kr/synth_datasets \
  && ls code/synth_datasets/3_palindrome_dataset_vary_length code_kr/synth_datasets/long_range_memory_dataset > /dev/null && echo STAGED \$(cat code/COMMIT)"
