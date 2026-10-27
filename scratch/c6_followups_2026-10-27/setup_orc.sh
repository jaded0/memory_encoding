#!/bin/bash
# Put the code for the C6 follow-ups on ORC: ~/c6fu_2026-10-27/{code,logs,runs_L0,runs_F6}. Run on a host with the ORC ssh master
# (manticore), from a memory_encoding checkout: bash scratch/c6_followups_2026-10-27/setup_orc.sh [REF] (default: HEAD).
# Refuses to replace the code once any run directory has a train.log (a launched batch keeps its code).
set -e
ref=${1:-HEAD}; commit=$(git rev-parse $ref)
ssh orc "bash -lc 'ls ~/c6fu_2026-10-27/runs_*/*/train.log 2>/dev/null | head -1'" | grep -q train.log && { echo "runs exist: not replacing the code"; exit 1; }
git archive --format=tar $commit | ssh orc "rm -rf ~/c6fu_2026-10-27/code && mkdir -p ~/c6fu_2026-10-27/code ~/c6fu_2026-10-27/logs ~/c6fu_2026-10-27/runs_L0 ~/c6fu_2026-10-27/runs_F6 && tar -x -C ~/c6fu_2026-10-27/code && echo $commit > ~/c6fu_2026-10-27/code/COMMIT && ln -s ~/fslt_2026-10-22/synth_datasets ~/c6fu_2026-10-27/code/synth_datasets"
echo "ORC ~/c6fu_2026-10-27/code = $commit"
