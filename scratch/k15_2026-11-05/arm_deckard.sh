#!/usr/bin/env bash
# From manticore only, after external preregistration and manifest approval.
# Starts idle waiters, never changes K13. Refuses any already-armed directory.
set -euo pipefail
ssh deckard 'bash -s' <<'REMOTE_ARM'
set -euo pipefail
root=$HOME/k15_2026-11-05
[[ -f $root/PREPARED && ! -e $root/ARMED && ! -s $root/status.txt ]] || exit 2
(( $(df --output=avail -B1 "$root" | tail -1) >= 100 * 1024**3 )) || exit 2
cd "$root/code"
~/miniforge3/envs/hebby/bin/python scratch/k15_2026-11-05/gate.py "$root/manifest.json"
# Refuse a used queue even if status was accidentally emptied.
~/miniforge3/envs/hebby/bin/python - "$root" <<'PY'
import json,pathlib,sys
r=pathlib.Path(sys.argv[1]); s=json.loads((r/'queue_state.json').read_text())
assert len(s['pending'])==20 and not any((s['active'],s['attempts'],s['done'],s['giveup']))
assert all(t.startswith('bigkey ') for t in s['pending'])
PY
date -Is > "$root/ARMED"
for i in 0 1 2 3; do
  setsid nohup bash "$root/code/scratch/k15_2026-11-05/deckard_lanes.sh" "$i" > "$root/logs/lane$i.log" 2>&1 < /dev/null &
  pid=$!
  printf 'lane%s pid%s %s\n' "$i" "$pid" "$(date -Is)" | tee -a "$root/lane_pids.txt"
done
REMOTE_ARM
