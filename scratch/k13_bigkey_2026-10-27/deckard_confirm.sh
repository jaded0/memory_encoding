#!/usr/bin/env bash
# Deckard lane for the 3-arm K13bigkey confirmation (addendum 2026-10-29, gate profile deckard3).
# Usage: deckard_confirm.sh LANE   (lanes 0-3; GPU = LANE % 2). Run detached (setsid nohup).
# A lane waits until its GPU has no non-K13 compute process (the B2fu-A runs), then pulls
# tasks from a shared flock'd queue, longest arm first, until the queue is empty.
# Addendum 2026-11-02: before each task the lane waits for K13B_START_FREE_GB free on the run
# disk; a failed run (not a timeout) is put back on the queue and resumed from its latest
# checkpoint, at most K13B_MAX_ATTEMPTS times in all. Each event is a line in confirm_status.txt.
set -eo pipefail
lane=${1:?lane}; g=$((lane % 2))
source ~/miniforge3/etc/profile.d/conda.sh; conda activate hebby
root=${K13B_ROOT:-$HOME/k13b_2026-10-28}
code=${K13B_CODE_DIR:-$root/code_d4}  # clean checkout of the addendum-2026-11-02 commit
cd "$code"
export K13B_CODE=$PWD K13B_COMMIT=$(git rev-parse HEAD) K13B_GPU=A6000 K13B_PILOT=0
export K13B_GATE_MANIFEST=$root/confirm_manifest.json
export K13B_RESUME=1 K13B_KEEP_EVERY=${K13B_KEEP_EVERY:-25000} K13B_MIN_FREE_GB=${K13B_MIN_FREE_GB:-15}
start_free_gb=${K13B_START_FREE_GB:-40} max_attempts=${K13B_MAX_ATTEMPTS:-3}
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # match nvidia-smi -i numbering
python scratch/k13_bigkey_2026-10-27/gate.py validate "$K13B_GATE_MANIFEST"
read -r data n < <(python -c "import json,sys; m=json.load(open(sys.argv[1])); print(m['dataset'], m['n_iters'])" "$K13B_GATE_MANIFEST")
queue=$root/confirm_queue.txt status=$root/confirm_status.txt

foreign() {  # PIDs on GPU g that are not confirmation runs
  local pid
  for pid in $(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader); do
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "$root/confirm/" || echo "$pid"
  done
}
echo "WAIT lane$lane gpu$g $(date -Is)" >> "$status"
while [[ -n $(foreign) ]]; do sleep 600; done

while true; do
  task=$(flock "$queue.lock" bash -c 'head -n1 "$1"; sed -i 1d "$1"' _ "$queue")
  [[ -n $task ]] || break
  read -r arm seed <<< "$task"
  waited=0
  while (( $(df --output=avail -B1G "$root" | tail -1) < start_free_gb )); do
    (( waited )) || echo "DISKWAIT-START $arm s$seed lane$lane $(date -Is)" >> "$status"
    waited=1; sleep 600
  done
  attempt=$(( $(grep -c "^START $arm s$seed " "$status" || true) + 1 ))
  echo "START $arm s$seed lane$lane gpu$g attempt$attempt $(date -Is)" >> "$status"
  set +e
  CUDA_VISIBLE_DEVICES=$g bash scratch/k13_bigkey_2026-10-27/train.sh "$data" "$n" "$arm" "$seed" "$root/confirm"
  rc=$?
  set -e
  echo "EXIT $rc $arm s$seed lane$lane $(date -Is)" >> "$status"
  if [[ $rc != 0 && $rc != 124 ]]; then
    if (( attempt < max_attempts )); then
      flock "$queue.lock" bash -c 'printf "%s\n" "$2" >> "$1"' _ "$queue" "$arm $seed"
      echo "REQUEUE $arm s$seed after attempt$attempt $(date -Is)" >> "$status"
      sleep 300
    else
      echo "GIVEUP $arm s$seed after attempt$attempt $(date -Is)" >> "$status"
    fi
  fi
done
echo "LANE DONE lane$lane $(date -Is)" >> "$status"
