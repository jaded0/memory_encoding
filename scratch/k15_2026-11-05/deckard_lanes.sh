#!/usr/bin/env bash
# One detached waiter per lane, 0..3; no launch/arming action in this script itself.
set -euo pipefail
lane=${1:?lane 0..3}
[[ $lane =~ ^[0-3]$ ]] || exit 2
g=$((lane % 2))
root=${K15_ROOT:-$HOME/k15_2026-11-05}
k13=$HOME/k13b_2026-10-28
source ~/miniforge3/etc/profile.d/conda.sh
conda activate hebby
export PATH="$HOME/miniforge3/bin:$PATH" CUDA_DEVICE_ORDER=PCI_BUS_ID
exec 9>"$root/lane$lane.lock"
flock -n 9 || { echo 'Refusing duplicate lane' >&2; exit 2; }
[[ ! -e $root/lane$lane.done ]] || { echo 'Lane already completed; never relaunch' >&2; exit 2; }
cd "$root/code"
scripts=$PWD/scratch/k15_2026-11-05
export K15_ROOT=$root
python "$scripts/gate.py" "$root/manifest.json"
# A failed nvidia-smi or unreadable live /proc entry blocks rather than falsely frees a GPU.
k13_gpu_clear() {
  local pids pid cmd cwd
  pids=$(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader) || return 1
  for pid in $pids; do
    [[ $pid =~ ^[0-9]+$ ]] || return 1
    [[ -d /proc/$pid ]] || continue
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline") || return 1
    cwd=$(readlink "/proc/$pid/cwd") || { [[ ! -d /proc/$pid ]] && continue; return 1; }
    [[ $cmd != *"$k13"* && $cwd != "$k13"/* ]] || return 1
  done
}
echo "WAIT lane$lane gpu$g $(date -Is)" >> "$root/status.txt"
until grep -Eq "^LANE DONE lane$lane [^ ]+$" "$k13/confirm_status.txt" && k13_gpu_clear; do sleep 300; done
echo "READY lane$lane gpu$g $(date -Is)" >> "$root/status.txt"
while true; do
  # Gate before claiming, so disk waiting does not reserve a task or weaken the barrier.
  waited=0
  while (( $(df --output=avail -B1 "$root" | tail -1) < 40 * 1024**3 )); do
    (( waited )) || echo "DISKWAIT-START lane$lane $(date -Is)" >> "$root/status.txt"
    waited=1; sleep 300
  done
  task=$(python "$scripts/k15_queue.py" claim --root "$root" --k13-status "$k13/confirm_status.txt" --lane "$lane")
  case "$task" in
    WAIT) sleep 60; continue;;
    DONE) break;;
  esac
  read -r phase arm seed <<< "$task"
  set +e
  CUDA_VISIBLE_DEVICES=$g bash "$scripts/train.sh" "$phase" "$arm" "$seed"
  rc=$?
  set -e
  python "$scripts/k15_queue.py" finish --root "$root" --k13-status "$k13/confirm_status.txt" --lane "$lane" --rc "$rc"
  if (( rc )); then sleep 300; fi
done
touch "$root/lane$lane.done"
echo "LANE DONE lane$lane $(date -Is)" >> "$root/status.txt"
