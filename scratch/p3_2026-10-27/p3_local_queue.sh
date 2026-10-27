#!/bin/bash
# Launch with a managed systemd user service; see README.md. One or two lanes only.
# Each lane is sequential; failures are recorded without preventing other arms running.
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate hebby
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export SG_THREADS=${SG_THREADS:-4}
out=${OUT:-$HOME/p3_2026-10-27/runs}
lanes=${CONCURRENCY:-2}
[[ $lanes = 1 || $lanes = 2 ]] || { echo 'CONCURRENCY must be 1 or 2'; exit 2; }
mkdir -p "$out/logs"
rm -f "$out/QUEUE_DONE" "$out/QUEUE_FAILED"
jobs=("$@")
lane() {
  local i arm seed rc failed=0
  for ((i=$1; i<${#jobs[@]}; i+=lanes)); do
    arm=${jobs[i]%%:*}; seed=${jobs[i]##*:}
    echo "$(date -Is) START $arm $seed lane $1" >> "$out/status.txt"
    if bash scratch/p3_2026-10-27/p3_train.sh "$arm" "$seed" "$out" "${N:-100000}" > "$out/logs/KR-$arm-s$seed.out" 2>&1; then
      rc=0
    else
      rc=$?; failed=1
    fi
    echo "$(date -Is) EXIT $rc $arm $seed lane $1" >> "$out/status.txt"
  done
  return "$failed"
}
pids=()
for ((l=0; l<lanes; l++)); do lane "$l" & pids+=("$!"); done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
python scratch/p3_2026-10-27/p3_collect.py "$out" || failed=1
if (( failed )); then touch "$out/QUEUE_FAILED"; else touch "$out/QUEUE_DONE"; fi
echo "$(date -Is) queue finished, failed=$failed"
exit "$failed"
