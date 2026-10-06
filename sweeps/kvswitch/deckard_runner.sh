#!/bin/bash
# Waiting runner: one kvswitch arm at a time from a queue file, only on a Deckard GPU that is
# below 30% utilisation (mean of 5 samples over 10 s) with under 8 GB in use; polls every 120 s.
# Stops (between arms) when $ROOT/STOP exists or free disk falls below 40 GB.
# usage: setsid nohup sweeps/kvswitch/deckard_runner.sh QUEUE > $ROOT/runner.log 2>&1 &
#   QUEUE lines: NAME DATASET SEED FORGET PLASTICITY ITERS [WIPE_EVERY [FAST_CLAMP]]; '#' comments.
# ONLY_GPU=i restricts the runner to GPU i (one runner per GPU when several run at once).
# Finished arms are appended to $ROOT/done.txt and skipped on a restart.
set -uo pipefail
QUEUE=$1
ROOT=${ROOT:-$HOME/kvswitch_2026-10-02}
CODE=${CODE:-$ROOT/code}
POLL=${POLL:-120}
MIN_FREE_GB=${MIN_FREE_GB:-40}
touch "$ROOT/done.txt"
log() { echo "$(date '+%F %T') $*"; }

idle_gpu() {  # prints the index of a GPU below 30% mean utilisation and 8 GB used, else nothing
    local samples
    samples=$(for i in 1 2 3 4 5; do
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits; sleep 2; done)
    echo "$samples" | awk -F', ' -v only="${ONLY_GPU:-}" '{u[$1]+=$2; m[$1]=($3>m[$1]?$3:m[$1]); n[$1]++}
        END {for (g in u) if ((only == "" || g == only) && u[g]/n[g] < 30 && m[g] < 8000) {print g; exit}}'
}

while read -r name dataset seed f alpha iters wipe clamp; do
    [[ -z ${name:-} || $name == \#* ]] && continue
    grep -qx "$name" "$ROOT/done.txt" && { log "skip $name (done)"; continue; }
    while true; do
        [[ -e $ROOT/STOP ]] && { log "STOP file: exiting before $name"; exit 0; }
        free_gb=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9)
        (( free_gb < MIN_FREE_GB )) && { log "only ${free_gb} GB free (< $MIN_FREE_GB): exiting"; exit 1; }
        gpu=$(idle_gpu)
        [[ -n $gpu ]] && break
        log "no idle GPU for $name; waiting ${POLL}s"
        sleep "$POLL"
    done
    log "start $name on GPU $gpu ($dataset seed $seed f $f alpha $alpha iters $iters wipe ${wipe:-1024} clamp ${clamp:-1})"
    if GPU=$gpu ROOT=$ROOT CODE=$CODE bash "$CODE/sweeps/kvswitch/run_arm.sh" \
            "$name" "$dataset" "$seed" "$f" "$alpha" "$iters" ${wipe:-} ${clamp:-}; then
        echo "$name" >> "$ROOT/done.txt"; log "finished $name"
    else
        log "FAILED $name (exit $?); see $ROOT/logs/$name.log"
    fi
done < "$QUEUE"
log "queue empty"
