#!/bin/bash
# forward_signals.sh - let a training run receive SLURM's pre-timeout warning.
#
# Source this from an sbatch script that has "#SBATCH --signal=B:USR1@600", then launch with:
#     source ./forward_signals.sh
#     forward_signals python -u train.py ...
#
# "--signal=B:USR1@600" sends USR1 only to the batch shell itself, 600 s before the wall-time limit.
# forward_signals sets traps in that shell (which is why this file is sourced, not executed), runs the
# command in the background, forwards USR1/TERM to it, keeps waiting while it checkpoints and exits, and
# returns its exit status. train.py treats USR1 as "time limit approaching": it stops at the next
# iteration, saves a checkpoint (if --checkpoint_save_freq > 0), records end_reason=time_limit in W&B,
# and exits with code 124.

forward_signals() {
    "$@" &
    local child=$!
    trap "kill -USR1 $child 2>/dev/null" USR1
    trap "kill -TERM $child 2>/dev/null" TERM

    # wait returns early when a trapped signal arrives; keep waiting until the child has exited.
    local status
    wait "$child"
    status=$?
    while kill -0 "$child" 2>/dev/null; do
        wait "$child"
        status=$?
    done
    trap - USR1 TERM
    return "$status"
}
