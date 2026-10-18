#!/bin/bash
# ADDENDUM 1: stop the from-scratch K >= 2 runs once their 100k checkpoint is written; rf_run.sh then runs the final
# held-out eval on that checkpoint (latest_checkpoint.pth) and logs EXIT.
R=~/reforward_2026-10-16
for j in K4-s620101 K4-s620203 K4-s620307 K2-s620203; do
  ( f=$R/runs/$j/checkpoint_00100000.pth
    until [ -e $f ] && [ $(( $(date +%s) - $(stat -c %Y $f) )) -gt 60 ]; do sleep 60; done
    pkill -f -- "--checkpoint_dir $R/runs/$j\$" ; pkill -f -- "--checkpoint_dir $R/runs/$j "
    echo "$(date +%F\ %T) STOPPED at 100k $j" >> $R/runs/status.txt ) &
done
wait
