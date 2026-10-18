#!/bin/bash
# ADDENDUM 1 queues. K = 1 x 3 start now (sharing GPUs with the from-scratch runs until those stop at 100k); warm-start runs
# each wait for their K = 1 50k checkpoint, then run. Two jobs per GPU once the from-scratch runs have stopped.
cd ~/reforward_2026-10-16
(./rf_run.sh 1 620101 0; ./rf_warm.sh 4 620203 0) &
(./rf_run.sh 1 620203 1; ./rf_warm.sh 2 620307 1) &
(./rf_run.sh 1 620307 1; ./rf_warm.sh 4 620307 0) &
(./rf_warm.sh 4 620101 0; ./rf_warm.sh 2 620203 1) &
(./rf_warm.sh 2 620101 1) &
wait
echo "$(date +%F\ %T) QUEUES DONE" >> runs/status.txt
