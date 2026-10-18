#!/bin/bash
# Re-forward drift test queues: two jobs at a time per GPU, longest first, balanced by estimated GPU time.
cd ~/reforward_2026-10-16
printf "%s\n" "4 620101" "4 620203" "2 620101" "1 620101" | xargs -P 2 -L 1 bash -c "./rf_run.sh \$0 \$1 0" &
printf "%s\n" "4 620307" "2 620203" "2 620307" "1 620203" "1 620307" | xargs -P 2 -L 1 bash -c "./rf_run.sh \$0 \$1 1" &
wait
echo "$(date +%F\ %T) QUEUES DONE" >> runs/status.txt
