#!/bin/bash
# Stage a code snapshot for Batch F9 / S7 on a remote host (run on manticore, from the repo).
# Usage: bash scratch/f9_s7_2026-10-27/stage_code.sh SSH_HOST REMOTE_CODE_DIR DATA_DIR [COMMIT]
#   ORC F9:   stage_code.sh orc /home/jaded79/f9s7_2026-10-27/code_v2data /home/jaded79/memory_encoding_data/current/synth_datasets
#   ORC S7D:  stage_code.sh orc /home/jaded79/f9s7_2026-10-27/code_batterydata /home/jaded79/battery_2026-10/datasets
#   deckard:  stage_code.sh jaden@deckard /home/jaden/f9s7_2026-10-27/code /home/jaden/loop_trace_2026-09-30/code/synth_datasets
# Writes CODE_COMMIT (read by run_arm.sh / s7_r4ext.sh into train.log), links synth_datasets to DATA_DIR, makes ../logs and ../runs.
set -eu
host=$1; dir=$2; data=$3; commit=${4:-HEAD}
sha=$(git rev-parse "$commit")
git archive "$sha" | ssh "$host" "mkdir -p $dir && tar -x -C $dir && echo $sha > $dir/CODE_COMMIT && ln -sfn $data $dir/synth_datasets \
  && ls $dir/synth_datasets/3_palindrome_dataset_vary_length/train > /dev/null && mkdir -p $dir/../logs $dir/../runs && echo staged $sha on $host:$dir"
