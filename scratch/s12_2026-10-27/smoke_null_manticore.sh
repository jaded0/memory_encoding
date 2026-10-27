#!/bin/bash
# Smoke test of ln_null_eval.py on one local checkpoint (manticore, GPU ${GPU:-1}): round-1 KR plain ln s1414 at 30k
# (a drifted seed), 32 held-out batches. Code = ~/bfr_kr_2026-10-22/code (git archive b26e88d with RN's KR data).
here=$(cd $(dirname $0) && pwd)
source ~/miniforge3/etc/profile.d/conda.sh; conda activate hebby
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-1}
out=${OUT:-~/s12_smoke}; mkdir -p $out
cd ~/bfr_kr_2026-10-22/code
start=$(date +%s)
python $here/ln_null_eval.py --checkpoint ~/bfr_kr_2026-10-22/runs/kr_plain_ln_s1414/checkpoint_00030000.pth \
  --batches ${BATCHES:-32} --json $out/null_kr_plain_ln_s1414_30k_b${BATCHES:-32}.json
echo "EXIT $? after $(( $(date +%s) - start )) s"
