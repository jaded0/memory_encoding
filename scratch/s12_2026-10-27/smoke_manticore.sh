#!/bin/bash
# Smoke test of the three Batch S12F training launchers on manticore (GPU ${GPU:-1}), each run as the real sbatch file with BASE pointed
# at a local stage and a short N_ITERS (task 0 of each array; absrun_floor tasks 0 and 3 cover c = 10 and c = 100).
# Usage: bash smoke_manticore.sh CODE_TAR   (CODE_TAR = git archive of the commit under test)
# Local stage: ~/s12_smoke/{code,code_kr} with the 3pal data of ~/bfr_kr_2026-10-22/code3 and RN's KR data.
set -euo pipefail
here=$(cd $(dirname $0) && pwd); tar_=$1
S=${SMOKE:-~/s12_smoke}; rm -rf $S/code $S/code_kr $S/runs $S/runs_kr $S/ratios; mkdir -p $S/code
tar -x -C $S/code -f $tar_; echo "smoke $(date -Iseconds)" > $S/code/COMMIT; cp -a $S/code $S/code_kr
ln -s ~/bfr_kr_2026-10-22/code3/synth_datasets $S/code/synth_datasets
ln -s /home/jaden/readnorm_2026-10-16/repo/synth_datasets $S/code_kr/synth_datasets
export BASE=$S CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-1}
run() {  # name, file, task, iters
  local t0=$(date +%s)
  SLURM_ARRAY_TASK_ID=$3 N_ITERS=$4 CKPT_EVERY=${SMOKE_CKPT_EVERY:-1000} timeout 600 bash $here/$2 > $S/smoke_$1.log 2>&1
  echo "$1: exit $? in $(( $(date +%s) - t0 )) s"
}
run lndc2 s12_lndc2_lc.sbatch 0 ${SMOKE_ITERS:-2000}
run afl_lc_c10 s12_absfloor_lc.sbatch 0 ${SMOKE_ITERS:-2000}
run afl_lc_c100 s12_absfloor_lc.sbatch 3 ${SMOKE_ITERS:-2000}
run afl_kr_c10 s12_absfloor_kr.sbatch 0 ${SMOKE_ITERS:-2000}
run afl_kr_c100 s12_absfloor_kr.sbatch 3 ${SMOKE_ITERS:-2000}
echo SMOKE DONE
