#!/bin/bash
# run_full_arm.sh ARM GPU [START_CKPT]   (Deckard). Root ~/causal_cap_2026-10-02
ROOT=$HOME/causal_cap_2026-10-02; ARM=$1; GPU=$2; CKPT=${3:-$ROOT/ckpt/checkpoint_00150000.pth}
source ~/miniforge3/etc/profile.d/conda.sh; conda activate hebby
cd $ROOT/code; source causal_cap/flags.sh; source causal_cap/flags_full.sh
arm_flags_full $ARM $ROOT/specs_norm || exit 2
D=$ROOT/runs/$ARM; mkdir -p $D
FLAGS=("${CTRL[@]}" "${COMMON_FULL[@]}" "${ARM_FLAGS[@]}")
[ -f $D/latest_checkpoint.pth ] || FLAGS+=(--resume_checkpoint $CKPT)
export CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled HF_DATASETS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d /tmp/inductor_ccf_XXXXXX)
echo "=== arm $ARM start $(date) gpu $GPU" >> $D/train.log
echo "=== flags: ${FLAGS[*]} --seed 3141" >> $D/train.log
python -u train.py "${FLAGS[@]}" --seed 3141 --checkpoint_dir $D >> $D/train.log 2>&1
echo "=== exit $? $(date)" >> $D/train.log
rm -rf $TORCHINDUCTOR_CACHE_DIR
