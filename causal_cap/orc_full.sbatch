#!/bin/bash --login
# Full-spectrum norm-cap arm on ORC (A100 only). sbatch --export=ALL,CODE_COMMIT=<sha>,ARM=FROB[,START_CKPT=...] causal_cap/orc_full.sbatch
#SBATCH --time=0-06:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus=1
#SBATCH --mem=30G
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --exclude=cs-1-5
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --job-name=ccapf
#SBATCH --output=/home/jaded79/memory_encoding_benchmarks/causal_cap_2026-10-02/logs/%x_%A.out
set -eo pipefail
ROOT=/home/jaded79/memory_encoding_benchmarks/causal_cap_2026-10-02
CODE=$ROOT/code_${CODE_COMMIT:?}
ARM=${ARM:?}; CKPT=${START_CKPT:-$ROOT/ckpt/checkpoint_00150000.pth}
conda activate hebby
set -u
export WANDB_MODE=disabled HF_DATASETS_OFFLINE=1 HF_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d "${TMPDIR:-/tmp}/inductor_${SLURM_JOB_ID:-local}_XXXXXX")
cd $CODE; source causal_cap/flags.sh; source causal_cap/flags_full.sh
arm_flags_full $ARM $ROOT/specs_norm
D=$ROOT/runs/$ARM; mkdir -p $D
FLAGS=("${CTRL[@]}" "${COMMON_FULL[@]}" "${ARM_FLAGS[@]}")
[[ -f $D/latest_checkpoint.pth ]] || FLAGS+=(--resume_checkpoint "$CKPT")
echo "=== arm $ARM start $(date) $(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader
echo "=== flags: ${FLAGS[*]} --seed 3141" | tee -a $D/train.log
source sweeps/forward_signals.sh
set +e
forward_signals "$CONDA_PREFIX/bin/python" -u train.py "${FLAGS[@]}" --seed 3141 --checkpoint_dir "$D" >> "$D/train.log" 2>&1
st=$?; set -e
echo "=== exit $st $(date)" >> $D/train.log
if [[ $st == 124 ]]; then scontrol requeue $SLURM_JOB_ID; exit 0; fi
exit $st
