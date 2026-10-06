#!/bin/bash --login
# Causal cap test arms (see the vault note "ephemeral weights causal cap test 2026-10-02").
# Submit: sbatch --export=ALL,CODE_COMMIT=<sha>,N_ITERS=210000 --array=0-5%4 causal_cap/orc_arms.sbatch
#SBATCH --time=0-06:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus=1
#SBATCH --mem=30G
#SBATCH --partition=cs,cs2
#SBATCH --qos=cs
#SBATCH --exclude=cs-1-5
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --job-name=ccap
#SBATCH --output=/home/jaded79/memory_encoding_benchmarks/causal_cap_2026-10-02/logs/%x_%A_%a.out
set -eo pipefail
readonly ROOT=/home/jaded79/memory_encoding_benchmarks/causal_cap_2026-10-02
readonly COMMIT=${CODE_COMMIT:?}
readonly CODE=$ROOT/code_$COMMIT
readonly -a ARMS=(CAP8 CTRL SHAM SHAM2 CAP16 SHAM3)
arm=${ARM:-${ARMS[${SLURM_ARRAY_TASK_ID:?}]}}
N_ITERS=${N_ITERS:-210000}
CKPT=${START_CKPT:-$ROOT/ckpt/checkpoint_00150000.pth}
D=$ROOT/runs/$arm
conda activate hebby
set -u
export WANDB_MODE=disabled HF_DATASETS_OFFLINE=1 HF_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d "${TMPDIR:-/tmp}/inductor_${SLURM_JOB_ID:-local}_XXXXXX")
cd "$CODE"
source causal_cap/flags.sh
arm_flags "$arm" "$ROOT/specs"
FLAGS=("${CTRL[@]}" "${COMMON[@]}" "${ARM_FLAGS[@]}" --n_iters "$N_ITERS" ${EXTRA_FLAGS:-})
mkdir -p "$D"
[[ -f $D/latest_checkpoint.pth ]] || FLAGS+=(--resume_checkpoint "$CKPT")
echo "=== arm $arm start $(date) host $(hostname) code $COMMIT"; nvidia-smi --query-gpu=name --format=csv,noheader
echo "=== flags: ${FLAGS[*]} --seed 3141"
source sweeps/forward_signals.sh
set +e
forward_signals "$CONDA_PREFIX/bin/python" -u train.py "${FLAGS[@]}" --seed 3141 --checkpoint_dir "$D" >> "$D/train.log" 2>&1
status=$?
set -e
echo "=== exit $status $(date)" >> "$D/train.log"
if [[ $status == 124 ]]; then scontrol requeue "$SLURM_JOB_ID"; exit 0; fi
exit $status
