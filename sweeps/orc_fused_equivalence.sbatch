#!/bin/bash --login
# Does --fused_update change the training dynamics? Paired runs of the unfused and fused DFA step
# on 3-char reversal at lr * alpha = 3 (plasticity 3e3), the regime in which the fast weights
# learn (benchmarks/memory_tasks_head.md), for 200k iterations. Submit once per GPU type, so the
# unfused A100-vs-H100 difference calibrates what "same dynamics" means:
#   sbatch -p cs --export=ALL,GPU_LABEL=a100 sweeps/orc_fused_equivalence.sbatch
#   sbatch -p cs2 --export=ALL,GPU_LABEL=h100 sweeps/orc_fused_equivalence.sbatch
# The code is an exported copy of $COMMIT with a REVISION file checked below.

#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus=1
#SBATCH --mem=30G
#SBATCH --qos=cs
#SBATCH --requeue
#SBATCH --job-name=fusedeq
#SBATCH --array=0-5
#SBATCH --output=/home/jaded79/memory_encoding_benchmarks/fused_equivalence/logs/%x_%A_%a.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jaden.lorenc@gmail.com

set -eo pipefail

readonly RUN_ROOT=/home/jaded79/memory_encoding_benchmarks/fused_equivalence
readonly CODE=$RUN_ROOT/code
readonly COMMIT=6fb78e2
readonly SERIES=fused_equivalence_v1
readonly GPU_LABEL=${GPU_LABEL:?Set GPU_LABEL (a100 or h100) with --export}

readonly -a SEEDS=(2718 3141 4241)
readonly -a FUSED=(false true)
task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
seed=${SEEDS[$((task_id / 2))]:?No seed for array task $task_id}
fused=${FUSED[$((task_id % 2))]}
arm=$([[ $fused == true ]] && echo fused || echo unfused)_$GPU_LABEL
checkpoint_dir=$RUN_ROOT/checkpoints/$arm/seed-$seed
wandb_dir=$RUN_ROOT/wandb/$arm/seed-$seed

conda activate hebby
set -u
export WANDB_MODE=offline
export WANDB_EXECUTABLE=$CONDA_PREFIX/bin/python
export HF_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCHINDUCTOR_CACHE_DIR=${TMPDIR:-/tmp}/inductor_$SLURM_JOB_ID

echo "Array job: ${SLURM_ARRAY_JOB_ID}_${task_id}"
echo "Arm: $arm (fused_update $fused), seed $seed, code $COMMIT"
echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader

[[ $(cat "$CODE/REVISION" 2>/dev/null) == "$COMMIT"* ]] || {
    echo "Expected an export of $COMMIT in $CODE"
    exit 1
}
mkdir -p "$checkpoint_dir" "$wandb_dir"
export WANDB_DIR=$wandb_dir
export WANDB_NAME="fusedeq-$arm-s$seed"

cd "$CODE"
source sweeps/forward_signals.sh
set +e
forward_signals "$CONDA_PREFIX/bin/python" -u train.py \
    --model_type ephemeral \
    --updater dfa \
    --enable_recurrence false \
    --fused_update "$fused" \
    --dataset 3_palindrome_dataset_vary_length \
    --input_mode last_one \
    --learning_rate 1e-3 \
    --plasticity 3e3 \
    --forget_rate 0.01 \
    --ephemeral_fraction 0.2 \
    --ephemeral_update_clamp 0 \
    --grad_norm_clip 0 \
    --unit_norm_weights false \
    --weight_clamp 1 \
    --hidden_size 1024 \
    --num_layers 3 \
    --residual_connection false \
    --positional_encoding_dim 0 \
    --batch_size 16 \
    --n_iters 200000 \
    --print_freq 5000 \
    --checkpoint_dir "$checkpoint_dir" \
    --checkpoint_save_freq 25000 \
    --resume true \
    --track true \
    --group "$SERIES" \
    --tags fused_equivalence orc "$arm" "$COMMIT" "seed$seed" \
    --notes "Fused-update equivalence: series=$SERIES arm=$arm commit=$COMMIT seed=$seed array=${SLURM_ARRAY_JOB_ID}_${task_id}" \
    --seed "$seed" \
    --deterministic true
status=$?
set -e

if [[ $status == 124 ]]; then
    echo "Time-limit checkpoint saved; requeueing task $SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
elif [[ $status != 0 ]]; then
    echo "$arm failed with exit $status"
    exit "$status"
fi

echo "--- Finished $arm, seed $seed ---"
