#!/bin/bash --login
# Missing factorial arm: old sibling-head layout with default-initialized slow weights.

#SBATCH --time=1-00:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus=1
#SBATCH --mem=30G
#SBATCH --partition=m9g,m13l,m13h
#SBATCH --qos=gpu
#SBATCH --requeue
#SBATCH --job-name=3pal_legacy_default
#SBATCH --array=0-7%5
#SBATCH --output=/home/jaded79/memory_encoding_benchmarks/orc_3pal_250k_legacy_default/logs/%x_%A_%a.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jaden.lorenc@gmail.com

set -eo pipefail

readonly WORKTREE=/home/jaded79/memory_encoding_benchmark_worktrees/legacy_default_aef0697
readonly RUN_ROOT=/home/jaded79/memory_encoding_benchmarks/orc_3pal_250k_legacy_default
readonly DATA_ROOT=/home/jaded79/memory_encoding_benchmark_data/synth_datasets
readonly SERIES=3pal_250k_multiseed_v1
readonly BASE_COMMIT=aef0697
readonly MODEL_SHA256=2391a040f1eb0e16871f385f2760dba752b32d2e18799bed98d4a3b51ea8d82f

readonly -a SEEDS=(1729 2718 3141 4241 5153 8677 10007 65537)

task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
seed=${SEEDS[$task_id]:?No seed for array task $task_id}
checkpoint_dir=$RUN_ROOT/checkpoints/legacy_default/seed-$seed
wandb_dir=$RUN_ROOT/wandb/legacy_default/seed-$seed

conda activate hebby
set -u
export WANDB_MODE=offline
export WANDB_EXECUTABLE=$CONDA_PREFIX/bin/python
export HF_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "Array job: ${SLURM_ARRAY_JOB_ID}_${task_id}"
echo "Seed: $seed"
echo "Configuration: legacy_default ($BASE_COMMIT plus default-init patch)"
echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader

[[ -d $WORKTREE ]] || { echo "Missing worktree: $WORKTREE"; exit 1; }
actual_commit=$(git -C "$WORKTREE" rev-parse --short=7 HEAD)
[[ $actual_commit == "$BASE_COMMIT" ]] || {
    echo "Wrong base revision in $WORKTREE: expected $BASE_COMMIT, found $actual_commit"
    exit 1
}
actual_model_sha256=$(sha256sum "$WORKTREE/ephemeral_model.py" | cut -d' ' -f1)
[[ $actual_model_sha256 == "$MODEL_SHA256" ]] || {
    echo "Wrong initialization patch in $WORKTREE: expected $MODEL_SHA256, found $actual_model_sha256"
    exit 1
}
[[ -d $WORKTREE/synth_datasets/3_palindrome_dataset_vary_length ]] || {
    echo "Missing shared 3-palindrome dataset in $WORKTREE"
    exit 1
}

mkdir -p "$checkpoint_dir" "$wandb_dir"
export WANDB_DIR=$wandb_dir
export WANDB_NAME="3pal-legacy_default-orc-s$seed"

echo "--- Starting legacy_default at $BASE_COMMIT plus default-init patch, seed $seed ---"
cd "$WORKTREE"
source sweeps/forward_signals.sh
set +e
forward_signals "$CONDA_PREFIX/bin/python" -u train.py \
    --model_type ephemeral \
    --updater dfa \
    --input_mode last_one \
    --learning_rate 1e-3 \
    --plasticity 1e3 \
    --ephemeral_update_clamp 0 \
    --forget_rate 0.01 \
    --unit_norm_weights false \
    --weight_clamp 1 \
    --hidden_size 1024 \
    --num_layers 3 \
    --residual_connection false \
    --positional_encoding_dim 0 \
    --dataset 3_palindrome_dataset_vary_length \
    --batch_size 16 \
    --n_iters 250000 \
    --print_freq 5000 \
    --checkpoint_dir "$checkpoint_dir" \
    --checkpoint_save_freq 5000 \
    --resume true \
    --track true \
    --group "$SERIES" \
    --tags benchmark multiseed orc legacy_default "$BASE_COMMIT" init_patch_8e9b45f "seed$seed" \
    --notes "Missing factorial arm: series=$SERIES config=legacy_default base=$BASE_COMMIT init_patch=8e9b45f model_sha256=$MODEL_SHA256 seed=$seed array=${SLURM_ARRAY_JOB_ID}_${task_id}" \
    --ephemeral_fraction 0.2 \
    --enable_recurrence false \
    --seed "$seed" \
    --deterministic true \
    --self_grad 0
status=$?
set -e

if [[ $status == 124 ]]; then
    echo "Time-limit checkpoint saved; requeueing task $SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
elif [[ $status != 0 ]]; then
    echo "legacy_default failed with exit $status"
    exit "$status"
fi

echo "--- Finished legacy_default at $BASE_COMMIT plus default-init patch, seed $seed ---"
