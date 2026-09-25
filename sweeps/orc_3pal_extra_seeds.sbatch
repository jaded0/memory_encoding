#!/bin/bash --login
# Five additional paired seeds for the 3-palindrome commit-chain benchmark.
# Each array task runs one configuration so SLURM can use any available GPU.

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
#SBATCH --job-name=3pal_extra5
#SBATCH --array=0-14%5
#SBATCH --output=/home/jaded79/memory_encoding_benchmarks/orc_3pal_250k_extra5/logs/%x_%A_%a.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jaden.lorenc@gmail.com

set -eo pipefail

readonly WORKTREE_ROOT=/home/jaded79/memory_encoding_benchmark_worktrees
readonly RUN_ROOT=/home/jaded79/memory_encoding_benchmarks/orc_3pal_250k_extra5
readonly DATA_ROOT=/home/jaded79/memory_encoding_benchmark_data/synth_datasets
readonly SERIES=3pal_250k_multiseed_v1

readonly -a SEEDS=(4241 5153 8677 10007 65537)
readonly -a CONFIGS=(legacy_zero corrected_zero corrected_default)
readonly -a COMMITS=(aef0697 59db471 8e9b45f)

task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
seed_index=$((task_id / ${#CONFIGS[@]}))
config_index=$((task_id % ${#CONFIGS[@]}))
seed=${SEEDS[$seed_index]:?No seed for array task $task_id}
config=${CONFIGS[$config_index]}
commit=${COMMITS[$config_index]}
worktree=$WORKTREE_ROOT/$commit
checkpoint_dir=$RUN_ROOT/checkpoints/$config/seed-$seed
wandb_dir=$RUN_ROOT/wandb/$config/seed-$seed

conda activate hebby
set -u
export WANDB_MODE=offline
export WANDB_EXECUTABLE=$CONDA_PREFIX/bin/python
export HF_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "Array job: ${SLURM_ARRAY_JOB_ID}_${task_id}"
echo "Seed: $seed"
echo "Configuration: $config ($commit)"
echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader

[[ -d $worktree ]] || { echo "Missing worktree: $worktree"; exit 1; }
actual_commit=$(git -C "$worktree" rev-parse --short=7 HEAD)
[[ $actual_commit == "$commit" ]] || {
    echo "Wrong revision in $worktree: expected $commit, found $actual_commit"
    exit 1
}
[[ -d $worktree/synth_datasets/3_palindrome_dataset_vary_length ]] || {
    echo "Missing shared 3-palindrome dataset in $worktree"
    exit 1
}

mkdir -p "$checkpoint_dir" "$wandb_dir"
export WANDB_DIR=$wandb_dir
export WANDB_NAME="3pal-$config-orc-s$seed"

extra_args=()
if [[ $commit == aef0697 || $commit == 59db471 ]]; then
    extra_args+=(--self_grad 0)
fi

echo "--- Starting $config at $commit, seed $seed ---"
cd "$worktree"
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
    --tags benchmark multiseed orc extra5 "$config" "$commit" "seed$seed" \
    --notes "Paired ORC replicate: series=$SERIES config=$config commit=$commit seed=$seed array=${SLURM_ARRAY_JOB_ID}_${task_id}" \
    --ephemeral_fraction 0.2 \
    --enable_recurrence false \
    --seed "$seed" \
    --deterministic true \
    "${extra_args[@]}"
status=$?
set -e

if [[ $status == 124 ]]; then
    echo "Time-limit checkpoint saved; requeueing task $SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
elif [[ $status != 0 ]]; then
    echo "$config failed with exit $status"
    exit "$status"
fi

echo "--- Finished $config at $commit, seed $seed ---"
