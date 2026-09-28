#!/bin/bash --login
# Reproduce Jaden's Sep-Oct 2025 recipe, which converged on 3-char reversal without any clamp
# (benchmarks/old_recipe_reproduction.md): DFA, recurrence off, lr 1e-4, plasticity 1e5
# (lr * alpha = 10 with a 10x slower slow-weight rate than the 2026 benchmarks), 10% fast
# entries, weight clamp off, no output tanh. In 2025, 13 of 23 such runs passed 0.9 final-character
# accuracy, at 2.8M-9.4M iterations. Current code, --fused_update, three seeds, 5M iterations.
# Runs past cs's 1-day wall limit continue through the USR1 checkpoint and requeue.

#SBATCH --time=1-00:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus=1
#SBATCH --mem=30G
#SBATCH --partition=cs,cs2
#SBATCH --qos=cs
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --job-name=oldrecipe
#SBATCH --array=0-2
#SBATCH --output=/home/jaded79/memory_encoding_benchmarks/old_recipe/logs/%x_%A_%a.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jaden.lorenc@gmail.com

set -eo pipefail

readonly RUN_ROOT=/home/jaded79/memory_encoding_benchmarks/old_recipe
readonly COMMIT=${CODE_COMMIT:?Set CODE_COMMIT to the exported commit with --export}
readonly CODE=$RUN_ROOT/code_$COMMIT
readonly SERIES=old_recipe_v1
readonly N_ITERS=${N_ITERS:-5000000}

readonly -a SEEDS=(2718 3141 4241)
task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
seed=${SEEDS[$task_id]:?No seed for array task $task_id}
checkpoint_dir=$RUN_ROOT/checkpoints/seed-$seed
wandb_dir=$RUN_ROOT/wandb/seed-$seed

conda activate hebby
set -u
export WANDB_MODE=offline
export WANDB_EXECUTABLE=$CONDA_PREFIX/bin/python
export HF_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCHINDUCTOR_CACHE_DIR=${TMPDIR:-/tmp}/inductor_$SLURM_JOB_ID

echo "Array job: ${SLURM_ARRAY_JOB_ID}_${task_id} (restart count ${SLURM_RESTART_COUNT:-0})"
echo "Run: old recipe, seed $seed, $N_ITERS iterations, code $COMMIT"
echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader

[[ $(cat "$CODE/REVISION" 2>/dev/null) == "$COMMIT"* ]] || {
    echo "Expected an export of $COMMIT in $CODE"
    exit 1
}
mkdir -p "$checkpoint_dir" "$wandb_dir"
export WANDB_DIR=$wandb_dir
export WANDB_NAME="oldrecipe-s$seed"

cd "$CODE"
source sweeps/forward_signals.sh
set +e
forward_signals "$CONDA_PREFIX/bin/python" -u train.py \
    --model_type ephemeral \
    --updater dfa \
    --enable_recurrence false \
    --fused_update true \
    --dataset 3_palindrome_dataset_vary_length \
    --input_mode last_one \
    --learning_rate 1e-4 \
    --plasticity 1e5 \
    --forget_rate 0.01 \
    --ephemeral_fraction 0.1 \
    --ephemeral_update_clamp 0 \
    --grad_norm_clip 0 \
    --unit_norm_weights false \
    --weight_clamp 0 \
    --fast_weight_clamp 0 \
    --slow_weight_decay 0 \
    --output_tanh false \
    --hidden_size 1024 \
    --num_layers 3 \
    --residual_connection false \
    --positional_encoding_dim 0 \
    --batch_size 16 \
    --n_iters "$N_ITERS" \
    --print_freq 5000 \
    --checkpoint_dir "$checkpoint_dir" \
    --checkpoint_save_freq 50000 \
    --resume true \
    --track true \
    --group "$SERIES" \
    --tags old_recipe orc "$COMMIT" "seed$seed" \
    --notes "Old-recipe reproduction: series=$SERIES commit=$COMMIT seed=$seed array=${SLURM_ARRAY_JOB_ID}_${task_id}" \
    --seed "$seed" \
    --deterministic true
status=$?
set -e

if [[ $status == 124 ]]; then
    echo "Time-limit checkpoint saved; requeueing task $SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
elif [[ $status != 0 ]] && grep -q "^Early stopping: Loss has been" \
        "$RUN_ROOT/logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${task_id}.out"; then
    echo "--- Collapsed: old recipe, seed $seed (loss early stop) ---"
    exit 0
elif [[ $status != 0 ]]; then
    echo "old recipe seed $seed failed with exit $status"
    exit "$status"
fi

echo "--- Finished old recipe, seed $seed ---"
