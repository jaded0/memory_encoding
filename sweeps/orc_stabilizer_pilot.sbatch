#!/bin/bash --login
# Which stabilizer stops the late divergence? 3-char reversal at lr 1e-3 and plasticity 1e4
# (lr * alpha = 10), where every run of benchmarks/memory_tasks_head.md learned, peaked and then
# collapsed (205k-355k). Each arm changes one setting from that control. 400k iterations,
# seeds 3141 and 2718, --fused_update. train.py stops a run whose loss stays above 5 for 10
# intervals, which marks a collapse.
#
# RUN_SET=saturation (sbatch --array=0-3 --export=ALL,RUN_SET=saturation) asks whether fast
# weights pinned at the clamp cost memory: output tanh with the weight clamp off against the
# pilot's output tanh at clamp 1 (about 10% of fast entries saturated; slow entries never reach
# 1), three seeds each, the pilot's two clamp-1 runs reused. Clamp 1 also binds some slow entries
# under output tanh (up to 11% of the last trunk layer's), so RUN_SET=saturation_fast adds output
# tanh with only the fast entries clamped at 1 (--fast_weight_clamp; needs CODE_COMMIT with it):
#   sbatch --array=0-2 --export=ALL,RUN_SET=saturation_fast,CODE_COMMIT=<commit> ...

#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus=1
#SBATCH --mem=30G
#SBATCH --partition=cs,cs2
#SBATCH --qos=cs
#SBATCH --requeue
#SBATCH --job-name=stabilize
#SBATCH --array=0-13%8
#SBATCH --output=/home/jaded79/memory_encoding_benchmarks/stabilizer_pilot/logs/%x_%A_%a.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jaden.lorenc@gmail.com

set -eo pipefail

readonly RUN_ROOT=/home/jaded79/memory_encoding_benchmarks/stabilizer_pilot
readonly COMMIT=${CODE_COMMIT:-8fdd980}
readonly CODE=$([[ $COMMIT == 8fdd980 ]] && echo "$RUN_ROOT/code" || echo "$RUN_ROOT/code_$COMMIT")
readonly SERIES=stabilizer_pilot_v1

readonly -a SEEDS=(3141 2718)
readonly -a ARMS=(control slow_decay_1e-5 slow_decay_1e-6 weight_clamp_0.3 weight_clamp_0.1 output_tanh update_clamp_100)
task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
runs=()
if [[ ${RUN_SET:-pilot} == saturation ]]; then
    runs=("output_tanh 4241" "output_tanh_noclamp 3141" "output_tanh_noclamp 2718" "output_tanh_noclamp 4241")
elif [[ ${RUN_SET:-pilot} == saturation_fast ]]; then
    runs=("output_tanh_fastclamp 3141" "output_tanh_fastclamp 2718" "output_tanh_fastclamp 4241")
else
    for arm in "${ARMS[@]}"; do
        for seed in "${SEEDS[@]}"; do runs+=("$arm $seed"); done
    done
fi
[[ -n ${runs[$task_id]:-} ]] || { echo "No run for array task $task_id"; exit 1; }
read -r arm seed <<< "${runs[$task_id]}"
weight_clamp=1 extra=()
case $arm in
    control) ;;
    slow_decay_1e-5) extra=(--slow_weight_decay 1e-5) ;;
    slow_decay_1e-6) extra=(--slow_weight_decay 1e-6) ;;
    weight_clamp_0.3) weight_clamp=0.3 ;;
    weight_clamp_0.1) weight_clamp=0.1 ;;
    output_tanh) extra=(--output_tanh true) ;;
    output_tanh_noclamp) weight_clamp=0 extra=(--output_tanh true) ;;
    output_tanh_fastclamp) weight_clamp=0 extra=(--output_tanh true --fast_weight_clamp 1) ;;
    update_clamp_100) extra=(--ephemeral_update_clamp 100) ;;
esac
checkpoint_dir=$RUN_ROOT/checkpoints/$arm/seed-$seed
wandb_dir=$RUN_ROOT/wandb/$arm/seed-$seed

conda activate hebby
set -u
export WANDB_MODE=offline
export WANDB_EXECUTABLE=$CONDA_PREFIX/bin/python
export HF_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# A fresh directory per start: a requeued job keeps its job ID, and on some nodes (cs-1-5,
# 2026-09-30) the previous start's directory was not writable, so torch.compile failed.
export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d "${TMPDIR:-/tmp}/inductor_${SLURM_JOB_ID:-local}_XXXXXX")

echo "Array job: ${SLURM_ARRAY_JOB_ID}_${task_id}"
echo "Arm: $arm (weight clamp $weight_clamp, ${extra[*]:-no extra flags}), seed $seed, code $COMMIT"
echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader

[[ $(cat "$CODE/REVISION" 2>/dev/null) == "$COMMIT"* ]] || {
    echo "Expected an export of $COMMIT in $CODE"
    exit 1
}
mkdir -p "$checkpoint_dir" "$wandb_dir"
export WANDB_DIR=$wandb_dir
export WANDB_NAME="stabilize-$arm-s$seed"

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
    --learning_rate 1e-3 \
    --plasticity 1e4 \
    --forget_rate 0.01 \
    --ephemeral_fraction 0.2 \
    --ephemeral_update_clamp 0 \
    --grad_norm_clip 0 \
    --unit_norm_weights false \
    --weight_clamp "$weight_clamp" \
    --hidden_size 1024 \
    --num_layers 3 \
    --residual_connection false \
    --positional_encoding_dim 0 \
    --batch_size 16 \
    --n_iters 400000 \
    --print_freq 5000 \
    --checkpoint_dir "$checkpoint_dir" \
    --checkpoint_save_freq 25000 \
    --resume true \
    --track true \
    --group "$SERIES" \
    --tags stabilizer_pilot orc "$arm" "$COMMIT" "seed$seed" \
    --notes "Stabilizer pilot: series=$SERIES arm=$arm commit=$COMMIT seed=$seed array=${SLURM_ARRAY_JOB_ID}_${task_id}" \
    --seed "$seed" \
    --deterministic true \
    "${extra[@]}"
status=$?
set -e

if [[ $status == 124 ]]; then
    echo "Time-limit checkpoint saved; requeueing task $SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
elif [[ $status != 0 ]] && grep -q "^Early stopping: Loss has been" \
        "$RUN_ROOT/logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${task_id}.out"; then
    # The outcome under test, not a crash: record it without failing the task.
    echo "--- Collapsed: $arm, seed $seed (loss early stop) ---"
    exit 0
elif [[ $status != 0 ]]; then
    echo "$arm failed with exit $status"
    exit "$status"
fi

echo "--- Finished $arm, seed $seed ---"
