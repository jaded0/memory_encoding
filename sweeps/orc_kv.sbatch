#!/bin/bash --login
# Key-value panel, 2026-09-29: the key-value tasks (kv_tasks.py) at the setting that solved key
# recall (long_range_memory_dataset) by ~200k in the overnight panel (keyrecall_a3e4): DFA,
# recurrence off, lr 1e-4, plasticity 3e4 (1e5 diverged there), 10% fast, no clamp. One run per
# array task:
#   kv           ephemeral DFA on kv_unique_{2,4,8} and kv_reassign_{2,4}, with held-out recall
#                (heldout.py) every 250k.
#   kv_bptt      SimpleRNN + BPTT with recurrence on, at the same lr, on kv_unique_4 and
#                kv_reassign_4.
# Every run logs recall_acc and the answer classes kv_correct / kv_stale / kv_wrong_key /
# kv_other (metrics.IntervalMetrics).
# Submit: sbatch --export=ALL,CODE_COMMIT=<exported commit> sweeps/orc_kv.sbatch

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
#SBATCH --job-name=kv
#SBATCH --array=0-6
#SBATCH --output=/home/jaded79/memory_encoding_benchmarks/kv/logs/%x_%A_%a.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jaden.lorenc@gmail.com

set -eo pipefail

readonly RUN_ROOT=${RUN_ROOT:-/home/jaded79/memory_encoding_benchmarks/kv}
readonly COMMIT=${CODE_COMMIT:?Set CODE_COMMIT to the exported commit with --export}
readonly CODE=${CODE_DIR:-$RUN_ROOT/code_$COMMIT}
readonly SERIES=kv_2026-09-29

# arm dataset seed iterations
readonly -a RUNS=(
    "kv kv_unique_2 2718 1000000" "kv kv_unique_4 2718 1000000" "kv kv_unique_8 2718 1000000"
    "kv kv_reassign_2 2718 1000000" "kv kv_reassign_4 2718 1000000"
    "kv_bptt kv_unique_4 2718 1000000" "kv_bptt kv_reassign_4 2718 1000000"
)

task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
[[ -n ${RUNS[$task_id]:-} ]] || { echo "No run for array task $task_id (${#RUNS[@]} runs)"; exit 1; }
read -r arm dataset seed n_iters <<< "${RUNS[$task_id]}"
n_iters=${N_ITERS:-$n_iters}
print_freq=${PRINT_FREQ:-5000}
heldout_every=${HELDOUT_EVERY:-250000}
save_freq=${SAVE_FREQ:-50000}

model=(--model_type ephemeral --updater dfa --enable_recurrence false --fused_update true)
lr=1e-4 plasticity=3e4 fraction=0.1 extra=()
case $arm in
    kv) extra=(--heldout_eval_every "$heldout_every" --heldout_batches 4) ;;
    kv_bptt) model=(--model_type rnn --updater bptt --enable_recurrence true --fused_update false) ;;
    *) echo "Unknown arm $arm"; exit 1 ;;
esac

run_name=$arm-$dataset-s$seed
checkpoint_dir=$RUN_ROOT/checkpoints/$run_name
wandb_dir=$RUN_ROOT/wandb/$run_name

conda activate hebby
set -u
export WANDB_MODE=offline
export WANDB_EXECUTABLE=$CONDA_PREFIX/bin/python
export HF_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCHINDUCTOR_CACHE_DIR=${TMPDIR:-/tmp}/inductor_${SLURM_JOB_ID:-local}

echo "Array job: ${SLURM_ARRAY_JOB_ID:-local}_${task_id} (restart count ${SLURM_RESTART_COUNT:-0})"
echo "Run: $arm, $dataset, seed $seed, $n_iters iterations, lr $lr, plasticity $plasticity," \
     "fraction $fraction, ${extra[*]:-no extra flags}, code $COMMIT"
echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader

[[ $(cat "$CODE/REVISION" 2>/dev/null) == "$COMMIT"* ]] || {
    echo "Expected an export of $COMMIT in $CODE"
    exit 1
}
[[ -d $CODE/synth_datasets/$dataset ]] || { echo "Missing $dataset in $CODE/synth_datasets"; exit 1; }
mkdir -p "$checkpoint_dir" "$wandb_dir"
export WANDB_DIR=$wandb_dir
export WANDB_NAME="kv-$run_name"

cd "$CODE"
source sweeps/forward_signals.sh
set +e
forward_signals "$CONDA_PREFIX/bin/python" -u train.py \
    "${model[@]}" \
    --dataset "$dataset" \
    --input_mode last_one \
    --learning_rate "$lr" \
    --plasticity "$plasticity" \
    --forget_rate 0.01 \
    --ephemeral_fraction "$fraction" \
    --ephemeral_update_clamp 0 \
    --unit_norm_weights false \
    --weight_clamp 0 \
    --slow_weight_decay 0 \
    --hidden_size 1024 \
    --num_layers 3 \
    --residual_connection false \
    --positional_encoding_dim 0 \
    --batch_size 16 \
    --n_iters "$n_iters" \
    --print_freq "$print_freq" \
    --checkpoint_dir "$checkpoint_dir" \
    --checkpoint_save_freq "$save_freq" \
    --resume true \
    --track true \
    --group "$SERIES" \
    --tags kv orc "$arm" "$dataset" "$COMMIT" "seed$seed" \
    --notes "Key-value panel: series=$SERIES arm=$arm dataset=$dataset commit=$COMMIT seed=$seed array=${SLURM_ARRAY_JOB_ID:-local}_${task_id}" \
    --seed "$seed" \
    --deterministic true \
    "${extra[@]}"
status=$?
set -e

log=$RUN_ROOT/logs/${SLURM_JOB_NAME:-kv}_${SLURM_ARRAY_JOB_ID:-local}_${task_id}.out
if [[ $status == 124 ]]; then
    echo "Time-limit checkpoint saved; requeueing task $SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
elif [[ $status != 0 ]] && [[ -f $log ]] && grep -q "^Early stopping: Loss has been" "$log"; then
    echo "--- Collapsed: $run_name (loss early stop) ---"
    exit 0
elif [[ $status != 0 ]]; then
    echo "$run_name failed with exit $status"
    exit "$status"
fi

echo "--- Finished $run_name ---"
