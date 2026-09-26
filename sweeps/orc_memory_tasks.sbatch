#!/bin/bash --login
# Does the ephemeral memory replace a clipped recurrent connection? One run per array task, on
# six memory tasks (2-, 3- and 4-char reversal, binary 3- and 4-char reversal, 4-period
# resequencing; long_range_memory_dataset is left out because train.py preprocesses its 10M rows
# on the fly at every start, which takes over 5 minutes):
#   ephemeral  ephemeral + DFA with recurrence off, lr 1e-3 and plasticity 1e4 (lr * alpha = 10,
#              the regime in which 2025 runs reached 0.9+ on 3-char palindromes)
#   no_fast    the same model with --ephemeral_fraction 0: no fast entries, so no fast memory
#   rnn_bptt   SimpleRNN + BPTT with recurrence on: the recurrent connection being replaced
# Everything else is the 3-palindrome panel's settings (sweeps/orc_3pal_head_panel.sbatch).
# The code is an exported copy of $COMMIT with a REVISION file checked below.

#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus=1
#SBATCH --mem=30G
#SBATCH --partition=cs,cs2
#SBATCH --qos=cs
#SBATCH --requeue
#SBATCH --job-name=memtasks
#SBATCH --array=0-24%10
#SBATCH --output=/home/jaded79/memory_encoding_benchmarks/memory_tasks_head/logs/%x_%A_%a.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jaden.lorenc@gmail.com

set -eo pipefail

readonly RUN_ROOT=/home/jaded79/memory_encoding_benchmarks/memory_tasks_head
readonly CODE=$RUN_ROOT/code
readonly COMMIT=7226291
readonly SERIES=memory_tasks_head_v1

readonly -a TASKS=(2_palindrome_dataset_vary_length 3_palindrome_dataset_vary_length
                   4_palindrome_dataset_vary_length 3_small_palindrome_dataset_vary_length
                   4_small_palindrome_dataset_vary_length 4_resequence)
runs=()
for task in "${TASKS[@]}"; do
    runs+=("$task ephemeral 2718" "$task ephemeral 3141")
    [[ $task == 3_palindrome_dataset_vary_length ]] && runs+=("$task ephemeral 4241")
    runs+=("$task no_fast 2718" "$task rnn_bptt 2718")
done

task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
[[ -n ${runs[$task_id]:-} ]] || { echo "No run for array task $task_id (${#runs[@]} runs)"; exit 1; }
read -r dataset arm seed <<< "${runs[$task_id]}"
case $arm in
    ephemeral) model=(--model_type ephemeral --updater dfa --enable_recurrence false --ephemeral_fraction 0.2)
               n_iters=500000 ;;
    no_fast) model=(--model_type ephemeral --updater dfa --enable_recurrence false --ephemeral_fraction 0)
             n_iters=250000 ;;
    rnn_bptt) model=(--model_type rnn --updater bptt --enable_recurrence true --ephemeral_fraction 0.2)
              n_iters=250000 ;;
esac
checkpoint_dir=$RUN_ROOT/checkpoints/$dataset/$arm/seed-$seed
wandb_dir=$RUN_ROOT/wandb/$dataset/$arm/seed-$seed

conda activate hebby
set -u
export WANDB_MODE=offline
export WANDB_EXECUTABLE=$CONDA_PREFIX/bin/python
export HF_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "Array job: ${SLURM_ARRAY_JOB_ID}_${task_id}"
echo "Run: $dataset $arm seed $seed, $n_iters iterations, code $COMMIT"
echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader

[[ $(cat "$CODE/REVISION" 2>/dev/null) == "$COMMIT"* ]] || {
    echo "Expected an export of $COMMIT in $CODE"
    exit 1
}
[[ -d $CODE/synth_datasets/$dataset ]] || { echo "Missing $dataset in $CODE/synth_datasets"; exit 1; }

mkdir -p "$checkpoint_dir" "$wandb_dir"
export WANDB_DIR=$wandb_dir
export WANDB_NAME="memtasks-$dataset-$arm-s$seed"

cd "$CODE"
source sweeps/forward_signals.sh
set +e
forward_signals "$CONDA_PREFIX/bin/python" -u train.py \
    "${model[@]}" \
    --dataset "$dataset" \
    --input_mode last_one \
    --learning_rate 1e-3 \
    --plasticity 1e4 \
    --forget_rate 0.01 \
    --ephemeral_update_clamp 0 \
    --grad_norm_clip 0 \
    --unit_norm_weights false \
    --weight_clamp 1 \
    --hidden_size 1024 \
    --num_layers 3 \
    --residual_connection false \
    --positional_encoding_dim 0 \
    --batch_size 16 \
    --n_iters "$n_iters" \
    --print_freq 5000 \
    --checkpoint_dir "$checkpoint_dir" \
    --checkpoint_save_freq 25000 \
    --resume true \
    --track true \
    --group "$SERIES" \
    --tags memory_tasks orc head "$arm" "$dataset" "$COMMIT" "seed$seed" \
    --notes "Memory-task verification: series=$SERIES dataset=$dataset arm=$arm commit=$COMMIT seed=$seed array=${SLURM_ARRAY_JOB_ID}_${task_id}" \
    --seed "$seed" \
    --deterministic true
status=$?
set -e

if [[ $status == 124 ]]; then
    echo "Time-limit checkpoint saved; requeueing task $SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
elif [[ $status != 0 ]]; then
    echo "$dataset $arm failed with exit $status"
    exit "$status"
fi

echo "--- Finished $dataset $arm, seed $seed ---"
