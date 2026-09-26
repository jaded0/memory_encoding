#!/bin/bash --login
# Current-code 3-palindrome panel: ephemeral + DFA against the SimpleRNN baseline under DFA and
# BPTT, on the seven ORC-paired seeds and the shared settings of the architecture/initialization
# factorial (benchmarks/3pal_architecture_initialization.md). The arms differ only in model,
# updater and recurrence: SimpleRNN needs recurrence for any memory, and the ephemeral arm keeps
# the factorial's recurrence-off setting. Clipping is matched: --weight_clamp 1 in every arm,
# --grad_norm_clip and --ephemeral_update_clamp off.
#
# qos=cs is 0.1x fairshare with priority 100 on the cs (A100) and cs2 (H100) partitions, whose
# GPUs the torch 2.5.1/cu121 hebby env supports. Each task runs one arm/seed. The code is an
# exported copy of $COMMIT (not a git worktree), with a REVISION file checked below.

#SBATCH --time=05:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus=1
#SBATCH --mem=30G
#SBATCH --partition=cs,cs2
#SBATCH --qos=cs
#SBATCH --requeue
#SBATCH --job-name=3pal_head
#SBATCH --array=0-20%8
#SBATCH --output=/home/jaded79/memory_encoding_benchmarks/orc_3pal_250k_head/logs/%x_%A_%a.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jaden.lorenc@gmail.com

set -eo pipefail

readonly RUN_ROOT=/home/jaded79/memory_encoding_benchmarks/orc_3pal_250k_head
readonly CODE=$RUN_ROOT/code
readonly COMMIT=5e088af
readonly SERIES=3pal_250k_head_v1

readonly -a SEEDS=(2718 3141 4241 5153 8677 10007 65537)
readonly -a ARMS=(ephemeral_dfa rnn_dfa rnn_bptt)

task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
seed=${SEEDS[$((task_id / ${#ARMS[@]}))]:?No seed for array task $task_id}
arm=${ARMS[$((task_id % ${#ARMS[@]}))]}
case $arm in
    ephemeral_dfa) model=ephemeral updater=dfa recurrence=false ;;
    rnn_dfa) model=rnn updater=dfa recurrence=true ;;
    rnn_bptt) model=rnn updater=bptt recurrence=true ;;
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

echo "Array job: ${SLURM_ARRAY_JOB_ID}_${task_id}"
echo "Arm: $arm ($model, $updater, recurrence $recurrence), seed $seed, code $COMMIT"
echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader

[[ $(cat "$CODE/REVISION" 2>/dev/null) == "$COMMIT"* ]] || {
    echo "Expected an export of $COMMIT in $CODE"
    exit 1
}
[[ -d $CODE/synth_datasets/3_palindrome_dataset_vary_length ]] || {
    echo "Missing 3-palindrome dataset in $CODE/synth_datasets"
    exit 1
}

mkdir -p "$checkpoint_dir" "$wandb_dir"
export WANDB_DIR=$wandb_dir
export WANDB_NAME="3pal-head-$arm-orc-s$seed"

cd "$CODE"
source sweeps/forward_signals.sh
set +e
forward_signals "$CONDA_PREFIX/bin/python" -u train.py \
    --model_type "$model" \
    --updater "$updater" \
    --enable_recurrence "$recurrence" \
    --input_mode last_one \
    --learning_rate 1e-3 \
    --plasticity 1e3 \
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
    --dataset 3_palindrome_dataset_vary_length \
    --batch_size 16 \
    --n_iters 250000 \
    --print_freq 5000 \
    --checkpoint_dir "$checkpoint_dir" \
    --checkpoint_save_freq 5000 \
    --resume true \
    --track true \
    --group "$SERIES" \
    --tags benchmark multiseed orc head "$arm" "$COMMIT" "seed$seed" \
    --notes "Current-code panel: series=$SERIES arm=$arm commit=$COMMIT seed=$seed array=${SLURM_ARRAY_JOB_ID}_${task_id}" \
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
