#!/bin/bash --login
# --layer_norm panel, 2026-09-29: does affine-free LayerNorm on the trunk activations (both models'
# --layer_norm, after each trunk GELU) harm memory, as paper/paper_content.tex:195 claims? Every
# arm is the overnight panel's recipe (sweeps/orc_overnight_panel.sbatch: DFA, recurrence off,
# lr 1e-4, 10% fast, no clamp, no output tanh) plus --layer_norm true. One run per array task:
#   keyrecall_ln  long_range_memory_dataset at plasticity 3e4, 1M. Reference without LayerNorm:
#                 the overnight panel's keyrecall_a3e4 (seeds 2718, 3141: 100% recall by ~200k).
#   3pal_ln       3-char reversal at the recipe (plasticity 1e5), 5M. Reference: the old-recipe
#                 reproduction (seed 3141: 95% at 5M; benchmarks/old_recipe_reproduction.md).
# Every arm logs held-out recall (heldout.py) every 250k.
# Submit: sbatch --export=ALL,CODE_COMMIT=<exported commit> sweeps/orc_layer_norm.sbatch

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
#SBATCH --job-name=layer_norm
#SBATCH --array=0-3
#SBATCH --output=/home/jaded79/memory_encoding_benchmarks/layer_norm/logs/%x_%A_%a.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jaden.lorenc@gmail.com

set -eo pipefail

readonly RUN_ROOT=${RUN_ROOT:-/home/jaded79/memory_encoding_benchmarks/layer_norm}
readonly COMMIT=${CODE_COMMIT:?Set CODE_COMMIT to the exported commit with --export}
readonly CODE=${CODE_DIR:-$RUN_ROOT/code_$COMMIT}
readonly SERIES=layer_norm_2026-09-29

readonly P3=3_palindrome_dataset_vary_length KR=long_range_memory_dataset
# arm dataset seed iterations
readonly -a RUNS=(
    "keyrecall_ln $KR 2718 1000000" "keyrecall_ln $KR 3141 1000000"
    "3pal_ln $P3 2718 5000000" "3pal_ln $P3 3141 5000000"
)

task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
[[ -n ${RUNS[$task_id]:-} ]] || { echo "No run for array task $task_id (${#RUNS[@]} runs)"; exit 1; }
read -r arm dataset seed n_iters <<< "${RUNS[$task_id]}"
n_iters=${N_ITERS:-$n_iters}
print_freq=${PRINT_FREQ:-5000}
heldout_every=${HELDOUT_EVERY:-250000}
save_freq=${SAVE_FREQ:-50000}

# The recipe, with --layer_norm; each arm sets its plasticity.
model=(--model_type ephemeral --updater dfa --enable_recurrence false --fused_update true)
lr=1e-4 plasticity=1e5 fraction=0.1 weight_clamp=0 extra=(--layer_norm true) heldout=true
case $arm in
    keyrecall_ln) plasticity=3e4 ;;
    3pal_ln) ;;
    *) echo "Unknown arm $arm"; exit 1 ;;
esac
[[ $heldout == true ]] && extra+=(--heldout_eval_every "$heldout_every" --heldout_batches 4)

run_name=$arm-s$seed
checkpoint_dir=$RUN_ROOT/checkpoints/$run_name
wandb_dir=$RUN_ROOT/wandb/$run_name

conda activate hebby
set -u
export WANDB_MODE=offline
export WANDB_EXECUTABLE=$CONDA_PREFIX/bin/python
export HF_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d "${TMPDIR:-/tmp}/inductor_${SLURM_JOB_ID:-local}_XXXXXX")

echo "Array job: ${SLURM_ARRAY_JOB_ID:-local}_${task_id} (restart count ${SLURM_RESTART_COUNT:-0})"
echo "Run: $arm, $dataset, seed $seed, $n_iters iterations, lr $lr, plasticity $plasticity," \
     "fraction $fraction, weight clamp $weight_clamp, ${extra[*]:-no extra flags}, code $COMMIT"
echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader

[[ $(cat "$CODE/REVISION" 2>/dev/null) == "$COMMIT"* ]] || {
    echo "Expected an export of $COMMIT in $CODE"
    exit 1
}
[[ -d $CODE/synth_datasets/$dataset ]] || { echo "Missing $dataset in $CODE/synth_datasets"; exit 1; }
mkdir -p "$checkpoint_dir" "$wandb_dir"
export WANDB_DIR=$wandb_dir
export WANDB_NAME="layer_norm-$run_name"

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
    --weight_clamp "$weight_clamp" \
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
    --tags layer_norm orc "$arm" "$dataset" "$COMMIT" "seed$seed" \
    --notes "LayerNorm panel: series=$SERIES arm=$arm dataset=$dataset commit=$COMMIT seed=$seed array=${SLURM_ARRAY_JOB_ID:-local}_${task_id}" \
    --seed "$seed" \
    --deterministic true \
    "${extra[@]}"
status=$?
set -e

log=$RUN_ROOT/logs/${SLURM_JOB_NAME:-layer_norm}_${SLURM_ARRAY_JOB_ID:-local}_${task_id}.out
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
