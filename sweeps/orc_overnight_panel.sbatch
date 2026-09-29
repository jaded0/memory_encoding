#!/bin/bash --login
# Overnight panel, 2026-09-29: open questions at the 2025 recipe (sweeps/orc_old_recipe.sbatch:
# DFA, recurrence off, lr 1e-4, plasticity 1e5, 10% fast, no clamp, no output tanh). One run per
# array task, in priority order:
#   4pal         4-char reversal at the recipe. 2025 never ran it past 495k (untested, not failed).
#   4pal_bptt    SimpleRNN + BPTT with recurrence on, the same task, at the recipe's lr.
#   a7e4         3-char reversal at plasticity 7e4, the 2025 cell that converged 7 of 7.
#   bptt         SimpleRNN + BPTT on 3-char at lr 1e-4: the fair head-to-head at the good recipe.
#   no_fast      the recipe with --ephemeral_fraction 0: no fast memory.
#   tanh         the recipe with --output_tanh: does tanh cost anything where it isn't needed?
#   fastclamp    the recipe with --fast_weight_clamp 1: clamp erasure in the convergence phase.
#   normclip_old / normclip_2026  --grad_norm_clip 1e30 never binds (bit-identical to off) but
#                logs the per-sequence gradient-norm distribution, to set the clipping panel's
#                thresholds: at the recipe, and at the 2026 recipe with output tanh
#                (sweeps/orc_stabilizer_pilot.sbatch's output_tanh arm).
#   keyrecall    long_range_memory_dataset (Fig. 1's task, whose data is lost) at the recipe.
# Every ephemeral DFA arm with fast entries logs held-out recall (heldout.py) every 250k.
# Runs past cs's 1-day wall limit continue through the USR1 checkpoint and requeue.
# Submit: sbatch --export=ALL,CODE_COMMIT=<exported commit> sweeps/orc_overnight_panel.sbatch

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
#SBATCH --job-name=overnight
#SBATCH --array=0-18
#SBATCH --output=/home/jaded79/memory_encoding_benchmarks/overnight_2026-09-29/logs/%x_%A_%a.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jaden.lorenc@gmail.com

set -eo pipefail

readonly RUN_ROOT=${RUN_ROOT:-/home/jaded79/memory_encoding_benchmarks/overnight_2026-09-29}
readonly COMMIT=${CODE_COMMIT:?Set CODE_COMMIT to the exported commit with --export}
readonly CODE=${CODE_DIR:-$RUN_ROOT/code_$COMMIT}
readonly SERIES=overnight_2026-09-29

readonly P3=3_palindrome_dataset_vary_length P4=4_palindrome_dataset_vary_length
readonly KR=long_range_memory_dataset
# arm dataset seed iterations
readonly -a RUNS=(
    "4pal $P4 2718 10000000" "4pal $P4 3141 10000000" "4pal $P4 4241 10000000"
    "4pal_bptt $P4 2718 2000000"
    "a7e4 $P3 2718 10000000" "a7e4 $P3 3141 10000000" "a7e4 $P3 4241 10000000"
    "bptt $P3 2718 2000000" "bptt $P3 3141 2000000"
    "no_fast $P3 2718 1000000" "no_fast $P3 3141 1000000"
    "tanh $P3 2718 10000000" "tanh $P3 3141 10000000"
    "fastclamp $P3 2718 10000000" "fastclamp $P3 3141 10000000"
    "normclip_old $P3 2718 400000" "normclip_2026 $P3 2718 400000"
    "keyrecall $KR 2718 5000000" "keyrecall $KR 3141 5000000"
    # Added after 4pal diverged within the first 5k iterations (loss 50-120, collapsed by ~60k):
    # 4-char's lr * alpha ceiling is below 10. Submit with --array=19-23.
    "4pal_a3e4 $P4 2718 10000000" "4pal_a3e4 $P4 3141 10000000"
    "4pal_a5e4 $P4 2718 10000000" "4pal_a5e4 $P4 3141 10000000"
    "4pal_tanh $P4 2718 10000000"
    # keyrecall diverged the same way (loss 1e4-1e6 from the first interval). Submit --array=24-26.
    "keyrecall_a3e4 $KR 2718 5000000" "keyrecall_a3e4 $KR 3141 5000000"
    "keyrecall_tanh $KR 2718 5000000"
    # keyrecall_a3e4 reached 100% recall by 200k: the no-fast control. Submit --array=27.
    "keyrecall_no_fast $KR 2718 1000000"
    # Key-recall figure: more seeds at alpha 3e4 (solved by ~200k) and the BPTT baseline, 1M each.
    # Submit --array=28-32.
    "keyrecall_a3e4 $KR 4241 1000000" "keyrecall_a3e4 $KR 1618 1000000" "keyrecall_a3e4 $KR 5772 1000000"
    "keyrecall_bptt $KR 2718 1000000" "keyrecall_bptt $KR 3141 1000000"
)

task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
[[ -n ${RUNS[$task_id]:-} ]] || { echo "No run for array task $task_id (${#RUNS[@]} runs)"; exit 1; }
read -r arm dataset seed n_iters <<< "${RUNS[$task_id]}"
n_iters=${N_ITERS:-$n_iters}
print_freq=${PRINT_FREQ:-5000}
heldout_every=${HELDOUT_EVERY:-250000}
save_freq=${SAVE_FREQ:-50000}

# The recipe; each arm overrides what it tests.
model=(--model_type ephemeral --updater dfa --enable_recurrence false --fused_update true)
lr=1e-4 plasticity=1e5 fraction=0.1 weight_clamp=0 extra=() heldout=true
case $arm in
    4pal|keyrecall) ;;
    a7e4) plasticity=7e4 ;;
    4pal_a3e4) plasticity=3e4 ;;
    4pal_a5e4) plasticity=5e4 ;;
    4pal_tanh|keyrecall_tanh) extra=(--output_tanh true) ;;
    keyrecall_a3e4) plasticity=3e4 ;;
    4pal_bptt|bptt|keyrecall_bptt) model=(--model_type rnn --updater bptt --enable_recurrence true --fused_update false)
                    heldout=false ;;
    no_fast) fraction=0 heldout=false ;;
    keyrecall_no_fast) plasticity=3e4 fraction=0 heldout=false ;;
    tanh) extra=(--output_tanh true) ;;
    fastclamp) extra=(--fast_weight_clamp 1) ;;
    normclip_old) extra=(--grad_norm_clip 1e30) ;;
    normclip_2026) lr=1e-3 plasticity=1e4 fraction=0.2 weight_clamp=1
                   extra=(--grad_norm_clip 1e30 --output_tanh true) ;;
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
export TORCHINDUCTOR_CACHE_DIR=${TMPDIR:-/tmp}/inductor_${SLURM_JOB_ID:-local}

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
export WANDB_NAME="overnight-$run_name"

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
    --tags overnight orc "$arm" "$dataset" "$COMMIT" "seed$seed" \
    --notes "Overnight panel: series=$SERIES arm=$arm dataset=$dataset commit=$COMMIT seed=$seed array=${SLURM_ARRAY_JOB_ID:-local}_${task_id}" \
    --seed "$seed" \
    --deterministic true \
    "${extra[@]}"
status=$?
set -e

log=$RUN_ROOT/logs/${SLURM_JOB_NAME:-overnight}_${SLURM_ARRAY_JOB_ID:-local}_${task_id}.out
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
