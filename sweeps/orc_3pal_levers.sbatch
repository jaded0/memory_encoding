#!/bin/bash --login
# Plateau levers on 3-character reversal at the multi-million-iteration scale: four arms x four
# seeds at the 2025 recipe (DFA, recurrence off, lr 1e-4, plasticity 1e5, 10% fast, no clamp, no
# output tanh). Question: do the key-recall plateau levers (readout learning rate, aligned
# feedback) shorten the 3pal plateau, and does the per-position answer margin close earlier?
# Design note: ~/Documents/brain/ephemeral-lowrank/three_char_orc/design.md. Background: the 625k
# window (three_char_plateau/report.md) saw no plateau exit in any arm, with readout x10 worse than
# random at plasticity 1e5; the 2025 recipe converged at a median of ~6M iterations.
#
# Arms (every arm logs --log_margin: per-position margin, p_correct, accuracy, whole-sequence
# accuracy every PRINT_FREQ iterations; held-out recall every 250k):
#   random      control: random feedback, readout lr x1
#   ro3         --readout_lr_scale 3 (x10 was worse than random at 625k; x3 is the milder value)
#   aligned     --feedback_init aligned (best fixed B at init, at random B's Frobenius norm)
#   aligned_ro2 aligned + --readout_lr_scale 2: the gain-conservative combination (x10 + aligned
#               was worse than aligned alone at 625k; keeping the readout boost small)
# Tasks are seed-major, so any prefix of the array holds all four arms for each completed seed.
# Seeds 2718/3141/4241 are the 2025-recipe reproduction's seeds (random-arm reference, converged at
# 8.7M / 2.69M / 9.7M); 1618 is new.
#
# Runs past cs's 1-day wall limit continue through the USR1 checkpoint and requeue (an 8M run takes
# ~1.3 days on an A100). A run that holds whole-sequence accuracy >= STOP_ACC over the last
# STOP_INTERVALS print intervals is stopped (a watcher sends USR1, the run checkpoints and exits;
# a requeued start also checks the log first). Loss > 5 for 10 intervals is a collapse, not a failure.
#
# Dry run (no cluster, no training, prints each task's command):
#   for t in $(seq 0 15); do SLURM_ARRAY_TASK_ID=$t DRY_RUN=1 CODE_COMMIT=abc1234 bash sweeps/orc_3pal_levers.sbatch; done
# Submit (DO NOT before the checklist in three_char_orc/design.md; the logs dir must exist first):
#   sbatch --export=ALL,CODE_COMMIT=<exported commit> sweeps/orc_3pal_levers.sbatch
# Extend past the 8M default by resubmitting a task with --export=ALL,CODE_COMMIT=...,N_ITERS=12000000.

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
#SBATCH --job-name=3pal_levers
#SBATCH --array=0-15%8
#SBATCH --output=/home/jaded79/memory_encoding_benchmarks/3pal_levers_2026-10/logs/%x_%A_%a.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jaden.lorenc@gmail.com

set -eo pipefail

readonly RUN_ROOT=${RUN_ROOT:-/home/jaded79/memory_encoding_benchmarks/3pal_levers_2026-10}
readonly COMMIT=${CODE_COMMIT:?Set CODE_COMMIT to the exported commit with --export}
readonly CODE=${CODE_DIR:-$RUN_ROOT/code_$COMMIT}
readonly SERIES=3pal_levers_2026-10

readonly P3=${DATASET:-3_palindrome_dataset_vary_length}
# arm seed iterations
readonly -a RUNS=(
    "random 3141 8000000" "ro3 3141 8000000" "aligned 3141 8000000" "aligned_ro2 3141 8000000"
    "random 2718 8000000" "ro3 2718 8000000" "aligned 2718 8000000" "aligned_ro2 2718 8000000"
    "random 4241 8000000" "ro3 4241 8000000" "aligned 4241 8000000" "aligned_ro2 4241 8000000"
    "random 1618 8000000" "ro3 1618 8000000" "aligned 1618 8000000" "aligned_ro2 1618 8000000"
)

task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
[[ -n ${RUNS[$task_id]:-} ]] || { echo "No run for array task $task_id (${#RUNS[@]} runs)"; exit 1; }
read -r arm seed n_iters <<< "${RUNS[$task_id]}"
n_iters=${N_ITERS:-$n_iters}
print_freq=${PRINT_FREQ:-1000}
heldout_every=${HELDOUT_EVERY:-250000}
save_freq=${SAVE_FREQ:-50000}
keep_every=${KEEP_EVERY:-1000000}
stop_acc=${STOP_ACC:-0.8}
stop_intervals=${STOP_INTERVALS:-50}
watch_every=${WATCH_EVERY:-600}

# The recipe; each arm overrides what it tests.
model=(--model_type ephemeral --updater dfa --enable_recurrence false --fused_update "${FUSED:-true}")
lr=1e-4 plasticity=1e5 fraction=0.1 weight_clamp=0 extra=()
case $arm in
    random) ;;
    ro3) extra=(--readout_lr_scale 3) ;;
    aligned) extra=(--feedback_init aligned) ;;
    aligned_ro2) extra=(--feedback_init aligned --readout_lr_scale 2) ;;
    *) echo "Unknown arm $arm"; exit 1 ;;
esac
extra+=(--log_margin true --heldout_eval_every "$heldout_every" --heldout_batches 4)

run_name=$arm-s$seed
checkpoint_dir=$RUN_ROOT/checkpoints/$run_name
wandb_dir=$RUN_ROOT/wandb/$run_name
log=$RUN_ROOT/logs/${SLURM_JOB_NAME:-3pal_levers}_${SLURM_ARRAY_JOB_ID:-local}_${task_id}.out

# converged LOG: true when the last STOP_INTERVALS interval blocks have a mean answer_seq_acc
# (the metric --log_margin prints each interval) >= STOP_ACC.
converged() {
    [[ -f $1 ]] || return 1
    awk -v need="$stop_intervals" -v thr="$stop_acc" '
        /^  answer_seq_acc: / { v[++n] = $2 + 0 }
        END { if (n < need) exit 1; s = 0; for (i = n - need + 1; i <= n; i++) s += v[i];
              exit !(s / need >= thr) }' "$1"
}

cmd=(train.py
    "${model[@]}"
    --dataset "$P3"
    --input_mode last_one
    --learning_rate "$lr"
    --plasticity "$plasticity"
    --forget_rate 0.01
    --ephemeral_fraction "$fraction"
    --ephemeral_update_clamp 0
    --unit_norm_weights false
    --weight_clamp "$weight_clamp"
    --slow_weight_decay 0
    --hidden_size 1024
    --num_layers 3
    --residual_connection false
    --positional_encoding_dim 0
    --batch_size 16
    --n_iters "$n_iters"
    --print_freq "$print_freq"
    --checkpoint_dir "$checkpoint_dir"
    --checkpoint_save_freq "$save_freq"
    --checkpoint_keep_every "$keep_every"
    --resume true
    --track true
    --group "$SERIES"
    --tags 3pal_levers orc "$arm" "$P3" "$COMMIT" "seed$seed"
    --notes "3pal levers: series=$SERIES arm=$arm dataset=$P3 commit=$COMMIT seed=$seed array=${SLURM_ARRAY_JOB_ID:-local}_${task_id}"
    --seed "$seed"
    --deterministic true
    "${extra[@]}")

if [[ ${DRY_RUN:-} == 1 ]]; then
    echo "task $task_id: $run_name (lr $lr, plasticity $plasticity, fraction $fraction, ${extra[*]})"
    printf 'CMD: python -u'; printf ' %q' "${cmd[@]}"; printf '\n'
    exit 0
fi

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

echo "Array job: ${SLURM_ARRAY_JOB_ID:-local}_${task_id} (restart count ${SLURM_RESTART_COUNT:-0})"
echo "Run: $arm, $P3, seed $seed, $n_iters iterations, lr $lr, plasticity $plasticity," \
     "fraction $fraction, weight clamp $weight_clamp, ${extra[*]}, code $COMMIT"
echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader

[[ $(cat "$CODE/REVISION" 2>/dev/null) == "$COMMIT"* ]] || {
    echo "Expected an export of $COMMIT in $CODE"
    exit 1
}
[[ -d $CODE/synth_datasets/$P3 ]] || { echo "Missing $P3 in $CODE/synth_datasets"; exit 1; }
mkdir -p "$checkpoint_dir" "$wandb_dir"

if converged "$log"; then
    echo "--- Already converged (answer_seq_acc >= $stop_acc over $stop_intervals intervals): $run_name ---"
    exit 0
fi

export WANDB_DIR=$wandb_dir
export WANDB_NAME="3pal_levers-$run_name"

cd "$CODE"
source sweeps/forward_signals.sh

# Convergence watcher: after the first hour, every WATCH_EVERY seconds, ask the run to checkpoint
# and stop (USR1, as for the wall limit) once the log shows convergence.
batch_pid=$$
(
    sleep 3600
    while sleep "$watch_every"; do
        if converged "$log"; then
            echo "Convergence watcher: stopping $run_name" >> "$log"
            touch "$checkpoint_dir/CONVERGED"
            kill -USR1 "$batch_pid"
            break
        fi
    done
) &
watcher=$!

set +e
forward_signals "$CONDA_PREFIX/bin/python" -u "${cmd[@]}"
status=$?
set -e
kill "$watcher" 2>/dev/null || true

if [[ $status == 124 && -f $checkpoint_dir/CONVERGED ]]; then
    echo "--- Converged, stopped by the watcher: $run_name ---"
    exit 0
elif [[ $status == 124 ]]; then
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
