#!/bin/bash --login
# Benchmark-task panel, 2026-10-02: do the project's conclusions hold on tasks beyond key recall?
# Tasks: kv_unique_4 (bridge to the 09-29 kv panel), mqar_4, mod3_8, selcopy_3_9 (bench_tasks.py).
# Arms (every ephemeral arm: DFA, recurrence off, 10% fast, forget 0.01, no clamp, input last_one):
#   base       lr 1e-4, plasticity 3e4 (lr*alpha = 3), held-out (observed/strict/no_fast/free_running) every 50k
#   nofast     --ephemeral_fraction 0 (C1 control)
#   ln         base + --layer_norm true (C3: LayerNorm speeds convergence)
#   slow03     lr 3e-5, plasticity 1e5: slow lr x0.3 at an unchanged fast step lr*alpha = 3 (C3)
#   bptt_sgd   SimpleRNN + BPTT, recurrence on, SGD lr 1e-1, clip 1 (C2)
#   bptt_adam  SimpleRNN + BPTT, recurrence on, Adam lr 1e-4, clip 1, residual (C2)
#   Extras: bptt_sgdres (SGD lr 1e-1, clip 1, residual: the residual is what lets kv_unique_4 train)
#           and base_a1e4 (plasticity 1e4, mqar_4 only: C4 probe at the longest sequences).
# One run per array task, index = ((task * 6 + arm) * 2 + seed), then the extras. The sizing of every
# task's iterations is in ITERS. Runs past cs's 1-day limit continue through the USR1 checkpoint.
# Submit: sbatch --export=ALL,CODE_COMMIT=<exported commit> [--array=...] sweeps/orc_bench_panel.sbatch
# Timing: N_ITERS=400 RUN_ROOT=.../bench_timing (overrides every run's iterations and directory).

#SBATCH --time=1-00:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus=1
#SBATCH --mem=30G
#SBATCH --partition=cs,cs2
#SBATCH --qos=cs
#SBATCH --exclude=cs-1-5
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --job-name=benchpanel
#SBATCH --array=0-57%12
#SBATCH --output=/home/jaded79/memory_encoding_benchmarks/bench_panel/logs/%x_%A_%a.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jaden.lorenc@gmail.com

set -eo pipefail

readonly RUN_ROOT=${RUN_ROOT:-/home/jaded79/memory_encoding_benchmarks/bench_panel}
readonly COMMIT=${CODE_COMMIT:?Set CODE_COMMIT to the exported commit with --export}
readonly CODE=${CODE_DIR:-$RUN_ROOT/code_$COMMIT}
readonly SERIES=bench_panel_2026-10-02

readonly -a TASKS=(kv_unique_4 mqar_4 mod3_8 selcopy_3_9)
readonly -a ARMS=(base nofast ln slow03 bptt_sgd bptt_adam)
readonly -a SEEDS=(2718 3141)
declare -A ITERS=([kv_unique_4]=300000 [mqar_4]=300000 [mod3_8]=300000 [selcopy_3_9]=300000)

# Build "arm task seed" for every run, in array order.
RUNS=()
for task in "${TASKS[@]}"; do for arm in "${ARMS[@]}"; do for seed in "${SEEDS[@]}"; do
    RUNS+=("$arm $task $seed")
done; done; done
for seed in "${SEEDS[@]}"; do for task in "${TASKS[@]}"; do RUNS+=("bptt_sgdres $task $seed"); done; done
for seed in "${SEEDS[@]}"; do RUNS+=("base_a1e4 mqar_4 $seed"); done
readonly RUNS

task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
[[ -n ${RUNS[$task_id]:-} ]] || { echo "No run for array task $task_id (${#RUNS[@]} runs)"; exit 1; }
read -r arm dataset seed <<< "${RUNS[$task_id]}"
n_iters=${N_ITERS:-${ITERS[$dataset]}}
print_freq=${PRINT_FREQ:-5000}
heldout_every=${HELDOUT_EVERY:-50000}
save_freq=${SAVE_FREQ:-50000}

model=(--model_type ephemeral --updater dfa --enable_recurrence false --fused_update true)
lr=1e-4 plasticity=3e4 fraction=0.1 extra=() heldout=true
bptt=(--model_type rnn --updater bptt --enable_recurrence true --fused_update false)
case $arm in
    base) ;;
    base_a1e4) plasticity=1e4 ;;
    nofast) fraction=0 heldout=false ;;
    ln) extra=(--layer_norm true) ;;
    slow03) lr=3e-5 plasticity=1e5 ;;
    bptt_sgd) model=("${bptt[@]}"); lr=1e-1; heldout=false
              extra=(--optimizer sgd --grad_norm_clip 1) ;;
    bptt_sgdres) model=("${bptt[@]}"); lr=1e-1; heldout=false
                 extra=(--optimizer sgd --grad_norm_clip 1 --residual_connection true) ;;
    bptt_adam) model=("${bptt[@]}"); heldout=false
               extra=(--optimizer adam --grad_norm_clip 1 --residual_connection true) ;;
    *) echo "Unknown arm $arm"; exit 1 ;;
esac
[[ $heldout == true ]] && extra+=(--heldout_eval_every "$heldout_every" --heldout_batches 16)

run_name=$dataset-$arm-s$seed
checkpoint_dir=$RUN_ROOT/checkpoints/$run_name
wandb_dir=$RUN_ROOT/wandb/$run_name

conda activate hebby
set -u
export WANDB_MODE=offline
export WANDB_EXECUTABLE=$CONDA_PREFIX/bin/python
export HF_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# A fresh directory per start (a requeued job keeps its job ID; cs-1-5 once left an unwritable one).
export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d "${TMPDIR:-/tmp}/inductor_${SLURM_JOB_ID:-local}_XXXXXX")

echo "Array job: ${SLURM_ARRAY_JOB_ID:-local}_${task_id} (restart count ${SLURM_RESTART_COUNT:-0})"
echo "Run: $run_name arm $arm, $dataset, seed $seed, $n_iters iterations, lr $lr, plasticity $plasticity," \
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
export WANDB_NAME="bench-$run_name"

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
    --positional_encoding_dim 0 \
    --batch_size 16 \
    --n_iters "$n_iters" \
    --print_freq "$print_freq" \
    --checkpoint_dir "$checkpoint_dir" \
    --checkpoint_save_freq "$save_freq" \
    --resume true \
    --track true \
    --group "$SERIES" \
    --tags bench orc "$arm" "$dataset" "$COMMIT" "seed$seed" \
    --notes "Benchmark panel: series=$SERIES arm=$arm dataset=$dataset commit=$COMMIT seed=$seed array=${SLURM_ARRAY_JOB_ID:-local}_${task_id}" \
    --seed "$seed" \
    --deterministic true \
    "${extra[@]}"
status=$?
set -e

log=$RUN_ROOT/logs/${SLURM_JOB_NAME:-benchpanel}_${SLURM_ARRAY_JOB_ID:-local}_${task_id}.out
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
