#!/bin/bash --login
# Five-predictions follow-up: sbatch --array=0-3 orc_array.sbatch
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus=1
#SBATCH --mem=30G
#SBATCH --partition=cs,cs2
#SBATCH --qos=cs
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --job-name=driftpred
#SBATCH --output=/home/jaded79/memory_encoding_benchmarks/drift_predictions_2026-10-06/logs/%x_%A_%a.out
set -eo pipefail

ROOT=/home/jaded79/memory_encoding_benchmarks/drift_predictions_2026-10-06
CODE=${CODE:-$ROOT/code}
ARMS=(nlms label_smoothing alpha0 slow_nlms)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
RUN=$ROOT/runs/$ARM

conda activate hebby
set -u
mkdir -p "$RUN"

EXTRA=()
N_ITERS=600000
case "$ARM" in
  nlms) EXTRA+=(--readout_nlms true) ;;
  label_smoothing) EXTRA+=(--label_smoothing 0.1) ;;
  alpha0) EXTRA+=(--plasticity 0); N_ITERS=900000 ;;
  slow_nlms) EXTRA+=(--slow_nlms true) ;;
esac

echo "=== $ARM start $(date) host $(hostname)" >> "$RUN/train.log"
(cd "$CODE" && WANDB_MODE=disabled HF_DATASETS_OFFLINE=1 HF_OFFLINE=1 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCHINDUCTOR_CACHE_DIR=$(mktemp -d "${TMPDIR:-$HOME/.cache}/ind_XXXXXX") \
  "$CONDA_PREFIX/bin/python" -u train.py \
    --model_type ephemeral --updater dfa --enable_recurrence false --fused_update true \
    --dataset 3_palindrome_dataset_vary_length --input_mode last_one --learning_rate 1e-3 \
    --plasticity 1e4 --forget_rate 0.01 --ephemeral_fraction 0.2 --weight_clamp 1 \
    --hidden_size 1024 --num_layers 3 --residual_connection false --positional_encoding_dim 0 \
    --batch_size 16 --track false --deterministic true --seed 3141 --resume true \
    --n_iters "$N_ITERS" --print_freq 500 --trace_loop_every 500 \
    --checkpoint_save_freq 10000 --checkpoint_keep_every 50000 --checkpoint_keep_max 0 \
    --early_stop_window 0 --checkpoint_dir "$RUN" "${EXTRA[@]}" >> "$RUN/train.log" 2>&1)
echo "=== exit $? $(date)" >> "$RUN/train.log"
