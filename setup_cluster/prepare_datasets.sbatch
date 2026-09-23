#!/bin/bash --login
# ==============================================================================
# prepare_datasets.sbatch - COMPUTE NODE, offline: preprocess the downloaded HF
# datasets into the processed-data directory ($EPHEMERAL_DATA_DIR, default
# ./processed_datasets/), then smoke-test a few GPU iterations of the real
# training config.
#
#   sbatch setup_cluster/prepare_datasets.sbatch [dataset ...]   # from the repo root
#   REDO=1 sbatch setup_cluster/prepare_datasets.sbatch          # overwrite saved data
#
# Needs setup_cluster/download_datasets.sh first (compute nodes have no internet).
# The smoke test runs the train.py command from $CONFIG (default slurm_run.sh)
# with every flag it sets, overriding only length, tracking and checkpoint dir.
# Usually run through setup_cluster/setup.sh.
# ==============================================================================

#SBATCH --job-name=prepare_datasets
#SBATCH --qos=test               # 1 hour limit; preprocessing takes minutes
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16       # preprocess.py map workers (num_proc)
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --output=prepare_datasets_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jaden.lorenc@gmail.com

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1
[[ -f train.py ]] || { echo "Submit from the repo root: sbatch setup_cluster/prepare_datasets.sbatch"; exit 1; }

DATASETS=("$@")
[[ ${#DATASETS[@]} -gt 0 ]] || DATASETS=(roneneldan/tinystories)
CONFIG=${CONFIG:-slurm_run.sh}

conda activate hebby || { echo "conda activate hebby failed"; exit 1; }
export WANDB_MODE=offline
export WANDB_EXECUTABLE=$CONDA_PREFIX/bin/python
export HF_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ======================== Preprocess ==========================================
OVERWRITE=()
[[ ${REDO:-0} == 1 ]] && OVERWRITE=(--overwrite)
echo "--- Preprocessing ${DATASETS[*]} (num_proc ${SLURM_CPUS_PER_TASK:-auto}) ---"
python -u preprocess.py "${DATASETS[@]}" "${OVERWRITE[@]}" || { echo "Preprocessing failed"; exit 1; }
du -sh "${EPHEMERAL_DATA_DIR:-processed_datasets}"/*

# ======================== Smoke test ==========================================
# Load the launcher's settings (its plain VAR=value lines) and its train.py command.
eval "$(grep -E '^[A-Z_]+=' "$CONFIG")"
TRAIN_CMD=$(sed -nE '/^(forward_signals )?python -u train.py/,/[^\\]$/p' "$CONFIG" | sed -E 's/^forward_signals //')
[[ -n $TRAIN_CMD ]] || { echo "No 'python -u train.py' command found in $CONFIG"; exit 1; }

SMOKE_DIR="/tmp/${SLURM_JOB_ID:-$$}/smoke_checkpoint"
mkdir -p "$SMOKE_DIR"
trap 'rm -rf "$SMOKE_DIR"' EXIT
echo "--- Smoke test: $CONFIG config ($DATASET), 20 iterations ---"
eval "$TRAIN_CMD --n_iters 20 --print_freq 10 --track False --resume False \
    --checkpoint_dir $SMOKE_DIR --checkpoint_save_freq 20"
status=$?
echo "--- Smoke test exit $status ---"
[[ $status == 0 ]] && echo "Setup complete: $CONFIG is ready to sbatch."
exit $status
