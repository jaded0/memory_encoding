#!/bin/bash --login
# ==============================================================================
# setup.sh - one-time cluster setup, run from the LOGIN NODE:
#
#   setup_cluster/setup.sh [--redo] [--wait] [dataset ...]
#
# 1. download_datasets.sh here (network only, light on CPU)
# 2. sbatch prepare_datasets.sbatch (test QOS: preprocess offline + GPU smoke test)
#
# --redo  re-download the raw datasets and overwrite the processed-data directory
# --wait  block until the prepare job finishes, then print the end of its log
# Default dataset: roneneldan/tinystories.
# ==============================================================================

cd "$(dirname "$0")/.." || exit 1

REDO=0
WAIT=0
DATASETS=()
for arg in "$@"; do
    case $arg in
        --redo) REDO=1 ;;
        --wait) WAIT=1 ;;
        -*) echo "unknown option $arg (use --redo, --wait)"; exit 1 ;;
        *) DATASETS+=("$arg") ;;
    esac
done

REDO_FLAG=()
[[ $REDO == 1 ]] && REDO_FLAG=(--redo)
setup_cluster/download_datasets.sh "${REDO_FLAG[@]}" "${DATASETS[@]}" || { echo "Download failed"; exit 1; }

WAIT_FLAG=()
[[ $WAIT == 1 ]] && WAIT_FLAG=(--wait)
if [[ $WAIT == 1 ]]; then
    echo "Waiting for the prepare job (log: prepare_datasets_<jobid>.out) ..."
fi
job=$(REDO=$REDO sbatch --parsable "${WAIT_FLAG[@]}" setup_cluster/prepare_datasets.sbatch "${DATASETS[@]}")
status=$?
job=${job%%;*}
[[ -n $job ]] || { echo "sbatch failed"; exit 1; }
log="prepare_datasets_${job}.out"
if [[ $WAIT == 1 ]]; then
    tail -n 25 "$log"
    echo "Prepare job $job finished (exit $status); full log: $log"
    exit $status
fi
echo "Submitted prepare job $job; log: $log (you'll get an email when it ends)"
