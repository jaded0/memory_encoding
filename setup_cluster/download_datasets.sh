#!/bin/bash --login
# ==============================================================================
# download_datasets.sh - LOGIN NODE ONLY: download raw Hugging Face datasets into
# the HF cache (~/.cache/huggingface). No preprocessing; that runs on a compute
# node in prepare_datasets.sbatch. Network-bound; the only CPU work is HF's
# one-time parquet -> arrow conversion (about a minute for TinyStories).
#
#   setup_cluster/download_datasets.sh [--redo] [dataset ...]
#
# --redo forces a fresh download (download_mode=force_redownload).
# Default datasets: the Hugging Face ones the launchers reference.
# Use the exact spelling the launchers use (roneneldan/tinystories): HF keys its
# cache directory by that string, so roneneldan/TinyStories is a separate copy.
# Usually run through setup_cluster/setup.sh.
# ==============================================================================

cd "$(dirname "$0")/.." || exit 1

REDO=0
if [[ ${1:-} == --redo ]]; then REDO=1; shift; fi
DATASETS=("$@")
[[ ${#DATASETS[@]} -gt 0 ]] || DATASETS=(roneneldan/tinystories)

conda activate hebby || { echo "conda activate hebby failed"; exit 1; }
unset HF_OFFLINE HF_DATASETS_OFFLINE HF_HUB_OFFLINE

python -u - "$REDO" "${DATASETS[@]}" <<'EOF'
import sys, time
from datasets import load_dataset

redo = sys.argv[1] == "1"
for name in sys.argv[2:]:
    start = time.time()
    dataset = load_dataset(name, download_mode="force_redownload" if redo else "reuse_dataset_if_exists")
    sizes = ", ".join(f"{split}: {len(rows)} rows" for split, rows in dataset.items())
    print(f"downloaded {name} ({sizes}) in {time.time() - start:.0f}s")
EOF
