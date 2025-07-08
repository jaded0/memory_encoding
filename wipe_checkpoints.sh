#!/usr/bin/env bash
# wipe_checkpoints.sh
# Delete all *.pth checkpoint files under ./checkpoints

set -euo pipefail  # safer bash: exit on errors & undefined vars

ROOT_DIR="checkpoints"

# -type f  → only files
# -name '*.pth' → match PyTorch checkpoints
# -print         → list each file as it’s removed (optional)
# -delete        → actually remove it
find "$ROOT_DIR" -type f -name '*.pth' -print -delete
