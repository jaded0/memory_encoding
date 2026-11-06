#!/usr/bin/env bash
# From manticore, after the primary commits/pushes. Preparation only: never arms any lane.
# Usage: bash scratch/k15_2026-11-05/stage_deckard.sh FULL_COMMIT [--krx-supported]
set -euo pipefail
commit=${1:?full pushed commit}; krx=${2:-}
[[ $commit =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ -z $krx || $krx == --krx-supported ]] || exit 2
ssh deckard "bash -s -- '$commit'" <<'REMOTE'
set -euo pipefail
commit=$1; root=$HOME/k15_2026-11-05
export PATH="$HOME/miniforge3/bin:$PATH"
[[ ! -e $root/PREPARED && ! -e $root/ARMED && ! -e $root/code ]] || {
  echo 'Existing checkout/preparation: inspect manually, never relaunch or replace' >&2; exit 2;
}
[[ ! -s $root/status.txt ]] || exit 2
(( $(df --output=avail -B1 "$HOME" | tail -1) >= 100 * 1024**3 )) || exit 2
mkdir -p "$root/runs" "$root/logs"
touch "$root/status.txt"
git clone --no-checkout https://github.com/jaded0/memory_encoding.git "$root/code"
git -C "$root/code" checkout --detach "$commit"
src=$HOME/k13b_2026-10-28/code_d4/synth_datasets/mqar2_16_q4
dst=$root/code/synth_datasets/mqar2_16_q4
mkdir -p "$(dirname "$dst")"
cp -a "$src" "$dst"
(cd "$src"; find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$root/k13_data_source.sha256"
(cd "$dst"; sha256sum -c "$root/k13_data_source.sha256")
cp "$HOME/k13b_2026-10-28/confirm_manifest.json" "$root/k13_confirm_manifest.json"
REMOTE
if [[ $krx == --krx-supported ]]; then
  # Do NOT generate data. This is BFR's regenerated RN key-recall dataset.
  src=$HOME/bfr_kr_2026-10-22/code/synth_datasets/long_range_memory_dataset
  [[ -f $src/dataset_dict.json ]] || exit 2
  (cd "$src"; find . -type f -print0 | sort -z | xargs -0 sha256sum) | \
    ssh deckard 'cat > ~/k15_2026-11-05/krx_data_source.sha256'
  rsync -a "$src/" deckard:k15_2026-11-05/code/synth_datasets/long_range_memory_dataset/
  ssh -n deckard 'cd ~/k15_2026-11-05/code/synth_datasets/long_range_memory_dataset && sha256sum -c ~/k15_2026-11-05/krx_data_source.sha256'
fi
ssh -n deckard "cd ~/k15_2026-11-05/code; ~/miniforge3/envs/hebby/bin/python scratch/k15_2026-11-05/stage_init.py --root \$HOME/k15_2026-11-05 --commit '$commit' $krx"
echo 'PREPARED ONLY: manifest launch_approved=false; no lane armed.'
