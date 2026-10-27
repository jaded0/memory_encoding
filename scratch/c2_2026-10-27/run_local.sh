#!/bin/bash
# Run C2 tasks on a local GPU, one after another (omen, manticore, deckard).
# usage: run_local.sh CODE_DIR OUT_DIR N_ITERS TASK [TASK ...]
#   smoke (prep, 2026-10-26): run_local.sh ~/worktrees/c2-backprop-parity /tmp/c2_smoke 2000 0 3 6 9 12
# Records the commit next to each run and refuses a dirty checkout.
set -u
code=$1 out=$2 iters=$3; shift 3
source ~/miniforge3/etc/profile.d/conda.sh; conda activate hebby
source "$code/scratch/c2_2026-10-27/c2_arms.sh"
head=$(git -C "$code" rev-parse HEAD)
[[ -z $(git -C "$code" status --porcelain --untracked-files=no) ]] || { echo "code checkout is dirty"; exit 1; }
export CUBLAS_WORKSPACE_CONFIG=:4096:8
for t in "$@"; do
    c2_arm "$t" || exit 1
    d=$out/$NAME; mkdir -p "$d"; echo "$head" > "$d/code_commit.txt"
    echo "task $t -> $NAME ($iters its), code $head"
    (cd "$code" && python -u train.py "${ARGV[@]}" --n_iters "$iters" --checkpoint_dir "$d" > "$d/train.log" 2>&1)
    echo "EXIT $? $NAME" | tee -a "$out/status.txt"
done
