#!/bin/bash
#SBATCH --time=0-12:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus=1
#SBATCH --mem=30G
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --exclude=cs-1-5,cs-1-3
#SBATCH --requeue
#SBATCH --job-name=c2
#SBATCH --array=0-14
#SBATCH --output=/home/jaded79/c2_2026-10-27/logs/%x_%A_%a.out
# Batch C2 (prep agent on omen, 2026-10-26; PREREG_fastlayers.md "Batch C2"): backprop step parity (tasks 0-5)
# and the host-agnostic B_fast arm (tasks 6-14). Arms in c2_arms.sh. Setup on ORC, once:
#   mkdir -p ~/c2_2026-10-27/{logs,runs} && cd ~/c2_2026-10-27 && git clone <repo> code && git -C code checkout <commit>
#   ln -s <synth_datasets> code/synth_datasets      (3_palindrome_dataset_vary_length)
# Launch: CODE_COMMIT=<commit> sbatch --export=ALL,CODE_COMMIT=<commit> c2.sbatch   (subset: --array=0-5)
# A timed-out task requeues and resumes from latest_checkpoint.pth (--resume true).
source ~/.bashrc >/dev/null 2>&1
conda activate hebby
set -u
t=${SLURM_ARRAY_TASK_ID:?}
ROOT=$HOME/c2_2026-10-27; CODE=$ROOT/code
: "${CODE_COMMIT:?set CODE_COMMIT to the commit checked out in $CODE}"
head=$(git -C "$CODE" rev-parse HEAD)
[[ $head == "$CODE_COMMIT"* ]] || { echo "code is at $head, expected $CODE_COMMIT"; exit 1; }
[[ -z $(git -C "$CODE" status --porcelain --untracked-files=no) ]] || { echo "code checkout is dirty"; exit 1; }
source "$CODE/scratch/c2_2026-10-27/c2_arms.sh"
c2_arm "$t" || exit 1
d=$ROOT/runs/$NAME; mkdir -p "$d"
echo "$head" > "$d/code_commit.txt"
echo "task $t -> $NAME on $(hostname), code $head"; nvidia-smi --query-gpu=name --format=csv,noheader
mkdir -p "$HOME/.cache/inductor_c2"; export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d "$HOME/.cache/inductor_c2/XXXX")
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$CODE"
source sweeps/forward_signals.sh
forward_signals python -u train.py "${ARGV[@]}" --n_iters 400000 --checkpoint_dir "$d" >> "$d/train.log" 2>&1
rc=$?; echo "TRAIN EXIT $rc" | tee -a "$d/train.log"
if [[ $rc == 124 ]]; then scontrol requeue "$SLURM_JOB_ID"; exit 0; fi
# A loss early stop is the outcome under test (a collapse), not a failure.
if [[ $rc != 0 ]] && grep -q "^Early stopping: Loss has been" "$d/train.log"; then echo "COLLAPSED $NAME"; exit 0; fi
exit $rc
