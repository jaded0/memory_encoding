#!/bin/bash
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus=1
#SBATCH --mem=30G
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --exclude=cs-1-5,cs-1-3
#SBATCH --job-name=s12_lndc2
#SBATCH --array=0-4
#SBATCH --output=/home/jaded79/s12_2026-10-27/logs/%x_%A_%a.out
# S12 follow-up 1 (prep 2026-10-27; PREREG "Batch S12F", L1-L5): ln + DC2 on the 3pal late collapse.
# = Batch BFR's LC ln arm (batchBFR.sbatch tasks 0-4: the Batch Q F7 recipe + --fast_read_norm ln) plus --decouple_fast_feedback 2,
# same 5 seeds, so each run pairs with BFR-LC-ln-s<seed> (ORC ~/bfr_2026-10-22/runs/). Code: $BASE/code (git archive; see its COMMIT).
# Resubmitting the same file resumes every unfinished task from runs/<name>/latest_checkpoint.pth (--resume true).
# 1.2M instead of 600k: change the default on the next line to 1200000 (12 h covers ~600-900k on an A100: 1.2M needs one resubmit).
N_ITERS=${N_ITERS:-600000}
BASE=${BASE:-$HOME/s12_2026-10-27}
CKPT_EVERY=${CKPT_EVERY:-25000}  # smoke tests only: N_ITERS must be a multiple (latest_checkpoint.pth is saved on these)
source ~/.bashrc >/dev/null 2>&1; source ~/miniforge3/etc/profile.d/conda.sh 2>/dev/null
conda activate hebby
set -eo pipefail
t=${SLURM_ARRAY_TASK_ID:-0}
seeds5=(620101 620203 620307 620411 620509)
seed=${seeds5[$t]}; name=S12-LC-lnDC2-s$seed
common="--dataset 3_palindrome_dataset_vary_length --model_type ephemeral --updater dfa --hidden_size 1024 --batch_size 16 --input_mode last_one
  --learning_rate 0.001 --plasticity 10000 --ephemeral_fraction .2 --forget_rate .01 --weight_clamp 1 --enable_recurrence False
  --residual_connection False --layer_norm False --positional_encoding_dim 0 --track false --deterministic true --fused_update True
  --optimizer sgd --print_freq 500 --early_stop_window 0 --resume true --grad_norm_clip 0"
cd $BASE/code || exit 1
echo "task $t -> $name ($N_ITERS its, ln + DC2), code $(cat COMMIT 2>/dev/null)"; nvidia-smi --query-gpu=name --format=csv,noheader
mkdir -p $HOME/.cache/inductor_s12; export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d $HOME/.cache/inductor_s12/XXXX)
d=$BASE/runs/$name; mkdir -p $d
python -u train.py $common --num_layers 3 --seed $seed --n_iters $N_ITERS --checkpoint_dir $d \
  --fast_read_norm ln --decouple_fast_feedback 2 \
  --checkpoint_save_freq $CKPT_EVERY --checkpoint_keep_every 50000 --heldout_eval_every 50000 --heldout_batches 8
rc=$?; echo "TRAIN EXIT $rc"
[ $rc -eq 0 ] || exit $rc
python -u heldout.py --checkpoint $d/latest_checkpoint.pth --protocols observed strict no_fast --batches 128 --json $d/final_eval.json
echo "EVAL EXIT $?"
# Follow-up 2's valid no_fast on the final checkpoint (a few minutes).
python -u scratch/s12_2026-10-27/ln_null_eval.py --checkpoint $d/latest_checkpoint.pth --batches 128 --json $d/null_eval.json
echo "NULL EXIT $?"
