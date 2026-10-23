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
#SBATCH --job-name=bfr2
#SBATCH --array=0-4
#SBATCH --output=/home/jaded79/bfr_2026-10-22/logs/%x_%A_%a.out
# Batch BFR2 LC2x (agent A round 2, 2026-10-23; PREREG_fastlayers.md "Batch BFR2"): the 5 LC ln runs, 600k -> 1.2M, exact continuation into runs2/.
# First submission resumes from the round-1 checkpoint_00600000.pth; a resubmission (after a timeout) resumes from runs2/<name>/latest_checkpoint.pth.
source ~/.bashrc >/dev/null 2>&1
conda activate hebby
t=$SLURM_ARRAY_TASK_ID
seeds5=(620101 620203 620307 620411 620509)
seed=${seeds5[$t]}; name=BFR-LC-ln-s$seed
common="--dataset 3_palindrome_dataset_vary_length --model_type ephemeral --updater dfa --hidden_size 1024 --batch_size 16 --input_mode last_one
  --learning_rate 0.001 --plasticity 10000 --ephemeral_fraction .2 --forget_rate .01 --weight_clamp 1 --enable_recurrence False
  --residual_connection False --layer_norm False --positional_encoding_dim 0 --track false --deterministic true --fused_update True
  --optimizer sgd --print_freq 500 --early_stop_window 0 --grad_norm_clip 0"
echo "task $t -> $name, 600k -> 1.2M (fast_read_norm ln)"; nvidia-smi --query-gpu=name --format=csv,noheader
cd ~/bfr_2026-10-22/code_main
mkdir -p $HOME/.cache/inductor_bfr; export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d $HOME/.cache/inductor_bfr/XXXX)
d=../runs2/$name; mkdir -p $d
if [ -f $d/latest_checkpoint.pth ]; then res="--resume true"; else res="--resume_checkpoint ../runs/$name/checkpoint_00600000.pth"; fi
echo "resume: $res"
python -u train.py $common --num_layers 3 --seed $seed --n_iters 1200000 --checkpoint_dir $d --fast_read_norm ln $res \
  --checkpoint_save_freq 25000 --checkpoint_keep_every 150000 --heldout_eval_every 150000 --heldout_batches 8
rc=$?; echo "TRAIN EXIT $rc"
python -u heldout.py --checkpoint $d/latest_checkpoint.pth --protocols observed strict no_fast --batches 128 --json $d/final_eval.json
echo "EVAL EXIT $?"
