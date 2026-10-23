#!/bin/bash
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus=1
#SBATCH --mem=30G
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --exclude=cs-1-5,cs-1-3
#SBATCH --job-name=bfr2_kr
#SBATCH --array=0-9
#SBATCH --output=/home/jaded79/bfr_2026-10-22/logs/%x_%A_%a.out
# Batch BFR2 KRx (agent A round 2, 2026-10-23; PREREG "Batch BFR2"): plain ln (0-4) and DC2 ln (5-9), seeds 2718 3141 1618 1414 5772, 30k -> 100k.
# Resumes from the round-1 30k checkpoint (plain ln's copied from manticore into runs_kr/) into runs2_kr/; a resubmission resumes from runs2_kr/<name>.
source ~/.bashrc >/dev/null 2>&1
conda activate hebby
t=$SLURM_ARRAY_TASK_ID
seeds=(2718 3141 1618 1414 5772); arms=(plain DC2)
arm=${arms[$((t/5))]}; seed=${seeds[$((t%5))]}; name=kr_${arm}_ln_s$seed
extra="--fast_read_norm ln"; [ $arm = DC2 ] && extra="--decouple_fast_feedback 2 --fast_read_norm ln"
echo "task $t -> $name, 30k -> 100k"; nvidia-smi --query-gpu=name --format=csv,noheader
cd ~/bfr_2026-10-22/code_kr
mkdir -p $HOME/.cache/inductor_bfr; export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d $HOME/.cache/inductor_bfr/XXXX)
d=../runs2_kr/$name; mkdir -p $d
if [ -f $d/latest_checkpoint.pth ]; then res="--resume true"; else res="--resume_checkpoint ../runs_kr/$name/checkpoint_00030000.pth"; fi
echo "resume: $res" >> $d/train.log
python -u train.py --dataset long_range_memory_dataset --model_type ephemeral --updater dfa \
  --hidden_size 1024 --num_layers 3 --batch_size 16 --input_mode last_one --learning_rate 0.001 \
  --plasticity 10000 --ephemeral_fraction .2 --forget_rate .01 --weight_clamp 1 --enable_recurrence False \
  --residual_connection False --layer_norm False --positional_encoding_dim 0 --track false \
  --deterministic true --fused_update True --optimizer sgd --grad_norm_clip 0 --print_freq 100 \
  --early_stop_window 0 --checkpoint_save_freq 10000 --checkpoint_keep_every 10000 --seed $seed --n_iters 100000 \
  --heldout_eval_every 0 --checkpoint_dir $d $res $extra >> $d/train.log 2>&1
echo "EXIT $?" >> $d/train.log
python heldout.py --checkpoint $d/latest_checkpoint.pth --protocols observed strict no_fast --batches 128 \
  --json $d/heldout.json > $d/heldout.log 2>&1
echo "EVAL EXIT $?"
