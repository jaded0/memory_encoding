#!/bin/bash
#SBATCH --time=0-05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gpus=1
#SBATCH --mem=24G
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --job-name=b2_smoke
#SBATCH --array=0-13
#SBATCH --output=/home/jaded79/b2_2026-10-22/logs/%x_%A_%a.out
# B2 TinyStories bridge, recipe smoke (Claude session 2026-10-22, agent C). Pre-registered in PREREG_fastlayers.md "B2".
# 3000 iterations of TinyStories cropped to 512 characters, one seed, to pick a recipe that trains stably AND uses its fast weights.
source ~/.bashrc >/dev/null 2>&1
conda activate hebby
export EPHEMERAL_DATA_DIR=$HOME/memory_encoding/processed_datasets
t=$SLURM_ARRAY_TASK_ID
common="--dataset roneneldan/tinystories --model_type ephemeral --updater dfa --hidden_size 1024 --num_layers 3 --batch_size 16
  --input_mode last_one --ephemeral_fraction .2 --forget_rate .01 --weight_clamp 1 --enable_recurrence False
  --residual_connection False --positional_encoding_dim 0 --track false --deterministic true --fused_update True
  --optimizer sgd --print_freq 100 --early_stop_window 0 --checkpoint_save_freq 1000 --resume true
  --max_seq_len 512 --n_iters 3000 --seed 7101"
RB="--learning_rate 0.001 --plasticity 10000"        # R-break: c-phi = lr*alpha*frac = 2
WK="--learning_rate 0.0001 --plasticity 15000"       # weak writes at lr 1e-4: c-phi 0.3
WH="--learning_rate 0.001 --plasticity 1500"         # weak writes at lr 1e-3: c-phi 0.3
names=(rb-LN rb-rms rb-LN-rms rb-L01 rb-DC2 rb-L0 wk-plain wk-LN wk-rms wh-plain wh-LN nofast-1e-3 nofast-1e-4 wh-L0)
flags=("$RB --layer_norm True" "$RB --fast_read_norm rms" "$RB --layer_norm True --fast_read_norm rms" "$RB --fast_layers 0,1"
       "$RB --decouple_fast_feedback 2" "$RB --fast_layers 0" "$WK" "$WK --layer_norm True" "$WK --fast_read_norm rms"
       "$WH" "$WH --layer_norm True" "--learning_rate 0.001 --plasticity 0" "--learning_rate 0.0001 --plasticity 0" "$WH --fast_layers 0")
name=smoke-${names[$t]}
echo "task $t -> $name: ${flags[$t]}"; nvidia-smi --query-gpu=name --format=csv,noheader
cd ~/b2_2026-10-22/code
mkdir -p $HOME/.cache/inductor_b2 $HOME/.cache/tmp_b2; export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d $HOME/.cache/inductor_b2/XXXX); export TMPDIR=$(mktemp -d $HOME/.cache/tmp_b2/XXXX)  # cs-1-3/cs-1-5 /tmp is not writable
d=../runs/$name; mkdir -p $d
python -u train.py $common ${flags[$t]} --checkpoint_dir $d
echo "TRAIN EXIT $?"
