#!/bin/bash --login
# replication: array 0 = Q1 (B150 reseed cells, 10 cells), 1,2 = lineage seeds 2718, 4241 (train to 150k then kicks)
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus=1
#SBATCH --mem=30G
#SBATCH --partition=cs,cs2
#SBATCH --qos=cs
#SBATCH --exclude=cs-1-5
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --job-name=twomodes2
#SBATCH --output=/home/jaded79/memory_encoding_benchmarks/two_modes_2026-10-03/logs/%x_%A_%a.out
set -eo pipefail
ROOT=/home/jaded79/memory_encoding_benchmarks/two_modes_2026-10-03
conda activate hebby
set -u
nvidia-smi --query-gpu=name --format=csv,noheader; hostname
export CODE=$ROOT/code3 CKPT=$ROOT/ckpt RUNS=$ROOT/runs PY=$CONDA_PREFIX/bin/python
case $SLURM_ARRAY_TASK_ID in
 0|1|2|3) bash $ROOT/run_cells.sh $ROOT/cells_T$SLURM_ARRAY_TASK_ID.txt ;;
 1|2)
  sd=(0 2718 4241); sd=${sd[$SLURM_ARRAY_TASK_ID]}
  D=$RUNS/L${sd}_train; mkdir -p $D
  if ! grep -q "^=== exit 0" $D/train.log 2>/dev/null; then
    (cd $CODE && WANDB_MODE=disabled HF_DATASETS_OFFLINE=1 HF_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCHINDUCTOR_CACHE_DIR=$(mktemp -d "${TMPDIR:-/tmp}/ind_XXXXXX") \
     $PY -u train.py --model_type ephemeral --updater dfa --enable_recurrence false --fused_update true \
     --dataset 3_palindrome_dataset_vary_length --input_mode last_one --learning_rate 1e-3 --plasticity 1e4 --forget_rate 0.01 \
     --ephemeral_fraction 0.2 --weight_clamp 1 --hidden_size 1024 --num_layers 3 --residual_connection false --positional_encoding_dim 0 \
     --batch_size 16 --track false --deterministic true --resume true --n_iters 150000 --print_freq 500 --trace_loop_every 500 \
     --checkpoint_save_freq 10000 --checkpoint_keep_every 10000 --checkpoint_keep_max 100 --early_stop_window 0 --seed $sd \
     --checkpoint_dir $D >> $D/train.log 2>&1; echo "=== exit $? $(date)" >> $D/train.log)
  fi
  ls $D | head -20
  bash $ROOT/run_cells.sh $ROOT/cells_L${sd}.txt ;;
esac
