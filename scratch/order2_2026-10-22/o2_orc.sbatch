#!/bin/bash
#SBATCH --time=0-14:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus=1
#SBATCH --mem=30G
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --exclude=cs-1-5,cs-1-3
#SBATCH --job-name=o2
#SBATCH --array=0-104
#SBATCH --output=/home/jaded79/order2_2026-10-22/logs/%x_%A_%a.out
# Batch O2 (agent D, 2026-10-22; PREREG_fastlayers.md "Batch O2"). 0-24 KR {lin,par2,par1,rep2,rep1} x 5 seeds, 100k;
# 25-99 mqar_{2,4,8} x 5 arms x 5 seeds, 200k; 100-104 mqar_8 par2t16 x 5 seeds, 200k.
source ~/.bashrc >/dev/null 2>&1
conda activate hebby
export WANDB_MODE=offline CUBLAS_WORKSPACE_CONFIG=:4096:8
t=$SLURM_ARRAY_TASK_ID
seeds=(2718 3141 1618 1414 1732); arms=(lin par2 par1 rep2 rep1); ks=(2 4 8)
if [ $t -lt 25 ]; then data=long_range_memory_dataset; n=100000; arm=${arms[$((t / 5))]}; seed=${seeds[$((t % 5))]}
elif [ $t -lt 100 ]; then i=$((t - 25)); data=mqar_${ks[$((i / 25))]}; n=200000; arm=${arms[$(((i % 25) / 5))]}; seed=${seeds[$((i % 5))]}
else data=mqar_8; n=200000; arm=par2t16; seed=${seeds[$((t - 100))]}; fi
echo "task $t -> $data $arm s$seed"; nvidia-smi --query-gpu=name --format=csv,noheader
cd ~/order2_2026-10-22/code
mkdir -p $HOME/.cache/inductor_o2; export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d $HOME/.cache/inductor_o2/XXXX)
bash scratch/order2_2026-10-22/o2_train.sh $data $n $arm $seed $HOME/nobackup/order2_2026-10-22/runs
