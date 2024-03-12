#!/bin/bash

#SBATCH --time=72:00:00   # walltime.  hours:minutes:seconds
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=16000M   # 8G memory per CPU core
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --job-name=memory_encoding
#BATCH --output ./memory_encoding.out
#SBATCH --mail-user jaden.lorenc@gmail.com

# some helpful debugging options
set -e
set -u

nvidia-smi

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
source /apps/miniconda3/latest/etc/profile.d/conda.sh
conda activate hebby
export WANDB_EXECUTABLE=$CONDA_PREFIX/bin/python
export WANDB_MODE=offline

export HF_DATASETS_OFFLINE=1

bash run_training.sh