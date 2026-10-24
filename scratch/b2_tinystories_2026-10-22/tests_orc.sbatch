#!/bin/bash
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --job-name=b2_tests
#SBATCH --output=/home/jaded79/b2_2026-10-22/logs/%x_%j.out
# B2 (2026-10-22): the full test suite in ORC's pinned hebby env, CPU only.
source ~/.bashrc >/dev/null 2>&1
conda activate hebby
cd ~/b2_2026-10-22/code
CUDA_VISIBLE_DEVICES="" python -m pytest tests/ -q
echo "PYTEST EXIT $?"
