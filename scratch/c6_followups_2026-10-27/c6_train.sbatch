#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus=1
#SBATCH --mem=30G
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --exclude=cs-1-5,cs-1-3
#SBATCH --job-name=c6_train
#SBATCH --signal=B:USR1@600
#SBATCH --output=/home/jaded79/c6fu_2026-10-27/logs/%x_%A_%a.out
# C6 follow-ups 2 and 6 (prep agent, 2026-10-27; PREREG_fastlayers.md "Batch C6F"): one training chunk of task SLURM_ARRAY_TASK_ID of tasks.sh.
# A run that needs more than one day continues in the next chunk (submit.sh chains chunks with --dependency=aftercorr): USR1 10 min
# before the limit stops train.py with a checkpoint, and this script exits 0 so the next chunk starts. Submit with submit.sh, not directly.
source ~/.bashrc >/dev/null 2>&1
conda activate hebby
cd ~/c6fu_2026-10-27/code
echo "task $SLURM_ARRAY_TASK_ID, code $(cat COMMIT)"; nvidia-smi --query-gpu=name --format=csv,noheader
mkdir -p $HOME/.cache/inductor_c6; export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d $HOME/.cache/inductor_c6/XXXX)
source sweeps/forward_signals.sh
forward_signals bash scratch/c6_followups_2026-10-27/run_c6.sh train $SLURM_ARRAY_TASK_ID
exit 0
