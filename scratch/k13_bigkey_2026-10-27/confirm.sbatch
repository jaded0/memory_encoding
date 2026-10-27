#!/usr/bin/env bash
# MANUAL submission only after gate.py build + explicit approval; no pilot auto-chain.
# Submit with --time=MINUTES from manifest per_run_wall_seconds / 60.
# Header is the absolute ceiling (25 x 96 minutes = 40 A100-hours), not permission
# to bypass the smaller manifest allocation. The runtime guard below rejects that.
#SBATCH --time=0-01:36:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus=1
#SBATCH --mem=30G
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --exclude=cs-1-5,cs-1-3
#SBATCH --job-name=k13b-confirm
#SBATCH --array=0-24%4
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
source "$HOME/.bashrc" >/dev/null 2>&1
conda activate hebby
: "${K13B_CODE:?clean checkout}" "${K13B_COMMIT:?expected commit}"
: "${K13B_GATE_MANIFEST:?explicitly approved frozen gate manifest}"
cd "$K13B_CODE"
python scratch/k13_bigkey_2026-10-27/gate.py validate "$K13B_GATE_MANIFEST"
python - "$K13B_GATE_MANIFEST" <<'PY'
import json, os, sys
m = json.load(open(sys.argv[1]))
wall = m['per_run_wall_seconds']
assert 25 * wall <= 40 * 3600
# Slurm supplies start/end times in seconds. Fail closed if unavailable; manual
# launch must request the gate-selected walltime, not just a favorable estimate.
actual = int(os.environ['SLURM_JOB_END_TIME']) - int(os.environ['SLURM_JOB_START_TIME'])
assert 0 < actual <= wall, (actual, wall)
PY
read -r data n < <(python - "$K13B_GATE_MANIFEST" <<'PY'
import json, sys
m = json.load(open(sys.argv[1])); print(m['dataset'], m['n_iters'])
PY
)
seeds=(2718 3141 1618 1414 1732); arms=(lin par2 par1 rep2 spar)
t=${SLURM_ARRAY_TASK_ID:?}
[[ $t =~ ^[0-9]+$ && $t -ge 0 && $t -lt 25 ]]
export K13B_PILOT=0
bash scratch/k13_bigkey_2026-10-27/train.sh "$data" "$n" \
  "${arms[$((t / 5))]}" "${seeds[$((t % 5))]}" \
  "${K13B_OUTROOT:-$HOME/nobackup/archive/k13_bigkey_2026-10-27/confirm}"
