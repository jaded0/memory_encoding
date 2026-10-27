# K13bigkey — approved lin pilot, confirmation gated

Branch `k13-bigkey`, base `ea22bd5`, no merge. New `mqar2_K_q4` and
`mqar2gap_K_q1` benchmark families use overlapping two-letter keys through last_two.
Legacy task streams and model defaults stay unchanged. Three1024-wide layers,
KRC1 recipe, B16; transformed arms use layers0,1,2 so L0 crosstalk is addressed.
100 addresses do not mean100 independent dimensions (rank19; full-set PR18.18).

Context/motivation/status and frozen preregistration, on manticore:
- `~/Documents/brain/ephemeral-lowrank/queue_2026-10-27/K13 bigkey - context, design and status.md`
- `~/Documents/brain/ephemeral-lowrank/queue_2026-10-27/prereg_drafts/K13bigkey.md`

The prereg was written before the pilot/smoke. Jaden explicitly approved the lin-only
pilot, then confirmation only if the frozen regime/stability/budget gates pass.
Never edit the proposals page. Never turn an unfinished pilot into a positive gate.

## Local validation (manticore only)

```bash
CUDA_VISIBLE_DEVICES= ~/miniforge3/envs/hebby/bin/python -m pytest tests -q
CUDA_VISIBLE_DEVICES=0 timeout 580 ~/miniforge3/envs/hebby/bin/python \
  scratch/k13_bigkey_2026-10-27/smoke.py ~/k13bigkey_prep/smoke
CUDA_VISIBLE_DEVICES=0 timeout 580 ~/miniforge3/envs/hebby/bin/python \
  scratch/k13_bigkey_2026-10-27/analyze.py \
  ~/k13bigkey_prep/smoke/latest_checkpoint.pth --datasets mqar2_32_q4 \
  --batches 1 --raw-records --out ~/k13bigkey_prep/analysis_smoke.json
```

GPU0 only, unfused1080Ti; smoke allocator25% (<3GiB), analysis30% (<4GiB).
Three-iteration full-width/B16/K32 training smoke finished with peak allocated
1,222,862,336bytes. Tiny local data is plumbing evidence, not convergence evidence.
Generate standard datasets fresh on ORC; never copy the smoke synth_datasets there.
Full suite484 passed/443 subtests before the ORC activation regression tests;
final results live in the vault handoff. Analysis smoke passed in11.01s at739.5MiB.

## Exact approved pilot launch (ORC)

Use a **clean checkout of the pushed commit**, not another agent's working tree.
Environment activation on ORC is `source ~/.bashrc; conda activate hebby`.

```bash
cd ~/me_k13b
python bench_tasks.py mqar2_4_q4 mqar2_8_q4 mqar2_16_q4 mqar2_32_q4
mkdir -p logs
export K13B_CODE=$PWD K13B_COMMIT=$(git rev-parse HEAD)
sbatch --export=ALL scratch/k13_bigkey_2026-10-27/pilot.sbatch
```

One array0-3%4, lin seed2718 at K4,8,16,32. A100/cs/qos cs, excludes
cs-1-5 and cs-1-3; per-task HOME inductor cache. Up to200k iterations with75min
training timeout, checkpoint/validation each5000, then bounded frozen analysis.
Each allocation90min; all four hard-capped6 A100-hours. Timeout124 is recorded
explicitly; latest complete checkpoint is analyzed but not called converged.
Root: `~/nobackup/archive/k13_bigkey_2026-10-27/pilot/`.
Each run records code_commit, train.log, pilot_run.json, learning_curve.json,
latest_checkpoint.pth, checkpoints and analysis.json/log. Existing run dirs are
refused, preventing an accidental second pilot budget. Record array ID in local
`~/queue_2026-10-27/jobs_orc.txt` as `JOBID K13bigkey-pilot` for the watcher.

## Evidence and frozen gate

At each checkpoint, `heldout.py` uses the same model and canonical fast-only writes
as training. `analyze.py CKPT --datasets NAME --out FILE --batches 8 --raw-records`
scores128 validation episodes: strict/observed/no_fast, first answers, post-store
suppression, store-only, paired single-store gap, L0/L1/L2 unit-key PR and wrong
cosine distributions, and hypothetical kernels using sparse_screen's observer.
Hypothetical kernels never alter replay state. Source indices are exact; lag buckets
are capped57. Post-store masks preserve inputs and decay, retain the desired store,
and remove later writes including value/next-letter/query-prefix errors. Store-only
also removes pre-store writes. The paired gap is OOD, diagnostic only.

Gate: all-target and first strict recall in [.20,.85], post-store first recall
improves>=.10 and restores>=80% of error, pooled store-key PR L0>=4 and L1>=2.
Choose smallest qualifying K. Require stable final two5000-step curve points within
.03, then earliest point within .02 of final for fixed iterations. No point qualifies
if learning is still moving. Per-episode PR is diagnostic, not the pooled gate.

## Conditional confirmation (do NOT run merely because pilot was submitted)

Once results land, start a **fresh agent** from the vault handoff. Do not babysit
Slurm with an agent. Check all four analyses, curves, exit statuses and PR/kernel
screens. Build a conservative training+scoring runtime estimate with evidence paths;
the most expensive arm must be included. Estimate>40 A100-hours blocks even the core.
No larger-K extension is prepped; core only is safer inside this cap.

```bash
python scratch/k13_bigkey_2026-10-27/gate.py build \
  --pilot-root ~/nobackup/archive/k13_bigkey_2026-10-27/pilot \
  --runtime-estimate /ABSOLUTE/runtime_estimate.json --out /ABSOLUTE/gate.json
# Only if review agrees and every frozen gate passes:
python scratch/k13_bigkey_2026-10-27/gate.py build \
  --pilot-root ~/nobackup/archive/k13_bigkey_2026-10-27/pilot \
  --runtime-estimate /ABSOLUTE/runtime_estimate.json --out /ABSOLUTE/gate.json \
  --approve --approval-reference 'Jaden chat 2026-10-27: try the higher-dimensional-key task; evaluate order two vs expand-and-sparsify'
export K13B_CODE=$PWD K13B_COMMIT=$(git rev-parse HEAD) K13B_GATE_MANIFEST=/ABSOLUTE/gate.json
minutes=$(python -c 'import json,os; print(json.load(open(os.environ["K13B_GATE_MANIFEST"]))["per_run_wall_seconds"]//60)')
sbatch --time="$minutes" --export=ALL scratch/k13_bigkey_2026-10-27/confirm.sbatch
```

Gate validates evidence hashes, exact recipe, all four pilot runs, approved seeds,
and authorization reference, not just two booleans. Read its runtime JSON schema
and allocation cap before submission; confirmation must not exceed40h even by wall
limits. Append each real array ID to jobs_orc.txt. Five arms×five seeds, strict test
scoring primary, paired differences for predictions; validation was used for selection.
No gate pass => launch nothing else; preserve failed evidence and propose a new fix.

For collection use rsync from ORC (checkpoints are large; JSON/log files suffice for
first review). Read final_eval.json's `heldout_strict/recall_acc`; average each arm's
five seeds, report paired seed differences, counts positive, all invalid/failure rows.
The earlier o2 collector is not assumed to understand the new task or analysis schema.
Use the dedicated summary (missing/invalid rows are never silently dropped):

```bash
python scratch/k13_bigkey_2026-10-27/summarize.py "$PILOT_ROOT" --mode pilot
python scratch/k13_bigkey_2026-10-27/summarize.py "$CONFIRM_ROOT" --mode confirm
```

ORC environment caveat: conda's MKL activate hook reads an optional unset variable.
Both sbatch scripts deliberately disable nounset during activation, then restore it.
This was found during CPU dataset preparation, before any GPU allocation/submission.
