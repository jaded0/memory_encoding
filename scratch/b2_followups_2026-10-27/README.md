# B2fu: ready for launch review (no production jobs launched)

Branch **b2-followups**, isolated omen worktree `~/me_b2fu`, base `origin/main`
`a4970cfc625a1d7012c9854c7763d583402bf0e6`. Code commit: see `git rev-parse HEAD`
and the vault **B2fu ready.md** (the final hash is recorded there after committing).
Do not merge this branch; main session reviews/merges. Original main checkout untouched.

Pre-registration draft (both parts, predictions before GPU smoke):
`~/Documents/brain/ephemeral-lowrank/queue_2026-10-27/prereg_drafts/B2fu.md`.
The proposals page and canonical PREREG were read only, never edited.

## Prepared arms

* **A**: DEEP-W16 (`--train_wipe_chars 16`), DEEP-F03, DEEP-F10.
  Three independent interventions, no combined arm; 5 seeds × 20k each.
  Jaden approved retrain, but main session must reread his log before launch.
* **B**: REC-NOFAST, REC-DEEP, RNN-BPTT; 5 seeds × 20k each.
  **Needs Jaden's explicit yes.** All have recurrence on. True RNN baseline is
  SimpleRNN/BPTT, Adam lr1e-4, clip1, residual, as fixed in README.
* `arms.sh` is the single recipe table; tests parse every arm through train.parse_args.
  Shared B2 H1024×3, B16, TinyStories crop512, LN, last_one, deterministic.
* `--wipe_every` on main counts **sequences**. New default-off
  `--train_wipe_chars` counts within-story **characters**, wiping only fast state,
  never consolidating slow weights or resetting hidden/time counters. Resume changes refused.
* Recurrent evaluator OFF keeps recurrence and disables only fast writing.
  RESET replays the **full prefix**, wiping only fast entries at mention_start−32.
  WIPE1==OFF is asserted only without recurrence: hidden can preserve prior fast effects.
  SimpleRNN ON/OFF/RESET are identical and fast-key PR is explicitly unavailable.

**Fairness:** equal **logical forward scalar slots** (not equal compute or serialized
state): V73, H1024, d1097, **4,818,024** weights+biases each. Ephemeral fast slots
replace slow slots; their B16 dynamic copies, masks, feedback, traces/plasticity are
extra working/storage overhead, not extra forward slots. REC-NOFAST leaves its .2
masked slots zero and therefore has fewer adaptable slots; RNN has all slots slow
trainable. RNN also has prescribed Adam/residual/clamp differences. This conservative
comparison does not claim equal slow parameter count or a clean updater-only ablation.

## Stage inputs (no training yet)

On ORC, after branch push, create an isolated checkout (or use its reviewed merge):

```bash
ssh orc 'bash -lc "cd ~/memory_encoding && git fetch origin && git worktree add ~/me_b2fu origin/b2-followups -b b2-followups"'
ssh orc 'bash -lc "mkdir -p ~/b2_followups_2026-10-27/{logs,inputs/original}"'
```

Copy the **original** B2 `val8192.json` and `names.json` into
`~/b2_followups_2026-10-27/inputs/` (original ORC location was
`~/b2_2026-10-22/b2/`). Do not regenerate names or replace the validation prefix.
Record SHA256s; each run also records both hashes and code commit.
Stage `inputs/SHA256SUMS` containing the two **original-source-verified** hashes
with relative names `val8192.json` and `names.json`; launchers verify it, not merely
record new hashes. Obtain the manifest from the original B2 source, not an unverified
replacement sample. Actual frozen hashes could not be recovered during headless prep.
Link/copy original 20k NOFAST and DEEP checkpoints (5 seeds each) to
`inputs/original/{NOFAST,DEEP}-s{7201,7302,7403,7504,7605}.pth`.
Original ORC run directories `~/b2_2026-10-22/runs/ARM-sSEED/`, and deckard has
s7201; verify filenames/location rather than guessing.
Store a source-verified `inputs/original/SHA256SUMS` with relative checkpoint names;
control_eval verifies it and records all ten checkpoint hashes. These manifests
and original checkpoint identities are a **required launch preflight**, still open.
Existing preprocessed
TinyStories first1M cache must be at `$HOME/memory_encoding/processed_datasets`
or set `EPHEMERAL_DATA_DIR`; do not build a different subset for production.

**External preflight not completed:** headless `ssh orc 'bash -lc "..."'`
returns `Permission denied (keyboard-interactive)`; deckard name resolution also
failed on this host. No sbatch was issued. Refresh ORC login and verify A100 GRES,
cache availability and staged original inputs before launching. The script uses
`--gpus=a100:1`, cs partition/qos, exclude cs-1-5 and HOME inductor/tmp caches.

## Exact launch commands (only after approval)

Dry-run on ORC after authentication and staging (does not submit a job):

```bash
ssh orc 'bash -lc "cd ~/me_b2fu && sbatch --test-only --export=ALL,B2FU_PART=A,B2FU_APPROVED=A scratch/b2_followups_2026-10-27/b2fu.sbatch"'
```

After the main session rereads Jaden's log, Part A:

```bash
ssh orc 'bash -lc "cd ~/me_b2fu && sbatch --export=ALL,B2FU_PART=A,B2FU_APPROVED=A scratch/b2_followups_2026-10-27/b2fu.sbatch"'
```

Only after a **separate explicit yes**, Part B:

```bash
ssh orc 'bash -lc "cd ~/me_b2fu && sbatch --export=ALL,B2FU_PART=B,B2FU_APPROVED=B scratch/b2_followups_2026-10-27/b2fu.sbatch"'
```

Each array has 15 tasks, arm=t%3, seed=t/3; max 3 concurrent to respect shared
fairshare. The gate is deliberate: run.sh refuses if the matching B2FU_APPROVED
is absent. A timeout can be resubmitted with `sbatch --array=TASK ...`; resume
uses unchanged settings and the latest checkpoint. No automatic resubmission or
extension is armed. Failures propagate through `set -euo pipefail`.

Local/manual equivalent (production-length, **do not run during prep**):
`B2FU_APPROVED=A PYTHON=~/miniforge3/envs/hebby/bin/python bash scratch/b2_followups_2026-10-27/run.sh DEEP-W16 7201`.
Set CUDA_VISIBLE_DEVICES explicitly for the chosen host/GPU.

### Evaluation and scoring

run.sh trains then evaluate.sh runs the original b2_eval modes: final tiny4096,
horizon1024 R2..256, curve256 at every retained 2500-step checkpoint. All crop512;
ON is unrestricted even for a W16-trained model. Scoring consumes the same
final JSON keys as B2's b2_aggregate.py, plus horizon CE keys.

Stage/re-evaluate **old** controls, once, on an allocated GPU (eval only, still gated):

```bash
B2FU_APPROVED=A bash scratch/b2_followups_2026-10-27/control_eval.sh
python scratch/b2_followups_2026-10-27/score.py --root ~/b2_followups_2026-10-27 --part A --out scores-A.json
python scratch/b2_followups_2026-10-27/score.py --root ~/b2_followups_2026-10-27 --part B --out scores-B.json
```

Control eval also measures original DEEP WIPE16 on **4096** stories so A5 never
compares mismatched validation counts. Score fails closed if any seed/control is
missing or not at20k. It reports paired seed values, mean/SD/95% t intervals, tax
by class, accumulation penalty, and old-DEEP improvements. Re-mention/PR/curves
remain in the unchanged evaluator output schemas for the full prediction scorecard.

## Resource estimate / placement

Prefer **ORC A100**, one arm per GPU; arrays capped at3. Historical B2 at the
same size/data/20k costs **6–7 GPU-h/train**: A15 trains =90–105, plus final,
curves, horizon and old-control evaluation roughly15–30 => **105–135 GPU-h**.
Part B is provisional **110–220 GPU-h** including recurrence/BPTT and slower
full-prefix RESET evaluation; update using measured production-node throughput.
These are resource reservations, not a claim smoke predicts convergence.
On omen (mobile4090) the full512 smoke is roughly .3 it/s: ≥17 GPU-h per20k arm,
so it is a smoke/overflow host, not the preferred production placement. The old
deckard estimate was ~17h/run with three sharing a GPU. Avoid sharing for timing
comparisons; report observed runtime and peak memory alongside CE.

## Validation evidence

* GPU smoke is bounded to590s per arm including **both** tiny and horizon eval,
  100 training iterations at **full H1024/B16/crop512**; evaluations use **16 stories
  at length64** (production eval is512). Seed7901, real TinyStories first100
  train rows, first32 validation rows staged separately via HF rows API. The small
  loader is patched only by smoke_train.py; never write this sample into the
  production cache. CUDA allocator cap .25 of omen's16GiB (<5GB including context).
* Logs/checkpoints/sample JSON outside vault/repo: `/tmp/opencode/b2fu-smoke/`.
  Committed `smoke_evidence.json` is the lightweight timings/memory/status record;
  scores are not used to select arms. Reproduce with staged sample JSON:
  `B2FU_SMOKE_ROOT=/tmp/opencode/b2fu-smoke bash scratch/b2_followups_2026-10-27/smoke.sh`.
* CPU suite has the **pre-existing golden fixture environment mismatch**:
  fixture torch2.5.1 vs installed2.5.1+cu121. Origin/main has the same failure.
  Removing only the version suffix in a one-off process exposes a tiny existing
  fixture drift (−.06160417199 vs −.06160435081); no fixtures were regenerated and
  no tolerances changed. Stronger check: **all seven default characterization
  traces are byte-identical to origin/main**, 343,122 bytes, in the same env.
* New flag and recurrent-evaluator/launcher CPU tests pass; exact final suite
  counts and GPU evidence are in the vault ready note. bash -n and diff --check pass.

## Open questions for Jaden / main session

1. **Approve Part B?** Recurrence-on comparison is prepped, not authorized by this note.
2. Accept logical-forward-slot matching (with RNN's extra usable slow slots and
   fixed Adam/residual recipe), rather than a new compute-matched tuning study?
3. Restore ORC auth and stage original B2 inputs/controls; confirm typed A100 GRES
   with test-only. The submission scripts are syntax-tested locally, not cluster-tested.
4. Decide launch order/fairshare alongside the other overnight prep branches.

No production jobs, merges, proposal-page edits, or long-running watchers were started.
