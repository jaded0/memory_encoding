---
title: K13r2 ready for review — NOT ready to launch
date: 2026-10-27
status: prep-blocked-regime
reading: unread
reading_priority: 2
reading_focus: "Lin interference is real, but L2 geometry fails the predeclared spread gate."
project: ephemeral-weights
---

# K13r2: prep complete; regime gate failed

**No real runs, submissions, or main merge. Do not launch the 20-run array.**
Branch `k13-crosstalk`, worktree `manticore:~/me_k13x`, code/evidence commit
`4a4cb898fb7e143758ef73ff2696561192f8c446`. A later documentation-only commit
adds this handoff; `git rev-parse origin/k13-crosstalk` identifies the final branch tip.
The proposals page was only read and was not edited.

## Ready
- Opt-in seeded `kr_load_tasks.py` datasets: `long_range_memory_dataset_load_D` and
  delay-matched `gap_D`, D=0,4,8,16,32,48. Existing KR generation/model path unchanged.
  One answer per episode, restricted to `123,.` (chance .20), no extra query markers.
- Generator/registration/charset/chance/lag tests and screening-gate tests.
- Frozen-scaffold screening, identical round-1 strict scoring/key kernels, plus unit-key PR
  and wrong-key cosine tails. Committed raw JSON under `evidence/`.
- Four fail-closed launch arms: lin, par2, par1, rep2; five round-1 seeds. ORC array 0-19,
  max four concurrent tasks, cs/qos cs, excluded cs-1-5 and cs-1-3, A100 runtime guard,
  per-task `$HOME` inductor cache, exact clean-commit check and per-run commit recording.
- Round-1 `o2_collect.py`/`o2_summary.py` now recognize the new dataset names and final iteration.
- Vault preregistration draft: `ephemeral-lowrank/queue_2026-10-27/prereg_drafts/K13r2.md`.
  Written before screening results; primary prediction: par2 − lin >=.05, P=.40.

## Smoke evidence: interference exists, but the full desired regime is unverified
Frozen round-1 lin s2718 checkpoint copied **read-only** from ORC
`~/nobackup/archive/order2_2026-10-22/runs/KR-lin-s2718/latest_checkpoint.pth`
to `manticore:~/k13r2_prep/KR-lin-s2718.pth`. SHA256 is in every evidence JSON.
Round-1 source commit was 2ccec4e; replay uses current main-derived model code. The round-1
pre-/post-rebase identity caveat remains; this is not a new training replication.

128 episodes per point; frozen slow weights; same `o2_keys.py` measurements as round 1.
Selective replay suppresses all post-store writes but keeps inputs, timing and forgetting.

| D | lin strict | gap control | selective writes | L1 PR | L2 PR | L1 mean abs wrong cos | L2 mean abs wrong cos |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.0000 | — | — | 2.234 | 1.907 | .451 | .363 |
| 4 | .8984 | 1.0000 | 1.0000 | 1.979 | 1.482 | .457 | .561 |
| 8 | .5625 | 1.0000 | 1.0000 | 1.902 | 1.361 | .452 | .612 |
| 16 | .4219 | .0000 | 1.0000 | 2.012 | 1.293 | .439 | .650 |
| 32 | .3359 | .0000 | 1.0000 | 2.347 | 1.323 | .413 | .665 |

At D=8, suppressing interfering writes restores 1.000 with identical forgetting/inputs;
the matched zero-gap control is also 1.000. L1 linear signal share falls .540 → .183;
L1 order-two/order-one key ratios are 5.021/2.590. This is encouraging evidence that
interference binds in the *frozen OOD scaffold*. However, **L2 PR falls below the preregistered
1.5 spread threshold** (1.361 at D=8). D=4 is both above the .85 ceiling gate and just below
the PR threshold. D>=16 additionally fails the delay-control gate. No load qualifies.
The criterion was not relaxed after seeing results; `screen.py` returns selected_load=null.
Neither a defensible trained-load iteration budget nor a production load has been selected.

GPU0 was selected at start (51 MiB used versus GPU1 3540 MiB). All our work stayed on GPU0;
no other processes were touched. Unfused path because GPU0 is a 1080 Ti. The 13 screening
points totaled 143.6 seconds of timed replay, max 25.1 seconds per point, max allocated
827.4 MiB; allocator capped at 30% of the 11 GiB device, below 4 GB. Startup time is additional.
A separate **3-iteration** full-width/B16 training smoke on load_8 completed training,
checkpoint, heldout.py observed/strict/no_fast, and o2_keys.py (one batch each). It only
checks plumbing, not learning or predictions; outputs remain `~/k13r2_prep/train_smoke/`.
Screening JSON `code_commit` records base a4970cf because preparation was then uncommitted;
the replay/generator source is pinned by the code/evidence commit above.

Final CPU suite: **439 passed, 438 subtests passed**, 3 existing FutureWarnings, 44.05 seconds;
golden/identity tests pass. Log: `~/k13r2_prep/tests_complete.log`.
Shell syntax and Python compilation pass. Production launcher correctly refuses absent approval
manifest without creating a run directory. No `sbatch` or `sbatch --test-only` was invoked.

## Exact reproduction / conditional launch commands
From the worktree on manticore, screening only (each point timeout 580 seconds):
```bash
cd ~/me_k13x
CUDA_VISIBLE_DEVICES=0 bash scratch/k13_crosstalk_2026-10-27/smoke.sh \
  ~/k13r2_prep/KR-lin-s2718.pth ~/k13r2_prep/recheck
CUDA_VISIBLE_DEVICES= ~/miniforge3/envs/hebby/bin/python -m pytest tests -q
python3 scratch/k13_crosstalk_2026-10-27/screen.py \
  scratch/k13_crosstalk_2026-10-27/evidence
```

The following is a **template, not a launch recommendation**. Only after a newly verified
regime, fixed iterations, a fresh log read and Jaden's launch gate, on ORC:
```bash
cd "$K13_CODE"                 # clean checkout of approved branch commit
python kr_load_tasks.py "$K13_DATASET"
mkdir -p logs
export K13_CODE K13_DATASET K13_ITERS K13_COMMIT K13_GATE_MANIFEST
sbatch scratch/k13_crosstalk_2026-10-27/orc.sbatch
```
All five exported variables must be filled; K13_COMMIT is a full hash. Gate JSON must contain
`regime_verified:true`, `launch_approved:true`, `dataset`, `n_iters`, `approval_reference`,
and `pilot_evidence`. **No such manifest was issued.** Script paths are relative to repo root.
For approved single-host sequential execution: `bash scratch/k13_crosstalk_2026-10-27/train.sh
"$K13_DATASET" "$K13_ITERS" par2 2718 "$OUTROOT"` with the same gate env.

Score after collecting full runs (not screening JSON):
```bash
python scratch/order2_2026-10-22/o2_collect.py "$OUTROOT"/* > k13r2_runs.jsonl
python scratch/order2_2026-10-22/o2_summary.py k13r2_runs.jsonl > k13r2_summary.md
```

## GPU-hours and decision
No production GPU-hours requested: validity gate failed. For planning only, a 100k reference
budget at D=8 would be approximately 25-35 A100 GPU-hours for 20 runs plus scoring, scaled
from round-1 KR's ~50 min/run by 13/8.5 sequence length; not a measured new throughput or
a pilot-selected budget. ORC A100 array concurrency <=4 is preferable; respect shared fairshare.
Manticore GPU0 is unsuitable for fused production; no deckard/omen launch was attempted.

==Should the next prep task build a genuinely higher-dimensional key/pair load task, rather
than accepting D=8's L2 PR=1.36 as "spread"?== D=8 provides a causal interference screen,
but calling the whole two-layer regime verified would overstate the evidence. A trained-load
pilot and a fixed iteration budget are still needed before the 4-arm/5-seed confirmatory run.
