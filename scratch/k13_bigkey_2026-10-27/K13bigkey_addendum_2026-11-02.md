---
title: K13bigkey addendum — confirmation restart after the full-disk failure
date: 2026-11-02
amends: K13bigkey.md (sha256 c547bb9c…, unchanged) and K13bigkey_addendum_2026-10-29.md (unchanged)
status: approved-before-relaunch
---

# K13bigkey addendum, 2026-11-02

Written before the relaunch and before any confirmation run was scored on the test split.
Decision (Jaden, chat 2026-11-02): "restart K13 from scratch after fixing the code to make it a
bit more robust to similar issues so that we don't have to go from scratch again."

## What happened to attempt 1
Launched 2026-10-30 at c7028e6 (`~/k13b_2026-10-28/confirm`). On 2026-10-31 at about 04:45 deckard's
root disk filled: every 5000 iterations each run kept a numbered checkpoint (~373 MB each), and
other experiments were also using the disk. Six runs exited 1. Four spar runs (s2718 and s3141
at ~155k, s1414 and s1618 at ~100–105k) failed while writing a checkpoint or their logs. par2
s2718 and spar s1732 failed at start. The other nine runs (par2 ×4, lin ×5) never started. No
run was scored on the test split, and no `final_eval.json` or `analysis.json` exists. While
diagnosing, the assistant saw only exit codes, tracebacks and one training-interval block of spar
s1414 at 104–105k (training recall ≈ 0.2–0.35 by lag). It did not compare arms. Attempt 1 is
moved to `confirm_attempt1_2026-10-30/` and is reported only as a failed infrastructure attempt.
None of its runs or checkpoints is used in scoring.

## Restart
All 15 runs (lin, par2 and spar × seeds 2718, 3141, 1618, 1414, 1732) start again from iteration
0 with the same gate manifest (profile `deckard3`, the approval reference of 2026-10-29) and the
same recipe, data, iterations (195000), lanes and queue order (spar, par2, lin).

## Changes (infrastructure only; the training computation is unchanged)
1. **Disk guard** (`train.py --min_free_disk_gb`, set to 15). This is checked every 500
   iterations and before each periodic checkpoint. While less space is free, training pauses
   instead of crashing, and a `DISKWAIT` line goes to `confirm_status.txt`. A failed checkpoint
   write is retried; the save is atomic, so the previous checkpoint survives. The guard reads only
   free space. A guarded CPU run equals an unguarded one bit for bit (`tests/test_disk_guard.py`).
2. **Resume instead of restart** (`train.sh`, `K13B_RESUME=1`, confirmation only). A run directory
   that is not complete continues from its `latest_checkpoint.pth`, but only at the same code
   commit and gate manifest. A run that never wrote a checkpoint starts over in place, keeping its
   old log. The training timeout from the gate is the total over all segments
   (`train_seconds`). Every resume is recorded in `resumes.txt` and in `pilot_run.json`
   (`resumes`). Where a segment re-logs a curve point, the learning curve keeps the later value.
   Checked on deckard's A6000s with the real recipe: 40 straight iterations versus 20 + resume +
   20 gave identical model, optimizer, RNG (python, numpy, torch, CUDA), data-stream and
   program state for spar and par2, plus the same loss and held-out strict recall.
   `train.py`'s resume code is the one already tested in `tests/test_seed_resume.py`.
3. **Lane retry** (`deckard_confirm.sh`). A run that fails (any non-zero exit other than the
   timeout code 124) goes back on the queue and resumes, with at most 3 attempts in all; after that
   the lane writes `GIVEUP`. Before starting a task, a lane waits for 40 GB free.
4. **Checkpoint retention:** a numbered checkpoint every 25000 iterations instead of every 5000.
   `latest_checkpoint.pth` and validation are still every 5000. No confirmatory analysis uses
   numbered checkpoints: scoring reads `latest_checkpoint.pth`, and the curve comes from
   `train.log`. Disk per run drops from ~14.5 GB to ~3 GB.

Unchanged: task, recipe, placement, temperature, sparsity, seeds, iteration rule, gate, test
split reserved for final scoring, metrics, BK3 (primary) and the secondary paired comparisons.
Report every run including failures, timeouts and resumes. There is no best-seed selection.
