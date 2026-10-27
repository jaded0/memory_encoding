# Batches F9 and S7 (queue 2026-10-27): launchers, prep agent hand-back

Specs: proposals page `### F9 · DC2 with the weight clamp off` and `### S7 · Slow weight decay`. Pre-registration drafts:
vault `ephemeral-lowrank/queue_2026-10-27/prereg_drafts/F9.md` and `S7.md`. **Nothing has been launched.**

Branch `f9-s7-launchers` (from origin/main 364a07f). Code staged at **3b730f2** (`CODE_COMMIT` in each code dir); later
commits on the branch touch only this README. No new model code: `--decouple_fast_feedback`, `--weight_clamp 0`,
`--slow_weight_decay` and `--early_stop_window` are already on main. CPU suite on the branch: 411 passed.

## What is ready

| Test | Runs | Where | Code dir (staged) | Data (3pal draw) |
|---|---|---|---|---|
| F9 | V2's DC2 R-break command, `--weight_clamp 0`; seeds 620101 620203 620307 × 600k | ORC, `f9.sbatch` (array 0-2) | `orc:~/f9s7_2026-10-27/code_v2data` | `~/memory_encoding_data/current` (V2's) |
| S7 part b (S7D) | plain R-break (battery A1 plain) + `--slow_weight_decay 3e-6`; same seeds × 600k | ORC, `s7d.sbatch` (array 0-2) | `orc:~/f9s7_2026-10-27/code_batterydata` | `~/battery_2026-10/datasets` (battery's) |
| S7 part a (R4ext) | R4 (decay 3e-6 from B_late-s3141 150k) 300k → 600k, exact continuation | deckard, `s7_r4ext.sh` | `deckard:~/f9s7_2026-10-27/code` | loop_trace_2026-09-30 (R4's) |

Files: `run_arm.sh` (one from-scratch run + final `heldout.py`, used by both sbatch files), `f9.sbatch`, `s7d.sbatch`,
`s7_r4ext.sh`, `stage_code.sh` (git archive → host, CODE_COMMIT, data link), `score.py` (V2's measures: onset, held-out,
readout shares in range(B_2) / range(B_fast), trunk-2 slow SV, per-layer max |slow w|).

Divergence guard (all three): `--early_stop_window 100` at print 500 = stop after 50k iterations of windows > 5. V2 used 0.
The longest recovered excursion at R-break is ~29 windows (battery A1 plain s620411, 5k-19k); V2's onset rule needs 5. NaN/inf
always stops. A guard-stopped run is not resumed by a resubmission.

R4 checkpoint: found. `deckard:~/overnight_2026-10-01/runs/INT2_R4/latest_checkpoint.pth` (iter 300000, code_version 12) is the
only R4 checkpoint left (INT_R4's and the numbered copies were deleted on 10-01). `s7_r4ext.sh` copies it into the run
directory before training and never writes to the source directory.

## Launch commands (after Jaden's go)

```bash
ssh orc 'bash -lc "cd ~/f9s7_2026-10-27/code_v2data && sbatch scratch/f9_s7_2026-10-27/f9.sbatch"'
ssh orc 'bash -lc "cd ~/f9s7_2026-10-27/code_batterydata && sbatch scratch/f9_s7_2026-10-27/s7d.sbatch"'
ssh jaden@deckard 'cd ~/f9s7_2026-10-27/code && setsid nohup bash scratch/f9_s7_2026-10-27/s7_r4ext.sh 0 > ../logs/s7_r4ext.out 2>&1 < /dev/null &'
```
(deckard GPU 0 or 1, whichever is free.) A timed-out ORC task: resubmit the same sbatch with `--array=<task>`, and it resumes.

Scoring:
```bash
ssh orc 'bash -lc "conda activate hebby; cd ~/f9s7_2026-10-27/code_v2data && python scratch/f9_s7_2026-10-27/score.py ../runs/Rbreak-F9-s* > ../f9_scores.jsonl && python scratch/f9_s7_2026-10-27/score.py ../runs/Rbreak-S7D-s* > ../s7d_scores.jsonl"'
ssh jaden@deckard 'source ~/miniforge3/etc/profile.d/conda.sh; conda activate hebby; cd ~/f9s7_2026-10-27/code && python scratch/f9_s7_2026-10-27/score.py --ref150 ~/overnight_2026-10-01/analysis/ckptB/checkpoint_00150000.pth ../runs/R4ext-s3141 > ../r4ext_scores.jsonl'
```
References: V2 DC2 per seed in `orc:~/fastlayers_2026-10-13/v2_final.jsonl` / PREREG "Batch V2 OUTCOMES"; battery A1 plain
late onsets 81.5k / 259.5k / 304.5k (seeds 620101/203/307; train.logs on deckard `~/battery_2026-10/runs/`).

## Cost and placement

- F9: 3 × ~2.7 A100-h = ~8 A100-h, ORC (6 h limit per task, resumable).
- S7D: 3 × ~2.7 A100-h = ~8 A100-h, ORC.
- R4ext: one A6000 for ~1.5-2.5 h (INT2_R4 ran at 67-70 it/s with tracing). Deckard, because R4 ran on an A6000 and the
  trajectory is hardware-sensitive (see below).
- `sbatch --test-only` (10-27 ~00:50): both arrays valid; the scheduler's start estimate was 2026-11-03 (49 jaded79 jobs in
  the queue at the time). Omen (4090) could take S7D or F9 if ORC is slow; run_arm.sh works anywhere the code is staged.

## Smoke-test evidence (manticore, 10-27)

- R4 reproduction, B 150k → 152.5k with R4's exact flags, 1080 Ti: main (b6f3449) and R4's own `code_intervention`
  (e1b5b90 + inert lr override) give identical interval metrics and bit-identical loop traces (14 shared trace arrays; main
  logs 2 extra). Interval loss 1.889 vs R4's logged 2.865 on deckard: the difference is the GPU (same on both codes), so the
  exact R4 trajectory continues only on an A6000.
- R4ext launcher, 300k → 305k (SRC/RUNDIR/ITERS overrides): resumes at 300001 with seed 3141 from the checkpoint, only the
  intended config diffs, loss 1.21-1.39 (one window 2.57), checkpoint and `final_eval.json` written (strict 0.378).
- Existing checkpoints, `heldout.py` 128 batches: B 150k strict 0.542; R4 300k strict 0.411. score.py: share in range(B_2)
  0.002 → 0.150, trunk-2 SV 16.5 → 6.46 (B 150k → R4 300k).
- run_arm.sh, 1k iterations, seed 620101, checkpoints every 500 (`SAVE_FREQ`): F9 windows 4.74, 4.39 (2080 Ti fused 4.63,
  4.55), S7D 2.33, 2.20, clamp-1 DC2 2.32, 2.10. Checkpoint configs show the intended flags. Held-out eval ran for all.
  < 2.5 GB GPU memory each.
- V2 DC2 checkpoints (ORC): trunk-2 fast entries at the clamp ≤ 0.13%, no slow entry near 1. The clamp is nearly slack late in
  training, but the F9 smoke shows that it binds early.

## Open questions for Jaden

1. **Two different 3pal data draws.** V2 (and BFR) on ORC read `~/memory_encoding_data/current` (train md5 aaf938…). The battery
   (ORC and deckard), R4, deckard and manticore read another draw (28fb6a…). V2 was compared with battery controls across the two. F9 uses V2's draw
   and S7D uses the battery's, so each matches its reference. OK?
2. F9's early phase: with the clamp off, the first windows are 4.4-4.7, against 2.0-2.4 for V2. If an early excursion appears
   but no late collapse, the draft's claim rule says "DC2 removes the need for capping for the late collapse" only.
3. Decay value 3e-6 (R4's) for S7D, versus the pilot's 1e-6. One value only, 3 seeds.
4. R4ext prints every 500 (R4 printed every 2500) so that the onset windows match V2's. Print steps run the unfused step, a
   rounding-level change. Keep 2500 for strict continuity instead?
5. The guard (`--early_stop_window 100`) is a departure from V2's 0. It saves compute on runaway collapses (plain reaches ~1e7) and
   cannot stop a run before its onset is recorded.
