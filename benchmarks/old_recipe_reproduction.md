# Reproducing the 2025 no-clamp recipe on current code (started 2026-09-27)

Status: **running; one seed has converged** (interim results below, 2026-09-28 13:10 MDT). ORC array
13904287, code 953dc0e, outputs in `~/memory_encoding_benchmarks/old_recipe` on ORC. Seeds 2718 and
4241 continue to 10M as jobs 13909497 and 13909498, which wait (`afterok`) on their 5M runs.

## Why

The 2026 benchmarks (`3pal_head_panel.md`, `memory_tasks_head.md`, `stabilizer_pilot.md`) never
approached the 0.93–0.97 final-character accuracy Jaden's 2025 runs reached on 3-char reversal.
They also diverged late unless the output head was bounded. Reading the archive
(`archive/wandb_export/runs.jsonl` plus W&B histories in `jadens_team/hebby`) showed that the 2026
runs did not test his recipe or his time scale.

## What the 2025 runs did

Metric: old `avg_accuracy` is **final-character accuracy** of batch item 0, averaged per logging
interval (`hebby.py` at af3c049). It matches today's `final_char_acc`, measured there on the whole
batch. On 3-char reversal, always predicting padding scores about 0.67, because two-thirds of halves
are shorter than 3. Scoring above that requires lag-5 recall.

The converged recipe (19 no-clamp runs reached ≥ 0.9; commits 3430868, f92c191, f4932d8;
Sep–Oct 2025):

| Setting | Value |
|---|---|
| updater, recurrence | DFA, off |
| lr / plasticity α | **1e-4 / 1e5** (14 of 19); also 3e-4/3e4, 5e-4/1e4 |
| fast fraction | **0.1** |
| weight clamp | **0 (off)** |
| forget rate | 0.01 |
| normalize | false |
| model | 3 layers × 1024, batch 16, `last_one`, no positional encoding |
| length | 5M–10M iterations |

Of 23 runs with exactly lr 1e-4, α 1e5, 1024 wide and no clamp, **13 reached ≥ 0.9**, 3 collapsed
to 0, and 7 stalled at 0.60–0.87. Learning curves of ten converged runs:

| Run | acc @200k | @400k | @1M | @2M | first ≥ 0.9 | final |
|---|---:|---:|---:|---:|---:|---:|
| helpful-spaceship-9162 | .72 | .75 | .76 | .80 | 2.97M | .94 @ 5M |
| wild-star-9304 | .73 | .75 | .73 | .81 | 3.73M | .94 @ 5M |
| wild-eon-9604 | .73 | .76 | .72 | .77 | 9.44M | .93 @ 10M |
| swept-jazz-9639 | .68 | .67 | .72 | .74 | 7.89M | .90 @ 8.2M |
| magic-lake-9650 | .74 | .76 | .76 | .80 | 2.81M | .97 @ 10M |
| sweet-mountain-9681 (5e-4/1e4) | .71 | .70 | .71 | .73 | 4.24M | .92 @ 10M |
| glad-armadillo-9705 (5e-4/1e4) | .72 | .73 | .67 | .73 | 3.51M | .95 @ 10M |
| upbeat-oath-9751 | .73 | .73 | .72 | .79 | 5.60M | .97 @ 8.2M |
| drawn-cloud-9762 | .73 | .76 | .74 | .79 | 3.05M | .97 @ 10M |
| distinctive-aardvark-9763 | .66 | .66 | .70 | .72 | 9.43M | .91 @ 10M |

After converging, their loss stayed ≤ 1.5 to the end: stable with no clamp and no output tanh.

## How the 2026 runs differed

| | 2025 converged recipe | 2026 benchmarks |
|---|---|---|
| lr / α | 1e-4 / 1e5 | 1e-3 / 1e4 (plus 3e3 and 1e3) |
| fast rate lr·α | 10 | 10 (also 3, 1) |
| **slow rate lr** | 1e-4 | 1e-3: **10× faster** |
| fast fraction | 0.1 | 0.2 |
| weight clamp | 0 | 1 |
| length | 5–10M | 0.25–0.5M |
| slow-weight init | **zero** (every weight) | `nn.Linear` default (since 8e9b45f) |
| output head | reads the trunk, no tanh | same, until `--output_tanh` |
| `i2o` fast entries | 10% (masked, decayed, wiped) | none (slow-only since 5fae219) |
| forget order | before the update | after the update (092d434) |
| `i2h` | not trained by DFA; irrelevant with recurrence off | direct DFA; irrelevant with recurrence off |

Two findings reconcile the results:
1. **Time scale.** At 400k the 2025 runs sat at 0.66–0.76. The 2026 runs at 400k sit at 0.71–0.75
   (`stabilizer_pilot.md`). Neither had converged; 2025 convergence came at 2.8M–9.4M.
2. **Slow learning rate.** In the 2025 sweep, lr 1e-3 almost never worked on this task at any α (2
   of 152 long runs, `docs/archive_2025_sweep_observations.md`). The late divergence diagnosed in
   2026 is driven by slow-weight growth. A 10× slower slow rate plus zero initialization plausibly
   explains why the 2025 recipe was stable without any clamp.

## Design

`sweeps/orc_old_recipe.sbatch`: the 2025 recipe on current code, with lr 1e-4, α 1e5, fraction
0.1, clamp 0, no output tanh, `--fused_update`, seeds 2718, 3141 and 4241, and 5M iterations
(`N_ITERS` overrides). Initialization, the `i2o` mask and the forget order stay at current code,
so this is a test of the current code on the old recipe. Each run takes about 20 h at roughly
70 it/s and continues across cs's 1-day wall limit through the USR1 checkpoint and requeue. A
loss early stop (loss > 5 for 10 intervals) is recorded as a collapse.

To extend a run still climbing at 5M, resubmit the same task with a larger `N_ITERS`:
```bash
sbatch --array=<task> --export=ALL,CODE_COMMIT=953dc0e,N_ITERS=10000000 orc_old_recipe.sbatch
```
It resumes from the checkpoint; a changed `n_iters` is allowed on resume.

## How to read it

- **Converges (final-character accuracy ≥ 0.9 by about 5–10M) on most seeds:** the current code is
  fine. Adopt this recipe for the benchmarks and retire lr 1e-3.
- **Stalls or collapses:** switch on one 2025 difference at a time, starting with zero
  initialization of the slow weights (most recent change, and directly affects the slow-weight gain
  that drives divergence), then fast entries in `i2o`, then forget before the update.

Compare on `final_char_acc` (the 2025 metric) as well as `recall_acc` and lag-5 recall.

## Results

### Interim, 2026-09-28 13:10 MDT

| Seed | Iteration | Final char | Recall | Lag 1 / 3 / 5 | Exact recall | Loss |
|---|---:|---:|---:|---|---:|---:|
| 3141 | 4.00M | **0.951** | **0.936** | .986 / .902 / .853 | .887 | 0.640 |
| 2718 | 3.19M | 0.755 | 0.740 | .937 / .682 / .264 | .582 | 0.949 |
| 4241 | 3.71M | 0.743 | 0.699 | .976 / .517 / .227 | .513 | 0.960 |

- **The current code reproduces the 2025 result.** Seed 3141 passed 0.9 final-character accuracy at
  **2.69M** iterations, inside the 2025 range of 2.8M–9.4M. At 4.0M its recall-position accuracy is
  93.6%, with lag 5 at 85.3%, and it is still improving. Every 2026 run at lr 1e-3 stayed at or below
  about 30% lag-5 recall.
- **Stable without a clamp or tanh.** No seed collapsed; the maximum interval loss is about 2.0, in one
  early interval per seed. At lr 1e-3 the 2026 recipe collapsed at 200k–350k unless the output was
  tanh-bounded. This supports the slow-learning-rate explanation (`stabilizer_pilot.md`).
- **The late jump is lags learned in order.**
  - Lag 1 locks in first, jumping from about 0.65 to 0.94 around 1.25M–1.75M in all three seeds.
  - In seed 3141, lag 3 follows (0.84 by 1.5M), then lag 5 climbs from 0.22 at 1.5M to 0.85 at 3.75M.
  - Final-character accuracy needs lag 5 on a third of sequences, so it rises last and looks like a
    sudden late jump.
  - Seeds 2718 and 4241 are stuck between the lag-1 and lag-5 stages (lag 5 about 0.23–0.26).
- **Recall dips at 0.75M–1.0M, just before lag 1 locks in**: clearly in 2718 and 4241 (0.61 → 0.45–0.48,
  loss up to 1.07–1.09), mildly in 3141.
- **The old final-character metric tracked real recall here.** 0.95 final char comes with 93.6% recall
  and 85% lag-5 recall.
