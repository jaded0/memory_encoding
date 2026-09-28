# Observations from the 2025 sweeps (read 2026-09-27)

Source: `archive/wandb_export/runs.jsonl` (4,701 runs, 4,477 with a final metric, 2025-05-01 to
2025-11-03) and W&B histories in `jadens_team/hebby`. The metric throughout is the old
`avg_accuracy`: **final-character accuracy of batch item 0**, averaged over the logging interval
(one noisy sample per iteration, `hebby.py` at af3c049). "Success" below means a final value ≥ 0.9.
Always predicting padding gives about 0.67 on 3-char reversal and about 0.75 on 4-char. Runs were
unseeded, so identical configs are independent draws.

## 1. The 3-char reversal landscape: two conditions, not one

The comprehensive sweep (`comprehensive_sweep`, 2025-08-21 to 11-03, 2,071 runs) is the best
evidence. This table covers DFA with recurrence off, 3 × 1024, fraction 0.1, and runs of at least
1M iterations. Entries are successes out of runs, per (lr, α):

| lr \ α | 1e3 | 7e3 | 1e4 | 3e4 | 5e4 | 7e4 | 1e5 | 3e5 | 1e6 |
|---|---|---|---|---|---|---|---|---|---|
| 1e-5 | | | 0/9 | | | | 0/21 | | 0/9 |
| 3e-5 | | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 1/5 | |
| 5e-5 | | 0/7 | 0/6 | 0/7 | 0/7 | 0/7 | 1/7 | 2/7 | |
| 7e-5 | | 0/7 | 0/7 | 0/7 | 1/7 | 1/7 | **5/7** | | |
| **1e-4** | 0/12 | 0/7 | 0/28 | 3/7 | **10/17** | **7/7** | **56/98** | | 0/3 |
| 2e-4 | | 0/4 | 0/4 | 2/4 | 2/4 | 0/4 | | | |
| 3e-4 | | 0/7 | 4/7 | **38/107** | 0/3 | | | | |
| 5e-4 | | 3/7 | 7/16 | | | | | | |
| 7e-4 | | 0/7 | 0/6 | | | | | | |
| **1e-3** | 0/19 | | **2/19** | | | | 0/14 | | |
| 1e-2 | 0/4 | | 0/3 | | | | | | |

Success needs **both** lr·α in about [3, 10] **and** a slow rate lr of about 7e-5 to 5e-4:
- By lr·α over all long runs: 0 of every setting below 3; 7/19 at 3; 31/93 at 5; 12/20 at 7;
  39/112 at 9; 62/187 at 10; 0/4 at 14; 2/10 at 15; 0/41 at 100.
- By lr: 1e-4 91/313, 3e-4 42/124, 5e-4 10/23, but **1e-3 2/152** and 1e-5 0/42.

**The lr 1e-3 used for every 2026 benchmark is outside this region.** At lr·α = 10 it succeeded in
2 of 19 runs, where lr 1e-4 succeeded in 56 of 98.

Best cells: lr 1e-4 with α 7e4 (7/7, lr·α = 7), lr 7e-5 with α 1e5 (5/7), lr 1e-4 with α 1e5 (56/98).

## 2. Other factors on 3-char reversal (DFA, recurrence off, ≥ 1M iterations)

| Factor | Successes / runs |
|---|---|
| width 1024 / 512 | **158/693** / 2/158 |
| fast fraction 0.01 / 0.05 / **0.1** / 0.15 / 0.2 / 0.3 / 0.5 | 0/10 / 0/10 / **147/742** / 3/10 / 9/66 / 1/10 / 0/3 |
| weight clamp 0 / 0.01 / 0.1 / 0.5 / 1 / 2 / 5 / 7 / 10 / 100 | 19/131 / 0/15 / 0/68 / 9/15 / 19/133 / 6/15 / 10/15 / 9/15 / 77/429 / 11/15 |

- **Clamping at 0.1 or below removes the memory** (0 of 83). From 0.5 up, the clamp makes no
  consistent difference: the high rates at 0.5, 2, 5, 7 and 100 come from one 15-run block per
  value at a good lr/α. At a matched config, clamp 0 is as good as any.
- **Width matters:** 512 almost never solved 3-char reversal.
- **Fraction 0.1 is the sweet spot**, and 0.2, the 2026 default, did worse (9/66).

## 3. Convergence is late and sudden

Ten converged no-clamp runs (lr 1e-4/α 1e5 and 5e-4/α 1e4) sat at 0.66–0.76 at 400k and 0.67–0.76
at 1M. They first crossed 0.9 between **2.8M and 9.4M** iterations, typically after a slow rise
from about 2M (table in `benchmarks/old_recipe_reproduction.md`). Nothing shorter than about 3M
can show whether a configuration works. Even the best cell (56/98) leaves 40% of runs
unconverged or collapsed.

## 4. Collapse was common and a clamp did not prevent it

Of 851 long 3-char DFA runs, 139 collapsed (final loss > 100, NaN, or accuracy 0). The collapses
were spread across lr·α (0.1: 25, 1: 34, 10: 49, 100: 31) and across clamps (0: 47, 1: 40,
10: 44, 0.1: 8). A clamp of 1 or 10 gave no protection. In 2026, the same kind of late divergence
at lr 1e-3 was stopped only by `--output_tanh` (`benchmarks/stabilizer_pilot.md`).

## 5. 4-char reversal was never solved without recurrence

1 success in 209 runs. The best recurrence-off DFA runs reached 0.81–0.83, against a padding
baseline of about 0.75: `easy-water-8756`, `quiet-lion-8783` (lr 1e-3, α 1e3 or 1e2, clamp 10) and
`fluent-leaf-8277`. The one success, `rare-mountain-8798`, is BPTT **with** recurrence. 2026's
lr·α = 3 run reached 53% recall at 500k and was still rising (`memory_tasks_head.md`), so 4-char
reversal may need the long-horizon recipe plus more capacity than 1024 × 10%.

## 6. Updaters and recurrence on the easier tasks (runs ≥ 500k)

| Task | Updater, recurrence | Successes / runs |
|---|---|---|
| 2-char reversal | DFA, off | 155/582 |
| | `static_plastic_candidate` | 50/270 |
| | backprop, off | 1/62 |
| binary 2-char reversal | `static_plastic_candidate` | **20/21** |
| | BPTT, on | 17/21 |
| | DFA, off | 93/388 |
| | DFA, on | 66/246 |
| | `nocycle` | 27/90 |
| | backprop, off / on | 22/369 / 18/270 |
| | BPTT, off | 0/28 |

- Per-step backprop rarely works for the ephemeral model (the α² issue in the README applies).
- Recurrence did not help DFA on the binary task (66/246 on against 93/388 off).
- The early `static_plastic_candidate` rule (May–Aug 2025) was the most reliable on the binary
  task. It is gone from the current code; `archive/checkpoints/*/run_used.sh` shows how it was
  launched.

## 7. Timeline

- **May 2025, all failures:** `mega_clip_sweep` 0/204, `check_phenomenon` 0/33, the long requeue
  and self_grad groups.
- **Late May–July, first reliable successes on 2-char tasks:** `normlog_4_4` 15/25,
  `quite_forgetful_speedier` 12/28, `nocycles` 19/39. `normlog_5_5` 0/23 is a near-twin that failed.
- **August, larger sweeps:** `arch_sweep` 52/200, `bench_sweep` 133/940, `grad_clip_sweep` 46/284,
  all on the binary 2-char task. Then `comprehensive_sweep` (2-, 3- and 4-char), which supplies
  almost all the 3-char evidence above.
- About 9/31 TinyStories runs log ≥ 0.9 on this metric, but for text it is just next-character
  accuracy at the end of a line, not a memory test.

## Caveats

- The metric is one sequence's last character per iteration, so it is noisy and blind to lags
  other than the longest half.
- The code changed across 2025 (update rules, forget order, normalize semantics, masks). Configs
  from different months are not strictly comparable, which is why sections 1–2 stay inside one
  sweep.
- Unseeded runs: the success fractions are the seed-to-seed reliability of a configuration.
