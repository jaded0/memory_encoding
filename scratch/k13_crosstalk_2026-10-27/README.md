---
title: K13r2 sparse arm — prep and frozen screening
date: 2026-10-27
status: prep-blocked-regime
reading: unread
reading_priority: 2
reading_focus: "Sparse matches L1 key ratio but not selectivity; wide kernels separate L2 better."
project: ephemeral-weights
---

# K13r2 sparse arm

Branch `k13-crosstalk`, manticore `~/me_k13x`; implementation `a944c07`,
code/evidence `e399097e9e4ef999d20f0e1bff1a80fff067e6da`. The following documentation
commit adds this hand-back; `origin/k13-crosstalk` identifies the final pushed tip.
No merge, training, production run or Slurm submission. Proposals page read only.
The earlier four-arm README is superseded by this five-arm hand-back; the original
[[K13r2 ready]] retains the failed regime-gate evidence and conditional launch details.

## Added
- `--order2_map sparse`: same seeded P/z16 as order1/order2, independent seeded Gaussian
  expansion R, **signed top-k by magnitude**, retaining signed amplitudes, then normalize
  to unit length and scale by `sqrt(fraction)*|x|`. Self-match is `fraction*|x|²`.
  Zero/projected-null keys stay zero; generic nonzero keys have exactly k active entries.
  Signed selection avoids adding a ReLU positive-overlap floor; temperature cancels.
- `--order2_sparse_m 273`, `--order2_sparse_k 17` defaults: same feature count as order2
  at d16, 6.2% density as a compromise between collisions and tiny unstable supports.
  **Defaults fixed before screening, not chosen from the results.** R is a checkpoint
  buffer, not a learned weight. utils.py checks both settings on resume.
- `spar` parallel launcher arm; ORC array **0-24%4**, five arms × five original seeds.
  Manifest, clean exact-commit, A100 guard, concurrency limit and recording gates remain.
  No approval manifest exists. No task/load/iteration budget has been selected.
- Frozen-screen observer on the original `o2_keys.py` replay; no map changes replay state.
  Wide full-input m4096/k64,256 is kernel-only, not a training flag. Tests cover support,
  shape, scale, seed/RNG isolation, checkpoint settings/defaults and sparse write-path parity.

## Screening (no training)
Frozen `~/k13r2_prep/KR-lin-s2718.pth`; strict replay, 128 identical episodes per load,
episode seed 20261027. All original accuracy and layer summaries exactly equal the prior
screen at every load. The baseline re-emitted kernel metrics are asserted against o2_keys.
This is one frozen seed, OOD task evidence, **not trained capacity or accuracy evidence**.

Cells: **median key ratio / median selectivity**. Ratio = absolute store kernel divided
by median absolute wrong kernel, per episode; selectivity = signed store kernel divided
by RMS wrong kernel. Linear is the expected unmasked scalar kernel, not the masked vector read.
`z-k` uses m273 from z16; `full-k` uses m4096 from full input.

| D | layer | linear | order1 | order2 | z-k8 | z-k17 | z-k34 | full-k64 | full-k256 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | L1 | 2.201 / 2.177 | 2.706 / 2.700 | 5.136 / 5.065 | 2.314 / 2.221 | 2.690 / 2.262 | 2.593 / 2.212 | 4.730 / 4.680 | 3.714 / 3.579 |
| 0 | L2 | 2.965 / 2.518 | 2.260 / 1.885 | 3.232 / 2.512 | 4.663 / 3.297 | 3.388 / 2.572 | 2.952 / 2.296 | 9.782 / 8.083 | 6.221 / 4.941 |
| 4 | L1 | 1.902 / 1.917 | 2.650 / 2.347 | 5.021 / 4.282 | 3.206 / 2.848 | 4.436 / 3.074 | 4.172 / 2.830 | 4.235 / 4.198 | 2.928 / 3.027 |
| 4 | L2 | 1.847 / 1.915 | 1.442 / 1.551 | 1.784 / 1.875 | 1.970 / 2.075 | 1.766 / 1.867 | 1.707 / 1.813 | 3.923 / 3.794 | 3.130 / 3.107 |
| 8 | L1 | 1.886 / 1.865 | 2.590 / 2.241 | 5.021 / 4.050 | 3.227 / 2.909 | 5.042 / 3.237 | 4.075 / 2.838 | 4.321 / 4.232 | 2.919 / 2.961 |
| 8 | L2 | 1.761 / 1.812 | 1.433 / 1.501 | 1.712 / 1.794 | 1.924 / 1.987 | 1.757 / 1.794 | 1.722 / 1.776 | 3.438 / 3.381 | 2.728 / 2.783 |
| 16 | L1 | 1.849 / 1.822 | 2.404 / 2.111 | 4.531 / 3.684 | 3.152 / 2.819 | 4.723 / 3.028 | 4.046 / 2.762 | 4.376 / 4.276 | 2.916 / 2.942 |
| 16 | L2 | 1.694 / 1.726 | 1.416 / 1.446 | 1.643 / 1.686 | 1.776 / 1.791 | 1.731 / 1.749 | 1.666 / 1.696 | 3.160 / 3.105 | 2.551 / 2.570 |
| 32 | L1 | 1.811 / 1.797 | 2.404 / 2.031 | 4.777 / 3.472 | 3.063 / 2.595 | 4.394 / 2.602 | 3.792 / 2.548 | 4.550 / 4.304 | 2.946 / 2.970 |
| 32 | L2 | 1.621 / 1.644 | 1.386 / 1.408 | 1.605 / 1.649 | 1.709 / 1.713 | 1.704 / 1.709 | 1.652 / 1.662 | 2.881 / 2.863 | 2.397 / 2.409 |

Raw JSON: `scratch/k13_crosstalk_2026-10-27/evidence/sparse/load{0,4,8,16,32}.json`.
Includes every episode's store/wrong kernel, store/wrong cosine, ratio, selectivity,
pooled signed min/p05/p25/median/p75/p95/max and absolute medians, for **every load/map/layer**.
`evidence/sparse/REPORT.md` tabulates all 80 wrong-key cosine distributions;
`sparse_report.py` regenerates it. D8 examples below (signed pooled distributions,
1,280 wrong keys per layer/map; not distributions of episode means):

| layer | map | min | p05 | p25 | median | p75 | p95 | max |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| L1 | linear | -.438 | -.438 | .433 | .459 | .466 | .470 | .495 |
| L1 | order1 | -.443 | -.443 | .219 | .313 | .383 | .542 | .683 |
| L1 | order2 | .039 | .072 | .105 | .140 | .227 | .321 | .487 |
| L1 | z-k8 | -.297 | -.297 | .000 | .105 | .127 | .215 | .281 |
| L1 | z-k17 | -.351 | -.351 | -.015 | .046 | .118 | .196 | .309 |
| L1 | z-k34 | -.426 | -.426 | -.005 | .087 | .177 | .286 | .401 |
| L1 | full-k64 | -.118 | -.118 | .079 | .116 | .130 | .156 | .189 |
| L1 | full-k256 | -.157 | -.157 | .178 | .203 | .215 | .238 | .242 |
| L2 | linear | -.283 | -.269 | .548 | .658 | .710 | .781 | .834 |
| L2 | order1 | .023 | .126 | .706 | .791 | .835 | .887 | .946 |
| L2 | order2 | .039 | .054 | .518 | .640 | .709 | .794 | .899 |
| L2 | z-k8 | .000 | .000 | .344 | .528 | .655 | .772 | .997 |
| L2 | z-k17 | -.048 | .000 | .409 | .515 | .600 | .712 | .819 |
| L2 | z-k34 | -.077 | .008 | .473 | .567 | .635 | .726 | .827 |
| L2 | full-k64 | -.060 | -.056 | .191 | .274 | .337 | .422 | .535 |
| L2 | full-k256 | -.085 | -.073 | .274 | .368 | .421 | .502 | .592 |

## Pre-registration and interpretation
Sparse section appended to vault `prereg_drafts/K13r2.md` **before** screening;
snapshot in `evidence/sparse/PREREG.md`. Numbered outcomes at D8:
1. SP1 (P=.65): **falsified**. z-k17 exceeds order1 but narrowly exceeds rather than
   falls below order2's L1 ratio (5.042 vs 5.021). This tiny ratio difference is not a win in selectivity.
2. SP2 (P=.70): **held**. L2 z-k17/order2 ratio = 1.026, below 1.5.
3. SP3 (P=.55): **held**. Wide k256 L2 ratio/selectivity 2.728/2.783 > 1.757/1.794.
4. SP4 (P=.70): **falsified**. k8 absolute wrong-cosine median exceeds k34 at L1
   (.126 vs .111), though it is lower at L2 (.528 vs .567).

In plain language: the size-matched sparse read removed much of the typical overlap
between wrong keys, but it did not uniformly clean up the read better than order two.
At D8 in layer 1 it matched order two's key ratio while its selectivity was lower
(3.237 vs 4.050), so tails and store matching matter, not only typical wrong-key overlap.
In layer 2 its key ratio and selectivity were almost the same as order two's. The wide
full-input sparse maps separated layer-2 keys more clearly, but width, density and input
source all change together, so this does not isolate projection loss as the cause.
These kernel swaps do not show that a trained sparse arm improves task accuracy.

## Validation, resources, reproduction
CPU suite: **441 passed, 443 subtests passed**, 3 existing FutureWarnings, 45.47s; golden/identity
tests pass. Log `~/k13r2_prep/sparse_tests_complete.log`. Shell syntax and Python compilation pass.
GPU0 was idle at start (129 MiB); unfused 1080 Ti path, .30 allocator cap (~3.3 GiB).
Five separately timeout-580s points: total timed replay 32.2s, max point 8.3s,
max allocated 892.0 MiB, below 4 GB. No other processes touched.
Checkpoint SHA256 `e4a30b0a67853abaa7baa4c1fcfe32aea0b9da950e010d7c30bcb8ff0adccbf0`.

From `~/me_k13x`, screening only:
```bash
for d in 0 4 8 16 32; do
  CUDA_VISIBLE_DEVICES=0 timeout 580 ~/miniforge3/envs/hebby/bin/python \
    scratch/k13_crosstalk_2026-10-27/sparse_screen.py ~/k13r2_prep/KR-lin-s2718.pth \
    --load "$d" --out "scratch/k13_crosstalk_2026-10-27/evidence/sparse/load$d.json"
done
CUDA_VISIBLE_DEVICES= ~/miniforge3/envs/hebby/bin/python -m pytest tests -q
python3 scratch/k13_crosstalk_2026-10-27/sparse_report.py
```
No production GPU-hour request or launch recommendation. The previous L2 spread gate
still fails: changing a hypothetical read kernel is not evidence that the underlying
linear-scaffold keys satisfy that gate. **Higher-dimensional-key task versus accepting
D8 and restricting claims remains Jaden's open decision.**
