---
title: K13 round 2 preregistration draft — interference-limited KR
date: 2026-10-27
status: draft-gated
---

# Batch K13r2: order-two fast read where crosstalk binds

Draft written before the screening pilot produced results. **Not authorization to launch.**
Jaden's 10-26 23:33 approval motivates screening; the shared queue brief still gates real runs.

## Design and validity gate
New opt-in datasets `long_range_memory_dataset_load_D`, D = 0,4,8,16,32,48:
`0?V<random distractors>!V`, V and distractors uniform in `123,.`.
Delay-matched `gap_D` uses zeroes instead of distractors. The original KR dataset is unchanged.
Markers never appear among distractors. One recall target per episode regardless of load avoids
round 1's MQAR target-count/training-speed confound. Chance is 0.20 for this restricted answer set.

Screen a frozen round-1 lin scaffold first, in separately bounded <=580-second points, 128
episodes per point, fixed seed 20261027. Reuse `o2_keys.py` for strict accuracy, store/wrong-key
cosines, raw/projected order-one/order-two selectivity and masked linear signal share; add
uncentered unit-key PR = tr(C)^2/tr(C^2), and wrong-key absolute cosines/95th percentile.
Frozen-scaffold screening is **OOD evidence, not evidence of trained capacity**.

Before a confirmatory run is suggested, require lin accuracy <=0.85 and >=0.30 at one load,
a >=0.10 decline against load zero, L1 and L2 PR >=1.5 and mean absolute wrong-key cosine <=0.70.
At the selected load also require delay-matched gap accuracy >= load accuracy +0.10, and
a selective-write replay (mask distractor writes but retain identical forgetting and inputs)
improves recall >=0.10. Otherwise loss can be delay, OOD routing, or forgetting, not crosstalk.
Choose the smallest qualifying D. If none qualify, **no confirmatory launch; report blocked**.
Further screening/trained pilot requires a fresh explicit decision, not quietly relaxing this gate.

## Conditional arms and budget
At qualifying D: lin (no order2 flag), par2 (`--order2_layers 1,2`), par1 (same plus
`--order2_map order1`), rep2 (same plus `--order2_mode replace`). Default t=4, projection dim=16.
Seeds 2718,3141,1618,1414,1732; paired by seed. KRC1: SGD/DFA, lr 1e-4, alpha 3e4,
fraction .1, forget .01, 3x1024, B16, last_one, no recurrence/residual/PE/clamps, deterministic.
The pilot is frozen evaluation, so it cannot select a defensible training iteration count.
100k iterations is a **provisional round-1 reference only**, not a selected budget; launchers
require explicit N_ITERS and a separately approved gate manifest. No arms are run in prep.
Evaluate heldout.py strict/observed/no_fast (128 batches), o2_keys.py (64 batches) at final,
and checkpoint at 25k intervals. Same round-1 collect/summary scripts and output schema.

## Numbered predictions (conditional on the validity gate and fixed budget)
1. **K13r2-1:** par2 minus lin mean strict recall >=0.05. P=0.40. Falsifier: gap <0.05;
   also report paired differences and count of positive seeds. This is the primary outcome.
2. **K13r2-2:** par2 > par1 in >=4/5 paired seeds and mean advantage >=0.03. P=0.55.
   Falsifier: either condition fails. Tests order beyond adding a projected fast path.
3. **K13r2-3:** rep2 minus lin mean strict >=0.05. P=0.35. Falsifier: gap <0.05.
4. **K13r2-4:** order-two kernel key ratio > order-one in both deep layers in >=4/5 lin seeds.
   P=0.80. Falsifier: either layer misses 4/5; sharpening need not yield better outcomes.
5. **K13r2-5:** lin deep-layer PR >=1.5 and mean absolute wrong-key cosine <=0.70 in >=4/5
   seeds at final. P=0.60. Falsifier: either layer fails; if false, the intended regime failed,
   even if an arm wins. Also flag lin >=0.95 as renewed ceiling, <=0.25 as floor.

No choosing best seed/checkpoint, no temperature sweep, no changed predictions after screening.
Report invalid regimes, failures, all seeds and raw scores. Frozen-screen outcomes belong in
the ready note, not as confirmatory results here. Main session appends this draft only after
load and iteration budget are fixed and Jaden's current log is re-read.

Implementation correction during prep: maximum supported D is 48 rather than the initially
listed 64, because the existing metrics reject recall lag >=60. No thresholds or predictions
were changed. Screening only used D=0,4,8,16,32.

## Sparse arm — preregistered 2026-10-27 before sparse screening
Prep only; no training or submission. Frozen KR-lin-s2718, 128 episodes/load,
episode RNG 20261027, D=0,4,8,16,32; layers 1 and 2. Reuse strict replay and
store/median-absolute-other key ratio and signed-store/RMS-other selectivity.
Report medians over episodes and pooled wrong-key cosine distributions (quantiles
and raw samples) for linear, order1, order2, sparse m273/k8,17,34 from z16,
and sparse m4096/k64,256 directly from the full input. No slow updates.
Use signed top-k by magnitude, retaining amplitudes and signs, then unit-normalize
and multiply by sqrt(fraction)*|x|. This avoids a rectification-induced positive
overlap floor and guarantees k active coordinates for generic nonzero inputs.
Expansion is fixed Gaussian with its own seed/layer stream, independent of P.
Default k17 (~6.2% density) is a compromise between collisions and unstable tiny
supports; keep it for the launcher regardless of the screen (no winner selection).
Conditional fifth training arm: spar, parallel, m273/k17, same recipe/seeds,
iteration-budget gate and metrics as above; task choice remains Jaden's.

1. **SP1 (P=.65):** at D8 L1, m273/k17 key ratio exceeds order1's but is below
   order2's. Falsifier: either inequality fails. Signed selection can amplify
   differences, but finite-support noise will compete with Taylor sharpening.
2. **SP2 (P=.70):** at D8 L2, m273/k17 key ratio is less than 1.5 times order2's.
   Falsifier: >=1.5 times. Collinearity should survive this expansion.
3. **SP3 (P=.55):** at D8 L2, wide k256 has larger key ratio AND selectivity than
   m273/k17. Falsifier: either is not larger. Bypassing P may recover discarded
   distinctions, though the full-input near-collinearity may still dominate.
4. **SP4 (P=.70):** at D8, k8 has lower pooled median absolute wrong-key cosine
   than k34 at both layers. Falsifier: either layer fails. Sparse supports should
   reduce typical overlap, without guaranteeing better store matching.
No frozen-screen win establishes learned accuracy or repairs the failed L2 PR gate.
