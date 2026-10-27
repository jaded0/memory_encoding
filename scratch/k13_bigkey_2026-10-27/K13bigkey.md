---
title: K13bigkey preregistration — overlapping two-character addresses
date: 2026-10-27
status: pilot-approved-confirmation-gated
---

# Batch K13bigkey

Written before the pilot and before new-task smoke results. Exact approval_reference:
`Jaden chat 2026-10-27: try the higher-dimensional-key task; evaluate order two vs expand-and-sparsify`.
The proposals page is read-only. This draft is not a result.

## Task, arms, seeds, iterations
`mqar2_K_q4`, K=4,8,16,32: K distinct pairs from a-j × a-j (100 addresses), independent
digits 0–9, table `ab3cd7...`, then four queries `?cd7?ab3...`. No separator between
key and value: last_two must expose the intact pair on the value-prediction step.
Own shared charset, core-hashed train/validation/test splits. Chance .10 per answer.
Fixed Q=4 avoids the prior target-density confound. Total length 3K+16; exact lag
metadata retained, training metric lag capped at57 as for mqarw. Padding is zero input.
`mqar2gap_K_q1` retains one queried table slot and replaces the others with `...`;
it matches first-answer delay, not the full query distribution. Also make paired
gap replays from pilot episodes; gap inputs can be OOD, so causal write suppression
(same input, same decay) is the primary interference test.

Recipe: KRC1, 3×1024, B16, DFA/SGD lr1e-4, alpha3e4, fraction .1, forget .01,
last_two, no recurrence/residual/PE/clamps. lin pilot seed2718 at each K, up to200k
iterations but **75 minutes training wall-clock per task**, save/evaluate each5000.
Use latest complete checkpoint on timeout, report actual iterations and curves.
Each A100 allocation hard-limited90min, all four <=6 A100-hours including analysis;
max4 concurrent. Hardware A100, qos cs, exclude cs-1-5,cs-1-3, HOME inductor cache.

Conditional arms: lin; par2 (`--order2_layers 0,1,2`); par1 (same, order1);
rep2 (same, replace); spar (same, sparse m273/k17). Default t4, dim16, fixed P/R.
Include layer0 because that is where partially overlapping keys are guaranteed.
All transformed arms use the same placement; deep geometry is reported, not assumed
healthy. This tests whole-network changes, not an isolated L0 mechanism. Order1 has
17 rather than273 features; sparse and order2 are size-matched. Parallel adds memory
and self-gain; rep2 helps distinguish replacement from merely adding a read.
Five paired seeds2718,3141,1618,1414,1732. No confirmatory run before gate passes.

## Regime gate (frozen before the pilot)
At least one K must have validation strict recall in [.20,.85]; first-answer strict
recall also in [.20,.85]. On128 heldout episodes, suppressing all writes strictly
after the first query's source-store step (including value and following steps),
without changing inputs or forgetting, must improve first-answer recall >=.10 AND
recover >=80% of its error toward1. Report store-only replay separately, since
post-store suppression leaves pre-store interference intact. Require unit-key
uncentered PR >=4 at L0 AND >=2 at L1, measured over actual distinct table-store
keys pooled across heldout episodes, plus signed/absolute wrong-key cosine quantiles.
Layer2 is diagnostic, not a gate: this task deliberately anchors addressing at L0.
Choose smallest qualifying K, never relax thresholds. A delay-matched gap is a useful
diagnostic but cannot replace the same-input suppression criterion.

Iteration budget: use earliest recorded5000-step point within .02 of final validation
strict recall, provided the final two curve points differ <=.03; otherwise no stable
budget, no confirmation. Pilot timeout is not convergence. Confirmation estimate must
include scoring and the most expensive arm, cap40 A100-hours. If core25 runs alone
exceed40h, do not launch. Optional larger K with3 seeds only if total including core
stays<=40h. Gate manifest records actual evidence, selected K, iterations and estimate;
`regime_verified:true`, `launch_approved:true`, approval_reference above, pilot_evidence.

## Metrics
Strict/observed/no_fast recall and first-answer recall; exact-sequence recall; curves,
runtime/memory/actual iterations. Test split reserved for final confirmatory scoring;
pilot selection uses validation only. Per-layer normalized key PR, correct/wrong signed
cosine distributions, kernel ratio and selectivity. Kernel-only swaps on lin checkpoints:
linear/order1/order2/z16-sparse273-k17/full-input sparse4096-k64,k256, reuse sparse_screen.
Report all K, failed/unfinished runs and suppressed-write/gap evidence, not best seed.
Inspect writes under separator '?', value, next first-letter, and their overlap with
the true pair; they are not silently treated as clean distractor-free stores.

## Numbered predictions and falsifiers
1. **BK1**, P=.65: at least one pilot K passes the gate. Falsified if none does or
   no stable iteration budget is obtained within the budget. No confirmation then.
2. **BK2**, P=.70: lin strict recall at32 is at least .10 below8 at the selected
   common learning-curve checkpoint. Falsified otherwise; training-speed confounds
   remain possible, so this is not by itself a capacity measurement.
3. **BK3**, P=.45, primary conditional outcome: spar minus par2 mean test strict
   recall >=0 at selected K. Falsified if negative. Report paired differences and
   number positive; equality is not evidence of equivalence.
4. **BK4**, P=.55: par2 beats par1 in >=4/5 seeds and mean gap >=.03.
   Falsified if either fails. Isolates second-order term on the same projection.
5. **BK5**, P=.45: rep2 minus lin mean strict >=.05. Falsified if smaller.
6. **BK6**, P=.65: on pilot lin L0, z-k17 median absolute wrong-key cosine is
   lower than order2's, but order2's median selectivity is greater. Falsified if
   either inequality fails at selected K (or K16 if gate fails).
7. **BK7**, P=.55: if larger-K confirmation runs, max(spar,par2) minus lin
   mean gain is >=.03 larger there. Falsified otherwise; unscored if not launched.

Do not change task, thresholds, placement, temperature, sparsity or predictions from
pilot results. Any repair is a new preregistration and needs a new launch decision.
