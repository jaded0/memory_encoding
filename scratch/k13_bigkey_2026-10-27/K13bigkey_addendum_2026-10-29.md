---
title: K13bigkey addendum — three-arm deckard confirmation
date: 2026-10-29
amends: K13bigkey.md (sha256 c547bb9c…, unchanged)
status: approved-before-launch
---

# K13bigkey addendum, 2026-10-29

Written before any confirmation run and before any trained transformed-arm result.
Exact approval_reference (gate profile `deckard3`):
`Jaden chat 2026-10-29: option 1, lin/par2/spar x 5 seeds x 195k on deckard after B2fu-A`.

## What was known when this was decided
The four lin pilots (deckard, 200k each) and the gate review of 2026-10-28: K16 passes
every frozen scientific criterion; frozen curve rule gives 195000 iterations. The frozen
A100 runtime gate blocks because the 25-run core exceeds 40 A100-hours. Also seen: the
kernel-only screen on the K16 lin checkpoint (BK6 falsified: z-k17 selectivity 6.497 vs
order2 4.732). That screen swaps kernels on a frozen lin model; it is not a trained-arm
accuracy and was the only arm-related evidence seen.

## Changes (cost-driven, not outcome-driven)
1. **Arms:** lin, par2, spar only. par1 and rep2 are dropped to cut cost; they are the
   arms the primary question (order two vs expand-and-sparsify, Jaden 10-27) needs least.
2. **Hardware:** deckard 2× RTX A6000, four lanes (two per GPU), queued after the
   B2fu-A deckard runs leave each GPU. Not ORC; the 40 A100-hour cap is replaced by a
   504 lane-hour cap on the arm-specific conservative estimate (≈472 lane-h measured
   basis: `~/k13b_2026-10-28/runtime_estimate.json`). Per-run timeout from the gate.
3. **Gate:** `gate.py build --profile deckard3`. The pilot wall bound is 86400 s (the
   deckard pilots ran the full 200k under explicit authorization), not 4510 s. All other
   regime, stability and evidence-integrity checks are unchanged; K and iterations are
   re-derived from the same pilot evidence (expected mqar2_16_q4, 195000).

Unchanged: task, recipe, placement, temperature, sparsity, seeds (2718, 3141, 1618,
1414, 1732), iteration rule, test split reserved for final scoring, metrics.

## Predictions
- **BK3 (primary)** unchanged: spar minus par2 mean test strict recall ≥ 0 at K16.
- **BK4, BK5** unscored (par1, rep2 not run). **BK7** unscored (no larger K).
- **BK1** passed (K16), **BK2** and **BK6** are scored from pilot evidence as before.
- Secondary, reported not gated: par2 minus lin and spar minus lin mean test strict
  recall, paired by seed, with the number of positive seeds.

Report every run including failures and timeouts; no best-seed selection.
