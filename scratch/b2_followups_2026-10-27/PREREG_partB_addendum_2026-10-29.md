# B2fu Part B: launch addendum (2026-10-29, written before any Part B result)

Approved by Jaden 2026-10-29 ("Yes to B2 pB", plus the RNN lr sweep). Part B's design and
predictions 6–9 are the draft in the vault (`queue_2026-10-27/prereg_drafts/B2fu.md`,
"Part B"), copied into `PREREG_fastlayers.md` under "Batch B2fu-B". This addendum records the
changes from that draft. Code: main af1364f (RNN_LR override, `lr_select.sh`) plus this file.

## 1. Recipe change: RNN lr sweep (the draft said "no hyperparameter search")
- RNN-BPTT arm, seed 7201, 20k, lr ∈ {3e-5, 1e-4, 3e-4, 1e-3}; everything else as `arms.sh`.
  Set through `RNN_LR` (arms.sh; refused for any other arm). Runs live in
  `$ROOT/lrsweep/lr$LR/` and also run the normal evaluate.sh outputs (not used to select).
- **Selection:** `lr_select.sh`: CE (`ce.all.on`, nats/char, crop 512) of the 20k checkpoint on
  **validation stories 4096–8191 of `val8192.json`**. No B2/B2fu test evaluation reads those
  stories (final = 0–4095, horizon = 0–1023, curves = 0–255). Lowest CE wins; ties are not
  expected and would go to the smaller lr. Test numbers are never consulted for the pick.
- The 5-seed RNN-BPTT arm uses the picked lr. If the pick is not 1e-4 this changes the draft's
  fixed baseline; it is disclosed as tuning on one seed.
- **Reuse:** the sweep run at the picked lr is Part B's RNN-BPTT s7201 (same code, seed, recipe,
  deterministic), copied rather than rerun. Its test score was not used in selection.
- A diverged/nonfinite sweep run counts as "not picked" only if it finished with exit 0 and a
  20k checkpoint; a crash halts the chain and goes to Jaden (no automatic substitute).
- Predictions (sweep): S1 the pick is 3e-4 or 1e-3 (P .55). S2 1e-3 trains without a nonfinite
  loss (P .8). S3 validation CE spread across the four lrs ≥ .05 nats/char (P .6).

## 2. Placement (hardware is a recorded nuisance, never confounded with arm)
- Bounded timing 10-29 (25 min each, `--print_freq 25`): REC-DEEP DFA on an RTX 2080 Ti (roc,
  compiled step) 0.60–0.64 it/s; on GTX 1080 / 1080 Ti (sm61, no Triton, eager step) fewer than
  25 iterations in ~15 min (< .03 it/s): unusable. RNN-BPTT: 1.65 it/s on valkyrie's GTX 1080
  (CPU-bound, ~50% GPU), 0.77–1.0 it/s on wyrmling/roc/formic.
- Two DFA runs sharing one 2080 Ti ran 0.27 + 0.28 it/s, below one run's 0.64, so one run per GPU.
  REC-NOFAST ran 0.64 it/s alone. RNN-BPTT four at once on valkyrie: 1.07 it/s each.
- **All 10 DFA runs (REC-NOFAST, REC-DEEP) on roc's two RTX 2080 Ti**, one per GPU, seeds in
  order 7201→7605: GPU0 runs REC-NOFAST, GPU1 REC-DEEP, so both arms of a seed run side by side.
  ~9 h train + eval per run, ~2.2 days in all. Moving a seed elsewhere (ORC, deckard after K13)
  moves both of its arms together.
- **All RNN-BPTT runs on valkyrie's two GTX 1080**: 4 sweep runs at once (two per GPU), then
  selection, then seeds 7302–7605 at the pick (two per GPU).
- Data: ORC's TinyStories first-1M processed cache copied byte-identical (all 7 file sha256s
  match ORC) to `~/b2_followups_2026-10-27/processed_datasets` on each box; `inputs/` copied
  from deckard and pass `SHA256SUMS`. Detached checkout `~/b2fu_B/code`.
- Wall-clock and peak memory differ by host; report them per run, not as arm effects.

## 3. Unchanged
Predictions 6–9, scoring (`score.py --part B`), fairness disclosure (logical-forward-slot
matching; the RNN's extra slow slots, Adam, residual, no LN), failure handling (report,
resume same settings only, no silent substitution).
