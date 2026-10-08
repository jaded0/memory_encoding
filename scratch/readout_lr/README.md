# Readout-only vs trunk-only slow lr from B 150k (2026-10-07)

Write-up: vault note "ephemeral weights readout-only slow lr test 2026-10-07".

- `run.sh NAME GPU` (NAME in CTRL, RO3, TR3): the deckard launcher, run from
  `~/readout_lr_2026-10-07` with an immutable code snapshot of f03ea4b in `code/`.
- `collect.py`: interval metrics of CTRL/RO3/TR3 and the global x0.3 lineage
  (INT_R2 -> INT2_R2 -> INT3_R2) as CSV; run on deckard with `python3 -`.
- `fig.py SCRATCH_DIR OUT.png`: the three-panel figure from `metrics.csv` and the
  `scratch/drift_predictions/summarize.py` output `summary.json` in SCRATCH_DIR.

Result: onset CTRL 180k, readout-only 260k, trunk-only 215k, global none through 450k.
