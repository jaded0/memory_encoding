# P6 cheap look — analysis only, 2026-11-05

No training, library changes, or main merge. `PROTOCOL.md` fixes definitions and thresholds before measurement. `measure.py` replays checkpoints with their training code's forward, fast projection and fast-only DFA step. `probe_layers.py` localizes MQAR recall dependence by subtracting one fast read at recall prediction only. `test_measure.py` checks algebra, padding, ties, overlap, reuse, and baseline with deterministic toy data.

Host: manticore. Checkpoint/data/code archives/results: `/home/jaden/p6_look_2026-11-05/`. Production recipe: `run.sh` (GPU1, verified free of compute). All source deckard reads/copies occur through manticore; no deckard writes or GPU use.

KR: actual trained `b26e88dc25ee829b93b65f59f678a0acc87c4385`; dataset symlink to original BFR stored episodes. MQAR pilot `code_commit` and `pilot_run.json` name `983ca8be8d7713b8f4c7f8f52fb74eb90017a515` (not brief's 0f512f8); use actual trained archive and rsync dataset from deckard's `~/k13b_2026-10-28/code/synth_datasets/`. All four checkpoint source/destination SHA256 pairs match.

Eight batches per checkpoint, seeds 2718/3141/1618 for each KR arm, one pilot seed per MQAR K. Canonical observed writes, frozen slow scaffold, wiped fast state per sequence. M2 KR store-vs-other is not literal store-store: one store per episode. M1 shares for activation-selected neurons still measure energy, not activation mass. Layer width includes concatenated input and hidden state: compute baselines from actual N. i2h is reported but cannot carry recall with recurrence disabled.

Run unit tests:

```sh
cd /home/jaden/memory_encoding/scratch/p6_look_2026-11-05
CUDA_VISIBLE_DEVICES= ~/miniforge3/envs/hebby/bin/python -m unittest test_measure -v
```

Reproduce analysis (overwrites matching result names; do not run concurrently):

```sh
bash /home/jaden/memory_encoding/scratch/p6_look_2026-11-05/run.sh
```

Per-checkpoint `.json`: all distributions, mean/IQR Jaccard, counts, reuse. Per-layer `.npz`: step energies, abs activations, validity and store masks for independent auditing or pooling. `report.py` pools the preregistered three training seeds for display, preserves individual-seed decisions, writes the final vault note on manticore. It must run only after all 13 checkpoint results and four localization files exist. Optional additive_dense skipped: no finished checkpoint found in local C6 directories on manticore or omen; no ORC pull.

Revival audit: `revival_verification.json` verifies six core replay/import files per training-code archive against git and all four copied checkpoint hashes against the pre-existing deckard manifest and measured results. Analysis replays had finished before revival; none were relaunched. GPU1 was compute-free again at 17:40 for the remaining read-knockout diagnostics. The analysis log’s trailing EOF was from an incomplete old shell wrapper after all 13 primary result files were already saved; the repaired `run.sh` passes `bash -n`. No checkpoint or result was discarded.
