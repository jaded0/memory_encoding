# Baseline Behavior Characterization

This baseline freezes current training behavior before mechanics cleanup. It is
an observational contract, not a claim that every captured behavior is
desirable.

## Reproducible Runs

Training remains unseeded by default. To seed Python, NumPy, Torch, Hugging Face
dataset shuffling, DataLoader sampling, and DataLoader workers, pass `--seed`:

```bash
python hebby.py --seed 1729
```

Strict mode additionally enables deterministic Torch operations, disables
cuDNN benchmarking and TF32, and configures deterministic cuBLAS behavior:

```bash
python hebby.py --seed 1729 --deterministic True
```

`--deterministic` without `--seed` is rejected. A resumed checkpoint must use
the same seed and deterministic setting as the original run; mismatches are
rejected rather than producing a hybrid run.

Checkpoints preserve Python, NumPy, Torch CPU, and available Torch CUDA RNG
states. Legacy checkpoints remain loadable because every added RNG key is
optional.

The DataLoader sampler cursor and prefetched worker batches are not currently
checkpointed. A mid-epoch seeded resume therefore recreates its sampler from
the seed instead of continuing at the exact next batch. Do not treat resumed
data order as bitwise-continuous until a resumable sampler is implemented.

## Golden Trace

`tests/fixtures/training_traces.json` records a fixed, network-free CPU call to
the real `hebby.train()` entry point for each updater. Each trace uses:

- Seed 1729 and strict deterministic mode.
- A fixed two-example, five-token in-memory sequence over `abcd`.
- One recurrent layer, hidden size 4, and `last_two` input mode.
- Recurrence enabled, matching the CLI default.
- Normalization and weight clipping disabled so raw mechanics remain visible.
- Learning rate 0.01, gradient clip 0.2, plasticity 3.0, forgetting 0.25, and plastic proportion 0.5.

The schema captures inputs, loss, per-step outputs and labels, final layer
state, candidate gradients, masks, plasticity and forgetting tensors, update
norms, and instrumented before/after boundaries for forgetting, gradient
scaling, and unified updates.

The fixture was generated on `manticore` using CPU execution with Python
3.11.12, Torch 2.5.1, NumPy 2.2.5, and one Torch thread. The test reports a
clear environment mismatch if the Torch version changes; regenerate only after
reviewing the behavior delta.

## Captured Asymmetries

| Updater | Loss | Forget cycles | Gradient-scale cycles | Unified linear updates | Unified i2h updates | `training_instance` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DFA | 1.4015415907 | 4 | 0 | 4 | 0 | 4 |
| Backprop | 1.4014136195 | 4 | 4 | 4 | 4 | 4 |
| BPTT | 1.3995014429 | 1 | 1 | 0 | 0 | 0 |

BPTT updates parameters through its manual sequence-level step rather than
`apply_unified_updates()`. DFA does not apply a unified update to `i2h`, even
with recurrence enabled. These facts are deliberately frozen for review; they
are not silently normalized by the characterization harness.

## Verification

Run the complete assertion suite from the repository root:

```bash
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s . -t . -v
```

Regenerate the fixture intentionally with:

```bash
CUDA_VISIBLE_DEVICES="" python -m tests.generate_characterization_fixtures
```

The suite checks the three real training paths, strict seeding behavior,
Python/NumPy/Torch checkpoint RNG round-trips, seed-sensitive plasticity masks,
and checkpoint plasticity updates. CUDA RNG capture is implemented but the
golden fixture and default assertion command remain CPU-only for portability.
