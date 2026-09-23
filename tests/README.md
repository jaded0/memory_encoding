# Tests and the golden-trace baseline

Run everything from the repository root (CPU only, network-free, about 5 s):

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest tests/ -q
```

`pytest` is needed because `tests/legacy/` has no `__init__.py`, so
`python -m unittest discover -s . -t .` does not collect it.

| Module | Covers |
| --- | --- |
| `test_characterization.py` | Golden traces for DFA, backprop and BPTT; a different seed changes the plasticity mask |
| `test_smoke_updaters.py` | All three updaters produce a finite loss, finite outputs and non-zero, finite `i2o` weights |
| `test_reproducibility.py` | Seeding, strict deterministic mode, RNG capture/restore, seeded data order and workers |
| `test_failure_paths.py` | Checkpoint compatibility, missing or unreadable checkpoints, explicit resume, non-finite loss, time-limit (124) and SIGTERM (143) exits |
| `test_metrics.py` | Interval metrics, recall targets and chance levels |
| `legacy/test_plast_clip_update.py` | Changing `--plast_clip` on resume updates checkpoint plasticity; RNG round-trip |

## What the golden trace is

`fixtures/training_traces.json` records one call to the real `train.train()` per
updater, built by `characterization.py`: seed 1729, strict deterministic mode,
one Torch thread, CPU. The input is two five-token sequences over `abcd`. The
model has one layer and hidden size 4, `last_two` input and recurrence on, with
normalization and weight clipping off. lr 0.01, `grad_clip` 0.2, α
(`plast_clip`) 3.0, `forget_rate` 0.25, `plast_proportion` 0.5. These are
deliberately not the CLI defaults. Every argument is passed explicitly, so CLI
default changes never reach the trace.

The test compares every recorded value at rel 1e-6: inputs, per-step outputs
and labels, the final state of every EphemeralLinear layer (candidate weights,
masks, plasticity, forgetting, update norms), and before/after summaries around
each forget step, gradient scaling and unified update. It fails if the Torch
version differs from the one recorded in the fixture; it does not check Python
or NumPy versions. The loss alone is a weak signal: all three losses sit near
ln 4, and the tensor comparison is what catches changes.

Not covered: `normalize=True`, `clip_weights != 0`, positional encoding,
`self_grad > 0`, more than one layer, the SimpleRNN baseline, and metric
outputs. In a single call BPTT moves only `i2o` and the biases; its other
candidate weights are still zero afterwards.

## Behaviour the trace currently freezes

| Updater | Loss | Forget calls | Gradient-scale calls | Unified-update calls (linear / `i2h`) | `training_instance` |
| --- | ---: | ---: | ---: | ---: | ---: |
| DFA | 1.4015415907 | 4 | 0 | 4 / 0 | 4 |
| Backprop | 1.4014136195 | 4 | 4 | 4 / 4 (all no-ops: `i2h` gradient is `None`) | 4 |
| BPTT | 1.3995014429 | 1 | 1 | 0 / 0 (manual SGD step) | 0 |

### Pinned known bugs

The trace is observational: it freezes today's behaviour, including the items
below, which are documented in the main README under "Known issues / behaviours
under review". A fix to any of them is expected to fail the golden test.

| Behaviour | Where | Trace that changes when it is fixed |
| --- | --- | --- |
| Backprop's ephemeral step is α² (`scale_gradients` multiplies by α, then `apply_unified_updates` multiplies by `plasticity`); DFA and BPTT apply α once | `train.py` backprop branch; `ephemeral_model.py` `scale_gradients`, `apply_unified_updates` | backprop |
| DFA never populates or updates `i2h` (by design; under review) | `train.py` DFA branch | DFA, if `i2h` starts being updated |
| Backprop's `i2h` gradient is always `None` because the hidden state is detached every step (by design; under review) | `train.py` hidden detach; `EphemeralRNN.forward` | backprop, only if truncation changes |
| `grad_clip` clamps the α-scaled update on masked entries only; it binds in the DFA trace at 0.2 | `apply_unified_updates` | DFA |
| Ephemeral BPTT ignores `grad_clip`, `clip_weights` and `normalize` | `train.py` BPTT branch | BPTT only if clipping is added (weight clip and normalize are off in the trace) |
| Ephemeral BPTT never increments `training_instance` | `train.py` BPTT branch | BPTT |
| Forgetting runs before the update, as `1 - forget_rate` (the paper puts γ after) | `train.py` all branches; `apply_forget_step` | all three |

The 1/B factor in backprop and BPTT (batch-mean loss before `backward()`) is
also frozen, but it was a deliberate choice rather than a bug.

## Regenerating

Policy: each future bug fix that changes the trace regenerates the fixture in
the same commit, and adds a row to the log below explaining why. Review the
diff of the fixture before committing it.

```bash
CUDA_VISIBLE_DEVICES="" python -m tests.generate_characterization_fixtures
```

Pass `--output PATH` to write somewhere else for comparison.

### Regeneration log

| Date | Commit | Reason |
| --- | --- | --- |
| 2026-08-28 | 43b34c4 | Initial baseline (manticore, CPU, Python 3.11.12, Torch 2.5.1, NumPy 2.2.5, one thread) |
