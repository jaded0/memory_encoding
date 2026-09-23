# Tests and the golden-trace baseline

Run everything from the repository root (CPU only, network-free, about 5 s):

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest tests/ -q
```

`pytest` is needed because `tests/legacy/` has no `__init__.py`, so
`python -m unittest discover -s . -t .` does not collect it.

| Module | Covers |
| --- | --- |
| `test_characterization.py` | Golden traces for DFA, backprop and BPTT, in a base case and a `normalize_clip_2seq` case; a different seed changes the plasticity mask |
| `test_smoke_updaters.py` | All three updaters produce a finite loss, finite outputs and non-zero, finite `i2o` weights |
| `test_reproducibility.py` | Seeding, strict deterministic mode, RNG capture/restore, seeded data order and workers |
| `test_failure_paths.py` | Checkpoint compatibility, missing or unreadable checkpoints, explicit resume, non-finite loss, time-limit (124) and SIGTERM (143) exits |
| `test_metrics.py` | Interval metrics, recall targets and chance levels |
| `legacy/test_plast_clip_update.py` | Changing `--plast_clip` on resume updates checkpoint plasticity; RNG round-trip |

## What the golden trace is

`fixtures/training_traces.json` records calls to the real `train.train()`,
built by `characterization.py`: seed 1729, strict deterministic mode, one Torch
thread, CPU. There are two cases per updater:

- **Base** (keys `dfa`, `backprop`, `bptt`): one call on a batch of two
  five-token sequences over `abcd`. The model has one layer and hidden size 4,
  `last_two` input and recurrence on, with normalization and weight clipping
  off. lr 0.01, `grad_clip` 0.2, α (`plast_clip`) 3.0, `forget_rate` 0.25,
  `plast_proportion` 0.5.
- **`normalize_clip_2seq`** (keys `<updater>/normalize_clip_2seq`): the same
  model and seed with `normalize=True`, `clip_weights` 0.2 and lr 1.0, and two
  consecutive calls on the same model (the base batch, then a second batch),
  so the second call starts from the first call's weights and `wipe()`. The
  trace stores each call's inputs, outputs and loss under `calls`, and the
  model state and event log after both. `clip_weights` is 0.2 because
  `normalize` runs first and leaves no entry above 1, so a clip of 1 never
  binds. lr is 1.0 so that the step BPTT takes on the hidden layer after the
  second sequence (about lr², since it goes through the `i2o` weights the first
  sequence set) sits well above the comparison's `abs_tol` of 1e-7.

These settings are deliberately not the CLI defaults. Every argument is passed
explicitly, so CLI default changes never reach the trace.

The test compares every recorded value at rel 1e-6: inputs, per-step outputs
and labels, the final state of every EphemeralLinear layer (candidate weights,
masks, plasticity, forgetting, update norms), and before/after summaries around
each forget step, gradient scaling and unified update. It fails if the Torch
version differs from the one recorded in the fixture; it does not check Python
or NumPy versions. The loss alone is a weak signal: the three base losses sit
near ln 4, and the tensor comparison is what catches changes.

Not covered: positional encoding, `self_grad > 0`, more than one layer, the
SimpleRNN baseline, and metric outputs. In the base case's single call BPTT
moves only `i2o` and the biases, and its other candidate weights are still zero
afterwards. The second call of `normalize_clip_2seq` also moves the hidden
layer's slow weights. `i2h` never gets a non-zero update in any trace: under
BPTT its gradient passes through the hidden layer's weights, which are still
zero during the second sequence, so it would take a third.

## Behaviour the trace currently freezes

| Updater | Loss | Forget calls | Gradient-scale calls | Unified-update calls (linear / `i2h`) | `training_instance` |
| --- | ---: | ---: | ---: | ---: | ---: |
| DFA | 1.4015091658 | 4 | 0 | 4 / 0 | 4 |
| Backprop | 1.4013973176 | 4 | 4 | 4 / 4 (all no-ops: `i2h` gradient is `None`) | 4 |
| BPTT | 1.3995014429 | 1 | 1 | 0 / 0 (manual SGD step) | 0 |
| DFA, `normalize_clip_2seq` | 1.8621133566, 1.6398608685 | 8 | 0 | 8 / 0 | 8 |
| Backprop, `normalize_clip_2seq` | 1.9314314723, 1.6064965129 | 8 | 8 | 8 / 8 (`i2h` all no-ops) | 8 |
| BPTT, `normalize_clip_2seq` | 1.3995014429, 1.3863281012 | 2 | 2 | 0 / 0 (manual SGD step) | 0 |

In `normalize_clip_2seq`, BPTT's first loss equals the base case's, because the
update comes after the last step and BPTT ignores `normalize` and
`clip_weights`.

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
| Ephemeral BPTT ignores `grad_clip`, `clip_weights` and `normalize` | `train.py` BPTT branch | `bptt/normalize_clip_2seq` if weight clipping or normalization is added; `bptt` (base) if `grad_clip` is |
| `normalize` rescales every float parameter of a layer to unit norm after each update, including `plasticity`, `forgetting_factor`, the bias, the feedback weights, the traces and the logged update norms (which end up at about 1). After the first update α and `forget_rate` are no longer the values passed in (in the trace, α 3.0 becomes about 0.11 in the hidden layer) | `EphemeralLinear._apply_regularization` | DFA and backprop `normalize_clip_2seq` |
| `normalize` does not touch `i2h` under backprop, because `apply_unified_updates` returns before `_apply_regularization` when the gradient is `None` | `apply_unified_updates` | backprop `normalize_clip_2seq` |
| Ephemeral BPTT never increments `training_instance` | `train.py` BPTT branch | BPTT |

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
| 2026-09-23 | 8860326 | Added cases `dfa/normalize_clip_2seq`, `backprop/normalize_clip_2seq` and `bptt/normalize_clip_2seq` for normalize/clip_weights/2×BPTT. The three base entries are byte-identical (patience diff: additions only). Same machine and versions as the initial baseline |
| 2026-09-23 | this commit (see `git log -- tests/fixtures/training_traces.json`) | Forget step moved after the update in all three updaters (the paper's order, `w ← (1 − forget_rate)·(w − lr·α·g)`); under DFA and backprop it now also follows the clamp and normalize. All six traces change. DFA and backprop losses move by 1.6e-5 to 1e-2 (base about 3e-5 and 2e-5 lower; `normalize_clip_2seq` 0.6e-3 to 1e-2 lower). BPTT losses are unchanged (its update comes after the last step, and `wipe()` zeroes the decayed entries before the next forward pass); only its final state and event log differ |
