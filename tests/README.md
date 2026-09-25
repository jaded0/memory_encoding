# Tests and the golden-trace baseline

Run everything from the repository root (CPU only, network-free, about 5 s):

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest tests/ -q
```

`pytest` is needed because `tests/legacy/` has no `__init__.py`, so
`python -m unittest discover -s . -t .` does not collect it.

| Module | Covers |
| --- | --- |
| `test_characterization.py` | Golden traces for DFA, backprop and BPTT, in a base case and a `normalize_clip_2seq` case, and for the SimpleRNN baseline under DFA (`dfa/rnn`); a different seed changes `ephemeral_mask` |
| `test_smoke_updaters.py` | All three updaters produce a finite loss, finite outputs and non-zero, finite `i2o` weights |
| `test_reproducibility.py` | Seeding, strict deterministic mode, RNG capture/restore, seeded data order and workers |
| `test_failure_paths.py` | Checkpoint compatibility (including a changed `forget_rate`, `dataset` or `learning_rate`, and a different or missing `code_version`, in `load_checkpoint` and in a `--resume` that then trains nothing), the resume config diff (including an old config key), missing or unreadable checkpoints, explicit resume, non-finite loss, time-limit (124) and SIGTERM (143) exits |
| `test_legacy_checkpoints.py` | Checkpoints written before the 2026-09 renames (`fixtures/legacy_names`, made at 1775121 by `make_legacy_checkpoints.py` there). They have no `code_version`, so resuming them with the old or the new flag names is refused and leaves them untouched. At the unit level, `upgrade_legacy_state_dict` maps their state dicts exactly as the test's own independent renamer does, and `upgrade_legacy_config` maps their configs to the new keys. A bad `forgetting_factor`, or a missing or unexpected state-dict key, fails the load |
| `test_cli_aliases.py` | Old flag names parse to the new settings, each with one deprecation line, and none appears in `--help`. `--grad_clip` follows `--model_type`. Removed flags are ignored, and conflicting old and new values are an error. Old config keys map to what the old flags parse to |
| `test_dfa_error_signals.py` | The DFA path's per-layer error tensors (every layer, `i2h` included, is populated each step), projected errors, gradients and bias steps, checked at populate and at update time against values computed independently. It fails on an in-place change to the shared `output_error` (see the main README) |
| `test_preprocess.py` | Processed-dataset naming (the code hash ignores comment-only and docstring-only edits, not code edits), saved rows and batches (against the old one-hot pipeline), and a missing processed dataset: prepared automatically outside SLURM, the setup-hint error under SLURM, and the `EPHEMERAL_AUTO_PREPROCESS` override both ways (raw download mocked) |
| `test_metrics.py` | Interval metrics, recall targets and chance levels |
| `test_layer_mechanics.py` | Single mechanics checked in isolation: `--unit_norm_weights` rescales each sequence's slice independently. In the forked layout, output and state read the shared trunk independently; `i2h` gets explicit DFA every step, no same-step backprop gradient, and future credit under BPTT; recurrence-off still executes both heads but feeds back zeros. SimpleRNN DFA mechanics and clipping are checked independently |
| `legacy/test_plast_clip_update.py` | Changing `--plasticity` on resume updates checkpoint plasticity; RNG round-trip |

## What the golden trace is

`fixtures/training_traces.json` records calls to the real `train.train()`,
built by `characterization.py`: seed 1729, strict deterministic mode, one Torch
thread, CPU. There are two cases per updater, and one SimpleRNN case under DFA:

- **Base** (keys `dfa`, `backprop`, `bptt`): one call on a batch of two
  five-token sequences over `abcd`. The model has one layer and hidden size 4,
  `last_two` input and recurrence on, with `unit_norm_weights` and
  `weight_clamp` off. lr 0.01, `ephemeral_update_clamp` 0.2, α (`plasticity`)
  3.0, `forget_rate` 0.25, `ephemeral_fraction` 0.5.
- **`normalize_clip_2seq`** (keys `<updater>/normalize_clip_2seq`; the name
  predates the flag renames): the same model and seed with
  `unit_norm_weights=True`, `weight_clamp` 0.2 and lr 1.0, and two consecutive
  calls on the same model (the base batch, then a second batch), so the second
  call starts from the first call's weights and `start_sequence_wipe()`. The
  trace stores each call's inputs, outputs and loss under `calls`, and the
  model state and event log after both. `weight_clamp` is 0.2 because the
  unit-norm rescaling runs first and leaves no entry above 1, so a clamp of 1
  never binds. lr is 1.0 so that the step BPTT takes on `i2h` after the
  second sequence (about lr², since it goes through the `i2o` weights the first
  sequence set) sits well above the comparison's `abs_tol` of 1e-7.
- **`rnn`** (key `dfa/rnn`, DFA only): the SimpleRNN baseline (one layer,
  hidden size 4, `last_two` input, recurrence on) with `updater='dfa'`, lr 0.1,
  `grad_norm_clip` 0, and the same two consecutive calls. It stores each call
  as above, and after both calls each layer's `weight`, `bias`, their
  gradients, `feedback_weights` (none for `i2o`) and input trace, plus a
  before/after summary around every `apply_dfa_update` call.

These settings are deliberately not the CLI defaults. Every argument is passed
explicitly, so CLI default changes never reach the trace.

The test compares every recorded value at rel 1e-6: inputs, per-step outputs
and labels, the final state of every EphemeralLinear layer (`per_sample_weights`,
`ephemeral_mask`, plasticity, the per-entry forget rate `forget_rate * ephemeral_mask`
under the key `forgetting_factor`, update norms), and before/after summaries around
each forget step, gradient scaling and `apply_update` call. It fails if the Torch
version differs from the one recorded in the fixture; it does not check Python
or NumPy versions. The loss alone is a weak signal: the three base losses sit
near ln 4, and the tensor comparison is what catches changes.

Not covered: positional encoding, more than one layer, the SimpleRNN baseline under backprop
and BPTT, and metric outputs. The focused mechanics tests cover the initialization invariant:
slow entries use the already-drawn `nn.Linear.weight` values in every per-sequence copy, while
fast entries start at zero. The output head has no fast entries and therefore starts entirely
from the default initialization.

## Behaviour the trace currently freezes

| Updater | Loss | Forget calls | Gradient-scale calls | `apply_update` calls (linear / `i2h`) | `training_instance` |
| --- | ---: | ---: | ---: | ---: | ---: |
| DFA | 1.4024642706 | 4 | 0 | 4 / 4 | 4 |
| Backprop | 1.4026465416 | 4 | 4 | 4 / 4 (all `i2h` no-ops) | 4 |
| BPTT | 1.4003649950 | 1 | 1 | 0 / 0 (manual SGD step) | 0 |
| DFA, `normalize_clip_2seq` | 1.5828732252, 1.5748171806 | 8 | 0 | 8 / 8 | 8 |
| Backprop, `normalize_clip_2seq` | 1.6082669795, 1.5170544088 | 8 | 8 | 8 / 8 (`i2h` all no-ops) | 8 |
| BPTT, `normalize_clip_2seq` | 1.4003649950, 1.3702043295 | 2 | 2 | 0 / 0 (manual SGD step) | 0 |
| DFA, `rnn` (SimpleRNN) | 1.4461380243, 1.4681148529 | 0 | 0 | 8 / 8 (`apply_dfa_update`; `i2o` 8 too) | 8 |

In `normalize_clip_2seq`, BPTT's first loss equals the base case's, because the
update comes after the last step and BPTT ignores `unit_norm_weights` and
`weight_clamp`.

### Pinned known bugs

The trace is observational: it freezes today's behaviour, including the items
below, which are documented in the main README under "Known issues / behaviours
under review". A fix to any of them is expected to fail the golden test.

| Behaviour | Where | Trace that changes when it is fixed |
| --- | --- | --- |
| Backprop's ephemeral step is α² (`scale_ephemeral_grads` multiplies by α, then `apply_update` multiplies by `plasticity`); DFA and BPTT apply α once | `train.py` backprop branch; `ephemeral_model.py` `scale_ephemeral_grads`, `apply_update` | backprop |
| `ephemeral_update_clamp` clamps the α-scaled update on masked entries only; it binds in the DFA trace at 0.2 | `apply_update` | DFA |
| Ephemeral BPTT ignores `ephemeral_update_clamp`, `weight_clamp` and `unit_norm_weights` | `train.py` BPTT branch | `bptt/normalize_clip_2seq` if weight clamping or unit-norm rescaling is added; `bptt` (base) if the update clamp is |
| Forked `i2h` gets direct same-step DFA as a surrogate for temporal credit, while per-step backprop cannot train it | `EphemeralRNN.forward`, `SimpleRNN.forward`; `train.py` DFA branch | all seven if topology or credit routing changes |
| DFA omits the activation derivative: hidden layers and `i2h` use `output_error @ feedback_weights` without ⊙ f′(a) (Nøkland 2016 includes it); under review, not a confirmed bug (main README) | `ephemeral_model.py` `dfa_projected_error`, shared by `EphemeralLinear` and `DFALinear` | `dfa`, `dfa/normalize_clip_2seq`, `dfa/rnn` |
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
| 2026-09-23 | 092d434 | Forget step moved after the update in all three updaters (the paper's order, `w ← (1 − forget_rate)·(w − lr·α·g)`); under DFA and backprop it now also follows the clamp and normalize. All six traces change. DFA and backprop losses move by 1.6e-5 to 1e-2 (base about 3e-5 and 2e-5 lower; `normalize_clip_2seq` 0.6e-3 to 1e-2 lower). BPTT losses are unchanged (its update comes after the last step, and `wipe()` zeroes the decayed entries before the next forward pass); only its final state and event log differ |
| 2026-09-23 | a017f44 | `--normalize` now rescales only each layer's `candidate_weights`, no longer `plasticity`, `forgetting_factor`, the bias, the feedback weights, the traces or the logged update norms. Only `dfa/normalize_clip_2seq` and `backprop/normalize_clip_2seq` change; the other four traces are identical. Their losses drop a lot: DFA 1.862 → 1.619 and 1.640 → 1.568, backprop 1.931 → 1.612 and 1.606 → 1.519 (α stays 3.0 instead of shrinking to about 0.1, and the logged norms are real values, not about 1) |
| 2026-09-23 | 5fae219 | Last layers (`i2o`, `self_grad`) now have an empty mask, so none of their entries are decayed or wiped. All six traces change: those layers' `mask` and `forgetting_factor` are all zero, and their high-plasticity update norm is 0. The hidden-layer and `i2h` masks and all feedback weights are identical. Losses rise slightly: DFA base +5.6e-5, backprop base +2.8e-5, DFA `normalize_clip_2seq` +9.0e-3 and +8.6e-3, backprop `normalize_clip_2seq` +1.6e-3 and +3.8e-3. BPTT's base loss is identical and its second `normalize_clip_2seq` loss moves by -2.7e-6 (the `i2o` update after the first sequence is no longer partly wiped) |
| 2026-09-23 | 79aab73 | Renames only; values identical. Event key `unified_update` → `update` (the method is now `apply_update`). Checked by renaming that key in the previous fixture and comparing: equal, and byte-identical when dumped |
| 2026-09-23 | eeb0d89 | Renames only; values identical. State-dict names in the module snapshots: `candidate_weights` → `per_sample_weights`, `candidate_gradient` → `per_sample_gradient`, `mask` → `ephemeral_mask`, `last_high_plast_update_norm` / `last_low_plast_update_norm` → `last_ephemeral_step_norm` / `last_slow_step_norm`. `forgetting_factor` is no longer stored; the snapshot records `forget_rate * ephemeral_mask` under the same key, and its values are identical. Checked by renaming those keys in the previous fixture and comparing: equal, and byte-identical when dumped |
| 2026-09-23 | d7905ac | Renames only; values identical. CLI and config names in each trace's `configuration`: `grad_clip` → `ephemeral_update_clamp`, `plast_clip` → `plasticity`, `plast_proportion` → `ephemeral_fraction`, `clip_weights` → `weight_clamp`, `normalize` → `unit_norm_weights`. Checked by renaming those keys in the previous fixture and comparing: equal, and byte-identical when dumped. The case key `normalize_clip_2seq` is unchanged |
| 2026-09-23 | aef0697 | `--unit_norm_weights` now normalises each sequence's `[out, in]` slice of `per_sample_weights` separately instead of the whole `[batch, out, in]` tensor, so a sequence's scale no longer depends on the others (Jaden's choice). Only `dfa/normalize_clip_2seq` and `backprop/normalize_clip_2seq` change; the other four traces are identical (BPTT ignores `unit_norm_weights`). Losses: DFA 1.6278 → 1.6381 and 1.5768 → 1.5860; backprop 1.6138 → 1.6136 and 1.5223 → 1.5339. `CHECKPOINT_CODE_VERSION` 2 → 3 |
| 2026-09-23 | 59db471 | **Elman layout ("option A", approved by Jaden). Needs full before/after benchmark runs; compare against its parent.** Both models now compute `h_t = tanh(i2h(combined))` and `y_t = i2o(h_t)` (before, `i2o` read `combined`); `self_grad` also reads `h_t`, so `i2o` and `self_grad` are `[B, vocab, hidden]` instead of `[B, vocab, input + hidden]`, and `i2h` gets a DFA error projection and update like the hidden layers. All six traces change. `i2h` is now updated in DFA (4 / 8 calls, non-zero from the first step) and backprop (non-zero from the second step), and BPTT's second `normalize_clip_2seq` sequence moves `i2h` instead of the hidden layer. Losses: DFA 1.4016 → 1.4558, backprop 1.4014 → 1.4556, BPTT 1.3995 → 1.4539; `normalize_clip_2seq` DFA 1.6381 → 1.6721 and 1.5860 → 1.6016, backprop 1.6136 → 1.5848 and 1.5339 → 1.5525, BPTT 1.3995 → 1.4539 and 1.3863 → 1.3889. Most of the base rise is initialisation, not learning: BPTT's first loss is the untrained model's, whose output is `i2o.bias`, and that bias is now drawn with bound 1/√4 instead of 1/√12 (and the RNG stream after `i2o` shifts). `CHECKPOINT_CODE_VERSION` 3 → 4 |
| 2026-09-23 | this commit (see `git log -- tests/fixtures/training_traces.json`) | Added case `dfa/rnn`: the SimpleRNN baseline under DFA, which until now changed no parameters and now takes the ephemeral model's DFA step without ephemeral weights (every layer, `i2h` included, every step; see the main README's Updaters). The six existing entries are byte-identical (each dumped the same, and the fixture's patience diff is additions only); `generated_with` is unchanged (same machine and versions). Losses 1.4805 and 1.4799. SimpleRNN under backprop and BPTT was also checked outside the fixture: bit-identical losses, state dicts and RNG afterwards against the parent commit. `CHECKPOINT_CODE_VERSION` 4 → 5 |
| 2026-09-23 | this commit (see `git log -- tests/fixtures/training_traces.json`) | Removed the `self_grad` head and `--self_grad` (an abandoned experiment; see `docs/self_grad.md`). The traces only lose the `self_grad` layer's entries; everything else, losses included, is identical (it was the last layer built and consumed no RNG afterwards, and `--self_grad 0` never touched the error). `CHECKPOINT_CODE_VERSION` 5 → 6 |
| 2026-09-23 | this commit (see `git log -- tests/fixtures/training_traces.json`) | Slow entries of every `EphemeralLinear.per_sample_weights` now start from the layer's already-drawn default `nn.Linear.weight`, repeated over the batch; fast entries remain zero. All six ephemeral traces change from the first forward pass. Masks and feedback matrices are identical because initialization adds no RNG draws, and `dfa/rnn` is byte-identical. Base losses: DFA 1.4558 → 1.4534, backprop 1.4556 → 1.4530, BPTT 1.4539 → 1.4507. `normalize_clip_2seq`: DFA 1.6721 → 1.6770 and 1.6016 → 1.6151; backprop 1.5848 → 1.6608 and 1.5525 → 1.5241; BPTT 1.4539 → 1.4507 and 1.3889 → 1.3847. `CHECKPOINT_CODE_VERSION` 6 → 7 |
| 2026-09-24 | this commit (see `git log -- tests/fixtures/training_traces.json`) | Restored a forked transition/emission graph while retaining tanh on output features: `h_t = tanh(i2h(combined))`, `y_t = i2o(tanh(combined))`. `i2h` remains explicitly direct-feedback-trained under DFA, gets no same-step per-step-backprop gradient, and receives future credit under BPTT. Every trace changes, including `dfa/rnn`; base losses are DFA 1.4025, backprop 1.4026, BPTT 1.4004. `CHECKPOINT_CODE_VERSION` 7 → 8. See `docs/tapped_vs_forked_rnn_report.md` |
