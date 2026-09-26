# Next steps (handoff, 2026-09-23)

Decisions Jaden made on 2026-09-23 that are **not implemented yet**. Each is a separate
commit. Follow the golden-trace policy in `tests/README.md`: a change that alters a trace
regenerates `tests/fixtures/training_traces.json` in the same commit, adds a row to the
regeneration log, and bumps `CHECKPOINT_CODE_VERSION` (`utils.py`) if the values change.
Run tests with the `hebby` env: `python -m pytest tests/ -q` (all passing at the time of
writing, including with the GPUs visible).

Guiding rule from Jaden: **both model architectures (`EphemeralRNN`, `SimpleRNN`) support the
same hyperparameters, except where it fundamentally makes no sense; in that case a curt
comment at the point of use says why.**

## 1. Hyperparameter parity audit (Jaden's rule above): done 2026-09-25
- `--grad_norm_clip` now applies to both models under every updater. The ephemeral model clips
  each sequence's raw gradient (its weight slices and bias shares) before α
  (`EphemeralRNN.clip_grad_norm_per_sequence`; Jaden chose per sequence over whole batch).
  SimpleRNN keeps `clip_grad_norm_` on its shared gradients. `tests/test_grad_norm_clip.py`.
- Ephemeral BPTT now applies `--weight_clamp` and `--unit_norm_weights`, which bound the slow
  weights it trains. A layer with no gradient in a step (`i2h` under per-step backprop) is now
  regularized too. `--ephemeral_update_clamp` stays ephemeral DFA/backprop only; README
  "Updaters" explains why. `CHECKPOINT_CODE_VERSION` 12.
- `run_training.sh` and `slurm_run.sh` set `GRAD_NORM_CLIP` and `EPHEMERAL_UPDATE_CLAMP`
  separately. The four old sweeps that passed one `$GRAD_CLIP` to both flags now pass only
  the model's historical one, so rerunning them never applies both clips.
- `--plasticity`, `--ephemeral_fraction`, `--forget_rate` remain ephemeral-only (no ephemeral
  entries in SimpleRNN).

## 2. Batch-mean DFA step for SimpleRNN's shared weights: done
The justification is in the README (Updaters, rnn + dfa) and at `DFALinear`.
`test_rnn_dfa_gradient_is_the_batch_mean_of_per_sequence_outer_products` pins the step.

## 3. Research, not code yet
- **DFA and f′** (README "Known issues"): examine whether DFA should include the activation
  derivative (Nøkland 2016). Any change goes in the shared `dfa_*` helpers for both models.
- **α² in backprop** and the **1/B factor**: fix only with full before/after runs.
- The historical topology-by-slow-initialization benchmark is complete. Seven fully ORC-paired
  seeds show a strong interaction: default initialization rescues most of the serial Elman
  layout's zero-init deficit, while the panel provides no clear evidence that it improves the
  historical sibling-head layout. See `benchmarks/3pal_architecture_initialization.md`. This does
  not directly compare current HEAD, whose restored fork trains `i2h` through direct DFA and has
  other subsequent architecture changes.

## 4. Test the paper's clipping claim in a separate worktree

The paper currently says that "gradient clipping ... fails," but the old ephemeral-model
`--grad_clip` flag was not conventional gradient-norm clipping. It element-wise clamped the
plasticity-scaled fast-weight update and is now named `--ephemeral_update_clamp`. Keep three
distinct interventions separate in code, run names, plots, and paper language:

- `--grad_norm_clip`: rescale a complete gradient vector by its norm before the step;
- `--ephemeral_update_clamp`: element-wise clamp the alpha-scaled update on fast entries; and
- `--weight_clamp`: clamp weights after the step (`--weight_clamp 1` is active in the current
  3-palindrome architecture/initialization benchmarks, while `--ephemeral_update_clamp 0` is
  off). Forgetting is damping, not clipping.

Do this work on its own branch/worktree so it does not mix with the architecture/init benchmark.

### 4a. Historical audit (no training)

1. Identify the commits, launch scripts, W&B groups, and exact code path behind every clipping
   result cited in `paper/paper_content.tex`, `paper/appendix.tex`, and `paper/pre-cut.tex`.
2. Record for each result whether it tested raw gradient-norm clipping, element-wise fast-update
   clipping, or post-update weight clipping, including the threshold and whether the clamp bound.
3. If the evidence used only the old ephemeral `--grad_clip` behavior, change the provisional
   interpretation from "gradient clipping fails" to "element-wise fast-update clipping did not
   help in that sweep." Do not edit the paper until the audit and controlled runs are reviewed.

### 4b. Implementation/characterization tests

Status 2026-09-25: `--grad_norm_clip` for the ephemeral model is implemented and tested
(section 1). A threshold that never binds (e.g. `1e30`) is bit-identical to no clip and logs
`grad_norm_mean`, `grad_norm_max` and `grad_norm_clip_fraction` per print interval, which is
the measurement the pilot needs. The α-scaled update norms are the existing
`*_ephemeral_update_norm` / `*_slow_update_norm` logs.

Before launching a new sweep, define `--grad_norm_clip` for the ephemeral model without changing
the existing two clamp operations. Match the SimpleRNN convention: compute the norm over all live
parameter gradients and rescale before the update. Explicitly document whether this is before or
after alpha scaling; log both the raw gradient norm and the actual alpha-scaled update norm so the
choice is visible. Add tests that pin:

- disabled clipping is trace-identical to the current control;
- global norm clipping binds at the requested norm and preserves gradient direction/ratios;
- `--ephemeral_update_clamp` changes only fast entries and clamps them element-wise after alpha;
- `--weight_clamp` bounds weights only after the update and does not alter gradients;
- the three mechanisms can be enabled independently and their update order is fixed; and
- DFA and backprop exercise the intended clipping path; treat BPTT separately because fast-weight
  updates are wiped before a subsequent training forward pass.

Any changed golden trace follows `tests/README.md`: regenerate the fixture, append its regeneration
log, and bump `CHECKPOINT_CODE_VERSION` when values change.

### 4c. Controlled training panel

Use the same model, dataset, seeds, initialization, training budget, and all non-clipping
hyperparameters in every arm. Start with ephemeral + DFA on the 3-palindrome task, because that is
the setting of the current claim. Use at least the seven fully ORC-paired benchmark seeds; if an
eighth seed is added, run every arm in the same environment. Compare one mechanism at a time
against an unclipped control:

- no clipping: all three clipping thresholds zero;
- global gradient-norm clipping: off plus several thresholds chosen from measured unclipped norms;
- element-wise fast-update clipping: off plus several thresholds chosen from measured unclipped
  fast-update magnitudes; and
- post-update weight clipping: off plus several thresholds, including 1 for continuity with the
  current launcher setting.

Use a short pilot only to choose thresholds that span never-binding, intermittently binding, and
nearly-always-binding regimes; do not select the final conclusion from the pilot. In final runs,
log the fraction of steps on which each mechanism binds, pre/post norm or magnitude, non-finite
events, recall accuracy, lag-specific recall, exact-sequence recall, loss, and time-to-threshold.
Report paired seed differences and uncertainty, not only the best threshold or mean curve.

Only make the broad paper claim about conventional gradient clipping if the global-norm panel
supports it. Otherwise name the exact operation tested and scope the conclusion to the updater,
task, threshold range, and training regime.

## Implemented from this handoff
- Slow entries of `per_sample_weights` start from `nn.Linear`'s default initialization,
  repeated over the batch without another RNG draw; fast entries start at zero. Implemented
  after Jaden accepted the proposed fix on 2026-09-23. Full before/after benchmarks remain.
- Main-code DFA throughput was benchmarked against the scratch harness and profiled. Allocation
  and gradient-clearing changes improved native throughput 15-20% without changing any golden
  trace; methodology and raw results are under `benchmarks/`.
- SimpleRNN now derives the same `input + hidden` trunk width, GELU activation, residual
  placement, forked heads, recurrence switch, and positional input width as EphemeralRNN.
- `--unit_norm_weights` and `--weight_clamp` now apply after every SimpleRNN update, using a
  weights-only helper shared with EphemeralLinear; normalization precedes the clamp.
- The 3-palindrome historical topology-by-initialization factorial completed across seven paired
  ORC seeds. Standard initialization improved serial-Elman mean recall by 27.47 percentage points
  but changed historical sibling-head recall by -3.11 points; the paired interaction was +30.57
  points (95% t interval [19.28, 41.87]). Full scope and caveats are in
  `benchmarks/3pal_architecture_initialization.md`.

## Already decided, nothing to do
- `EPHEMERAL_AUTO_PREPROCESS=1` forces preprocessing even under SLURM: already implemented
  (`preprocess.auto_preprocess_enabled`).
- The AST code hash depends on the Python version: accepted (`environment.yml` pins 3.11).
- One global `CHECKPOINT_CODE_VERSION`: kept.
- `self_grad`: removed (see `docs/self_grad.md`).
