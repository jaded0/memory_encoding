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

## 1. Weights start at zero: investigate and decide (behaviour change, needs benchmarks)
Finding: `per_sample_weights` (both slow and ephemeral entries) are initialised with
`torch.zeros` (`ephemeral_model.py`, `EphemeralLinear.__init__`), and nn.Linear's own
`self.weight` (default Kaiming-uniform init) is never read by the forward pass. So every
EphemeralRNN layer's weights start at exactly zero; only the biases are non-zero.

History (checked with `git log -S/-G`):
- 2024-03 → 2024-07: `self.weight` had `xavier_uniform_(gain=…)`; the forward used `self.weight`,
  and `candidate_weights` (today's `per_sample_weights`) started at zero as an add-on.
- 170ec86 (2024-07-05): reverted `self.weight` to nn.Linear's default init (still non-zero).
- **8dd8294 (2025-02-03, "implemented real batching")**: the forward switched to
  `torch.bmm(candidate_weights, …)` with `candidate_weights = zeros(B, out, in)`, and the
  `self.weight.data += update` lines were commented out. From then on the non-zero init no
  longer reached the forward pass. This matches Jaden's memory of "nonzero init for the
  parameters I defined, not the main weights" only in part: it was the main weights that
  were non-zero before 8dd8294.
- SimpleRNN uses ordinary nn.Linear init, so the two models already differ here, which also
  matters for the baseline-matching in §2.

Suggested fix (ask Jaden before landing): initialise the **slow** entries of
`per_sample_weights` from the layer's default `self.weight` init (repeated over the batch) and
leave the ephemeral entries at zero, since `start_sequence_wipe()` zeroes them each sequence
anyway. Consider then dropping or reusing `self.weight` so it is no longer an unused parameter.
Keep the RNG stream order in mind: `self.weight`'s init is already drawn, so reusing it adds no
draws. Needs full before/after benchmark runs (like the Elman change; Deckard for now).

## 2. SimpleRNN baseline matches the ephemeral model's architecture automatically
Jaden: "parameterized/computed to automatically match when run with otherwise same
parameters." Today SimpleRNN's hidden layers use **ReLU** (EphemeralRNN: **GELU**) and are
**`hidden_size` wide** (EphemeralRNN's hidden layers are `input + hidden` wide, `inner_size`).
- Make SimpleRNN compute its layer widths and activation the same way EphemeralRNN does, from
  the same constructor arguments, ideally via one shared helper so they cannot drift.
- Check residual connection and positional-encoding handling match too.
- Changes rnn/backprop, rnn/bptt and `dfa/rnn` behaviour: regenerate traces, bump the code
  version, and note that old SimpleRNN runs aren't comparable.
- The README's Updaters section ("The architectures still differ outside DFA …") must then be
  updated.

## 3. rnn + dfa (and SimpleRNN generally) applies `--weight_clamp` and `--unit_norm_weights`
Today SimpleRNN ignores both under every updater. Apply them after each update, the same way
`EphemeralLinear._apply_regularization` does (per-sequence unit norm there; SimpleRNN has one
shared weight copy, so the unit norm is over the layer's `[out, in]` weight). Put the logic in a
shared helper used by both `EphemeralLinear` and `DFALinear` (and by the SimpleRNN backprop/BPTT
path, per the rule above). Decide and document whether the bias is included (in the ephemeral
model it is not).

## 4. Hyperparameter parity audit (Jaden's rule above)
Go through every flag in `train.py`'s parser and make each apply to both models, or add a
curt comment where it cannot. Known cases:
- `--grad_norm_clip` (SimpleRNN; also under rnn+dfa) vs `--ephemeral_update_clamp` (ephemeral
  only). Keep `--grad_norm_clip` under rnn+dfa (Jaden: keep it) and add a comment that it breaks
  the exact match with the ephemeral model's DFA when non-zero. Consider whether the ephemeral
  model should support `--grad_norm_clip` too (on backprop/BPTT that is well defined).
- `--plasticity`, `--ephemeral_fraction`, `--forget_rate`: fundamentally ephemeral-only
  (SimpleRNN has no ephemeral entries); a one-line comment suffices.
- `--weight_clamp`, `--unit_norm_weights`: §3.
- Ephemeral BPTT ignores `--ephemeral_update_clamp`, `--weight_clamp`, `--unit_norm_weights`
  (tests/README.md, pinned behaviours). Decide whether that is "fundamental" (fast weights never
  reach a forward pass under BPTT) and comment accordingly.
- `run_training.sh` / `slurm_run.sh` choose `GRAD_CLIP_FLAG` by model type; simplify once flags
  are unified.

## 5. Batch-mean DFA step for SimpleRNN's shared weights: document the justification
Jaden: keep the batch mean, "but this should be well thought-out with documented
justification." Write it in the README (Updaters, rnn + dfa) and at `DFALinear`'s update:
- SimpleRNN has one weight copy for the whole batch. The ephemeral model's slow entries are
  per sequence during a sequence and are averaged over the batch at `start_sequence_wipe()`,
  so over a sequence their net step is the batch mean of the per-sequence steps.
  Taking the batch mean each step in SimpleRNN gives the same expected step size per sequence,
  independent of batch size, and matches the DFA bias step (already a batch mean in both).
- A batch sum would scale the effective learning rate with B, making `--learning_rate`
  mean different things at different batch sizes and in the two models.
- Caveat to state: averaging per step vs averaging at sequence end are not identical when the
  weights feed back into later steps within a sequence (the ephemeral model's per-sequence copies
  diverge during a sequence, SimpleRNN's don't). Also note the backprop/BPTT 1/B (batch-mean
  loss) is a related but separate choice (README "Known issues").
- Consider a small test pinning that the rnn+dfa step equals the mean of the per-sequence DFA
  gradients.

## 6. Research, not code yet
- **DFA and f′** (README "Known issues"): examine whether DFA should include the activation
  derivative (Nøkland 2016). Any change goes in the shared `dfa_*` helpers for both models.
- **α² in backprop** and the **1/B factor**: fix only with full before/after runs.
- **Benchmarks on Deckard** (`ssh jaden@deckard`): Elman layout, before = aef0697,
  after = 59db471 (HEAD behaves identically for ephemeral runs up to this doc's commit).
  §1 and §2 need the same treatment.

## Already decided, nothing to do
- `EPHEMERAL_AUTO_PREPROCESS=1` forces preprocessing even under SLURM: already implemented
  (`preprocess.auto_preprocess_enabled`).
- The AST code hash depends on the Python version: accepted (`environment.yml` pins 3.11).
- One global `CHECKPOINT_CODE_VERSION`: kept.
- `self_grad`: removed (see `docs/self_grad.md`).
