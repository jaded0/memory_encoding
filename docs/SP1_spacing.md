# SP1: spacing with plasticity bands (opt-in)

## Model and supported combinations

`--plasticity_bands '0.05:30000:0.03,0.05:3000:0.001'` specifies
whole-matrix `fraction:alpha:forget` triples. Fractions must be positive and sum
to at most 1; alpha is nonnegative and forgetting is in [0,1], all finite.
Their sum replaces `ephemeral_fraction`. Empty bands and default shared feedback
leave legacy behavior unchanged.

Bands require ephemeral **per-character DFA**, exclusive weights,
`slow_update_every=1`, and `fast_backward_per_forward=1`. They reject order-two
state, decoupled/readout-derived feedback overrides, fast-read normalization,
plasticity schedules, and gradient-norm clipping. They do not support K14
sequence-outer/backprop training. Existing fast-layer selection, activation
options, slow-gradient normalization/scales, write/weight clamps, carried fast
state, and eager fused/unfused DFA remain available. The standalone runner's
`--fused_update` is optional CUDA-only compiled DFA; CPU uses the unfused path.

Each layer partitions its **existing random ephemeral mask**, without redrawing
or enlarging support. Band counts are cumulative proportional rounding on that
mask's realized Bernoulli support, not exact requested whole-matrix counts.
Excluded layers and the output head keep their empty fast support. A local RNG
makes assignment deterministic without consuming global RNG state. Band IDs,
forgetting maps and configuration are checkpointed; incompatible band/feedback
configuration is refused. Bands add fixed buffers, **no new learnable parameters**.

`--band_feedback shared` uses the usual DFA projection for all bands.
`separate` uses deterministic independent fixed random projections per band and
non-output layer. All projections receive the **same output error**, already
including the fast read: this is not independent errors, an independent
magnitude source, distillation, or a memory-level delta rule. Slow updates retain
the original shared DFA projection.

## Flat-write control and inference API

`--write_gain_power -1` multiplies a nonzero projected error by its causal
running-median norm divided by its current norm. History uses training steps
only; inference reads but never advances it. Before history exists the gain is
1; zero errors remain zero. `--write_gain_window` sets the history window and
`--write_gain_max` can cap the gain (0 means uncapped). With separate feedback,
histories are per layer/projection. The uncapped control normalizes **projected
error norm**, not the complete outer-product write: input/key norms, mask
geometry, alpha, clamps and forgetting still affect actual weight changes.

```python
model.eval()
model.wipe_bands([0])              # zero selected fast entries, no consolidation
with model.band_forget_override({0: 0.01}):
    # Run fast-only DFA evaluation; slow weights and biases stay frozen.
    ...
```

Indices are zero-based. Forget overrides are inference-only (`eval()` required),
finite rates in [0,1], and restore on exit, including nested contexts/exceptions.
They do not change training configuration. Use `wipe_fast_state()` to wipe all
fast entries without consolidating slow copies.

## Standalone runner and evaluation

`spacing_run.py` selects `--arm shared|separate|flat|oneband`; `--bands` aliases
`--plasticity_bands`. Flat uses shared feedback with power -1; oneband uses a
single fraction-0.1 band with `--plasticity` and `--forget_rate`.
`--checkpoint`, `--load`, and `--eval_only` support atomic saves and exact
stream/RNG resume. `--train_steps` is an absolute **row-batch** target on resume,
not streams or character steps.

`--eval_protocols no_wipe wipe_band0 wipe_band1 wipe_all` chooses query
interventions; `--eval_wipe_bands 0,1` appends a custom selected-band wipe.
Oneband defaults to no-wipe/all-wipe and refuses the named two-band protocols.
Wipes occur **before the entire query row**, including its filler prefix, not
immediately before the answer. Query-prefix writes can therefore rebuild state.
`--forget_multipliers` changes only band 0 during inference. Explicit
`--forget_rate_override 0:0.02,1:0.005` requires `--eval_only` and
`--forget_multipliers 1`.

Evaluation uses a private model with slow copies consolidated once, then freezes
slow entries, biases and output head. It leaves the training model, mode, fast
state, gain history and RNG unchanged. Bindings/query identities are paired
across spacing, forgetting and wipe cells; shared prefixes are reused for query
forks. Recall is unrestricted vocabulary argmax on the first/only query answer,
scored **before** its teacher-forced write. All ordinary target writes remain
enabled; padding tail streams do not contribute to metrics.

## Stream semantics and production gate

`spacing_task.py` generates fresh random persistent key/value tables per stream
and batch row, without finite-table epoch replay. Training mixes spacing and
retention schedules; held-out splits are disjoint. Each stream presents the same
table R times (default 3), then queries it once. Hidden state resets each row;
fast state carries within the stream and wipes at stream starts. Slow weights
can adapt during training, but are frozen in evaluation.

I is onset-to-onset spacing in **sequences** (I=1 is consecutive/massed). D counts
complete distractor sequences after the last presentation and before the query.
Each row has `2*K+1` characters and `2*K` next-character updates. Filler rows
perform ordinary writes and forgetting, not idle/no-write time. A common
left-padded length fixes compute and final query position across spacing cells.
Forgetting is per **character update**, not per sequence; half-life is
`log(0.5)/log(1-f)` updates for 0<f<1. Report sequence and character-update units.

The independent no-wipe learnability pilot uses disjoint seed 91105, not
production seeds or after-wipe peaks: select a horizon only after shared-arm
held-out no-wipe recall is at least 0.20 at nominal forgetting, I=1 and the chosen
D. All four arms then share that horizon/recipe except their named intervention;
other-arm learning failures are reported, not tuned away. Production no-wipe is
a safeguard, not a seed-exclusion rule. Pilot learnability does **not** satisfy
the separate WS1 mechanism gate: production may auto-launch **only on a recorded
WS1 PASS**; FAIL, AMBIGUOUS or a missing verdict holds production. No pilot
outcomes are asserted here.
