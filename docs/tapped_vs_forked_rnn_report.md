# Tapped versus forked recurrent state

> **Decision: use the forked transition/emission topology.** The project intentionally favors
> separate persistent-state and current-emission paths over strict textbook-Elman form. Describe
> it as a minimal transition/emission RNN and retain tapped Elman as a sensitivity baseline; do
> not claim that the fork is the canonical vanilla RNN.

## Question

The ephemeral-memory experiments need a small recurrent activation-memory baseline. Two
topologies are plausible:

```text
Tapped/shared state                         Forked transition and emission
z_t = M([x_t, h_(t-1)])                    z_t = M([x_t, h_(t-1)])
u_t = T(z_t)                               u_t = T(z_t)
h_t = tanh(u_t)                            h_t = tanh(u_t)
y_t = O(h_t)                               y_t = O(z_t)
```

The tapped model makes the recurrent state the representation read by the output. The forked
model gives the current output and persistent state separate heads over a common deep trunk.
The practical question is whether the baseline should prioritize textbook recognizability or
separation between what is remembered and what is emitted now.

Tanh is a separate design variable. It can bound the state fed through recurrence without
necessarily being placed on the immediate readout path. For example:

```text
u_t = T(z_t)
h_t = tanh(u_t)           # bounded recurrent carrier
y_t = O(ReLU(u_t))        # readout does not pass through tanh
```

## Historical context

Tapped state is the canonical vanilla-RNN design. Elman's hidden units both feed the output and
are copied into context for the next step. Standard textbook equations likewise use
`h_t = tanh(...)` and `y_t = O(h_t)`. GRUs use the same visible hidden state for recurrence and
readout. LSTMs distinguish protected cell state `c_t` from exposed state `h_t`, but the output
normally reads `h_t`, which also participates in the next recurrent computation. Linear and
modern structured state-space models similarly apply an emission projection to the updated
persistent state.

A fork is therefore not the most canonical "vanilla RNN." It is better described as a compact
transition/emission model. That separation is common and defensible in latent state-space
models, world models, and partially observed control, where future-relevant state and current
emission have different jobs. Transformer KV caches provide only a loose analogy: persistent
keys and values and final logits are different projections, but a growing content-addressable
cache is unlike a fixed-dimensional recurrent state.

For this project, either framing is defensible:

- Use tapped state when the principal claim is a canonical vanilla-RNN baseline.
- Use a fork when the principal claim is a deliberately simplified activation-memory proxy whose
  emission path should not be forced through recurrence-specific machinery.
- Do not claim that the fork is the historically standard RNN or a faithful Transformer/LSTM.
  Call it a transition/emission RNN and state why it is used.

## Temporal credit

Under full backpropagation through time, tapped state receives both current-output and future
temporal gradients. In a fork, the common trunk receives both, but the state head `T` is trained
only through future outputs. This makes the fork's state branch more distant from the loss, even
though the representation can specialize more freely.

The project's original failure to update forked `i2h` under DFA was an implementation oversight,
not an intended property of the topology. If the fork is restored, `i2h` will receive its own
fixed direct-feedback projection. This fixes the missing spatial learning signal. A same-step
projection is still a temporal surrogate because `i2h_t` affects later rather than current
outputs; standard DFA does not by itself reproduce BPTT's future-loss gradient. The exact
temporal DFA rule must therefore be reported explicitly.

### Future route: delayed DFA over retained activations

A future experiment should retain each timestep's `i2h` input/activation or a compressed
eligibility trace, then use later output errors to update the earlier state-writing event. In an
unrolled view, a future error `e_(t+k)` would be projected through a fixed feedback matrix and
paired with the retained local trace from time `t`; updates for tied recurrent weights would be
aggregated across time. This could provide temporally delayed direct feedback without multiplying
through the full chain of recurrent Jacobians.

This is a research direction, not the current update rule. It requires explicit decisions about
which future errors supervise each trace, trace lifetime/decay, online versus end-of-sequence
updates, memory cost, and whether stale activations remain valid after intervening online weight
updates. It is related in spirit to temporal DFA and eligibility-trace methods and should be
evaluated separately from the topology decision.

## Standalone experiment

A fresh standalone PyTorch experiment compared three equal-parameter deep vanilla RNNs under
Adam and full BPTT:

- `tapped_post_tanh`: output reads `tanh(u_t)`.
- `tapped_pre_tanh_relu`: output reads `ReLU(u_t)` while recurrence stores `tanh(u_t)`.
- `forked`: output reads common trunk representation `z_t`; recurrence stores `tanh(T(z_t))`.

Every arm used the same embedding, two-layer ReLU residual trunk, state projection, output
projection, initialization, batches, training budget, and 5,884 trainable parameters. The
expanded panel used 20 paired seeds per arm and task, deterministic CPU execution, Adam, full
BPTT, and recall-only loss. Tasks were delayed copying and delayed reversal of two independently
presented blocks. Fixed reversal panels evaluated block lengths three and four on 1,024 samples
per seed.

The primary comparison was forked versus textbook post-tanh tapped. The pre-tanh ReLU arm was a
secondary activation/readout diagnostic rather than the main topology test.

| Evaluation | Tapped token / exact | Forked token / exact | Paired fork advantage (95% CI) |
|---|---:|---:|---:|
| Copy, lengths 2-4 | 87.18% / 68.18% | **89.77% / 73.50%** | +2.59 pp [0.58, 4.61] / +5.31 pp [1.28, 9.34] |
| Two reversals, lengths 2-3 | 75.93% / 23.98% | **80.54% / 36.95%** | +4.61 pp [2.60, 6.62] / +12.97 pp [7.29, 18.65] |
| Two reversals, fixed length 3 | 69.12% / 8.47% | **73.81% / 15.12%** | +4.69 pp [2.78, 6.60] / +6.66 pp [3.71, 9.60] |
| Two reversals, fixed length 4 | 55.12% / 0.48% | **57.53% / 0.82%** | +2.41 pp [1.15, 3.68] / +0.34 pp [0.17, 0.52] |

The fork reached 90% copy-token accuracy in 15/20 seeds, versus 7/20 for textbook tapped and
3/20 for pre-tanh ReLU. Its advantage was largest on the two-block reversal task. All models
generalized poorly beyond trained lengths, and fixed-length-four exact recall was near zero, so
the experiment supports a trainability/topology effect rather than robust algorithmic
generalization.

The natural architectures do not isolate topology perfectly. The fork reads signed, relatively
unbounded `z_t`; textbook tapped reads bounded `tanh(u_t)`; and pre-tanh tapped reads sparse
nonnegative `ReLU(u_t)`. Consequently, the result establishes that the complete forked design
trained better on these tasks, not that branching alone has a precisely measured causal effect.
A strict factorial follow-up would hold the readout activation fixed while varying only whether
it receives `z_t` or `u_t`.

## Recurrent activation after locking the fork

Once the fork is selected, tanh is no longer on the immediate emission path. Its remaining role
is solely to form the recurrent carrier:

```text
z_t = M([x_t, h_(t-1)])
y_t = O(z_t)
u_t = T(z_t)
h_t = activation(u_t)
```

A separate 20-seed panel held the forked readout fixed and compared `tanh`, ReLU, identity, and
softsign recurrence. With global gradient clipping at 1.0 and identity-centered `T`, identity
looked decisively best on the short training distributions:

| Evaluation | tanh token / exact | identity token / exact |
|---|---:|---:|
| Copy, lengths 2-4 | 89.77% / 73.50% | **95.23% / 87.27%** |
| Two reversals, lengths 2-3 | 80.54% / 36.95% | **97.27% / 87.79%** |
| Two reversals, fixed length 3 | 73.81% / 15.12% | **95.00% / 74.21%** |
| Two reversals, fixed length 4 | 57.53% / 0.82% | **63.85% / 2.03%** |

That result depended strongly on clipping. Identity bound the 1.0 global gradient clip on about
90% of copy steps and 99.7% of palindrome steps, and produced recurrent-state maxima around
37-50 even in distribution, versus exactly 1.0 under tanh. A 20-seed robustness panel therefore
crossed tanh versus identity with clipping enabled/disabled and identity-centered versus
orthogonal recurrent initialization:

| `T` initialization | Gradient clipping | Identity minus tanh, copy | Identity minus tanh, palindrome |
|---|---|---:|---:|
| Identity | norm 1.0 | +5.46 pp [3.87, 7.05] | +16.74 pp [14.82, 18.66] |
| Identity | disabled | -17.55 pp [-22.59, -12.50] | +4.19 pp [0.47, 7.91] |
| Orthogonal | norm 1.0 | +5.33 pp [4.35, 6.32] | +2.05 pp [1.26, 2.85] |
| Orthogonal | disabled | -0.02 pp [-2.94, 2.90] | +3.50 pp [-0.44, 7.44] |

No condition became nonfinite, but identity produced extreme finite carriers. On stress copy
lengths 8-10, mean per-run maximum state/logit magnitudes reached roughly 441/246 to 664/668,
while tanh state remained bounded at 1.0 and maximum logits stayed around 15-19. Identity lost
every stress token-accuracy comparison: by 3.50-14.97 points on copy lengths 8-10 and by
6.64-9.77 points on palindrome block length 6.

The defensible interpretation is that clipping can substitute for tanh on short trained
horizons and allow a near-linear accumulator to optimize very quickly. It does not make the
unbounded carrier generally stable or length-robust. **Retain tanh for the recurrent carrier,
but keep it off the forked immediate-output path.** This preserves its intended magnitude-control
role without compressing ephemeral fast-weight effects before emission.

## Output activation after locking the fork

Separate scratch panels held recurrent state fixed at `tanh(i2h(z_t))` and varied only whether
the output head read `z_t` or `tanh(z_t)`.

Under full BPTT, 20 paired seeds favored direct output on delayed copy: +2.49 token-accuracy
points [0.96, 4.02] and +5.10 exact-recall points [1.60, 8.59]. Longer-copy effects were +2.24
[0.06, 4.42] and +1.07 [0.24, 1.91]. Palindrome point estimates also favored direct output but
their intervals crossed zero. All runs remained finite.

The recurrence-clipped fast-weight DFA harness gave the stronger mechanistic result. In an
aggressive fast-learning panel, direct output beat output tanh by +16.13 token points
[13.42, 18.84] on repeated copy and +6.12 [3.69, 8.55] on two-palindrome. On copy, output tanh
reduced feature RMS from 11.99 to .281 and the slow-only `i2o` update norm from 196.38 to 7.05.
Bounded short checks kept features almost entirely in tanh's linear regime and found no
meaningful difference, but did not learn enough to rank the arms.

Thus output tanh is approximately harmless when it behaves approximately as identity, and can
be strongly harmful when fast weights use activation magnitude. It offers no recurrent-stability
benefit in the fork because recurrent state has its own tanh. The selected graph is therefore:

```text
output_t = i2o(z_t)
hidden_t = tanh(i2h(z_t))
```

The combined scratch summary is `scratch/output_activation_report.md`; the standalone DFA
implementation and full results are under `scratch/dfa_forked_output_activation/`.

## Decision argument

The literature alone favors tapped state as the recognizable canonical baseline. The project's
scientific framing and empirical evidence favor the fork:

- It preserves a direct deep-trunk-to-output path for ephemeral fast-weight effects.
- It allows persistent state and immediate emission to specialize.
- It avoids making the recurrence-clipped model execute a recurrence-specific tanh bottleneck.
- It was more reliable in the project's matched zero-initialization runs.
- It also outperformed tapped BPTT RNNs across 20 paired seeds, especially on two-memory reversal.

The forked topology is selected. The defensible claim is not "forked is the standard vanilla
RNN." The claim is:

> We use a minimal transition/emission RNN as the activation-memory comparator. It retains a
> familiar recurrent state trained with direct feedback while allowing the shared deep
> representation to support current emission independently. A canonical tapped Elman model is
> reported as a sensitivity baseline.

Tanh placement should remain an independent ablation. Bounding the recurrent carrier does not
require forcing the immediate output through tanh.

## Reproduction archive

The standalone implementation and complete artifacts are under
`scratch/tapped_vs_forked_rnn/`:

- `experiment.py`: fresh model, task generation, training, evaluation, and aggregation code.
- `expanded_results.json`: complete 120-run configuration, curves, seed-level results, paired
  differences, confidence intervals, and fixed-length panels.
- `expanded_summary.csv`: compact machine-readable summaries.
- `README.md`: commands, original pilot, expanded panel, and caveats.
- `expanded_delayed_copy.png` and `expanded_two_palindrome.png`: learning-curve plots.
- `activation_ablation.py` and `activation_ablation_results.json`: four-way recurrent-activation
  comparison with fixed forked readout.
- `activation_robustness.py` and `activation_robustness_results.json`: clipping-by-initialization
  robustness panel and long-unroll stress evaluations.

Expanded command:

```bash
/home/jaden/miniforge3/bin/python experiment.py \
  --seeds 20 --workers 12 --output expanded_results.json
```

## References

- Elman, *Finding Structure in Time* (1990):
  <https://onlinelibrary.wiley.com/doi/10.1207/s15516709cog1402_1>
- Goodfellow, Bengio, and Courville, *Deep Learning*, Chapter 10 (2016):
  <https://www.deeplearningbook.org/contents/rnn.html>
- Pascanu et al., *How to Construct Deep Recurrent Neural Networks* (2014):
  <https://arxiv.org/abs/1312.6026>
- Hochreiter and Schmidhuber, *Long Short-Term Memory* (1997):
  <https://doi.org/10.1162/neco.1997.9.8.1735>
- Cho et al., *Learning Phrase Representations using RNN Encoder-Decoder* (2014):
  <https://doi.org/10.3115/v1/D14-1179>
- Gu, Goel, and Re, *Efficiently Modeling Long Sequences with Structured State Spaces* (2021):
  <https://arxiv.org/abs/2111.00396>
- Nøkland, *Direct Feedback Alignment Provides Learning in Deep Neural Networks* (2016):
  <https://proceedings.neurips.cc/paper/2016/hash/d490d7b4576290fa60eb31b5fc917ad1-Abstract.html>
- Werbos, *Backpropagation Through Time: What It Does and How to Do It* (1990):
  <https://doi.org/10.1109/5.58337>
