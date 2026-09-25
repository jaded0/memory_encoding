# Forked DFA fast-weight output activation

## Question

With topology fixed to a fork and recurrence clipped, should the immediate output head read the
deep fast-weight representation directly or through tanh?

```text
z_t = deep ephemeral trunk([x_t, 0])
discarded recurrent candidate = tanh(i2h(z_t))

identity output: logits_t = i2o(z_t)
tanh output:     logits_t = i2o(tanh(z_t))
```

Tanh is applied to output **features**, not logits. It provides no recurrence-stability benefit
here because the recurrent candidate has its own tanh and is discarded when recurrence is off.

## Harness

`experiment.py` is a standalone transcription of the project-style mechanism:

- Per-sequence fast and slow weights.
- Default `nn.Linear` initialization on slow entries and zero fast entries.
- Twenty percent ephemeral entries in hidden layers; output head is slow-only.
- Online DFA updates after every token, including presentation tokens.
- Fast-weight plasticity, forgetting after each update, and sequence-boundary wipe.
- Deep GELU trunk with forked `i2h` and `i2o` heads.
- Direct DFA projection to every hidden layer and `i2h`.
- Recurrence candidate computed but next hidden state forced to zero.
- DFA activation derivatives omitted, matching the current project rule.

Validation asserts paired initialization, zero/default fast/slow initialization, clipped
recurrence, active `i2h`, correct output traces, sequence wipe semantics, and that output-feature
activation is the only difference between paired arms.

The generated fast tasks are a four-symbol repeated-copy sequence and a two-symbol-half
palindrome. Loss and DFA updates occur at every prediction step; reported accuracy is restricted
to recall positions.

## Fast-learning panel

Command:

```bash
/home/jaden/miniforge3/bin/python experiment.py \
  --seeds 12 --workers 12 --output results.json
```

Configuration: two 64-wide trunk layers, batch 16, 600 training batches, learning rate `1e-4`,
plasticity `1e5`, forgetting `0.01`, and no weight clamp. Runtime was 480 seconds. All 72 runs
(identity, tanh, and ReLU) completed without nonfinite values.

Primary paired effects are identity minus tanh:

| Task | Token accuracy | Exact recall | Learning AUC |
|---|---:|---:|---:|
| Repeated copy | **+16.13 pp** [13.42, 18.84] | **+1.56 pp** [0.72, 2.41] | **+.1270** [.1030, .1511] |
| Two-palindrome | **+6.12 pp** [3.69, 8.55] | **+2.99 pp** [0.96, 5.03] | **+.0354** [.0153, .0555] |

Identity won token accuracy in 12/12 copy seeds and 11/12 palindrome seeds, with one tie.

Tanh saturation at `|feature| >= .99` was only 4.69% on copy and 0.76% on palindrome, but tanh
still compressed feature RMS strongly:

| Task | Identity feature RMS | Tanh feature RMS | Identity/tanh `i2o` update norm |
|---|---:|---:|---:|
| Repeated copy | 11.99 | 0.281 | 196.38 / 7.05 |
| Two-palindrome | 1.143 | 0.138 | 7.89 / 3.23 |

This is the direct readout-side effect: `i2o` learns from its input trace, so tanh bounds both the
effect of fast-weight features on logits and the activation factor in the slow output-head DFA
update. Upstream DFA does not include an activation derivative; its raw update is affected only
indirectly through the changed output error.

## Bounded checks

Two 12-seed identity-versus-tanh checks added `weight_clamp=1`:

1. Project update scale: learning rate `1e-3`, plasticity `1e3`.
2. Aggressive update scale: learning rate `1e-4`, plasticity `1e5`.

Commands and outputs are in `project_scale_results.json` and `bounded_fast_results.json`.
Neither short 600-batch panel learned the tasks well enough to rank output activations. Features
remained predominantly in tanh's linear regime:

- Project-scale copy feature RMS was `.0852` identity versus `.0842` tanh, with zero tanh
  saturation; recall was statistically indistinguishable.
- Aggressive bounded copy feature RMS was `.250` versus `.169`, with `.09%` tanh saturation;
  identity-minus-tanh recall was +0.28 pp with CI [-0.39, +0.94].
- Aggressive bounded palindrome identity-minus-tanh recall was -0.52 pp with CI
  [-1.30, +0.26].

These checks show that tanh behaves nearly like identity when the trunk remains small. They are
not successful-task evidence that tanh helps.

## Interpretation

The standalone full-BPTT panel in `../tapped_vs_forked_rnn/` independently found that direct
forked output improved copy accuracy by 2.49 points [0.96, 4.02] over `tanh(z)`, with no
significant palindrome difference.

Across BPTT and DFA there is no evidence that output tanh improves learned recall. Its effect is
small when features stay linear and substantially harmful when fast weights generate larger
features. Because recurrent stability is already handled by `tanh(i2h(z_t))`, the supported
design is:

```text
output_t = i2o(z_t)
hidden_t = tanh(i2h(z_t))
```

## Caveats

- This is a small CPU harness, not the full 1024-wide project trainer or three-palindrome
  dataset.
- The strongest DFA result uses no weight clamp and very large finite fast-weight features. It
  deliberately stress-tests the output bottleneck but is not the retained benchmark
  hyperparameter configuration.
- The bounded project-scale panels were too short to achieve useful recall.
- Exact recall is floor-limited, especially for four-symbol copy.
- `i2h` receives direct same-step DFA even though its clipped recurrent output cannot affect the
  task; this matches the intended future implementation but is not temporal credit assignment.

## Artifacts

- `experiment.py`: standalone implementation and validations.
- `results.json`: primary 12-seed fast-learning panel.
- `project_scale_results.json`: bounded project-update-scale check.
- `bounded_fast_results.json`: bounded aggressive-update check.
- `run.log`, `project_scale_run.log`, `bounded_fast_run.log`: console summaries and runtimes.
- `summary.csv`: summary from the most recent bounded run; full summaries remain embedded in
  each JSON artifact.
