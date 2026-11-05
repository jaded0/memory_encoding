# K14: sequence outer loop (opt-in)

`--sequence_outer true --updater backprop --model_type ephemeral
--backprop_step_parity true --max_seq_len 10` selects a **new sequence-level
outer SGD updater**, not the existing per-character backprop updater. Add
`--fast_write_grad through` or `--fast_write_grad detached` for the matched arms.
Without `--sequence_outer`, the default `detached` leaves all legacy paths
unchanged. `through` without the new updater is an error.

At sequence start, slow copies are consolidated with the existing batch mean
and fast entries are wiped. Slow entries and shared biases remain fixed during
the rollout. Each character predicts before its target supplies the DFA write:

```
e_t = softmax(logits_t) * target_t.sum() - target_t
F[t+1] = keep * clamp_weight(F[t] + lr * mask * clamp_write(-alpha * (e_t B) outer key_t))
```

The output head has no fast mask. Feedback matrices are fixed. Fast values are
functional tensors, not mutated leaves; input keys and the softmax error retain
their graph in `through`. In `detached`, only the write increment is detached.
Both arms keep ordinary BPTT through recurrent hidden state and query/read paths.
They have bit-identical predictions, losses, and fast writes before their first
outer update; they differ only in the outer gradient. No `create_graph` or
differentiated gradient-computation routine is used.

The outer objective is **time mean of batch sum**. Each slow copy receives its
own sequence gradient, without `1/B`; shared bias gradients are divided by batch
size. SGD updates slow entries/biases once after the sequence; no outer update
trains fast entries and alpha is applied exactly once in each DFA write. Slow
copies are averaged at the next sequence start, as in the existing model.
Training-instance counts still count prediction steps (10 characters = 9 steps).

The implementation supports exclusive masks/`fast_layers`, recurrence on/off,
last-one/last-two inputs, PE, layer norm, residual connection, output tanh,
write clamp, weight clamp, and ordinary alpha schedules. It explicitly refuses
fused updates, clipping, slow decay, fast weight clamp, multi-pass/subsampled
writes, slow update windows, carried fast state, mid-sequence wipes, normalized
reads/writes, alternate feedback/write rules, order-two state, write gains,
label smoothing, and loop tracing. Restrictions are checked at CLI and direct
training entry points. This narrow unfused implementation is intended for the
short K14 pilot, not as a replacement for all online updater combinations.

`max_seq_len=10` is an upper bound, **not an exact-length data generator**.
Rollouts accept 2–10 characters including padding, which produces zero-error
writes on padded targets. Generate exact ten-character episodes separately if
that is required by the experimental design. Training does not silently pad,
truncate, or regenerate data itself.

Held-out evaluation (`--heldout_eval_every`, or `heldout.py --checkpoint ...`)
supports the opt-in outer model with the existing frozen-slow eager DFA
fast-only evaluator, including observed/strict/no-fast/free-running protocols.
It does not backpropagate or update slow entries/biases; both gradient modes
evaluate identically for identical weights. Evaluation state is restored before
training resumes. Ordinary backprop checkpoints still cannot use that evaluator.

Both new flags are recorded in checkpoints and experiment config. Resume refuses
changing either flag; old checkpoints imply `sequence_outer=false` and
`fast_write_grad=detached`. Outer-vs-online comparisons are **not matched** and
must not be represented as the detached-vs-through K14 comparison.

CPU tests: `CUDA_VISIBLE_DEVICES= ~/miniforge3/envs/hebby/bin/python -m pytest tests/test_sequence_outer.py -q`.
They include full float64 gradcheck, finite differences with/without recurrence,
independent detached-gradient reference, C2 scaling/bias checks, eager fast-only
equivalence with clamps/padding, CLI/refusal checks, legacy bit identity, and
checkpoint/heldout/resume integration. No GPU or production launch is required.
