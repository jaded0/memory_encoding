# Output tanh ablation

## Decision supported by the scratch panels

Keep tanh on the forked recurrent-state branch and leave the immediate output branch unbounded:

```text
z_t = M([x_t, h_(t-1)])
output_t = i2o(z_t)
hidden_t = tanh(i2h(z_t))
```

Output tanh has no recurrent-stability role in this topology. It only compresses the deep
representation before emission and bounds the trace used to train `i2o`.

## Full-BPTT result

The standalone forked RNN held recurrence fixed at `tanh(i2h(z))` and compared `i2o(z)` with
`i2o(tanh(z))` over 20 paired seeds.

| Panel | Identity minus tanh token accuracy | Exact recall |
|---|---:|---:|
| Copy, trained lengths | **+2.49 pp** [0.96, 4.02] | **+5.10 pp** [1.60, 8.59] |
| Copy, longer lengths | **+2.24 pp** [0.06, 4.42] | **+1.07 pp** [0.24, 1.91] |
| Two-palindrome, trained lengths | +0.99 pp [-1.14, 3.13] | +3.67 pp [-1.90, 9.24] |

All 120 runs were finite. Tanh reduced feature/logit magnitude and clipping frequency, but only
about 7% of its output features were saturated. Identity therefore has a modest copy advantage
under ordinary BPTT and no demonstrated palindrome cost.

Artifacts: `tapped_vs_forked_rnn/output_activation_bptt_*`.

## Recurrence-clipped DFA fast-weight result

A standalone project-style harness used forked topology, direct DFA, zero-initialized fast
entries, default-initialized slow entries, online per-token updates, forgetting, sequence wipes,
and recurrence forced to zero. Twelve paired seeds gave:

| Fast task | Identity minus tanh token accuracy | Exact recall | Learning AUC |
|---|---:|---:|---:|
| Repeated copy | **+16.13 pp** [13.42, 18.84] | **+1.56 pp** [0.72, 2.41] | **+.1270** [.1030, .1511] |
| Two-palindrome | **+6.12 pp** [3.69, 8.55] | **+2.99 pp** [0.96, 5.03] | **+.0354** [.0153, .0555] |

Identity won 12/12 copy seeds and 11/12 palindrome seeds, with one tie. On copy, tanh reduced
feature RMS from 11.99 to .281 and the mean `i2o` update norm from 196.38 to 7.05. Thus output
tanh can directly suppress both the expression of fast-weight features and learning in the
slow-only readout head.

The strongest DFA result intentionally used an aggressive unclamped fast-learning regime.
Bounded short checks with `weight_clamp=1` kept features mostly in tanh's linear regime and found
no meaningful difference, but also failed to learn enough to rank the activations. This supports
the conditional mechanism: output tanh is approximately harmless when it is approximately
identity, and harmful when fast weights make substantial use of activation magnitude.

Artifacts and detailed caveats: `dfa_forked_output_activation/README.md`.

## Conclusion

No successful panel showed a benefit from output tanh. Direct output was modestly better under
BPTT and substantially better when the recurrence-clipped DFA fast-weight mechanism generated
large useful features. Since the separate recurrent branch retains tanh, removing output tanh
does not sacrifice recurrent magnitude control.
