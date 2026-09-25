# DFA throughput benchmark

`benchmark_dfa_throughput.py` compares three single-thread CPU paths on identical pre-generated
batches and matched 6-symbol, 64-hidden, two-layer, batch-16 models:

- `main_native`: production `train_batch`, including loss/autograd and metric bookkeeping.
- `main_core`: production model, DFA update, forgetting, and wipe without trainer bookkeeping.
- `scratch_core`: the standalone scratch model with diagnostics removed from its update loop.

Run both sequence lengths with:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python benchmarks/benchmark_dfa_throughput.py --task repeated_copy
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python benchmarks/benchmark_dfa_throughput.py --task palindrome2
```

The committed runs used 50 warmup batches, 500 timed batches, and seven fresh model repetitions.
Rates are medians; parenthesized values are IQRs.

| Workload | Path | Before optimization, seq/s | After optimization, seq/s |
|---|---|---:|---:|
| Repeated copy, 8 steps | Main native | 926 (28) | **1,108 (11)** |
| Repeated copy, 8 steps | Main core | 1,112 (12) | **1,324 (63)** |
| Repeated copy, 8 steps | Scratch core | 1,593 (57) | unchanged |
| Palindrome, 4 steps | Main native | 1,767 (25) | **2,028 (86)** |
| Palindrome, 4 steps | Main core | 2,117 (36) | **2,431 (114)** |
| Palindrome, 4 steps | Scratch core | 2,843 (38) | unchanged |

The trace-preserving optimization improved native throughput by 19.6% on repeated copy and 14.8%
on palindrome. Main core improved by 19.1% and 14.8%. The remaining main-core gap to scratch is
16.9% and 14.5%; native trainer bookkeeping widens it to 30.4% and 28.7%.

## Profile and changes

A 200-batch `cProfile` run identified full-tensor clones in `populate_dfa_gradients` and
`apply_update`, out-of-place forgetting, materialized sequence-wipe repeats, and two recursive
`Module.zero_grad` traversals per token as avoidable costs. The production path now:

- assigns the newly allocated DFA outer product directly to `.grad`;
- negates that gradient without cloning it first;
- forgets in place;
- broadcasts the aggregate and mask during sequence wipe instead of repeating them; and
- clears only the per-sample gradients that DFA manually populated.

Regenerated characterization output was byte-identical to the committed golden fixture, and the
full suite passed. Two further candidates were rejected because they changed floating-point
traces: in-place `add_(..., alpha=learning_rate)` and replacing cross-entropy autograd with the
analytic `softmax(output) - target` error.

Raw measurements are in `dfa_throughput_*.json`. The unoptimized files contain all three paths;
the optimized files rerun the two changed main paths against the unchanged scratch baselines.
