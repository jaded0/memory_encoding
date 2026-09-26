# Benchmarks

- `3pal_architecture_initialization.md`: seven-seed historical topology-by-slow-initialization
  factorial on the three-palindrome task.
- `3pal_head_panel.md`: seven-seed panel at current code (5e088af): ephemeral + DFA against
  SimpleRNN under DFA and BPTT on the three-palindrome task.

## DFA throughput benchmark

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

## GPU throughput at the 3-palindrome benchmark size (2026-09-25)

`sweeps/orc_speed_benchmark.sbatch` runs the same script with `--device cuda --task palindrome3
--symbols 7 --hidden-size 1024 --num-layers 3 --batch-size 16` (lr `1e-3`, plasticity `1e3`,
weight clamp 1; 20 warmup and 200 timed batches, three repeats), then `train.py` end to end for
each panel arm. `fused_core` is a speed ceiling, not a production path: `main_core` with each
layer's outer product, plasticity, update, weight clamp and forget compiled into one kernel by
`torch.compile`. After 50 batches its weights differ from `main_core`'s by at most 4.8e-7
(relative). Triton, which it needs, does not support the P100. The bandwidth floor is the
rate if each step only read every per-sequence weight tensor for the forward pass and read and
wrote it once for the update. It uses the measured copy bandwidth.

Batches (iterations) per second, medians:

| GPU | Main native | Main core | Scratch core | Fused core | Bandwidth floor | `train.py`: ephemeral DFA / rnn DFA / rnn BPTT |
|---|---:|---:|---:|---:|---:|---|
| P100 (m9g) | 14.9 | 16.6 | 18.5 | n/a | 98 | 14.6 / 39.5 / 43.4 |
| RTX A6000 (Deckard) | 21.1 | 22.7 | 25.4 | 69.6 | 125 | |
| A100 (cs) | 39.9 | 54.8 | 61.3 | 87.8 | 318 | 36.7 / 63.8 / 59.7 |
| H100 (cs2) | 56.8 | 87.5 | 97.4 | 117.8 | 548 | 52.7 / 86.2 / 74.7 |

Deterministic mode costs nothing measurable. On the P100 and A6000 the main path is GPU-bound.
About 90% of GPU time goes to separate elementwise passes (mul, add, neg, clamp, in-place mul)
over the `[B, out, in]` per-sample weights, and the forward `bmm` is about 5%. Fusing them
gives 3.1× on the A6000. On the A100 and H100 `train_batch` is CPU/launch-bound: its
bookkeeping costs 27–35% relative to `main_core` (a `.item()` host sync every step is one
source), and even `fused_core` stays far from the bandwidth floor. Raw results are on ORC under
`~/memory_encoding_speed/2026-09-25_head-panel-speed/speed_results`.
