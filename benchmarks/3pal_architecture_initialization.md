# Three-palindrome architecture and initialization factorial

## Question

Two historical changes were initially confounded: commit `59db471` replaced the sibling-head
layout at `aef0697` with a serial Elman layout, and commit `8e9b45f` initialized slow weights from
`nn.Linear` instead of zero. This benchmark asks whether the observed behavior came from topology,
slow-weight initialization, or their interaction.

This is a historical mechanics experiment, not a direct comparison with current HEAD. In
particular, the sibling-head arm at `aef0697` did not apply direct DFA updates to `i2h`; current
HEAD restores sibling transition/emission heads while explicitly training `i2h` through fixed
direct feedback.

## Design

The four arms form a 2x2 factorial:

| Arm | Topology | Slow-weight initialization | Revision |
|---|---|---|---|
| `legacy_zero` | Sibling state/output heads | Zero | `aef0697` |
| `legacy_default` | Sibling state/output heads | `nn.Linear` default | `aef0697` plus the isolated initialization patch from `8e9b45f` |
| `corrected_zero` | Serial Elman | Zero | `59db471` |
| `corrected_default` | Serial Elman | `nn.Linear` default | `8e9b45f` |

Every run used ephemeral DFA, no recurrence, `last_one` input, the
`3_palindrome_dataset_vary_length` dataset, batch 16, three 1024-wide hidden layers, learning rate
`1e-3`, plasticity `1e3`, forget rate `0.01`, 20% ephemeral weights, update clamp off, unit
normalization off, weight clamp 1, and 250,000 iterations. Seeds were paired across arms:
`2718`, `3141`, `4241`, `5153`, `8677`, `10007`, and `65537`. Runs were deterministic and executed
on ORC GPUs. Metrics below are from each run's final 5,000-training-iteration interval, not a held-out
evaluation set.

The original seed-1729 arms were split between Deckard and ORC. Three same-cluster reruns were
started to make an eight-seed panel, then cancelled when the seven fully ORC-paired seeds were
judged sufficient. They are excluded from every result below.

Launchers are `sweeps/orc_3pal_extra_seeds.sbatch`,
`sweeps/orc_3pal_legacy_default.sbatch`, and `sweeps/orc_3pal_missing_seeds.sbatch`. Successful
SLURM arrays were `13880257`, `13885677`, and `13888852`; every included task completed with exit
code zero.

## Results

| Arm | Loss | Token accuracy | Recall accuracy | Exact recall | Lag-1 recall | Lag-3 recall | Lag-5 recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| `legacy_zero` | **1.265** | **61.29%** | **48.12%** | **30.22%** | **78.16%** | **24.20%** | 5.89% |
| `legacy_default` | 1.298 | 60.69% | 45.01% | 25.79% | 73.58% | 22.09% | 5.20% |
| `corrected_zero` | 1.513 | 48.93% | 13.78% | 10.69% | 24.89% | 2.94% | 2.14% |
| `corrected_default` | 1.370 | 56.81% | 41.24% | 20.95% | 65.22% | 22.28% | **7.29%** |

Paired effects below are percentage-point differences. Confidence intervals are two-sided 95%
Student-t intervals over the seven paired seed differences (`df=6`). With only seven seeds and no
held-out evaluation, they describe this panel rather than establishing general task performance.

| Contrast | Recall accuracy | Exact recall | Lag-1 recall | Lag-3 recall | Lag-5 recall |
|---|---:|---:|---:|---:|---:|
| Elman minus sibling, zero init | -34.34 [-44.40, -24.28] | -19.53 [-23.56, -15.50] | -53.28 [-70.16, -36.39] | -21.26 [-27.26, -15.25] | -3.75 [-7.25, -0.25] |
| Elman minus sibling, default init | -3.77 [-16.81, 9.28] | -4.84 [-18.32, 8.64] | -8.36 [-21.14, 4.41] | 0.19 [-19.11, 19.48] | 2.09 [-2.79, 6.98] |
| Default minus zero init, sibling | -3.11 [-7.38, 1.17] | -4.43 [-9.57, 0.72] | -4.58 [-8.79, -0.37] | -2.11 [-9.34, 5.12] | -0.69 [-1.85, 0.46] |
| Default minus zero init, Elman | +27.47 [18.66, 36.28] | +10.27 [-2.49, 23.02] | +40.34 [27.37, 53.30] | +19.34 [3.24, 35.43] | +5.15 [0.38, 9.92] |
| Initialization-by-topology interaction | +30.57 [19.28, 41.87] | +14.69 [-0.30, 29.69] | +44.91 [30.86, 58.97] | +21.44 [0.53, 42.36] | +5.84 [1.52, 10.17] |

Seed-level recall accuracy shows both the consistency of the zero-init topology result and the
variance of the default-initialized Elman arm:

| Seed | `legacy_zero` | `legacy_default` | `corrected_zero` | `corrected_default` |
|---:|---:|---:|---:|---:|
| 2718 | 47.13% | 42.25% | 5.26% | 23.46% |
| 3141 | 47.30% | 41.77% | 12.37% | 55.22% |
| 4241 | 40.95% | 46.82% | 5.84% | 23.21% |
| 5153 | 43.53% | 42.48% | 11.19% | 41.05% |
| 8677 | 49.37% | 46.66% | 7.95% | 41.47% |
| 10007 | 51.37% | 42.66% | 8.19% | 39.24% |
| 65537 | 57.17% | 52.42% | 45.64% | 65.06% |

## Interpretation

The effects are not additive. Standard slow-weight initialization is strongly beneficial in the
serial Elman topology, while this panel provides no clear evidence that it improves the historical
sibling-head topology. It recovers most of the large zero-initialized Elman deficit: the mean
recall gap changes from -34.34 points with zero initialization to -3.77 points with default
initialization. The residual default-init topology contrast is uncertain and varies in sign across
seeds.

Therefore, the initialization fix is justified as removing a severe topology-dependent failure
mode, not as a universal performance improvement. This panel also does not justify selecting the
historical sibling implementation over current HEAD: current HEAD has materially different DFA
credit to the state head, trunk architecture, and output activation. Any direct current-topology
claim requires a matched benchmark of the current implementations.
