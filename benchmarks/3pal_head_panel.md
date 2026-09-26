# Three-palindrome panel at current code (5e088af)

## Question

How does the current implementation (forked transition/emission heads, direct DFA to `i2h`, no
output tanh, default slow-weight initialization) do on the 3-palindrome task? How does it compare
with the SimpleRNN baseline under the same learning rule (DFA) and under BPTT, now that SimpleRNN
matches the ephemeral model's trunk?

## Design

Launcher: `sweeps/orc_3pal_head_panel.sbatch`, ORC array 13896047. All 21 tasks completed.

| Arm | Model | Updater | Recurrence |
|---|---|---|---|
| `ephemeral_dfa` | EphemeralRNN | DFA | off (memory only in fast weights) |
| `rnn_dfa` | SimpleRNN | DFA | on |
| `rnn_bptt` | SimpleRNN | BPTT | on |

Shared settings are those of `3pal_architecture_initialization.md`:
- task: `3_palindrome_dataset_vary_length`, `last_one` input
- model: three 1024-wide layers, batch 16
- update: learning rate `1e-3`, plasticity `1e3` (so lr·α = 1), forget rate `0.01`, 20%
  ephemeral weights
- clipping: weight clamp 1, update clamp and grad-norm clip off
- 250,000 iterations, deterministic
- seeds `2718`, `3141`, `4241`, `5153`, `8677`, `10007`, `65537`

Runs executed on `-p cs,cs2 --qos cs` (A100 and H100). Metrics are from the final
5,000-iteration training interval. Recall chance is 1/7 = 14.3%.

## Results

Mean ± SD over seven seeds:

| Arm | Loss | Recall | Exact recall | Lag 1 | Lag 3 | Lag 5 | Final char |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ephemeral_dfa` | 1.299 ± .037 | 45.0% ± 3.9 | 25.8% ± 4.9 | 73.6% ± 4.4 | 22.1% ± 5.2 | 5.2% ± 1.2 | 68.4% |
| `rnn_dfa` | 1.130 ± .085 | 26.6% ± 13.5 | 12.7% ± 8.3 | 22.6% ± 7.5 | 28.7% ± 17.6 | 34.1% ± 24.5 | 78.0% |
| `rnn_bptt` | 0.510 ± .001 | 100% | 100% | 100% | 100% | 100% | 100% |

Seed-level recall:

| Seed | `ephemeral_dfa` | `rnn_dfa` | `rnn_bptt` |
|---:|---:|---:|---:|
| 2718 | 42.2% | 37.6% | 100% |
| 3141 | 41.8% | 16.6% | 100% |
| 4241 | 46.8% | 41.9% | 100% |
| 5153 | 42.5% | 16.1% | 100% |
| 8677 | 46.7% | 43.1% | 100% |
| 10007 | 42.7% | 15.2% | 100% |
| 65537 | 52.4% | 15.4% | 100% |

Mean recall over training:

| Arm | 25k | 50k | 100k | 150k | 200k | 250k |
|---|---:|---:|---:|---:|---:|---:|
| `ephemeral_dfa` | 33.9% | 36.3% | 30.5% | 38.1% | 40.7% | 45.0% |
| `rnn_dfa` | 15.0% | 14.1% | 14.7% | 17.8% | 23.5% | 26.5% |
| `rnn_bptt` | 2.7% | 19.8% | 96.9% | 100% | 100% | 100% |

## Interpretation

- **The ephemeral model uses its fast weights at short lags only.** Lag-1 recall is 73.6%
  against a 14.3% chance. Lag 5, the first character of a three-character half recalled at the
  sequence's end, is 5.2%, below chance, because the model mostly predicts end-of-sequence
  padding there. Final-character accuracy, 68.4%, is barely above the 67.0% of always
  predicting padding (two-thirds of halves are shorter than three). The fast-weight memory has
  not yet replaced the recurrent connection on this task at these settings.
- **These settings are below the regime that worked in 2025.** The 2025 DFA runs with
  recurrence off that reached 90–97% last-character accuracy on this task used lr·α = 10
  (lr `1e-3` with α `1e4`, or `1e-4` with `1e5`), weight clamp 1, and 1–3 million
  iterations, with a sudden jump late in training (W&B `jadens_team/hebby`, e.g.
  `spring-durian-8684`, `wobbly-haze-9197`). This panel runs at lr·α = 1 for 250k
  iterations. Recall is still rising at 250k.
- The HEAD ephemeral arm is 3.8 points above the serial-Elman default-init arm of the factorial
  (41.2%) and 3.1 points below its historical sibling zero-init arm (48.1%). These are
  different panels on the same seeds, not a paired comparison.
- SimpleRNN under DFA is bimodal: three seeds learn some long-lag recall through the direct-DFA
  surrogate on `i2h`, and four stay near chance. Under BPTT it solves the task on every seed by
  150k iterations.

The follow-up verification across tasks at lr·α = 10 is `sweeps/orc_memory_tasks.sbatch`.
