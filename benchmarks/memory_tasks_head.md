# Do fast weights replace a clipped recurrent connection? (current code, 2026-09-25)

## Question

With recurrence clipped (`--enable_recurrence false`), does the ephemeral fast-weight memory
carry the information a recurrent connection would, across several memory tasks? The 3-char
panel (`3pal_head_panel.md`) ran at lr·α = 1 and found memory mostly at lag 1. The 2025 runs
that solved 3-char palindromes used lr·α = 10.

## Design

Launcher: `sweeps/orc_memory_tasks.sbatch`. ORC arrays 13897313 (main set) and 13897527
(`RUN_SET=low_alpha`). Code 7226291 (PR #2; DFA with clipping off is trace-identical to
5e088af). Settings are the panel's except plasticity: three 1024-wide layers, batch 16,
lr `1e-3`, forget rate `0.01`, 20% ephemeral weights, weight clamp 1, deterministic.

| Arm | Model | Recurrence | Fast weights | α | Iterations | Seeds |
|---|---|---|---|---|---:|---|
| ephemeral | EphemeralRNN + DFA | off | 20% | 1e4 (lr·α = 10) | 500k | 2718, 3141 (+4241 on 3-char) |
| ephemeral_a3e3 | EphemeralRNN + DFA | off | 20% | 3e3 (lr·α = 3) | 500k | 2718, 3141 (3- and 4-char only) |
| no_fast | EphemeralRNN + DFA | off | none (`--ephemeral_fraction 0`) | n/a | 250k | 2718 |
| rnn_bptt | SimpleRNN + BPTT | on | n/a | n/a | 250k | 2718 |

Tasks:
- Reversal (`n_palindrome_dataset_vary_length`, half length 1..n, 7 recallable symbols):
  2-, 3- and 4-char, with lags up to 3, 5 and 7.
- Binary 3- and 4-char reversal (`n_small_palindrome`, chance 50%).
- 4-period resequencing (`4_resequence`, lags 1–3, chance 11%). The first recall position is
  ambiguous about the period, so no model can reach 100%.

`long_range_memory_dataset` was left out: train.py preprocesses its 10M rows on the fly at every
start, for over 5 minutes.

## Results

Recall on the final 5,000-iteration interval, and the best interval, averaged over seeds. A run
that train.py stopped early (loss above 5 for 10 consecutive intervals) is reported at its last
interval. **Bold** marks the best memoryless-recurrence arm per task.

| Task | Chance | ephemeral α 1e4: final / best | ephemeral α 3e3: final / best | no_fast | rnn_bptt |
|---|---:|---|---|---:|---:|
| 2-char reversal | 14% | **70.1% / 70.1%** (2/2 completed, still rising at 500k) | | 8.3% | 100% |
| 3-char reversal | 14% | 46.4% / 69.8% (0/3 completed) | **62.9% / 80.1%** (2/2 completed, one collapsing) | 6.7% | 100% |
| 4-char reversal | 14% | 21.9% / 24.8% (0/2) | **33.0% / 50.7%** (1/2 completed, still rising at 500k) | 5.3% | 100% |
| Binary 3-char | 50% | **66.7% / 74.6%** (0/2 completed) | | 38.9% | 100% |
| Binary 4-char | 50% | 33.3% / 36.8% (0/2, diverged by 10k) | | 50.5% | 100% |
| 4-period resequencing | 11% | **53.2% / 54.2%** (2/2 completed, flat after 150k) | | 0.0% | 66.3% |

Per-run 3-char reversal curves (recall / loss):

| Run | 100k | 200k | 250k | 300k | 400k | 500k |
|---|---|---|---|---|---|---|
| α 1e4 s2718 | .50/1.7 | .75/1.2 | .75/1.4 | .62/5.2 | stopped at 345k (.47/23.4) | |
| α 1e4 s3141 | .61/2.3 | .48/20.4 | stopped at 205k | | | |
| α 1e4 s4241 | .54/1.1 | .70/1.1 | .68/2.6 | .60/6.7 | stopped at 355k (.45/31.2) | |
| α 3e3 s2718 | .41/1.2 | .68/1.0 | .76/1.0 | .79/1.0 | .67/2.2 | .45/10.9 |
| α 3e3 s3141 | .48/1.1 | .75/1.0 | .80/0.9 | .77/0.9 | .80/0.9 | .81/1.0 |

## Interpretation

- **The fast weights are the memory.** On every task where training stayed stable, the
  ephemeral arm recalls far above the same model without fast weights. The ablation sits at or
  below chance everywhere, because a memoryless model mostly predicts padding at recall
  positions. On 3-char reversal the best interval reaches 81% recall, against 45% at lr·α = 1
  in the panel, including the lag-5 recall and final character that the panel missed.
- **They do not yet replace the recurrent connection.** SimpleRNN + BPTT with recurrence solves
  every reversal task (100%) and reaches the ambiguity-limited ceiling on resequencing (66%),
  where the ephemeral arm stops at 53–54%.
- **Training is unstable late in runs.** Runs learn, peak, and then the loss grows and recall
  collapses. This happened in every α 1e4 run on tasks of 7 tokens or more, and in 2 of 4 α 3e3
  runs. In the logs of `3-char α 1e4 s2718`, the slow-weight norm grows monotonically for the
  whole run, and the average weight norm goes from 197 to 313 between 200k and 345k. Over the
  same stretch `i2o`'s slow update norm, which scales with the trunk features `i2o` reads,
  rises from 245 to 619 as the loss takes off. `--weight_clamp 1` bounds each entry, not a
  layer's gain. Likely cause, not yet tested: unbounded growth of the trunk's slow weights,
  together with the untanh'd output path (tanh on `i2o`'s input was removed on 2026-09-24),
  lets the logits grow until predictions are confidently wrong.
  - Candidate stabilizers to test on 3-char reversal: `--unit_norm_weights`, output tanh,
    slow-weight decay, a tighter weight clamp. The 2026-09-22 audit found that a clamp of
    0.1 or less removes the memory.
