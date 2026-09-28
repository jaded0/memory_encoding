# Stabilizer pilot for the late divergence (2026-09-27)

## Question

On 3-char reversal at lr·α = 10, every run of `memory_tasks_head.md` learned, peaked and then
collapsed: the loss exploded and recall fell. Which single change keeps the learning and removes
the collapse?

## Design

Launcher: `sweeps/orc_stabilizer_pilot.sbatch`, ORC array 13901513, code 8fdd980,
`--fused_update`. The control is ephemeral + DFA with recurrence off, lr `1e-3`, plasticity
`1e4`, forget rate `0.01`, 20% ephemeral weights, weight clamp 1, three 1024-wide layers,
batch 16. Each arm changes one setting. The runs were 400k iterations on seeds 3141 and 2718.
train.py stops a run whose loss stays above 5 for 10 intervals; that counts as a collapse.

## Results

Recall on the 5,000-iteration interval (chance 14%):

| Arm | Outcome (s3141, s2718) | Best recall | Recall at 400k | Lag-5 at 400k | Loss at 400k |
|---|---|---:|---:|---:|---:|
| control | collapsed at 215k, 330k | 67%, 74% | | | |
| `--slow_weight_decay 1e-5` | stable | 36%, 36% | 32%, 31% | 15%, 14% | 1.39, 1.40 |
| `--slow_weight_decay 1e-6` | stable | 63%, 68% | 51%, 61% | 26%, 29% | 1.16, 1.13 |
| `--weight_clamp 0.3` | collapsed at 345k, 335k | **81%, 72%** | | | |
| `--weight_clamp 0.1` | collapsed at 265k, 230k | 64%, 58% | | | |
| `--output_tanh` | **stable** | 69%, 71% | **68%, 66%** | 16%, 25% | **0.93, 0.98** |
| `--ephemeral_update_clamp 100` | collapsed at 280k, 225k | 55%, 53% | | | |

Fast weights in trunk layers 1 and 2, traced through one sequence of 3-char halves from each
run's last checkpoint. Layer 0 is left out because, with recurrence off, 1,024 of its 1,033
inputs are the zeroed hidden state and those fast entries are never written.

| Run (checkpoint) | At the clamp, steps 0 / 2 / 5 | Median \|fast w\| step 5 | Cosine with the step-0 write at step 5 | Max \|logit\| step 5 |
|---|---|---:|---:|---:|
| control s2718 (325k, collapsing) | 0.1% / 1.1% / 4.4% | 0.098 | 0.20 | 5,487 |
| clamp 0.3 s3141 (325k, collapsing) | 7.7% / 15.6% / 18.7% | 0.115 | 0.20 | 3,212 |
| clamp 0.1 s3141 (250k, collapsed) | 18.0% / 26.4% / 24.2% | 0.057 | 0.12 | 106 |
| output tanh s3141 (400k) | 4.5% / 3.8% / 10.3% | 0.108 | 0.35 | 5 |
| slow decay 1e-6 s2718 (400k) | 0.0% / 0.0% / 0.1% | 0.029 | 0.15 | 2,382 |

## Interpretation

- **Output tanh is the only arm both stable and at the control's recall.** Recall is 66–68% at
  400k with the lowest loss, and logits stay below 6 through a sequence. The collapse was on the
  output side: unbounded trunk features gave unbounded logits.
- **Slow-weight decay stabilizes, at a cost.** At 1e-6 the runs are stable but end at 51–61%;
  at 1e-5 the slow weights cannot learn and recall stays near 31%.
- **Tighter weight clamps and the fast-update clamp do not prevent the collapse.** A clamp of
  0.3 reached the highest peaks (72–81%) with 8–19% of fast entries saturated, then collapsed
  anyway. A clamp of 0.1 saturated 18–26% of fast entries, peaked lower and collapsed earlier.
  At 0.1 the clamp also binds much of the output head's slow weights (mean |w| 0.2–0.35), so its
  lower recall cannot be pinned on fast-weight saturation alone.
- **Saturation and memory capacity.** Pinning entries at ±clamp erases their magnitudes and
  turns additive superposition into last-write-wins, so heavy saturation should cost capacity.
  The 2026-09-22 audit found that a clamp of 0.1 or less removes the memory, consistent with
  that. In this pilot, moderate saturation (clamp 0.3, output tanh at clamp 1) did not lower
  recall. The one arm with about 25% saturated did peak lower, but the output-head confound
  applies. A direct test is output tanh with the clamp off against output tanh at clamp 1,
  compared on lag-3 and lag-5 recall.
