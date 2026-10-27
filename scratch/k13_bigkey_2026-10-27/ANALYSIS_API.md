# Analysis/launcher contract

`analyze.py CHECKPOINT --datasets mqar2_32_q4 --batches 8 --raw-records --out FILE`
uses stored validation data, or `--generated` generates hash-filtered validation
cores. All launches remain separately gated; the analyzer never approves training.

The preregistered geometry field is
`results[i].layers.L0.items.key_pr_pooled` (likewise L1). `items.key_pr` is an
explicit alias for **pooled**, not median per-episode PR. The accumulator sums
unit-key `xx^T` over nonblank table store steps **once per episode**, not once per
query. `key_pr_per_episode_median` and `key_pr_per_episode_distribution` retain
the episode diagnostic. `pooled_stores` and `pooled_episodes` audit the population.
For the complete 100-address L0 key set with no PE, pooled PR is 200/11≈18.18,
rank19; individual K4 episode PR is at most4, but pooled PR is not bounded by4.

Gate checks both all-target strict and first-answer strict in [.20,.85], first-
answer post-store suppression improvement >=.10 and >=80% of error recovered,
pooled PR L0>=4 and L1>=2, and at least128 evaluated episodes. B16×8 batches
meets the preregistered 128-episode population. A one-batch smoke is insufficient
to establish the gate, even if its other metrics happen to pass.

`acc` reports strict/observed/no_fast all-target, `_first`, and `_exact` (all
answers correct in an episode) scores. `post_store`, `store_only`, `paired_gap`
are first-answer-only scores. The paired gap keeps only the first queried table
slot and the unchanged query suffix; it is not the all-Q-retained standalone gap
task. `--raw-records` includes geometry and kernel evidence; summaries always
include signed and absolute cosine distributions. Kernel swaps never affect state.
