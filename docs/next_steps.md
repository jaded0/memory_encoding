# Next steps (handoff; state as of 2026-09-27)

## State as of 2026-09-27 (read this first)

### Running
- **Old-recipe reproduction:** ORC array **13904287**, `sweeps/orc_old_recipe.sbatch`, code 953dc0e
  (branch `fast-clamp`), outputs `~/memory_encoding_benchmarks/old_recipe`. It runs Jaden's 2025 recipe
  on current code: lr 1e-4, α 1e5, fraction 0.1, no clamp, no output tanh. 3-char reversal, seeds
  2718/3141/4241, 5M iterations, about 20 h each, continuing across the 1-day `cs` limit by requeue.
  Design and decision rule: `benchmarks/old_recipe_reproduction.md`. Read `final_char_acc` (the 2025
  metric; ≥ 0.9 means converged), `recall_acc` and lag-5 recall from the interval blocks in
  `logs/oldrecipe_13904287_<task>.out`. Extend a run still climbing with `N_ITERS=10000000`.
  **2026-09-28: seed 3141 converged** (0.951 final char, 93.6% recall, lag 5 85% at 4.0M; first
  ≥ 0.9 at 2.69M). Seeds 2718 and 4241 are on the plateau (lag 5 about 0.25) and continue to 10M as
  jobs 13909497 and 13909498, which wait on their 5M runs; their logs append to the same files.
  Interim tables are in `benchmarks/old_recipe_reproduction.md`.

### What we know (details in the linked documents)
1. **Fast weights carry the memory when recurrence is clipped**, on every memory task tested. Without
   them the same model is at or below chance (`benchmarks/memory_tasks_head.md`). At the 2026
   recipe they reach 60–80% recall on 3-char reversal, against 100% for SimpleRNN + BPTT with
   recurrence.
2. **The 2026 recipe (lr 1e-3/α 1e4, fraction 0.2, clamp 1, 0.25–0.5M iterations) differs from the
   one that worked in 2025** (lr 1e-4/α 1e5, fraction 0.1, no clamp, 5–10M iterations, zero-init
   slow weights).
   - In the 2025 sweep, lr 1e-3 succeeded in 2 of 152 long 3-char runs; lr 1e-4 to 3e-4 in about
     30%.
   - 2025 convergence came at 2.8M–9.4M iterations. At 400k both years sit on the same 0.66–0.76
     plateau. See `docs/archive_2025_sweep_observations.md` and
     `benchmarks/old_recipe_reproduction.md`.
3. **Late divergence at the 2026 recipe** comes from slow-weight gain growth. Trunk activations
   explode within a sequence, and so do the logits. Of seven single changes, only `--output_tanh`
   was stable and kept full recall (`benchmarks/stabilizer_pilot.md`).
   - Correction: fast weights do not saturate en masse. At clamp 1, 0–4% sit at the clamp.
4. **Saturation test** (same document, follow-up section): with output tanh, clamp 1, a fast-only
   clamp and no clamp are indistinguishable at 400k. The clamp does erase older writes (step-0
   write retention 0.08–0.10 against 0.39–0.45), but that does not show in recall on the plateau.
5. **Speed:** `--fused_update` gives 1.9× on A100 and 1.5× on H100 end to end, with the same
   dynamics as the unfused step (12 paired runs). The per-step host syncs are gone (byte-identical
   traces). See `benchmarks/README.md`.
6. **With recurrence off, about a third of all fast entries are never written:** 1,024 of trunk
   layer 0's 1,033 inputs are the zeroed hidden state.

### Branches, PRs, worktrees
- Merged: #1 characterization; #2 per-sequence `--grad_norm_clip` and regularization parity; #3
  panel and GPU benchmark; #4 sync removal and `--fused_update`; #5 `--slow_weight_decay`,
  `--output_tanh`, and the stabilizer pilot.
- Branch `fast-clamp` (worktree `../memory_encoding_fastclamp`): `--fast_weight_clamp`, the
  saturation run sets, the old-recipe launcher, and these documents.
- Held-out evaluator, reworked on branch `heldout-eval-v2` (supersedes PR #6; not merged yet).
  It carries PR #6's evaluator (31106db, 22554a2), reworked (README "Held-out evaluation"):
  - The fast writes are the training step itself: `EphemeralRNN.fast_only_dfa_step` runs
    `dfa_layer_step` with `freeze_slow=True`, with the same projected errors, clip and clamps.
    There is no second copy of the rule. A test pins it against `train_batch` bit for bit.
  - Run flag `--heldout_eval_every N` (off by default) and the standalone
    `python heldout.py --checkpoint PATH`. Both report the protocols `observed`, `strict` and
    `no_fast`, with per-lag recall and `first_answer_acc`.
  - `--unit_norm_weights` was refused, with the reason (the flag was removed on 2026-09-29;
    `--layer_norm` is the normalization now, and held-out evaluation supports it).
  - Removed as unneeded: the `continue` mode, per-row resets and `HeldOutBatch`.
  - The key-value generator (Pile B, 699b93e) stays parked on branch `heldout-eval`.
- The main checkout `/home/jaden/memory_encoding` is on branch `heldout-eval` (moved there so the
  other session's files were not disturbed).

### Next steps, in order
1. Read the old-recipe result (above). If it converges, adopt it as the benchmark recipe and redo
   the head-to-head against SimpleRNN at it. If not, bisect the 2025 differences, zero init
   first.
2. Review and merge `heldout-eval-v2` (held-out evaluator). Rerun `python heldout.py` on the
   final checkpoints before publication.
3. Only then revisit clamp and saturation, in the convergence phase: the 2025 recipe with fast-only
   clamp 1 against no clamp, past 3M iterations.
4. Older items below (DFA f′, α² and 1/B in backprop, the clipping-claim audit) remain open.

### Operational notes
- **ORC:** `ssh orc` works non-interactively. The login banner swallows the first stdout line of
  a remote command, so print a guard line first. Use `bash -lc` for Slurm tools and
  `conda activate hebby`.
- **QOS:** `-p cs,cs2 --qos cs` is the cheap, high-priority default (A100/H100, 1-day wall).
  `--qos test` on `m9g` (P100, 1 h) is for quick checks. B200 (`cs3`) needs a newer torch than the
  hebby env's 2.5.1+cu121.
- **Launchers** run from an exported copy of a commit (`git archive` plus a `REVISION` file) under
  `~/memory_encoding_benchmarks/<experiment>`. Never from a git worktree: another session cleans
  those up. They record a loss early stop (loss > 5 for 10 intervals) as a collapse, not a failed
  task.
- **Local env:** use `~/miniforge3/envs/hebby/bin/python`; the default `python` is a different
  torch 2.9 and fails the golden test. The local GPU driver is broken until a reboot (NVML
  mismatch). Deckard (`ssh jaden@deckard`, 2× RTX A6000) is free for quick GPU checks.
- **Shell pitfall:** `name=$(false-returning command)` exits under `set -e`. Use `if` instead
  (it bit `orc_speed_benchmark.sbatch` once).

## Handoff of 2026-09-23 (partly superseded; sections 1, 2 and 5 are done)

Decisions Jaden made on 2026-09-23 that are **not implemented yet**. Each is a separate
commit. Follow the golden-trace policy in `tests/README.md`: a change that alters a trace
regenerates `tests/fixtures/training_traces.json` in the same commit, adds a row to the
regeneration log, and bumps `CHECKPOINT_CODE_VERSION` (`utils.py`) if the values change.
Run tests with the `hebby` env: `python -m pytest tests/ -q` (all passing at the time of
writing, including with the GPUs visible).

Guiding rule from Jaden: **both model architectures (`EphemeralRNN`, `SimpleRNN`) support the
same hyperparameters, except where it fundamentally makes no sense; in that case a curt
comment at the point of use says why.**

## 1. Hyperparameter parity audit (Jaden's rule above): done 2026-09-25
- `--grad_norm_clip` now applies to both models under every updater. The ephemeral model clips
  each sequence's raw gradient (its weight slices and bias shares) before α
  (`EphemeralRNN.clip_grad_norm_per_sequence`; Jaden chose per sequence over whole batch).
  SimpleRNN keeps `clip_grad_norm_` on its shared gradients. `tests/test_grad_norm_clip.py`.
- Ephemeral BPTT now applies `--weight_clamp` and `--unit_norm_weights`, which bound the slow
  weights it trains. A layer with no gradient in a step (`i2h` under per-step backprop) is now
  regularized too. `--ephemeral_update_clamp` stays ephemeral DFA/backprop only; README
  "Updaters" explains why. `CHECKPOINT_CODE_VERSION` 12.
- `run_training.sh` and `slurm_run.sh` set `GRAD_NORM_CLIP` and `EPHEMERAL_UPDATE_CLAMP`
  separately. The four old sweeps that passed one `$GRAD_CLIP` to both flags now pass only
  the model's historical one, so rerunning them never applies both clips.
- `--plasticity`, `--ephemeral_fraction`, `--forget_rate` remain ephemeral-only (no ephemeral
  entries in SimpleRNN).

## 2. Batch-mean DFA step for SimpleRNN's shared weights: done
The justification is in the README (Updaters, rnn + dfa) and at `DFALinear`.
`test_rnn_dfa_gradient_is_the_batch_mean_of_per_sequence_outer_products` pins the step.

## 3. Research, not code yet
- **DFA and f′** (README "Known issues"): examine whether DFA should include the activation
  derivative (Nøkland 2016). Any change goes in the shared `dfa_*` helpers for both models.
- **α² in backprop** and the **1/B factor**: fix only with full before/after runs.
- The historical topology-by-slow-initialization benchmark is complete. Seven fully ORC-paired
  seeds show a strong interaction: default initialization rescues most of the serial Elman
  layout's zero-init deficit, while the panel provides no clear evidence that it improves the
  historical sibling-head layout. See `benchmarks/3pal_architecture_initialization.md`. This does
  not directly compare current HEAD, whose restored fork trains `i2h` through direct DFA and has
  other subsequent architecture changes.

## 4. Test the paper's clipping claim in a separate worktree

The paper currently says that "gradient clipping ... fails," but the old ephemeral-model
`--grad_clip` flag was not conventional gradient-norm clipping. It element-wise clamped the
plasticity-scaled fast-weight update and is now named `--ephemeral_update_clamp`. Keep three
distinct interventions separate in code, run names, plots, and paper language:

- `--grad_norm_clip`: rescale a complete gradient vector by its norm before the step;
- `--ephemeral_update_clamp`: element-wise clamp the alpha-scaled update on fast entries; and
- `--weight_clamp`: clamp weights after the step (`--weight_clamp 1` is active in the current
  3-palindrome architecture/initialization benchmarks, while `--ephemeral_update_clamp 0` is
  off). Forgetting is damping, not clipping.

Do this work on its own branch/worktree so it does not mix with the architecture/init benchmark.

### 4a. Historical audit (no training)

1. Identify the commits, launch scripts, W&B groups, and exact code path behind every clipping
   result cited in `paper/paper_content.tex`, `paper/appendix.tex`, and `paper/pre-cut.tex`.
2. Record for each result whether it tested raw gradient-norm clipping, element-wise fast-update
   clipping, or post-update weight clipping, including the threshold and whether the clamp bound.
3. If the evidence used only the old ephemeral `--grad_clip` behavior, change the provisional
   interpretation from "gradient clipping fails" to "element-wise fast-update clipping did not
   help in that sweep." Do not edit the paper until the audit and controlled runs are reviewed.

### 4b. Implementation/characterization tests

Status 2026-09-25: `--grad_norm_clip` for the ephemeral model is implemented and tested
(section 1). A threshold that never binds (e.g. `1e30`) is bit-identical to no clip and logs
`grad_norm_mean`, `grad_norm_max` and `grad_norm_clip_fraction` per print interval, which is
the measurement the pilot needs. The α-scaled update norms are the existing
`*_ephemeral_update_norm` / `*_slow_update_norm` logs.

Before launching a new sweep, define `--grad_norm_clip` for the ephemeral model without changing
the existing two clamp operations. Match the SimpleRNN convention: compute the norm over all live
parameter gradients and rescale before the update. Explicitly document whether this is before or
after alpha scaling; log both the raw gradient norm and the actual alpha-scaled update norm so the
choice is visible. Add tests that pin:

- disabled clipping is trace-identical to the current control;
- global norm clipping binds at the requested norm and preserves gradient direction/ratios;
- `--ephemeral_update_clamp` changes only fast entries and clamps them element-wise after alpha;
- `--weight_clamp` bounds weights only after the update and does not alter gradients;
- the three mechanisms can be enabled independently and their update order is fixed; and
- DFA and backprop exercise the intended clipping path; treat BPTT separately because fast-weight
  updates are wiped before a subsequent training forward pass.

Any changed golden trace follows `tests/README.md`: regenerate the fixture, append its regeneration
log, and bump `CHECKPOINT_CODE_VERSION` when values change.

### 4c. Controlled training panel

Use the same model, dataset, seeds, initialization, training budget, and all non-clipping
hyperparameters in every arm. Start with ephemeral + DFA on the 3-palindrome task, because that is
the setting of the current claim. Use at least the seven fully ORC-paired benchmark seeds; if an
eighth seed is added, run every arm in the same environment. Compare one mechanism at a time
against an unclipped control:

- no clipping: all three clipping thresholds zero;
- global gradient-norm clipping: off plus several thresholds chosen from measured unclipped norms;
- element-wise fast-update clipping: off plus several thresholds chosen from measured unclipped
  fast-update magnitudes; and
- post-update weight clipping: off plus several thresholds, including 1 for continuity with the
  current launcher setting.

Use a short pilot only to choose thresholds that span never-binding, intermittently binding, and
nearly-always-binding regimes; do not select the final conclusion from the pilot. In final runs,
log the fraction of steps on which each mechanism binds, pre/post norm or magnitude, non-finite
events, recall accuracy, lag-specific recall, exact-sequence recall, loss, and time-to-threshold.
Report paired seed differences and uncertainty, not only the best threshold or mean curve.

Only make the broad paper claim about conventional gradient clipping if the global-norm panel
supports it. Otherwise name the exact operation tested and scope the conclusion to the updater,
task, threshold range, and training regime.

## 5. Current-code benchmark and speed (2026-09-25)
- Done: `sweeps/orc_3pal_head_panel.sbatch` (ORC array 13896047), results in
  `benchmarks/3pal_head_panel.md`. At lr·α = 1 and 250k iterations the ephemeral model reaches
  45% recall (chance 14%), mostly at lag 1. Its lag-5 and final-character accuracy are at the
  always-predict-padding baseline, while SimpleRNN + BPTT solves the task on every seed.
- Done: `sweeps/orc_memory_tasks.sbatch` (arrays 13897313, 13897527), results in
  `benchmarks/memory_tasks_head.md`. With recurrence clipped, the fast weights clearly carry the
  memory on every task where training stays stable: 3-char reversal peaks at 81% recall against
  7% without fast weights. They stay short of SimpleRNN + BPTT, which gets 100% on reversals.
  Runs are unstable late: slow-weight norms grow until the loss explodes and recall collapses
  (every α 1e4 run on 7+ token tasks, 2 of 4 at α 3e3). **Next:** test stabilizers on 3-char
  reversal before long runs (`--unit_norm_weights`, output tanh, slow-weight decay, tighter
  weight clamp).
- Speed (2026-09-26, Jaden: "make sure the speed changes are truly just that"):
  - The per-step host syncs are gone, with the fixture byte-identical.
  - The update is refactored into shared helpers, again byte-identical.
  - `--fused_update` compiles those same helpers: bit-identical uncompiled, equal to rounding
    compiled, 3.4× on an A6000.
  - Done: `sweeps/orc_fused_equivalence.sbatch` (arrays 13899359, 13899360). Fused and unfused
    runs agree on every logged metric over 200k iterations, and their weight differences are of
    the same order as A100 vs H100 (`benchmarks/README.md`). End to end: A100 36.7 → 68.5 it/s,
    H100 52.7 → 81.1. Jaden accepted the rounding difference on 2026-09-26.
- Speed (`sweeps/orc_speed_benchmark.sbatch`, `benchmarks/benchmark_dfa_throughput.py
  --device cuda`). Main is 1.2× (P100) to 1.7× (H100) slower than the scratch harness. Both
  are 5–10× below the memory-bandwidth floor, because every step makes about 15 elementwise
  passes over each `[B, out, in]` per-sample weight tensor. On H100 the trainer is also
  CPU/launch-bound (`.item()` every step in `train_batch`). A fused `torch.compile` path
  (`fused_core`) measures the ceiling: 3.1× on an A6000, 1.6× on A100, 1.35× on H100
  (`benchmarks/README.md`). Candidate changes: drop the per-step sync, then optionally fuse
  the update behind a flag. Fusion changes GPU floating point (FMA), so treat
  it as a new numerical path.

## Implemented from this handoff
- Slow entries of `per_sample_weights` start from `nn.Linear`'s default initialization,
  repeated over the batch without another RNG draw; fast entries start at zero. Implemented
  after Jaden accepted the proposed fix on 2026-09-23. Full before/after benchmarks remain.
- Main-code DFA throughput was benchmarked against the scratch harness and profiled. Allocation
  and gradient-clearing changes improved native throughput 15-20% without changing any golden
  trace; methodology and raw results are under `benchmarks/`.
- SimpleRNN now derives the same `input + hidden` trunk width, GELU activation, residual
  placement, forked heads, recurrence switch, and positional input width as EphemeralRNN.
- `--unit_norm_weights` and `--weight_clamp` now apply after every SimpleRNN update, using a
  weights-only helper shared with EphemeralLinear; normalization precedes the clamp.
- The 3-palindrome historical topology-by-initialization factorial completed across seven paired
  ORC seeds. Standard initialization improved serial-Elman mean recall by 27.47 percentage points
  but changed historical sibling-head recall by -3.11 points; the paired interaction was +30.57
  points (95% t interval [19.28, 41.87]). Full scope and caveats are in
  `benchmarks/3pal_architecture_initialization.md`.

## Already decided, nothing to do
- `EPHEMERAL_AUTO_PREPROCESS=1` forces preprocessing even under SLURM: already implemented
  (`preprocess.auto_preprocess_enabled`).
- The AST code hash depends on the Python version: accepted (`environment.yml` pins 3.11).
- One global `CHECKPOINT_CODE_VERSION`: kept.
- `self_grad`: removed (see `docs/self_grad.md`).
