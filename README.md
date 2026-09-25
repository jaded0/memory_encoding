# Memory Encoding with EphemeralRNN

This repository implements and compares different weight update mechanisms for recurrent neural networks, with a focus on the EphemeralRNN architecture, whose ephemeral weights (a fraction of each layer with a high learning-rate multiplier) act as short-term memory.

## Overview

The project explores three different training approaches:
1. **DFA (Direct Feedback Alignment)** - Uses fixed random feedback weights for error propagation
2. **Backprop** - True gradients with an update after every step, and no gradient through time (hidden state detached)
3. **BPTT (Backpropagation Through Time)** - Accumulates the loss across the whole sequence and updates once at the end

## Key Features

### EphemeralRNN Architecture
- **Ephemeral weights**: A subset of weights with plasticity α > 1 that adapt rapidly, for short-term memory; the rest are slow weights
- **Per-Batch Adaptation**: Each sequence in a batch has independent weight adaptations
- **Forgetting Mechanism**: Controlled decay of the ephemeral weights after each update

"Ephemeral" means this fast-weights-plus-decay mechanism. The updater (DFA, backprop, BPTT)
and the model (`--model_type ephemeral` or the `rnn` baseline) are independent axes, and
experiments test permutations of them; the tables under [Updaters](#updaters) describe
every combination as the code behaves today.

### Training Methods

#### DFA (Direct Feedback Alignment)
- Computes error signals using fixed random feedback weights
- Updates weights immediately at each time step
- Independent gradient computation for each sequence in batch
- No temporal gradient flow (hidden state detached)

#### Backprop
- Standard gradient computation through the network
- Updates weights immediately at each time step
- Independent gradient computation for each sequence in batch
- No temporal gradient flow (hidden state detached)

#### BPTT (Backpropagation Through Time)
- Accumulates loss across entire sequence
- Updates weights only after processing the complete sequence
- Temporal gradient flow through hidden states (only with `--enable_recurrence True`; off by default)
- Sequence-level optimization

## Updaters

Each EphemeralLinear layer holds `per_sample_weights` of shape `[batch, out, in]`, one copy
per sequence. Slow entries start from the layer's standard `nn.Linear` initialization, copied
identically to every sequence without an additional RNG draw; fast entries start at zero. A
fixed random `ephemeral_mask` marks `--ephemeral_fraction` of each layer's entries as
ephemeral, with plasticity α = `--plasticity`. The rest are slow weights with plasticity 1.
The output layer `i2o` has no ephemeral entries, so all of its weights use the standard init:
its mask is empty, so it has plasticity 1 everywhere and is never decayed or wiped
(the W&B `nominal_ephemeral_lr` and `nominal_mean_lr` config values therefore describe the
hidden layers and `i2h` only). At the start of every sequence, `start_sequence_wipe()`
replaces each sequence's copy with the batch mean and zeroes the ephemeral entries.
Forgetting multiplies the ephemeral entries by `1 - forget_rate`. Layers without ephemeral
entries log no ephemeral norms.

Both models use a forked transition/emission layout. At each step,
`combined = hidden_layers(cat(x_t, h_{t-1}))` (plus the residual, if on),
`h_t = tanh(i2h(combined))`, and `y_t = i2o(tanh(combined))`. The state and output heads can
therefore specialize over a shared deep representation. With `--enable_recurrence False`, both
heads still execute but zeros are fed to the next step instead of `h_t`.

| `--updater` | `--model_type ephemeral` | `--model_type rnn` (SimpleRNN baseline) |
| --- | --- | --- |
| `dfa` | Every step: DFA gradients, `apply_update`, forget | Every step: the same DFA gradients without ephemeral weights (all layers, `i2h` included; see below), then `w -= lr * grad`; `--grad_norm_clip` is a global grad-norm clip |
| `backprop` | Every step: `backward()` on the batch-mean step loss, `scale_ephemeral_grads`, `apply_update`, forget; the forked `i2h` gets no same-step gradient | Every step: `torch.optim.SGD` on the batch-mean step loss; forked `i2h` gets no same-step gradient; `--grad_norm_clip` is a global grad-norm clip |
| `bptt` | After the last step: `backward()` on the batch-mean summed loss, `scale_ephemeral_grads`, plain `p -= lr * p.grad`, forget | After the last step: `torch.optim.SGD`; `--grad_norm_clip` is a global grad-norm clip |

For the ephemeral model (`g` is the gradient of one sequence's own loss, `B` is `--batch_size`):

| | DFA | Backprop | BPTT |
| --- | --- | --- | --- |
| Error signal | Per-sequence `output_error`; hidden layers receive it through fixed random `feedback_weights` | Autograd | Autograd |
| Hidden state | Detached every step | Detached every step | Not detached |
| When weights change | Every step | Every step | Once, after the last step |
| Order per update | Update (incl. clamp and normalize), then forget | Update (incl. clamp and normalize), then forget | Update, then forget (once) |
| Step on an ephemeral weight | `lr·α·g` | `lr·α²·g/B` | `lr·α·g/B`, zeroed by the next `start_sequence_wipe()` |
| Step on a slow weight | `lr·g` | `lr·g/B` | `lr·g/B` |
| Layers that change | Hidden layers, `i2h` (direct feedback), `i2o` | Hidden layers and `i2o`; `i2h` only through future loss under BPTT | Every parameter with a gradient (hidden layers, `i2h`, `i2o`) |
| `--ephemeral_update_clamp` | Element-wise clamp on α-scaled ephemeral updates | Same as DFA | Ignored |
| `--weight_clamp`, `--unit_norm_weights` | Applied after each update | Applied after each update | Ignored |

`--grad_norm_clip` applies only to the `rnn` baseline, and `--ephemeral_update_clamp` only to
the ephemeral model. They used to be one flag, `--grad_clip`; see [Renamed flags](#renamed-flags-2026-09).

**DFA in the SimpleRNN baseline** (since 2026-09; before, `--model_type rnn --updater dfa`
ran but changed no parameters). It is the ephemeral model's DFA with the ephemeral parts
removed, so the two models can be compared under DFA. SimpleRNN's layers are `DFALinear`
(an `nn.Linear`), which shares the `dfa_*` helpers in `ephemeral_model.py` with
`EphemeralLinear`:

- The error is the same per-sequence `output_error` (softmax − target, from the unreduced loss).
- The hidden layers and `i2h` project it through a fixed random feedback matrix `[vocab, out]`,
  initialised like `EphemeralLinear.feedback_weights` (`xavier_normal_`). `i2o` takes it directly.
- Each layer's gradient is the outer product of its projected error with its input at that
  step. The step is `w ← w − lr·g` and `b ← b − lr·mean(projected error)`, and the hidden
  state is detached every step.
- SimpleRNN has one weight shared by the batch, not one copy per sequence, so the
  per-sequence outer products are averaged over the batch: the step on a weight is
  `lr·mean_B(g)`. The bias step is already a batch mean. In the ephemeral model each copy of
  a slow weight takes `lr·g` for its own sequence, and `start_sequence_wipe()` sets every copy
  to the batch mean before the next sequence, so each step's contribution to the batch-mean
  weight is the same `lr·mean_B(g)`.
- There is no plasticity, ephemeral mask, forgetting or wiping. `--ephemeral_update_clamp`
  is ignored, since it only clamps ephemeral entries. `--weight_clamp` and `--unit_norm_weights`
  are ignored, as SimpleRNN ignores them under every updater. `--grad_norm_clip` clips the
  global norm of the DFA gradients before the step, as it does for the SGD gradients under
  backprop and BPTT.
- The feedback matrices are drawn after every layer is initialised, so rnn + dfa starts
  from the same weights as rnn + backprop at the same seed. They are buffers, and only an
  rnn + dfa model has them (`CHECKPOINT_CODE_VERSION` 5). rnn + backprop and rnn + BPTT
  are unchanged.
- The architectures still differ outside DFA: SimpleRNN's hidden layers use ReLU (the
  ephemeral model's use GELU), and they are `--hidden_size` wide (the ephemeral model's are
  input + hidden).

Under DFA, `output_error` (`train.py:136`) is a single tensor, and the same object is passed
to every layer's `populate_dfa_gradients`. It is ∂loss/∂output per sequence. It used to have two names,
`global_error` and `reward_update`, but they were always one object. `i2o`
keeps a reference to it for its bias update, which runs after other layers' updates, so it
must never be modified in place during a step. `tests/test_dfa_error_signals.py` fails if it is.

The entries that differ between columns are explained, with evidence, in the next section.

## Known issues / behaviours under review

Planned, decided-but-unimplemented work (baseline architecture matching and hyperparameter
parity) is listed in `docs/next_steps.md`.

These describe the current code. They are recorded here, not changed, until they can be
re-examined with full training runs before and after. Line numbers were last checked after
the 2026-09 change that added DFA to the SimpleRNN baseline.

- **Backprop applies α twice (α²) on ephemeral weights.** The backprop branch calls
  `rnn.scale_ephemeral_grads(plasticity)` (`train.py:207`), which multiplies masked gradients
  by α (`ephemeral_model.py:293-301`). `apply_update` then multiplies by the `plasticity`
  tensor, which is α on the mask (`ephemeral_model.py:94`, `:195`). DFA does not call
  `scale_ephemeral_grads`, so it applies α once. BPTT calls it and then takes a plain SGD step
  (`train.py:274-285`), so it also applies α once. Measured with α = 7, the masked step is
  49·lr·g under backprop and 7·lr·g under DFA; slow weights get 1·lr·g under both.
- **Backprop and BPTT carry a 1/B factor that DFA does not.** Backprop calls `backward()`
  on `step_loss.mean()` over the batch (`train.py:203`), and BPTT on
  `accumulated_loss.mean()` (`train.py:270`), so each sequence's `per_sample_weights` gradient
  is 1/B of its own loss gradient. DFA takes each sequence's own error, using
  `grad_outputs=ones` on the unreduced loss (`train.py:136`; the reduction is forced to
  `'none'` at `train.py:327-329`). This was a deliberate choice at the time, and it helped
  the loss numbers. Combined with α², backprop's per-sequence ephemeral step is α/B times
  DFA's at the same `--learning_rate` and `--plasticity`.
- **Forked state and emission with direct DFA to `i2h` (2026-09).** The shared deep
  representation feeds separate current-output and recurrent-state heads. This preserves a
  direct route from ephemeral features to emission and lets memory state specialize separately.
  The state head receives its own fixed DFA projection every step even though it affects only
  future outputs; this is an explicit local surrogate for temporal credit, not the gradient of a
  future loss. Per-step backprop does not train the forked `i2h`, because hidden state is detached
  between steps, while BPTT trains it through later outputs. The topology decision and supporting
  BPTT experiments are documented in `docs/tapped_vs_forked_rnn_report.md`. Checkpoints from the
  preceding serial Elman layout are refused (`CHECKPOINT_CODE_VERSION` 8).
- **Ephemeral + BPTT: fast weights are frozen within a sequence.** BPTT is the contrast to
  per-step backprop and DFA in the permutation grid above. Its only update comes after the
  last step (`train.py:267`), and `start_sequence_wipe()` zeroes the ephemeral entries at the
  start of the next sequence (`train.py:80`, `ephemeral_model.py:120-125`). Updates to ephemeral
  entries therefore never reach a training forward pass, and only slow weights and biases
  learn. As a result, `--plasticity` and `--forget_rate` do not affect ephemeral BPTT
  training. (The forget set is exactly `ephemeral_mask`, `ephemeral_model.py:83-99`.) This
  path also ignores `--ephemeral_update_clamp`, `--weight_clamp` and `--unit_norm_weights`,
  because it never calls `apply_update` or `_apply_regularization` (`train.py:281-285`), and
  it does not increment `training_instance`. Checked on a small model (under the old flag
  names): changing α, `--forget_rate`, the update clamp or the weight clamp leaves a
  four-sequence BPTT loss trajectory bit-identical.
- **`--unit_norm_weights` rescales the `per_sample_weights` only, one sequence at a time.**
  `_apply_regularization` divides each sequence's `[out, in]` slice of a layer's
  `per_sample_weights` by that slice's own L2 norm after each update
  (`ephemeral_model.py:252-264`), so one sequence's scale does not depend on the others in
  the batch. (Until 2026-09 the norm was taken over the whole `[batch, out, in]` tensor.)
  `plasticity`, the bias, the feedback weights,
  the traces and the logged update norms are left alone (until 2026-09 they were all
  rescaled, together with the then-stored `forgetting_factor`, so α and the forget rate
  drifted from the CLI values after the first update). `--weight_clamp` is applied after the
  normalization, so a clamp of 1 or more never binds when `--unit_norm_weights` is on.
- **DFA omits the activation derivative f′ (to examine; not changed).** Every non-output
  layer's DFA error is the output error projected straight through its feedback matrix,
  `projected = output_error @ feedback_weights` (`dfa_projected_error`,
  `ephemeral_model.py:21-27`), and that is used as is for the weight step (outer product with
  the input) and the bias step. It is never multiplied by the derivative of the layer's
  activation: gelu′ for the ephemeral model's hidden layers (`ephemeral_model.py:440`),
  relu′ for SimpleRNN's (`:633`), and tanh′ for `i2h` in both (`:452`, `:638`). This differs
  from Nøkland's formulation (2016, "Direct Feedback Alignment Provides Learning in Deep
  Neural Networks"), where a hidden layer's update is δa_l = (B_l·e) ⊙ f′(a_l), with a_l the
  layer's pre-activation, e the output error, and δW_l = −δa_l·h_{l−1}ᵀ. Only the output
  layer takes e directly, as `i2o` does here. Jaden wants to examine whether the
  update *should* include f′. The SimpleRNN DFA baseline (`DFALinear`, 2026-09) omits it too,
  on purpose, so that the two models' DFA is the same computation. Any future change must be
  applied to both, most simply in the shared `dfa_*` helpers (`EphemeralLinear` already records
  each step's pre-activation as `out_traces`; `DFALinear` records only its input). It would
  change every DFA golden trace.
- **`EphemeralLinear._update_bias` is dead code with a flipped sign.** Nothing calls it,
  and it adds `+lr·projected_error` (`ephemeral_model.py:243-250`). The live bias update is
  `_update_bias_from_grad`, which subtracts (`ephemeral_model.py:215-226`).

## Paper settings & stability

The paper trains with plain SGD at a base learning rate of 1e-4
(`paper/paper_content.tex:135`), ephemeral plasticity α = 1e4 or 1e5 (`:104`), and a
"forgetting rate coefficient" of 0.7 applied after each update (`:124-127`). In code terms
that is `--forget_rate 0.3`. The current CLI
defaults are lr 1e-4, α 1e5 and `--forget_rate` 0.01, which keeps 1 − forget_rate = 0.99 of
each ephemeral weight per step (`train.py:391-396`).

### Terminology: paper vs code

The code's convention is the one to use: `forget_rate` is the fraction of each ephemeral
weight removed per step, and 1 − forget_rate is the fraction kept. The paper's text
reports the fraction kept; its figure legends use code `--forget_rate` values (the Fig. 1
key-recall legend "ephemeral 0.0001 0.5" is lr 1e-4, `--forget_rate 0.5`).

| Paper term | Code / CLI name | Meaning | Formula | Conversion |
| --- | --- | --- | --- | --- |
| "Forgetting rate coefficient" in the text (`paper_content.tex:127`); forget rate in the figure legends | `--forget_rate`; config and W&B key `forget_rate`; `FORGET_RATE` in the run scripts; `forget_rate=` in the `EphemeralRNN`/`EphemeralLinear` constructors | Fraction of each ephemeral weight removed per step | `w ← (1 − forget_rate)·w` | The text's coefficient is 1 − `forget_rate`: its "forgetting rate coefficient 0.7" is `--forget_rate 0.3`. Legend values are already `--forget_rate` values |
| (none) | `forget_rate * ephemeral_mask`, computed in `EphemeralLinear.apply_forget_step` (checkpoints before 2026-09 stored it as the tensor `forgetting_factor`) | Per-entry forget rate: `forget_rate` on the ephemeral mask, 0 elsewhere. A removal fraction, not a multiplier | `w ← (1 − forget_rate·mask)·w`, element-wise | As above, per entry |
| Plasticity α_k (`:100-107`) | `--plasticity` (formerly `--plast_clip`); config and W&B key `plasticity`; `plasticity` tensor | Learning-rate multiplier on ephemeral weights (1 on slow weights) | step `lr·α·g` (DFA) | `--plasticity` = α |
| Ephemeral (fast) weights | Entries where `ephemeral_mask` is true, a `--ephemeral_fraction` (formerly `--plast_proportion`) of each hidden layer and `i2h` | Weights with plasticity α that are decayed and wiped | | |
| Slow weights | The other entries of `per_sample_weights` (formerly `candidate_weights`) | Plasticity 1, never decayed or wiped | | |

Ordering: as in the paper (`:124-127`), the decay comes after each update in all three
updaters (`train.py:167`, `:231`, `:288`), so one step is
`w ← (1 − forget_rate)·(w − lr·α·g)`. Under DFA and backprop the update it follows
includes `--ephemeral_update_clamp`, `--unit_norm_weights` and `--weight_clamp`. (The class constructors used to
default to `forget_rate=0.7`, a leftover of the paper's coefficient that would have kept
only 0.3 of each weight. They now default to 0.01, matching the CLI; `train.py` always
passed `--forget_rate` explicitly, so no run changed.)

These settings are part of the paper's method, and they were tuned under today's α² and
1/B behaviour. Fixing either changes backprop's effective fast-weight step at the same
flags, and fixing 1/B also changes every BPTT step. Backprop and BPTT settings will
therefore need re-tuning or re-deriving. DFA's step involves neither factor. At the
defaults, the per-sequence ephemeral step multiplier is lr·α = 10 for DFA and
lr·α²/B = 62,500 for backprop with B = 16.

The removed `UNIFIED_UPDATES.md` (Aug 2025, still in git history) recorded lr 1e-4,
`PLAST_CLIP` 1e3 and `FORGET_RATE` 0.01 as the settings that stopped backprop producing NaNs.
They were found while backprop was applying α². Its forget rate of 0.01, like today's
default, disagrees with the paper's text value, which is `--forget_rate 0.3`.

## Installation

```bash
# Clone the repository
git clone https://github.com/jaded0/memory_encoding.git
cd memory_encoding

# Create conda environment (recommended)
conda env create -f environment.yml
conda activate hebby

# Or install dependencies manually
pip install torch wandb matplotlib numpy psutil
```

## Usage

### Basic Training

The defaults are the configuration the run scripts use: ephemeral model, DFA, no recurrence,
`last_one` input, no normalization or clipping, batch 16, and the hyperparameters
(lr 1e-4, plasticity 1e5, forget rate 0.01, hidden 1024, 10% ephemeral weights) that solve
3-char palindromes. `python train.py --help` lists every default. W&B tracking is on by
default and needs a login and network; pass `--track False` to run offline.

```bash
# Train with DFA updates (the defaults)
python train.py

# Train with backprop updates
python train.py --updater backprop --model_type ephemeral

# Train with BPTT updates
python train.py --updater bptt --model_type ephemeral
```

### Key Parameters

- `--updater`: Choose between `dfa`, `backprop`, or `bptt`
- `--model_type`: Choose between `rnn` or `ephemeral`
- `--learning_rate`: Learning rate for weight updates
- `--plasticity`: Plasticity (learning-rate multiplier, alpha) of the ephemeral weights
- `--ephemeral_fraction`: Fraction of each hidden layer's weights that are ephemeral
- `--forget_rate`: Fraction of each ephemeral weight removed per step
- `--ephemeral_update_clamp` (ephemeral model) / `--grad_norm_clip` (`rnn` baseline): update clamp or gradient-norm clip
- `--unit_norm_weights`, `--weight_clamp`: rescaling and clamping of the weights after each update
- `--resume` / `--resume_checkpoint PATH`: Resume from `latest_checkpoint.pth`, or from an explicit checkpoint
- `--batch_size`: Number of sequences processed together
- `--seed`: Seed Python, NumPy, Torch, dataset shuffling, and DataLoader sampling (unset = drawn from the OS; see below)
- `--deterministic`: Require deterministic Torch operations

Resuming is opt-in: without `--resume` or `--resume_checkpoint`, training starts from
scratch even if `latest_checkpoint.pth` exists. `--resume` with no checkpoint present starts
from scratch; a missing explicit `--resume_checkpoint` is an error. Once a checkpoint is
chosen, any load failure aborts the run, including a mismatch in hidden size, layer count,
updater, model type, charset size, `--forget_rate`, `--dataset` or `--learning_rate`
(`utils.py`, `load_checkpoint`). A changed `--forget_rate` is refused because it would
change a running experiment's decay (before 2026-09 the checkpoint stored the per-entry rate
as `forgetting_factor`, and the new value was silently ignored). A changed dataset or learning rate is refused because it
means a different experiment: `slurm_run.sh` keys checkpoints by SLURM job name and always
passes `--resume true`, so a reused job name would otherwise silently continue an old
checkpoint. Every CLI argument is saved in the checkpoint's config, and on resume
`load_checkpoint` prints every field that differs (`checkpoint -> this run`) before these
checks. Other differences, such as `--n_iters` or `--print_freq`, are only printed;
`--plasticity` is printed and then applied to the loaded plasticity as before. Checkpoints
that did not record a field (older runs) are not checked on it. The seed and `--deterministic`
come from the checkpoint; passing a different value is an error.

Every checkpoint records `code_version`, the value of `CHECKPOINT_CODE_VERSION` in
`utils.py` when it was written. A resume first checks it and refuses, with an error saying
the run must start fresh, if it differs from the current value or is missing (every
checkpoint written before 2026-09 has none). `CHECKPOINT_CODE_VERSION` is bumped whenever the
training mechanics change in a way that makes an in-flight run's continuation meaningless
(see the comment on it); continuing such a run would mix two different algorithms in one
experiment. To continue work from a refused checkpoint, start a new run.

The state dict must match the model exactly: a missing or unexpected tensor is an error,
not a freshly initialised tensor. `load_checkpoint` still maps the tensor names used before
the 2026-09 naming cleanup (`candidate_weights` → `per_sample_weights`, `mask` →
`ephemeral_mask`, `last_high_plast_update_norm` / `last_low_plast_update_norm` →
`last_ephemeral_step_norm` / `last_slow_step_norm`), checks that a stored `forgetting_factor`
equals `forget_rate` on the mask and drops it, and maps old config keys (see below). Those
checkpoints have no `code_version`, so a resume refuses them before the mapping runs; the
mapping is kept, and unit-tested, for reading old checkpoints outside a resume.

### Renamed flags (2026-09)

The old names still work. Each prints a one-line `DEPRECATED:` note, so old scripts and the
frozen `checkpoints/<run>/run_used.sh` copies that `sweeps/bulk_restart.sh` resubmits run
unchanged. (A resume of the checkpoints written next to those copies is refused by the
`code_version` check, since they predate it; such runs have to start fresh.) Configs, checkpoints and W&B record only the new names. Giving an old and a new
name with different values is an error. The old names are hidden from `python train.py --help`
(and its usage line); this table is their reference.

| Old flag | New flag | Notes |
| --- | --- | --- |
| `--plast_clip` | `--plasticity` | α |
| `--plast_proportion` | `--ephemeral_fraction` | |
| `--grad_clip` | `--ephemeral_update_clamp` with `--model_type ephemeral`, `--grad_norm_clip` with `--model_type rnn` | Each model only ever used the one that applies to it |
| `--clip_weights` | `--weight_clamp` | |
| `--normalize` | `--unit_norm_weights` | |
| `--plast_learning_rate`, `--imprint_rate` | (removed) | Were unused; still accepted and ignored |

W&B also renamed its keys: `high_lr` → `nominal_ephemeral_lr`, `effective_lr` →
`nominal_mean_lr`, and `avg_high_plast_*` / `avg_low_plast_*` → `avg_ephemeral_*` /
`avg_slow_*`.

### Seeds & reproducibility

Every run is seeded. On a fresh start without `--seed`, `train.py` draws a 63-bit seed from the OS
(`secrets.randbits`; never from time or job ID, so array tasks that start together get independent
seeds) and prints it near the top of the log: `Seed: 8338083689295822420 (generated), deterministic: False`.
`--seed N` picks it yourself (`(from --seed)`). The seed is saved in every checkpoint and in the W&B
config (`seed`, `seed_source`). Under SLURM it is also written to the job comment (best effort), so
`squeue -o "%i %j %k"` or `sacct -o JobID,Comment` shows it.

On resume the checkpoint is the source of truth: the seed and `--deterministic` are read from it
(`(from checkpoint)`) and need not be passed again. Checkpoints also store the Python, NumPy and Torch
RNG states and the DataLoader's position (`DataStream` in `reproducibility.py`), so preemption, requeue
or `sweeps/bulk_restart.sh` (a new SLURM job ID) continues the same random and data stream: N steps,
resume, M steps equals N+M uninterrupted steps (`tests/test_seed_resume.py`). Checkpoints from before
this change have `seed: None` and no `code_version`, so a resume refuses them; the code still resumes a
checkpoint with `seed: None` unseeded, with a warning. A job preempted before its first
checkpoint starts fresh with a new seed.

To rerun an experiment exactly, start fresh with `--seed <logged seed>` (plus `--deterministic True`
for bitwise-deterministic Torch ops; on GPU this sets `CUBLAS_WORKSPACE_CONFIG`).

### First-time cluster setup

Compute nodes have no internet, so a SLURM job never preprocesses Hugging Face
datasets itself: if the processed copy is missing, training stops with a setup
hint. Before the first `sbatch slurm_run.sh`, run this once from the repo root on
the login node:

```bash
setup_cluster/setup.sh          # add --redo to re-download and overwrite the processed data
```

This command downloads the raw datasets on the login node. It then submits a
short test-QOS job that preprocesses them offline into `$EPHEMERAL_DATA_DIR`
(default `./processed_datasets/`) and smoke-tests the `slurm_run.sh` config on a
GPU. See [setup_cluster/README.md](setup_cluster/README.md).

Local runs (no `SLURM_JOB_ID`) need no setup step: when the processed copy of a
Hugging Face dataset is missing, `train.py` prints a notice, runs the same
preparation as `python preprocess.py <name>` (downloading the raw split if it is
not in the HF cache; about a minute for TinyStories), saves it to the
processed-data directory and continues. `EPHEMERAL_AUTO_PREPROCESS=0` makes a
local run fail with the setup hint instead, and `EPHEMERAL_AUTO_PREPROCESS=1`
makes even a SLURM job prepare it (only useful where the job has the raw data
and time to spare).

The processed directory's name includes a hash of the code of `utils.filter_text`
and `utils.text_to_indices` (`preprocess.preprocessing_code_hash`), so a real code
change there makes every saved copy stale. The hash is taken over the functions'
AST with the docstrings removed, so comment-only and docstring-only edits keep it.
**That change (2026-09) itself changed the hash once**, from `code-e0abe65b67` to
`code-9c6837022f` for today's code: saves made before it are no longer found, so
rerun `setup_cluster/setup.sh` on the cluster (a local run re-prepares by itself,
or run `python preprocess.py roneneldan/tinystories`). The AST dump format belongs
to the Python version, so prepare the data with the same Python (the `hebby`
environment) that trains on it.

### SLURM time limits

The sbatch scripts request `#SBATCH --signal=B:USR1@600` and launch training through
`forward_signals` (from `sweeps/forward_signals.sh`). Ten minutes before the wall-time limit,
`train.py` stops at the next iteration, saves `latest_checkpoint.pth` (if `--checkpoint_save_freq > 0`),
records `end_reason: time_limit` in W&B, and exits with code 124; resume with `--resume`. A SIGTERM
stops the same way with `end_reason: terminated` and exit code 143.

### Advanced Features

- **Positional Encoding**: Add positional information with `--positional_encoding_dim N`
- **Residual Connections**: Enable/disable with `--residual_connection True/False`
- **Weight Normalization**: Enable/disable with `--unit_norm_weights True/False`
- **Input Modes**: Choose between `--input_mode last_one` or `--input_mode last_two`

## Testing

Run the network-free assertion suite from the repository root:

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest tests/ -q
```

`pytest` is not in `environment.yml`; install it with `pip install pytest`.

The suite includes fixed-input golden traces through the real `train.train()`
path for DFA, backprop, and BPTT (and for the SimpleRNN baseline under DFA), a finite-update smoke test for all three
(`tests/test_smoke_updaters.py`), and checkpoint/failure-path, metrics and
reproducibility tests. Others check the old flag names and checkpoints
(`tests/test_cli_aliases.py`, `tests/test_legacy_checkpoints.py`), and the per-layer
error tensors of the DFA path (`tests/test_dfa_error_signals.py`). See [tests/README.md](tests/README.md) for what the
golden traces pin, which known behaviours they currently freeze, and how to
regenerate them.

## Project Structure

- `train.py`: Main training script; `train_batch` runs one batch under any of the three updaters
- `ephemeral_model.py`: Implementation of EphemeralRNN and EphemeralLinear layers
- `preprocess.py`: Data loading and preprocessing utilities
- `reproducibility.py`: Seed resolution, RNG and data-stream checkpoint state
- `utils.py`: Helper functions and utilities
- `tests/`: Test suite; `tests/README.md` has the golden-trace contract, pinned known behaviours and regeneration log
- `tests/fixtures/training_traces.json`: CPU golden traces for all update paths

## Results and Monitoring

The training process logs metrics to Weights & Biases (WandB) including:
- Loss values
- Accuracy metrics
- Weight and gradient norms
- Plasticity parameter statistics

To disable WandB tracking, use `--track False`.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests to ensure nothing is broken
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- Direct Feedback Alignment papers
- Hebbian learning principles
- Recurrent neural network training methods
