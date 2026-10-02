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

### Keeping fast weights across sequences (`--wipe_every N`)

`--wipe_every N` (default 1, every sequence) zeroes the fast entries only at the start of every
N-th sequence (batch): the first batch of a run, then every N-th after it. The wipe does two
separable things, and only the second is skipped:

- **Slow consolidation, every sequence.** Each copy's slow entries are set to the batch mean,
  as before. This is how the slow weights learn from the batch, so it never stops. The mean of
  the slow entries is the same whether or not the fast entries are then zeroed.
- **Fast reset, every N-th sequence.** Between wipes, each batch row keeps its own fast entries
  from the sequence it just finished, not averaged across rows. The next sequence in the same
  row starts from them, after they have been forgotten at `--forget_rate` every step (and
  clamped by `--fast_weight_clamp`, if set). The data loader shuffles, so consecutive
  sequences in a row are unrelated, and what carries over is interference.
- **Hidden state.** It still starts at zero every sequence, so only the fast weights carry
  over. With `--enable_recurrence False`, the default, the fed-back state is zero anyway.

The count of sequences started is `sequence_count` in the checkpoint's `main_program_state`
(present only when N > 1), and the carried fast entries are part of `per_sample_weights`, so a
resume continues the cycle. A checkpoint without the count (N = 1, or older) wipes at its
first resumed batch. Held-out evaluation (`--heldout_eval_every`, `heldout.py`) still starts
every episode from wiped fast entries and restores the carried ones afterwards. Under BPTT the
carried entries are the post-update ones, which `--ephemeral_update_clamp` does not bound. The
SimpleRNN baseline has no fast weights, so `--model_type rnn` refuses N > 1. With N = 1 the
code path is the old one, and the golden traces are byte-identical, so `CHECKPOINT_CODE_VERSION` stays 12: a version-12 checkpoint resumes with or without the flag. `tests/test_wipe_every.py`
pins the behaviour.

### How often the slow weights update (`--slow_update_every`)

`--slow_update_every` (ephemeral + DFA only; default `1`) sets how often the slow parameters take
their DFA step, a third plasticity dimension next to the learning rate and the forget rate. Its
value is a positive integer N or `sequence`. The default is today's per-step update, bit for bit:
the golden traces are unchanged and no checkpoint key is added.

Under any other value the **fast entries still take the per-step update** (same projected error,
`--grad_norm_clip`, α, clamps and forgetting), but everything slow is frozen within the window:
the slow entries of every layer (i2o included) and every bias. Each step's slow gradient is
accumulated instead, and applied when the window ends:

- **`N`**: every N steps, and at the end of every sequence for any partial window, so a window
  never spans a wipe. Each batch row's copy takes its own summed gradient, which is what N
  per-step updates give without the drift in between. The wipe then averages the copies as usual.
- **`sequence`**: one window per sequence. The step is the batch mean of the per-sequence sums,
  written to every copy, so the slow entries stay one shared matrix and the wipe skips the mean
  (the mean of equal copies can differ in the last bit). Within a sequence the slow weights
  provably do not change, which is the setting for asking whether fast and slow weights really
  act on different timescales. It is also what a later compact fast-weight storage and a CUDA
  graph would need.
- **Stabilizers** keep their meaning but act on the window: `--weight_clamp` once per window,
  `--slow_weight_decay` as (1 − d)^steps (the same total as per step), and `--grad_norm_clip` per
  step on each step's gradient. Under `sequence`, a binding `--weight_clamp` clamps the shared
  mean, where the per-step update clamps each copy before the wipe's mean, so the two differ
  even at T = 1.
- **Together with other flags.** `--dfa_fprime` and `--layer_norm` work: the windowed step shares
  their error projection and forward pass with the per-step update, and a test checks a one-step
  sequence against the per-step result with each. `--wipe_every N` works: the slow entries are
  shared and consolidated every sequence, and only the fast entries are kept between wipes.
  Backprop, BPTT and `--model_type rnn` are refused.
- **Resuming** with a different value is refused (a checkpoint without the setting counts as 1).

`--slow_update_every` changes the training dynamics, so a run that uses it is comparable to the
default only as a different arm. Record it with the run, and keep a `1` arm in any comparison, so a
speed gain is not mistaken for an effect of the update rate. `tests/test_slow_update_every.py`
pins the behaviour.

### Backward passes per forward pass (`--fast_backward_per_forward`)

Three counters are easy to confuse: **forwards per character** (one, plus the re-forwards below),
**backward passes per forward** (this flag, for the fast entries), and **apply events** (when a
gradient is written to a parameter: `--slow_update_every` is an apply period for the slow group,
and does not change how many backward passes there are). `--fast_backward_per_forward R`
(ephemeral + DFA only; default `1`) sets the backward ratio of the fast entries. R is a positive
integer K or `1/N` for an integer N >= 2. The default is today's one DFA step per character, bit
for bit: the golden traces are unchanged and no checkpoint key is added.

- **`K >= 2`**: the character's step is split in two. Pass 1 is the ordinary forward, and its DFA
  step is applied to the fast entries only (with the character's one forgetting step); its
  projected errors and layer inputs are saved. Then K - 1 more times the forward pass is re-run on
  the same character, from the same incoming hidden state, with the updated fast weights; the
  output error is recomputed from that fresh output and a fast-only DFA step is applied, without
  forgetting (`--grad_norm_clip`, clamps and alpha as usual). After the last pass the slow half of
  pass 1's step is applied from the saved errors and inputs: the slow entries, i2o and biases,
  `--weight_clamp` and `--slow_weight_decay` (which rides on the forget step in the single-kernel
  step, so it moves into this half), per step, or accumulated as `--slow_update_every` says (an N
  window that fills up is applied after the extra passes). So every pass sees the same slow
  weights, and the slow stream, loss, metrics and `--grad_norm_clip` statistics are pass 1's alone,
  in every `--slow_update_every` mode. The hidden state passed to the next character is the last
  pass's. With K = 1 the single fused step is used, unchanged.
- **`1/N`**: only every N-th character of a sequence (the first, N + 1-th, ...) gets a fast update.
  The other characters run the forward pass, and the fast entries only forget (forgetting is
  time-based and applies every character). The slow stream is unchanged: every character
  contributes its gradient as `--slow_update_every` says.
- **Cost.** K >= 2 costs about K forwards per character plus two update kernels per layer (the
  fast half and the slow half) instead of one, so expect more than K times fewer iterations per
  second; 1/N costs no more than 1.
- **Together with other flags.** `--fused_update` (the extra passes use the same compiled fast-only
  step, with forgetting turned off), `--slow_update_every`, `--dfa_fprime`, `--layer_norm`,
  `--wipe_every` and the clamps work. Resuming with a different value is refused (a checkpoint
  without the setting counts as 1). Backprop, BPTT and `--model_type rnn` are refused for now; support
  for them would be a later change. Held-out evaluation (`heldout.py`) keeps 1:1.

`tests/test_backward_ratio.py` pins the behaviour. Like `--slow_update_every`, it changes the
dynamics, so compare it against a `1` arm.

Both models use a forked transition/emission layout. At each step,
`combined = hidden_layers(cat(x_t, h_{t-1}))` (plus the residual, if on),
`h_t = tanh(i2h(combined))`, and `y_t = i2o(combined)`. The state and output heads can
therefore specialize over a shared deep representation. With `--enable_recurrence False`, both
heads still execute but zeros are fed to the next step instead of `h_t`.

| `--updater` | `--model_type ephemeral` | `--model_type rnn` (SimpleRNN baseline) |
| --- | --- | --- |
| `dfa` | Every step: DFA gradients, `apply_update`, forget | Every step: the same DFA gradients without ephemeral weights (all layers, `i2h` included; see below), then `w -= lr * grad`; `--grad_norm_clip` is a global grad-norm clip |
| `backprop` | Every step: `backward()` on the batch-mean step loss, `scale_ephemeral_grads`, `apply_update`, forget; the forked `i2h` gets no same-step gradient | Every step: `--optimizer` (SGD, or Adam) on the batch-mean step loss; forked `i2h` gets no same-step gradient; `--grad_norm_clip` is a global grad-norm clip |
| `bptt` | After the last step: `backward()` on the batch-mean summed loss, `scale_ephemeral_grads`, plain `p -= lr * p.grad`, forget | After the last step: `--optimizer` (SGD, or Adam); `--grad_norm_clip` is a global grad-norm clip |

For the ephemeral model (`g` is the gradient of one sequence's own loss, `B` is `--batch_size`):

| | DFA | Backprop | BPTT |
| --- | --- | --- | --- |
| Error signal | Per-sequence `output_error`; hidden layers receive it through fixed random `feedback_weights` | Autograd | Autograd |
| Hidden state | Detached every step | Detached every step | Not detached |
| When weights change | Every step | Every step | Once, after the last step |
| Order per update | Grad-norm clip, α, update clamp, update, weight clamp, then forget | Same as DFA | Grad-norm clip, α, update, weight clamp, then forget (once) |
| Step on an ephemeral weight | `lr·α·g` | `lr·α²·g/B` | `lr·α·g/B`, zeroed by the next `start_sequence_wipe()` |
| Step on a slow weight | `lr·g` | `lr·g/B` | `lr·g/B` |
| Layers that change | Hidden layers, `i2h` (direct feedback), `i2o` | Hidden layers and `i2o`; `i2h` only through future loss under BPTT | Every parameter with a gradient (hidden layers, `i2h`, `i2o`) |
| `--grad_norm_clip` | Each sequence's raw gradient rescaled to norm ≤ c, before α | Same as DFA | Same, on the gradient summed over the sequence |
| `--ephemeral_update_clamp` | Element-wise clamp on α-scaled ephemeral updates | Same as DFA | Ignored |
| `--weight_clamp` | Applied after each update | Applied after each update | Applied after the update (since `CHECKPOINT_CODE_VERSION` 12) |

`--optimizer {sgd,adam}` (default `sgd`) picks `torch.optim.SGD` or `torch.optim.Adam` (default
betas and eps) at `--learning_rate`. It applies only to the SimpleRNN baseline under `backprop` and
`bptt`; any other combination with `adam` is an error. The ephemeral model has no optimizer step:
its fast entries are per-sequence copies wiped at every sequence start, so Adam's moment estimates
would average the gradients of unrelated sequences, and Adam's per-coordinate normalization would
cancel the α scaling that defines plasticity (the step would be about `lr` whatever α is). DFA,
for both models, applies its update by hand from the projected error. A checkpoint stores the
optimizer's state, and resuming with a different `--optimizer` is refused (checkpoints written
before the flag existed count as `sgd`).

With `sgd` at the default lr 1e-4, the 3-layer SimpleRNN learns very slowly through its state:
each step's transition is four linear layers (three GELU trunk layers and `i2h`) at PyTorch's
default initialization, so the per-step state Jacobian has spectral norm about 0.05 at
initialization. Key recall (lag 2–3, no lag-1 targets) stays at chance for 1M iterations
(palindromes have lag-1 targets, which the state path learns first). For a baseline that
represents standard BPTT training, use `--optimizer adam --learning_rate 1e-4 --grad_norm_clip 1
--residual_connection true` (hidden size 1024, 3 layers). Adam alone is enough for key recall
(0.99 by 10k iterations) but leaves `kv_unique_4` (lags up to 8) at chance; the residual gives the
state an identity path through the trunk, and with it `kv_unique_4` reaches the 0.37 that a
from-scratch GRU or LSTM reaches in the same number of steps. At that width Adam at 3e-4 or 1e-3
stays at chance.

Three separate clipping mechanisms exist, and only the first is gradient clipping in the usual
sense. They used to be confused under one flag, `--grad_clip`; see [Renamed flags](#renamed-flags-2026-09).

- `--grad_norm_clip c` (both models, every updater) rescales a gradient to norm at most `c`
  before the step, with torch's coefficient `min(1, c / (norm + 1e-6))`. SimpleRNN clips the
  global norm of its shared gradients (`clip_grad_norm_`). The ephemeral model has one weight
  copy per sequence, so it clips each sequence's gradient separately
  (`EphemeralRNN.clip_grad_norm_per_sequence`). The norm covers that sequence's slice of every
  layer's `per_sample_weights.grad` and its share of every bias gradient: the projected error
  under DFA, and under backprop and BPTT the gradient of the layer's output, which is retained
  for this. The threshold then does not depend on the batch size, one sequence never rescales
  another, and at batch size 1 this is exactly `clip_grad_norm_`. It is taken on the raw
  gradient, before α. After α, the α-scaled fast entries would dominate the norm, and the clip
  would act mainly on fast-weight updates. A threshold that never binds (e.g. `1e30`) leaves
  training bit-identical and logs `grad_norm_mean`, `grad_norm_max` and
  `grad_norm_clip_fraction` each print interval; use it to measure unclipped norms.
- `--ephemeral_update_clamp v` (ephemeral only, DFA and backprop): element-wise clamp of the
  α-scaled update of each ephemeral entry to `[-v, v]`. SimpleRNN has no ephemeral entries.
  This is what the paper's old "gradient clipping" sweeps tested.
- `--weight_clamp w` (both models): element-wise clamp of the weights after each update.
- `--fast_weight_clamp v` (ephemeral only): the same clamp on the ephemeral (fast) entries only,
  after `--weight_clamp`. It separates the cost of pinning fast weights from that of clamping slow
  ones (`benchmarks/stabilizer_pilot.md`). SimpleRNN has no fast entries and ignores it.
  With `w = 1` the clamp rarely binds. At lr·α ≤ 3 no fast entry reaches it, and at lr·α = 10
  0–2% of fast entries sit at ±0.99 (the clamp after forgetting) by the end of a sequence. It
  binds more with `--output_tanh` (about 10%) and with `w = 0.3` or `0.1` (8–26%);
  `benchmarks/stabilizer_pilot.md` has the measurements.

Two further stabilizers, both off by default:
- `--slow_weight_decay λ` (both models): after each update, every slow weight keeps `1 − λ`.
  The ephemeral model applies it in the forget step (fast entries keep `1 − forget_rate`),
  SimpleRNN to all its weights. Biases are excluded.
- `--output_tanh` (both models): the output head reads `tanh` of the shared trunk. This was the
  default until 2026-09-24 (`docs/tapped_vs_forked_rnn_report.md`).

One normalization, off by default:
- `--layer_norm` (both models, every updater): LayerNorm over the features of each trunk
  layer's output, after its GELU: `combined = LN(gelu(layer(combined)))`, zero mean and unit
  variance per sequence and step (`trunk_layer_norm`, eps 1e-5). It has no learnable gain or
  bias, so it adds no parameters: an affine pair would be slow weights that DFA has no rule
  for, and the next layer's weights and bias can absorb any scale and shift. It goes after the
  activation because DFA's outer product uses each layer's recorded input (`in_traces`): placed
  there, the normalized features are exactly what the next trunk layer, `i2h` and `i2o` read and
  what their DFA gradients use, and every such input row has norm √(input + hidden), which bounds
  the input side of the rank-1 DFA gradient `p·xᵀ`. The first layer's input (`x_t`, `h_{t−1}`) is
  not normalized, nor is the `tanh`-bounded recurrent state; with `--residual_connection` the
  residual is added after the last LayerNorm. DFA still omits every activation derivative, the
  LayerNorm Jacobian included (see Known issues); backprop and BPTT differentiate through it.
  A resume refuses a changed `--layer_norm`. Held-out evaluation supports it. The paper's
  "layer normalization actively harms performance" (`paper/paper_content.tex:195`) had no code
  behind it before this flag.

`--dfa_fprime` (both models, DFA only; default off): the standard DFA rule of Nøkland (2016).
Each non-output layer's projected error is multiplied element-wise by the derivative of that
layer's nonlinearity at this step's pre-activation, δ_l = (output_error @ B_l) ⊙ f′(a_l):
gelu′ (exact erf form, as `F.gelu`) for the trunk layers and tanh′ for `i2h`; `i2o` keeps the
raw output error. `--output_tanh` and `--residual_connection` change only what `i2o` and `i2h`
read, not a layer's own nonlinearity, so they leave f′ alone. The weight step is still the outer
product with the layer input and the bias step the batch mean of δ_l, so `--grad_norm_clip`'s
rank-1 closed form, `--fused_update` and held-out evaluation all take the same δ_l (it is
applied in the shared `dfa_projected_error`). `EphemeralLinear` already records the
pre-activation as `out_traces`; `DFALinear` records it only when the flag is on. Off, nothing
changes (the golden traces are byte-identical). A resume refuses a checkpoint with a different
`dfa_fprime`. `tests/test_dfa_fprime.py` checks it against an autograd reference on a tiny
model, for both models, and fused against unfused.

Ephemeral BPTT ignores `--ephemeral_update_clamp` by design: it clamps only fast-weight updates,
and under BPTT those are wiped before any forward pass reads them (see Known issues).

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
  is ignored, since it only clamps ephemeral entries. `--weight_clamp` clamps each shared
  layer's weight entries after every DFA or SGD update; biases and feedback matrices are excluded. `--grad_norm_clip` clips
  the global norm of the DFA gradients before the step, as it does for the SGD gradients under
  backprop and BPTT. Non-zero, it no longer matches the ephemeral model's DFA step exactly,
  because the ephemeral model clips each sequence's gradient separately.
- The feedback matrices are drawn after every layer is initialised, so rnn + dfa starts
  from the same weights as rnn + backprop at the same seed. They are buffers, and only an
  rnn + dfa model has them (added in `CHECKPOINT_CODE_VERSION` 5).
- Both models use the same `input + hidden`-wide GELU trunk, forked heads, optional residual,
  recurrence switch, and positional dimensions. They differ in the weights and updates:
  SimpleRNN has one shared `nn.Linear` copy per layer, while EphemeralRNN has per-sequence
  fast/slow weights, plasticity, forgetting, and sequence wipes (`CHECKPOINT_CODE_VERSION` 10).

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
- **No tanh on the forked output path (2026-09).** Tanh remains on `i2h`'s recurrent carrier,
  where it bounds values fed repeatedly through recurrence, but `i2o` reads the shared deep
  representation directly. Matched scratch panels found no learned-task benefit from output
  tanh: direct output modestly improved BPTT copy and substantially improved recurrence-clipped
  DFA recall when fast weights generated large features. See
  `docs/tapped_vs_forked_rnn_report.md`, section "Output activation after locking the fork,"
  and `scratch/output_activation_report.md`. `CHECKPOINT_CODE_VERSION` is 9.
- **Ephemeral + BPTT: fast weights are frozen within a sequence.** BPTT is the contrast to
  per-step backprop and DFA in the permutation grid above. Its only update comes after the
  last step (`train.py:267`), and `start_sequence_wipe()` zeroes the ephemeral entries at the
  start of the next sequence (`train.py:80`, `ephemeral_model.py:120-125`). Updates to ephemeral
  entries therefore never reach a training forward pass, and only slow weights and biases
  learn. As a result, `--plasticity` and `--forget_rate` do not affect ephemeral BPTT
  training. (The forget set is exactly `ephemeral_mask`, `ephemeral_model.py:83-99`.) This
  path also ignores `--ephemeral_update_clamp`, because it never calls `apply_update`, and it
  does not increment `training_instance`. Checked on a small model (under the old flag
  names): changing α, `--forget_rate` or the update clamp leaves a four-sequence BPTT loss
  trajectory bit-identical. Until `CHECKPOINT_CODE_VERSION` 12 it also ignored
  `--weight_clamp` (and the since-removed `--unit_norm_weights`). That clamp bounds the slow
  weights, which do learn, so it now applies after the SGD step, as SimpleRNN does under BPTT.
- **DFA omits the activation derivative f′ by default (to examine; `--dfa_fprime` adds it,
  2026-09-29).** Every non-output
  layer's DFA error is the output error projected straight through its feedback matrix,
  `projected = output_error @ feedback_weights` (`dfa_projected_error`,
  `ephemeral_model.py:21-27`), and that is used as is for the weight step (outer product with
  the input) and the bias step. It is never multiplied by the derivative of the layer's
  activation: gelu′ for both models' hidden layers and tanh′ for `i2h` in both. This differs
  from Nøkland's formulation (2016, "Direct Feedback Alignment Provides Learning in Deep
  Neural Networks"), where a hidden layer's update is δa_l = (B_l·e) ⊙ f′(a_l), with a_l the
  layer's pre-activation, e the output error, and δW_l = −δa_l·h_{l−1}ᵀ. Only the output
  layer takes e directly, as `i2o` does here. Jaden wants to examine whether the
  update *should* include f′. The SimpleRNN DFA baseline (`DFALinear`, 2026-09) omits it too,
  on purpose, so that the two models' DFA is the same computation. Any future change must be
  applied to both, most simply in the shared `dfa_*` helpers (`EphemeralLinear` already records
  each step's pre-activation as `out_traces`; `DFALinear` records only its input). It would
  change every DFA golden trace. **Update 2026-09-29:** `--dfa_fprime` implements exactly that,
  opt-in, in the shared helpers (see Updaters), so both variants can be compared; the default
  is unchanged until the comparison decides it.
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
includes `--ephemeral_update_clamp`, `--weight_clamp` and `--fast_weight_clamp`. (The class constructors used to
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
- `--optimizer`: `sgd` (default) or `adam`, for the `rnn` baseline under `backprop`/`bptt`
- `--plasticity`: Plasticity (learning-rate multiplier, alpha) of the ephemeral weights
- `--ephemeral_fraction`: Fraction of each hidden layer's weights that are ephemeral
- `--forget_rate`: Fraction of each ephemeral weight removed per step
- `--ephemeral_update_clamp` (ephemeral model) / `--grad_norm_clip` (`rnn` baseline): update clamp or gradient-norm clip
- `--weight_clamp`: clamping of the weights after each update
- `--layer_norm`: affine-free LayerNorm on each trunk layer's post-GELU activations (both models)
- `--resume` / `--resume_checkpoint PATH`: Resume from `latest_checkpoint.pth`, or from an explicit checkpoint
- `--batch_size`: Number of sequences processed together
- `--seed`: Seed Python, NumPy, Torch, dataset shuffling, and DataLoader sampling (unset = drawn from the OS; see below)
- `--deterministic`: Require deterministic Torch operations
- `--heldout_eval_every N`, `--heldout_batches K`: held-out, frozen-slow-weight recall every N iterations (0 = off; see "Held-out evaluation")

Resuming is opt-in: without `--resume` or `--resume_checkpoint`, training starts from
scratch even if `latest_checkpoint.pth` exists. `--resume` with no checkpoint present starts
from scratch; a missing explicit `--resume_checkpoint` is an error. Once a checkpoint is
chosen, any load failure aborts the run, including a mismatch in hidden size, layer count,
updater, model type, charset size, `--optimizer`, `--forget_rate`, `--dataset` or `--learning_rate`
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
| `--normalize` | `--unit_norm_weights` (since removed; see below) | |
| `--plast_learning_rate`, `--imprint_rate` | (removed) | Were unused; still accepted and ignored |
| `--unit_norm_weights`, `--normalize` | (removed 2026-09) | `false` is still accepted, with a note, and ignored; `true` is an error. For a normalization use `--layer_norm` |

`--unit_norm_weights` (the old `--normalize`) divided each sequence's whole `[out, in]` weight
slice by its L2 norm after every update. On a 1033-wide layer that leaves entries around 1e-3,
far below the fast-weight magnitudes the memory needs, so it removed the memory for a trivial
reason; 2 of the 4,701 archived 2025 runs used it (best final-character accuracy 0.75, near the
no-memory ceiling). It was removed rather than kept as a trap. A checkpoint whose config has it
false (every run since 2025) loads as before, and the key is dropped so the resume diff does not
show it; one with it true is refused by `load_checkpoint` (and so by `heldout.py`), since this
code cannot run that model. `CHECKPOINT_CODE_VERSION` stays 12: with it false, as every
checkpoint that can still load had it, the mechanics are unchanged.

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

### Speed: `--fused_update` (ephemeral + DFA)

Almost all of a DFA step's GPU time is spent in elementwise passes over each layer's
`[B, out, in]` per-sample weights: outer product, plasticity, clamp, add, forget. `--fused_update`
compiles each layer's step into one kernel with `torch.compile`, and it never materializes the
gradient. At the 3-palindrome benchmark size, end-to-end training goes from 48 to 68 it/s on an
A100 and from 72 to 81 on an H100 (both already without the old per-step host syncs), and
`train_batch` goes from 22.5 to 76 batches/s on an RTX A6000. Over 200k iterations, fused and
unfused runs agree on every logged metric (`benchmarks/README.md`).

The step is `dfa_layer_step` in `ephemeral_model.py`. It is composed from the same helpers
`apply_update` and `apply_forget_step` use, in the same order, so the update rule is written once.
- Run uncompiled, it is bit-identical to the unfused step. Compiled, it is the same math with
  different rounding (fused multiply-adds): about 2e-7 relative per step, and the gap stays
  below 5e-7 over 200 batches, because the sequence wipe resets the fast weights.
- Fused runs are deterministic run to run on the same hardware.
- Steps that log update norms take the unfused step. `--grad_norm_clip` works (closed-form
  rank-1 norms, equal to rounding).
- Only `--model_type ephemeral --updater dfa` accepts the flag. A GPU below compute capability
  7.0 (the P100, which Triton does not support) falls back to the unfused step, and
  `fused_update_active` records which path ran. `tests/test_fused_update.py` pins all of this.

Changing the update rule means editing a helper, which changes both paths. A new operation in
the step has to be added to `dfa_layer_step` too. The bit-identity test fails if the two
diverge.

### Held-out evaluation (`--heldout_eval_every`, `heldout.py`)

Off by default. It scores a DFA EphemeralRNN on episodes it did not train on, with its slow
weights frozen, so the score measures what the fast weights remember. Each episode starts as a
training sequence does (`start_sequence_wipe`, always the full wipe, even with `--wipe_every` > 1). At each step the model predicts and is scored,
and only then is the target revealed. The target writes the fast entries with the training
step itself: `EphemeralRNN.fast_only_dfa_step` runs `dfa_layer_step` with
`freeze_slow=True`. It uses the same projected errors, `--grad_norm_clip`,
`--ephemeral_update_clamp`, `--weight_clamp`, `--fast_weight_clamp` and forgetting as
training. `--output_tanh` and the input settings come through the same forward pass and
`utils.model_input`. Slow entries, biases and the output head `i2o` stay bit for bit, and
`--slow_weight_decay` does not act. `tests/test_heldout.py` pins the fast entries against
`train_batch`'s own step with slow updates undone: bit for bit, and to rounding with
`--grad_norm_clip`, since training clips the materialized gradient and the evaluator uses the
closed form. There are four protocols:

| Protocol | Fast writes |
| --- | --- |
| `observed` | Every target writes, as in training: teacher-forced writes during the answer |
| `strict` | None from the step that predicts the first recall target onward. The fast entries still forget every step |
| `no_fast` | No fast weights: they stay at their wiped zeros, leaving only the slow scaffold |
| `free_running` | Self-targets (efference copy). As `observed` up to `strict`'s boundary; from the step that predicts the first recall target on, the model's own argmax replaces the truth as the next input and as the write's target (error softmax − onehot(own)). Still scored against the true targets. `heldout.py --free_running_sample SEED` samples from the softmax instead |

Each protocol reports `metrics.IntervalMetrics` (`recall_acc`, `recall_acc_lag_<k>`,
`final_char_acc`, ...) under `heldout_<protocol>/`. It also reports `first_answer_acc`,
accuracy on each episode's first recall target, which no answer write can have helped.

- `--heldout_eval_every N` (0 = off) evaluates the first `--heldout_batches` (default 4)
  batches of the synthetic dataset's `validation` split every N iterations. The metrics are
  logged with the next print interval, so use a multiple of `--print_freq`. The model's state
  is restored afterwards and the evaluation draws no random numbers, so the training run is
  unchanged.
- A saved checkpoint: `python heldout.py --checkpoint PATH [--dataset NAME] [--protocols
  observed strict no_fast free_running] [--free_running_sample SEED] [--batches 0] [--json
  out.json]`. The default is the whole
  validation split.
- Only `--model_type ephemeral --updater dfa` is supported. `--layer_norm` works: it acts on
  activations, not weights, so the fast-only step is unchanged.
- The batch size is the model's (fast weights are `[B, out, in]`).
- The synthetic tasks are small (3-char palindromes have 399 distinct strings), so validation
  strings also occur in training. Held out means fresh fast state and frozen slow weights, not
  unseen strings.

### Feedback-loop traces (`--trace_loop_every`, `trace_replay.py`)

Observation only: nothing about training changes (tests check weights and losses bit for bit). Every
`--trace_loop_every N` iterations (a multiple of `--print_freq`; 0 = off) the batch's per-step,
within-sequence traces are recorded by `loop_trace.LoopTracer`: trunk activation norm, fast-weight
norm, the fast write norm, the per-step **loop gain** (write norm at step t over step t-1; sustained
above 1 means amplification), max logit and logit norm, per-step loss, and each layer's fast and
slow contribution to its pre-activation. Summaries (`trace/...`) join the interval metrics; the
arrays go to `<checkpoint_dir>/traces/trace_<iter>.pt`. `--checkpoint_keep_every N` (with
`--checkpoint_keep_max M`) keeps numbered checkpoints for the replay tool:

```bash
python trace_replay.py --checkpoints runs/control --batches 2 --out replay.pt   # every checkpoint_*.pth
python plots/loop_figures.py replay.pt --out figures/loop   # see the script's --help
```

It works with `--fused_update`, every `--slow_update_every` and `--fast_backward_per_forward` (the
fast write is pass 1 only; fast_delta spans all passes; slow_delta is zero between window ends). The collection interface and what the traces mean
are in the vault design note "ephemeral weights feedback-loop instrumentation design 2026-09-30".

### Tools for studying blowups (`--early_stop_window`, `--plasticity_schedule`, trace additions)

- `--early_stop_window N` (default 10): the run stops when the interval loss has been above 5 for N
  consecutive `--print_freq` intervals (10 intervals at `--print_freq 500` is 5k iterations, which cut
  off early blowups that recover on their own). `0` turns the loss stop off; a NaN or inf loss always stops.
- `--plasticity_schedule "ITER:VALUE,ITER:VALUE,..."` (default empty = off; ephemeral only): sets the
  plasticity alpha to VALUE from iteration ITER on, applied at the start of each iteration;
  `--plasticity` holds before the first entry (use `0:VALUE` to replace it). On resume the value in
  force at the resumed iteration is applied, and resuming with a different schedule is allowed. The
  active alpha is logged each interval as `plasticity`. This one changes training (everything else here
  is observation only). With `--fused_update` every new alpha recompiles the fused step, so the torch
  compile cache limit is raised to 64.
- Two more traces with `--trace_loop_every`: `h_sat` (fraction of the tanh state's units with
  |h| > 0.99) and `i2h_pre_norm`, computed from the i2h pre-activation (with `--enable_recurrence false`
  the state is not fed back, which is why `hidden_norm` is 0 there). `trace_replay.py` prints both.

### Key-value memory tasks (`kv_tasks.py`)

Associative recall as ordinary synthetic datasets (`--dataset kv_unique_4`, ...). An episode is
K key-value pairs (keys `a`-`j`, values `0`-`9`), optional distractors `.`, the query marker
`?`, a key and its value, which is the only recall target:

| Dataset | Example | Answer |
| --- | --- | --- |
| `kv_unique_<K>` | `c7a2h2e9?a2` | K distinct keys (Ba et al. 2016); values may repeat |
| `kv_reassign_<K>` | `d5d1j0d6?d6` | the queried key is assigned c ~ U{1..K} times; its latest value, which differs from the previous one |
| `..._d<D>` | `c7a2h2e9........?a2` | D distractors before `?` (lag + D) |

Generate with `python kv_tasks.py [names] [--seed 0]` (default: K = 2, 4, 8 in both modes;
1,000,000 / 5,000 / 20,000 rows, seeded per split). Besides `recall_acc` (chance 1/10), the
metrics classify each answer as `kv_correct`, `kv_stale` (an earlier value of the queried key),
`kv_wrong_key` (a value bound to another key in the episode) or `kv_other`, in training and
under each held-out protocol.

### Advanced Features

- **Positional Encoding**: Add positional information with `--positional_encoding_dim N`
- **Residual Connections**: Enable/disable with `--residual_connection True/False`
- **Layer Normalization**: Enable/disable with `--layer_norm True/False` (trunk activations)
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
- `synth_datasets.py`, `kv_tasks.py`: synthetic dataset generators (`kv_tasks.py`: key-value tasks)
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
