# Memory Encoding with EphemeralRNN

This repository implements and compares different weight update mechanisms for recurrent neural networks, with a focus on the EphemeralRNN architecture that features high-plasticity weights for short-term memory.

## Overview

The project explores three different training approaches:
1. **DFA (Direct Feedback Alignment)** - Uses fixed random feedback weights for error propagation
2. **Backprop** - True gradients with an update after every step, and no gradient through time (hidden state detached)
3. **BPTT (Backpropagation Through Time)** - Accumulates the loss across the whole sequence and updates once at the end

## Key Features

### EphemeralRNN Architecture
- **High-Plasticity Weights**: A subset of weights that can adapt rapidly for short-term memory
- **Per-Batch Adaptation**: Each sequence in a batch has independent weight adaptations
- **Forgetting Mechanism**: Controlled decay of high-plasticity weights between updates

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

Each EphemeralLinear layer holds per-sequence `candidate_weights` of shape
`[batch, out, in]`, starting at zero. A fixed random mask marks `--plast_proportion` of each
layer's entries as ephemeral, with plasticity α = `--plast_clip`. The rest are slow weights
with plasticity 1, and the output layers (`i2o`, `self_grad`) have plasticity 1 everywhere.
At the start of every sequence, `wipe()` replaces each sequence's copy with the batch mean
and zeroes the ephemeral entries in every layer. Forgetting multiplies the ephemeral
entries by `1 - forget_rate`.

| `--updater` | `--model_type ephemeral` | `--model_type rnn` (SimpleRNN baseline) |
| --- | --- | --- |
| `dfa` | Every step: DFA gradients, forget, `apply_unified_updates` | No parameter changes: the DFA branch only updates an EphemeralRNN |
| `backprop` | Every step: `backward()` on the batch-mean step loss, forget, `scale_gradients`, `apply_unified_updates` | Every step: `torch.optim.SGD` on the batch-mean step loss; `--grad_clip` is a global grad-norm clip |
| `bptt` | After the last step: `backward()` on the batch-mean summed loss, forget, `scale_gradients`, plain `p -= lr * p.grad` | After the last step: `torch.optim.SGD`; `--grad_clip` is a global grad-norm clip |

For the ephemeral model (`g` is the gradient of one sequence's own loss, `B` is `--batch_size`):

| | DFA | Backprop | BPTT |
| --- | --- | --- | --- |
| Error signal | Per-sequence output error; hidden layers receive it through fixed random `feedback_weights` | Autograd | Autograd |
| Hidden state | Detached every step | Detached every step | Not detached |
| When weights change | Every step | Every step | Once, after the last step |
| Order per update | Forget, then update | Forget, then update | Forget, then update (once) |
| Step on an ephemeral weight | `lr·α·g` | `lr·α²·g/B` | `lr·α·g/B`, zeroed by the next `wipe()` |
| Step on a slow weight | `lr·g` | `lr·g/B` | `lr·g/B` |
| Layers that change | Hidden layers, `i2o`, `self_grad` | Hidden layers, `i2o` | Every parameter with a gradient |
| `--grad_clip` | Element-wise clamp on α-scaled ephemeral updates | Same as DFA | Ignored |
| `--clip_weights`, `--normalize` | Applied after each update | Applied after each update | Ignored |

The entries that differ between columns are explained, with evidence, in the next section.

## Known issues / behaviours under review

These describe the current code. They are recorded here, not changed, until they can be
re-examined with full training runs before and after. Line numbers were last checked when
the forget-rate terminology table (next section) was added.

- **Backprop applies α twice (α²) on ephemeral weights.** The backprop branch calls
  `rnn.scale_gradients(plast_clip)` (`train.py:183`), which multiplies masked gradients by
  α (`ephemeral_model.py:253-259`). `apply_unified_updates` then multiplies by
  `plasticity`, which is α on the mask (`ephemeral_model.py:49`, `:156`). DFA does not call
  `scale_gradients`, so it applies α once. BPTT calls it and then takes a plain SGD step
  (`train.py:250-261`), so it also applies α once. Measured with α = 7, the masked step is
  49·lr·g under backprop and 7·lr·g under DFA; slow weights get 1·lr·g under both.
- **Backprop and BPTT carry a 1/B factor that DFA does not.** Backprop calls `backward()`
  on `step_loss.mean()` over the batch (`train.py:176`), and BPTT on
  `accumulated_loss.mean()` (`train.py:243`), so each sequence's candidate-weight gradient
  is 1/B of its own loss gradient. DFA takes each sequence's own error, using
  `grad_outputs=ones` on the unreduced loss (`train.py:134`; the reduction is forced to
  `'none'` at `train.py:300-302`). This was a deliberate choice at the time, and it helped
  the loss numbers. Combined with α², backprop's per-sequence ephemeral step is α/B times
  DFA's at the same `--learning_rate` and `--plast_clip`.
- **`i2h` gets no gradient under DFA or backprop (by design; under review).** The DFA
  branch never populates or updates `i2h` (`train.py:145-157`). Under backprop, the hidden
  state is detached every step (`train.py:91-92`), and `i2h`'s output feeds only the next
  step (`ephemeral_model.py:399-406`). So `i2h.candidate_weights.grad` is `None`, and
  `apply_unified_updates` returns immediately (`ephemeral_model.py:144-145`). This is
  probably intended. The ephemeral weights are meant to replace the recurrent connection
  as the short-term memory (paper Fig. 1 caption, `paper/paper_content.tex:117`), while the
  slow weights keep learning as usual. `i2h` is likely vestigial in the ephemeral model,
  kept for parity with the SimpleRNN baseline, which can use recurrence through BPTT. (The
  baseline's `i2h` also gets no gradient under backprop, for the same detach reason.)
  Consequence: `i2h`'s candidate weights stay at their initial zeros, so under DFA or
  backprop the hidden state is `tanh(i2h.bias)` with `--enable_recurrence True` and zero
  with it off (`ephemeral_model.py:400-406`). It is constant either way, so recurrence
  contributes no information.
- **Ephemeral + BPTT: fast weights are frozen within a sequence.** BPTT is the contrast to
  per-step backprop and DFA in the permutation grid above. Its only update comes after the
  last step (`train.py:240`), and `wipe()` zeroes the ephemeral entries at the start of the
  next sequence (`train.py:80`, `ephemeral_model.py:78-83`). Updates to ephemeral entries
  therefore never reach a training forward pass, and only slow weights and biases learn. As
  a result, `--plast_clip` and `--forget_rate` do not affect ephemeral BPTT training. (The
  forget set lies inside the mask when `--plast_proportion` ≥ 0.01,
  `ephemeral_model.py:47-58`.) This path also ignores `--grad_clip`, `--clip_weights` and
  `--normalize`, because it never calls `apply_unified_updates` or `_apply_regularization`
  (`train.py:257-261`), and it does not increment `training_instance`. Checked on a small
  model: changing `--plast_clip`, `--forget_rate`, `--grad_clip` or `--clip_weights` leaves
  a four-sequence BPTT loss trajectory bit-identical.
- **`--normalize` also rescales plasticity and forgetting.** `_apply_regularization`
  divides every float parameter of the layer by its L2 norm after each update
  (`ephemeral_model.py:216-225`): the candidate weights, but also `plasticity`,
  `forgetting_factor`, the bias, the feedback weights and the traces. After the first
  update the ephemeral α and forget rate are no longer `--plast_clip` and `--forget_rate`
  (in the golden trace, α 3.0 becomes about 0.11). The logged high/low-plasticity update
  norms are scalar parameters too, so they are rescaled to about 1 before they are logged.
  Layers whose update returns early (`i2h` under backprop) are not rescaled.
  `--clip_weights` is applied after the normalization, so a clip of 1 or more never binds
  when `--normalize` is on.
- **`EphemeralLinear._update_bias` is dead code with a flipped sign.** Nothing calls it,
  and it adds `+lr·projected_error` (`ephemeral_model.py:207-214`). The live bias update is
  `_update_bias_from_grad`, which subtracts (`ephemeral_model.py:176-190`).
- **Forget-step ordering differs from the paper.** The paper multiplies each ephemeral
  weight by its coefficient 0.7 after each update (`paper/paper_content.tex:124-127`). The
  code multiplies by `1 - forget_rate` before the update, in all three updaters
  (`train.py:151`, `:180`, `:247`; `ephemeral_model.py:237`). (The class constructors used
  to default to `forget_rate=0.7`, a leftover of the paper's coefficient that would have
  kept only 0.3 of each weight. They now default to 0.01, matching the CLI; `train.py`
  always passed `--forget_rate` explicitly, so no run changed.)

## Paper settings & stability

The paper trains with plain SGD at a base learning rate of 1e-4
(`paper/paper_content.tex:135`), ephemeral plasticity α = 1e4 or 1e5 (`:104`), and a
"forgetting rate coefficient" of 0.7 applied after each update (`:124-127`). In code terms
that is `--forget_rate 0.3`, applied before the update as noted above. The current CLI
defaults are lr 1e-4, α 1e5 and `--forget_rate` 0.01, which keeps 1 − forget_rate = 0.99 of
each ephemeral weight per step (`train.py:313-317`).

### Terminology: paper vs code

The code's convention is the one to use: `forget_rate` is the fraction of each ephemeral
weight removed per step, and 1 − forget_rate is the fraction kept. The paper's text
reports the fraction kept; its figure legends use code `--forget_rate` values (the Fig. 1
key-recall legend "ephemeral 0.0001 0.5" is lr 1e-4, `--forget_rate 0.5`).

| Paper term | Code / CLI name | Meaning | Formula | Conversion |
| --- | --- | --- | --- | --- |
| "Forgetting rate coefficient" in the text (`paper_content.tex:127`); forget rate in the figure legends | `--forget_rate`; config and W&B key `forget_rate`; `FORGET_RATE` in the run scripts; `forget_rate=` in the `EphemeralRNN`/`EphemeralLinear` constructors | Fraction of each ephemeral weight removed per step | `w ← (1 − forget_rate)·w` | The text's coefficient is 1 − `forget_rate`: its "forgetting rate coefficient 0.7" is `--forget_rate 0.3`. Legend values are already `--forget_rate` values |
| (none) | `EphemeralLinear.forgetting_factor` (state-dict tensor) | Per-entry forget rate: `forget_rate` on the ephemeral mask, 0 elsewhere. A removal fraction, not a multiplier | `w ← (1 − forgetting_factor)·w`, element-wise (`apply_forget_step`) | As above, per entry |
| Plasticity α_k (`:100-107`) | `--plast_clip`; `plasticity` tensor | Learning-rate multiplier on ephemeral weights (1 on slow weights) | step `lr·α·g` (DFA) | `--plast_clip` = α |

Ordering: the paper applies the decay after each update (`:124-127`). The code forgets
before the update in all three updaters (`train.py:151`, `:180`, `:247`), so one step is
`w ← (1 − forget_rate)·w − lr·α·g` rather than the paper's
`w ← (1 − forget_rate)·(w − lr·α·g)`. The two
differ only in whether the newest update is decayed once before the next prediction.

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
- `--plast_clip`: Plasticity (learning-rate multiplier, alpha) of the ephemeral weights
- `--plast_proportion`: Proportion of weights that are high-plasticity
- `--forget_rate`: Fraction of each ephemeral weight removed per step
- `--resume` / `--resume_checkpoint PATH`: Resume from `latest_checkpoint.pth`, or from an explicit checkpoint
- `--batch_size`: Number of sequences processed together
- `--seed`: Seed Python, NumPy, Torch, dataset shuffling, and DataLoader sampling (unset = drawn from the OS; see below)
- `--deterministic`: Require deterministic Torch operations

Resuming is opt-in: without `--resume` or `--resume_checkpoint`, training starts from
scratch even if `latest_checkpoint.pth` exists. `--resume` with no checkpoint present starts
from scratch; a missing explicit `--resume_checkpoint` is an error. Once a checkpoint is
chosen, any load failure aborts the run, including a mismatch in hidden size, layer count,
updater, model type or charset size (`utils.py`, `load_checkpoint`). The seed and `--deterministic`
come from the checkpoint; passing a different value is an error.

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
this change with `seed: None` resume unseeded, with a warning. A job preempted before its first
checkpoint starts fresh with a new seed.

To rerun an experiment exactly, start fresh with `--seed <logged seed>` (plus `--deterministic True`
for bitwise-deterministic Torch ops; on GPU this sets `CUBLAS_WORKSPACE_CONFIG`).

### SLURM time limits

The sbatch scripts request `#SBATCH --signal=B:USR1@600` and launch training through
`forward_signals` (from `sweeps/forward_signals.sh`). Ten minutes before the wall-time limit,
`train.py` stops at the next iteration, saves `latest_checkpoint.pth` (if `--checkpoint_save_freq > 0`),
records `end_reason: time_limit` in W&B, and exits with code 124; resume with `--resume`. A SIGTERM
stops the same way with `end_reason: terminated` and exit code 143.

### Advanced Features

- **Positional Encoding**: Add positional information with `--positional_encoding_dim N`
- **Residual Connections**: Enable/disable with `--residual_connection True/False`
- **Weight Normalization**: Enable/disable with `--normalize True/False`
- **Input Modes**: Choose between `--input_mode last_one` or `--input_mode last_two`

## Testing

Run the network-free assertion suite from the repository root:

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest tests/ -q
```

`pytest` is not in `environment.yml`; install it with `pip install pytest`.

The suite includes fixed-input golden traces through the real `train.train()`
path for DFA, backprop, and BPTT, a finite-update smoke test for all three
(`tests/test_smoke_updaters.py`), and checkpoint/failure-path, metrics and
reproducibility tests. See [tests/README.md](tests/README.md) for what the
golden traces pin, which known behaviours they currently freeze, and how to
regenerate them.

## Project Structure

- `train.py`: Main training script with unified training loop
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
