# self_grad (removed 2026-09-23)

A record of an abandoned experiment, removed so the codebase stays focused. Recover the code
with `git show f2147ee:ephemeral_model.py` / `git show f2147ee:train.py`.

## What it was
- A second output head on `EphemeralRNN`: `self.self_grad`, an `EphemeralLinear`
  (`hidden_size → vocab`, last layer, no ephemeral entries), the same shape as `i2o`, reading
  `h_t` after the Elman change (`combined` before).
- `forward` returned `(output, next_hidden, self_grad)`; SimpleRNN returned `None` in that slot.
- Under DFA only, with `--self_grad S > 0`, `train.py` added `clamp(self_grad_output, -S, S)` **in
  place** to `output_error` before any layer was populated, so every layer (and the head itself)
  learned from `∂loss/∂output + clamp(self_grad_output)`. The head was also trained by DFA on
  that same error. Under backprop/BPTT it was built but never trained.
- Default `--self_grad 0`: the head was built, updated by DFA, wiped and decayed, but never
  influenced any other layer. Removing it left every golden trace identical (only its own
  entries disappeared).

## Why it existed
Introduced in 557b82d (2025-02-03, "reset parametric memory") as `reward_update += self_grad * 1e-5`,
then parametrised as `--self_grad` in 5e52e07 (2025-02-19, "try self_grad again, parametrize
it"). The idea: a learned, gradient-shaped signal the network emits about itself, as a
"grad based replacement for recurrence" (the old `--help` text), letting the model inject its
own error term alongside the true output error. It was open-ended exploration without a clear
learning objective for the head (it was trained on the very error it perturbed), and no run
showed a benefit.

## If revisiting
- Give the head a defined target (e.g. predict the next step's error, or a synthetic-gradient
  style target, cf. Jaderberg et al. 2017 "Decoupled Neural Interfaces using Synthetic
  Gradients"), rather than training it on its own perturbed error.
- Add it as a separate tensor, not an in-place add to `output_error` (the in-place add is why
  `tests/test_dfa_error_signals.py` had to pin aliasing).
- Removal bumped `CHECKPOINT_CODE_VERSION` 5 → 6.
