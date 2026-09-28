"""Prequential held-out evaluation for the DFA EphemeralRNN.

The evaluator freezes all slow entries, shared biases, feedback matrices, and the slow-only
output head. Only masked ephemeral entries can be written or forgotten. It deliberately does
not call the training update path: that path also updates slow state and applies whole-weight
regularization. Because ``unit_norm_weights`` would rescale frozen slow entries, models configured
to use it are rejected rather than evaluated with changed update semantics. ``weight_clamp`` is
applied only to fast entries after a permitted write.

Each step is ordered: selected-row reset, prediction, scoring, target reveal/write, forgetting.
There is currently no separate validity mask, so every batch/step position is a valid passage
of time and forgets even when ``update_mask`` is false.

By default evaluation starts a fresh held-out sequence: existing row-specific slow copies are
consolidated with the training-time sequence-start operation and all fast state is wiped. Explicit
``initial_state='continue'`` instead preserves every weight copy and requires recurrent hidden
state from the preceding chunk.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ephemeral_model import EphemeralRNN, dfa_per_sample_gradient, dfa_projected_error, ephemeral_update


TARGET_SUM_ATOL = 1e-6


@dataclass(frozen=True)
class HeldOutBatch:
    """A dense held-out stream with independent inputs and probability targets.

    Every target row must be finite, non-negative, and sum to one within absolute tolerance
    ``TARGET_SUM_ATOL`` (1e-6, with zero relative tolerance). This guarantees that
    ``softmax(logits) - target`` is the cross-entropy gradient used for DFA writes.
    """

    inputs: torch.Tensor
    targets: torch.Tensor
    score_mask: torch.Tensor
    update_mask: torch.Tensor
    reset_mask: torch.Tensor

    def to(self, device=None, dtype=None):
        """Return a moved copy; only floating inputs/targets adopt ``dtype``.

        Boolean masks remain boolean. ``dtype``, when supplied, must be a floating Torch dtype.
        """
        if dtype is not None and not torch.empty((), dtype=dtype).is_floating_point():
            raise ValueError("HeldOutBatch dtype must be a floating point dtype")
        floating = {"device": device}
        if dtype is not None:
            floating["dtype"] = dtype
        return HeldOutBatch(
            self.inputs.to(**floating), self.targets.to(**floating),
            self.score_mask.to(device=device), self.update_mask.to(device=device),
            self.reset_mask.to(device=device))

    def __post_init__(self):
        if self.inputs.ndim != 3:
            raise ValueError(f"inputs must have shape [batch, steps, input], got {tuple(self.inputs.shape)}")
        if self.targets.ndim != 3:
            raise ValueError(f"targets must be one-hot/probabilities [batch, steps, output], got {tuple(self.targets.shape)}")
        prefix = self.inputs.shape[:2]
        if self.targets.shape[:2] != prefix:
            raise ValueError("inputs and targets must have the same batch and step dimensions")
        for name in ("score_mask", "update_mask", "reset_mask"):
            mask = getattr(self, name)
            if mask.dtype != torch.bool or mask.shape != prefix:
                raise ValueError(f"{name} must be boolean with shape {tuple(prefix)}")
        tensors = (self.targets, self.score_mask, self.update_mask, self.reset_mask)
        if any(t.device != self.inputs.device for t in tensors):
            raise ValueError("all held-out batch tensors must be on the same device")
        if not self.inputs.is_floating_point() or not self.targets.is_floating_point():
            raise ValueError("inputs and targets must be floating point tensors")
        if not torch.isfinite(self.inputs).all():
            raise ValueError("inputs must contain only finite values")
        if not torch.isfinite(self.targets).all():
            raise ValueError("targets must contain only finite values")
        if (self.targets < 0).any():
            raise ValueError("targets must be non-negative probabilities")
        row_sums = self.targets.sum(dim=2)
        if not torch.isclose(row_sums, torch.ones_like(row_sums), rtol=0,
                             atol=TARGET_SUM_ATOL).all():
            raise ValueError(f"every target row must sum to 1 within atol={TARGET_SUM_ATOL}")


@dataclass(frozen=True)
class HeldOutResult:
    """Detached step outputs plus aggregates over exactly ``score_mask``."""

    logits: torch.Tensor       # [B, T, output]
    predictions: torch.Tensor  # [B, T], class indices
    losses: torch.Tensor       # [B, T], irrespective of score_mask
    scored_loss: float
    scored_accuracy: float
    scored_count: int
    final_hidden: torch.Tensor  # [B, hidden], for explicit continuation into another chunk


def _fast_dfa_write(model, output_error, rows, learning_rate, update_clamp):
    """Apply one target-driven DFA write to fast entries of selected rows only."""
    if not rows.any():
        return
    for layer in model.trained_layers():
        # i2o has no fast entries. Skipping it is also what freezes its weights and bias.
        if not layer.ephemeral_mask.any():
            continue
        projected = dfa_projected_error(output_error, layer.feedback_weights, False)
        gradient = dfa_per_sample_gradient(projected, layer.in_traces.data)
        update = ephemeral_update(gradient, layer.plasticity, layer.ephemeral_mask,
                                  update_clamp, False)
        selected = rows[:, None, None] & layer.ephemeral_mask[None, :, :]
        layer.per_sample_weights.data[selected] += learning_rate * update[selected]
        # Element-wise clipping can safely be restricted to fast state. Models using whole-matrix
        # unit normalization are rejected by evaluate_held_out before reaching this update.
        if layer.weight_clamp:
            layer.per_sample_weights.data[selected] = layer.per_sample_weights.data[selected].clamp(
                -layer.weight_clamp, layer.weight_clamp)


def _forget_fast(model):
    """Advance time for all rows while keeping slow state bitwise frozen."""
    for layer in model.trained_layers():
        if layer.ephemeral_mask.any():
            fast = layer.ephemeral_mask.unsqueeze(0).expand_as(layer.per_sample_weights)
            layer.per_sample_weights.data[fast] *= 1 - layer.forget_rate


@torch.no_grad()
def evaluate_held_out(model: EphemeralRNN, batch: HeldOutBatch, learning_rate: float,
                      update_clamp: float = 0.0, initial_state: str = "fresh",
                      initial_hidden: torch.Tensor | None = None) -> HeldOutResult:
    """Run an adaptive held-out stream with frozen slow state and DFA fast writes.

    The model must have been built with ``updater='dfa'`` and the same fixed batch size as
    ``batch``. Targets are consumed directly, never reconstructed by shifting inputs.

    ``initial_state='fresh'`` (the default) applies ``start_sequence_wipe()`` once before step 0
    and starts recurrent state at zero; supplying ``initial_hidden`` is an error.
    ``initial_state='continue'`` preserves all existing fast/slow row state and requires an
    explicit ``initial_hidden``. In either mode, ``reset_mask`` remains a per-step, per-row reset.
    """
    if not isinstance(model, EphemeralRNN) or model.updater != "dfa":
        raise ValueError("held-out fast-memory evaluation requires an EphemeralRNN with updater='dfa'")
    if any(layer.unit_norm_weights for layer in model.trained_layers()):
        raise ValueError("held-out fast-memory evaluation does not support unit_norm_weights=True: "
                         "whole-matrix normalization would change frozen slow entries")
    if learning_rate < 0 or update_clamp < 0:
        raise ValueError("learning_rate and update_clamp must be non-negative")
    if initial_state not in ("fresh", "continue"):
        raise ValueError("initial_state must be 'fresh' or 'continue'")
    batch_size, steps, input_size = batch.inputs.shape
    if steps == 0:
        raise ValueError("held-out batches must contain at least one step")
    if batch_size != model.batch_size:
        raise ValueError(f"batch size {batch_size} does not match model batch size {model.batch_size}")
    expected_input = model.linear_layers[0].in_features - model.hidden_size
    if input_size != expected_input:
        raise ValueError(f"input width {input_size} does not match model input size {expected_input}")
    if batch.targets.shape[2] != model.i2o.out_features:
        raise ValueError("target width does not match model output size")
    if batch.inputs.device != next(model.parameters()).device:
        raise ValueError("held-out batch and model must be on the same device")
    model_dtype = next(model.parameters()).dtype
    if batch.inputs.dtype != model_dtype or batch.targets.dtype != model_dtype:
        raise ValueError(f"inputs and targets must have the model dtype ({model_dtype})")
    if initial_state == "fresh":
        if initial_hidden is not None:
            raise ValueError("initial_hidden must not be supplied when initial_state='fresh'")
    else:
        if initial_hidden is None:
            raise ValueError("initial_hidden is required when initial_state='continue'")
        if initial_hidden.shape != (batch_size, model.hidden_size):
            raise ValueError(f"initial_hidden must have shape {(batch_size, model.hidden_size)}")
        if initial_hidden.device != batch.inputs.device:
            raise ValueError("initial_hidden and held-out batch must be on the same device")
        if initial_hidden.dtype != model_dtype:
            raise ValueError(f"initial_hidden must have the model dtype ({model_dtype})")
        if not torch.isfinite(initial_hidden).all():
            raise ValueError("initial_hidden must contain only finite values")

    was_training = model.training
    model.eval()
    try:
        if initial_state == "fresh":
            model.start_sequence_wipe()
            hidden = model.initHidden(batch_size)
        else:
            hidden = initial_hidden.detach().clone()
        logits, predictions, losses = [], [], []
        for step in range(steps):
            resets = batch.reset_mask[:, step]
            model.reset_ephemeral_rows(resets)
            hidden[resets] = 0

            # Prediction is detached and recorded before this step's target is used anywhere.
            output, hidden = model(batch.inputs[:, step], hidden)
            target = batch.targets[:, step]
            step_logits = output.detach().clone()
            step_loss = -(target * F.log_softmax(output, dim=1)).sum(dim=1)
            logits.append(step_logits)
            predictions.append(step_logits.argmax(dim=1))
            losses.append(step_loss.detach().clone())

            rows = batch.update_mask[:, step]
            _fast_dfa_write(model, F.softmax(output, dim=1) - target, rows,
                            learning_rate, update_clamp)
            _forget_fast(model)

        logits = torch.stack(logits, dim=1)
        predictions = torch.stack(predictions, dim=1)
        losses = torch.stack(losses, dim=1)
        scored = batch.score_mask
        count = int(scored.sum().item())
        if count:
            scored_loss = losses[scored].mean().item()
            truth = batch.targets.argmax(dim=2)
            scored_accuracy = (predictions[scored] == truth[scored]).float().mean().item()
        else:
            scored_loss = scored_accuracy = float("nan")
        return HeldOutResult(logits, predictions, losses, scored_loss, scored_accuracy, count,
                             hidden.detach().clone())
    finally:
        model.train(was_training)


# A descriptive alias for callers that name the operation after the adapted memory.
evaluate_fast_memory = evaluate_held_out
