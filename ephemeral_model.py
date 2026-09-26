import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
from utils import initialize_charset
import numpy as np
import torch.nn.utils.parametrize as parametrize
# from memory_profiler import profile


# DFA pieces shared by EphemeralLinear (the ephemeral model) and DFALinear (the SimpleRNN
# baseline), so the two models' DFA stays the same computation. Note: neither multiplies the
# projected error by the layer's activation derivative; see README "Known issues".

def init_feedback_weights(vocab_size, out_features):
    """A layer's fixed random DFA feedback matrix B, [vocab, out]: xavier_normal_."""
    return torch.nn.init.xavier_normal_(torch.empty(vocab_size, out_features))


def dfa_projected_error(error_signal, feedback_weights, is_last_layer):
    """The error a layer's DFA update uses, [B, out]: the output error itself for a last layer
    (i2o), else error_signal @ feedback_weights. error_signal is train.py's
    output_error, [B, vocab]; it is never modified, and a last layer gets that same object."""
    if is_last_layer:
        return error_signal
    return error_signal @ feedback_weights


def dfa_per_sample_gradient(projected_error, input):
    """Per-sequence DFA gradient, [B, out, in]: the outer product of each sequence's projected
    error [B, out] with its layer input (the input trace) [B, in]."""
    out = projected_error.unsqueeze(2)  # [batch_size, out_features, 1]
    return out * input.unsqueeze(1)  # [batch_size, 1, in_features] -> [batch_size, out_features, in_features]


def dfa_bias_update(projected_error, learning_rate):
    """DFA bias step, [out]: -learning_rate times the batch mean of the projected error."""
    bias_update = -learning_rate * projected_error.mean(dim=0)
    if len(bias_update.shape) > 1:
        bias_update = bias_update.mean(dim=0)
    return bias_update


def recurrent_trunk_size(input_size, hidden_size):
    """Width shared by the recurrent models' concatenated input/state trunk."""
    return input_size + hidden_size


def regularized_weight(weight, unit_norm_weights, weight_clamp, norm_dims):
    """Return weights normalized over norm_dims, then element-wise clamped; biases are excluded."""
    if unit_norm_weights:
        norms = torch.linalg.vector_norm(weight, ord=2, dim=norm_dims, keepdim=True)
        weight = weight / (norms + 1e-6)
    if weight_clamp != 0:
        weight.clamp_(-weight_clamp, weight_clamp)
    return weight


def per_sequence_clip_scale(norms, max_norm):
    """--grad_norm_clip's factor for each sequence, [B]: min(1, max_norm / (norm + 1e-6)), the
    same coefficient torch.nn.utils.clip_grad_norm_ uses. It is exactly 1 where the clip does
    not bind, so a threshold that never binds leaves every gradient bit-identical."""
    return (max_norm / (norms + 1e-6)).clamp(max=1.0)


class GradNormClipStats:
    """--grad_norm_clip statistics over a print interval: the mean and max pre-clip gradient norm
    and the fraction of norms that exceeded the threshold. The ephemeral model records one norm
    per sequence per update, SimpleRNN one per update. Accumulated on the device, so recording
    adds no host sync; summary() syncs once and resets."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.norm_sum = self.norm_max = self.clipped = None
        self.count = 0

    def record(self, norms, max_norm):
        norms = norms.detach().reshape(-1)
        clipped = (norms > max_norm).sum()
        if self.count == 0:
            self.norm_sum, self.norm_max, self.clipped = norms.sum(), norms.max(), clipped
        else:
            self.norm_sum = self.norm_sum + norms.sum()
            self.norm_max = torch.maximum(self.norm_max, norms.max())
            self.clipped = self.clipped + clipped
        self.count += norms.numel()

    def summary(self):
        if self.count == 0:
            return {}
        result = {"grad_norm_mean": self.norm_sum.item() / self.count,
                  "grad_norm_max": self.norm_max.item(),
                  "grad_norm_clip_fraction": self.clipped.item() / self.count}
        self.reset()
        return result


class EphemeralLinear(nn.Linear):
    def __init__(self, in_features, out_features, charset, bias=True, unit_norm_weights=True, weight_clamp=0, updater='dfa', requires_grad=False, is_last_layer=False, plasticity=1, batch_size=1, forget_rate=0.01, ephemeral_fraction=0.2):
        """forget_rate: fraction of each ephemeral weight removed per forget step,
        w <- (1 - forget_rate) * w (see apply_forget_step). The paper's "forgetting rate
        coefficient 0.7" is 1 - forget_rate, i.e. forget_rate = 0.3. Same meaning as --forget_rate.
        plasticity: alpha, the learning-rate multiplier on the ephemeral entries (--plasticity).
        ephemeral_fraction: fraction of entries that are ephemeral (--ephemeral_fraction).
        unit_norm_weights, weight_clamp: applied after each update (--unit_norm_weights,
        --weight_clamp; see _apply_regularization)."""
        super(EphemeralLinear, self).__init__(in_features, out_features, bias)

        # Set requires_grad for the base class parameters
        self.weight.requires_grad = False # Base weights are not trained directly
        if bias:
            # For backprop/bptt, bias needs requires_grad=True so PyTorch computes bias.grad
            # For DFA, we set it to False since we handle bias manually
            self.bias.requires_grad = (updater in ['backprop', 'bptt'])

        self.unit_norm_weights = unit_norm_weights
        self.weight_clamp = weight_clamp
        self.updater = updater
        self.is_last_layer = is_last_layer
        self.in_traces = nn.Parameter(torch.zeros(batch_size, in_features), requires_grad=requires_grad)
        self.out_traces = nn.Parameter(torch.zeros(batch_size, out_features), requires_grad=requires_grad)

        self.last_ephemeral_step_norm = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.last_slow_step_norm = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.register_buffer('t', torch.tensor(1.0))
        self.batch_size = batch_size
        # Backprop and BPTT reduce the bias gradient over the batch, so --grad_norm_clip keeps
        # each forward's output to read the per-sequence shares from (see sequence_bias_grads).
        self.retain_sequence_bias_grads = False
        self._retained_outputs = []


        # Initialize weights with the adjusted gain
        self.feedback_weights = nn.Parameter(init_feedback_weights(len(charset), out_features), requires_grad=requires_grad)
        # self.weight.data = torch.nn.init.xavier_uniform_(torch.empty(out_features, in_features), gain=gain)

        # per_sample_weights hold one copy of the layer's weights per sequence in the batch,
        # with both the ephemeral and the slow entries. They are filled after the mask is drawn.
        # They require gradients only if we are using the backprop or bptt updater.
        self.per_sample_weights = nn.Parameter(torch.zeros(self.batch_size, out_features, in_features), requires_grad=(updater in ['backprop', 'bptt']))
        distribution = torch.ones_like(self.weight)
        rand_vals = torch.rand_like(self.weight)
        # The mask marks the ephemeral entries. Last layers (i2o) have none, so
        # nothing there is decayed, wiped or treated as ephemeral. rand_vals is drawn for
        # every layer anyway, which keeps the RNG stream the same for the layers after it.
        if self.is_last_layer:
            ephemeral = torch.zeros_like(self.weight, dtype=torch.bool)
        else:
            ephemeral = rand_vals < ephemeral_fraction
        self.ephemeral_mask = nn.Parameter(ephemeral, requires_grad=False)
        # Reuse nn.Linear's already-drawn default initialization for slow weights, without
        # consuming more RNG. Fast weights intentionally begin at zero and are wiped there at
        # the start of every sequence.
        with torch.no_grad():
            initial_weights = self.weight.unsqueeze(0).expand_as(self.per_sample_weights)
            self.per_sample_weights.copy_(initial_weights)
            self.per_sample_weights.masked_fill_(self.ephemeral_mask.unsqueeze(0), 0)
        distribution[self.ephemeral_mask] = plasticity

        # A plain attribute, not state: the per-entry forget rate is forget_rate on the ephemeral
        # mask and 0 elsewhere, computed in apply_forget_step. (Checkpoints from before the
        # rename stored it as the tensor forgetting_factor; utils.load_checkpoint checks and drops it.)
        self.forget_rate = forget_rate

        # Initialize plasticity parameters with the generated values
        if self.is_last_layer == False:
            self.plasticity = nn.Parameter(distribution, requires_grad=requires_grad)
        else:
            self.plasticity = nn.Parameter(torch.ones_like(self.weight), requires_grad=requires_grad)
        print(f"Number of non-zero values in self.plasticity: {torch.count_nonzero(self.plasticity).item()}")

        self.plasticity_feedback_weights = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(len(charset), out_features)), requires_grad=requires_grad)

    def start_sequence_wipe(self):
        """Start of a sequence: set every sequence's per_sample_weights to the batch mean, then
        zero the ephemeral entries (also in the unused base weight) and reset the time counter."""
        # Suppose per_sample_weights is of shape [B, out_features, in_features]
        # Aggregate across the batch (e.g., average) to get a single copy:
        aggregated = self.per_sample_weights.mean(dim=0, keepdim=True)
        # Then set every sequence's copy in the batch to this aggregated value:
        self.per_sample_weights.data.copy_(aggregated.expand_as(self.per_sample_weights))

        # Apply the mask
        # masked_fill_, not boolean indexing: the same values without a host sync.
        self.weight.data.masked_fill_(self.ephemeral_mask, 0)
        self.per_sample_weights.data.masked_fill_(self.ephemeral_mask.unsqueeze(0), 0)
        # Reset the time counter at the start of the sequence
        self.t.fill_(0.0)
        self._retained_outputs = []

    def forward(self, input):
        batch_size = input.size(0)

        # Perform a batched matrix multiplication.
        # input: [B, in_features] -> reshape to [B, in_features, 1]
        input_unsq = input.unsqueeze(2)

        # The output will be [B, out_features, 1] and then we can squeeze the last dimension.
        output = torch.bmm(self.per_sample_weights, input_unsq).squeeze(2)

        # Optionally add a bias if needed.
        if self.bias is not None:
            output = output + self.bias
        self.update_imprints(input, output)
        if self.retain_sequence_bias_grads and output.requires_grad:
            output.retain_grad()
            self._retained_outputs.append(output)
        return output

    def update_imprints(self, input, output):
        self.in_traces.data = input
        self.out_traces.data = output

    def populate_dfa_gradients(self, error_signal):
        """Populate gradients using DFA feedback weights for gradient-based update.

        error_signal is train.py's output_error, [B, vocab], the same object for every layer.
        Last layers use it as is: _last_projected_error is then that shared object, not a copy,
        and _update_bias_from_grad reads it. Other layers project it with feedback_weights into a
        new tensor. Nothing here modifies error_signal. The new gradient tensor is assigned
        directly to .grad after train.py clears the preceding DFA step's value."""
        # Project error signal using feedback weights (DFA-specific); last layers use it as is.
        # error_signal: [batch_size, vocab_size] -> projected_error: [batch_size, out_features]
        projected_error = dfa_projected_error(error_signal, self.feedback_weights, self.is_last_layer)

        # Store projected error for bias updates
        self._last_projected_error = projected_error

        # Per-sequence gradient: outer product with the input trace, [batch_size, out_features, in_features]
        gradient = dfa_per_sample_gradient(projected_error, self.in_traces.data)

        # Populate per_sample_weights.grad
        if self.per_sample_weights.grad is None:
            self.per_sample_weights.grad = gradient
        else:
            self.per_sample_weights.grad.copy_(gradient)

    def sequence_bias_grads(self):
        """Each sequence's share of this layer's bias gradient, [B, out], or None: its gradient
        with respect to a per-sequence copy of the bias, on the same footing as its slice of
        per_sample_weights.grad. Under DFA that is the projected error (the bias step is -lr
        times its batch mean); under backprop and BPTT it is the retained outputs' gradients,
        summed over steps (bias.grad is their sum over the batch)."""
        if self.bias is None:
            return None
        if self.updater == 'dfa':
            return getattr(self, '_last_projected_error', None)
        grads = [output.grad for output in self._retained_outputs if output.grad is not None]
        return torch.stack(grads).sum(0) if grads else None

    def scale_sequence_grads(self, scale):
        """Multiplies sequence b's weight gradient and bias share by scale[b] (--grad_norm_clip).
        The DFA projected error is replaced, not modified: i2o's is train.py's output_error."""
        if self.per_sample_weights.grad is not None:
            self.per_sample_weights.grad.mul_(scale.view(-1, 1, 1))
        shares = self.sequence_bias_grads()
        if shares is not None:
            if self.updater == 'dfa':
                self._last_projected_error = shares * scale.unsqueeze(1)
            else:
                # bias.grad + sum_b (s_b - 1) share_b = sum_b s_b share_b, and adds exactly 0
                # where the clip does not bind, leaving autograd's reduction untouched.
                self.bias.grad.add_(((scale - 1).unsqueeze(1) * shares).sum(0))
        self._retained_outputs = []

    def apply_update(self, learning_rate, update_clamp, state):
        """Update step shared by DFA and backprop.

        Args:
            learning_rate: Learning rate for updates
            update_clamp: Element-wise clamp on the alpha-scaled update of the ephemeral
                entries (--ephemeral_update_clamp; 0 = off)
            state: Training state dictionary for logging
        """
        if self.per_sample_weights.grad is None:
            # A layer with no gradient this step (i2h under per-step backprop) is still
            # regularized after every update, as SimpleRNN's layers are.
            self._apply_regularization()
            return

        # Get the gradient (already populated by either DFA or backprop)
        update = -self.per_sample_weights.grad

        # Apply plasticity scaling and masking (same for both methods)
        if not self.is_last_layer:
            plasticity_expanded = self.plasticity.unsqueeze(0)  # [1, out_features, in_features]
            mask_expanded = self.ephemeral_mask.unsqueeze(0)  # [1, out_features, in_features]

            # Scale by plasticity and mask
            update = update * plasticity_expanded
            # update = update * mask_expanded

            # Clamp the ephemeral entries of the update element-wise
            if update_clamp > 0:
                update = torch.where(mask_expanded,
                                    torch.clamp(update, -update_clamp, update_clamp),
                                    update)

        self.per_sample_weights.data = self.per_sample_weights.data + learning_rate * update

        # Log norms if requested
        if state.get("log_norms_now", False):
            self._log_update_norms(update)

        # Update bias using the gradient if this is DFA
        self._update_bias_from_grad(learning_rate)
        # Apply normalization and weight clipping if enabled
        self._apply_regularization()

    def _update_bias_from_grad(self, learning_rate):
        """Helper method to update bias using stored gradients."""
        if hasattr(self, 'bias') and self.bias is not None:
            if self.updater == 'dfa':
                # For DFA, manually update bias using projected error
                if hasattr(self, '_last_projected_error'):
                    self.bias.data += dfa_bias_update(self._last_projected_error, learning_rate)
            elif self.updater in ['backprop', 'bptt']:
                # For backprop/bptt, manually update bias using the computed bias gradient
                if self.bias.grad is not None:
                    bias_update = -learning_rate * self.bias.grad
                    self.bias.data += bias_update


    def _log_update_norms(self, update):
        """Helper method to log update norms."""
        with torch.no_grad():
            mask_expanded = self.ephemeral_mask.unsqueeze(0).expand_as(update)

            ephemeral_update = update[mask_expanded]
            slow_update = update[~mask_expanded]

            ephemeral_norm = torch.norm(ephemeral_update).item() if ephemeral_update.numel() > 0 else 0.0
            self.last_ephemeral_step_norm.data.fill_(ephemeral_norm)

            slow_norm = torch.norm(slow_update).item() if slow_update.numel() > 0 else 0.0
            self.last_slow_step_norm.data.fill_(slow_norm)

    def _update_bias(self, projected_error, learning_rate):
        """Helper method to update bias consistently."""
        if hasattr(self, 'bias') and self.bias is not None and self.updater == 'dfa':
            # For DFA, manually update bias. For backprop, optimizer handles it.
            bias_update = learning_rate * projected_error.mean(dim=0)
            if len(bias_update.shape) > 1:
                bias_update = bias_update.mean(dim=0)
            self.bias.data += bias_update

    def _apply_regularization(self):
        """Helper method to apply normalization and weight clipping."""
        # Each sequence's [out, in] slice is rescaled independently. Plasticity, biases,
        # feedback matrices, traces, and logged norms are intentionally excluded.
        self.per_sample_weights.data = regularized_weight(
            self.per_sample_weights.data, self.unit_norm_weights, self.weight_clamp, (1, 2))


    def apply_forget_step(self):
        """Decays the ephemeral entries: w <- (1 - forget_rate * ephemeral_mask) * w, element-wise,
        so each call keeps
        1 - forget_rate of every ephemeral weight. train.py calls this after each update (after
        the clamp and normalization too), as in the paper: w <- (1 - forget_rate) * (w - lr*alpha*g).
        This is done through .data under no_grad to avoid recording the update in autograd."""
        with torch.no_grad():
            # forget_rate * bool mask is float32 forget_rate on the mask and 0 elsewhere, the same
            # values the old stored forgetting_factor tensor held.
            self.per_sample_weights.data.mul_(1 - self.forget_rate * self.ephemeral_mask)

    def scale_ephemeral_grads(self, plasticity):
        """Scales the gradients of the ephemeral weights by plasticity (alpha) before the update."""
        if self.per_sample_weights.grad is None:
            return

        # Do not scale gradients for the final layer, mirroring the DFA update rule.
        if self.is_last_layer:
            return

        with torch.no_grad():
            # Create a scaling tensor based on the plasticity mask.
            # The scaling factor is `plasticity`, making the effective learning rate
            # for ephemeral weights `learning_rate * plasticity`, which
            # mirrors the logic in the DFA updater.
            lr_scale = plasticity
            # self.ephemeral_mask is [out, in], grad is [B, out, in]
            scaling_factor = torch.ones_like(self.ephemeral_mask, dtype=torch.float)
            scaling_factor[self.ephemeral_mask] = lr_scale

            # Apply scaling
            self.per_sample_weights.grad *= scaling_factor.unsqueeze(0)

    def get_norms(self):
        """Calculates and returns weight and last update norms."""
        with torch.no_grad():
            weights = self.per_sample_weights.data
            # Ensure mask is broadcastable for indexing
            mask_expanded = self.ephemeral_mask.unsqueeze(0).expand_as(weights)

            combined_weight_norm = torch.norm(weights).item()

            # Check if any ephemeral weights exist before calculating norm
            ephemeral_weights = weights[mask_expanded]
            ephemeral_norm = torch.norm(ephemeral_weights).item() if ephemeral_weights.numel() > 0 else 0.0

            # Check if any slow weights exist
            slow_weights = weights[~mask_expanded]
            slow_norm = torch.norm(slow_weights).item() if slow_weights.numel() > 0 else 0.0

            # update_norm = self.last_update_norm.item()

        norms = {
            'weight_norm': combined_weight_norm,
            'ephemeral_weight_norm': ephemeral_norm,
            'slow_weight_norm': slow_norm,
            'ephemeral_update_norm': self.last_ephemeral_step_norm.item(),
            'slow_update_norm': self.last_slow_step_norm.item(),
        }
        if not self.ephemeral_mask.any():
            # No ephemeral entries (last layers): report no ephemeral norms rather than
            # zeros, so they do not pull down the averages logged to W&B.
            del norms['ephemeral_weight_norm'], norms['ephemeral_update_norm']
        return norms

    def store_grad_norms(self):
        """Calculates the norm of the current gradient and stores it."""
        if self.per_sample_weights.grad is None:
            self.last_ephemeral_step_norm.data.fill_(0.0)
            self.last_slow_step_norm.data.fill_(0.0)
            return

        with torch.no_grad():
            grad = self.per_sample_weights.grad
            mask_expanded = self.ephemeral_mask.unsqueeze(0).expand_as(grad)

            ephemeral_grad = grad[mask_expanded]
            slow_grad = grad[~mask_expanded]

            ephemeral_norm = torch.norm(ephemeral_grad).item() if ephemeral_grad.numel() > 0 else 0.0
            self.last_ephemeral_step_norm.data.fill_(ephemeral_norm)

            slow_norm = torch.norm(slow_grad).item() if slow_grad.numel() > 0 else 0.0
            self.last_slow_step_norm.data.fill_(slow_norm)

    def set_plasticity(self, value):
        """Sets the plasticity (alpha) of the ephemeral weights; slow weights keep 1."""
        with torch.no_grad():
            # Only update plasticity values where the mask is True (ephemeral weights)
            self.plasticity.data[self.ephemeral_mask] = value
            print(f"Set plasticity to {value} for {torch.sum(self.ephemeral_mask).item()} ephemeral weights")

class EphemeralRNN(torch.nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, num_layers, charset,
        dropout_rate=0, residual_connection=False, init_type='zero',
        unit_norm_weights=True, weight_clamp=0, updater='dfa',
        plasticity=1, batch_size=1, forget_rate=0.01, ephemeral_fraction=0.2,
        enable_recurrence=True, retain_sequence_bias_grads=False
    ):
        """forget_rate: fraction of each ephemeral weight removed per forget step,
        w <- (1 - forget_rate) * w (see EphemeralLinear).
        retain_sequence_bias_grads: needed by clip_grad_norm_per_sequence under backprop and
        BPTT (train.py sets it when --grad_norm_clip > 0)."""
        super(EphemeralRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.init_type = init_type
        inner_size = recurrent_trunk_size(input_size, hidden_size)
        self.residual_connection = residual_connection
        self.batch_size = batch_size
        self.forget_rate = forget_rate
        self.enable_recurrence = enable_recurrence

        # Using EphemeralLinear instead of Linear
        self.linear_layers = torch.nn.ModuleList([
            EphemeralLinear(
                inner_size, inner_size, charset,
                unit_norm_weights=unit_norm_weights, weight_clamp=weight_clamp,
                updater=updater, plasticity=plasticity,
                batch_size=batch_size, forget_rate=forget_rate,
                ephemeral_fraction=ephemeral_fraction
            )
        ])
        for _ in range(1, num_layers):
            self.linear_layers.append(EphemeralLinear(
                inner_size, inner_size, charset,
                unit_norm_weights=unit_norm_weights, weight_clamp=weight_clamp,
                updater=updater, plasticity=plasticity,
                batch_size=batch_size, forget_rate=forget_rate,
                ephemeral_fraction=ephemeral_fraction
            ))

        # Dropout layers
        self.dropout = nn.Dropout(dropout_rate)

        # Forked transition/emission layout: i2h and i2o both read the deep trunk. i2h is a
        # hidden layer like the trunk layers (ephemeral entries and a DFA feedback matrix), while
        # i2o is the slow-only emission head.
        self.i2h = EphemeralLinear(
            inner_size, hidden_size, charset,
            unit_norm_weights=unit_norm_weights, weight_clamp=weight_clamp,
            updater=updater, plasticity=plasticity,
            batch_size=batch_size, forget_rate=forget_rate,
            ephemeral_fraction=ephemeral_fraction
        )
        self.i2o = EphemeralLinear(
            inner_size, output_size, charset,
            unit_norm_weights=unit_norm_weights, weight_clamp=weight_clamp,
            updater=updater, requires_grad=False, is_last_layer=True,
            plasticity=plasticity, batch_size=batch_size, forget_rate=forget_rate,
            ephemeral_fraction=ephemeral_fraction
        )
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.updater = updater
        for layer in self.trained_layers():
            layer.retain_sequence_bias_grads = retain_sequence_bias_grads
        self.grad_clip_stats = GradNormClipStats()

    def trained_layers(self):
        """Every EphemeralLinear, in update order: the hidden layers, i2h, then i2o."""
        return [*self.linear_layers, self.i2h, self.i2o]

    def apply_regularization(self):
        """--unit_norm_weights then --weight_clamp on every layer's per_sample_weights. DFA and
        backprop apply it inside apply_update; BPTT calls this after its plain SGD step."""
        for layer in self.trained_layers():
            layer._apply_regularization()

    def clip_grad_norm_per_sequence(self, max_norm):
        """--grad_norm_clip for the ephemeral model: conventional gradient-norm clipping applied
        to each sequence's own gradient, since each sequence has its own weights.

        Sequence b's norm is taken over its slice of every layer's per_sample_weights.grad and
        its share of every bias gradient (sequence_bias_grads), and all of them are multiplied
        by min(1, max_norm / norm). The threshold therefore does not depend on the batch size,
        one sequence's gradient never rescales another's, and with batch size 1 this is
        torch.nn.utils.clip_grad_norm_ over every trained gradient. It runs on the raw gradient,
        before plasticity (alpha) scales the ephemeral entries and before
        --ephemeral_update_clamp and --weight_clamp. Returns the pre-clip norms, [B]."""
        if self.updater != 'dfa' and not self.i2o.retain_sequence_bias_grads:
            raise RuntimeError("Build EphemeralRNN with retain_sequence_bias_grads=True to clip "
                               "per-sequence gradients under backprop or BPTT.")
        squared = None
        for layer in self.trained_layers():
            parts = []
            if layer.per_sample_weights.grad is not None:
                parts.append(torch.linalg.vector_norm(layer.per_sample_weights.grad, dim=(1, 2)))
            shares = layer.sequence_bias_grads()
            if shares is not None:
                parts.append(torch.linalg.vector_norm(shares, dim=1))
            for part in parts:
                squared = part.square() if squared is None else squared + part.square()
        if squared is None:
            return None
        norms = squared.sqrt()
        scale = per_sequence_clip_scale(norms, max_norm)
        for layer in self.trained_layers():
            layer.scale_sequence_grads(scale)
        self.grad_clip_stats.record(norms, max_norm)
        return norms

    def forward(self, input, hidden):
        # print(f"input shape: {input.shape}, hidden shape: {hidden.shape}")
        combined = torch.cat((input, hidden), dim=1)
        if self.residual_connection:
            residual = combined.clone()  # Store the original combined tensor for residual connection

        # Pass through the ephemeral linear layers with ReLU and Dropout
        for layer in self.linear_layers:
            combined = layer(combined)
            combined = F.gelu(combined)
            # combined = self.dropout(combined)

        # Add the residual (original combined tensor) to the output of the layers
        # print(f"residual_shape: {residual.shape}, combined shape: {combined.shape}")
        if self.residual_connection:
            combined += residual

        # Forked transition/emission layout. The state head remains bounded for recurrence,
        # while the output head reads a sibling transform of the shared deep representation.
        hidden_t = torch.tanh(self.i2h(combined))
        output = self.i2o(combined)
        # --enable_recurrence False still executes both heads but feeds back zeros.
        next_hidden = hidden_t if self.enable_recurrence else torch.zeros_like(hidden)

        # output.requires_grad = True # This is now handled in the training loop for DFA.
        # output = self.dropout(output)  # Apply dropout to the output before softmax
        # output = self.softmax(output)
        return output, next_hidden

    def initHidden(self, batch_size):
        device = next(self.parameters()).device
        return torch.zeros(batch_size, self.hidden_size, device=device, requires_grad=False)

    def apply_forget_step(self):
        """Calls apply_forget_step on all EphemeralLinear layers."""
        for layer in self.linear_layers:
            layer.apply_forget_step()
        self.i2h.apply_forget_step()
        self.i2o.apply_forget_step()

    def clear_dfa_gradients(self):
        """Clear the only gradients manually populated by the DFA updater."""
        for layer in self.linear_layers:
            layer.per_sample_weights.grad = None
        self.i2h.per_sample_weights.grad = None
        self.i2o.per_sample_weights.grad = None

    def scale_ephemeral_grads(self, plasticity):
        """Calls scale_ephemeral_grads on all EphemeralLinear layers."""
        for layer in self.linear_layers:
            layer.scale_ephemeral_grads(plasticity)
        self.i2h.scale_ephemeral_grads(plasticity)
        self.i2o.scale_ephemeral_grads(plasticity)


    def get_all_norms(self):
        """Aggregates norms from all EphemeralLinear layers."""
        all_norms = {}

        def _collect_norms(layers_list, prefix):
            for i, layer in enumerate(layers_list):
                if isinstance(layer, EphemeralLinear):
                    layer_norms = layer.get_norms()
                    for key, value in layer_norms.items():
                        all_norms[f'{prefix}_{i}_{key}'] = value

        _collect_norms(self.linear_layers, 'linear')
        _collect_norms([self.i2h], 'i2h')
        _collect_norms([self.i2o], 'i2o')

        return all_norms

    def store_all_grad_norms(self):
        """Calls store_grad_norms on all EphemeralLinear layers that are trained."""
        for layer in self.linear_layers:
            layer.store_grad_norms()
        self.i2h.store_grad_norms()
        self.i2o.store_grad_norms()

    def start_sequence_wipe(self):
        """Calls start_sequence_wipe on all EphemeralLinear layers."""
        for layer in self.linear_layers:
            layer.start_sequence_wipe()
        self.i2h.start_sequence_wipe()
        self.i2o.start_sequence_wipe()

    def set_plasticity(self, value):
        """Sets the ephemeral plasticity (alpha) in all EphemeralLinear layers (used on resume)."""
        print(f"Setting plasticity from checkpoint resume: {value}")

        # Update all linear layers
        for i, layer in enumerate(self.linear_layers):
            if isinstance(layer, EphemeralLinear):
                layer.set_plasticity(value)

        # Update i2h layer
        if isinstance(self.i2h, EphemeralLinear):
            self.i2h.set_plasticity(value)

        # Note: i2o is a last layer, so it doesn't use plasticity scaling
        # in the same way, but we'll update them for consistency
        if isinstance(self.i2o, EphemeralLinear) and not self.i2o.is_last_layer:
            self.i2o.set_plasticity(value)



class DFALinear(nn.Linear):
    """nn.Linear that the SimpleRNN baseline can train with DFA, the same way EphemeralLinear is
    trained (same error projection, input trace, outer product, learning rate and bias step,
    via the shared dfa_* helpers above). Without enable_dfa it is a plain nn.Linear with the same
    state dict, so backprop and BPTT are unchanged.

    SimpleRNN has one weight shared by the batch rather than one copy per sequence, so the
    per-sequence gradients are averaged over the batch, as the bias step already is. That is
    what EphemeralRNN's slow weights amount to: each copy takes its own sequence's step, and
    start_sequence_wipe() sets every copy to the batch mean."""

    def __init__(self, in_features, out_features, bias=True,
                 unit_norm_weights=False, weight_clamp=0):
        super().__init__(in_features, out_features, bias)
        self.unit_norm_weights = unit_norm_weights
        self.weight_clamp = weight_clamp
        self.is_last_layer = False
        self.register_buffer('feedback_weights', None)  # set by enable_dfa for non-last layers
        self.in_traces = None  # this step's input, recorded by forward (not saved)
        self._last_projected_error = None

    def enable_dfa(self, vocab_size, is_last_layer):
        """Gives a non-last layer its fixed random feedback matrix [vocab, out], initialised as
        EphemeralLinear's feedback_weights; a last layer (i2o) gets the output error directly."""
        self.is_last_layer = is_last_layer
        if not is_last_layer:
            self.feedback_weights = init_feedback_weights(vocab_size, self.out_features).to(self.weight.device)

    def forward(self, input):
        self.in_traces = input.detach()
        return super().forward(input)

    def populate_dfa_gradients(self, error_signal):
        """Sets weight.grad to the batch mean of the per-sequence DFA gradients and bias.grad to
        the batch mean of the projected error. error_signal is not modified."""
        projected_error = dfa_projected_error(error_signal, self.feedback_weights, self.is_last_layer)
        self._last_projected_error = projected_error
        self.weight.grad = dfa_per_sample_gradient(projected_error, self.in_traces).mean(dim=0)
        if self.bias is not None:
            # So apply_dfa_update's bias step, -lr * bias.grad, is dfa_bias_update(projected_error, lr).
            self.bias.grad = projected_error.mean(dim=0)

    def apply_dfa_update(self, learning_rate):
        """w <- w - lr * w.grad and b <- b - lr * b.grad, with the grads populate_dfa_gradients
        set (clipped first by train.py if --grad_norm_clip is on)."""
        with torch.no_grad():
            self.weight -= learning_rate * self.weight.grad
            if self.bias is not None:
                self.bias -= learning_rate * self.bias.grad
            self.apply_regularization()

    def apply_regularization(self):
        self.weight.data = regularized_weight(
            self.weight.data, self.unit_norm_weights, self.weight_clamp, (0, 1))


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_rate=0.1,
                 init_type='zero', enable_recurrence=True, updater=None,
                 residual_connection=False, unit_norm_weights=False, weight_clamp=0):
        """updater: 'dfa' gives the hidden layers and i2h fixed random DFA feedback matrices (drawn
        after every layer is initialised, so the layers start the same as under the other
        updaters at the same seed). Other values leave it a plain backprop/BPTT model."""
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.init_type = init_type
        self.enable_recurrence = enable_recurrence
        self.updater = updater
        self.residual_connection = residual_connection
        inner_size = recurrent_trunk_size(input_size, hidden_size)

        # Standard linear layers (DFALinear is an nn.Linear that can also take DFA updates)
        layer_options = {"unit_norm_weights": unit_norm_weights, "weight_clamp": weight_clamp}
        self.linear_layers = nn.ModuleList([DFALinear(inner_size, inner_size, **layer_options)])
        for _ in range(1, num_layers):
            self.linear_layers.append(DFALinear(inner_size, inner_size, **layer_options))

        # Dropout layers
        self.dropout = nn.Dropout(dropout_rate)

        # Forked transition and emission heads over the shared deep representation.
        self.i2h = DFALinear(inner_size, hidden_size, **layer_options)
        self.i2o = DFALinear(inner_size, output_size, **layer_options)
        self.softmax = nn.LogSoftmax(dim=1)

        if updater == 'dfa':
            for layer in self.dfa_layers():
                layer.enable_dfa(output_size, is_last_layer=layer is self.i2o)
        self.grad_clip_stats = GradNormClipStats()

    def dfa_layers(self):
        """Every layer DFA trains, in update order: the hidden layers, i2h, then i2o."""
        return [*self.linear_layers, self.i2h, self.i2o]

    def apply_regularization(self):
        """Apply the configured weights-only normalization and clamp after an SGD step."""
        with torch.no_grad():
            for layer in self.dfa_layers():
                layer.apply_regularization()

    def forward(self, input, hidden):
        # print(f"input shape: {input.shape}, hidden shape: {hidden.shape}")
        combined = torch.cat((input, hidden), dim=1)
        if self.residual_connection:
            residual = combined.clone()

        # Match EphemeralRNN's shared-width GELU trunk.
        for layer in self.linear_layers:
            combined = layer(combined)
            combined = F.gelu(combined)
            # combined = self.dropout(combined)

        if self.residual_connection:
            combined += residual

        # Forked transition/emission layout, as in EphemeralRNN. The output is independent of
        # this step's state head; --enable_recurrence False feeds back zeros.
        hidden_t = torch.tanh(self.i2h(combined))
        output = self.i2o(combined)
        next_hidden = hidden_t if self.enable_recurrence else torch.zeros_like(hidden)
        # output = self.dropout(output)
        # output = self.softmax(output)
        return output, next_hidden

    def get_all_norms(self):
        """Calculates weight and gradient norms for SimpleRNN."""
        all_norms = {}
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.requires_grad:
                    all_norms[f'{name}_weight_norm'] = torch.norm(param.data).item()
                    if param.grad is not None:
                        all_norms[f'{name}_grad_norm'] = torch.norm(param.grad).item()
                    else:
                        all_norms[f'{name}_grad_norm'] = 0.0 # No grad yet/available
        return all_norms

    def initHidden(self, batch_size):
        device = next(self.parameters()).device
        return torch.zeros(batch_size, self.hidden_size, device=device)
