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
# baseline), so the two models' DFA stays the same computation. By default neither multiplies
# the projected error by the layer's activation derivative; --dfa_fprime does, in both (Nøkland
# 2016); see README "Known issues".

def init_feedback_weights(vocab_size, out_features):
    """A layer's fixed random DFA feedback matrix B, [vocab, out]: xavier_normal_."""
    return torch.nn.init.xavier_normal_(torch.empty(vocab_size, out_features))


_INV_SQRT2 = 1 / math.sqrt(2)
_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)


def dfa_activation_derivative(pre_activation, activation):
    """f'(a), [B, out], for the nonlinearity that follows a layer, at its pre-activation a
    (--dfa_fprime). 'gelu' is the exact (erf) GELU that F.gelu computes by default:
    Phi(a) + a * phi(a). 'tanh': 1 - tanh(a)^2."""
    if activation == 'gelu':
        cdf = 0.5 * (1 + torch.erf(pre_activation * _INV_SQRT2))
        pdf = torch.exp(-0.5 * pre_activation.square()) * _INV_SQRT_2PI
        return cdf + pre_activation * pdf
    if activation == 'tanh':
        return 1 - torch.tanh(pre_activation).square()
    raise ValueError(f"No DFA activation derivative for activation {activation!r}")


def dfa_projected_error(error_signal, feedback_weights, is_last_layer, activation_derivative=None):
    """The error a layer's DFA update uses, [B, out]: the output error itself for a last layer
    (i2o), else error_signal @ feedback_weights, multiplied element-wise by
    activation_derivative (f'(a) at the layer's pre-activation, [B, out]) when it is given
    (--dfa_fprime; Nøkland 2016: delta_l = (B_l e) * f'(a_l)). error_signal is train.py's
    output_error, [B, vocab]; it is never modified, and a last layer gets that same object."""
    if is_last_layer:
        return error_signal
    projected = error_signal @ feedback_weights
    if activation_derivative is not None:
        projected = projected * activation_derivative
    return projected


def dfa_layer_activation_derivative(layer):
    """--dfa_fprime: f'(a) for a non-last layer (EphemeralLinear or DFALinear) at the
    pre-activation its last forward recorded, or None (flag off, or a last layer, whose error is
    the output error itself). layer.activation names the nonlinearity the model applies to the
    layer's output ('gelu' for the trunk, 'tanh' for i2h)."""
    if not layer.dfa_fprime or layer.is_last_layer:
        return None
    return dfa_activation_derivative(layer.pre_activation(), layer.activation)


def dfa_per_sample_gradient(projected_error, input):
    """Per-sequence DFA gradient, [B, out, in]: the outer product of each sequence's projected
    error [B, out] with its layer input (the input trace) [B, in]."""
    out = projected_error.unsqueeze(2)  # [batch_size, out_features, 1]
    return out * input.unsqueeze(1)  # [batch_size, 1, in_features] -> [batch_size, out_features, in_features]


def normalized_readout_error(projected_error, inputs, enabled, eps=1e-6):
    """NLMS scaling for --readout_nlms.

    Divides each sequence's readout step by eps + ||x||^2. The same factor applies to the bias,
    treating it as part of the per-sequence readout step. Disabled returns the original error
    object unchanged.
    """
    if not enabled:
        return projected_error
    scale = 1 / (inputs.square().sum(1) + eps)
    return projected_error * scale.unsqueeze(1)


def normalize_slow_gradient(gradient, inputs, ephemeral_mask, enabled, eps=1e-6):
    """--slow_nlms: divide only slow weight entries by eps + ||x||^2 per sequence."""
    if not enabled:
        return gradient
    scale = 1 / (inputs.square().sum(1) + eps)
    normalized = gradient * scale.view(-1, 1, 1)
    return torch.where(ephemeral_mask.unsqueeze(0), gradient, normalized)


def dfa_bias_update(projected_error, learning_rate):
    """DFA bias step, [out]: -learning_rate times the batch mean of the projected error."""
    bias_update = -learning_rate * projected_error.mean(dim=0)
    if len(bias_update.shape) > 1:
        bias_update = bias_update.mean(dim=0)
    return bias_update


def recurrent_trunk_size(input_size, hidden_size):
    """Width shared by the recurrent models' concatenated input/state trunk."""
    return input_size + hidden_size


def trunk_layer_norm(activations):
    """--layer_norm: LayerNorm over the feature dimension of one trunk layer's post-GELU
    activations, [B, width], with no learnable affine (no gain or bias, so no extra slow
    parameters for DFA to train; the next layer's weights and bias can absorb any scale and
    shift). Each row gets zero mean and unit variance (eps 1e-5, torch's default). It sits after
    the activation, so it is exactly what the next layer (trunk, i2h or i2o) reads: that layer's
    forward records it as its input trace, and the DFA outer product uses it."""
    return F.layer_norm(activations, activations.shape[-1:])


def regularized_weight(weight, weight_clamp):
    """--weight_clamp: the weights element-wise clamped to [-c, c] (0 = off), in place; biases
    are excluded. (--unit_norm_weights, which rescaled whole weight slices here first, was
    removed in 2026-09; see README "Renamed flags".)"""
    if weight_clamp != 0:
        weight.clamp_(-weight_clamp, weight_clamp)
    return weight


def ephemeral_update(gradient, plasticity, ephemeral_mask, update_clamp, is_last_layer):
    """The update a layer's per_sample_weights take, before the learning rate, [B, out, in]:
    -plasticity * gradient, with the ephemeral entries clamped element-wise to
    [-update_clamp, update_clamp] (0 = off). A last layer (i2o) has no ephemeral entries: -gradient.

    plasticity is the layer's [out, in] tensor, or a float alpha (EphemeralLinear.fused_plasticity):
    alpha on the mask and 1 elsewhere, built from the bool mask. Those are the tensor's float32
    values, so the result is bit-identical, and a compiled step reads one byte per entry (the
    mask) instead of the mask plus the four-byte plasticity."""
    update = -gradient
    if not is_last_layer:
        if not torch.is_tensor(plasticity):
            plasticity = torch.where(ephemeral_mask, plasticity, 1.0)
        update = update * plasticity.unsqueeze(0)
        if update_clamp > 0:
            update = torch.where(ephemeral_mask.unsqueeze(0),
                                 torch.clamp(update, -update_clamp, update_clamp),
                                 update)
    return update


def forget_keep(forget_rate, ephemeral_mask, slow_weight_decay=0.0):
    """What each forget step keeps of an entry, [out, in]: 1 - forget_rate on the ephemeral mask
    and 1 - slow_weight_decay elsewhere (--slow_weight_decay; 0 = off). forget_rate * bool mask is
    float32 forget_rate on the mask and 0 elsewhere, the same values the old stored
    forgetting_factor tensor held."""
    keep = 1 - forget_rate * ephemeral_mask
    if slow_weight_decay:
        keep = keep - slow_weight_decay * ~ephemeral_mask
    return keep


def clamp_fast_entries(weights, ephemeral_mask, fast_weight_clamp):
    """--fast_weight_clamp: clamp only the ephemeral entries of per_sample_weights [B, out, in]
    to [-c, c], after --weight_clamp (0 = off). Slow entries are untouched."""
    if fast_weight_clamp:
        weights = torch.where(ephemeral_mask.unsqueeze(0),
                              weights.clamp(-fast_weight_clamp, fast_weight_clamp), weights)
    return weights


def dfa_output_error(output, target, criterion, label_smoothing=0.0):
    """The DFA output error dL/d(output), [B, vocab], and the per-sequence loss [B], for the
    logits of one step. criterion is train.py's CrossEntropyLoss(reduction='none'). An all-zero
    (padding) target row gives zero error, so that step writes nothing. The error is a new
    tensor, not a view of output. --label_smoothing eps replaces the one-hot target with
    target*(1-eps) + eps/V; all-zero padding rows remain zero-error rows."""
    with torch.enable_grad():
        output.requires_grad_(True)
        if label_smoothing:
            valid = target.sum(1, keepdim=True)
            smoothed = target * (1 - label_smoothing) + label_smoothing * valid / target.shape[1]
            loss = criterion(output, smoothed)
        else:
            loss = criterion(output, target)
        error = torch.autograd.grad(loss, output, grad_outputs=torch.ones_like(loss), retain_graph=False)[0]
    return loss, error


def dfa_layer_step(weights, bias, projected_error, inputs, plasticity, ephemeral_mask,
                   forget_rate: float, learning_rate: float, update_clamp: float,
                   weight_clamp: float, is_last_layer: bool,
                   slow_weight_decay: float = 0.0, fast_weight_clamp: float = 0.0,
                   freeze_slow: bool = False, freeze_fast: bool = False, fast_forget: bool = True,
                   slow_nlms: bool = False):
    """One EphemeralLinear's whole DFA step, in place on weights and bias: the DFA gradient, the
    update and weight clamp (apply_update), then forgetting (apply_forget_step).
    It is built from the same helpers as those methods, in the same order, so run eagerly it
    gives bit-identical results; --fused_update compiles it into one kernel per layer
    (EphemeralRNN.enable_fused_update), which never materializes the [B, out, in] gradient.

    freeze_slow (held-out evaluation, heldout.py): the ephemeral entries take exactly this step
    and every other entry keeps its value. Pass bias=None to freeze the bias too. The element-wise
    clamps act entry by entry, so the fast entries come out as in training.

    freeze_fast (--fast_backward_per_forward 1/N, on a character that gets no fast update): the
    slow entries and the bias take their step as usual, and the ephemeral entries only forget
    (fast_forget=False: they are left exactly as they are, the slow half of a K >= 2 character,
    whose fast half already forgot)."""
    gradient = normalize_slow_gradient(dfa_per_sample_gradient(projected_error, inputs), inputs,
                                       ephemeral_mask, slow_nlms)
    update = ephemeral_update(gradient, plasticity,
                              ephemeral_mask, update_clamp, is_last_layer)
    updated = clamp_fast_entries(
        regularized_weight(weights + learning_rate * update, weight_clamp),
        ephemeral_mask, fast_weight_clamp)
    updated = updated * forget_keep(forget_rate, ephemeral_mask, slow_weight_decay)
    if freeze_slow:
        updated = torch.where(ephemeral_mask.unsqueeze(0), updated, weights)
    if freeze_fast:
        forgotten = weights * forget_keep(forget_rate, ephemeral_mask, slow_weight_decay) if fast_forget else weights
        updated = torch.where(ephemeral_mask.unsqueeze(0), forgotten, updated)
    weights.copy_(updated)
    if bias is not None:
        bias_error = normalized_readout_error(projected_error, inputs, slow_nlms)
        bias.add_(dfa_bias_update(bias_error, learning_rate))


def parse_slow_update_every(value):
    """--slow_update_every: a positive integer N (steps) or 'sequence'."""
    if isinstance(value, str) and value.strip().lower() == "sequence":
        return "sequence"
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = 0
    if isinstance(value, bool) or number < 1 or str(number) != str(value).strip():
        raise ValueError(f"--slow_update_every takes a positive integer or 'sequence', not {value!r}")
    return number


def parse_fast_backward_per_forward(value):
    """--fast_backward_per_forward: a positive integer K (K DFA steps per forward pass: the first
    as usual, K - 1 more after re-running the forward pass), or '1/N' for an integer N >= 2 (a fast
    DFA step on every N-th character only). Returns the int K, or the string '1/N'."""
    text = str(value).strip()
    bad = ValueError(f"--fast_backward_per_forward takes a positive integer K or '1/N' with an "
                     f"integer N >= 2, not {value!r}")
    if isinstance(value, bool):
        raise bad
    if "/" in text:
        top, _, bottom = text.partition("/")
        if top != "1" or not bottom.isdigit() or str(int(bottom)) != bottom or int(bottom) < 2:
            raise bad
        return f"1/{int(bottom)}"
    if not text.isdigit() or str(int(text)) != text or int(text) < 1:
        raise bad
    return int(text)


def fast_backward_counts(value):
    """(iterations K, subsample N) of a parsed --fast_backward_per_forward: (K, 1) or (1, N)."""
    value = parse_fast_backward_per_forward(value)
    return (1, int(value[2:])) if isinstance(value, str) else (value, 1)


def slow_window_step(weights, gradient_sum, ephemeral_mask, learning_rate: float, weight_clamp: float,
                     decay: float):
    """The slow entries' update at the end of a --slow_update_every window, [..., out, in]:
    weights - learning_rate * gradient_sum, then --weight_clamp, then the window's
    --slow_weight_decay (decay = (1 - slow_weight_decay) ** steps; 1 = off): the order a per-step
    update takes. Fast entries (the mask) keep their values."""
    updated = weights - learning_rate * gradient_sum
    if weight_clamp != 0:
        updated = updated.clamp(-weight_clamp, weight_clamp)
    if decay != 1:
        updated = updated * decay
    return torch.where(ephemeral_mask, weights, updated)


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


def set_dfa_fprime(model, enabled):
    """--dfa_fprime for EphemeralRNN or SimpleRNN: each non-output layer's projected DFA error is
    multiplied by the derivative of the nonlinearity the model's forward applies to that layer's
    output, at this step's pre-activation: gelu' for the trunk layers (F.gelu, exact erf form)
    and tanh' for i2h (hidden_t = tanh(i2h(.))). i2o keeps the raw output error. --output_tanh
    and the residual connection change only what i2o and i2h read, not any layer's own
    nonlinearity, so they do not change the derivatives. Off (the default), nothing is computed
    or recorded beyond what the models already do, and the step is unchanged."""
    for layer in model.linear_layers:
        layer.dfa_fprime, layer.activation = enabled, 'gelu'
    model.i2h.dfa_fprime, model.i2h.activation = enabled, 'tanh'
    model.i2o.dfa_fprime, model.i2o.activation = enabled, None


class EphemeralLinear(nn.Linear):
    def __init__(self, in_features, out_features, charset, bias=True, weight_clamp=0, updater='dfa', requires_grad=False, is_last_layer=False, plasticity=1, batch_size=1, forget_rate=0.01, ephemeral_fraction=0.2, slow_weight_decay=0, fast_weight_clamp=0, slow_update_every=1):
        """forget_rate: fraction of each ephemeral weight removed per forget step,
        w <- (1 - forget_rate) * w (see apply_forget_step). The paper's "forgetting rate
        coefficient 0.7" is 1 - forget_rate, i.e. forget_rate = 0.3. Same meaning as --forget_rate.
        plasticity: alpha, the learning-rate multiplier on the ephemeral entries (--plasticity).
        ephemeral_fraction: fraction of entries that are ephemeral (--ephemeral_fraction).
        weight_clamp: applied after each update (--weight_clamp; see _apply_regularization).
        slow_update_every: --slow_update_every, 1 (every step), N or 'sequence'. Anything but 1
        allocates the window's accumulators (see accumulate_slow_gradient)."""
        super(EphemeralLinear, self).__init__(in_features, out_features, bias)

        # Set requires_grad for the base class parameters
        self.weight.requires_grad = False # Base weights are not trained directly
        if bias:
            # For backprop/bptt, bias needs requires_grad=True so PyTorch computes bias.grad
            # For DFA, we set it to False since we handle bias manually
            self.bias.requires_grad = (updater in ['backprop', 'bptt'])

        self.weight_clamp = weight_clamp
        self.updater = updater
        self.is_last_layer = is_last_layer
        self.normalize_step = False  # --readout_nlms, set only on i2o
        self.slow_nlms = False  # --slow_nlms, set on every trained layer
        # --dfa_fprime (set by EphemeralRNN): scale the projected DFA error by f'(pre-activation),
        # where activation names the nonlinearity the model applies to this layer's output.
        self.dfa_fprime = False
        self.activation = None
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
        # --slow_weight_decay: the fraction of each slow entry the same forget step removes.
        self.slow_weight_decay = slow_weight_decay
        self.fast_weight_clamp = fast_weight_clamp  # --fast_weight_clamp (clamp_fast_entries)

        # Initialize plasticity parameters with the generated values
        if self.is_last_layer == False:
            self.plasticity = nn.Parameter(distribution, requires_grad=requires_grad)
        else:
            self.plasticity = nn.Parameter(torch.ones_like(self.weight), requires_grad=requires_grad)
        print(f"Number of non-zero values in self.plasticity: {torch.count_nonzero(self.plasticity).item()}")

        self.plasticity_feedback_weights = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(len(charset), out_features)), requires_grad=requires_grad)
        self._fused_plasticity = None  # cache for fused_plasticity()

        # --slow_update_every other than 1: the slow gradients of the current window, in
        # non-persistent buffers (never in checkpoints: a window always ends with its sequence,
        # so they are zero between batches). 'sequence' keeps their batch sum, [out, in]; N keeps
        # each step's projected error and input, [N, B, out] and [N, B, in], since each copy then
        # takes its own sequence's sum. No random numbers are drawn.
        self.slow_update_every = parse_slow_update_every(slow_update_every)
        if self.slow_update_every == "sequence":
            self.register_buffer('slow_grad_sum', torch.zeros(out_features, in_features), persistent=False)
        elif self.slow_update_every != 1:
            n = self.slow_update_every
            self.register_buffer('window_errors', torch.zeros(n, batch_size, out_features), persistent=False)
            self.register_buffer('window_inputs', torch.zeros(n, batch_size, in_features), persistent=False)
        if self.slow_update_every != 1 and bias:
            self.register_buffer('bias_grad_sum', torch.zeros(out_features), persistent=False)

    def fused_plasticity(self):
        """alpha as a float when the plasticity tensor is alpha on the mask and 1 elsewhere (always
        so in training: see set_plasticity), else the tensor itself. The fused and windowed steps
        pass it to ephemeral_update, which rebuilds the same values from the bool mask. Checked
        once (a host sync) and cached until set_plasticity or a state-dict load."""
        if self._fused_plasticity is None:
            with torch.no_grad():
                plasticity, mask = self.plasticity, self.ephemeral_mask
                fast = plasticity[mask]
                alpha = float(fast[0]) if fast.numel() else 1.0
                uniform = bool((fast == alpha).all()) and bool((plasticity[~mask] == 1).all())
            self._fused_plasticity = alpha if uniform else plasticity
        return self._fused_plasticity

    def _load_from_state_dict(self, *args, **kwargs):
        self._fused_plasticity = None
        super()._load_from_state_dict(*args, **kwargs)

    def start_sequence_wipe(self, wipe_fast=True):
        """Start of a sequence: set every sequence's slow entries to the batch mean (consolidation),
        then, if wipe_fast, zero the ephemeral entries (also in the unused base weight); reset the
        time counter either way.

        wipe_fast=False (--wipe_every N > 1, on the sequences between wipes) still averages the
        slow entries, which is how the slow weights learn, but leaves each batch row's fast
        entries as the previous sequence in that row left them (already forgotten at forget_rate
        per step). The fast entries are not averaged across rows: each row's fast state carries
        into the next sequence in the same row.

        With --slow_update_every sequence the slow entries are already one shared matrix (the
        previous sequence ended with the batch-mean step, written to every copy), so the mean is
        skipped and they are left exactly as they are: the mean of B equal copies can differ in
        the last bit. The fast entries are wiped or kept as above."""
        if self.slow_update_every != "sequence":
            # Suppose per_sample_weights is of shape [B, out_features, in_features]
            # Aggregate across the batch (e.g., average) to get a single copy:
            aggregated = self.per_sample_weights.mean(dim=0, keepdim=True)
            if wipe_fast:
                # Then set every sequence's copy in the batch to this aggregated value:
                self.per_sample_weights.data.copy_(aggregated.expand_as(self.per_sample_weights))
            else:
                # Slow entries take the batch mean; fast entries keep their per-row values.
                self.per_sample_weights.data.copy_(torch.where(
                    self.ephemeral_mask.unsqueeze(0), self.per_sample_weights.data,
                    aggregated.expand_as(self.per_sample_weights)))
        if wipe_fast:
            # masked_fill_, not boolean indexing: the same values without a host sync.
            self.weight.data.masked_fill_(self.ephemeral_mask, 0)
            self.per_sample_weights.data.masked_fill_(self.ephemeral_mask.unsqueeze(0), 0)
        # Reset the time counter at the start of the sequence
        self.t.fill_(0.0)
        self._retained_outputs = []

    def accumulate_slow_gradient(self, projected_error, slot):
        """--slow_update_every other than 1: adds this step's DFA gradient (projected_error [B, out]
        times the input trace) to the window instead of applying it. 'sequence' adds the batch sum
        sum_b p_b x_b^T to slow_grad_sum [out, in] (one GEMM, no [B, out, in] tensor); N stores
        the step in window slot `slot`. The bias share, the batch mean of the projected error,
        goes to bias_grad_sum. Every entry is accumulated; only the slow ones are ever applied."""
        inputs = self.in_traces.data
        if self.slow_update_every == "sequence":
            self.slow_grad_sum.addmm_(projected_error.t(), inputs)
        else:
            self.window_errors[slot].copy_(projected_error)
            self.window_inputs[slot].copy_(inputs)
        if self.bias is not None:
            self.bias_grad_sum.add_(projected_error.mean(dim=0))

    def apply_slow_window(self, learning_rate, steps):
        """Ends a --slow_update_every window of `steps` steps: the slow entries and the bias take
        the accumulated step, and the accumulators are zeroed.

        'sequence': the batch mean of the per-sequence sums, S <- S - lr * (1/B) sum_b sum_t g_bt,
        written to every copy, so the slow entries stay one shared matrix. N: each copy takes its
        own sequence's sum, w_b <- w_b - lr * sum_t g_bt, which is what `steps` per-step updates
        give without the drift in between (start_sequence_wipe still averages the copies). Then
        --weight_clamp, then --slow_weight_decay as (1 - d) ** steps, so a window decays as much as
        `steps` per-step updates. The bias takes -lr times the summed batch-mean projected errors."""
        if steps == 0:
            return
        weights = self.per_sample_weights.data
        decay = (1 - self.slow_weight_decay) ** steps if self.slow_weight_decay else 1.0
        if self.slow_update_every == "sequence":
            shared = slow_window_step(weights[0], self.slow_grad_sum / self.batch_size, self.ephemeral_mask,
                                      learning_rate, self.weight_clamp, decay)
            weights.copy_(torch.where(self.ephemeral_mask, weights, shared.unsqueeze(0)))
            self.slow_grad_sum.zero_()
        else:
            # [B, out, steps] @ [B, steps, in]: each sequence's own summed gradient
            gradient_sum = torch.bmm(self.window_errors[:steps].permute(1, 2, 0),
                                     self.window_inputs[:steps].transpose(0, 1))
            weights.copy_(slow_window_step(weights, gradient_sum, self.ephemeral_mask.unsqueeze(0),
                                           learning_rate, self.weight_clamp, decay))
        if self.bias is not None:
            self.bias.data.add_(-learning_rate * self.bias_grad_sum)
            self.bias_grad_sum.zero_()

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

    def pre_activation(self):
        """This step's pre-activation (the output before the model's nonlinearity), [B, out]."""
        return self.out_traces.data

    def populate_dfa_gradients(self, error_signal):
        """Populate gradients using DFA feedback weights for gradient-based update.

        error_signal is train.py's output_error, [B, vocab], the same object for every layer.
        Last layers use it as is: _last_projected_error is then that shared object, not a copy,
        and _update_bias_from_grad reads it. Other layers project it with feedback_weights into a
        new tensor. Nothing here modifies error_signal. The new gradient tensor is assigned
        directly to .grad after train.py clears the preceding DFA step's value."""
        # Project error signal using feedback weights (DFA-specific); last layers use it as is.
        # error_signal: [batch_size, vocab_size] -> projected_error: [batch_size, out_features]
        projected_error = dfa_projected_error(error_signal, self.feedback_weights, self.is_last_layer,
                                              dfa_layer_activation_derivative(self))
        projected_error = normalized_readout_error(
            projected_error, self.in_traces.data, self.is_last_layer and self.normalize_step)

        # Per-sequence gradient: outer product with the input trace, [batch_size, out_features, in_features]
        gradient = dfa_per_sample_gradient(projected_error, self.in_traces.data)
        gradient = normalize_slow_gradient(
            gradient, self.in_traces.data, self.ephemeral_mask, self.slow_nlms)

        # Biases are slow, so --slow_nlms scales their per-sequence error too.
        self._last_projected_error = normalized_readout_error(
            projected_error, self.in_traces.data, self.slow_nlms)

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

        # The gradient was populated by DFA or backprop; plasticity scaling and the update clamp
        # are shared with dfa_layer_step (--fused_update).
        update = ephemeral_update(self.per_sample_weights.grad, self.plasticity, self.ephemeral_mask,
                                  update_clamp, self.is_last_layer)

        self.per_sample_weights.data = self.per_sample_weights.data + learning_rate * update

        # Log norms if requested
        if state.get("log_norms_now", False):
            self._log_update_norms(update)

        # Update bias using the gradient if this is DFA
        self._update_bias_from_grad(learning_rate)
        # Apply the weight clamps if enabled
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
        """--weight_clamp, then --fast_weight_clamp, on per_sample_weights. Plasticity, biases,
        feedback matrices, traces, and logged norms are intentionally excluded."""
        self.per_sample_weights.data = clamp_fast_entries(
            regularized_weight(self.per_sample_weights.data, self.weight_clamp),
            self.ephemeral_mask, self.fast_weight_clamp)


    def apply_forget_step(self):
        """Decays the ephemeral entries: w <- (1 - forget_rate * ephemeral_mask) * w, element-wise,
        so each call keeps
        1 - forget_rate of every ephemeral weight. train.py calls this after each update (after
        the clamps too), as in the paper: w <- (1 - forget_rate) * (w - lr*alpha*g).
        This is done through .data under no_grad to avoid recording the update in autograd."""
        with torch.no_grad():
            self.per_sample_weights.data.mul_(forget_keep(self.forget_rate, self.ephemeral_mask,
                                                          self.slow_weight_decay))

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
            self._fused_plasticity = None
            print(f"Set plasticity to {value} for {torch.sum(self.ephemeral_mask).item()} ephemeral weights")

class EphemeralRNN(torch.nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, num_layers, charset,
        dropout_rate=0, residual_connection=False, init_type='zero',
        weight_clamp=0, updater='dfa',
        plasticity=1, batch_size=1, forget_rate=0.01, ephemeral_fraction=0.2,
        enable_recurrence=True, retain_sequence_bias_grads=False,
        slow_weight_decay=0, output_tanh=False, fast_weight_clamp=0, layer_norm=False,
        dfa_fprime=False, slow_update_every=1, fast_backward_per_forward=1,
        readout_nlms=False, slow_nlms=False
    ):
        """forget_rate: fraction of each ephemeral weight removed per forget step,
        w <- (1 - forget_rate) * w (see EphemeralLinear).
        retain_sequence_bias_grads: needed by clip_grad_norm_per_sequence under backprop and
        BPTT (train.py sets it when --grad_norm_clip > 0).
        slow_weight_decay: --slow_weight_decay, applied with each forget step.
        output_tanh: --output_tanh, i2o reads tanh of the trunk instead of the trunk itself.
        layer_norm: --layer_norm, trunk_layer_norm after each trunk layer's GELU.
        dfa_fprime: --dfa_fprime, see set_dfa_fprime.
        slow_update_every: --slow_update_every (see windowed_dfa_step); 1 is the per-step update.
        fast_backward_per_forward: --fast_backward_per_forward (see extra_fast_iterations and
        skip_fast_dfa_step); 1 is one fast DFA step per character.
        readout_nlms: --readout_nlms, NLMS scaling on i2o's whole DFA step.
        slow_nlms: --slow_nlms, NLMS scaling on each layer's slow entries and bias only."""
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
        self.output_tanh = output_tanh
        self.slow_update_every = parse_slow_update_every(slow_update_every)
        if self.slow_update_every != 1:
            if updater != 'dfa':
                raise ValueError(f"--slow_update_every {self.slow_update_every} supports only the DFA "
                                 f"updater, not {updater!r}")
        self.pending_slow_steps = 0  # steps accumulated in the current --slow_update_every window
        self.fast_backward_per_forward = parse_fast_backward_per_forward(fast_backward_per_forward)
        self.fast_iterations, self.fast_subsample = fast_backward_counts(fast_backward_per_forward)
        if self.fast_backward_per_forward != 1 and updater != 'dfa':
            raise ValueError(f"--fast_backward_per_forward {self.fast_backward_per_forward} supports "
                             f"only the DFA updater, not {updater!r}")
        self._force_split_step = False  # tests: take the K >= 2 split step even for K = 1
        self.layer_norm = layer_norm

        # Using EphemeralLinear instead of Linear
        self.linear_layers = torch.nn.ModuleList([
            EphemeralLinear(
                inner_size, inner_size, charset,
                weight_clamp=weight_clamp,
                updater=updater, plasticity=plasticity,
                batch_size=batch_size, forget_rate=forget_rate,
                ephemeral_fraction=ephemeral_fraction, slow_weight_decay=slow_weight_decay,
            fast_weight_clamp=fast_weight_clamp, slow_update_every=slow_update_every
            )
        ])
        for _ in range(1, num_layers):
            self.linear_layers.append(EphemeralLinear(
                inner_size, inner_size, charset,
                weight_clamp=weight_clamp,
                updater=updater, plasticity=plasticity,
                batch_size=batch_size, forget_rate=forget_rate,
                ephemeral_fraction=ephemeral_fraction, slow_weight_decay=slow_weight_decay,
            fast_weight_clamp=fast_weight_clamp, slow_update_every=slow_update_every
            ))

        # Dropout layers
        self.dropout = nn.Dropout(dropout_rate)

        # Forked transition/emission layout: i2h and i2o both read the deep trunk. i2h is a
        # hidden layer like the trunk layers (ephemeral entries and a DFA feedback matrix), while
        # i2o is the slow-only emission head.
        self.i2h = EphemeralLinear(
            inner_size, hidden_size, charset,
            weight_clamp=weight_clamp,
            updater=updater, plasticity=plasticity,
            batch_size=batch_size, forget_rate=forget_rate,
            ephemeral_fraction=ephemeral_fraction, slow_weight_decay=slow_weight_decay,
            fast_weight_clamp=fast_weight_clamp, slow_update_every=slow_update_every
        )
        self.i2o = EphemeralLinear(
            inner_size, output_size, charset,
            weight_clamp=weight_clamp,
            updater=updater, requires_grad=False, is_last_layer=True,
            plasticity=plasticity, batch_size=batch_size, forget_rate=forget_rate,
            ephemeral_fraction=ephemeral_fraction, slow_weight_decay=slow_weight_decay,
            fast_weight_clamp=fast_weight_clamp, slow_update_every=slow_update_every
        )
        self.i2o.normalize_step = readout_nlms
        for layer in self.trained_layers():
            layer.slow_nlms = slow_nlms
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.updater = updater
        for layer in self.trained_layers():
            layer.retain_sequence_bias_grads = retain_sequence_bias_grads
        self.grad_clip_stats = GradNormClipStats()
        self.fused_layer_step = None  # set by enable_fused_update (--fused_update)
        set_dfa_fprime(self, dfa_fprime)

    def enable_fused_update(self, compile=True):
        """--fused_update: train.py's DFA step goes through fused_dfa_step. compile=False runs the
        same dfa_layer_step eagerly, which is bit-identical to the unfused step when
        --grad_norm_clip is off (tests/test_fused_update.py)."""
        if self.updater != 'dfa':
            raise ValueError(f"--fused_update supports only the DFA updater, not {self.updater!r}")
        self.fused_layer_step = torch.compile(dfa_layer_step, dynamic=False) if compile else dfa_layer_step

    @torch.no_grad()
    def fused_dfa_step(self, output_error, learning_rate, update_clamp, grad_norm_clip=0):
        """The DFA step train.py takes (populate_dfa_gradients on every layer, the optional
        per-sequence --grad_norm_clip, apply_update, then apply_forget_step), with each layer's
        update done by self.fused_layer_step and no gradient tensor materialized.

        Compiled, the math is the same but the rounding is not (fused multiply-adds and another
        evaluation order: about 1e-7 relative per step). With --grad_norm_clip the norms use the
        closed form for a rank-1 gradient, |p_b x_b^T| = |p_b| |x_b|, and the clip scales the
        projected error, so the clipped step matches the unfused one only to rounding."""
        layers = self.trained_layers()
        projected, norms = self.dfa_step_errors(output_error, grad_norm_clip)
        if norms is not None:
            self.grad_clip_stats.record(norms, grad_norm_clip)
        for layer, error in zip(layers, projected):
            layer._last_projected_error = error
            self.fused_layer_step(
                layer.per_sample_weights.data, None if layer.bias is None else layer.bias.data,
                error, layer.in_traces.data, layer.fused_plasticity(), layer.ephemeral_mask,
                layer.forget_rate, learning_rate, update_clamp,
                layer.weight_clamp, layer.is_last_layer, layer.slow_weight_decay,
                layer.fast_weight_clamp, slow_nlms=layer.slow_nlms)

    def dfa_step_errors(self, output_error, grad_norm_clip=0):
        """Each trained layer's projected error for one DFA step, and the pre-clip per-sequence
        norms (None without --grad_norm_clip). The clip uses the closed form for a rank-1
        gradient, |p_b x_b^T| = |p_b| |x_b|, and scales the projected errors."""
        layers = self.trained_layers()
        projected = [dfa_projected_error(output_error, layer.feedback_weights, layer.is_last_layer,
                                         dfa_layer_activation_derivative(layer))
                     for layer in layers]
        projected = [normalized_readout_error(error, layer.in_traces.data,
                                              layer.is_last_layer and layer.normalize_step)
                     for layer, error in zip(layers, projected)]
        if grad_norm_clip <= 0:
            return projected, None
        squared = None
        for layer, error in zip(layers, projected):
            # weight gradient |p_b|^2 |x_b|^2, plus the bias share |p_b|^2
            term = error.square().sum(1) * (layer.in_traces.data.square().sum(1)
                                            + (1 if layer.bias is not None else 0))
            squared = term if squared is None else squared + term
        norms = squared.sqrt()
        scale = per_sequence_clip_scale(norms, grad_norm_clip)
        return [error * scale.unsqueeze(1) for error in projected], norms

    def check_fast_only_step(self):
        """fast_only_dfa_step needs the DFA updater."""
        if self.updater != 'dfa':
            raise ValueError("held-out evaluation needs an EphemeralRNN trained with --updater dfa: "
                             "its fast writes are the DFA step")

    @torch.no_grad()
    def fast_only_dfa_step(self, output_error, learning_rate, update_clamp, grad_norm_clip=0,
                           forget=True):
        """The DFA step of fused_dfa_step (the same projected errors, clip, update, clamps and
        forgetting, through dfa_layer_step) restricted to the ephemeral entries: slow entries,
        biases and i2o stay bit for bit, and --slow_weight_decay does not act. Rows whose
        output_error is zero only forget. Used by heldout.py; it records no clip statistics.
        forget=False leaves out the forgetting (the extra passes of --fast_backward_per_forward K)."""
        self.check_fast_only_step()
        projected, _ = self.dfa_step_errors(output_error, grad_norm_clip)
        for layer, error in zip(self.trained_layers(), projected):
            if layer.is_last_layer:
                continue  # i2o has no ephemeral entries: the frozen step would leave it unchanged
            self._fast_entry_step(layer, error, learning_rate, update_clamp, forget)

    def _fast_entry_step(self, layer, error, learning_rate, update_clamp, forget=True):
        """dfa_layer_step with freeze_slow: this step on the ephemeral entries only (bias frozen).
        forget=False: no forgetting (forget rate 0; decay 0, which only acts on slow entries)."""
        step = self.fused_layer_step or dfa_layer_step
        step(layer.per_sample_weights.data, None, error, layer.in_traces.data,
             layer.fused_plasticity(), layer.ephemeral_mask, layer.forget_rate if forget else 0.0,
             learning_rate, update_clamp, layer.weight_clamp, layer.is_last_layer,
             layer.slow_weight_decay if forget else 0.0, layer.fast_weight_clamp, True,
             slow_nlms=layer.slow_nlms)

    @property
    def split_step(self):
        """K >= 2 (or the testing hook): a character's DFA step is split in two, see split_dfa_step."""
        return self.fast_iterations > 1 or self._force_split_step

    @torch.no_grad()
    def split_dfa_step(self, output_error, learning_rate, update_clamp, grad_norm_clip=0,
                       log_norms=False):
        """--fast_backward_per_forward K >= 2, --slow_update_every 1: the first half of a character's
        DFA step. The fast entries take pass 1's step now (the same projected errors, clip, alpha,
        clamps and the character's one forgetting step, as fast_only_dfa_step). Everything slow
        waits: pass 1's projected errors and layer inputs are saved and returned, and
        apply_slow_step applies the slow half from them after the extra passes, so every pass sees
        the same slow weights and the slow gradient is pass 1's, from the weights before the step."""
        projected, norms = self.dfa_step_errors(output_error, grad_norm_clip)
        if norms is not None:
            self.grad_clip_stats.record(norms, grad_norm_clip)
        saved = []
        for layer, error in zip(self.trained_layers(), projected):
            layer._last_projected_error = error
            inputs = layer.in_traces.data
            if log_norms:
                gradient = normalize_slow_gradient(
                    dfa_per_sample_gradient(error, inputs), inputs,
                    layer.ephemeral_mask, layer.slow_nlms)
                layer._log_update_norms(ephemeral_update(
                    gradient, layer.plasticity,
                    layer.ephemeral_mask, update_clamp, layer.is_last_layer))
            saved.append((error, inputs))
            if not layer.is_last_layer:
                self._fast_entry_step(layer, error, learning_rate, update_clamp)
        return saved

    @torch.no_grad()
    def apply_slow_step(self, saved, learning_rate, update_clamp):
        """The second half of split_dfa_step: every layer's slow entries (i2o included), bias,
        --weight_clamp and --slow_weight_decay (which rides on the forget step in the fused
        step), from the saved pass-1 errors and inputs. The fast entries are left as they are:
        their forgetting was in the first half."""
        step = self.fused_layer_step or dfa_layer_step
        for layer, (error, inputs) in zip(self.trained_layers(), saved):
            step(layer.per_sample_weights.data, None if layer.bias is None else layer.bias.data,
                 error, inputs, layer.fused_plasticity(), layer.ephemeral_mask,
                  layer.forget_rate, learning_rate, update_clamp, layer.weight_clamp,
                  layer.is_last_layer, layer.slow_weight_decay, layer.fast_weight_clamp,
                  False, True, False, slow_nlms=layer.slow_nlms)

    @torch.no_grad()
    def extra_fast_iterations(self, input, hidden, target, criterion, learning_rate, update_clamp,
                              grad_norm_clip=0, tracer=None, step=0):
        """--fast_backward_per_forward K >= 2, after the character's first pass: K - 1 times,
        forward the same input from the same incoming hidden state with the fast weights as the
        last step left them (the slow weights are still the pre-step ones), take the output error
        of that fresh output, and apply a fast-only DFA step without forgetting (forgetting
        happened once, in the first pass). Slow entries, i2o and biases do not change, and no clip
        statistics or loss are recorded. Returns the hidden state of the final pass (the incoming
        one if K = 1, for the testing hook). tracer (loop_trace.py), if given, sees each extra
        pass (tracer.after_pass(step, pass_index, ...)) after its forward and before its write;
        it only reads."""
        final_hidden = None
        for extra in range(self.fast_iterations - 1):
            output, final_hidden = self(input, hidden)
            loss, error = dfa_output_error(output, target, criterion)
            if tracer is not None:
                tracer.after_pass(step, extra + 1, output, error, loss)
            self.fast_only_dfa_step(error, learning_rate, update_clamp, grad_norm_clip, forget=False)
        return final_hidden

    @torch.no_grad()
    def skip_fast_dfa_step(self, output_error, learning_rate, update_clamp, grad_norm_clip=0,
                           log_norms=False):
        """The per-step DFA step (--slow_update_every 1) on a character that gets no fast update
        (--fast_backward_per_forward 1/N): the slow entries, biases and i2o take their ordinary
        step, and the ephemeral entries only forget. Through dfa_layer_step (compiled if
        --fused_update), so it matches the unfused step to rounding."""
        projected, norms = self.dfa_step_errors(output_error, grad_norm_clip)
        if norms is not None:
            self.grad_clip_stats.record(norms, grad_norm_clip)
        step = self.fused_layer_step or dfa_layer_step
        for layer, error in zip(self.trained_layers(), projected):
            layer._last_projected_error = error
            if log_norms:
                gradient = normalize_slow_gradient(
                    dfa_per_sample_gradient(error, layer.in_traces.data), layer.in_traces.data,
                    layer.ephemeral_mask, layer.slow_nlms)
                layer._log_update_norms(ephemeral_update(
                    gradient, layer.plasticity,
                    layer.ephemeral_mask, update_clamp, layer.is_last_layer))
            step(layer.per_sample_weights.data, None if layer.bias is None else layer.bias.data,
                 error, layer.in_traces.data, layer.fused_plasticity(), layer.ephemeral_mask,
                  layer.forget_rate, learning_rate, update_clamp, layer.weight_clamp,
                  layer.is_last_layer, layer.slow_weight_decay, layer.fast_weight_clamp,
                  False, True, slow_nlms=layer.slow_nlms)

    @torch.no_grad()
    def windowed_dfa_step(self, output_error, learning_rate, update_clamp, grad_norm_clip=0,
                          log_norms=False, skip_fast=False, defer_apply=False):
        """The DFA step under --slow_update_every N or 'sequence' (train.py's DFA branch).

        The fast entries take this step's update exactly as under the per-step update (the same
        projected errors, --grad_norm_clip, alpha, clamps and forgetting, through dfa_layer_step
        with freeze_slow, as fast_only_dfa_step), fused if --fused_update. Everything slow (the
        slow entries of every layer, i2o included, and every bias) is left as it is, and this
        step's gradient is accumulated instead (accumulate_slow_gradient). After N steps, and at
        the end of every sequence (train.py calls apply_pending_slow_update), the window's sum is
        applied (EphemeralLinear.apply_slow_window). log_norms also records the per-step update
        norms the per-step path logs (the slow part as the would-be per-step update).
        skip_fast (--fast_backward_per_forward 1/N, a character with no fast update): the fast
        entries only forget (a step with zero error), the slow gradient is accumulated as usual.
        defer_apply (K >= 2): a window that fills up is not applied here; the caller calls
        finish_slow_window after the extra passes, so they see the same slow weights."""
        projected, norms = self.dfa_step_errors(output_error, grad_norm_clip)
        if norms is not None:
            self.grad_clip_stats.record(norms, grad_norm_clip)
        slot = self.pending_slow_steps
        for layer, error in zip(self.trained_layers(), projected):
            layer._last_projected_error = error
            if log_norms:
                gradient = normalize_slow_gradient(
                    dfa_per_sample_gradient(error, layer.in_traces.data), layer.in_traces.data,
                    layer.ephemeral_mask, layer.slow_nlms)
                layer._log_update_norms(ephemeral_update(
                    gradient, layer.plasticity,
                    layer.ephemeral_mask, update_clamp, layer.is_last_layer))
            slow_error = normalized_readout_error(error, layer.in_traces.data, layer.slow_nlms)
            layer.accumulate_slow_gradient(slow_error, slot)
            if not layer.is_last_layer:
                self._fast_entry_step(layer, torch.zeros_like(error) if skip_fast else error,
                                      learning_rate, update_clamp)
        self.pending_slow_steps += 1
        if not defer_apply:
            self.finish_slow_window(learning_rate)

    @torch.no_grad()
    def finish_slow_window(self, learning_rate):
        """Applies the window if it is full (windowed_dfa_step's last move, deferred for K >= 2)."""
        if self.pending_slow_steps == self.slow_update_every:
            self.apply_pending_slow_update(learning_rate)

    @torch.no_grad()
    def apply_pending_slow_update(self, learning_rate):
        """Applies the current --slow_update_every window's accumulated slow step (if any) to
        every layer and starts a new window. train.py calls it at the end of every sequence, so
        a window never spans a wipe and nothing is pending between batches (or in a checkpoint)."""
        for layer in self.trained_layers():
            layer.apply_slow_window(learning_rate, self.pending_slow_steps)
        self.pending_slow_steps = 0

    def trained_layers(self):
        """Every EphemeralLinear, in update order: the hidden layers, i2h, then i2o."""
        return [*self.linear_layers, self.i2h, self.i2o]

    def apply_regularization(self):
        """--weight_clamp (and --fast_weight_clamp) on every layer's per_sample_weights. DFA and
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
            if self.layer_norm:
                combined = trunk_layer_norm(combined)
            # combined = self.dropout(combined)

        # Add the residual (original combined tensor) to the output of the layers
        # print(f"residual_shape: {residual.shape}, combined shape: {combined.shape}")
        if self.residual_connection:
            combined += residual

        # Forked transition/emission layout. The state head remains bounded for recurrence,
        # while the output head reads a sibling transform of the shared deep representation.
        hidden_t = torch.tanh(self.i2h(combined))
        output = self.i2o(torch.tanh(combined) if self.output_tanh else combined)
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

    def start_sequence_wipe(self, wipe_fast=True):
        """Calls start_sequence_wipe on all EphemeralLinear layers: the slow entries always take
        the batch mean; the fast entries are zeroed only if wipe_fast (see --wipe_every)."""
        for layer in self.linear_layers:
            layer.start_sequence_wipe(wipe_fast)
        self.i2h.start_sequence_wipe(wipe_fast)
        self.i2o.start_sequence_wipe(wipe_fast)

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
                 weight_clamp=0, slow_weight_decay=0):
        super().__init__(in_features, out_features, bias)
        self.weight_clamp = weight_clamp
        self.slow_weight_decay = slow_weight_decay
        self.is_last_layer = False
        self.register_buffer('feedback_weights', None)  # set by enable_dfa for non-last layers
        self.in_traces = None  # this step's input, recorded by forward (not saved)
        # --dfa_fprime (set by SimpleRNN): out_traces, this step's pre-activation, is then
        # recorded too (not saved), and activation names the model's nonlinearity after this layer.
        self.dfa_fprime = False
        self.activation = None
        self.out_traces = None
        self._last_projected_error = None

    def enable_dfa(self, vocab_size, is_last_layer):
        """Gives a non-last layer its fixed random feedback matrix [vocab, out], initialised as
        EphemeralLinear's feedback_weights; a last layer (i2o) gets the output error directly."""
        self.is_last_layer = is_last_layer
        if not is_last_layer:
            self.feedback_weights = init_feedback_weights(vocab_size, self.out_features).to(self.weight.device)

    def forward(self, input):
        self.in_traces = input.detach()
        output = super().forward(input)
        if self.dfa_fprime:
            self.out_traces = output.detach()
        return output

    def pre_activation(self):
        """This step's pre-activation (the output before the model's nonlinearity), [B, out]."""
        return self.out_traces

    def populate_dfa_gradients(self, error_signal):
        """Sets weight.grad to the batch mean of the per-sequence DFA gradients and bias.grad to
        the batch mean of the projected error. error_signal is not modified."""
        projected_error = dfa_projected_error(error_signal, self.feedback_weights, self.is_last_layer,
                                              dfa_layer_activation_derivative(self))
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
        self.weight.data = regularized_weight(self.weight.data, self.weight_clamp)
        # --slow_weight_decay after every update, as the ephemeral model's forget step decays its
        # slow entries (every SimpleRNN weight is slow). Biases are excluded in both models.
        if self.slow_weight_decay:
            self.weight.data.mul_(1 - self.slow_weight_decay)


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_rate=0.1,
                 init_type='zero', enable_recurrence=True, updater=None,
                 residual_connection=False, weight_clamp=0,
                 slow_weight_decay=0, output_tanh=False, layer_norm=False, dfa_fprime=False):
        """updater: 'dfa' gives the hidden layers and i2h fixed random DFA feedback matrices (drawn
        after every layer is initialised, so the layers start the same as under the other
        updaters at the same seed). Other values leave it a plain backprop/BPTT model.
        dfa_fprime: --dfa_fprime, see set_dfa_fprime."""
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.init_type = init_type
        self.enable_recurrence = enable_recurrence
        self.output_tanh = output_tanh
        self.layer_norm = layer_norm  # --layer_norm, as in EphemeralRNN (trunk_layer_norm)
        self.updater = updater
        self.residual_connection = residual_connection
        inner_size = recurrent_trunk_size(input_size, hidden_size)

        # Standard linear layers (DFALinear is an nn.Linear that can also take DFA updates)
        layer_options = {"weight_clamp": weight_clamp, "slow_weight_decay": slow_weight_decay}
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
        set_dfa_fprime(self, dfa_fprime)

    def dfa_layers(self):
        """Every layer DFA trains, in update order: the hidden layers, i2h, then i2o."""
        return [*self.linear_layers, self.i2h, self.i2o]

    def apply_regularization(self):
        """Apply the configured weight clamp (and slow-weight decay) after an SGD step."""
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
            if self.layer_norm:
                combined = trunk_layer_norm(combined)
            # combined = self.dropout(combined)

        if self.residual_connection:
            combined += residual

        # Forked transition/emission layout, as in EphemeralRNN. The output is independent of
        # this step's state head; --enable_recurrence False feeds back zeros.
        hidden_t = torch.tanh(self.i2h(combined))
        output = self.i2o(torch.tanh(combined) if self.output_tanh else combined)
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
