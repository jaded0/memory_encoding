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


def readout_direction_error(output_error, readout_weights, pre_activation, slow_error, scale=1.0):
    """--fast_write_from_readout: the last trunk layer's fast-write error along the exact readout
    gradient, [B, out]. The direction is (e^T W_o) * GELU'(a) per sequence, with e the output error,
    W_o i2o's per-sequence weights before this step's update and a the layer's pre-activation (the
    exact gradient of the loss with respect to a when i2o reads GELU(a) directly). It is rescaled
    to scale times the norm of the layer's ordinary DFA error slow_error, so the writes keep their
    DFA size and only change direction (--fast_write_from_readout_scale, the 10-13 "x s")."""
    direction = torch.bmm(output_error.unsqueeze(1), readout_weights).squeeze(1)
    direction = direction * dfa_activation_derivative(pre_activation, 'gelu')
    return direction * (scale * slow_error.norm(dim=1, keepdim=True)
                        / (direction.norm(dim=1, keepdim=True) + 1e-12))


def decoupled_fast_feedback(vocab_size, out_features, layer_index):
    """--decouple_fast_feedback: the separate feedback matrix B_fast for trunk layer layer_index's
    fast writes, [vocab, out], drawn from its own generator (seed 918273 + layer_index) so the
    global RNG stream, and with it everything else in the run, is unchanged. Xavier-normal scale,
    like init_feedback_weights. Deterministic, so it is not stored in checkpoints."""
    generator = torch.Generator().manual_seed(918273 + layer_index)
    return (torch.randn(vocab_size, out_features, generator=generator)
            * math.sqrt(2.0 / (vocab_size + out_features)))


FAST_READ_NORMS = ('off', 'running', 'rms', 'ln', 'absrun')
# Modes that keep per-sequence state over the sequence's earlier reads (one write per forward).
FAST_READ_NORMS_WITH_HISTORY = ('running', 'absrun')


def fast_read_scale(mode, fast, slow, key_mass=None, input=None, coef=0.0, eps=1e-12):
    """--fast_read_norm: the per-row factor [B, 1] a layer multiplies its fast read F x by (F = the
    ephemeral entries; fast and slow are F x and S x, [B, out], bias excluded). Never above 1, so
    it only ever quiets the fast read.

    running: 1 / max(1, n), n = coef * (key_mass . |x|), coef = lr * alpha * (fraction of fast
      entries) and key_mass the decayed running sum of |x_s| over the sequence's earlier steps. n
      bounds (in expectation over the mask) the total key-query weight the read sums its stored
      values with, so above 1 the read becomes a weighted average of the stored values, as in
      normalized linear attention; below 1 it is left alone.
    absrun: 1 / max(1, n), n = coef * key_mass, where key_mass [B, 1] is already the decayed sum
      of the absolute key-query scores sum_s w_s |x_s . x| (fast_abs_score_mass). The same floor
      as running, with the exact score magnitude in place of running's element-wise bound
      sum_j |x_s,j| |x_j| >= |x_s . x|, so only stored keys that actually match the query count.
    rms: min(1, rms(S x) / rms(F x)): the fast part is never louder than the slow part.
    ('ln' is not a scale; see fast_read_layer_norm.)"""
    if mode == 'running':
        n = coef * (key_mass * input.abs()).sum(dim=1, keepdim=True)
        return 1.0 / n.clamp_min(1.0)
    if mode == 'absrun':
        return 1.0 / (coef * key_mass).clamp_min(1.0)
    if mode == 'rms':
        fast_rms = fast.pow(2).mean(dim=1, keepdim=True).sqrt()
        slow_rms = slow.pow(2).mean(dim=1, keepdim=True).sqrt()
        return (slow_rms / fast_rms.clamp_min(eps)).clamp(max=1.0)
    raise ValueError(f"--fast_read_norm {mode!r}: expected one of {FAST_READ_NORMS}")


FAST_READ_LN_EPS = 1e-5


def fast_read_layer_norm(fast):
    """--fast_read_norm ln: LayerNorm of the fast read alone, per batch row over the layer's
    outputs, (F x - mean) / sqrt(var + 1e-5), with no gain or bias (as --layer_norm's
    trunk_layer_norm). F x = 0 (an empty fast store, e.g. the first step of a sequence) reads 0.
    Unlike the scale modes it sets the fast read's size instead of capping it: any non-negligible
    F x reads back with rms ~1 per unit, so a faint read is amplified and a loud one quieted."""
    return F.layer_norm(fast, fast.shape[-1:], eps=FAST_READ_LN_EPS)


def fast_abs_score_mass(keys, key_weights, input):
    """--fast_read_norm absrun's normalizer before its coefficient, [B, 1]: sum_s w_s |x_s . x| over
    the keys stored this sequence (keys [B, S, in], or None for none yet, then 0), with w_s
    (key_weights [S]) the decay the fast entries have applied to step s's write by now."""
    if keys is None:
        return input.new_zeros(input.shape[0], 1)
    scores = torch.bmm(keys, input.unsqueeze(2)).squeeze(2)  # [B, S]
    return (scores.abs() * key_weights).sum(dim=1, keepdim=True)
ORDER2_MAPS = ('order2', 'order1')
ORDER2_MODES = ('parallel', 'replace')


def order2_projection(in_features, dim, seed, layer_index):
    """--order2_layers: the fixed random projection [dim, in] of a layer's input that the order-two
    feature map expands, drawn from its own generator (seeded by the run's seed and the layer) so the
    global RNG stream, and with it every other initial weight, is the same as without the flag. Its
    scale does not matter: order2_features normalizes the projected vector."""
    generator = torch.Generator().manual_seed((int(seed) % 2**31) * 101 + 31337 + int(layer_index))
    return torch.randn(dim, in_features, generator=generator)


def order2_kernel_norm(order, temperature):
    """The order-two (or order-one) kernel at a perfect match, k(1): 1 + t + t^2/2 (or 1 + t)."""
    return 1 + temperature + temperature ** 2 / 2 if order == 'order2' else 1 + temperature


def order2_feature_size(order, dim):
    """The number of features order2_features gives: 1 + d + d^2 (order2, Based's second-order Taylor
    map: 273 for d = 16) or 1 + d (order1)."""
    return 1 + dim + dim * dim if order == 'order2' else 1 + dim


def order2_features(x, projection, order, temperature, fraction, eps=1e-12):
    """The features psi(x), [B, F], a layer's order-two fast matrix reads and writes with.

    z = sqrt(t) * P x / |P x| (t = --order2_temperature), so z.z' = t * c with c the cosine of the
    projected inputs. phi(z) = [1, z, vec(z z^T) / sqrt(2)] (order2) or [1, z] (order1), so that
    phi(z).phi(z') = 1 + t c + t^2 c^2 / 2, the first three terms of exp(t c) (order1: 1 + t c).
    psi(x) = sqrt(fraction / k(1)) * |x| * phi(z), which makes the kernel
        psi(x).psi(y) = fraction * |x| |y| * k(c) / k(1),
    the linear masked read's own expected key factor fraction * x.y = fraction |x| |y| c with the
    cosine c replaced by k(c) / k(1). A perfect match therefore reads exactly as large as on the
    linear fast entries, at the same alpha, learning rate and forget rate, and with the same
    dependence on |x|; only the profile over cosines differs (size-matched by construction).
    A zero input gives zero features."""
    u = x @ projection.t()
    z = u * (math.sqrt(temperature) / u.norm(dim=1, keepdim=True).clamp_min(eps))
    parts = [torch.ones_like(z[:, :1]), z]
    if order == 'order2':
        parts.append((z.unsqueeze(2) * z.unsqueeze(1)).flatten(1) * _INV_SQRT2)
    phi = torch.cat(parts, dim=1)
    scale = math.sqrt(fraction / order2_kernel_norm(order, temperature)) * x.norm(dim=1, keepdim=True)
    return phi * scale


def order2_step(fast, error, features, alpha, learning_rate, forget_rate, update_clamp, weight_clamp,
                fast_weight_clamp):
    """A layer's order-two fast matrix G [B, out, F] takes its DFA write in place, in the order the
    linear fast entries take theirs (dfa_layer_step): -alpha * (error outer features), the
    element-wise --ephemeral_update_clamp, G + lr * update, --weight_clamp and --fast_weight_clamp,
    then forgetting, G <- (1 - forget_rate) G. error is the layer's fast-write error [B, out] (its
    projected DFA error, or the --decouple_fast_feedback / --fast_write_from_readout override), so the
    write is "error times whatever this layer sees", with psi(x) in place of x. A zero error row
    only forgets."""
    update = -alpha * (error.unsqueeze(2) * features.unsqueeze(1))
    if update_clamp > 0:
        update = update.clamp(-update_clamp, update_clamp)
    updated = regularized_weight(fast + learning_rate * update, weight_clamp)
    if fast_weight_clamp:
        updated = updated.clamp(-fast_weight_clamp, fast_weight_clamp)
    fast.copy_(updated * (1 - forget_rate))


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


def normalize_slow_gradient(gradient, inputs, ephemeral_mask, enabled, eps=1e-6, slow_scale=1.0):
    """--slow_nlms: divide only slow weight entries by eps + ||x||^2 per sequence.
    slow_scale (--readout_slow_lr_scale / --trunk_slow_lr_scale) multiplies the slow entries
    too, so the layer's slow step is slow_scale * lr while its fast writes are unchanged."""
    if not enabled and slow_scale == 1:
        return gradient
    if enabled:
        scale = (slow_scale / (inputs.square().sum(1) + eps)).view(-1, 1, 1)
    else:
        scale = slow_scale
    return torch.where(ephemeral_mask.unsqueeze(0), gradient, gradient * scale)


def slow_bias_error(projected_error, inputs, slow_nlms, slow_scale=1.0):
    """The per-sequence error the (slow) bias steps with: --slow_nlms's NLMS factor and the
    layer's slow_scale applied to projected_error. Both off returns the same object."""
    error = normalized_readout_error(projected_error, inputs, slow_nlms)
    return error if slow_scale == 1 else error * slow_scale


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


def dfa_step_gradient(projected_error, inputs, ephemeral_mask, slow_nlms=False, slow_lr_scale=1.0,
                      fast_projected_error=None):
    """A layer's DFA step gradient, [B, out, in], before alpha and the clamps: the outer product
    of projected_error with the inputs, its ephemeral entries from fast_projected_error instead
    when one is given (see dfa_layer_step), then --slow_nlms and the slow lr scale on the slow
    entries."""
    gradient = dfa_per_sample_gradient(projected_error, inputs)
    if fast_projected_error is not None:
        gradient = torch.where(ephemeral_mask.unsqueeze(0),
                               dfa_per_sample_gradient(fast_projected_error, inputs), gradient)
    return normalize_slow_gradient(gradient, inputs, ephemeral_mask, slow_nlms,
                                   slow_scale=slow_lr_scale)


def dfa_layer_step(weights, bias, projected_error, inputs, plasticity, ephemeral_mask,
                   forget_rate: float, learning_rate: float, update_clamp: float,
                   weight_clamp: float, is_last_layer: bool,
                   slow_weight_decay: float = 0.0, fast_weight_clamp: float = 0.0,
                   freeze_slow: bool = False, freeze_fast: bool = False, fast_forget: bool = True,
                   slow_nlms: bool = False, slow_lr_scale: float = 1.0,
                   fast_projected_error=None):
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
    whose fast half already forgot).

    fast_projected_error (--decouple_fast_feedback, --fast_write_from_readout): the ephemeral
    entries' gradient uses this error instead of projected_error; the slow entries and the bias
    keep projected_error. Every operation after the gradient is element-wise, so this is the same
    as a fast-only step with the one error followed by a slow-only step with the other."""
    gradient = dfa_step_gradient(projected_error, inputs, ephemeral_mask, slow_nlms, slow_lr_scale,
                                 fast_projected_error)
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
        bias_error = slow_bias_error(projected_error, inputs, slow_nlms, slow_lr_scale)
        bias.add_(dfa_bias_update(bias_error, learning_rate))


FAST_STRUCTURES = ('exclusive', 'additive_masked', 'additive_dense')


def parse_fast_structure(value):
    """--fast_structure: how a layer's fast weights sit on its slow weights.

    exclusive (the default, and the only structure before 2026-10-22): a fast connection holds only
    its fast value, so the slow weight under it is absent; W = S * (1 - m) + F * m.
    additive_masked: every connection has a slow weight S and the fast part F lives on the same
    fixed random mask m, W = S + F * m.
    additive_dense: F on every connection, W = S + F (the fast-weight-programmer form, whose fast
    part is an exact sum of outer products).

    Matched gain (README "Fast weights on top of slow ones"): the expected fast read of one write
    is lr * alpha * phi_eff * (x . q) p, with phi_eff = --ephemeral_fraction for exclusive and
    additive_masked and 1 for additive_dense, so alpha_dense = fraction * alpha_exclusive matches
    the exclusive arm's c-phi."""
    if value not in FAST_STRUCTURES:
        raise ValueError(f"--fast_structure takes one of {FAST_STRUCTURES}, not {value!r}")
    return value


def dfa_additive_layer_step(slow, fast, bias, projected_error, inputs, alpha: float, fast_mask, no_mask,
                            forget_rate: float, learning_rate: float, update_clamp: float,
                            weight_clamp: float, slow_weight_decay: float = 0.0,
                            fast_weight_clamp: float = 0.0, fast_only: bool = False,
                            fast_forget: bool = True):
    """One additive EphemeralLinear's whole DFA step, in place (--fast_structure additive_*),
    W = S + F. Two calls of dfa_layer_step, so there is no new update arithmetic:

    - the slow part S (skipped if fast_only: held-out evaluation and the fast-only step): an empty
      mask and alpha 1, so every entry takes S <- (1 - slow_weight_decay) clamp_wc(S - lr g), and
      the bias takes its step;
    - the fast part F: freeze_slow on fast_mask, so the entries on the mask take
      F <- (1 - forget_rate) clamp_fwc(F + lr clamp_uc(-alpha g)) and every other entry of F stays
      exactly 0. --weight_clamp acts on S only; --fast_weight_clamp acts on F (the analogue of the
      exclusive structure's weight clamp on its fast entries).

    g = p x^T per sequence with the same projected error p for both parts (one shared feedback
    channel, as in the exclusive structure). fast_mask is the layer's random mask (additive_masked)
    or all ones (additive_dense); no_mask is all False. fast_forget=False leaves out F's forgetting
    (the forget-free fast-only step EphemeralRNN.fast_only_dfa_step takes with forget=False)."""
    if not fast_only:
        dfa_layer_step(slow, bias, projected_error, inputs, 1.0, no_mask, 0.0, learning_rate, 0.0,
                       weight_clamp, False, slow_weight_decay)
    dfa_layer_step(fast, None, projected_error, inputs, alpha, fast_mask,
                   forget_rate if fast_forget else 0.0, learning_rate, update_clamp, 0.0, False, 0.0,
                   fast_weight_clamp, True)


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
    def __init__(self, in_features, out_features, charset, bias=True, weight_clamp=0, updater='dfa', requires_grad=False, is_last_layer=False, plasticity=1, batch_size=1, forget_rate=0.01, ephemeral_fraction=0.2, slow_weight_decay=0, fast_weight_clamp=0, slow_update_every=1, fast_structure='exclusive'):
        """forget_rate: fraction of each ephemeral weight removed per forget step,
        w <- (1 - forget_rate) * w (see apply_forget_step). The paper's "forgetting rate
        coefficient 0.7" is 1 - forget_rate, i.e. forget_rate = 0.3. Same meaning as --forget_rate.
        plasticity: alpha, the learning-rate multiplier on the ephemeral entries (--plasticity).
        ephemeral_fraction: fraction of entries that are ephemeral (--ephemeral_fraction).
        weight_clamp: applied after each update (--weight_clamp; see _apply_regularization).
        slow_update_every: --slow_update_every, 1 (every step), N or 'sequence'. Anything but 1
        allocates the window's accumulators (see accumulate_slow_gradient).
        fast_structure: --fast_structure (parse_fast_structure). The additive structures keep S in
        per_sample_weights (every entry slow, none zeroed at init) and F in the extra fast_state
        parameter; they need the DFA updater and slow_update_every 1. A last layer (i2o) and a
        layer with ephemeral_fraction 0 (--fast_layers) have no fast weights and are always
        'exclusive', which for them is the same thing: all-slow."""
        super(EphemeralLinear, self).__init__(in_features, out_features, bias)
        fast_structure = parse_fast_structure(fast_structure)
        self.fast_structure = ('exclusive' if is_last_layer or ephemeral_fraction == 0
                               else fast_structure)
        if self.is_additive:
            if updater != 'dfa':
                raise ValueError(f"--fast_structure {fast_structure} supports only the DFA updater, "
                                 f"not {updater!r}")
            if parse_slow_update_every(slow_update_every) != 1:
                raise ValueError(f"--fast_structure {fast_structure} does not support "
                                 f"--slow_update_every {slow_update_every}")

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
        self.slow_lr_scale = 1.0  # --readout_slow_lr_scale (i2o) / --trunk_slow_lr_scale (others)
        # --dfa_fprime (set by EphemeralRNN): scale the projected DFA error by f'(pre-activation),
        # where activation names the nonlinearity the model applies to this layer's output.
        self.dfa_fprime = False
        self.activation = None
        # --decouple_fast_feedback / --fast_write_from_readout (set by EphemeralRNN): where this
        # layer's fast writes take their error from instead of feedback_weights (see
        # EphemeralRNN.fast_write_errors). Not persistent: B_fast is redrawn deterministically.
        self.register_buffer('fast_feedback_weights', None, persistent=False)
        self.fast_write_from_readout = False
        # --fast_read_norm (set by EphemeralRNN.set_fast_read_norm on layers with fast entries):
        # 'off' keeps the one-bmm forward exactly; otherwise forward splits the read into S x + F x
        # and rescales F x by fast_read_scale (or, for 'ln', normalizes it). fast_key_mass is
        # 'running''s decayed sum of |x|; fast_keys [B, S, in] and fast_key_weights [S] are
        # 'absrun''s stored keys of the sequence's earlier steps and their decay weights.
        self.fast_read_norm = 'off'
        self.fast_read_coef = 0.0
        self.register_buffer('fast_key_mass', None, persistent=False)
        self.register_buffer('fast_keys', None, persistent=False)
        self.register_buffer('fast_key_weights', None, persistent=False)
        # --order2_layers (set by EphemeralRNN.set_order2): a separate fast matrix order2_fast
        # [B, out, F] read and written through order2_features of a fixed projection order2_proj.
        # Both are None (and so absent from the state dict) when the flag is off.
        self.order2 = None
        self.order2_replace = False
        self.order2_temperature = 0.0
        self.order2_fraction = 0.0
        self.order2_trace = None
        self.register_buffer('order2_proj', None)
        self.register_buffer('order2_fast', None)
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
            if not self.is_additive:
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

        # --fast_structure additive_*: the fast part F, one dense [out, in] copy per sequence, 0 off
        # its support. A saved parameter, so a checkpoint carries it (a --wipe_every N > 1 run keeps
        # F between batches); exclusive layers have none, so their state dicts are unchanged. The
        # masks are derived from ephemeral_mask (non-persistent, nothing new to save or draw).
        if self.is_additive:
            self.fast_state = nn.Parameter(torch.zeros_like(self.per_sample_weights), requires_grad=False)
            self.register_buffer('no_fast_mask', torch.zeros_like(self.ephemeral_mask.data), persistent=False)
            if self.fast_structure == 'additive_dense':
                self.register_buffer('dense_fast_mask', torch.ones_like(self.ephemeral_mask.data),
                                     persistent=False)

    @property
    def is_additive(self):
        """--fast_structure additive_masked or additive_dense on this layer (W = S + F)."""
        return self.fast_structure != 'exclusive'

    @property
    def fast_mask(self):
        """Where this layer's fast weights live: the random ephemeral mask (exclusive and
        additive_masked) or every connection (additive_dense)."""
        return self.dense_fast_mask if self.fast_structure == 'additive_dense' else self.ephemeral_mask

    def fast_alpha(self):
        """alpha (a float) for the additive fast step: the plasticity of the mask's entries
        (--plasticity, or the --plasticity_schedule value in force)."""
        alpha = self.fused_plasticity()
        if torch.is_tensor(alpha):
            raise ValueError(f"--fast_structure {self.fast_structure} needs a uniform plasticity alpha")
        return alpha

    def fast_weights(self):
        """The fast part of this layer's per-sequence weights, [B, out, in]: the masked entries of
        per_sample_weights (exclusive), or F (additive). A new tensor for exclusive, F itself for
        additive. Used by observers (loop_trace.py), not by the step."""
        if self.is_additive:
            return self.fast_state.data
        return self.per_sample_weights.data * self.ephemeral_mask

    def slow_weights(self):
        """The slow part, [B, out, in]: the unmasked entries (exclusive), or all of S (additive)."""
        if self.is_additive:
            return self.per_sample_weights.data
        return self.per_sample_weights.data * ~self.ephemeral_mask

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
        the last bit. The fast entries are wiped or kept as above.

        Additive structures: every entry of per_sample_weights is slow (S), so all of it takes the
        batch mean; fast_state (F) is zeroed if wipe_fast, else each row keeps its own."""
        if self.is_additive:
            aggregated = self.per_sample_weights.mean(dim=0, keepdim=True)
            self.per_sample_weights.data.copy_(aggregated.expand_as(self.per_sample_weights))
            if wipe_fast:
                self.fast_state.data.zero_()
            self.t.fill_(0.0)
            self._retained_outputs = []
            return
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
            if self.fast_key_mass is not None:
                self.fast_key_mass.zero_()
            self.fast_keys = self.fast_key_weights = None
            if self.order2_fast is not None:
                self.order2_fast.zero_()
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
        if self.is_additive:
            # (S + F) x as S x + F x: two reads of [B, out, in] instead of three plus a write.
            output = torch.baddbmm(torch.bmm(self.per_sample_weights, input_unsq),
                                   self.fast_state, input_unsq).squeeze(2)
        elif self.fast_read_norm == 'off':
            output = torch.bmm(self.per_sample_weights, input_unsq).squeeze(2)
        else:
            output = self._normalized_read(input, input_unsq)
        if self.order2 is not None:
            # --order2_layers: + G psi(x), the order-two fast read (parallel to the linear one, or,
            # under 'replace', the only fast read: the linear fast entries are then never written).
            self.order2_trace = order2_features(input.detach(), self.order2_proj, self.order2,
                                                self.order2_temperature, self.order2_fraction)
            output = output + torch.bmm(self.order2_fast, self.order2_trace.unsqueeze(2)).squeeze(2)

        # Optionally add a bias if needed.
        if self.bias is not None:
            output = output + self.bias
        self.update_imprints(input, output)
        if self.retain_sequence_bias_grads and output.requires_grad:
            output.retain_grad()
            self._retained_outputs.append(output)
        return output

    def _normalized_read(self, input, input_unsq):
        """--fast_read_norm: S x + scale * F x, or S x + LN(F x) for 'ln', [B, out] (bias added by
        forward). The writes are unchanged. 'running' and 'absrun' then fold this step's key into
        their normalizer the way the fast entries take this step's write: add, then forget, so the
        next read's normalizer covers exactly the writes its F holds (one write per forward;
        EphemeralRNN refuses the other step paths)."""
        weights, mask = self.per_sample_weights, self.ephemeral_mask.unsqueeze(0)
        fast = torch.bmm(weights.masked_fill(~mask, 0), input_unsq).squeeze(2)
        slow = torch.bmm(weights.masked_fill(mask, 0), input_unsq).squeeze(2)
        if self.fast_read_norm == 'ln':
            return slow + fast_read_layer_norm(fast)
        if self.fast_read_norm in FAST_READ_NORMS_WITH_HISTORY:
            alpha = self.fused_plasticity()
            alpha = alpha if isinstance(alpha, float) else float(alpha[self.ephemeral_mask].mean())
            coef = self.fast_read_coef * alpha
            key = input.detach()
        if self.fast_read_norm == 'running':
            if self.fast_key_mass is None or self.fast_key_mass.shape != input.shape:
                self.fast_key_mass = torch.zeros_like(input)
            scale = fast_read_scale('running', fast, slow, self.fast_key_mass, input, coef)
            self.fast_key_mass = (1 - self.forget_rate) * (self.fast_key_mass + key.abs())
        elif self.fast_read_norm == 'absrun':
            if self.fast_keys is not None and self.fast_keys.shape[0] != input.shape[0]:
                self.fast_keys = self.fast_key_weights = None
            mass = fast_abs_score_mass(self.fast_keys, self.fast_key_weights, input)
            scale = fast_read_scale('absrun', fast, slow, mass, input, coef)
            new_weight = key.new_ones(1)
            if self.fast_keys is None:
                self.fast_keys, self.fast_key_weights = key.unsqueeze(1), new_weight
            else:
                self.fast_keys = torch.cat([self.fast_keys, key.unsqueeze(1)], dim=1)
                self.fast_key_weights = torch.cat([self.fast_key_weights, new_weight])
            self.fast_key_weights = (1 - self.forget_rate) * self.fast_key_weights
        else:
            scale = fast_read_scale(self.fast_read_norm, fast, slow)
        return slow + scale * fast

    def order2_alpha(self):
        """alpha for the order-two fast matrix: the layer's (uniform) fast plasticity."""
        alpha = self.fused_plasticity()
        return alpha if isinstance(alpha, float) else float(alpha[self.ephemeral_mask].mean())

    def order2_write(self, error, learning_rate, update_clamp, forget=True):
        """This step's DFA write on the order-two fast matrix (order2_step), with the features this
        layer's last forward recorded. forget=False leaves out the forgetting."""
        order2_step(self.order2_fast, error, self.order2_trace, self.order2_alpha(), learning_rate,
                    self.forget_rate if forget else 0.0, update_clamp, self.weight_clamp,
                    self.fast_weight_clamp)

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
            gradient, self.in_traces.data, self.ephemeral_mask, self.slow_nlms,
            slow_scale=self.slow_lr_scale)

        # Biases are slow, so --slow_nlms and the slow lr scale apply to their error too.
        self._last_projected_error = slow_bias_error(
            projected_error, self.in_traces.data, self.slow_nlms, self.slow_lr_scale)

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

        if self.is_additive:
            # The whole additive step (S, bias, F and F's forgetting) in one call;
            # apply_forget_step does nothing for this layer. It uses the projected error
            # populate_dfa_gradients stored, which --grad_norm_clip has already rescaled, and the
            # gradient's own entries are not needed. Update norms are not logged.
            self.additive_step(self._last_projected_error, learning_rate, update_clamp)
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

    def additive_step(self, projected_error, learning_rate, update_clamp, step=None, fast_only=False,
                      fast_forget=True):
        """An additive layer's DFA step (dfa_additive_layer_step) with this layer's settings, in
        place. step is the function to call (EphemeralRNN's compiled one under --fused_update);
        fast_only writes F only (S and the bias frozen); fast_forget=False skips F's forgetting."""
        (step or dfa_additive_layer_step)(
            self.per_sample_weights.data, self.fast_state.data,
            None if self.bias is None or fast_only else self.bias.data, projected_error,
            self.in_traces.data, self.fast_alpha(), self.fast_mask, self.no_fast_mask,
            self.forget_rate, learning_rate, update_clamp, self.weight_clamp,
            self.slow_weight_decay, self.fast_weight_clamp, fast_only, fast_forget)

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
        feedback matrices, traces, and logged norms are intentionally excluded. Additive layers:
        --weight_clamp on S and --fast_weight_clamp on F, as in their step."""
        if self.is_additive:
            regularized_weight(self.per_sample_weights.data, self.weight_clamp)
            if self.fast_weight_clamp:
                self.fast_state.data.clamp_(-self.fast_weight_clamp, self.fast_weight_clamp)
            return
        self.per_sample_weights.data = clamp_fast_entries(
            regularized_weight(self.per_sample_weights.data, self.weight_clamp),
            self.ephemeral_mask, self.fast_weight_clamp)


    def apply_forget_step(self):
        """Decays the ephemeral entries: w <- (1 - forget_rate * ephemeral_mask) * w, element-wise,
        so each call keeps
        1 - forget_rate of every ephemeral weight. train.py calls this after each update (after
        the clamps too), as in the paper: w <- (1 - forget_rate) * (w - lr*alpha*g).
        This is done through .data under no_grad to avoid recording the update in autograd.
        Additive layers forget inside their step (additive_step): nothing to do here."""
        if self.is_additive:
            return
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
        if self.is_additive:
            # S is slow_weight_norm and F is ephemeral_weight_norm (the composite S + F is not
            # formed); additive steps do not record update norms.
            with torch.no_grad():
                slow = torch.norm(self.per_sample_weights.data).item()
                fast = torch.norm(self.fast_state.data).item()
            return {'weight_norm': (slow ** 2 + fast ** 2) ** 0.5, 'ephemeral_weight_norm': fast,
                    'slow_weight_norm': slow, 'ephemeral_update_norm': 0.0, 'slow_update_norm': 0.0}
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

def check_fast_structure(fast_structure, updater='dfa', slow_update_every=1, fast_backward_per_forward=1,
                         slow_nlms=False, trunk_slow_lr_scale=1.0, decouple_fast_feedback=None,
                         fast_write_from_readout=None, fast_read_norm='off'):
    """Raises ValueError for settings the additive structures do not define. They take the per-step
    DFA step (one fast write per character, the slow step every character) with one projected
    error shared by S and F. Refused: other updaters (backprop's alpha scaling and BPTT's update
    are written for the exclusive mask), --slow_update_every N, --fast_backward_per_forward other
    than 1 (other step paths, not ported), --decouple_fast_feedback and --fast_write_from_readout
    (a second fast-write error), --slow_nlms and --trunk_slow_lr_scale (they rescale the slow part
    of the shared error), --fast_read_norm (a split read of the exclusive mask). Accepted:
    --fused_update, --grad_norm_clip (the raw p x^T, as for exclusive), --ephemeral_update_clamp
    and --fast_weight_clamp (on F), --weight_clamp and --slow_weight_decay (on S), --fast_layers,
    --wipe_every, --layer_norm, --dfa_fprime, --readout_nlms and --readout_slow_lr_scale (i2o only,
    which is slow-only in every structure), --plasticity_schedule."""
    if fast_structure == 'exclusive':
        return
    refused = []
    if updater != 'dfa':
        refused.append(f"--updater {updater}")
    if parse_slow_update_every(slow_update_every) != 1:
        refused.append(f"--slow_update_every {slow_update_every}")
    if parse_fast_backward_per_forward(fast_backward_per_forward) != 1:
        refused.append(f"--fast_backward_per_forward {fast_backward_per_forward}")
    if slow_nlms:
        refused.append("--slow_nlms")
    if trunk_slow_lr_scale != 1:
        refused.append(f"--trunk_slow_lr_scale {trunk_slow_lr_scale}")
    if decouple_fast_feedback:
        refused.append("--decouple_fast_feedback")
    if fast_write_from_readout:
        refused.append("--fast_write_from_readout")
    if fast_read_norm != 'off':
        refused.append(f"--fast_read_norm {fast_read_norm}")
    if refused:
        raise ValueError(f"--fast_structure {fast_structure} does not support {', '.join(refused)} "
                         "(see check_fast_structure)")


class EphemeralRNN(torch.nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, num_layers, charset,
        dropout_rate=0, residual_connection=False, init_type='zero',
        weight_clamp=0, updater='dfa',
        plasticity=1, batch_size=1, forget_rate=0.01, ephemeral_fraction=0.2,
        enable_recurrence=True, retain_sequence_bias_grads=False,
        slow_weight_decay=0, output_tanh=False, fast_weight_clamp=0, layer_norm=False,
        dfa_fprime=False, slow_update_every=1, fast_backward_per_forward=1,
        readout_nlms=False, slow_nlms=False, readout_slow_lr_scale=1.0, trunk_slow_lr_scale=1.0,
        fast_layers=None, decouple_fast_feedback=None, fast_write_from_readout=None,
        fast_write_from_readout_scale=1.0, fast_read_norm='off', fast_read_norm_lr=None,
        fast_structure='exclusive',
        order2_layers=None, order2_map='order2', order2_mode='parallel', order2_temperature=4.0,
        order2_dim=16, order2_seed=0
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
        slow_nlms: --slow_nlms, NLMS scaling on each layer's slow entries and bias only.
        readout_slow_lr_scale / trunk_slow_lr_scale: multiply the slow step (slow entries and bias)
        of i2o / of every other trained layer; fast writes are unchanged. 1 is the original step.
        fast_layers: --fast_layers, the trunk layer indices that keep ephemeral entries (None: all);
        the others are all-slow (ephemeral_fraction 0). i2h keeps its entries. The mask draw
        consumes the same random numbers either way.
        decouple_fast_feedback: --decouple_fast_feedback, trunk layer indices whose fast writes
        project the output error through their own feedback matrix B_fast
        (decoupled_fast_feedback) while their slow learning keeps feedback_weights ("DC2").
        fast_write_from_readout / fast_write_from_readout_scale: --fast_write_from_readout, the
        last trunk layer's fast writes follow the exact readout gradient (readout_direction_error,
        "aimed write"), scaled by --fast_write_from_readout_scale.
        fast_read_norm / fast_read_norm_lr: --fast_read_norm (see set_fast_read_norm); 'running'
        and 'absrun' need the learning rate.
        fast_structure: --fast_structure (parse_fast_structure) for the trunk layers that have fast
        entries and i2h; i2o is always the slow-only head. The additive structures take the
        per-step DFA path (unfused or --fused_update, held-out evaluation, the loop tracer); the
        flags that add other step paths or a second fast-write error are refused (see
        check_fast_structure).
        order2_layers / order2_map / order2_mode / order2_temperature / order2_dim / order2_seed:
        --order2_layers and friends, an order-two (or order-one control) fast matrix on a fixed
        projection of those trunk layers' inputs (see set_order2). None: off."""
        super(EphemeralRNN, self).__init__()
        self.fast_structure = parse_fast_structure(fast_structure)
        check_fast_structure(self.fast_structure, updater=updater, slow_update_every=slow_update_every,
                             fast_backward_per_forward=fast_backward_per_forward, slow_nlms=slow_nlms,
                             trunk_slow_lr_scale=trunk_slow_lr_scale,
                             decouple_fast_feedback=decouple_fast_feedback,
                             fast_write_from_readout=fast_write_from_readout,
                             fast_read_norm=fast_read_norm)
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
        layer_sets = {'fast_layers': fast_layers, 'decouple_fast_feedback': decouple_fast_feedback,
                      'fast_write_from_readout': fast_write_from_readout}
        for name, indices in layer_sets.items():
            if indices is not None and any(not 0 <= i < num_layers for i in indices):
                raise ValueError(f"--{name} {list(indices)}: trunk layers are 0..{num_layers - 1}")
        fast_layers = None if fast_layers is None else set(fast_layers)
        decouple_fast_feedback = set(decouple_fast_feedback or ())
        fast_write_from_readout = set(fast_write_from_readout or ())
        if fast_write_from_readout and updater != 'dfa':
            raise ValueError("--fast_write_from_readout needs the DFA updater")
        # Under per-step backprop, B_fast writes the fast entries and backprop trains the slow ones
        # (train.py also requires --backprop_step_parity). BPTT never reads a fast write.
        if decouple_fast_feedback and updater not in ('dfa', 'backprop'):
            raise ValueError("--decouple_fast_feedback needs the DFA or per-step backprop updater")
        if decouple_fast_feedback & fast_write_from_readout:
            raise ValueError("a layer's fast writes take either --decouple_fast_feedback or "
                             "--fast_write_from_readout, not both")
        if fast_layers is not None and not (decouple_fast_feedback | fast_write_from_readout) <= fast_layers:
            raise ValueError("--decouple_fast_feedback / --fast_write_from_readout name a layer that "
                             "--fast_layers leaves without fast entries")
        if fast_write_from_readout and (fast_write_from_readout != {num_layers - 1} or layer_norm
                                        or output_tanh or residual_connection):
            raise ValueError("--fast_write_from_readout is the exact readout gradient only for the last "
                             "trunk layer read directly by i2o (no --layer_norm, --output_tanh or "
                             "--residual_connection)")
        self.fast_layers = None if fast_layers is None else sorted(fast_layers)
        self.fast_write_from_readout_scale = fast_write_from_readout_scale

        def trunk_fraction(index):
            return ephemeral_fraction if fast_layers is None or index in fast_layers else 0.0

        # Using EphemeralLinear instead of Linear
        self.linear_layers = torch.nn.ModuleList([
            EphemeralLinear(
                inner_size, inner_size, charset,
                weight_clamp=weight_clamp,
                updater=updater, plasticity=plasticity,
                batch_size=batch_size, forget_rate=forget_rate,
                ephemeral_fraction=trunk_fraction(0), slow_weight_decay=slow_weight_decay,
            fast_weight_clamp=fast_weight_clamp, slow_update_every=slow_update_every,
            fast_structure=fast_structure
            )
        ])
        for index in range(1, num_layers):
            self.linear_layers.append(EphemeralLinear(
                inner_size, inner_size, charset,
                weight_clamp=weight_clamp,
                updater=updater, plasticity=plasticity,
                batch_size=batch_size, forget_rate=forget_rate,
                ephemeral_fraction=trunk_fraction(index), slow_weight_decay=slow_weight_decay,
            fast_weight_clamp=fast_weight_clamp, slow_update_every=slow_update_every,
            fast_structure=fast_structure
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
            fast_weight_clamp=fast_weight_clamp, slow_update_every=slow_update_every,
            fast_structure=fast_structure
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
        for index in sorted(decouple_fast_feedback):
            layer = self.linear_layers[index]
            layer.fast_feedback_weights = decoupled_fast_feedback(*layer.feedback_weights.shape, index)
        for index in fast_write_from_readout:
            self.linear_layers[index].fast_write_from_readout = True
        for layer in self.trained_layers():
            layer.slow_nlms = slow_nlms
            layer.slow_lr_scale = readout_slow_lr_scale if layer is self.i2o else trunk_slow_lr_scale
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.updater = updater
        for layer in self.trained_layers():
            layer.retain_sequence_bias_grads = retain_sequence_bias_grads
        self.grad_clip_stats = GradNormClipStats()
        self.fused_layer_step = None  # set by enable_fused_update (--fused_update)
        self.fused_additive_step = None  # likewise, for additive layers (--fast_structure)
        set_dfa_fprime(self, dfa_fprime)
        self.set_fast_read_norm(fast_read_norm, fast_read_norm_lr)
        self.order2_layers = None
        if order2_layers is not None:
            self.set_order2(order2_layers, order2_map, order2_mode, order2_temperature, order2_dim,
                            order2_seed)

    def set_order2(self, layers, order='order2', mode='parallel', temperature=4.0, dim=16, seed=0):
        """--order2_layers: each listed trunk layer gets a second fast matrix G [B, out, F], read as
        G psi(x) next to its slow and linear fast read, with psi = order2_features of a fixed random
        projection to `dim` numbers (order2: 1 + d + d^2 features; order1, the control: 1 + d).
        DFA is unchanged: G is written with the same fast-write error as the layer's linear fast
        entries, times psi(x) instead of x, with the same alpha, learning rate, clamps and forget
        rate, and it is wiped with them. mode 'parallel' keeps the linear fast entries; 'replace'
        never writes them (their error is zero, so they stay at their wiped zero), which leaves G as
        the layer's only fast read and the slow entries exactly as they were. The layers must have
        fast entries (psi is scaled by the layer's fast fraction). One fast write per forward and the
        per-step slow update are assumed, so the DFA updater, --slow_update_every 1,
        --fast_backward_per_forward 1 and --fast_read_norm off are required."""
        layers = sorted(set(layers))
        if order not in ORDER2_MAPS:
            raise ValueError(f"--order2_map {order!r}: expected one of {ORDER2_MAPS}")
        if mode not in ORDER2_MODES:
            raise ValueError(f"--order2_mode {mode!r}: expected one of {ORDER2_MODES}")
        if not temperature > 0 or int(dim) < 1:
            raise ValueError("--order2_temperature must be > 0 and --order2_dim >= 1")
        if any(not 0 <= i < self.num_layers for i in layers) or not layers:
            raise ValueError(f"--order2_layers {layers}: trunk layers are 0..{self.num_layers - 1}")
        if self.updater != 'dfa':
            raise ValueError("--order2_layers supports only the DFA updater")
        if self.slow_update_every != 1 or self.fast_iterations != 1 or self.fast_subsample != 1:
            raise ValueError("--order2_layers needs --slow_update_every 1 and --fast_backward_per_forward 1")
        if self.fast_read_norm != 'off':
            raise ValueError("--order2_layers is not defined with --fast_read_norm")
        if any(layer.is_additive for layer in self.linear_layers):
            raise ValueError("--order2_layers is defined only for --fast_structure exclusive")
        for index in layers:
            layer = self.linear_layers[index]
            if not bool(layer.ephemeral_mask.any()):
                raise ValueError(f"--order2_layers {index}: the layer has no fast entries (--fast_layers)")
            device = layer.per_sample_weights.device
            layer.order2 = order
            layer.order2_replace = mode == 'replace'
            layer.order2_temperature = float(temperature)
            layer.order2_fraction = float(layer.ephemeral_mask.float().mean())
            layer.order2_proj = order2_projection(layer.in_features, int(dim), seed, index).to(device)
            layer.order2_fast = torch.zeros(layer.batch_size, layer.out_features,
                                            order2_feature_size(order, int(dim)), device=device)
        self.order2_layers = layers

    def order2_targets(self):
        """The trained layers with an order-two fast matrix."""
        return [layer for layer in self.linear_layers if layer.order2 is not None]

    @torch.no_grad()
    def order2_dfa_step(self, output_error, learning_rate, update_clamp, forget=True, projected=None):
        """The order-two fast matrices' write for this step (EphemeralLinear.order2_write), from the
        same fast-write errors the linear fast entries would take (before 'replace' zeroes those).
        Nothing else changes, so it can run before or after the other layers' step; it must run
        before i2o's update when --fast_write_from_readout reads i2o's weights. No-op when off."""
        if self.order2_layers is None:
            return
        if projected is None:
            projected, _ = self.dfa_step_errors(output_error)
        overrides = self._fast_error_overrides(output_error, projected)
        for layer, error, fast in zip(self.trained_layers(), projected, overrides):
            if layer.order2 is not None:
                layer.order2_write(error if fast is None else fast, learning_rate, update_clamp, forget)

    def set_fast_read_norm(self, mode, learning_rate=None):
        """--fast_read_norm on every layer with ephemeral entries (the trunk layers --fast_layers
        keeps, and i2h): 'off' (default, the one-bmm forward), 'running', 'absrun' or 'rms'
        (fast_read_scale), or 'ln' (fast_read_layer_norm). Forward-only: the DFA writes, the slow
        step and every update path are unchanged. 'running' and 'absrun' track one write per
        forward, so they need the DFA updater with one fast step per character."""
        if mode not in FAST_READ_NORMS:
            raise ValueError(f"--fast_read_norm {mode!r}: expected one of {FAST_READ_NORMS}")
        if mode != 'off' and self.updater != 'dfa':
            raise ValueError("--fast_read_norm supports only the DFA updater")
        if mode in FAST_READ_NORMS_WITH_HISTORY:
            if self.fast_iterations != 1 or self.fast_subsample != 1:
                raise ValueError(f"--fast_read_norm {mode} assumes one fast write per forward; it is not "
                                 "defined with --fast_backward_per_forward other than 1")
            if learning_rate is None:
                raise ValueError(f"--fast_read_norm {mode} needs the learning rate")
        self.fast_read_norm = mode
        for layer in [*self.linear_layers, self.i2h]:
            fast = bool(layer.ephemeral_mask.any())
            layer.fast_read_norm = mode if fast else 'off'
            layer.fast_read_coef = (float(learning_rate) * float(layer.ephemeral_mask.float().mean())
                                    if fast and mode in FAST_READ_NORMS_WITH_HISTORY else 0.0)
            layer.fast_key_mass = layer.fast_keys = layer.fast_key_weights = None

    def enable_fused_update(self, compile=True):
        """--fused_update: train.py's DFA step goes through fused_dfa_step. compile=False runs the
        same dfa_layer_step eagerly, which is bit-identical to the unfused step when
        --grad_norm_clip is off (tests/test_fused_update.py)."""
        if self.updater != 'dfa':
            raise ValueError(f"--fused_update supports only the DFA updater, not {self.updater!r}")
        self.fused_layer_step = torch.compile(dfa_layer_step, dynamic=False) if compile else dfa_layer_step
        if self.fast_structure != 'exclusive':
            self.fused_additive_step = (torch.compile(dfa_additive_layer_step, dynamic=False) if compile
                                        else dfa_additive_layer_step)

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
        fast_errors = self.fast_write_errors(output_error, projected)
        self.order2_dfa_step(output_error, learning_rate, update_clamp, projected=projected)
        for layer, error, fast_error in zip(layers, projected, fast_errors):
            layer._last_projected_error = error
            if layer.is_additive:
                layer.additive_step(error, learning_rate, update_clamp, self.fused_additive_step)
                continue
            self.fused_layer_step(
                layer.per_sample_weights.data, None if layer.bias is None else layer.bias.data,
                error, layer.in_traces.data, layer.fused_plasticity(), layer.ephemeral_mask,
                layer.forget_rate, learning_rate, update_clamp,
                layer.weight_clamp, layer.is_last_layer, layer.slow_weight_decay,
                layer.fast_weight_clamp, slow_nlms=layer.slow_nlms,
                 slow_lr_scale=layer.slow_lr_scale, fast_projected_error=fast_error)

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

    def fast_write_errors(self, output_error, projected):
        """Each trained layer's fast-write error where it differs from its DFA error projected
        (dfa_step_errors), else None: --decouple_fast_feedback projects output_error through the
        layer's B_fast (with f' under --dfa_fprime, as for the slow error); --fast_write_from_readout
        follows the readout gradient (readout_direction_error, from i2o's weights before this
        step's update). Every step path writes the fast entries with these, so training, norm
        logging, held-out evaluation and the loop tracer all see the same writes. Neither flag
        is defined with --grad_norm_clip (train.py rejects the combination).
        --order2_mode replace: a layer with an order-two fast matrix writes its linear fast entries
        with a zero error (they are never written); its order-two matrix takes the error this
        method would otherwise give (order2_dfa_step)."""
        errors = self._fast_error_overrides(output_error, projected)
        if self.order2_layers is None:
            return errors
        return [torch.zeros_like(error) if layer.order2_replace else fast
                for layer, error, fast in zip(self.trained_layers(), projected, errors)]

    def _fast_error_overrides(self, output_error, projected):
        """fast_write_errors before --order2_mode replace: the --decouple_fast_feedback or
        --fast_write_from_readout error of each trained layer, else None."""
        errors = []
        for layer, error in zip(self.trained_layers(), projected):
            if layer.fast_write_from_readout:
                errors.append(readout_direction_error(
                    output_error, self.i2o.per_sample_weights.data, layer.pre_activation(), error,
                    self.fast_write_from_readout_scale))
            elif layer.fast_feedback_weights is not None:
                errors.append(dfa_projected_error(output_error, layer.fast_feedback_weights, False,
                                                  dfa_layer_activation_derivative(layer)))
            else:
                errors.append(None)
        return errors

    def fast_write_projected(self, output_error, grad_norm_clip=0):
        """The error each trained layer's fast entries are written with this step: its DFA error,
        or the fast_write_errors override. loop_trace.py reads the writes through this."""
        projected, _ = self.dfa_step_errors(output_error, grad_norm_clip)
        return [error if fast is None else fast
                for error, fast in zip(projected, self.fast_write_errors(output_error, projected))]

    @torch.no_grad()
    def override_fast_gradients(self, output_error):
        """The unfused DFA step (train.py): after populate_dfa_gradients, replace the ephemeral
        entries' gradient of each layer with a fast_write_errors override. normalize_slow_gradient
        leaves the ephemeral entries as they are, so this equals dfa_layer_step's gradient."""
        layers = self.trained_layers()
        if not any(layer.fast_write_from_readout or layer.fast_feedback_weights is not None
                   or layer.order2_replace for layer in layers):
            return
        projected, _ = self.dfa_step_errors(output_error)
        for layer, fast_error in zip(layers, self.fast_write_errors(output_error, projected)):
            if fast_error is not None:
                grad = layer.per_sample_weights.grad
                grad.copy_(torch.where(layer.ephemeral_mask.unsqueeze(0),
                                       dfa_per_sample_gradient(fast_error, layer.in_traces.data), grad))

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
        if self.order2_layers is not None:
            self.order2_dfa_step(output_error, learning_rate, update_clamp, forget,
                                 projected=self.dfa_step_errors(output_error, grad_norm_clip)[0])
        for layer, error in zip(self.trained_layers(), self.fast_write_projected(output_error, grad_norm_clip)):
            if layer.is_last_layer:
                continue  # i2o has no ephemeral entries: the frozen step would leave it unchanged
            self._fast_entry_step(layer, error, learning_rate, update_clamp, forget)

    def _fast_entry_step(self, layer, error, learning_rate, update_clamp, forget=True):
        """dfa_layer_step with freeze_slow: this step on the ephemeral entries only (bias frozen).
        forget=False: no forgetting (forget rate 0; decay 0, which only acts on slow entries).
        Additive layers: this step on F only (S and the bias frozen)."""
        if layer.is_additive:
            layer.additive_step(error, learning_rate, update_clamp, self.fused_additive_step,
                                fast_only=True, fast_forget=forget)
            return
        step = self.fused_layer_step or dfa_layer_step
        step(layer.per_sample_weights.data, None, error, layer.in_traces.data,
             layer.fused_plasticity(), layer.ephemeral_mask, layer.forget_rate if forget else 0.0,
             learning_rate, update_clamp, layer.weight_clamp, layer.is_last_layer,
             layer.slow_weight_decay if forget else 0.0, layer.fast_weight_clamp, True,
             slow_nlms=layer.slow_nlms,
                 slow_lr_scale=layer.slow_lr_scale)

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
        fast_errors = self.fast_write_errors(output_error, projected)
        saved = []
        for layer, error, fast_error in zip(self.trained_layers(), projected, fast_errors):
            layer._last_projected_error = error
            inputs = layer.in_traces.data
            if log_norms:
                self._log_step_norms(layer, error, fast_error, update_clamp)
            saved.append((error, inputs))
            if not layer.is_last_layer:
                self._fast_entry_step(layer, error if fast_error is None else fast_error,
                                      learning_rate, update_clamp)
        return saved

    @staticmethod
    def _log_step_norms(layer, error, fast_error, update_clamp):
        """The per-step update norms the unfused step logs (_log_update_norms), for the split,
        skip-fast and windowed steps, which never materialize the update otherwise."""
        inputs = layer.in_traces.data
        gradient = dfa_step_gradient(error, inputs, layer.ephemeral_mask, layer.slow_nlms,
                                     layer.slow_lr_scale, fast_error)
        layer._log_update_norms(ephemeral_update(
            gradient, layer.plasticity, layer.ephemeral_mask, update_clamp, layer.is_last_layer))

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
                  False, True, False, slow_nlms=layer.slow_nlms,
                 slow_lr_scale=layer.slow_lr_scale)

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
        fast_errors = self.fast_write_errors(output_error, projected) if log_norms else [None] * len(projected)
        for layer, error, fast_error in zip(self.trained_layers(), projected, fast_errors):
            layer._last_projected_error = error
            if log_norms:
                self._log_step_norms(layer, error, fast_error, update_clamp)
            step(layer.per_sample_weights.data, None if layer.bias is None else layer.bias.data,
                 error, layer.in_traces.data, layer.fused_plasticity(), layer.ephemeral_mask,
                  layer.forget_rate, learning_rate, update_clamp, layer.weight_clamp,
                  layer.is_last_layer, layer.slow_weight_decay, layer.fast_weight_clamp,
                  False, True, slow_nlms=layer.slow_nlms,
                 slow_lr_scale=layer.slow_lr_scale)

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
        fast_errors = self.fast_write_errors(output_error, projected)
        for layer, error, fast_error in zip(self.trained_layers(), projected, fast_errors):
            layer._last_projected_error = error
            if log_norms:
                self._log_step_norms(layer, error, fast_error, update_clamp)
            slow_error = slow_bias_error(error, layer.in_traces.data, layer.slow_nlms,
                                         layer.slow_lr_scale)
            layer.accumulate_slow_gradient(slow_error, slot)
            if not layer.is_last_layer:
                fast_write = error if fast_error is None else fast_error
                self._fast_entry_step(layer, torch.zeros_like(error) if skip_fast else fast_write,
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

    @torch.no_grad()
    def average_bias_grads(self, batch_size):
        """--backprop_step_parity: backward() ran on the batch-summed loss, so each bias gradient
        is the sum of the sequences' shares; divide it by the batch size so the shared bias steps
        with their batch mean, as DFA's bias does (dfa_bias_update)."""
        for layer in self.trained_layers():
            if layer.bias is not None and layer.bias.grad is not None:
                layer.bias.grad.div_(batch_size)


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
