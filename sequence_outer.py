"""Opt-in K14 sequence SGD: frozen slow scaffold, functional DFA fast writes.

Both arms retain the recurrent graph; only the write increment is stop-gradient
in the detached arm. This is not the legacy online backprop comparator.
"""
import torch
import torch.nn.functional as F

from ephemeral_model import EphemeralRNN, trunk_layer_norm, forget_keep
from utils import model_input


def check_sequence_outer(config):
    """Refuse unsupported mechanics instead of silently ignoring experiment flags."""
    mode = config.get('fast_write_grad', 'detached')
    if mode not in ('through', 'detached'):
        raise ValueError('--fast_write_grad must be through or detached')
    if not config.get('sequence_outer', False):
        if mode != 'detached':
            raise ValueError('--fast_write_grad through requires --sequence_outer')
        return
    if (config.get('model_type') != 'ephemeral' or config.get('updater') != 'backprop'
            or not config.get('backprop_step_parity', False)):
        raise ValueError('--sequence_outer requires --model_type ephemeral --updater backprop '
                         '--backprop_step_parity true')
    if config.get('max_seq_len') != 10:
        raise ValueError('--sequence_outer requires --max_seq_len 10 (ten characters)')
    unsupported = {
        'fused_update': False, 'grad_norm_clip': 0, 'slow_weight_decay': 0,
        'fast_weight_clamp': 0, 'slow_update_every': 1, 'fast_backward_per_forward': 1,
        'wipe_every': 1, 'train_wipe_chars': 0, 'fast_structure': 'exclusive',
        'dfa_fprime': False, 'readout_nlms': False, 'slow_nlms': False,
        'readout_slow_lr_scale': 1.0, 'trunk_slow_lr_scale': 1.0,
        'decouple_fast_feedback': '', 'fast_write_from_readout': '',
        'fast_read_norm': 'off', 'order2_layers': '', 'write_gain_power': 0,
        'label_smoothing': 0, 'trace_loop_every': 0,
        'sv_cap_file': '', 'norm_cap_file': '', 'optimizer': 'sgd',
    }
    for key, default in unsupported.items():
        if config.get(key, default) != default:
            raise ValueError(f'--sequence_outer does not support --{key}')


def _check_rollout(rnn, onehot, config):
    check_sequence_outer(config)
    if not config.get('sequence_outer', False):
        raise ValueError('sequence rollout requires --sequence_outer')
    if not isinstance(rnn, EphemeralRNN) or rnn.updater != 'backprop':
        raise ValueError('--sequence_outer requires a backprop EphemeralRNN')
    if onehot.shape[0] != rnn.batch_size:
        raise ValueError('--sequence_outer batch size must match model copies')
    if not 2 <= onehot.shape[1] <= 10:
        raise ValueError('--sequence_outer expects 2..10 characters')
    criterion = config['criterion']
    if (not isinstance(criterion, torch.nn.CrossEntropyLoss) or criterion.reduction != 'none'
            or criterion.weight is not None or criterion.label_smoothing != 0):
        raise ValueError('--sequence_outer requires plain CrossEntropyLoss(reduction=none)')


def sequence_rollout(rnn, onehot, config):
    """Live losses/outputs and final fast tensors, without parameter mutation.

    Caller wipes/consolidates first. Slow per-copy leaves are fixed throughout.
    The explicit error softmax(logit)-target stays live in the through arm; its
    Jacobian is ordinary first-order autograd, not a differentiated backward().
    """
    _check_rollout(rnn, onehot, config)
    layers = rnn.trained_layers()
    slow = [torch.where(l.ephemeral_mask.unsqueeze(0), 0., l.per_sample_weights)
            for l in layers]
    fast = [torch.zeros_like(w) for w in slow]
    hidden = onehot.new_zeros(onehot.shape[0], rnn.hidden_size)
    losses, outputs = [], []
    for t in range(onehot.shape[1] - 1):
        x = model_input(onehot, t, config['input_mode'], config['pe_matrix'])
        combined = torch.cat((x, hidden), dim=1)
        residual = combined
        keys = []

        def read(index, key):
            keys.append(key)
            value = torch.bmm(slow[index] + fast[index], key.unsqueeze(2)).squeeze(2)
            if layers[index].bias is not None:
                value = value + layers[index].bias
            return value

        for index in range(len(rnn.linear_layers)):
            combined = F.gelu(read(index, combined))
            if rnn.layer_norm:
                combined = trunk_layer_norm(combined)
        if rnn.residual_connection:
            combined = combined + residual
        hidden_t = torch.tanh(read(len(layers) - 2, combined))
        output = read(len(layers) - 1, torch.tanh(combined) if rnn.output_tanh else combined)
        hidden = hidden_t if rnn.enable_recurrence else torch.zeros_like(hidden)
        target = onehot[:, t + 1]
        losses.append(config['criterion'](output, target))
        outputs.append(output)
        error = output.softmax(dim=1) * target.sum(dim=1, keepdim=True) - target
        updated = []
        for index, layer in enumerate(layers):
            projected = error if layer.is_last_layer else error @ layer.feedback_weights
            # alpha once; clamp before lr and forget after writing, as eager DFA.
            write = -(projected.unsqueeze(2) * keys[index].unsqueeze(1)) * layer.fast_alpha()
            clamp = config.get('ephemeral_update_clamp', 0)
            if clamp > 0:
                write = write.clamp(-clamp, clamp)
            write = write * layer.ephemeral_mask.unsqueeze(0)
            if config.get('fast_write_grad', 'detached') == 'detached':
                write = write.detach()
            value = fast[index] + config['learning_rate'] * write
            if layer.weight_clamp > 0:
                value = value.clamp(-layer.weight_clamp, layer.weight_clamp)
            updated.append(value * forget_keep(layer.forget_rate, layer.ephemeral_mask))
        fast = updated
    return torch.stack(losses), torch.stack(outputs), fast


def train_sequence_outer(onehot, rnn, config, state, log_outputs=False):
    _check_rollout(rnn, onehot, config)
    rnn.start_sequence_wipe()
    rnn.zero_grad(set_to_none=True)
    losses, outputs, fast = sequence_rollout(rnn, onehot, config)
    # C2 batch sum => each copy gets its own gradient, shared biases batch mean.
    # Time mean keeps the outer learning-rate scale independent of crop length.
    losses.sum(dim=1).mean().backward()
    rnn.average_bias_grads(onehot.shape[0])
    with torch.no_grad():
        for layer, final_fast in zip(rnn.trained_layers(), fast):
            weight = layer.per_sample_weights
            slow = weight - config['learning_rate'] * (weight.grad if weight.grad is not None else 0)
            if layer.weight_clamp > 0:
                slow = slow.clamp(-layer.weight_clamp, layer.weight_clamp)
            weight.copy_(torch.where(layer.ephemeral_mask.unsqueeze(0), final_fast, slow))
            if layer.bias is not None and layer.bias.grad is not None:
                layer.bias.add_(layer.bias.grad, alpha=-config['learning_rate'])
    rnn.zero_grad(set_to_none=True)
    state['training_instance'] += onehot.shape[1] - 1
    detached = outputs.detach()
    return (detached[-1], losses.detach().mean().item(), detached.argmax(dim=2), losses.detach(),
            list(detached[:, 0]) if log_outputs else [],
            list(onehot[0, 1:]) if log_outputs else [])
