"""Diagnostic: cosine between DFA's projected error and the true backpropagated signal.

For every hidden layer l of an EphemeralRNN (not i2h, whose current-step true gradient is zero, and not
i2o, whose DFA error is the true one), per batch row and step: cos(p_l, g_l), p_l = the projected error
the DFA step uses (EphemeralRNN.dfa_step_errors), g_l = dL/da_l, the autograd gradient of that row's
cross-entropy with respect to the layer's pre-activation output, through the actual forward pass
(slow and fast entries as they are, later layers, GELU, LayerNorm and i2o included).

measure_alignment replays one batch of sequences with the model's own DFA steps (so the fast weights
are written as in training) and restores the complete state dict afterwards: the call changes nothing.
"""
import torch
import torch.nn.functional as F


def measure_alignment(rnn, step_inputs, step_targets, learning_rate, update_clamp, answer_steps=None):
    """step_inputs: list of [B, in] model inputs per step, step_targets: list of [B, V] one-hot targets.
    answer_steps: optional LongTensor [B], the step index of each row's recall answer.
    Returns {layer index: {'cos', 'cos_answer', 'p_norm', 'g_norm', 'n'}} for the hidden layers; cos is
    the mean over rows and steps with a non-zero error, cos_answer the mean over answer steps only."""
    if rnn.updater != 'dfa':
        raise ValueError("measure_alignment needs the DFA updater")
    snapshot = {key: value.clone() for key, value in rnn.state_dict().items()}
    layers = list(rnn.linear_layers)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    totals = [dict(cos=0.0, n=0, cos_a=0.0, n_a=0, p=0.0, g=0.0) for _ in layers]
    try:
        rnn.start_sequence_wipe()
        hidden = rnn.initHidden(step_inputs[0].shape[0])
        for i, (x, target) in enumerate(zip(step_inputs, step_targets)):
            with torch.enable_grad():
                outs = []
                hooks = [layer.register_forward_hook(lambda m, inp, out, outs=outs: outs.append(out.requires_grad_(True)))
                         for layer in layers]
                try:
                    output, hidden = rnn(x, hidden.detach())
                finally:
                    for hook in hooks:
                        hook.remove()
                loss = criterion(output, target)
                grads = torch.autograd.grad(loss.sum(), [output] + outs)
            error, true_grads = grads[0].detach(), [g.detach() for g in grads[1:]]
            projected, _ = rnn.dfa_step_errors(error, 0)
            valid = error.norm(dim=1) > 0
            for k in range(len(layers)):
                p, g = projected[k], true_grads[k]
                cos = F.cosine_similarity(p, g, dim=1, eps=1e-30)
                ok = valid & (g.norm(dim=1) > 0)
                t = totals[k]
                t['cos'] += cos[ok].sum().item(); t['n'] += int(ok.sum())
                t['p'] += p.norm(dim=1)[ok].sum().item(); t['g'] += g.norm(dim=1)[ok].sum().item()
                if answer_steps is not None:
                    ok_a = ok & (answer_steps.to(ok.device) == i)
                    t['cos_a'] += cos[ok_a].sum().item(); t['n_a'] += int(ok_a.sum())
            # the model's own DFA step (fused if enabled), then continue with the written fast weights
            if rnn.fused_layer_step is not None:
                rnn.fused_dfa_step(error, learning_rate, update_clamp, 0)
            else:
                rnn.clear_dfa_gradients()
                for layer in rnn.trained_layers():
                    layer.populate_dfa_gradients(error)
                for layer in rnn.trained_layers():
                    layer.apply_update(learning_rate, update_clamp, {})
                rnn.apply_forget_step()
                rnn.clear_dfa_gradients()
    finally:
        rnn.load_state_dict(snapshot)
    return {k: dict(cos=t['cos'] / max(t['n'], 1), cos_answer=(t['cos_a'] / t['n_a'] if t['n_a'] else None),
                    p_norm=t['p'] / max(t['n'], 1), g_norm=t['g'] / max(t['n'], 1), n=t['n'])
            for k, t in enumerate(totals)}
