"""Diagnostic: cap the singular values of the slow trunk weights inside a fixed set of directions.

Off by default (--sv_cap_file ''). Used by the causal test "are the growing slow-weight directions the
cause of the late collapse?". It changes training, so it is NOT an observation-only tool.

Spec file (torch.save of a dict): {'layers': {'linear_layers.0': {'U': [out,k], 'V': [in,k], 'caps': [k]}, ...}}
with U, V orthonormal columns. For each listed layer, every `every` iterations:
    S = mean over the batch copies of per_sample_weights, fast (ephemeral) positions zeroed   (the slow matrix)
    M = U^T S V                      (k x k block of the slow matrix in the capped subspace)
    M = P diag(s) Q^T ;  s' = min(s, caps)  (caps sorted descending, paired with s sorted descending: HARD cap)
    D = U (P diag(s') Q^T - M) V^T ; per_sample_weights[:, slow positions] += D[slow positions]
The correction is masked to the slow positions (fast entries are never touched) and applied to every batch
copy; the masking leaves part of D unremoved, so `passes` repeated passes (default 3) are applied.
"""
import json

import torch


def load_cap_spec(path, device):
    spec = torch.load(path, map_location='cpu', weights_only=False)
    layers = {}
    for name, d in spec['layers'].items():
        layers[name] = {key: d[key].to(device=device, dtype=torch.float32) for key in ('U', 'V', 'caps')}
    return layers, spec.get('meta', {})


def cap_block(S, U, V, caps):
    """(block singular values before the cap, correction D [out, in] to ADD to S, unmasked). Pure."""
    M = U.T @ S @ V
    P, s, Qh = torch.linalg.svd(M)
    s_new = torch.minimum(s, caps)
    D = U @ (P @ torch.diag(s_new - s) @ Qh) @ V.T
    return s, D  # S + D has block singular values s_new


class SlowSvCap:
    def __init__(self, rnn, spec_path, every=5, start_iter=0, passes=3, log_path=None, log_every=500):
        device = next(rnn.parameters()).device
        self.layers, self.meta = load_cap_spec(spec_path, device)
        modules = dict(rnn.named_modules())
        self.targets = {name: modules[name] for name in self.layers}
        self.every, self.start_iter, self.passes = every, start_iter, passes
        self.log_path, self.log_every = log_path, log_every
        self.last_before = {}

    @staticmethod
    def slow_mean(layer):
        return layer.per_sample_weights.data.mean(dim=0) * (~layer.ephemeral_mask)

    @torch.no_grad()
    def apply(self, iteration):
        if iteration < self.start_iter or iteration % self.every != 0:
            return
        for name, layer in self.targets.items():
            spec = self.layers[name]
            slow = ~layer.ephemeral_mask
            for p in range(self.passes):
                s, D = cap_block(self.slow_mean(layer), spec['U'], spec['V'], spec['caps'])
                if p == 0:
                    self.last_before[name] = s
                if not bool((s > spec['caps']).any()):
                    break
                layer.per_sample_weights.data.add_((D * slow).unsqueeze(0))

    @torch.no_grad()
    def log(self, iteration):
        """Block singular values after the cap (and before it), the full slow matrix's top singular values."""
        if self.log_path is None or iteration % self.log_every != 0:
            return
        rec = {'iter': iteration, 'layers': {}}
        for name, layer in self.targets.items():
            spec = self.layers[name]
            S = self.slow_mean(layer)
            block = torch.linalg.svdvals(spec['U'].T @ S @ spec['V'])
            rec['layers'][name] = {
                'block_sv': block.tolist(), 'caps': spec['caps'].tolist(),
                'block_sv_before_cap': self.last_before.get(name, block).tolist(),
                'full_top20_sv': torch.linalg.svdvals(S)[:20].tolist(), 'fro': float(torch.linalg.matrix_norm(S))}
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(rec) + '\n')
