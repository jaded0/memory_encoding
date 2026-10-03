"""Diagnostic (off by default): general slow-weight norm caps, the full-spectrum follow-up of sv_cap.py.

--norm_cap_file is a torch file {'layers': {name: {'mode': 'spectrum'|'frobenius', 'caps': [r] or float}},
'biases': {name: max_norm}}; names are module names (linear_layers.0, i2o, ...).
  spectrum   every singular value of the slow matrix S (batch mean of per_sample_weights, fast positions
             zeroed) is hard-capped at caps[i] (descending order, paired with the descending singular values);
             D = U diag(min(s, caps) - s) V^T is added on the slow positions of every batch copy (fast entries never
             touched). The mask removes part of D, so `passes` passes (default 2). A full SVD is expensive, so it runs
             every --norm_cap_svd_every iterations; matrices with at most 32 rows run every --norm_cap_every.
  frobenius  slow entries of every batch copy scaled by min(1, caps / |S|_F), every --norm_cap_every.
  biases     each bias vector scaled to norm <= max_norm, every --norm_cap_every.
It changes training (not observation-only). Applied after the train step of iteration it, from --norm_cap_start."""
import json

import torch


class NormCap:
    def __init__(self, rnn, path, every=5, svd_every=100, start_iter=0, passes=2, log_path=None, log_every=500):
        spec = torch.load(path, map_location='cpu', weights_only=False)
        device = next(rnn.parameters()).device
        mods = dict(rnn.named_modules())
        self.layers = {n: (mods[n], d['mode'], torch.as_tensor(d['caps'], dtype=torch.float32, device=device))
                       for n, d in spec['layers'].items()}
        self.biases = {n: (mods[n], float(v)) for n, v in spec.get('biases', {}).items()}
        self.every, self.svd_every, self.start_iter, self.passes = every, svd_every, start_iter, passes
        self.log_path, self.log_every = log_path, log_every

    @staticmethod
    def slow_mean(layer):
        return layer.per_sample_weights.data.mean(dim=0) * (~layer.ephemeral_mask)

    @torch.no_grad()
    def apply(self, it):
        if it < self.start_iter:
            return
        for n, (layer, mode, caps) in self.layers.items():
            slow = ~layer.ephemeral_mask
            if mode == 'frobenius':
                if it % self.every == 0:
                    f = torch.linalg.matrix_norm(self.slow_mean(layer))
                    scale = torch.clamp(caps / f, max=1.0)
                    w = layer.per_sample_weights.data
                    w.copy_(torch.where(slow.unsqueeze(0), w * scale, w))
            else:
                period = self.every if min(layer.per_sample_weights.shape[1:]) <= 32 else self.svd_every
                if it % period != 0:
                    continue
                for _ in range(self.passes):
                    S = self.slow_mean(layer)
                    U, s, Vh = torch.linalg.svd(S, full_matrices=False)
                    s_new = torch.minimum(s, caps[: s.numel()])
                    if not bool((s > caps[: s.numel()]).any()):
                        break
                    D = (U * (s_new - s)) @ Vh
                    layer.per_sample_weights.data.add_((D * slow).unsqueeze(0))
        if it % self.every == 0:
            for n, (layer, cap) in self.biases.items():
                b = layer.bias.data
                nb = torch.linalg.vector_norm(b)
                b.mul_(torch.clamp(cap / nb, max=1.0))

    @torch.no_grad()
    def log(self, it):
        if self.log_path is None or it % self.log_every != 0:
            return
        rec = {'iter': it, 'layers': {}, 'biases': {}}
        for n, (layer, mode, caps) in self.layers.items():
            S = self.slow_mean(layer)
            s = torch.linalg.svdvals(S)
            if mode == 'frobenius':
                rec['layers'][n] = {'fro': float(s.norm()), 'top5': s[:5].tolist(), 'cap_fro': float(caps)}
            else:
                rec['layers'][n] = {'fro': float(s.norm()), 'top5': s[:5].tolist(),
                                    'max_ratio_to_cap': float((s / caps[: s.numel()]).max()),
                                    'n_above_cap': int((s > caps[: s.numel()] * 1.0001).sum())}
        for n, (layer, cap) in self.biases.items():
            rec['biases'][n] = {'norm': float(torch.linalg.vector_norm(layer.bias.data)), 'cap': cap}
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(rec) + '\n')
