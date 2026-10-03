import types

import torch

from norm_cap import NormCap


def layer(out=12, inn=10, batch=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    mask = torch.rand(out, inn, generator=g) < 0.2
    w = torch.nn.Parameter(torch.randn(batch, out, inn, generator=g), requires_grad=False)
    w.data[:, mask] = 5.0
    return types.SimpleNamespace(ephemeral_mask=mask, per_sample_weights=w, bias=torch.nn.Parameter(torch.ones(out) * 3, requires_grad=False))


def rnn_of(l):
    return types.SimpleNamespace(parameters=lambda: iter([l.per_sample_weights]), named_modules=lambda: [('l', l)])


def run(tmp_path, spec, **kw):
    l = layer(); path = str(tmp_path / 's.pt'); torch.save(spec, path)
    before = l.per_sample_weights.data.clone()
    c = NormCap(rnn_of(l), path, every=1, svd_every=1, **kw)
    return l, before, c


def test_spectrum_cap_caps_every_singular_value_and_spares_fast(tmp_path):
    l0 = layer(); s0 = torch.linalg.svdvals(l0.per_sample_weights.data.mean(0) * (~l0.ephemeral_mask))
    spec = {'layers': {'l': {'mode': 'spectrum', 'caps': 0.5 * s0}}}
    l, before, c = run(tmp_path, spec, passes=60)
    c.apply(1)
    assert torch.equal(l.per_sample_weights.data[:, l.ephemeral_mask], before[:, l.ephemeral_mask])
    s = torch.linalg.svdvals(c.slow_mean(l))
    assert bool((s <= 0.5 * s0 * 1.02).all())


def test_frobenius_and_bias_caps(tmp_path):
    l0 = layer(); f0 = float(torch.linalg.matrix_norm(l0.per_sample_weights.data.mean(0) * (~l0.ephemeral_mask)))
    spec = {'layers': {'l': {'mode': 'frobenius', 'caps': 0.5 * f0}}, 'biases': {'l': 1.0}}
    l, before, c = run(tmp_path, spec)
    c.apply(1)
    assert torch.equal(l.per_sample_weights.data[:, l.ephemeral_mask], before[:, l.ephemeral_mask])
    assert abs(float(torch.linalg.matrix_norm(c.slow_mean(l))) - 0.5 * f0) < 1e-3
    assert abs(float(l.bias.norm()) - 1.0) < 1e-5


def test_start_and_noop_when_under_cap(tmp_path):
    spec = {'layers': {'l': {'mode': 'frobenius', 'caps': 1e9}}, 'biases': {'l': 1e9}}
    l, before, c = run(tmp_path, spec, start_iter=5)
    c.apply(1)
    assert torch.equal(l.per_sample_weights.data, before)
    c.apply(5)
    assert torch.equal(l.per_sample_weights.data, before)
