import types

import torch

from sv_cap import SlowSvCap, cap_block


def make(tmp_path, out=12, inn=10, k=3, batch=4):
    g = torch.Generator().manual_seed(0)
    mask = torch.rand(out, inn, generator=g) < 0.2
    layer = types.SimpleNamespace(ephemeral_mask=mask,
                                  per_sample_weights=torch.nn.Parameter(torch.randn(batch, out, inn, generator=g), requires_grad=False))
    layer.per_sample_weights.data[:, mask] = 5.0  # fast entries: must never change
    U = torch.linalg.qr(torch.randn(out, k, generator=g))[0]
    V = torch.linalg.qr(torch.randn(inn, k, generator=g))[0]
    caps = torch.tensor([0.5, 0.4, 0.3])
    path = str(tmp_path / 'spec.pt')
    torch.save({'layers': {'l': {'U': U, 'V': V, 'caps': caps}}}, path)
    rnn = types.SimpleNamespace(parameters=lambda: iter([layer.per_sample_weights]), named_modules=lambda: [('l', layer)])
    return rnn, layer, U, V, caps, path


def test_cap_block_caps_block_singular_values():
    S = torch.randn(12, 10)
    U = torch.linalg.qr(torch.randn(12, 3))[0]
    V = torch.linalg.qr(torch.randn(10, 3))[0]
    caps = torch.tensor([0.5, 0.4, 0.3])
    s, D = cap_block(S, U, V, caps)
    s_after = torch.linalg.svdvals(U.T @ (S + D) @ V)
    assert torch.allclose(s_after, torch.minimum(s, caps), atol=1e-5)


def test_apply_caps_slow_and_leaves_fast_untouched(tmp_path):
    rnn, layer, U, V, caps, path = make(tmp_path)
    before = layer.per_sample_weights.data.clone()
    cap = SlowSvCap(rnn, path, every=1, passes=40)
    cap.apply(1)
    after = layer.per_sample_weights.data
    assert torch.equal(after[:, layer.ephemeral_mask], before[:, layer.ephemeral_mask])
    s = torch.linalg.svdvals(U.T @ cap.slow_mean(layer) @ V)
    assert bool((s <= caps * 1.01).all())


def test_every_and_start(tmp_path):
    rnn, layer, *_rest, path = make(tmp_path)
    before = layer.per_sample_weights.data.clone()
    cap = SlowSvCap(rnn, path, every=5, start_iter=10)
    cap.apply(7); cap.apply(5)
    assert torch.equal(layer.per_sample_weights.data, before)
    cap.apply(10)
    assert not torch.equal(layer.per_sample_weights.data, before)


def test_off_by_default():
    import train
    import sys
    argv = sys.argv
    sys.argv = ['train.py']
    try:
        parser_src = open(train.__file__).read()
    finally:
        sys.argv = argv
    assert "'--sv_cap_file', type=str, default=''" in parser_src
