"""HY1 invariants, kept small enough for CPU/sm61 eager smoke tests."""
import importlib
from dataclasses import asdict

import pytest
import torch
import torch.nn.functional as F

from hy1_model import HY1Config, HY1Model, evaluate_document, load_model, rms
from hy1_train import save_checkpoint, train_chunk


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def small(arm="A1", **kwargs):
    return HY1Model(HY1Config(vocab_size=11, arm=arm, d_model=16,
                            hidden_dim=32, n_heads=4, n_layers=2,
                            window=4, chunk_size=4, **kwargs))


def test_slow_initialization_and_no_added_trainable_parameters():
    base = small("A0")
    a1 = small("A1")
    assert sum(p.numel() for p in base.parameters()) == sum(p.numel() for p in a1.parameters())
    for arm in ("A1", "A2", "A3"):
        other = small(arm)
        for (name, p), (other_name, q) in zip(base.named_parameters(), other.named_parameters()):
            assert name == other_name
            if arm == "A1" and name == "blocks.0.fc2.weight":
                assert torch.equal(p*a1.slow_mask, q)
            else:
                assert torch.equal(p, q)
    assert "B_fast" not in dict(a1.named_parameters())
    assert a1.B_fast.data_ptr() != a1.output.weight.data_ptr()
    assert a1.fast_indices.numel() == round(16*32*.1)


def test_constructor_rng_and_existing_defaults_unchanged():
    torch.manual_seed(72)
    before = torch.random.get_rng_state().clone()
    small()
    assert torch.equal(before, torch.random.get_rng_state())
    # New module never imports or monkeypatches the existing model.
    import sys
    assert "ephemeral_model" not in importlib.import_module("hy1_model").__dict__


def test_within_chunk_causality_and_no_early_label_write():
    m = small()
    tokens = torch.tensor([1, 2, 3, 4])
    s = m.new_state(torch.randn(m.fast_indices.numel()))
    a, s1, pending = m.forward_chunk(tokens, s)
    b, _, _ = m.forward_chunk(torch.tensor([1, 2, 8, 9]), s)
    torch.testing.assert_close(a[:2], b[:2], rtol=0, atol=0)
    assert torch.equal(s.fast, s1.fast)
    u = m.commit(s1, pending, torch.tensor([2, 3, 4, 5]))
    v = m.commit(s1, pending, torch.tensor([7, 8, 9, 0]))
    assert not torch.equal(u.fast, v.fast)
    # Committing returns new state, never mutates old fast values or logits.
    assert torch.equal(s.fast, s1.fast)
    assert not u.fast.requires_grad
    assert pending.logits.grad_fn is None and pending.hidden.grad_fn is None
    assert all(not x.requires_grad for pair in u.cache for x in pair)
    later_u, _, _ = m.forward_chunk(tokens, u)
    later_v, _, _ = m.forward_chunk(tokens, v)
    assert not torch.equal(later_u, later_v)


@pytest.mark.parametrize("arm", ["A0", "A2", "A3"])
def test_real_attention_history_and_chunk_partition_equivalence(arm):
    # Same full prefix forwarded as one chunk vs several chunks: logits agree.
    # There is no fast state in this comparison; cached K/V values are exact.
    config = HY1Config(vocab_size=11, arm=arm, d_model=16, hidden_dim=32,
                       n_heads=4, n_layers=2, window=4, chunk_size=12)
    m = HY1Model(config)
    tokens = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
    with torch.no_grad():
        full, _, _ = m.forward_chunk(tokens)
        state = m.new_state()
        parts = []
        for part in tokens.split(3):
            out, state, _ = m.forward_chunk(part, state)
            parts.append(out)
        torch.testing.assert_close(full, torch.cat(parts), atol=1e-6, rtol=1e-5)
        length = 9 if arm == "A3" else m.config.attention_window-1
        assert all(k.shape[1] == length for k, v in state.cache)
        changed = tokens.clone()
        changed[2] = 10
        alt, _, _ = m.forward_chunk(changed)
        assert not torch.equal(full[3], alt[3])  # previous-chunk context matters


def test_read_rms_bound_and_detached_normalized_write():
    m = small()
    block = m.blocks[0]
    x = torch.randn(4, 16)
    dense = torch.randn(16, 32)*1e4*(1-m.slow_mask)
    got = []
    handle = block.fc1.register_forward_hook(lambda module, args, out: got.append(F.gelu(out)))
    a, _, _ = block(x, None, m.config, None, m.slow_mask)
    b, _, _ = block(x, None, m.config, dense, m.slow_mask)
    handle.remove()
    slow = F.linear(got[0], block.fc2.weight*m.slow_mask, block.fc2.bias)
    assert torch.all(rms(b-a) <= .1*rms(slow)+1e-7)
    _, state, pending = m.forward_chunk(torch.tensor([1, 2, 3, 4]))
    state = m.commit(state, pending, torch.tensor([2, 3, 4, 5]))
    torch.testing.assert_close(state.fast.square().mean().sqrt(), torch.tensor(1.))
    for _ in range(30):
        state = m.commit(state, pending, torch.tensor([2, 3, 4, 5]))
    assert state.fast.square().mean().sqrt() <= 10.00001


def test_exclusive_zero_grad_zero_parameters_unfused_adam():
    m = small()
    opt = torch.optim.Adam(m.parameters(), lr=3e-4, foreach=False, fused=False)
    old_feedback = m.B_fast.clone()
    state = m.new_state()
    for _ in range(3):
        loss, state, norm = train_chunk(m, opt, torch.tensor([1, 2, 3, 4]),
                                       torch.tensor([2, 3, 4, 5]), state)
        w = m.blocks[0].fc2.weight
        assert torch.count_nonzero(w[m.slow_mask == 0]) == 0
        assert torch.count_nonzero(w.grad[m.slow_mask == 0]) == 0
        assert torch.isfinite(torch.tensor(loss))
    assert torch.equal(old_feedback, m.B_fast)
    assert state.fast.abs().sum() > 0
    assert not opt.defaults["fused"] and not opt.defaults["foreach"]


def test_eval_reset_carry_wipe_and_frozen_slow():
    m = small()
    document = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    before = {k: v.clone() for k, v in m.state_dict().items()}
    losses, state = evaluate_document(m, document)
    again, _ = evaluate_document(m, document)
    assert losses.shape == (len(document)-1,)
    assert torch.equal(losses, again)
    assert m.training  # evaluate restores caller mode
    assert all(torch.equal(v, m.state_dict()[k]) for k, v in before.items())
    assert all(p.grad is None for p in m.parameters())
    carry, carried = evaluate_document(m, document, state=state)
    assert not torch.equal(carry[:4], losses[:4])
    assert carried.position == len(document)-1  # KV reset at doc boundary
    wiped, _ = evaluate_document(m, document, state=state, wipe_every_chunk=True)
    torch.testing.assert_close(wiped[:4], losses[:4], atol=0, rtol=0)
    assert not torch.equal(wiped[4:], losses[4:])
    assert torch.count_nonzero(m.new_state().fast) == 0


def test_checkpoint_and_state_accounting(tmp_path):
    m = small()
    opt = torch.optim.Adam(m.parameters(), foreach=False, fused=False)
    path = tmp_path/"test.pt"
    save_checkpoint(path, m, opt, "abcdef", 100, 2)
    loaded, checkpoint = load_model(path)
    for k, v in m.state_dict().items():
        assert torch.equal(v, loaded.state_dict()[k])
    assert checkpoint["tokens"] == 100
    standard = HY1Model(HY1Config(vocab_size=73, arm="A1"))
    assert standard.state_bytes() == {"fast_bytes": 104856, "A2_extra_KV_bytes": 524288}


def test_nonfinite_failfast():
    m = small()
    with torch.no_grad():
        m.output.weight.fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        m.forward_chunk(torch.tensor([1, 2]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_unfused_sm61_path():
    m = small().cuda()
    opt = torch.optim.Adam(m.parameters(), lr=3e-4, foreach=False, fused=False)
    loss, state, _ = train_chunk(m, opt, torch.tensor([1, 2, 3, 4], device="cuda"),
                                torch.tensor([2, 3, 4, 5], device="cuda"), m.new_state())
    assert torch.isfinite(torch.tensor(loss))
    assert state.fast.abs().sum() > 0
    assert torch.count_nonzero(m.blocks[0].fc2.weight[m.slow_mask == 0]) == 0


def test_window_includes_current_and_exactly_w_minus_one_past():
    c = HY1Config(vocab_size=11, arm="A0", d_model=16, hidden_dim=32,
                  n_heads=4, n_layers=1, window=4, chunk_size=8)
    m = HY1Model(c)
    x = torch.tensor([1, 2, 3, 4, 5])
    y = x.clone()
    y[0] = 10
    a, _, _ = m.forward_chunk(x)
    b, _, _ = m.forward_chunk(y)
    assert not torch.equal(a[3], b[3])  # distance W-1 included
    torch.testing.assert_close(a[4], b[4], atol=0, rtol=0)  # distance W excluded
