"""Spacing bands: support, exact per-band writes/forgetting, and eager/unfused proof."""
import contextlib
import io

import pytest
import torch

from ephemeral_model import parse_plasticity_bands, write_gain_factor, dfa_layer_step
from tests.test_write_gain import build, build_unclamped, one_step
from tests import scratch_flags_reference as ref
from train import build_model, parse_args, train

BANDS = '0.25:3:0.3,0.25:7:0.01'


def equal_state(a, b):
    assert a.keys() == b.keys()
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0, equal_nan=True, msg=key)


def test_defaults_identical_state_rng_and_run():
    a = build()
    rng = torch.get_rng_state()
    b = build(plasticity_bands='', band_feedback='shared')
    assert torch.equal(rng, torch.get_rng_state())
    equal_state(a.state_dict(), b.state_dict())
    assert not any('band_' in key for key in b.state_dict())
    ar, br = ref.run(a, train), ref.run(b, train)
    assert ar['losses'] == br['losses']
    equal_state(ar['state'], br['state'])


@pytest.mark.parametrize('feedback', ['shared', 'separate'])
def test_partition_same_support_parameters_and_rng(feedback):
    plain = build_unclamped()
    rng = torch.get_rng_state()
    bands = build_unclamped(plasticity_bands=BANDS, band_feedback=feedback)
    assert torch.equal(rng, torch.get_rng_state())
    assert dict(plain.named_parameters()).keys() == dict(bands.named_parameters()).keys()
    for a, b in zip(plain.trained_layers(), bands.trained_layers()):
        assert torch.equal(a.ephemeral_mask, b.band_ids >= 0)
        assert torch.equal(a.per_sample_weights, b.per_sample_weights)
        n = int(a.ephemeral_mask.sum())
        assert int((b.band_ids == 0).sum()) == round(n / 2)
        assert int((b.band_ids == 1).sum()) == n - round(n / 2)
        for band, (_, alpha, rate) in enumerate(parse_plasticity_bands(BANDS)):
            assert (b.plasticity[b.band_ids == band] == alpha).all()
            assert (b.forget_rate[b.band_ids == band] == rate).all()


@pytest.mark.parametrize('feedback', ['shared', 'separate'])
@pytest.mark.parametrize('power', [0, -1, 2])
def test_unfused_fused_logged_exact(feedback, power):
    torch.set_num_threads(1)
    kwargs = dict(plasticity_bands=BANDS, band_feedback=feedback,
                  write_gain_power=power, write_gain_window=3)
    a, b, c = build(**kwargs), build(**kwargs), build(**kwargs)
    b.enable_fused_update(compile=False)
    c.enable_fused_update(compile=False)
    ar, br, cr = ref.run(a, train), ref.run(b, train), ref.run(c, train, log_norms_every=3)
    assert ar['losses'] == br['losses'] == cr['losses']
    equal_state(ar['state'], br['state'])
    equal_state(ar['state'], cr['state'])


@pytest.mark.parametrize('feedback', ['shared', 'separate'])
def test_exact_write_forget_and_heldout(feedback):
    model = build_unclamped(plasticity_bands=BANDS, band_feedback=feedback)
    error = one_step(model, fused=False)
    error[-1] = 0
    projected, _ = model.dfa_step_errors(error)
    fast = model.fast_write_errors(error, projected)
    before = [(layer.per_sample_weights.clone(), layer.bias.clone()) for layer in model.trained_layers()]
    model.fast_only_dfa_step(error, 0.1, 0)
    for index, layer in enumerate(model.trained_layers()):
        initial, bias = before[index]
        assert torch.equal(bias, layer.bias)
        assert torch.equal(initial[:, ~layer.ephemeral_mask], layer.per_sample_weights[:, ~layer.ephemeral_mask])
        for band, (_, alpha, forget) in enumerate(model.plasticity_bands):
            mask = layer.band_ids == band
            e = projected[index] if feedback == 'shared' or layer.is_last_layer else \
                error @ layer.band_feedback_weights[band]
            expected = -0.1 * alpha * e.unsqueeze(2) * layer.in_traces.unsqueeze(1) * (1 - forget)
            torch.testing.assert_close(layer.per_sample_weights[:, mask], expected[:, mask], rtol=1e-6, atol=1e-7)
        assert torch.isfinite(layer.per_sample_weights).all()


def test_wipe_and_override_restore_including_exception_and_dtype_move():
    model = build_unclamped(plasticity_bands=BANDS).double().eval()
    for layer in model.trained_layers():
        assert layer.forget_rate.dtype == torch.float64
        layer.per_sample_weights.data.fill_(2)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    model.wipe_bands([0])
    for layer in model.trained_layers():
        assert (layer.per_sample_weights[:, layer.band_ids == 0] == 0).all()
        assert (layer.per_sample_weights[:, layer.band_ids != 0] == 2).all()
    with pytest.raises(RuntimeError):
        with model.band_forget_override({1: 0}):
            for layer in model.trained_layers():
                assert (layer.forget_rate[layer.band_ids == 1] == 0).all()
            with model.band_forget_override({0: 1}):
                for layer in model.trained_layers():
                    assert (layer.forget_rate[layer.band_ids == 0] == 1).all()
            raise RuntimeError('restore')
    for name, layer in model.named_modules():
        if hasattr(layer, 'band_forget_rates'):
            assert torch.equal(layer.forget_rate, before[name + '.band_forget_rates'])
    model.train()
    with pytest.raises(ValueError), model.band_forget_override({0: 0}):
        pass


@pytest.mark.parametrize('feedback', ['shared', 'separate'])
def test_flat_write_norm_is_median_and_zero_stays_zero(feedback):
    model = build_unclamped(plasticity_bands=BANDS, band_feedback=feedback, write_gain_power=-1)
    model.write_gain_history.fill_(2.5)
    error = one_step(model, fused=False)
    error[-1] = 0
    projected, _ = model.dfa_step_errors(error)
    gained = model.fast_write_errors(error, projected)
    for index in model.write_gain_layers:
        layer = model.trained_layers()[index]
        if feedback == 'shared':
            torch.testing.assert_close(gained[index].norm(dim=1), torch.tensor([2.5, 2.5, 0.]))
        else:
            for band in range(2):
                raw = error @ layer.band_feedback_weights[band]
                expected = raw * write_gain_factor(raw.norm(dim=1)[None], torch.tensor([2.5]), -1)[0, :, None]
                expanded = expected.unsqueeze(2).expand_as(gained[index])
                torch.testing.assert_close(gained[index][:, layer.band_ids == band], expanded[:, layer.band_ids == band])
        assert torch.isfinite(gained[index]).all()


def test_build_flags_and_checkpoint_reject_semantic_change():
    args = parse_args(['--model_type', 'ephemeral', '--updater', 'dfa', '--plasticity_bands', BANDS,
                       '--band_feedback', 'separate', '--write_gain_power', '-1'])
    assert args.ephemeral_fraction == 0.5
    config = dict(vars(args), n_hidden=4, n_layers=3, pe_matrix=None, batch_size=3)
    with contextlib.redirect_stdout(io.StringIO()):
        model = build_model(config, ref.CHARSET, len(ref.CHARSET))
    assert model.band_feedback == 'separate'
    state = model.state_dict()
    model.load_state_dict(state)
    wrong = build_unclamped(plasticity_bands=BANDS)
    with pytest.raises(RuntimeError):
        wrong.load_state_dict(state)


@pytest.mark.parametrize('feedback', ['shared', 'separate'])
def test_nonuniform_plasticity_cache_is_not_a_registered_parameter(feedback):
    model = build_unclamped(plasticity_bands=BANDS, band_feedback=feedback)
    keys = set(model.state_dict())
    parameters = set(dict(model.named_parameters()))
    model.enable_fused_update(compile=False)
    error = one_step(model, fused=True)
    assert set(model.state_dict()) == keys
    model.fast_only_dfa_step(error, 0.1, 0)
    assert set(model.state_dict()) == keys
    assert set(dict(model.named_parameters())) == parameters
    assert not any('_fused_plasticity' in key for key in model.state_dict())

    # A real serialized checkpoint must strictly load both before and after cache use.
    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    state = torch.load(checkpoint, weights_only=True)
    fresh = build_unclamped(plasticity_bands=BANDS, band_feedback=feedback)
    fresh.load_state_dict(state, strict=True)
    equal_state(model.state_dict(), fresh.state_dict())
    model.load_state_dict(state, strict=True)
    assert set(model.state_dict()) == keys

    layer = model.linear_layers[0]
    assert layer.fused_plasticity() is layer.plasticity
    model.double()
    assert layer.fused_plasticity() is layer.plasticity
    assert layer.fused_plasticity().dtype == torch.float64


def test_compiled_fullgraph_support_and_override_decay():
    model = build_unclamped(plasticity_bands=BANDS, band_feedback='separate').eval()
    eager = build_unclamped(plasticity_bands=BANDS, band_feedback='separate').eval()
    eager.enable_fused_update(compile=False)
    model.fused_layer_step = torch.compile(dfa_layer_step, backend='eager', fullgraph=True)
    for m in (model, eager):
        error = one_step(m, fused=True)
        for layer in m.trained_layers():
            layer.per_sample_weights.data[:, layer.ephemeral_mask] = 1
        with m.band_forget_override({0: 1, 1: 0}):
            m.fast_only_dfa_step(torch.zeros_like(error), 0.1, 0)
        for layer in m.trained_layers():
            assert (layer.per_sample_weights[:, layer.band_ids == 0] == 0).all()
            assert (layer.per_sample_weights[:, layer.band_ids == 1] == 1).all()
    equal_state(model.state_dict(), eager.state_dict())


@pytest.mark.parametrize('text', ['0:2:0.1', '0.8:2:0,0.4:3:0', '0.2:-2:0', 'nan:2:0', '0.2:2:1.1', '0.1:2'])
def test_invalid_band_spec(text):
    with pytest.raises(ValueError):
        parse_plasticity_bands(text)


@pytest.mark.parametrize('extra', [dict(updater='backprop'), dict(fast_structure='additive_dense'),
    dict(order2_layers=[0]), dict(slow_update_every=2), dict(fast_backward_per_forward=2),
    dict(decouple_fast_feedback=[0]), dict(fast_read_norm='rms')])
def test_unsupported_combinations(extra):
    with pytest.raises(ValueError):
        build(plasticity_bands=BANDS, **extra)


@pytest.mark.parametrize('feedback', ['shared', 'separate'])
def test_float_checkpoint_loads_into_fresh_model(feedback):
    source = build_unclamped(plasticity_bands=BANDS, band_feedback=feedback).float()
    fresh = build_unclamped(plasticity_bands=BANDS, band_feedback=feedback)
    fresh.load_state_dict(source.state_dict(), strict=True)
    for a, b in zip(source.trained_layers(), fresh.trained_layers()):
        assert torch.equal(a.band_ids, b.band_ids)
        assert torch.equal(a.band_forget_rates, b.band_forget_rates)
    wrong = build_unclamped(plasticity_bands='0.25:3:0.2,0.25:7:0.01', band_feedback=feedback)
    with pytest.raises(RuntimeError, match='configuration'):
        wrong.load_state_dict(source.state_dict())


@pytest.mark.parametrize('power', [0, -1, 2])
def test_separate_feedback_changes_only_fast_writes_on_same_forward(power):
    shared = build_unclamped(plasticity_bands=BANDS, write_gain_power=power)
    separate = build_unclamped(plasticity_bands=BANDS, band_feedback='separate', write_gain_power=power)
    for m in (shared, separate):
        m.enable_fused_update(compile=False)
        if power:
            m.write_gain_history.fill_(2.5)
    assert torch.equal(one_step(shared), one_step(separate))
    changed = False
    for a, b in zip(shared.trained_layers(), separate.trained_layers()):
        assert torch.equal(a.bias, b.bias)
        assert torch.equal(a.per_sample_weights[:, ~a.ephemeral_mask],
                           b.per_sample_weights[:, ~b.ephemeral_mask])
        changed |= not torch.equal(a.per_sample_weights[:, a.ephemeral_mask],
                                   b.per_sample_weights[:, b.ephemeral_mask])
    assert changed


@pytest.mark.parametrize('feedback', ['shared', 'separate'])
def test_flat_history_is_causal_per_projection_and_inference_frozen(feedback):
    m = build_unclamped(plasticity_bands=BANDS, band_feedback=feedback,
                        write_gain_power=-1, write_gain_window=2)
    m.enable_fused_update(compile=False)
    error = one_step(m, fused=False)
    error[-1] = 0
    projected, _ = m.dfa_step_errors(error)
    raw = m._fast_error_overrides(error, projected)
    expected = torch.stack([((projected[index] if raw[index] is None else raw[index])
                             if band is None else raw[index][band]).norm(dim=1)
                            for index, band in m.write_gain_slots])
    first = m.fast_write_errors(error, projected)
    assert torch.isnan(m.write_gain_history).all()
    for item in first:
        if item is not None:
            assert torch.isfinite(item).all()
    m.fused_dfa_step(error, 0.1, 0)
    torch.testing.assert_close(m.write_gain_history[:, :2], expected[:, :2], rtol=0, atol=0)
    assert torch.isnan(m.write_gain_history[:, 2:]).all()
    history = m.write_gain_history.clone()
    m.eval()
    with m.band_forget_override({0: 0}):
        m.fast_only_dfa_step(error, 0.1, 0)
    torch.testing.assert_close(m.write_gain_history, history, rtol=0, atol=0, equal_nan=True)
    assert int(m.write_gain_steps) == 1


@pytest.mark.parametrize('feedback', ['shared', 'separate'])
def test_fprime_slow_normalization_and_sparse_fast_layers_path_parity(feedback):
    kwargs = dict(plasticity_bands=BANDS, band_feedback=feedback, write_gain_power=-1,
                  write_gain_window=3, dfa_fprime=True, slow_nlms=True,
                  trunk_slow_lr_scale=0.7, readout_slow_lr_scale=0.3, fast_layers=[1])
    a, b = build(**kwargs), build(**kwargs)
    b.enable_fused_update(compile=False)
    ar, br = ref.run(a, train), ref.run(b, train)
    assert ar['losses'] == br['losses']
    equal_state(ar['state'], br['state'])
