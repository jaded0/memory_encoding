"""Fast-only cousin transforms: independent algebra, path parity, state and defaults."""
import contextlib
import copy
import io
import math

import pytest
import torch

from ephemeral_model import EphemeralRNN, fast_key_features, dfa_layer_step
from reproducibility import seed_everything
from tests import scratch_flags_reference as ref
from train import train, parse_args, build_model
from utils import load_checkpoint, CHECKPOINT_CODE_VERSION
from loop_trace import LoopTracer, row_norm


def build(**kwargs):
    return ref.build(EphemeralRNN, seed_everything, **kwargs)


def test_default_bit_identity_against_baseline_fixture():
    default = ref.run(build(), train)
    explicit = ref.run(build(fast_key_rope=0, fast_write_momentum=0, fast_key_shift=0), train)
    assert default['losses'] == explicit['losses']
    for key, value in default['state'].items():
        assert torch.equal(value, explicit['state'][key])
    fixture = torch.load(ref.__file__.replace('scratch_flags_reference.py',
                                             'fixtures/scratch_flags.pt'), weights_only=False)
    tolerance = dict(rtol=0, atol=0) if fixture['generated_with']['torch'] == str(torch.__version__) else dict(rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(torch.tensor(default['losses']),
                               torch.tensor(fixture['main']['plain']['losses']), **tolerance)
    for key, value in default['state'].items():
        torch.testing.assert_close(value, fixture['main']['plain']['state'][key], **tolerance)
    assert not any('fast_position' in k or 'fast_previous_input' in k or 'fast_momentum' in k
                   for k in build().state_dict())


def test_shift_then_rope_independent_odd_dimension():
    x = torch.tensor([[1., 2., 3., 4., 5.]])
    previous = torch.tensor([[5., 4., 3., 2., 1.]])
    expected = .7 * x + .3 * previous
    original = expected.clone()
    for j in range(2):
        theta = 3 * 100. ** (-2 * j / 5)
        a, b = original[:, 2*j], original[:, 2*j+1]
        expected[:, 2*j] = a * math.cos(theta) - b * math.sin(theta)
        expected[:, 2*j+1] = a * math.sin(theta) + b * math.cos(theta)
    torch.testing.assert_close(fast_key_features(x, previous, torch.tensor(3), 100., .3), expected)
    torch.testing.assert_close(fast_key_features(x, previous, torch.tensor(0), 100., 0), x)


@pytest.mark.parametrize('flags', [dict(fast_key_rope=100.), dict(fast_key_shift=.4),
                                  dict(fast_key_rope=100., fast_key_shift=.4)])
def test_transforms_only_fast_read_and_write_input(flags):
    model = build(**flags)
    layer = model.linear_layers[0]
    model.start_sequence_wipe()
    torch.manual_seed(2)
    layer.per_sample_weights.data.normal_()
    slow = layer.slow_weights().clone()
    fast = layer.fast_weights().clone()
    prev = torch.zeros_like(layer.in_traces)
    for t in range(3):
        x = torch.randn_like(prev)
        key = fast_key_features(x, prev, torch.tensor(t), flags.get('fast_key_rope', 0),
                                flags.get('fast_key_shift', 0))
        expected = torch.bmm(slow, x.unsqueeze(2)).squeeze(2) + torch.bmm(fast, key.unsqueeze(2)).squeeze(2) + layer.bias
        torch.testing.assert_close(layer(x), expected)
        assert torch.equal(layer.in_traces, x)
        torch.testing.assert_close(layer.fast_key_trace, key)
        prev = x
    # Gradient's slow entries still use raw x; only fast entries use transformed key.
    error = torch.randn(ref.BATCH_SIZE, len(ref.CHARSET))
    model.clear_dfa_gradients()
    for target in model.trained_layers():
        if target is not layer:
            target(torch.randn_like(target.in_traces))
        target.populate_dfa_gradients(error)
    raw = layer.per_sample_weights.grad.clone()
    projected, _ = model.dfa_step_errors(error)
    model.override_fast_gradients(error)
    expected = torch.where(layer.ephemeral_mask, projected[0].unsqueeze(2) * layer.fast_key_trace.unsqueeze(1), raw)
    torch.testing.assert_close(layer.per_sample_weights.grad, expected)


def test_momentum_algebra_mask_clamp_forget_and_zero_error():
    weights = torch.tensor([[[.2, .3], [.4, .5]]])
    mask = torch.tensor([[True, False], [False, True]])
    momentum = torch.zeros_like(weights)
    expected = weights.clone()
    velocity = torch.zeros_like(weights)
    for error in (torch.tensor([[2., -3.]]), torch.zeros(1, 2), torch.tensor([[-1., 4.]])):
        x = torch.tensor([[2., -1.]])
        gradient = error.unsqueeze(2) * x.unsqueeze(1)
        u = torch.where(mask, (-3 * gradient).clamp(-.7, .7), -gradient)
        velocity = torch.where(mask, .6 * velocity + u, 0)
        update = torch.where(mask, velocity, u)
        expected = (expected + .2 * update).clamp(-2, 2)
        expected = torch.where(mask, expected.clamp(-.8, .8), expected)
        expected *= torch.where(mask, .75, .9)
        dfa_layer_step(weights, None, error, x, 3., mask, .25, .2, .7, 2., False,
                       .1, .8, momentum=momentum, momentum_mu=.6)
        torch.testing.assert_close(weights, expected)
        torch.testing.assert_close(momentum, velocity)


@pytest.mark.parametrize('flags', [dict(fast_key_rope=100.), dict(fast_key_shift=.4),
                                  dict(fast_write_momentum=.7),
                                  dict(fast_key_rope=100., fast_key_shift=.4, fast_write_momentum=.7),
                                  dict(fast_key_rope=100., fast_key_shift=.4, fast_write_momentum=.7,
                                       decouple_fast_feedback=[1], slow_nlms=True),
                                  dict(fast_write_momentum=.7, write_gain_power=.5),
                                  dict(fast_key_rope=100., fast_read_norm='ln')])
def test_unfused_fused_eager_and_logging_parity_and_effect(flags):
    baseline = ref.run(build(), train)
    unfused = ref.run(build(**flags), train)
    model = build(**flags)
    model.enable_fused_update(compile=False)
    fused = ref.run(model, train)
    assert unfused['losses'] != baseline['losses']
    assert unfused['losses'] == fused['losses']
    for key, value in unfused['state'].items():
        assert torch.equal(value, fused['state'][key]), key
    model = build(**flags)
    model.enable_fused_update(compile=False)
    logged = ref.run(model, train, log_norms_every=2)
    assert logged['losses'] == unfused['losses']


def test_state_wipe_keep_checkpoint_and_frozen_slow():
    flags = dict(fast_key_rope=100., fast_key_shift=.4, fast_write_momentum=.7, fast_layers=[1])
    model = build(**flags)
    assert model.linear_layers[0].fast_position is None
    assert model.i2o.fast_position is None
    ref.run(model, train)
    saved = copy.deepcopy(model.state_dict())
    clone = build(**flags)
    clone.load_state_dict(saved)
    layer = model.linear_layers[1]
    prev, pos, mom = layer.fast_previous_input.clone(), layer.fast_position.clone(), layer.fast_momentum.clone()
    model.start_sequence_wipe(wipe_fast=False)
    assert torch.equal(layer.fast_previous_input, prev)
    assert torch.equal(layer.fast_position, pos)
    assert torch.equal(layer.fast_momentum, mom)
    # Identical checkpoint continuation, including carry across a no-wipe boundary.
    clone.start_sequence_wipe(wipe_fast=False)
    x = torch.randn(ref.BATCH_SIZE, len(ref.CHARSET))
    hidden = model.initHidden(ref.BATCH_SIZE)
    with torch.no_grad():
        a, _ = model(x, hidden)
        b, _ = clone(x, hidden)
    assert torch.equal(a, b)
    before = [(l.slow_weights().clone(), l.bias.clone()) for l in model.trained_layers()]
    model.fast_only_dfa_step(torch.randn_like(a), .1, .2)
    for l, (slow, bias) in zip(model.trained_layers(), before):
        assert torch.equal(l.slow_weights(), slow)
        assert torch.equal(l.bias, bias)
    model.start_sequence_wipe()
    assert not layer.fast_previous_input.any()
    assert not layer.fast_position.any()
    assert not layer.fast_momentum.any()
    assert layer.fast_key_trace is None
    # A mid-story fast wipe must not consolidate the slow copies.
    ref.run(model, train)
    slow_before = [l.slow_weights().clone() for l in model.trained_layers()]
    model.wipe_fast_state()
    for l, slow in zip(model.trained_layers(), slow_before):
        assert torch.equal(l.slow_weights(), slow)
        for state in (l.fast_previous_input, l.fast_position, l.fast_momentum):
            if state is not None:
                assert not state.any()


@pytest.mark.parametrize('flags', [dict(fast_key_rope=-1), dict(fast_key_rope=float('nan')),
                                  dict(fast_write_momentum=1), dict(fast_key_shift=-.1),
                                  dict(fast_key_shift=.4, updater='backprop'),
                                  dict(fast_write_momentum=.4, fast_backward_per_forward=2),
                                  dict(fast_key_rope=100, slow_update_every='sequence'),
                                  dict(fast_key_shift=.4, fast_structure='additive_dense'),
                                  dict(fast_key_rope=100, order2_layers=[1]),
                                  dict(fast_key_rope=100, fast_read_norm='absrun', fast_read_norm_lr=.1)])
def test_refuses_undefined_interactions(flags):
    with pytest.raises(ValueError):
        build(**flags)


def test_cli_and_builder():
    args = parse_args(['--fast_key_rope', '100', '--fast_write_momentum', '.7', '--fast_key_shift', '.4'])
    config = dict(vars(args), n_hidden=8, n_layers=3, pe_matrix=None)
    with contextlib.redirect_stdout(io.StringIO()):
        model = build_model(config, ref.CHARSET, len(ref.CHARSET))
    assert model.linear_layers[0].fast_key_rope == 100
    assert model.i2h.fast_write_momentum == .7
    with pytest.raises(SystemExit):
        parse_args(['--fast_key_shift', '.4', '--grad_norm_clip', '1'])


def test_resume_configuration_guards():
    args = parse_args(['--fast_key_rope', '100', '--fast_write_momentum', '.7',
                      '--fast_key_shift', '.4', '--batch_size', '2'])
    config = dict(vars(args), n_hidden=8, n_layers=3, pe_matrix=None)
    with contextlib.redirect_stdout(io.StringIO()):
        model = build_model(config, ref.CHARSET, len(ref.CHARSET))
        checkpoint = dict(code_version=CHECKPOINT_CODE_VERSION, config=dict(config),
                          model_state_dict=model.state_dict(), iter=1)
        load_checkpoint('unused', model, config, checkpoint=checkpoint)
        for key in ('fast_key_rope', 'fast_write_momentum', 'fast_key_shift'):
            checkpoint['config'] = dict(config, **{key: 0.0})
            with pytest.raises(RuntimeError, match='configuration mismatch'):
                load_checkpoint('unused', model, config, checkpoint=checkpoint)


def test_strict_mask_does_not_emit_residual_momentum():
    model = build(fast_write_momentum=.7)
    ref.run(model, train)
    before = [(l.fast_weights().clone(), None if l.fast_momentum is None else l.fast_momentum.clone())
              for l in model.trained_layers()]
    error = torch.zeros(ref.BATCH_SIZE, len(ref.CHARSET))
    model.fast_only_dfa_step(error, .1, .2, write_mask=torch.zeros(ref.BATCH_SIZE, dtype=torch.bool))
    for layer, (fast, momentum) in zip(model.trained_layers(), before):
        if layer.is_last_layer:
            continue
        torch.testing.assert_close(layer.fast_weights(), (1-layer.forget_rate) * fast)
        if momentum is not None:
            assert torch.equal(layer.fast_momentum, momentum)


def test_compiled_update_with_transformed_key_and_momentum():
    # CPU Inductor exercises the actual fused kernel, not merely the eager fused dispatcher.
    eager = build(fast_key_rope=100, fast_key_shift=.4, fast_write_momentum=.7)
    compiled = build(fast_key_rope=100, fast_key_shift=.4, fast_write_momentum=.7)
    eager.enable_fused_update(compile=False)
    compiled.enable_fused_update(compile=True)
    generator = torch.Generator().manual_seed(22)
    hidden = eager.initHidden(ref.BATCH_SIZE)
    for _ in range(3):
        x = torch.randn(ref.BATCH_SIZE, len(ref.CHARSET), generator=generator)
        error = torch.randn(ref.BATCH_SIZE, len(ref.CHARSET), generator=generator)
        with torch.no_grad():
            eager(x, hidden)
            compiled(x, hidden)
        eager.fused_dfa_step(error, .1, .2)
        compiled.fused_dfa_step(error, .1, .2)
    for key, value in eager.state_dict().items():
        torch.testing.assert_close(compiled.state_dict()[key], value, rtol=1e-5, atol=1e-6)


def test_tracer_observes_transformed_reads_and_momentum_without_mutation():
    model = build(fast_key_rope=100, fast_key_shift=.4, fast_write_momentum=.7)
    ref.run(model, train)
    x = torch.randn(ref.BATCH_SIZE, len(ref.CHARSET))
    with torch.no_grad():
        output, hidden = model(x, model.initHidden(ref.BATCH_SIZE))
    error = torch.randn_like(output)
    tracer = LoopTracer(model, .1, update_clamp=.2)
    tracer.begin(1, ref.BATCH_SIZE)
    saved = copy.deepcopy(model.state_dict())
    tracer.before_update(0, output, error, torch.ones(ref.BATCH_SIZE), hidden)
    for key, value in saved.items():
        assert torch.equal(model.state_dict()[key], value), key
    layer = model.linear_layers[0]
    expected_read = torch.bmm(layer.fast_weights(), layer.fast_key_trace.unsqueeze(2))
    torch.testing.assert_close(tracer.buffers['fast_drive'][0, :, 0], row_norm(expected_read))
    projected = model.fast_write_projected(error)[0]
    raw = -layer.plasticity * (projected.unsqueeze(2) * layer.fast_key_trace.unsqueeze(1))
    expected_write = .7 * layer.fast_momentum + raw.clamp(-.2, .2)
    torch.testing.assert_close(tracer.buffers['fast_write'][0, :, 0],
                               .1 * row_norm(expected_write * layer.fast_mask))
