"""K15: positive stored-key mass, null sink, actual DFA routes and exact continuation."""
import contextlib
import io
import subprocess
import tempfile
import types
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from ephemeral_model import EphemeralRNN, dfa_output_error, order2_features, order2_kernel_norm
from heldout import evaluate_protocols
from reproducibility import seed_everything
from tests import scratch_flags_reference as ref
from tests.test_order2 import build
from tests.test_seed_resume import run_main, latest
import train as train_module
from train import build_model, parse_args, train
from utils import CHECKPOINT_CODE_VERSION, load_checkpoint


def summed(**kwargs):
    return build(order2_layers=[0, 1, 2], order2_read_norm='sum', **kwargs)


def assert_state(a, b):
    assert a.keys() == b.keys()
    for key in a:
        assert torch.equal(a[key], b[key]), key


def test_defaults_bit_identical_to_0f512f8_par2_trace():
    # Read the exact historical source, rather than comparing two new-code flag settings.
    # This checkout is intentionally based on K13's 0f512f8 (not main).
    old = types.ModuleType('k13_order2_reference')
    source = subprocess.check_output(['git', 'show', '0f512f8:ephemeral_model.py'], text=True)
    exec(compile(source, '0f512f8:ephemeral_model.py', 'exec'), old.__dict__)
    for fused in (False, True):
        before = ref.build(old.EphemeralRNN, seed_everything, order2_layers=[0, 1, 2])
        after = build(order2_layers=[0, 1, 2])
        if fused:
            before.enable_fused_update(compile=False)
            after.enable_fused_update(compile=False)
        with patch.object(train_module, 'EphemeralRNN', old.EphemeralRNN):
            a = ref.run(before, train)
        b = ref.run(after, train)
        assert a['losses'] == b['losses']
        assert_state(before.state_dict(), after.state_dict())
        assert not any('order2_z' in k or 'norm_counts' in k for k in after.state_dict())


@pytest.mark.parametrize('temperature', [4., 16.])
@pytest.mark.parametrize('sink', ['off', 'fixed'])
def test_explicit_decayed_keys_positive_unit_mass_and_forward(temperature, sink):
    torch.set_num_threads(1)
    model = summed(order2_temperature=temperature, order2_sink=sink)
    layer = model.linear_layers[1]
    layer.weight_clamp = layer.fast_weight_clamp = 0
    stored, values = [], []
    torch.manual_seed(121)
    for i in range(9):
        x = torch.randn(ref.BATCH_SIZE, layer.in_features)
        error = torch.randn(ref.BATCH_SIZE, layer.out_features)
        if i == 3:
            error.zero_()  # It is still a key write, including for heldout's masked rows.
        layer(x)
        stored.append(layer.order2_trace.clone())
        # lr and alpha belong to the value: no lr*alpha in z or in the sink's units.
        lr = .03 * (i + 1)
        values.append(-lr * layer.order2_alpha() * error)
        layer.order2_write(error, lr, 0)
    decay = torch.tensor([(1-layer.forget_rate)**(len(stored)-i) for i in range(len(stored))])
    keys = torch.stack(stored)
    explicit = (decay[:, None, None] * keys).sum(0)
    torch.testing.assert_close(layer.order2_z, explicit)
    assert not layer.order2_z.requires_grad
    query = torch.randn(ref.BATCH_SIZE, layer.in_features)
    psi = order2_features(query, layer.order2_proj, 'order2', temperature, layer.order2_fraction)
    masses = (keys * psi).sum(-1) * decay[:, None]
    assert bool((masses > 0).all())
    denominator = masses.sum(0)
    if sink == 'fixed':
        denominator += layer.order2_fraction * query.square().sum(1) / order2_kernel_norm('order2', temperature)
    weights = masses / denominator
    if sink == 'off':
        torch.testing.assert_close(weights.sum(0), torch.ones(ref.BATCH_SIZE))
    else:
        assert bool((weights.sum(0) < 1).all())
    expected = (weights[:, :, None] * torch.stack(values)).sum(0)
    slow = torch.bmm(layer.per_sample_weights, query.unsqueeze(2)).squeeze(2) + layer.bias
    torch.testing.assert_close(layer(query) - slow, expected, rtol=1e-4, atol=1e-6)
    # No stored writes, or zero query: finite zero fast read, despite the epsilon floor.
    model.start_sequence_wipe()
    assert torch.isfinite(layer(query)).all()
    assert torch.isfinite(layer(torch.zeros_like(query))).all()


def test_reset_carry_and_no_forget_match_g():
    model = summed()
    layer = model.linear_layers[1]
    layer(torch.randn(ref.BATCH_SIZE, layer.in_features))
    psi = layer.order2_trace.clone()
    layer.order2_write(torch.ones(ref.BATCH_SIZE, layer.out_features), .1, 0, forget=False)
    assert torch.equal(layer.order2_z, psi)
    g, z = layer.order2_fast.clone(), layer.order2_z.clone()
    model.start_sequence_wipe(wipe_fast=False)
    assert torch.equal(layer.order2_fast, g)
    assert torch.equal(layer.order2_z, z)
    model.start_sequence_wipe()
    assert not layer.order2_z.any()
    assert not layer.order2_fast.any()


def test_unfused_fused_logging_and_heldout_use_identical_z_route():
    models = [summed() for _ in range(3)]
    models[1].enable_fused_update(compile=False)
    models[2].enable_fused_update(compile=False)
    runs = [ref.run(m, train, log_norms_every=3 if i == 2 else 0) for i, m in enumerate(models)]
    for m, result in zip(models[1:], runs[1:]):
        assert result['losses'] == runs[0]['losses']
        # The existing log-only norm fields intentionally differ on logging steps.
        states = [{k: v for k, v in candidate.state_dict().items()
                   if not k.endswith(('last_ephemeral_step_norm', 'last_slow_step_norm'))}
                  for candidate in (models[0], m)]
        assert_state(*states)
    assert models[0].linear_layers[1].order2_z.abs().sum() > 0
    model = models[0]
    saved = {k: v.clone() for k, v in model.state_dict().items()}
    texts = ['abcdab', 'bcdabc', 'cdabcd']
    onehot = F.one_hot(torch.tensor([[ref.CHARSET.index(c) for c in t] for t in texts]), 4).float()
    config = dict(input_mode='last_one', pe_matrix=None, learning_rate=.1, ephemeral_update_clamp=0)
    with contextlib.redirect_stdout(io.StringIO()):
        evaluate_protocols(model, [(texts, onehot)], config, 'palindrome_dataset',
                           protocols=('observed', 'strict', 'no_fast'))
    assert_state(saved, model.state_dict())  # z and cumulative counters restored, not just G


def test_beta_gets_no_gradient_in_actual_dfa_training():
    # Even a differentiable scalar in the read is not trained: DFA asks only for dL/dlogits,
    # then manually updates named layer weights/biases. There is no loss.backward().
    model = summed(order2_sink='fixed')
    layer = model.linear_layers[0]
    beta = torch.nn.Parameter(torch.zeros(()))
    layer.register_parameter('probe_beta', beta)
    original = layer.forward
    with patch.object(layer, 'forward', side_effect=lambda x: original(x) / beta.exp()):
        ref.run(model, train)
    assert beta.grad is None
    assert beta.item() == 0
    # A direct differentiable toy read confirms the same dfa_output_error contract.
    logits = torch.randn(3, 4) / beta.exp()
    dfa_output_error(logits, F.one_hot(torch.tensor([0, 1, 2]), 4).float(),
                     torch.nn.CrossEntropyLoss(reduction='none'))
    assert beta.grad is None


@pytest.mark.parametrize('kwargs', [dict(order2_sink='fixed'), dict(order2_sink='learned'),
    dict(order2_read_norm='sum'), dict(order2_layers=[1], order2_map='order1', order2_read_norm='sum'),
    dict(order2_layers=[1], order2_map='sparse', order2_read_norm='sum'),
    dict(order2_layers=[1], order2_read_norm='sum', updater='backprop'),
    dict(order2_layers=[1], order2_read_norm='sum', slow_update_every=2),
    dict(order2_layers=[1], order2_read_norm='sum', fast_backward_per_forward=2),
    dict(order2_layers=[1], order2_read_norm='sum', fast_read_norm='rms'),
    dict(order2_layers=[1], order2_read_norm='sum', forget_rate=1.1)])
def test_unsupported_paths_refused(kwargs):
    # forget_rate cannot be passed through the reference build's fixed keyword.
    if 'forget_rate' in kwargs:
        flags = ['--model_type', 'ephemeral', '--updater', 'dfa']
        for k, v in kwargs.items():
            flags.extend(['--'+k, ','.join(map(str, v)) if isinstance(v, list) else str(v)])
        with pytest.raises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            parse_args(flags)
    else:
        with pytest.raises(ValueError):
            build(**kwargs)


def test_flags_and_resume_mismatch_and_old_defaults():
    base = ['--model_type', 'ephemeral', '--updater', 'dfa', '--order2_layers', '0',
            '--hidden_size', '4', '--num_layers', '1', '--batch_size', '2']
    args = parse_args(base + ['--order2_read_norm', 'sum', '--order2_sink', 'fixed'])
    config = dict(vars(args), n_hidden=4, n_layers=1, pe_matrix=None)
    with contextlib.redirect_stdout(io.StringIO()):
        model = build_model(config, ref.CHARSET, 4)
    assert model.linear_layers[0].order2_read_norm == 'sum'
    checkpoint = dict(code_version=CHECKPOINT_CODE_VERSION, config=config,
                      model_state_dict=model.state_dict(), iter=1)
    for key in ('order2_sink', 'order2_read_norm'):
        with pytest.raises(RuntimeError), contextlib.redirect_stdout(io.StringIO()):
            load_checkpoint('unused', model, dict(config, **{key: 'off'}), checkpoint=checkpoint)
    for argv in (['--order2_sink', 'learned', '--order2_read_norm', 'sum'],
                 ['--order2_sink', 'fixed'], ['--order2_read_norm', 'sum', '--grad_norm_clip', '1']):
        with pytest.raises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            parse_args(base + argv)


def test_clamp_diagnostics_count_before_forget_and_restore():
    model = summed()
    layer = model.linear_layers[1]
    layer.weight_clamp = .001
    layer.order2_trace = torch.ones(ref.BATCH_SIZE, layer.order2_fast.shape[-1])
    layer.order2_write(torch.ones(ref.BATCH_SIZE, layer.out_features), 1., .5)
    summary = model.order2_norm_summary()
    assert summary['order2/layer_1/G_clamp_bound_fraction'] == 1.
    assert summary['order2/layer_1/update_clamp_fraction'] == 1.
    assert layer.order2_fast.abs().max() < .001  # forgetting moved every G off its bound


@pytest.mark.parametrize('wipe', ['1', '2'])
def test_live_z_checkpoint_resume_bit_exact(wipe):
    flags = ('--seed', '2718', '--deterministic', 'true', '--order2_layers', '0',
             '--order2_read_norm', 'sum', '--order2_sink', 'fixed', '--wipe_every', wipe)
    with tempfile.TemporaryDirectory() as straight, tempfile.TemporaryDirectory() as split:
        run_main(*flags, '--n_iters', '6', checkpoint_dir=straight)
        run_main(*flags, '--n_iters', '3', checkpoint_dir=split)
        checkpoint = latest(split)
        assert checkpoint['model_state_dict']['linear_layers.0.order2_z'].abs().sum() > 0
        run_main(*flags, '--resume', '--n_iters', '6', checkpoint_dir=split)
        a, b = latest(straight), latest(split)
        assert_state(a['model_state_dict'], b['model_state_dict'])
        for key, value in a['data_stream_state'].items():
            other = b['data_stream_state'][key]
            assert torch.equal(value, other) if torch.is_tensor(value) else value == other


def test_unnormalized_replace_clamp_logging_and_both_weight_bounds():
    model = build(order2_layers=[1], order2_mode='replace')
    layer = model.linear_layers[1]
    # The tighter fast clamp survives even if the weight clamp was applied first.
    for weight, fast in ((.1, .03), (.03, .1)):
        layer.weight_clamp, layer.fast_weight_clamp = weight, fast
        layer.order2_norm_counts.zero_()
        layer.order2_trace = torch.ones(ref.BATCH_SIZE, layer.order2_fast.shape[-1])
        layer.order2_write(torch.ones(ref.BATCH_SIZE, layer.out_features), 1., 0)
        assert model.order2_norm_summary()['order2/layer_1/G_clamp_bound_fraction'] == 1.
        torch.testing.assert_close(layer.order2_fast.abs(),
                                   torch.full_like(layer.order2_fast, .03 * (1-layer.forget_rate)))
    # K13 bigkey's no-clamp recipe explicitly reports zero, even on unnormalized arms.
    layer.weight_clamp = layer.fast_weight_clamp = 0
    layer.order2_norm_counts.zero_()
    layer.order2_write(torch.ones(ref.BATCH_SIZE, layer.out_features), .1, 0)
    assert set(model.order2_norm_summary().values()) == {0.}
    assert not any('norm_counts' in key for key in model.state_dict())


def test_legacy_par2_checkpoint_missing_new_flags_loads_with_off_defaults():
    args = parse_args(['--model_type', 'ephemeral', '--updater', 'dfa', '--order2_layers', '0',
                       '--hidden_size', '4', '--num_layers', '1', '--batch_size', '2'])
    config = dict(vars(args), n_hidden=4, n_layers=1, pe_matrix=None)
    with contextlib.redirect_stdout(io.StringIO()):
        model = build_model(config, ref.CHARSET, 4)
        old_config = {k: v for k, v in config.items() if k not in ('order2_read_norm', 'order2_sink')}
        checkpoint = dict(code_version=CHECKPOINT_CODE_VERSION, config=old_config,
                          model_state_dict=model.state_dict(), iter=1)
        load_checkpoint('unused', model, config, checkpoint=checkpoint)
