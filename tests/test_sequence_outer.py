"""K14: matched sequence outer loops; functional DFA writes and eager evaluation."""
import contextlib
import copy
import io
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from ephemeral_model import EphemeralRNN
from heldout import evaluate_held_out, evaluate_protocols
from sequence_outer import check_sequence_outer, sequence_rollout
from train import build_model, parse_args, train_batch
from utils import CHECKPOINT_CODE_VERSION, load_checkpoint


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def config(**extra):
    return dict(model_type='ephemeral', updater='backprop', sequence_outer=True,
                fast_write_grad='through', backprop_step_parity=True, max_seq_len=10,
                criterion=torch.nn.CrossEntropyLoss(reduction='none'),
                input_mode='last_one', pe_matrix=None, learning_rate=.03,
                ephemeral_update_clamp=0, **extra)


def model(batch=2, recurrent=True, **extra):
    torch.manual_seed(17)
    input_size = extra.pop('input_size', 4)
    with contextlib.redirect_stdout(io.StringIO()):
        result = EphemeralRNN(input_size, 5, 4, 2, list('abcd'), updater='backprop', batch_size=batch,
                              plasticity=3., ephemeral_fraction=.5, forget_rate=.15,
                              enable_recurrence=recurrent, **extra)
    result.sequence_outer = True
    return result.double()


def data(batch=2, length=10):
    generator = torch.Generator().manual_seed(11)
    return F.one_hot(torch.randint(4, (batch, length), generator=generator), 4).double()


def objective(m, x, cfg):
    return sequence_rollout(m, x, cfg)[0].sum(dim=1).mean()


def test_arms_have_bit_identical_forward_and_fast_state_but_different_gradients():
    m, x = model(), data()
    m.start_sequence_wipe()
    a = sequence_rollout(m, x, config())
    b = sequence_rollout(m, x, {**config(), 'fast_write_grad': 'detached'})
    for left, right in zip([a[0], a[1], *a[2]], [b[0], b[1], *b[2]]):
        assert torch.equal(left, right)
    p = m.linear_layers[0].per_sample_weights
    ga = torch.autograd.grad(a[0].sum(), p)[0]
    gb = torch.autograd.grad(b[0].sum(), p)[0]
    assert (ga - gb).abs().max() > 1e-8
    assert all(not f.requires_grad for f in b[2])
    assert any(f.requires_grad for f in a[2])


def test_future_loss_reaches_write_time_input_only_in_through_arm_without_recurrence():
    m = model(recurrent=False)
    results = []
    for mode in ('through', 'detached'):
        x = data(length=4).requires_grad_()
        losses = sequence_rollout(m, x, {**config(), 'fast_write_grad': mode})[0]
        results.append(torch.autograd.grad(losses[-1].sum(), x)[0][:, 0])
    assert results[0].abs().max() > 1e-8
    assert torch.equal(results[1], torch.zeros_like(results[1]))


@pytest.mark.parametrize('recurrent', [False, True])
def test_through_finite_difference_includes_keys_and_live_softmax(recurrent):
    m, x, cfg = model(recurrent=recurrent), data(length=5), config()
    m.start_sequence_wipe()
    # Check every slow leaf (weights AND shared biases), including state-head
    # credit through time when recurrence is on. No in-place update in rollout.
    params = [p for l in m.trained_layers() for p in (l.per_sample_weights, l.bias)]
    grads = torch.autograd.grad(objective(m, x, cfg), params, allow_unused=True)
    epsilon = 1e-5
    checked = 0
    for p, g in zip(params, grads):
        if g is None:
            continue
        # Largest-magnitude derivative avoids vacuous checks on masked entries.
        idx = g.abs().reshape(-1).argmax().item()
        old = p.reshape(-1)[idx].item()
        with torch.no_grad():
            p.reshape(-1)[idx] = old + epsilon
        plus = objective(m, x, cfg).item()
        with torch.no_grad():
            p.reshape(-1)[idx] = old - epsilon
        minus = objective(m, x, cfg).item()
        with torch.no_grad():
            p.reshape(-1)[idx] = old
        assert g.reshape(-1)[idx].item() == pytest.approx((plus-minus)/(2*epsilon), abs=1e-8, rel=1e-5)
        checked += 1
    assert checked >= 6


def test_full_float64_gradcheck_all_outer_parameters():
    # Functional-call supplies independent leaves for EVERY slow-copy entry and
    # bias, not just the selected finite-difference coordinates above.
    class Rollout(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = model(batch=1)

        def forward(self, x):
            return objective(self.rnn, x, config())

    wrapped, x = Rollout(), data(batch=1, length=4)
    selected = [(n, p) for n, p in wrapped.named_parameters()
                if n.endswith('per_sample_weights') or n.endswith('.bias')]
    names = [n for n, _ in selected]
    params = tuple(p.detach().clone().requires_grad_() for _, p in selected)
    assert torch.autograd.gradcheck(
        lambda *p: torch.func.functional_call(wrapped, dict(zip(names, p)), (x,)),
        params, eps=1e-5, atol=1e-7, rtol=1e-4, fast_mode=False)


def test_detached_gradient_matches_independent_eager_reads_with_frozen_write_history():
    m, x, cfg = model(), data(length=5), {**config(), 'fast_write_grad': 'detached'}
    m.start_sequence_wipe()
    reference = copy.deepcopy(m)
    # Prefix rollouts supply numeric fast history. The independent comparator
    # calls the actual model forward with each fixed history, retaining hidden
    # recurrence but never differentiating through the history calculation.
    histories = [[torch.zeros_like(l.per_sample_weights) for l in m.trained_layers()]]
    for length in range(2, x.shape[1]):
        histories.append([f.detach() for f in sequence_rollout(m, x[:, :length], cfg)[2]])
    layer_names = [name for name, layer in reference.named_modules()
                   if layer in reference.trained_layers()]
    hidden = x.new_zeros(x.shape[0], reference.hidden_size)
    losses = []
    for t, history in enumerate(histories):
        mapping = {}
        for name, layer, fast in zip(layer_names, reference.trained_layers(), history):
            mapping[name + '.per_sample_weights'] = torch.where(
                layer.ephemeral_mask.unsqueeze(0), fast, layer.per_sample_weights)
        output, hidden = torch.func.functional_call(reference, mapping, (x[:, t], hidden))
        losses.append(cfg['criterion'](output, x[:, t+1]))
    live = objective(m, x, cfg)
    independent = torch.stack(losses).sum(1).mean()
    actual_params = [p for l in m.trained_layers() for p in (l.per_sample_weights, l.bias)]
    ref_params = [p for l in reference.trained_layers() for p in (l.per_sample_weights, l.bias)]
    actual = torch.autograd.grad(live, actual_params, allow_unused=True)
    expected = torch.autograd.grad(independent, ref_params, allow_unused=True)
    for a, b in zip(actual, expected):
        if a is None:
            assert b is None
        else:
            torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize('mode', ['through', 'detached'])
def test_outer_step_is_per_copy_batch_sum_shared_bias_mean_no_alpha_squared(mode):
    m, x = model(), data()
    cfg = {**config(), 'fast_write_grad': mode}
    reference = copy.deepcopy(m)
    reference.start_sequence_wipe()
    losses, _, fast = sequence_rollout(reference, x, cfg)
    layers = reference.trained_layers()
    params = [p for l in layers for p in (l.per_sample_weights, l.bias)]
    grads = torch.autograd.grad(losses.sum(dim=1).mean(), params, allow_unused=True)
    state = {'training_instance': 0}
    result = train_batch(None, x, m, cfg, state)
    assert state['training_instance'] == 9
    assert result[3].shape == (9, 2)
    for index, (actual, before, final_fast) in enumerate(zip(m.trained_layers(), layers, fast)):
        g, bg = grads[2*index:2*index+2]
        expected = before.per_sample_weights - cfg['learning_rate'] * (0 if g is None else g)
        expected = torch.where(actual.ephemeral_mask.unsqueeze(0), final_fast, expected)
        torch.testing.assert_close(actual.per_sample_weights, expected, rtol=0, atol=0)
        expected_bias = before.bias - cfg['learning_rate'] * (0 if bg is None else bg / 2)
        torch.testing.assert_close(actual.bias, expected_bias, rtol=1e-14, atol=1e-16)


@pytest.mark.parametrize('mode', ['through', 'detached'])
def test_batch_duplication_does_not_shrink_per_copy_steps(mode):
    one, two = model(batch=1), model(batch=2)
    x = data(batch=1)
    cfg = {**config(), 'fast_write_grad': mode}
    # Initialization draws shape-independent weights/feedback/masks.
    for a, b in zip(one.trained_layers(), two.trained_layers()):
        assert torch.equal(a.per_sample_weights[0], b.per_sample_weights[0])
    train_batch(None, x, one, cfg, {'training_instance': 0})
    train_batch(None, x.expand(2, -1, -1), two, cfg, {'training_instance': 0})
    for a, b in zip(one.trained_layers(), two.trained_layers()):
        torch.testing.assert_close(a.per_sample_weights[0], b.per_sample_weights[0], rtol=1e-14, atol=1e-16)
        torch.testing.assert_close(a.bias, b.bias, rtol=1e-14, atol=1e-16)


@pytest.mark.parametrize('clamp,weight_clamp', [(0, 0), (.002, .1)])
def test_unfused_heldout_matches_functional_rollout_and_freezes_slow(clamp, weight_clamp):
    m, x = model(weight_clamp=weight_clamp), data()
    cfg = {**config(), 'ephemeral_update_clamp': clamp}
    m.start_sequence_wipe()
    reference = copy.deepcopy(m)
    loss, outputs, fast = sequence_rollout(reference, x, cfg)
    frozen = [(l.per_sample_weights.detach().clone(), l.bias.detach().clone()) for l in m.trained_layers()]
    pred, evaluated_loss = evaluate_held_out(m, x, torch.ones(2, 9, dtype=torch.bool), cfg)
    assert torch.equal(pred, outputs.argmax(2))
    torch.testing.assert_close(evaluated_loss, loss, rtol=1e-10, atol=1e-12)
    for l, f, (w, bias) in zip(m.trained_layers(), fast, frozen):
        mask = l.ephemeral_mask.unsqueeze(0).expand_as(w)
        torch.testing.assert_close(l.per_sample_weights[mask], f[mask], rtol=1e-10, atol=1e-12)
        assert torch.equal(l.per_sample_weights[~mask], w[~mask])
        assert torch.equal(l.bias, bias)


def test_padding_writes_zero_and_only_forgets_as_in_eager_evaluator():
    m, x = model(), data()
    x[0, 5:] = 0
    cfg = config()
    reference = copy.deepcopy(m)
    loss, outputs, fast = sequence_rollout(reference, x, cfg)
    pred, evaluated_loss = evaluate_held_out(m, x, torch.ones(2, 9, dtype=torch.bool), cfg)
    assert torch.equal(pred, outputs.argmax(2))
    torch.testing.assert_close(evaluated_loss, loss, rtol=1e-10, atol=1e-12)
    for l, f in zip(m.trained_layers(), fast):
        mask = l.ephemeral_mask.unsqueeze(0).expand_as(f)
        torch.testing.assert_close(l.per_sample_weights[mask], f[mask], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('options,inputs,pe', [
    ({'layer_norm': True, 'output_tanh': True}, 'last_one', None),
    ({'residual_connection': True}, 'last_one', None),
    ({'input_size': 10, 'fast_layers': [1]}, 'last_two', torch.randn(10, 2).double()),
])
def test_supported_forward_options_match_real_eager_model(options, inputs, pe):
    m, x = model(**options), data()
    cfg = {**config(), 'input_mode': inputs, 'pe_matrix': pe}
    expected = sequence_rollout(copy.deepcopy(m), x, cfg)
    actual = evaluate_held_out(m, x, torch.ones(2, 9, dtype=torch.bool), cfg)
    assert torch.equal(actual[0], expected[1].argmax(2))
    torch.testing.assert_close(actual[1], expected[0], rtol=1e-10, atol=1e-12)


def test_invalid_direct_config_and_length_refused_without_update():
    m, x = model(), data()
    saved = copy.deepcopy(m.state_dict())
    with pytest.raises(ValueError, match='grad_norm_clip'):
        train_batch(None, x, m, {**config(), 'grad_norm_clip': 1}, {'training_instance': 0})
    assert all(torch.equal(saved[k], v) for k, v in m.state_dict().items())
    with pytest.raises(ValueError, match='characters'):
        sequence_rollout(m, data(length=11), config())


def test_protocols_restore_state_and_both_arms_evaluate_identically():
    m, x = model().float(), data().float()
    before = copy.deepcopy(m.state_dict())
    cfg = config()
    batches = [('abcdabcdab abcdabcdab'.split(), x)]
    a = evaluate_protocols(m, batches, cfg, 'palindrome')
    b = evaluate_protocols(m, batches, {**cfg, 'fast_write_grad': 'detached'}, 'palindrome')
    assert a == b
    assert all(torch.equal(before[k], v) for k, v in m.state_dict().items())


BASE = ['--model_type', 'ephemeral', '--updater', 'backprop', '--sequence_outer',
        '--backprop_step_parity', '--max_seq_len', '10']


def test_cli_defaults_and_valid_modes():
    assert parse_args([]).fast_write_grad == 'detached'
    assert not parse_args([]).sequence_outer
    for mode in ('through', 'detached'):
        args = parse_args(BASE + ['--fast_write_grad', mode, '--heldout_eval_every', '10'])
        assert args.fast_write_grad == mode


@pytest.mark.parametrize('args', [
    ['--fast_write_grad', 'through'], BASE + ['--updater', 'dfa'],
    BASE + ['--backprop_step_parity', 'false'], BASE + ['--max_seq_len', '11'],
    BASE + ['--grad_norm_clip', '1'], BASE + ['--fused_update'],
    BASE + ['--slow_weight_decay', '.1'], BASE + ['--fast_read_norm', 'ln'],
    BASE + ['--decouple_fast_feedback', '1'], BASE + ['--wipe_every', '2'],
    BASE + ['--label_smoothing', '.1'], BASE + ['--train_wipe_chars', '2'],
])
def test_cli_refuses_incompatible_combinations(args):
    with pytest.raises(SystemExit):
        parse_args(args)


@pytest.mark.parametrize('key,value', [('sequence_outer', False), ('fast_write_grad', 'detached')])
def test_checkpoint_refuses_switching_outer_mechanics_before_weights_load(key, value):
    current = config()
    checkpoint = {'code_version': CHECKPOINT_CODE_VERSION, 'config': {**current, key: value}}
    with pytest.raises(RuntimeError, match='configuration mismatch'):
        load_checkpoint('not-read', model(), current, checkpoint=checkpoint)


def test_legacy_checkpoint_defaults_are_not_reinterpreted_as_outer():
    # Even if the legacy config lacks both new keys, enabling outer is a mismatch.
    current = config()
    saved = {k: v for k, v in current.items() if k not in ('sequence_outer', 'fast_write_grad')}
    with pytest.raises(RuntimeError, match='configuration mismatch'):
        load_checkpoint('not-read', model(), current,
                        checkpoint={'code_version': CHECKPOINT_CODE_VERSION, 'config': saved})


@pytest.mark.parametrize('mode', ['through', 'detached'])
def test_training_cli_checkpoint_exact_resume_and_heldout_cli(tmp_path, mode):
    import heldout
    import train as training
    from preprocess import make_dataloader
    from tests.test_seed_resume import DATASET, in_memory_items
    from tests.test_heldout import fake_heldout_batches

    def loader(dataset, batch_size, drop_last=True, seed=None, max_seq_len=None):
        assert max_seq_len == 10
        return make_dataloader(in_memory_items(), batch_size, drop_last=drop_last,
                               seed=seed, num_workers=0)

    def run(directory, steps, *extra):
        argv = ['train.py', *BASE, '--fast_write_grad', mode, '--dataset', DATASET,
                '--track', 'false', '--seed', '3', '--batch_size', '2',
                '--hidden_size', '5', '--num_layers', '2', '--plasticity', '3',
                '--n_iters', str(steps), '--print_freq', '1', '--checkpoint_save_freq', '1',
                '--checkpoint_dir', str(directory), '--heldout_eval_every', '1', *extra]
        with patch('sys.argv', argv), \
                patch.object(training, 'load_and_preprocess_data', side_effect=loader), \
                patch.object(training, 'load_heldout_batches', side_effect=fake_heldout_batches), \
                contextlib.redirect_stdout(io.StringIO()):
            training.main()
        return torch.load(directory / 'latest_checkpoint.pth', weights_only=False)

    whole, split = tmp_path / 'whole', tmp_path / 'split'
    expected = run(whole, 4)
    first = run(split, 2)
    assert first['config']['sequence_outer'] is True
    assert first['config']['fast_write_grad'] == mode
    resumed = run(split, 4, '--resume')
    for key, tensor in expected['model_state_dict'].items():
        assert torch.equal(tensor, resumed['model_state_dict'][key]), key
    with patch.object(heldout, 'load_heldout_batches', side_effect=fake_heldout_batches), \
            contextlib.redirect_stdout(io.StringIO()) as out:
        heldout.main(['--checkpoint', str(split / 'latest_checkpoint.pth'), '--device', 'cpu'])
    assert 'heldout_strict/recall_acc' in out.getvalue()


@pytest.mark.parametrize('updater', ['dfa', 'backprop', 'bptt'])
def test_explicit_new_defaults_are_bit_identical_to_absent(updater):
    m, x = model(recurrent=False), data(length=5)
    # Rebuild updater correctly (DFA needs no gradient leaves).
    torch.manual_seed(3)
    m = EphemeralRNN(4, 5, 4, 2, list('abcd'), updater=updater, batch_size=2,
                     plasticity=3, enable_recurrence=False)
    other = copy.deepcopy(m)
    cfg = dict(updater=updater, criterion=torch.nn.CrossEntropyLoss(reduction='none'),
               input_mode='last_one', pe_matrix=None, learning_rate=.03, plasticity=3,
               ephemeral_update_clamp=0, grad_norm_clip=0)
    for _ in range(3):
        a = train_batch(None, x.float(), m, cfg, {'training_instance': 0})
        b = train_batch(None, x.float(), other, {**cfg, 'sequence_outer': False,
                        'fast_write_grad': 'detached'}, {'training_instance': 0})
        assert a[1] == b[1]
        assert all(torch.equal(m.state_dict()[k], v) for k, v in other.state_dict().items())
