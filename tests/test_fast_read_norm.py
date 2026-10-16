"""--fast_read_norm (2026-10-16, cheap check RN): a forward-only rescaling of the fast read F x.

- Off (the default), the model is the one main gave: the same run, bit for bit (fixture "main/plain"
  of tests/test_fast_layer_flags.py, and an explicit 'off' against the default).
- On, a layer's pre-activation is S x + scale * F x + b with scale from fast_read_scale, checked
  against a direct computation; the scale is never above 1; 'running' leaves a read with a small
  key-weight sum alone and tracks the key mass as update-then-forget, reset by the wipe.
- Only layers with fast entries are touched; the writes are the same function of the error.
- Unsupported combinations are refused."""
import contextlib
import io
import unittest
from pathlib import Path

import torch

from ephemeral_model import EphemeralRNN, fast_read_scale
from reproducibility import seed_everything
from tests import scratch_flags_reference as ref
from train import build_model, parse_args, train

FIXTURE = Path(__file__).parent / "fixtures" / "scratch_flags.pt"


def build(**kwargs):
    return ref.build(EphemeralRNN, seed_everything, **kwargs)


class FastReadNormTest(unittest.TestCase):
    def test_off_matches_main_and_the_default(self):
        fixture = torch.load(FIXTURE, weights_only=False)
        expected = fixture["main"]["plain"]
        exact = fixture["generated_with"]["torch"] == str(torch.__version__)
        tol = dict(rtol=0, atol=0) if exact else dict(rtol=1e-5, atol=1e-6)
        explicit = ref.run(build(fast_read_norm='off', fast_read_norm_lr=ref.LEARNING_RATE), train)
        default = ref.run(build(), train)
        self.assertEqual(explicit["losses"], default["losses"])
        for key, value in default["state"].items():
            self.assertTrue(torch.equal(value, explicit["state"][key]), key)
        torch.testing.assert_close(torch.tensor(default["losses"]), torch.tensor(expected["losses"]), **tol)

    def test_on_changes_the_run_and_stays_finite(self):
        default = ref.run(build(), train)
        for mode in ('running', 'rms'):
            run = ref.run(build(fast_read_norm=mode, fast_read_norm_lr=ref.LEARNING_RATE), train)
            self.assertNotEqual(run["losses"], default["losses"], mode)
            self.assertTrue(all(torch.isfinite(torch.tensor(run["losses"]))), mode)

    def test_forward_is_slow_plus_scaled_fast(self):
        torch.manual_seed(0)
        for mode in ('running', 'rms'):
            model = build(fast_read_norm=mode, fast_read_norm_lr=0.5)
            layer = model.linear_layers[1]
            with torch.no_grad():
                layer.per_sample_weights.add_(torch.randn_like(layer.per_sample_weights) * 3)
            mass = torch.rand(ref.BATCH_SIZE, layer.in_features) * 4
            layer.fast_key_mass = mass.clone()
            x = torch.randn(ref.BATCH_SIZE, layer.in_features)
            w, m = layer.per_sample_weights, layer.ephemeral_mask
            fast = torch.einsum('boi,bi->bo', w * m, x)
            slow = torch.einsum('boi,bi->bo', w * ~m, x)
            if mode == 'running':
                alpha = layer.fused_plasticity()
                n = 0.5 * float(m.float().mean()) * alpha * (mass * x.abs()).sum(1, keepdim=True)
                scale = 1 / n.clamp_min(1)
            else:
                scale = (slow.pow(2).mean(1, keepdim=True).sqrt()
                         / fast.pow(2).mean(1, keepdim=True).sqrt()).clamp(max=1)
            with torch.no_grad():
                out = layer(x)
            torch.testing.assert_close(out, slow + scale * fast + layer.bias, rtol=1e-5, atol=1e-5)
            self.assertTrue(bool((scale <= 1).all()))
            if mode == 'running':
                torch.testing.assert_close(layer.fast_key_mass, (1 - layer.forget_rate) * (mass + x.abs()))

    def test_running_leaves_a_small_read_alone_and_resets_on_wipe(self):
        model = build(fast_read_norm='running', fast_read_norm_lr=1e-6)
        layer = model.linear_layers[0]
        torch.manual_seed(1)
        with torch.no_grad():
            layer.per_sample_weights.add_(torch.randn_like(layer.per_sample_weights))
        x = torch.randn(ref.BATCH_SIZE, layer.in_features)
        layer.fast_key_mass = torch.ones_like(x)
        with torch.no_grad():
            expected = torch.bmm(layer.per_sample_weights, x.unsqueeze(2)).squeeze(2) + layer.bias
            torch.testing.assert_close(layer(x), expected, rtol=1e-6, atol=1e-6)
        model.start_sequence_wipe()
        self.assertTrue(bool((layer.fast_key_mass == 0).all()))

    def test_only_layers_with_fast_entries_and_rms_never_amplifies(self):
        model = build(fast_read_norm='rms', fast_layers=[0, 1])
        self.assertEqual([l.fast_read_norm for l in model.linear_layers], ['rms', 'rms', 'off'])
        self.assertEqual(model.i2h.fast_read_norm, 'rms')
        self.assertEqual(model.i2o.fast_read_norm, 'off')
        fast = torch.tensor([[1e-3, 0.0]]); slow = torch.tensor([[5.0, 5.0]])
        self.assertEqual(float(fast_read_scale('rms', fast, slow)), 1.0)
        self.assertEqual(float(fast_read_scale('rms', torch.zeros(1, 2), slow)), 1.0)

    def test_invalid_combinations_are_refused(self):
        with self.assertRaises(ValueError):
            build(fast_read_norm='bogus')
        with self.assertRaises(ValueError):
            build(fast_read_norm='running')  # no learning rate
        with self.assertRaises(ValueError):
            build(fast_read_norm='running', fast_read_norm_lr=0.1, fast_backward_per_forward=2)
        with self.assertRaises(ValueError):
            build(fast_read_norm='rms', updater='backprop')
        base = ['--model_type', 'ephemeral', '--updater', 'dfa']
        for argv in (['--fast_read_norm', 'running', '--fast_backward_per_forward', '2'],
                     ['--fast_read_norm', 'rms', '--updater', 'backprop'],
                     ['--fast_read_norm', 'rms', '--model_type', 'rnn']):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                parse_args(base + argv)

    def test_build_model_reads_the_flag_from_a_config(self):
        args = parse_args(['--model_type', 'ephemeral', '--updater', 'dfa', '--fast_read_norm', 'running',
                           '--learning_rate', '0.01', '--hidden_size', '8', '--num_layers', '2',
                           '--batch_size', '2'])
        config = dict(vars(args), n_hidden=8, n_layers=2, pe_matrix=None)
        with contextlib.redirect_stdout(io.StringIO()):
            model = build_model(config, ref.CHARSET, len(ref.CHARSET))
        self.assertEqual(model.fast_read_norm, 'running')
        frac = float(model.linear_layers[0].ephemeral_mask.float().mean())
        self.assertAlmostEqual(model.linear_layers[0].fast_read_coef, 0.01 * frac)


if __name__ == "__main__":
    unittest.main()
