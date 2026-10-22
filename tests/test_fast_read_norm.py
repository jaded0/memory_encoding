"""--fast_read_norm (2026-10-16, cheap check RN; 'ln' and 'absrun' 2026-10-22): a forward-only
rescaling of the fast read F x.

- Off (the default), the model is the one main gave: the same run, bit for bit (fixture "main/plain"
  of tests/test_fast_layer_flags.py, and an explicit 'off' against the default).
- On, a layer's pre-activation is S x + scale * F x + b with scale from fast_read_scale, checked
  against a direct computation; the scale is never above 1; 'running' leaves a read with a small
  key-weight sum alone and tracks the key mass as update-then-forget, reset by the wipe.
- 'ln' is LayerNorm of F x alone (no gain or bias), F x = 0 reads 0; 'absrun' divides by
  max(1, coef * sum_s (1-f)^(t-s) |x_s . x_t|), checked against an explicit history of the layer's
  inputs through real training sequences (wiped at each sequence start).
- Only layers with fast entries are touched; the writes are the same function of the error.
- Unsupported combinations are refused."""
import contextlib
import io
import unittest
from pathlib import Path

import torch

import torch.nn.functional as F

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
        for mode in ('running', 'rms', 'ln', 'absrun'):
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

    def test_ln_is_layer_norm_of_the_fast_read_alone(self):
        torch.manual_seed(2)
        model = build(fast_read_norm='ln')
        layer = model.linear_layers[1]
        with torch.no_grad():
            layer.per_sample_weights.add_(torch.randn_like(layer.per_sample_weights) * 3)
        x = torch.randn(ref.BATCH_SIZE, layer.in_features, dtype=torch.float64).float()
        w, m = layer.per_sample_weights.double(), layer.ephemeral_mask
        fast = torch.einsum('boi,bi->bo', w * m, x.double())
        slow = torch.einsum('boi,bi->bo', w * ~m, x.double())
        centred = fast - fast.mean(1, keepdim=True)
        normed = centred / (centred.pow(2).mean(1, keepdim=True) + 1e-5).sqrt()  # biased variance
        with torch.no_grad():
            out = layer(x)
        torch.testing.assert_close(out.double(), slow + normed + layer.bias.double(), rtol=1e-5, atol=1e-5)
        # An empty fast store (the first step after the wipe) reads exactly 0: S x + b.
        model.start_sequence_wipe()
        with torch.no_grad():
            expected = torch.bmm(layer.per_sample_weights, x.unsqueeze(2)).squeeze(2) + layer.bias
            torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)

    def test_absrun_divides_by_the_decayed_abs_score_sum(self):
        torch.manual_seed(3)
        lr = 0.5
        model = build(fast_read_norm='absrun', fast_read_norm_lr=lr)
        layer = model.linear_layers[1]
        model.start_sequence_wipe()
        with torch.no_grad():
            layer.per_sample_weights.add_(torch.randn_like(layer.per_sample_weights) * 3)
        w, m = layer.per_sample_weights, layer.ephemeral_mask
        coef = lr * float(m.float().mean()) * layer.fused_plasticity()
        f = layer.forget_rate
        xs = [torch.randn(ref.BATCH_SIZE, layer.in_features) * 2 for _ in range(5)]
        for t, x in enumerate(xs):
            fast = torch.einsum('boi,bi->bo', w * m, x)
            slow = torch.einsum('boi,bi->bo', w * ~m, x)
            mass = sum((1 - f) ** (t - s) * (xs[s] * x).sum(1, keepdim=True).abs() for s in range(t))
            scale = 1 / (coef * mass).clamp_min(1) if t else torch.ones(ref.BATCH_SIZE, 1)
            with torch.no_grad():
                out = layer(x)
            torch.testing.assert_close(out, slow + scale * fast + layer.bias, rtol=1e-5, atol=1e-5)
            if t >= 2:
                self.assertTrue(bool((scale < 1).any()), t)  # the scale binds in this setting
        self.assertEqual(tuple(layer.fast_keys.shape), (ref.BATCH_SIZE, 5, layer.in_features))
        model.start_sequence_wipe()
        self.assertIsNone(layer.fast_keys)
        self.assertIsNone(layer.fast_key_weights)

    def test_absrun_quiets_no_more_than_running(self):
        # |x_s . x| <= sum_j |x_s,j| |x_j|, so on the same history absrun's scale >= running's.
        torch.manual_seed(4)
        scales = {}
        for mode in ('running', 'absrun'):
            model = build(fast_read_norm=mode, fast_read_norm_lr=0.5)
            layer = model.linear_layers[0]
            model.start_sequence_wipe()
            with torch.no_grad():
                layer.per_sample_weights.add_(torch.randn_like(layer.per_sample_weights))
            generator = torch.Generator().manual_seed(5)
            outs = []
            for _ in range(4):
                x = torch.randn(ref.BATCH_SIZE, layer.in_features, generator=generator)
                with torch.no_grad():
                    full = layer(x) - layer.bias
                w, m = layer.per_sample_weights, layer.ephemeral_mask
                fast = torch.einsum('boi,bi->bo', w * m, x)
                slow = torch.einsum('boi,bi->bo', w * ~m, x)
                outs.append(((full - slow) * fast).sum(1) / fast.pow(2).sum(1))
            scales[mode] = torch.stack(outs)
        self.assertTrue(bool((scales['absrun'] >= scales['running'] - 1e-5).all()))

    def test_history_modes_track_real_training_sequences(self):
        # Through train(): every forward of a fast layer reads S x + scale * F x + b, with the
        # normalizer built from exactly this sequence's earlier inputs to that layer (wiped at the
        # sequence start, decayed once per step, as the fast entries are).
        lr = 2.0  # large enough that the normalizer binds within these short sequences
        for mode in ('running', 'absrun'):
            model = build(fast_read_norm=mode, fast_read_norm_lr=lr)
            layers = [*model.linear_layers, model.i2h]
            records = {id(l): [] for l in layers}

            def hook(layer, args, output):
                records[id(layer)].append((args[0].detach().clone(), layer.per_sample_weights.detach().clone(),
                                           layer.bias.detach().clone(), output.detach().clone()))
            handles = [l.register_forward_hook(hook) for l in layers]
            data = ref.batches()[:3]
            bound = False
            config = {"updater": "dfa", "criterion": torch.nn.CrossEntropyLoss(reduction="none"),
                      "input_mode": "last_one", "pe_matrix": None, "learning_rate": lr,
                      "ephemeral_update_clamp": 0, "grad_norm_clip": 0}
            for i, batch in enumerate(data):
                for l in layers:
                    records[id(l)].clear()
                with contextlib.redirect_stdout(io.StringIO()):
                    train(batch, F.one_hot(batch, len(ref.CHARSET)).float(), model, config,
                          {"training_instance": i, "log_norms_now": False})
                for l in layers:
                    coef = lr * float(l.ephemeral_mask.float().mean()) * l.fused_plasticity()
                    history = []
                    self.assertEqual(len(records[id(l)]), ref.SEQUENCE_LENGTH - 1)
                    for x, w, b, out in records[id(l)]:
                        m = l.ephemeral_mask
                        fast = torch.einsum('boi,bi->bo', w * m, x)
                        slow = torch.einsum('boi,bi->bo', w * ~m, x)
                        t = len(history)
                        if mode == 'absrun':
                            mass = sum((1 - l.forget_rate) ** (t - s) * (history[s] * x).sum(1, keepdim=True).abs()
                                       for s in range(t)) if t else torch.zeros(x.shape[0], 1)
                        else:
                            mass = sum((1 - l.forget_rate) ** (t - s) * (history[s].abs() * x.abs()).sum(1, keepdim=True)
                                       for s in range(t)) if t else torch.zeros(x.shape[0], 1)
                        scale = 1 / (coef * mass).clamp_min(1)
                        torch.testing.assert_close(out, slow + scale * fast + b, rtol=1e-5, atol=1e-5)
                        bound = bound or bool((scale < 1).any())
                        history.append(x)
            for h in handles:
                h.remove()
            self.assertTrue(bound, mode)  # the normalizer did bind somewhere, so the check has teeth

    def test_fused_eager_matches_unfused_for_new_modes(self):
        for mode in ('ln', 'absrun'):
            unfused = ref.run(build(fast_read_norm=mode, fast_read_norm_lr=ref.LEARNING_RATE), train)
            model = build(fast_read_norm=mode, fast_read_norm_lr=ref.LEARNING_RATE)
            model.enable_fused_update(compile=False)
            fused = ref.run(model, train)
            self.assertEqual(unfused["losses"], fused["losses"], mode)
            for key, value in unfused["state"].items():
                self.assertTrue(torch.equal(value, fused["state"][key]), (mode, key))

    def test_new_modes_only_on_layers_with_fast_entries(self):
        for mode in ('ln', 'absrun'):
            model = build(fast_read_norm=mode, fast_read_norm_lr=0.1, fast_layers=[0, 1])
            self.assertEqual([l.fast_read_norm for l in model.linear_layers], [mode, mode, 'off'])
            self.assertEqual(model.i2h.fast_read_norm, mode)
            self.assertEqual(model.i2o.fast_read_norm, 'off')
            self.assertEqual(model.linear_layers[2].fast_read_coef, 0.0)

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
        for mode in ('absrun',):
            with self.assertRaises(ValueError):
                build(fast_read_norm=mode)  # no learning rate
            with self.assertRaises(ValueError):
                build(fast_read_norm=mode, fast_read_norm_lr=0.1, fast_backward_per_forward=2)
        with self.assertRaises(ValueError):
            build(fast_read_norm='ln', updater='backprop')
        for argv in (['--fast_read_norm', 'running', '--fast_backward_per_forward', '2'],
                     ['--fast_read_norm', 'absrun', '--fast_backward_per_forward', '1/2'],
                     ['--fast_read_norm', 'absrun', '--wipe_every', '2'],
                     ['--fast_read_norm', 'ln', '--updater', 'bptt'],
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
