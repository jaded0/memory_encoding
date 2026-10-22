"""--order2_layers (2026-10-22, video experiment 5): an order-two fast matrix on a 16-number projection.

- Off (the default), the model is the one main gave: the same run, bit for bit (fixture "main/plain").
- The features give the documented kernel: fraction * |x| |y| * k(c) / k(1) with k the second-order
  Taylor polynomial (or its first-order truncation) of the projected cosine, so a perfect match reads
  as large as on the linear fast entries.
- The forward adds G psi(x); the write is the layer's DFA fast error times psi(x), alpha, lr, then
  forgetting; the wipe zeroes G. The unfused step, the eager fused step and the held-out fast-only step
  write G the same way.
- 'replace' never writes the linear fast entries; 'parallel' leaves them as they were until G has been
  read. The projection uses its own generator: every other initial weight is unchanged.
- Unsupported combinations and resume with different settings are refused."""
import contextlib
import io
import math
import unittest

import torch
import torch.nn.functional as F

from ephemeral_model import (EphemeralRNN, order2_feature_size, order2_features, order2_kernel_norm,
                             order2_projection)
from reproducibility import seed_everything
from tests import scratch_flags_reference as ref
from train import build_model, parse_args, train
from utils import CHECKPOINT_CODE_VERSION, load_checkpoint

FIXTURE = ref.__file__.replace("scratch_flags_reference.py", "fixtures/scratch_flags.pt")


def build(**kwargs):
    return ref.build(EphemeralRNN, seed_everything, **kwargs)


def same_state(test, first, second):
    test.assertEqual(first.keys(), second.keys())
    for key in first:
        test.assertTrue(torch.equal(first[key], second[key]), key)


class Order2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_off_matches_main_and_the_default(self):
        fixture = torch.load(FIXTURE, weights_only=False)
        exact = fixture["generated_with"]["torch"] == str(torch.__version__)
        tol = dict(rtol=0, atol=0) if exact else dict(rtol=1e-5, atol=1e-6)
        default = ref.run(build(), train)
        explicit = ref.run(build(order2_layers=None, order2_map='order2', order2_mode='replace'), train)
        self.assertEqual(default["losses"], explicit["losses"])
        same_state(self, default["state"], explicit["state"])
        torch.testing.assert_close(torch.tensor(default["losses"]),
                                   torch.tensor(fixture["main"]["plain"]["losses"]), **tol)
        self.assertFalse(any("order2" in key for key in build().state_dict()))

    def test_kernel_is_the_size_matched_taylor_polynomial(self):
        torch.manual_seed(0)
        proj = torch.randn(16, 40)
        x, y = torch.randn(5, 40), torch.randn(5, 40)
        u, v = x @ proj.t(), y @ proj.t()
        c = F.cosine_similarity(u, v, dim=1)
        for order, size in (('order2', 273), ('order1', 17)):
            for t in (1.0, 4.0):
                px = order2_features(x, proj, order, t, 0.1)
                py = order2_features(y, proj, order, t, 0.1)
                self.assertEqual(px.shape, (5, size))
                self.assertEqual(order2_feature_size(order, 16), size)
                k = 1 + t * c + (t * c) ** 2 / 2 if order == 'order2' else 1 + t * c
                expected = 0.1 * x.norm(dim=1) * y.norm(dim=1) * k / order2_kernel_norm(order, t)
                torch.testing.assert_close((px * py).sum(1), expected, rtol=1e-5, atol=1e-6)
                # a perfect match reads like the linear entries' expected key factor fraction * |x|^2
                torch.testing.assert_close((px * px).sum(1), 0.1 * x.norm(dim=1) ** 2, rtol=1e-5, atol=1e-6)
        self.assertTrue(bool((order2_features(torch.zeros(2, 40), proj, 'order2', 4.0, 0.1) == 0).all()))

    def test_forward_adds_the_order_two_read(self):
        model = build(order2_layers=[1, 2])
        layer = model.linear_layers[1]
        torch.manual_seed(2)
        with torch.no_grad():
            layer.order2_fast.normal_()
            x = torch.randn(ref.BATCH_SIZE, layer.in_features)
            psi = order2_features(x, layer.order2_proj, 'order2', 4.0, layer.order2_fraction)
            expected = (torch.bmm(layer.per_sample_weights, x.unsqueeze(2)).squeeze(2) + layer.bias
                        + torch.einsum('bof,bf->bo', layer.order2_fast, psi))
            torch.testing.assert_close(layer(x), expected, rtol=1e-5, atol=1e-5)
        self.assertEqual(layer.order2_fast.shape, (ref.BATCH_SIZE, layer.out_features, 273))
        self.assertIsNone(model.linear_layers[0].order2_fast)

    def test_write_is_the_fast_error_times_psi_then_forget(self):
        for mode in ('parallel', 'replace'):
            model = build(order2_layers=[2], order2_mode=mode)
            model.enable_fused_update(compile=False)
            layer = model.linear_layers[2]
            torch.manual_seed(3)
            with torch.no_grad():
                layer.order2_fast.normal_()
            before = layer.order2_fast.clone()
            onehot = F.one_hot(torch.randint(0, 4, (ref.BATCH_SIZE,)), 4).float()
            out, _ = model(onehot, model.initHidden(ref.BATCH_SIZE))
            psi = layer.order2_trace.clone()
            error = torch.randn_like(out)
            projected, _ = model.dfa_step_errors(error)
            p = projected[2]
            model.fused_dfa_step(error, ref.LEARNING_RATE, 0)
            alpha = layer.order2_alpha()
            # the reference model has --weight_clamp 1, which the write applies before forgetting
            expected = ((before - ref.LEARNING_RATE * alpha * p.unsqueeze(2) * psi.unsqueeze(1)).clamp(-1, 1)
                        * (1 - layer.forget_rate))
            torch.testing.assert_close(layer.order2_fast, expected, rtol=1e-5, atol=1e-6)

    def test_unfused_fused_and_norm_logging_paths_agree(self):
        for kwargs in ({"order2_layers": [1, 2]}, {"order2_layers": [2], "order2_mode": "replace"},
                       {"order2_layers": [1], "order2_map": "order1"},
                       {"order2_layers": [2], "decouple_fast_feedback": [2]}):
            with self.subTest(**kwargs):
                unfused, fused, logged = build(**kwargs), build(**kwargs), build(**kwargs)
                fused.enable_fused_update(compile=False)
                logged.enable_fused_update(compile=False)
                a, b, c = ref.run(unfused, train), ref.run(fused, train), ref.run(logged, train, log_norms_every=3)
                self.assertEqual(a["losses"], b["losses"])
                self.assertEqual(a["losses"], c["losses"])
                same_state(self, a["state"], b["state"])
                same_state(self, a["state"], c["state"])
                for l1, l2 in zip(unfused.linear_layers, fused.linear_layers):
                    if l1.order2 is not None:
                        self.assertTrue(torch.equal(l1.order2_fast, l2.order2_fast))
                        self.assertGreater(float(l1.order2_fast.abs().sum()), 0)

    def test_on_changes_the_run_and_stays_finite(self):
        default = ref.run(build(), train)
        for kwargs in ({"order2_layers": [1, 2]}, {"order2_layers": [1, 2], "order2_mode": "replace"},
                       {"order2_layers": [1, 2], "order2_map": "order1"}):
            run = ref.run(build(**kwargs), train)
            self.assertNotEqual(run["losses"], default["losses"])
            self.assertTrue(all(math.isfinite(v) for v in run["losses"]))

    def test_parallel_matches_the_baseline_until_g_is_read(self):
        # Batch 0: G starts at zero and is read before it is written, so step 0 of the first sequence
        # gives the baseline's output and writes; later steps read G and differ.
        base, par = build(), build(order2_layers=[1, 2])
        onehot = F.one_hot(torch.randint(0, 4, (ref.BATCH_SIZE,)), 4).float()
        h = base.initHidden(ref.BATCH_SIZE)
        out_b, _ = base(onehot, h)
        out_p, _ = par(onehot, h)
        self.assertTrue(torch.equal(out_b, out_p))
        error = torch.randn_like(out_b)
        base.enable_fused_update(compile=False); par.enable_fused_update(compile=False)
        base.fused_dfa_step(error, 0.1, 0); par.fused_dfa_step(error, 0.1, 0)
        for lb, lp in zip(base.trained_layers(), par.trained_layers()):
            self.assertTrue(torch.equal(lb.per_sample_weights, lp.per_sample_weights))
        self.assertFalse(torch.equal(base(onehot, h)[0], par(onehot, h)[0]))

    def test_replace_never_writes_the_linear_fast_entries(self):
        model = build(order2_layers=[1, 2], order2_mode='replace')
        model.enable_fused_update(compile=False)
        ref.run(model, train, log_norms_every=4)
        for index, layer in enumerate(model.linear_layers):
            fast = layer.per_sample_weights[:, layer.ephemeral_mask]
            if index in (1, 2):
                self.assertTrue(bool((fast == 0).all()), index)
                self.assertGreater(float(layer.order2_fast.abs().sum()), 0)
            else:
                self.assertGreater(float(fast.abs().sum()), 0)

    def test_wipe_zeroes_g_unless_carried(self):
        model = build(order2_layers=[1])
        layer = model.linear_layers[1]
        layer.order2_fast.fill_(1.0)
        model.start_sequence_wipe(wipe_fast=False)
        self.assertTrue(bool((layer.order2_fast == 1).all()))
        model.start_sequence_wipe()
        self.assertTrue(bool((layer.order2_fast == 0).all()))

    def test_heldout_fast_only_step_writes_g_and_is_restored(self):
        from heldout import evaluate_protocols
        model = build(order2_layers=[1, 2], order2_mode='replace')
        texts = ["abcdab", "bcdabc", "cdabcd"]
        onehot = F.one_hot(torch.tensor([[ref.CHARSET.index(ch) for ch in t] for t in texts]), 4).float()
        layer = model.linear_layers[2]
        layer.order2_fast.fill_(0.5)
        config = {"input_mode": "last_one", "pe_matrix": None, "learning_rate": 0.1,
                  "ephemeral_update_clamp": 0, "grad_norm_clip": 0}
        hidden = model.initHidden(3)
        out, _ = model(onehot[:, 0], hidden)
        with torch.no_grad():
            model.fast_only_dfa_step(torch.randn_like(out), 0.1, 0)
        self.assertFalse(bool((layer.order2_fast == 0.5 * (1 - layer.forget_rate)).all()))
        snapshot = layer.order2_fast.clone()
        with contextlib.redirect_stdout(io.StringIO()):
            results = evaluate_protocols(model, [(texts, onehot)], config, "palindrome_dataset",
                                         protocols=("observed", "no_fast"))
        self.assertTrue(torch.equal(layer.order2_fast, snapshot))
        # replace mode: the observed protocol's only fast reads are the order-two ones
        self.assertNotEqual(results["heldout_observed/loss"], results["heldout_no_fast/loss"])

    def test_projection_has_its_own_generator(self):
        torch.manual_seed(5)
        state = torch.get_rng_state()
        off, on = build(), build(order2_layers=[0, 1, 2])
        for a, b in zip(off.trained_layers(), on.trained_layers()):
            self.assertTrue(torch.equal(a.per_sample_weights, b.per_sample_weights))
            self.assertTrue(torch.equal(a.ephemeral_mask, b.ephemeral_mask))
            self.assertTrue(torch.equal(a.feedback_weights, b.feedback_weights))
        self.assertTrue(torch.equal(order2_projection(10, 4, 7, 1), order2_projection(10, 4, 7, 1)))
        self.assertFalse(torch.equal(order2_projection(10, 4, 7, 1), order2_projection(10, 4, 8, 1)))
        self.assertFalse(torch.equal(order2_projection(10, 4, 7, 1), order2_projection(10, 4, 7, 2)))
        self.assertTrue(any("order2_fast" in key for key in on.state_dict()))
        self.assertTrue(any("order2_proj" in key for key in on.state_dict()))

    def test_invalid_combinations_are_refused(self):
        for kwargs in ({"order2_layers": [3]}, {"order2_layers": [1], "order2_map": "order3"},
                       {"order2_layers": [1], "order2_mode": "both"},
                       {"order2_layers": [1], "order2_temperature": 0},
                       {"order2_layers": [1], "updater": "backprop"},
                       {"order2_layers": [1], "slow_update_every": 2},
                       {"order2_layers": [1], "fast_backward_per_forward": 2},
                       {"order2_layers": [1], "fast_backward_per_forward": "1/2"},
                       {"order2_layers": [1], "fast_read_norm": "rms"},
                       {"order2_layers": [2], "fast_layers": [0, 1]},
                       {"order2_layers": [1], "fast_structure": "additive_masked"}):
            with self.subTest(**{k: str(v) for k, v in kwargs.items()}), self.assertRaises(ValueError):
                build(**kwargs)
        base = ['--model_type', 'ephemeral', '--updater', 'dfa', '--order2_layers', '1,2']
        for argv in (['--grad_norm_clip', '1'], ['--updater', 'backprop'], ['--model_type', 'rnn'],
                     ['--slow_update_every', '2'], ['--fast_read_norm', 'rms'],
                     ['--order2_temperature', '-1'], ['--order2_map', 'x'],
                     ['--fast_structure', 'additive_dense']):
            with self.subTest(argv=argv), self.assertRaises(SystemExit), \
                    contextlib.redirect_stderr(io.StringIO()):
                parse_args(base + argv)

    def test_build_model_reads_the_flags_and_resume_refuses_a_change(self):
        args = parse_args(['--model_type', 'ephemeral', '--updater', 'dfa', '--order2_layers', '1,2',
                           '--order2_map', 'order1', '--order2_mode', 'replace', '--order2_temperature', '2',
                           '--order2_dim', '8', '--hidden_size', '8', '--num_layers', '3',
                           '--batch_size', '2', '--seed', '9'])
        config = dict(vars(args), n_hidden=8, n_layers=3, pe_matrix=None)
        with contextlib.redirect_stdout(io.StringIO()):
            model = build_model(config, ref.CHARSET, len(ref.CHARSET))
        layer = model.linear_layers[1]
        self.assertEqual((layer.order2, layer.order2_replace, layer.order2_temperature), ('order1', True, 2.0))
        self.assertEqual(layer.order2_fast.shape[-1], 9)
        self.assertTrue(torch.equal(layer.order2_proj, order2_projection(layer.in_features, 8, 9, 1)))
        checkpoint = {"code_version": CHECKPOINT_CODE_VERSION, "config": dict(config, order2_temperature=4.0),
                      "model_state_dict": model.state_dict(), "iter": 1}
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            load_checkpoint("x.pth", model, config, checkpoint=checkpoint)
        checkpoint["config"] = dict(config, order2_layers='')
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            load_checkpoint("x.pth", model, config, checkpoint=checkpoint)


if __name__ == "__main__":
    unittest.main()
