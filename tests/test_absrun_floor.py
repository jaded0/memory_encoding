"""--absrun_floor c (2026-10-27, S12 follow-up 3): --fast_read_norm absrun divides the fast read by
max(1, n / c) instead of max(1, n), a weaker divisor for c > 1.

- Off (0, the default) is absrun as it was, bit for bit, and the default model is still main's (the
  fixture "main/plain" of tests/test_fast_layer_flags.py, through test_fast_read_norm's own check).
- On, every read is S x + F x / max(1, coef * mass / c) + b, checked against an explicit history of
  the layer's inputs through real training sequences; c = 1 is absrun.
- A larger c quiets less (the scale is monotone in c), and the flag reaches the model from a config
  (so heldout.py and resumes see it from the checkpoint's config).
- Unsupported combinations are refused."""
import contextlib
import io
import unittest

import torch
import torch.nn.functional as F

from ephemeral_model import EphemeralRNN
from reproducibility import seed_everything
from tests import scratch_flags_reference as ref
from train import build_model, parse_args, train

LR = 2.0  # large enough that absrun's normalizer binds within these short sequences


def build(**kwargs):
    return ref.build(EphemeralRNN, seed_everything, **kwargs)


def config(lr):
    return {"updater": "dfa", "criterion": torch.nn.CrossEntropyLoss(reduction="none"),
            "input_mode": "last_one", "pe_matrix": None, "learning_rate": lr,
            "ephemeral_update_clamp": 0, "grad_norm_clip": 0}


class AbsrunFloorTest(unittest.TestCase):
    def test_off_is_absrun_bit_for_bit(self):
        plain = ref.run(build(fast_read_norm='absrun', fast_read_norm_lr=ref.LEARNING_RATE), train)
        for floor in (0, 0.0, None):
            off = ref.run(build(fast_read_norm='absrun', fast_read_norm_lr=ref.LEARNING_RATE,
                                absrun_floor=floor), train)
            self.assertEqual(plain["losses"], off["losses"], floor)
            for key, value in plain["state"].items():
                self.assertTrue(torch.equal(value, off["state"][key]), (floor, key))
        # c = 1 is the same divisor, max(1, n / 1) = max(1, n).
        one = ref.run(build(fast_read_norm='absrun', fast_read_norm_lr=ref.LEARNING_RATE, absrun_floor=1.0), train)
        self.assertEqual(plain["losses"], one["losses"])

    def test_floor_changes_a_bound_run(self):
        base = ref.run(build(fast_read_norm='absrun', fast_read_norm_lr=LR), train)
        weak = ref.run(build(fast_read_norm='absrun', fast_read_norm_lr=LR, absrun_floor=10.0), train)
        self.assertNotEqual(base["losses"], weak["losses"])
        self.assertTrue(all(torch.isfinite(torch.tensor(weak["losses"]))))

    def test_reads_divide_by_max_one_n_over_c(self):
        # Through train(): every forward of a fast layer reads S x + F x / max(1, coef * mass / c) + b,
        # mass = sum_s (1-f)^(t-s) |x_s . x_t| over this sequence's earlier inputs to the layer.
        for floor in (3.0, 10.0):
            model = build(fast_read_norm='absrun', fast_read_norm_lr=LR, absrun_floor=floor)
            layers = [*model.linear_layers, model.i2h]
            for l in layers:
                self.assertEqual(l.fast_read_absrun_floor, floor if bool(l.ephemeral_mask.any()) else 0.0)
            records = {id(l): [] for l in layers}

            def hook(layer, args, output):
                records[id(layer)].append((args[0].detach().clone(), layer.per_sample_weights.detach().clone(),
                                           layer.bias.detach().clone(), output.detach().clone()))
            handles = [l.register_forward_hook(hook) for l in layers]
            bound = unbound_by_absrun = False
            for i, batch in enumerate(ref.batches()[:3]):
                for l in layers:
                    records[id(l)].clear()
                with contextlib.redirect_stdout(io.StringIO()):
                    train(batch, F.one_hot(batch, len(ref.CHARSET)).float(), model, config(LR),
                          {"training_instance": i, "log_norms_now": False})
                for l in layers:
                    coef = LR * float(l.ephemeral_mask.float().mean()) * l.fused_plasticity()
                    history = []
                    for x, w, b, out in records[id(l)]:
                        m = l.ephemeral_mask
                        fast = torch.einsum('boi,bi->bo', w * m, x)
                        slow = torch.einsum('boi,bi->bo', w * ~m, x)
                        t = len(history)
                        mass = (sum((1 - l.forget_rate) ** (t - s) * (history[s] * x).sum(1, keepdim=True).abs()
                                    for s in range(t)) if t else torch.zeros(x.shape[0], 1))
                        n = coef * mass
                        scale = 1 / (n / floor).clamp_min(1)
                        torch.testing.assert_close(out, slow + scale * fast + b, rtol=1e-5, atol=1e-5)
                        bound = bound or bool((scale < 1).any())
                        # absrun alone would have quieted these rows; the floor left them alone
                        unbound_by_absrun = unbound_by_absrun or bool(((n > 1) & (n <= floor)).any())
                        history.append(x)
            for h in handles:
                h.remove()
            self.assertTrue(bound, floor)  # the floor-weakened divisor still binds somewhere
            self.assertTrue(unbound_by_absrun, floor)  # and the check separates c from c = 1

    def test_larger_floor_quiets_less(self):
        # The same layer, the same weights and inputs: the effective scale grows with c.
        torch.manual_seed(4)
        scales = {}
        for floor in (0.0, 10.0, 100.0):
            model = build(fast_read_norm='absrun', fast_read_norm_lr=LR, absrun_floor=floor)
            layer = model.linear_layers[0]
            model.start_sequence_wipe()
            with torch.no_grad():
                layer.per_sample_weights.add_(torch.randn(layer.per_sample_weights.shape,
                                                          generator=torch.Generator().manual_seed(6)))
            generator = torch.Generator().manual_seed(5)
            outs = []
            for _ in range(6):
                x = torch.randn(ref.BATCH_SIZE, layer.in_features, generator=generator) * 3
                with torch.no_grad():
                    full = layer(x) - layer.bias
                w, m = layer.per_sample_weights, layer.ephemeral_mask
                fast = torch.einsum('boi,bi->bo', w * m, x)
                slow = torch.einsum('boi,bi->bo', w * ~m, x)
                outs.append(((full - slow) * fast).sum(1) / fast.pow(2).sum(1))
            scales[floor] = torch.stack(outs)
        self.assertTrue(bool((scales[10.0] >= scales[0.0] - 1e-6).all()))
        self.assertTrue(bool((scales[100.0] >= scales[10.0] - 1e-6).all()))
        self.assertTrue(bool((scales[10.0] > scales[0.0] + 1e-6).any()))  # it actually differs
        self.assertTrue(bool((scales[100.0] <= 1 + 1e-6).all()))

    def test_build_model_reads_the_flag_from_a_config(self):
        args = parse_args(['--model_type', 'ephemeral', '--updater', 'dfa', '--fast_read_norm', 'absrun',
                           '--absrun_floor', '10', '--learning_rate', '0.01', '--hidden_size', '8',
                           '--num_layers', '2', '--batch_size', '2'])
        self.assertEqual(args.absrun_floor, 10.0)
        cfg = dict(vars(args), n_hidden=8, n_layers=2, pe_matrix=None)
        with contextlib.redirect_stdout(io.StringIO()):
            model = build_model(cfg, ref.CHARSET, len(ref.CHARSET))
        self.assertEqual(model.linear_layers[0].fast_read_absrun_floor, 10.0)
        # An older checkpoint's config has no absrun_floor: off.
        cfg.pop('absrun_floor')
        with contextlib.redirect_stdout(io.StringIO()):
            model = build_model(cfg, ref.CHARSET, len(ref.CHARSET))
        self.assertEqual(model.linear_layers[0].fast_read_absrun_floor, 0.0)
        default = parse_args(['--model_type', 'ephemeral', '--updater', 'dfa'])
        self.assertEqual(default.absrun_floor, 0.0)

    def test_invalid_combinations_are_refused(self):
        with self.assertRaises(ValueError):
            build(fast_read_norm='running', fast_read_norm_lr=0.1, absrun_floor=10.0)
        with self.assertRaises(ValueError):
            build(fast_read_norm='ln', absrun_floor=10.0)
        with self.assertRaises(ValueError):
            build(absrun_floor=10.0)
        with self.assertRaises(ValueError):
            build(fast_read_norm='absrun', fast_read_norm_lr=0.1, absrun_floor=-1.0)
        base = ['--model_type', 'ephemeral', '--updater', 'dfa']
        for argv in (['--absrun_floor', '10'],
                     ['--fast_read_norm', 'ln', '--absrun_floor', '10'],
                     ['--fast_read_norm', 'absrun', '--absrun_floor', '-1']):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                parse_args(base + argv)


if __name__ == "__main__":
    unittest.main()
