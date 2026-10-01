"""--fast_structure {exclusive, additive_masked, additive_dense}: the additive layers (W = S + F)
against the exclusive layer and against the factored read of the note "ephemeral weights low-rank
fast weights and fast weight programmers" (sections 4, 5, 8). The exclusive path itself is guarded
by the golden traces (tests/test_characterization.py)."""
import contextlib
import copy
import io
import tempfile
import unittest

import torch

import heldout
from ephemeral_model import EphemeralLinear, EphemeralRNN, dfa_projected_error
from reproducibility import seed_everything
from tests.test_heldout import CHARSET, TEXTS, config, episodes
from tests.test_seed_resume import DATASET, latest, run_main
from utils import load_checkpoint

IN, OUT, BATCH = 7, 5, 3


def layer(structure, alpha=4.0, forget=0.05, fraction=0.3, **options):
    seed_everything(3, deterministic=True)
    with contextlib.redirect_stdout(io.StringIO()):
        return EphemeralLinear(IN, OUT, CHARSET, plasticity=alpha, batch_size=BATCH, forget_rate=forget,
                               ephemeral_fraction=fraction, fast_structure=structure, **options)


def step(lyr, x, error, lr):
    """The unfused train.py step for one layer: forward, populate, update, forget."""
    out = lyr(x)
    lyr.populate_dfa_gradients(error)
    lyr.apply_update(lr, 0, {})
    lyr.apply_forget_step()
    return out


def inputs(steps, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(BATCH, IN, generator=g), torch.randn(BATCH, len(CHARSET), generator=g)) for _ in range(steps)]


class MaskedAdditiveReducesToExclusiveTest(unittest.TestCase):
    """Set-up: the exclusive layer A and the additive_masked layer B are built from the same seed,
    so they have the same mask, initial slow weights and feedback matrix. B's slow weight under
    the mask is then zeroed (the exclusive model's 'absent' slow weight). Fast and off-mask slow
    entries must then follow the same update; B's slow weight under the mask would start to
    train after the first write (the exclusive layer has none), so for a multi-step comparison of the
    outputs the test zeroes it again after every step, which is exactly 'the slow part removed
    under the mask'."""

    def setUp(self):
        self.a, self.b = layer('exclusive'), layer('additive_masked')
        self.mask = self.a.ephemeral_mask.data
        self.assertTrue(torch.equal(self.mask, self.b.ephemeral_mask.data))
        self.assertTrue(torch.equal(self.a.feedback_weights, self.b.feedback_weights))
        self.b.per_sample_weights.data.masked_fill_(self.mask.unsqueeze(0), 0)

    def test_outputs_and_updates_match_over_steps_when_slow_under_mask_is_removed(self):
        for x, e in inputs(6):
            out_a, out_b = step(self.a, x, e, 0.3), step(self.b, x, e, 0.3)
            torch.testing.assert_close(out_b, out_a, rtol=0, atol=0)
            fast_a = self.a.per_sample_weights.data[:, self.mask]
            fast_b = self.b.fast_state.data[:, self.mask]
            torch.testing.assert_close(fast_b, fast_a, rtol=0, atol=0)
            torch.testing.assert_close(self.b.per_sample_weights.data[:, ~self.mask],
                                       self.a.per_sample_weights.data[:, ~self.mask], rtol=0, atol=0)
            torch.testing.assert_close(self.b.bias.data, self.a.bias.data, rtol=0, atol=0)
            self.assertEqual(self.b.fast_state.data[:, ~self.mask].abs().sum().item(), 0.0)  # F is 0 off the mask
            self.b.per_sample_weights.data.masked_fill_(self.mask.unsqueeze(0), 0)  # remove S under the mask

    def test_without_the_removal_the_fast_entries_still_match_and_the_slow_under_mask_trains(self):
        x, e = inputs(1)[0]
        step(self.a, x, e, 0.3), step(self.b, x, e, 0.3)
        torch.testing.assert_close(self.b.fast_state.data[:, self.mask],
                                   self.a.per_sample_weights.data[:, self.mask], rtol=0, atol=0)
        # The only difference: the slow weight under the mask took a plain step (alpha 1).
        self.assertGreater(self.b.per_sample_weights.data[:, self.mask].abs().sum().item(), 0)

    def test_the_exclusive_layer_is_untouched(self):
        self.assertFalse(self.a.is_additive)
        self.assertNotIn('fast_state', dict(self.a.named_parameters()))
        self.assertTrue(torch.equal(self.a.per_sample_weights.data[:, self.mask],
                                    torch.zeros_like(self.a.per_sample_weights.data[:, self.mask])))


class DenseAdditiveReadIsTheFactoredReadTest(unittest.TestCase):
    """Note section 5: F^(t) q = sum_s w_s p_s, w_s = -lr alpha (1-f)^(t-s+1) (x_s . q), per batch
    row, with the exponent t - s + 1 because the step updates and then forgets. The slow part S
    takes its own plain steps, so the fast read is (the layer's output) - S q - bias."""

    def check(self, steps, forget):
        old = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)  # forget_keep builds its keep factor in the default dtype
        try:
            lyr = layer('additive_dense', alpha=3.0, forget=forget)
            lr, alpha = 0.2, 3.0
            keys, values = [], []
            for x, e in inputs(steps, seed=1):
                step(lyr, x.double(), e.double(), lr)
                keys.append(x.double())
                values.append(dfa_projected_error(e.double(), lyr.feedback_weights, False))
            q = torch.randn(BATCH, IN, dtype=torch.float64)
            slow = lyr.per_sample_weights.data
            read = torch.bmm(slow + lyr.fast_state.data, q.unsqueeze(2)).squeeze(2)
            fast_read = read - torch.bmm(slow, q.unsqueeze(2)).squeeze(2)
            t = steps
            factored = torch.zeros(BATCH, OUT, dtype=torch.float64)
            for s, (k, v) in enumerate(zip(keys, values), start=1):
                w = -lr * alpha * (1 - forget) ** (t - s + 1) * (k * q).sum(1)
                factored += w.unsqueeze(1) * v
            torch.testing.assert_close(fast_read, factored, rtol=1e-10, atol=1e-12)
            # The wrong exponent (t - s) would be off by about a factor (1 - f) on the newest term.
            if forget > 0:
                wrong = sum((-lr * alpha * (1 - forget) ** (t - s) * (k * q).sum(1)).unsqueeze(1) * v
                            for s, (k, v) in enumerate(zip(keys, values), start=1))
                self.assertGreater((fast_read - wrong).abs().max().item(), 1e-6)
        finally:
            torch.set_default_dtype(old)

    def test_read_equals_factored_sum(self):
        for steps in (1, 3, 10):
            for forget in (0.0, 0.05, 0.3):
                with self.subTest(steps=steps, forget=forget):
                    self.check(steps, forget)

    def test_every_connection_is_fast(self):
        lyr = layer('additive_dense')
        x, e = inputs(1)[0]
        step(lyr, x, e, 0.3)
        self.assertTrue((lyr.fast_state.data != 0).all())
        masked = layer('additive_masked')
        step(masked, x, e, 0.3)
        self.assertTrue(((masked.fast_state.data != 0) == masked.ephemeral_mask.data.unsqueeze(0).expand_as(
            masked.fast_state.data)).all())


class WipeTest(unittest.TestCase):
    def test_wipe_zeroes_F_and_averages_all_of_S(self):
        for structure in ('additive_masked', 'additive_dense'):
            with self.subTest(structure):
                lyr = layer(structure)
                for x, e in inputs(3):
                    step(lyr, x, e, 0.3)
                self.assertGreater(lyr.fast_state.data.abs().sum().item(), 0)
                s_mean = lyr.per_sample_weights.data.mean(0, keepdim=True)
                self.assertFalse(torch.equal(lyr.per_sample_weights.data[0], lyr.per_sample_weights.data[1]))
                fast_before = lyr.fast_state.data.clone()
                lyr.start_sequence_wipe(wipe_fast=False)
                torch.testing.assert_close(lyr.fast_state.data, fast_before, rtol=0, atol=0)  # kept per row
                torch.testing.assert_close(lyr.per_sample_weights.data, s_mean.expand_as(lyr.per_sample_weights),
                                           rtol=0, atol=0)
                lyr.start_sequence_wipe()
                self.assertEqual(lyr.fast_state.data.abs().sum().item(), 0.0)


def build_rnn(structure, **options):
    seed_everything(7, deterministic=True)
    settings = dict(updater="dfa", batch_size=2, plasticity=50.0, forget_rate=0.2, ephemeral_fraction=0.5,
                    enable_recurrence=True, slow_weight_decay=0.05, fast_structure=structure)
    settings.update(options)
    with contextlib.redirect_stdout(io.StringIO()):
        return EphemeralRNN(10, 5, 4, 2, CHARSET, **settings)


def hconfig(**options):
    return config(ephemeral_update_clamp=0, **options)


class RnnIntegrationTest(unittest.TestCase):
    def test_i2o_is_always_exclusive_and_the_others_follow_the_flag(self):
        model = build_rnn('additive_dense')
        self.assertEqual([l.fast_structure for l in model.trained_layers()], ['additive_dense'] * 3 + ['exclusive'])
        self.assertNotIn('fast_state', dict(model.i2o.named_parameters()))

    def test_fused_eager_step_equals_unfused_step(self):
        import train as train_module
        for structure in ('additive_masked', 'additive_dense'):
            with self.subTest(structure):
                unfused, fused = build_rnn(structure), build_rnn(structure)
                fused.enable_fused_update(compile=False)
                onehot = episodes(TEXTS)
                for model in (unfused, fused):
                    with contextlib.redirect_stdout(io.StringIO()):
                        train_module.train_batch(None, onehot, model, hconfig(),
                                                 {"training_instance": 0, "log_norms_now": False})
                for a, b in zip(unfused.state_dict().values(), fused.state_dict().values()):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_unsupported_combinations_are_refused_with_a_clear_error(self):
        with self.assertRaisesRegex(ValueError, "slow_update_every"):
            build_rnn('additive_masked', slow_update_every='sequence')
        with self.assertRaisesRegex(ValueError, "DFA updater"):
            build_rnn('additive_dense', updater='backprop')
        with self.assertRaisesRegex(ValueError, "fast_weight_clamp"):
            build_rnn('additive_masked', fast_weight_clamp=0.1)
        with self.assertRaisesRegex(ValueError, "fast_structure takes one of"):
            build_rnn('bogus')
        model = build_rnn('additive_masked')
        with self.assertRaisesRegex(ValueError, "ephemeral_update_clamp"):
            model.fast_only_dfa_step(torch.zeros(2, 4), 0.1, 0.02)


class HeldOutTest(unittest.TestCase):
    """heldout.py supports the additive structures: its fast-only step writes F and nothing else."""

    def run_protocol(self, structure, mask_value):
        model = build_rnn(structure)
        onehot = episodes(TEXTS)
        # train one batch first so S differs between rows and across the model
        import train as train_module
        with contextlib.redirect_stdout(io.StringIO()):
            train_module.train_batch(None, onehot, model, hconfig(), {"training_instance": 0, "log_norms_now": False})
        before = copy.deepcopy(model.state_dict())
        update_mask = torch.full((2, 6), mask_value, dtype=torch.bool)
        heldout.evaluate_held_out(model, onehot, update_mask, hconfig())
        return model, before

    def test_fast_only_step_leaves_slow_weights_and_biases_alone(self):
        for structure in ('additive_masked', 'additive_dense'):
            with self.subTest(structure):
                model, before = self.run_protocol(structure, True)
                for layer_, name in zip(model.trained_layers(), ('linear_layers.0', 'linear_layers.1', 'i2h', 'i2o')):
                    s_mean = before[f'{name}.per_sample_weights'].mean(0, keepdim=True)
                    torch.testing.assert_close(layer_.per_sample_weights.data,
                                               s_mean.expand_as(layer_.per_sample_weights), rtol=0, atol=0)
                    torch.testing.assert_close(layer_.bias.data, before[f'{name}.bias'], rtol=0, atol=0)
                    if layer_.is_additive:
                        self.assertGreater(layer_.fast_state.data.abs().sum().item(), 0)

    def test_no_fast_protocol_keeps_F_at_zero(self):
        model = build_rnn('additive_dense')
        onehot = episodes(TEXTS)
        heldout.evaluate_held_out(model, onehot, None, hconfig())
        for layer_ in model.trained_layers():
            if layer_.is_additive:
                self.assertEqual(layer_.fast_state.data.abs().sum().item(), 0.0)

    def test_evaluate_protocols_restores_the_state(self):
        model = build_rnn('additive_masked')
        before = copy.deepcopy(model.state_dict())
        results = heldout.evaluate_protocols(model, [(TEXTS, episodes(TEXTS))], hconfig(), DATASET)
        self.assertIn("heldout_no_fast/recall_acc", results)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)


class CheckpointTest(unittest.TestCase):
    def test_state_round_trips_through_the_real_checkpoint_path(self):
        with tempfile.TemporaryDirectory() as directory:
            run_main("--fast_structure", "additive_masked", "--ephemeral_fraction", "0.5",
                     "--plasticity", "5", checkpoint_dir=directory)
            checkpoint = latest(directory)
            keys = [k for k in checkpoint["model_state_dict"] if k.endswith("fast_state")]
            self.assertEqual(len(keys), 2)  # 1 trunk layer and i2h; i2o has none
            self.assertTrue(any(checkpoint["model_state_dict"][k].abs().sum() > 0 for k in keys))
            self.assertEqual(checkpoint["config"]["fast_structure"], "additive_masked")
            # Resume refuses a different structure ...
            with self.assertRaises(RuntimeError):
                run_main("--resume", "--n_iters", "5", "--fast_structure", "exclusive", checkpoint_dir=directory)
            # ... and continues with the same one.
            run_main("--resume", "--n_iters", "5", "--fast_structure", "additive_masked",
                     "--ephemeral_fraction", "0.5", "--plasticity", "5", checkpoint_dir=directory)
            self.assertEqual(latest(directory)["iter"], 6)  # the next iteration

    def test_load_checkpoint_restores_F(self):
        source = build_rnn('additive_dense')
        import train as train_module
        with contextlib.redirect_stdout(io.StringIO()):
            train_module.train_batch(None, episodes(TEXTS), source, hconfig(),
                                     {"training_instance": 0, "log_norms_now": False})
        cfg = {"n_hidden": 5, "n_layers": 2, "updater": "dfa", "charset_size": 4, "model_type": "ephemeral",
               "forget_rate": 0.2, "fast_structure": "additive_dense"}
        checkpoint = {"config": cfg, "model_state_dict": source.state_dict(), "code_version": 12}
        target = build_rnn('additive_dense', plasticity=1.0)
        with contextlib.redirect_stdout(io.StringIO()):
            load_checkpoint("<memory>", target, cfg, checkpoint=checkpoint)
        for a, b in zip(source.trained_layers(), target.trained_layers()):
            if a.is_additive:
                torch.testing.assert_close(a.fast_state.data, b.fast_state.data, rtol=0, atol=0)
                self.assertGreater(a.fast_state.data.abs().sum().item(), 0)
        # A model of the other structure cannot load it, and a config mismatch is refused first.
        with self.assertRaises(RuntimeError), contextlib.redirect_stdout(io.StringIO()):
            load_checkpoint("<memory>", build_rnn('exclusive'), {**cfg, "fast_structure": "exclusive"},
                            checkpoint=checkpoint)


if __name__ == "__main__":
    unittest.main()
