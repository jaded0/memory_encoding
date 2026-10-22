"""--fast_structure {exclusive, additive_masked, additive_dense}: the additive layers (W = S + F)
against the exclusive layer, against the factored read of the vault note "ephemeral weights
low-rank fast weights and fast weight programmers" (sections 4, 5, 8), and through every path
they support (unfused and fused steps, --grad_norm_clip, the clamps, --wipe_every, --fast_layers,
--plasticity_schedule, held-out evaluation, the loop tracer, checkpoints). The exclusive path is
guarded by the golden traces (tests/test_characterization.py) and by the bit-identity checks below.

Ported 2026-10-22 from the 2026-09-30 pilot branch (worktree-agent-a42eab09d5da49c3c), whose
tests 1-5 are kept; the clamp semantics changed (F now takes --fast_weight_clamp and
--ephemeral_update_clamp instead of refusing them), so those tests are new."""
import contextlib
import copy
import io
import tempfile
import unittest

import torch

import heldout
import train as train_module
from ephemeral_model import (EphemeralLinear, EphemeralRNN, dfa_additive_layer_step, dfa_layer_step,
                             dfa_projected_error)
from loop_trace import LoopTracer
from reproducibility import seed_everything
from tests.test_heldout import CHARSET, TEXTS, config, episodes, fake_heldout_batches
from tests.test_seed_resume import DATASET, latest, run_main
from utils import load_checkpoint

IN, OUT, BATCH = 7, 5, 3
ADDITIVE = ('additive_masked', 'additive_dense')


def layer(structure, alpha=4.0, forget=0.05, fraction=0.3, **options):
    seed_everything(3, deterministic=True)
    with contextlib.redirect_stdout(io.StringIO()):
        return EphemeralLinear(IN, OUT, CHARSET, plasticity=alpha, batch_size=BATCH, forget_rate=forget,
                               ephemeral_fraction=fraction, fast_structure=structure, **options)


def step(lyr, x, error, lr, update_clamp=0):
    """The unfused train.py step for one layer: forward, populate, update, forget."""
    out = lyr(x)
    lyr.populate_dfa_gradients(error)
    lyr.apply_update(lr, update_clamp, {})
    lyr.apply_forget_step()
    return out


def inputs(steps, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(BATCH, IN, generator=g), torch.randn(BATCH, len(CHARSET), generator=g))
            for _ in range(steps)]


def assert_same(test, a, b):
    torch.testing.assert_close(a, b, rtol=0, atol=0)


class MaskedAdditiveReducesToExclusiveTest(unittest.TestCase):
    """The exclusive layer A and the additive_masked layer B are built from the same seed, so they
    have the same mask, initial slow weights and feedback matrix. B's slow weight under the mask is
    then zeroed (the exclusive model's absent slow weight). The fast and off-mask slow entries must
    then follow the same update. B's slow weight under the mask would start to train after the
    first write (A has none), so for a multi-step comparison of the outputs it is zeroed again after
    every step, which is exactly "the slow part removed under the mask"."""

    def check(self, a_options, b_options, update_clamp=0):
        a, b = layer('exclusive', **a_options), layer('additive_masked', **b_options)
        mask = a.ephemeral_mask.data
        self.assertTrue(torch.equal(mask, b.ephemeral_mask.data))
        self.assertTrue(torch.equal(a.feedback_weights, b.feedback_weights))
        b.per_sample_weights.data.masked_fill_(mask.unsqueeze(0), 0)
        for x, e in inputs(6):
            out_a = step(a, x, 3 * e, 0.3, update_clamp)
            out_b = step(b, x, 3 * e, 0.3, update_clamp)
            # The additive read is S x + F x (two products), so the output agrees to rounding;
            # the weights, computed from the same x and error, agree exactly.
            torch.testing.assert_close(out_b, out_a, rtol=1e-6, atol=1e-6)
            assert_same(self, b.fast_state.data[:, mask], a.per_sample_weights.data[:, mask])
            assert_same(self, b.per_sample_weights.data[:, ~mask], a.per_sample_weights.data[:, ~mask])
            assert_same(self, b.bias.data, a.bias.data)
            self.assertEqual(b.fast_state.data[:, ~mask].abs().sum().item(), 0.0)  # F is 0 off the mask
            b.per_sample_weights.data.masked_fill_(mask.unsqueeze(0), 0)  # remove S under the mask

    def test_outputs_and_updates_match_over_steps_when_slow_under_mask_is_removed(self):
        self.check({}, {})

    def test_with_slow_decay(self):
        self.check({'slow_weight_decay': 0.1}, {'slow_weight_decay': 0.1})

    def test_with_update_clamp(self):
        self.check({}, {}, update_clamp=0.05)

    def test_exclusive_weight_clamp_on_fast_entries_is_the_additive_fast_clamp(self):
        # Exclusive --weight_clamp c bounds every entry, the fast ones too; the additive layer
        # bounds S with --weight_clamp and F with --fast_weight_clamp, so c on both reproduces it.
        self.check({'weight_clamp': 0.05}, {'weight_clamp': 0.05, 'fast_weight_clamp': 0.05})
        self.check({'weight_clamp': 0.05, 'fast_weight_clamp': 0.02},
                   {'weight_clamp': 0.05, 'fast_weight_clamp': 0.02})

    def test_without_the_removal_the_fast_entries_still_match_and_the_slow_under_mask_trains(self):
        a, b = layer('exclusive'), layer('additive_masked')
        mask = a.ephemeral_mask.data
        b.per_sample_weights.data.masked_fill_(mask.unsqueeze(0), 0)
        x, e = inputs(1)[0]
        step(a, x, e, 0.3), step(b, x, e, 0.3)
        assert_same(self, b.fast_state.data[:, mask], a.per_sample_weights.data[:, mask])
        # The only difference: the slow weight under the mask took a plain step (alpha 1).
        self.assertGreater(b.per_sample_weights.data[:, mask].abs().sum().item(), 0)

    def test_the_exclusive_layer_is_untouched(self):
        a = layer('exclusive')
        default = layer('exclusive')
        seed_everything(3, deterministic=True)
        with contextlib.redirect_stdout(io.StringIO()):
            omitted = EphemeralLinear(IN, OUT, CHARSET, plasticity=4.0, batch_size=BATCH, forget_rate=0.05,
                                      ephemeral_fraction=0.3)
        self.assertFalse(a.is_additive)
        self.assertNotIn('fast_state', dict(a.named_parameters()))
        self.assertEqual(list(a.state_dict()), list(omitted.state_dict()))
        for key, value in omitted.state_dict().items():
            assert_same(self, default.state_dict()[key], value)
        mask = a.ephemeral_mask.data
        self.assertEqual(a.per_sample_weights.data[:, mask].abs().sum().item(), 0.0)


class AdditiveClampTest(unittest.TestCase):
    def test_weight_clamp_acts_on_S_and_fast_clamp_on_F(self):
        for structure in ADDITIVE:
            with self.subTest(structure):
                lyr = layer(structure, alpha=50.0, weight_clamp=0.2, fast_weight_clamp=0.01)
                for x, e in inputs(4):
                    step(lyr, x, 5 * e, 0.3)
                fast, slow = lyr.fast_state.data, lyr.per_sample_weights.data
                keep = 1 - 0.05  # forgetting follows the clamp
                self.assertLessEqual(fast.abs().max().item(), 0.01 * keep + 1e-7)
                self.assertGreater(fast.abs().max().item(), 0.01 * keep - 1e-6)  # it binds
                self.assertLessEqual(slow.abs().max().item(), 0.2 + 1e-7)
                self.assertGreater(slow.abs().max().item(), 0.2 - 1e-6)

    def test_update_clamp_bounds_each_fast_write(self):
        lyr = layer('additive_dense', alpha=50.0, forget=0.0)
        x, e = inputs(1)[0]
        step(lyr, x, 5 * e, 0.3, update_clamp=0.01)
        self.assertAlmostEqual(lyr.fast_state.data.abs().max().item(), 0.3 * 0.01, places=7)


class DenseAdditiveReadIsTheFactoredReadTest(unittest.TestCase):
    """Note section 5: F^(t) q = sum_s w_s p_s, w_s = -lr alpha (1-f)^(t-s+1) (x_s . q), per batch
    row, with the exponent t - s + 1 because the step updates and then forgets. The slow part S
    takes its own plain steps, so the fast read is (the layer's read) - S q."""

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
            fast_read = lyr(q) - torch.bmm(lyr.per_sample_weights.data, q.unsqueeze(2)).squeeze(2) - lyr.bias
            t = steps
            factored = torch.zeros(BATCH, OUT, dtype=torch.float64)
            for s, (k, v) in enumerate(zip(keys, values), start=1):
                w = -lr * alpha * (1 - forget) ** (t - s + 1) * (k * q).sum(1)
                factored += w.unsqueeze(1) * v
            torch.testing.assert_close(fast_read, factored, rtol=1e-10, atol=1e-12)
            if forget > 0:  # the wrong exponent t - s is off by about a factor (1 - f)
                wrong = sum((-lr * alpha * (1 - forget) ** (t - s) * (k * q).sum(1)).unsqueeze(1) * v
                            for s, (k, v) in enumerate(zip(keys, values), start=1))
                self.assertGreater((fast_read - wrong).abs().max().item(), 1e-6)
        finally:
            torch.set_default_dtype(old)

    def test_read_equals_factored_sum(self):
        for steps in (1, 3, 10, 30):
            for forget in (0.0, 0.05, 0.3):
                with self.subTest(steps=steps, forget=forget):
                    self.check(steps, forget)

    def test_support(self):
        x, e = inputs(1)[0]
        dense = layer('additive_dense')
        step(dense, x, e, 0.3)
        self.assertTrue((dense.fast_state.data != 0).all())
        masked = layer('additive_masked')
        step(masked, x, e, 0.3)
        self.assertTrue(torch.equal(masked.fast_state.data != 0,
                                    masked.ephemeral_mask.data.unsqueeze(0).expand_as(masked.fast_state.data)))
        # S starts from the full nn.Linear init in both (nothing zeroed under the mask).
        fresh = layer('additive_masked')
        self.assertTrue(torch.equal(fresh.per_sample_weights.data[0], fresh.weight.data))

    def test_matched_gain_equalizes_the_expected_read(self):
        """README "matched gain": over random masks the masked read of one write is phi times the
        dense read, so alpha_dense = phi * alpha_masked gives the same mean fast read."""
        torch.manual_seed(0)
        d, phi, trials = 400, 0.1, 64
        x = torch.randn(1, d)
        p = torch.randn(1, 30)
        ratios = []
        for _ in range(trials):
            m = torch.rand(30, d) < phi
            masked = ((p.T @ x) * m) @ x.T  # [(p x^T) * m] x, alpha_masked = 1
            dense = phi * (p.T @ x) @ x.T   # alpha_dense = phi
            ratios.append((masked.squeeze() / dense.squeeze()))
        ratio = torch.stack(ratios).mean().item()
        self.assertAlmostEqual(ratio, 1.0, delta=0.05)


class WipeTest(unittest.TestCase):
    def test_wipe_zeroes_F_and_averages_all_of_S(self):
        for structure in ADDITIVE:
            with self.subTest(structure):
                lyr = layer(structure)
                for x, e in inputs(3):
                    step(lyr, x, e, 0.3)
                self.assertGreater(lyr.fast_state.data.abs().sum().item(), 0)
                s_mean = lyr.per_sample_weights.data.mean(0, keepdim=True)
                self.assertFalse(torch.equal(lyr.per_sample_weights.data[0], lyr.per_sample_weights.data[1]))
                fast_before = lyr.fast_state.data.clone()
                weight_before = lyr.weight.data.clone()
                lyr.start_sequence_wipe(wipe_fast=False)
                assert_same(self, lyr.fast_state.data, fast_before)  # kept per row
                assert_same(self, lyr.per_sample_weights.data, s_mean.expand_as(lyr.per_sample_weights))
                lyr.start_sequence_wipe()
                self.assertEqual(lyr.fast_state.data.abs().sum().item(), 0.0)
                assert_same(self, lyr.weight.data, weight_before)  # the unused base weight is left alone


def build_rnn(structure, **options):
    seed_everything(7, deterministic=True)
    settings = dict(updater="dfa", batch_size=2, plasticity=50.0, forget_rate=0.2, ephemeral_fraction=0.5,
                    enable_recurrence=True, slow_weight_decay=0.05, fast_structure=structure)
    settings.update(options)
    with contextlib.redirect_stdout(io.StringIO()):
        return EphemeralRNN(10, 5, 4, 2, CHARSET, **settings)


def hconfig(**options):
    return config(**{"ephemeral_update_clamp": 0, **options})


def train_once(model, cfg=None, batches=1):
    state = {"training_instance": 0, "log_norms_now": False}
    for _ in range(batches):
        with contextlib.redirect_stdout(io.StringIO()):
            train_module.train_batch(None, episodes(TEXTS), model, cfg or hconfig(), state)
    return state


class RnnIntegrationTest(unittest.TestCase):
    def test_i2o_is_always_exclusive_and_the_others_follow_the_flag(self):
        model = build_rnn('additive_dense')
        self.assertEqual([l.fast_structure for l in model.trained_layers()],
                         ['additive_dense'] * 3 + ['exclusive'])
        self.assertNotIn('fast_state', dict(model.i2o.named_parameters()))

    def test_fast_layers_leave_the_other_trunk_layers_all_slow(self):
        model = build_rnn('additive_dense', fast_layers=[0])
        self.assertEqual([l.fast_structure for l in model.trained_layers()],
                         ['additive_dense', 'exclusive', 'additive_dense', 'exclusive'])
        self.assertFalse(model.linear_layers[1].ephemeral_mask.any())
        self.assertNotIn('fast_state', dict(model.linear_layers[1].named_parameters()))
        train_once(model)

    def test_exclusive_model_is_unchanged(self):
        seed_everything(7, deterministic=True)
        with contextlib.redirect_stdout(io.StringIO()):
            omitted = EphemeralRNN(10, 5, 4, 2, CHARSET, updater="dfa", batch_size=2, plasticity=50.0,
                                   forget_rate=0.2, ephemeral_fraction=0.5, enable_recurrence=True,
                                   slow_weight_decay=0.05)
        named = build_rnn('exclusive')
        for model in (omitted, named):
            train_once(model, batches=2)
        self.assertEqual(list(omitted.state_dict()), list(named.state_dict()))
        for key, value in omitted.state_dict().items():
            assert_same(self, named.state_dict()[key], value)

    def test_fused_eager_step_equals_unfused_step(self):
        for structure in ADDITIVE:
            for options, cfg in (({}, {}), ({'weight_clamp': 0.3, 'fast_weight_clamp': 0.02},
                                            {'ephemeral_update_clamp': 0.05})):
                with self.subTest(structure=structure, options=options):
                    unfused, fused = build_rnn(structure, **options), build_rnn(structure, **options)
                    fused.enable_fused_update(compile=False)
                    for model in (unfused, fused):
                        train_once(model, hconfig(**cfg), batches=2)
                    for key, value in unfused.state_dict().items():
                        assert_same(self, fused.state_dict()[key], value)

    def test_grad_norm_clip_fused_matches_unfused_to_rounding_and_binds(self):
        for structure in ADDITIVE:
            with self.subTest(structure):
                unfused, fused, free = (build_rnn(structure) for _ in range(3))
                fused.enable_fused_update(compile=False)
                for model in (unfused, fused):
                    train_once(model, hconfig(grad_norm_clip=0.05))
                train_once(free)
                for key, value in unfused.state_dict().items():
                    torch.testing.assert_close(fused.state_dict()[key], value, rtol=1e-5, atol=1e-6)
                self.assertGreater(fused.grad_clip_stats.count, 0)
                clipped = fused.linear_layers[0].fast_state.data.abs().sum()
                self.assertLess(clipped.item(), free.linear_layers[0].fast_state.data.abs().sum().item())

    def test_compiled_fused_step_matches_to_rounding(self):
        for structure in ADDITIVE:
            with self.subTest(structure):
                eager, compiled = build_rnn(structure), build_rnn(structure)
                compiled.enable_fused_update(compile=True)
                train_once(eager)
                try:
                    train_once(compiled)
                except Exception as exc:  # no C++ compiler for Inductor's CPU backend, for example
                    self.skipTest(f"torch.compile unavailable here: {type(exc).__name__}: {exc}")
                for key, value in eager.state_dict().items():
                    torch.testing.assert_close(compiled.state_dict()[key], value, rtol=1e-4, atol=1e-5)

    def test_plasticity_schedule_reaches_the_fast_step(self):
        for structure in ADDITIVE:
            with self.subTest(structure):
                low, set_low = build_rnn(structure, plasticity=5.0), build_rnn(structure)
                set_low.set_plasticity(5.0)
                for model in (low, set_low):
                    train_once(model)
                for key, value in low.state_dict().items():
                    if not key.endswith('plasticity'):
                        assert_same(self, set_low.state_dict()[key], value)

    def test_wipe_every_carries_F(self):
        model = build_rnn('additive_dense')
        cfg = hconfig()
        cfg['wipe_every'] = 2
        state = {"training_instance": 0, "log_norms_now": False}
        with contextlib.redirect_stdout(io.StringIO()):
            train_module.train_batch(None, episodes(TEXTS), model, cfg, state)
        carried = model.linear_layers[0].fast_state.data.clone()
        self.assertGreater(carried.abs().sum().item(), 0)
        model.start_sequence_wipe(wipe_fast=False)
        assert_same(self, model.linear_layers[0].fast_state.data, carried)
        with contextlib.redirect_stdout(io.StringIO()):
            train_module.train_batch(None, episodes(TEXTS), model, cfg, state)  # sequence 1: no wipe
            train_module.train_batch(None, episodes(TEXTS), model, cfg, state)  # sequence 2: wipe first
        self.assertEqual(state['sequence_count'], 3)

    def test_unsupported_combinations_are_refused_with_a_clear_error(self):
        cases = [
            (dict(slow_update_every='sequence'), "slow_update_every"),
            (dict(updater='backprop'), "updater backprop"),
            (dict(fast_backward_per_forward=2), "fast_backward_per_forward"),
            (dict(decouple_fast_feedback=[1]), "decouple_fast_feedback"),
            (dict(fast_write_from_readout=[1]), "fast_write_from_readout"),
            (dict(fast_read_norm='rms'), "fast_read_norm"),
            (dict(slow_nlms=True), "slow_nlms"),
            (dict(trunk_slow_lr_scale=0.5), "trunk_slow_lr_scale"),
        ]
        for structure in ADDITIVE:
            for options, message in cases:
                with self.subTest(structure=structure, options=options):
                    with self.assertRaisesRegex(ValueError, message):
                        build_rnn(structure, **options)
        with self.assertRaisesRegex(ValueError, "fast_structure takes one of"):
            build_rnn('bogus')
        # Exclusive keeps every combination it had.
        build_rnn('exclusive', fast_backward_per_forward=2, fast_read_norm='rms')

    def test_cli_refuses_unsupported_combinations(self):
        parser = train_module.build_parser()
        for extra in (["--updater", "bptt"], ["--slow_update_every", "4"], ["--fast_read_norm", "rms"],
                      ["--fast_backward_per_forward", "2"], ["--model_type", "rnn"]):
            with self.subTest(extra=extra):
                args = parser.parse_args(["--fast_structure", "additive_dense", *extra])
                with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                    train_module.check_argument_combinations(args, parser)
        args = parser.parse_args(["--fast_structure", "additive_dense", "--fused_update", "true",
                                  "--grad_norm_clip", "1", "--fast_weight_clamp", "1"])
        train_module.check_argument_combinations(args, parser)


class HeldOutTest(unittest.TestCase):
    """heldout.py supports the additive structures: its fast-only step writes F and nothing else,
    and F evolves exactly as training's own step writes it when the slow step is undone."""

    def test_fast_only_step_leaves_slow_weights_and_biases_alone(self):
        for structure in ADDITIVE:
            with self.subTest(structure):
                model = build_rnn(structure)
                train_once(model)  # so S differs between rows and across the model
                before = copy.deepcopy(model.state_dict())
                heldout.evaluate_held_out(model, episodes(TEXTS), torch.ones(2, 6, dtype=torch.bool), hconfig())
                for layer_, name in zip(model.trained_layers(), ('linear_layers.0', 'linear_layers.1', 'i2h', 'i2o')):
                    s_mean = before[f'{name}.per_sample_weights'].mean(0, keepdim=True)
                    assert_same(self, layer_.per_sample_weights.data, s_mean.expand_as(layer_.per_sample_weights))
                    assert_same(self, layer_.bias.data, before[f'{name}.bias'])
                    if layer_.is_additive:
                        self.assertGreater(layer_.fast_state.data.abs().sum().item(), 0)

    def test_fast_only_writes_equal_the_training_step_with_the_slow_step_undone(self):
        for structure in ADDITIVE:
            for cfg in ({}, {'ephemeral_update_clamp': 0.05}):
                with self.subTest(structure=structure, cfg=cfg):
                    trained = build_rnn(structure, weight_clamp=0.6, fast_weight_clamp=0.05)
                    evaluated = copy.deepcopy(trained)
                    frozen = {}

                    def wipe_then_snapshot(wipe=trained.start_sequence_wipe):
                        wipe()
                        for lyr in trained.trained_layers():
                            frozen[lyr] = (lyr.per_sample_weights.data.clone(), lyr.bias.data.clone())

                    def forget_then_restore():
                        for lyr in trained.trained_layers():
                            weights, bias = frozen[lyr]
                            lyr.per_sample_weights.data.copy_(weights)
                            lyr.bias.data.copy_(bias)

                    trained.start_sequence_wipe, trained.apply_forget_step = wipe_then_snapshot, forget_then_restore
                    onehot = episodes(TEXTS)
                    with contextlib.redirect_stdout(io.StringIO()):
                        train_module.train_batch(None, onehot, trained, hconfig(**cfg),
                                                 {"training_instance": 0, "log_norms_now": False})
                    heldout.evaluate_held_out(evaluated, onehot, torch.ones(2, 6, dtype=torch.bool), hconfig(**cfg))
                    for a, b in zip(trained.trained_layers(), evaluated.trained_layers()):
                        if a.is_additive:
                            assert_same(self, b.fast_state.data, a.fast_state.data)

    def test_no_fast_protocol_keeps_F_at_zero(self):
        model = build_rnn('additive_dense')
        heldout.evaluate_held_out(model, episodes(TEXTS), None, hconfig())
        for layer_ in model.trained_layers():
            if layer_.is_additive:
                self.assertEqual(layer_.fast_state.data.abs().sum().item(), 0.0)

    def test_evaluate_protocols_restores_the_state(self):
        model = build_rnn('additive_masked')
        train_once(model)
        before = copy.deepcopy(model.state_dict())
        results = heldout.evaluate_protocols(model, [(TEXTS, episodes(TEXTS))], hconfig(), DATASET)
        self.assertIn("heldout_no_fast/recall_acc", results)
        for key, value in model.state_dict().items():
            assert_same(self, value, before[key])


class LoopTraceTest(unittest.TestCase):
    def test_traces_read_F_and_S_and_change_nothing(self):
        for structure in ADDITIVE:
            with self.subTest(structure):
                traced, plain = build_rnn(structure), build_rnn(structure)
                cfg = hconfig()
                tracer = LoopTracer(traced, cfg["learning_rate"])
                state = {"training_instance": 0, "log_norms_now": False}
                onehot = episodes(TEXTS)
                with contextlib.redirect_stdout(io.StringIO()):
                    # the first step's read: before any write F is 0, so its fast drive is 0
                    train_module.train_batch(None, onehot, traced, cfg, dict(state), tracer=tracer)
                    traces = tracer.finish()
                    train_module.train_batch(None, onehot, plain, cfg, dict(state))
                for key, value in plain.state_dict().items():
                    assert_same(self, traced.state_dict()[key], value)
                self.assertEqual(traces["fast_drive"][0].abs().sum().item(), 0.0)
                self.assertGreater(traces["fast_drive"][1:].abs().sum().item(), 0.0)
                self.assertGreater(traces["fast_delta"].abs().sum().item(), 0.0)
                self.assertTrue(torch.isfinite(traces["loop_gain"][1:]).any())


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
            with self.assertRaises(RuntimeError):  # a resume refuses a different structure
                run_main("--resume", "--n_iters", "5", "--fast_structure", "exclusive", checkpoint_dir=directory)
            run_main("--resume", "--n_iters", "5", "--fast_structure", "additive_masked",
                     "--ephemeral_fraction", "0.5", "--plasticity", "5", checkpoint_dir=directory)
            self.assertEqual(latest(directory)["iter"], 6)  # the next iteration

    def test_heldout_cli_rebuilds_an_additive_model(self):
        from unittest.mock import patch
        import os
        with tempfile.TemporaryDirectory() as directory:
            run_main("--fast_structure", "additive_dense", "--plasticity", "5", checkpoint_dir=directory)
            output = io.StringIO()
            with patch.object(heldout, "load_heldout_batches", side_effect=fake_heldout_batches), \
                    contextlib.redirect_stdout(output):
                heldout.main(["--checkpoint", os.path.join(directory, "latest_checkpoint.pth"),
                              "--device", "cpu", "--protocols", "strict", "no_fast"])
            self.assertIn("heldout_strict/recall_acc", output.getvalue())

    def test_load_checkpoint_restores_F(self):
        source = build_rnn('additive_dense')
        train_once(source)
        cfg = {"n_hidden": 5, "n_layers": 2, "updater": "dfa", "charset_size": 4, "model_type": "ephemeral",
               "forget_rate": 0.2, "fast_structure": "additive_dense"}
        checkpoint = {"config": cfg, "model_state_dict": source.state_dict(), "code_version": 12}
        target = build_rnn('additive_dense', plasticity=1.0)
        with contextlib.redirect_stdout(io.StringIO()):
            load_checkpoint("<memory>", target, cfg, checkpoint=checkpoint)
        for a, b in zip(source.trained_layers(), target.trained_layers()):
            if a.is_additive:
                assert_same(self, a.fast_state.data, b.fast_state.data)
                self.assertGreater(a.fast_state.data.abs().sum().item(), 0)
        with self.assertRaises(RuntimeError), contextlib.redirect_stdout(io.StringIO()):
            load_checkpoint("<memory>", build_rnn('exclusive'), {**cfg, "fast_structure": "exclusive"},
                            checkpoint=checkpoint)


class StepFunctionTest(unittest.TestCase):
    def test_additive_step_is_two_dfa_layer_steps(self):
        """dfa_additive_layer_step = dfa_layer_step on S with an empty mask and alpha 1, then on F
        with freeze_slow on the fast mask (written out here, so a change to either half shows)."""
        g = torch.Generator().manual_seed(5)
        slow, fast = torch.randn(2, 4, 6, generator=g), torch.zeros(2, 4, 6)
        bias = torch.randn(4, generator=g)
        p, x = torch.randn(2, 4, generator=g), torch.randn(2, 6, generator=g)
        mask, none = torch.rand(4, 6, generator=g) < 0.5, torch.zeros(4, 6, dtype=torch.bool)
        s1, f1, b1 = slow.clone(), fast.clone(), bias.clone()
        dfa_additive_layer_step(s1, f1, b1, p, x, 7.0, mask, none, 0.1, 0.2, 0.3, 1.5, 0.01, 0.4)
        lr, alpha = 0.2, 7.0
        expected_slow = ((slow - lr * p.unsqueeze(2) * x.unsqueeze(1)).clamp(-1.5, 1.5)) * (1 - 0.01)
        write = (-alpha * p.unsqueeze(2) * x.unsqueeze(1)).clamp(-0.3, 0.3)
        expected_fast = torch.where(mask, (lr * write).clamp(-0.4, 0.4) * (1 - 0.1), torch.zeros(()))
        torch.testing.assert_close(s1, expected_slow)
        torch.testing.assert_close(f1, expected_fast)
        torch.testing.assert_close(b1, bias - lr * p.mean(0))
        # fast_only: S and the bias untouched.
        s2, f2, b2 = slow.clone(), fast.clone(), bias.clone()
        dfa_additive_layer_step(s2, f2, b2, p, x, 7.0, mask, none, 0.1, 0.2, 0.3, 1.5, 0.01, 0.4, fast_only=True)
        assert_same(self, s2, slow)
        assert_same(self, b2, bias)
        assert_same(self, f2, f1)


if __name__ == "__main__":
    unittest.main()
