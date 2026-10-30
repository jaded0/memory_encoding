"""B2 round 2 (scratch/b2_tinystories_2026-10-22/b2_eval.py): the WIPE R protocol of the horizon mode."""
import contextlib
import importlib.util
import io
import json
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from ephemeral_model import EphemeralRNN, SimpleRNN, dfa_output_error
from reproducibility import seed_everything
from utils import model_input

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = importlib.util.spec_from_file_location(
    "b2_eval", os.path.join(HERE, "scratch", "b2_tinystories_2026-10-22", "b2_eval.py"))
b2_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(b2_eval)

CHARSET = "23. "
TEXTS = ["23.32  3.22 3.2 2", "322.223 .3 2..3 3"]  # 17 characters, 16 predicted steps


def build(**options):
    # The B2 setting: recurrence off, last_one input, no positional encoding, so nothing but fast entries
    # carries context; a nonzero fast clamp and forget so the wipe has state to clear.
    settings = dict(updater="dfa", batch_size=len(TEXTS), plasticity=50.0, forget_rate=0.05,
                    ephemeral_fraction=0.5, enable_recurrence=False, weight_clamp=0.6, fast_weight_clamp=0.5)
    settings.update(options)
    seed_everything(7, deterministic=True)
    with contextlib.redirect_stdout(io.StringIO()):
        model = EphemeralRNN(4, 6, 4, 3, CHARSET, **settings)
    model.eval()
    return model


CONFIG = {"input_mode": "last_one", "pe_matrix": None, "learning_rate": 0.3, "ephemeral_update_clamp": 0.05,
          "grad_norm_clip": 0}


def onehot(texts):
    return F.one_hot(torch.tensor([[CHARSET.index(c) for c in text] for text in texts]), 4).float()


class WipeProtocolTest(unittest.TestCase):
    def setUp(self):
        self.model = build()
        self.hot = onehot(TEXTS)
        self.on = b2_eval.run_pass(self.model, self.hot, CONFIG, write=True)
        self.off = b2_eval.run_pass(self.model, self.hot, CONFIG, write=False)

    def test_fast_weights_matter(self):
        self.assertFalse(torch.equal(self.on, self.off))

    def test_wipe_1_is_off(self):
        self.assertTrue(torch.equal(b2_eval.run_pass(self.model, self.hot, CONFIG, write=True, wipe_every=1), self.off))

    def test_wipe_at_or_past_length_is_on(self):
        for r in (16, 17, 100):
            self.assertTrue(torch.equal(b2_eval.run_pass(self.model, self.hot, CONFIG, write=True, wipe_every=r),
                                        self.on))

    def test_each_block_equals_a_fresh_pass(self):
        r = 5
        wiped = b2_eval.run_pass(self.model, self.hot, CONFIG, write=True, wipe_every=r)
        self.assertFalse(torch.equal(wiped, self.on))
        steps = self.hot.shape[1] - 1
        for lo in range(0, steps, r):
            hi = min(lo + r, steps)
            fresh = b2_eval.run_pass(self.model, self.hot[:, lo:hi + 1], CONFIG, write=True)
            torch.testing.assert_close(wiped[:, lo:hi], fresh, rtol=0, atol=1e-6)

    def test_slow_entries_untouched(self):
        before = [layer.per_sample_weights.detach().clone() for layer in self.model.linear_layers]
        masks = [layer.ephemeral_mask for layer in self.model.linear_layers]
        b2_eval.run_pass(self.model, self.hot, CONFIG, write=True, wipe_every=3)
        for layer, old, mask in zip(self.model.linear_layers, before, masks):
            self.assertTrue(torch.equal(layer.per_sample_weights[:, ~mask], old[:, ~mask]))


@torch.no_grad()
def reference_pass(model, hot, write=False, boundary=None, reset_hidden=False, keys=False):
    """Explicit causal loop, also the pre-change default evaluation when boundary is None."""
    if b2_eval.has_fast_weights(model):
        model.start_sequence_wipe()
    hidden = model.initHidden(hot.shape[0])
    losses = []
    collected = [[] for _ in model.linear_layers] if keys else []
    for i in range(hot.shape[1] - 1):
        if i == boundary:
            b2_eval.wipe_fast(model)
            if reset_hidden:
                hidden = model.initHidden(hot.shape[0])
        output, hidden = model(model_input(hot, i, CONFIG["input_mode"], CONFIG["pe_matrix"]), hidden)
        for store, layer in zip(collected, model.linear_layers):
            store.append(layer.in_traces.data.clone())
        loss, error = dfa_output_error(output, hot[:, i + 1], torch.nn.CrossEntropyLoss(reduction="none"))
        losses.append(loss.detach())
        if write and b2_eval.has_fast_weights(model):
            model.fast_only_dfa_step(error, CONFIG["learning_rate"], CONFIG["ephemeral_update_clamp"],
                                     CONFIG["grad_norm_clip"])
    out = torch.stack(losses, 1)
    return (out, [torch.stack(store) for store in collected]) if keys else out


class RecurrentProtocolTest(unittest.TestCase):
    def setUp(self):
        self.model = build(enable_recurrence=True)
        self.hot = onehot(TEXTS)

    def test_legacy_default_losses_and_keys_are_bit_identical(self):
        for recurrence in (False, True):
            model = build(enable_recurrence=recurrence)
            for write in (False, True):
                expected, expected_keys = reference_pass(model, self.hot, write=write, keys=True)
                actual, keys = b2_eval.run_pass(model, self.hot, CONFIG, write=write, keys=True)
                self.assertTrue(torch.equal(actual, expected))
                self.assertEqual(len(keys), len(model.linear_layers))
                self.assertEqual(keys[0].shape[:2], (self.hot.shape[1] - 1, len(TEXTS)))
                for actual_key, expected_key in zip(keys, expected_keys):
                    self.assertTrue(torch.equal(actual_key, expected_key))

    def test_off_keeps_hidden_but_never_writes(self):
        with mock.patch.object(self.model, "fast_only_dfa_step", wraps=self.model.fast_only_dfa_step) as step:
            actual = b2_eval.run_pass(self.model, self.hot, CONFIG, write=False)
            step.assert_not_called()
        self.assertTrue(torch.equal(actual, reference_pass(self.model, self.hot)))
        no_recurrence = build(enable_recurrence=False)
        self.assertFalse(torch.equal(actual, b2_eval.run_pass(no_recurrence, self.hot, CONFIG, write=False)))

    def test_pass_and_story_hidden_state_is_fresh(self):
        expected = b2_eval.run_pass(self.model, self.hot, CONFIG, write=False)
        b2_eval.run_pass(self.model, self.hot.flip(1), CONFIG, write=True)
        self.assertTrue(torch.equal(expected, b2_eval.run_pass(self.model, self.hot, CONFIG, write=False)))
        # A different story in row 0 must not affect the story in row 1 (or its next pass).
        changed = self.hot.clone()
        changed[0] = self.hot[1].flip(0)
        actual = b2_eval.run_pass(self.model, changed, CONFIG, write=False)
        self.assertTrue(torch.equal(expected[1], actual[1]))
        swapped = b2_eval.run_pass(self.model, self.hot.flip(0), CONFIG, write=False)
        self.assertTrue(torch.equal(expected, swapped.flip(0)))

    def test_reset_preserves_hidden_and_prefix(self):
        boundary = 6
        actual = b2_eval.run_pass(self.model, self.hot, CONFIG, write=True, wipe_at=boundary)
        expected = reference_pass(self.model, self.hot, write=True, boundary=boundary)
        self.assertTrue(torch.equal(actual, expected))
        wrong = reference_pass(self.model, self.hot, write=True, boundary=boundary, reset_hidden=True)
        self.assertFalse(torch.equal(actual[:, boundary:], wrong[:, boundary:]))
        snippet = b2_eval.run_pass(self.model, self.hot[:, boundary:], CONFIG, write=True)
        self.assertFalse(torch.equal(actual[:, boundary:], snippet))

    def test_recurrent_reset_mention_scores_use_full_prefix(self):
        # Use a small R here to exercise the production RESET scorer on short CPU fixtures.
        pairs = [{"story": row, "m1": (8, 10), "m2": (12, 15)} for row in range(len(TEXTS))]
        jobs = [(i, which) for i in range(len(pairs)) for which in (1, 2)]
        with mock.patch.object(b2_eval, "R", 4):
            b2_eval.reset_mentions(self.model, CONFIG, TEXTS, pairs, jobs,
                                   {c: i for i, c in enumerate(CHARSET)}, 4, len(TEXTS), "cpu")
        for which in (1, 2):
            s, e = pairs[0][f"m{which}"]
            expected = reference_pass(self.model, self.hot[:, :e], write=True, boundary=s - 4)
            for row in range(len(TEXTS)):
                self.assertEqual(pairs[row][f"reset{which}"], b2_eval.mention_surprisal(expected[row], s, e))

    def test_horizon_skips_off_equivalence_assertion_with_recurrence(self):
        texts = ["23. 23 22. 23", "32. 32 33. 32"]
        # Letters are needed for every position class to have a nonzero count.
        charset = "ab. "
        texts = [t.translate(str.maketrans("23", "ab")) for t in texts]
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(device="cpu", checkpoint="unused", stories=2, length=len(texts[0]),
                                   val_file=None, wipe_every="1,4", out=os.path.join(directory, "horizon"))
            config = dict(CONFIG, batch_size=2)
            with mock.patch.object(b2_eval, "load_model", return_value=(self.model, config, charset, 0)), \
                    mock.patch.object(b2_eval, "load_validation", return_value=texts), \
                    mock.patch.object(b2_eval, "run_pass", wraps=b2_eval.run_pass) as run, \
                    contextlib.redirect_stdout(io.StringIO()):
                b2_eval.horizon(args)
            self.assertEqual(sum(c.kwargs.get("wipe_every") == 1 for c in run.call_args_list), 1)


class InferenceForgettingTest(unittest.TestCase):
    def test_parser_validation(self):
        parser = b2_eval.build_parser()
        base = ["curve", "--checkpoint", "unused", "--out", "unused"]
        self.assertIsNone(parser.parse_args(base).inference_forget_rate)
        for rate in (0, 0.01, 0.03, 0.05, 0.1, 1):
            self.assertEqual(parser.parse_args(base + ["--forget_rate_override", str(rate)]).
                             inference_forget_rate, rate)
        for rate in ("nan", "inf", "-0.01", "1.01", "invalid"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(base + ["--forget_rate_override", rate])

    def test_actual_unfused_decay_and_losses_all_deep_rates(self):
        hot = onehot(TEXTS)
        baseline = build(forget_rate=0.01)
        expected = b2_eval.run_pass(baseline, hot, CONFIG, write=True)
        expected_off = b2_eval.run_pass(baseline, hot, CONFIG, write=False)
        for rate in (0.03, 0.05, 0.1):
            config = dict(CONFIG, forget_rate=0.01)
            model = build(forget_rate=0.01)
            metadata = b2_eval.inference_forgetting(model, config, rate)
            self.assertEqual(config["forget_rate"], rate)
            self.assertEqual(metadata["checkpoint_forget_rate"], 0.01)
            self.assertEqual(metadata["effective_forget_rate"], rate)
            self.assertEqual(set(metadata["original_module_rates"].values()), {0.01})
            self.assertEqual(set(metadata["effective_module_rates"].values()), {rate})
            for layer in (*model.linear_layers, model.i2h, model.i2o):
                self.assertEqual(layer.forget_rate, rate)
            self.assertIsNone(model.fused_layer_step)
            # A zero-error actual fast-only updater call must decay each fast entry by 1-rate.
            model.start_sequence_wipe()
            model(hot[:, 0], model.initHidden(hot.shape[0]))
            layers = [*model.linear_layers, model.i2h]
            with torch.no_grad():
                for layer in layers:
                    layer.per_sample_weights[:, layer.ephemeral_mask] = 0.2
            before = [layer.per_sample_weights.detach().clone() for layer in layers]
            model.fast_only_dfa_step(torch.zeros(hot.shape[0], 4), 0.3, 0.05)
            for layer, old in zip(layers, before):
                torch.testing.assert_close(layer.per_sample_weights[:, layer.ephemeral_mask],
                                           old[:, layer.ephemeral_mask] * (1 - rate), rtol=0, atol=0)
                self.assertTrue(torch.equal(layer.per_sample_weights[:, ~layer.ephemeral_mask],
                                            old[:, ~layer.ephemeral_mask]))
            # Fresh model so the analytic fixture cannot influence the causal scoring comparison.
            model = build(forget_rate=0.01)
            b2_eval.inference_forgetting(model, config, rate)
            self.assertFalse(torch.equal(expected, b2_eval.run_pass(model, hot, config, write=True)))
            self.assertTrue(torch.equal(expected_off, b2_eval.run_pass(model, hot, config, write=False)))

    def test_checkpoint_default_equivalence_and_override(self):
        from train import build_model, build_parser
        from utils import initialize_charset, save_checkpoint
        config = vars(build_parser().parse_args([]))
        config.update(updater="dfa", n_hidden=6, n_layers=3, batch_size=2, input_mode="last_one",
                      positional_encoding_dim=0, forget_rate=0.01, fused_update=False)
        charset, c2i, _, size = initialize_charset(config["dataset"])
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            model = build_model(config, charset, size).eval()
            save_checkpoint({"config": config, "model_state_dict": model.state_dict(), "iter": 12},
                            directory, "dfa.pth")
            path = os.path.join(directory, "dfa.pth")
            loaded, loaded_config, _, _ = b2_eval.load_model(path, "cpu")
            hot, _ = b2_eval.onehot_batch([charset[:11], charset[::-1][:11]], c2i, size, 11, 2, "cpu")
            for write in (False, True):
                expected = b2_eval.run_pass(loaded, hot, loaded_config, write=write)
                metadata = b2_eval.inference_forgetting(loaded, loaded_config)
                self.assertEqual(metadata["original_module_rates"], metadata["effective_module_rates"])
                self.assertIsNone(metadata["override"])
                self.assertTrue(torch.equal(expected, b2_eval.run_pass(loaded, hot, loaded_config, write=write)))
            b2_eval.inference_forgetting(loaded, loaded_config, 0.1)
            self.assertEqual(loaded_config["forget_rate"], 0.1)
            self.assertTrue(all(layer.forget_rate == 0.1 for layer in
                                (*loaded.linear_layers, loaded.i2h, loaded.i2o)))
            reloaded, _, _, _ = b2_eval.load_model(path, "cpu")
            self.assertEqual(reloaded.forget_rate, 0.01)  # checkpoint was not rewritten

    def test_reject_nonfast_override(self):
        model = SimpleRNN(4, 6, 4, 3, updater="bptt")
        with self.assertRaisesRegex(ValueError, "requires a model with fast weights"):
            b2_eval.inference_forgetting(model, {"forget_rate": 0.01}, 0.1)

    def test_all_scoring_modes_write_original_and_effective_metadata(self):
        charset = "ab. "
        texts = ["ab. ab aa. ab", "ba. ba bb. ba"]
        with tempfile.TemporaryDirectory() as directory:
            for mode in ("tiny", "curve", "horizon", "synth"):
                config = dict(CONFIG, batch_size=2, dataset="unused", forget_rate=0.01)
                model = build(forget_rate=0.01)
                args = SimpleNamespace(device="cpu", checkpoint="unused", stories=2, length=len(texts[0]),
                                       val_file=None, names=None, mode=mode, batches=1, wipe_every="1,4",
                                       inference_forget_rate=0.05, out=os.path.join(directory, mode))
                with mock.patch.object(b2_eval, "load_model", return_value=(model, config, charset, 0)), \
                        mock.patch.object(b2_eval, "load_validation", return_value=texts), \
                        mock.patch("heldout.load_heldout_batches", return_value=[(TEXTS, onehot(TEXTS))]), \
                        contextlib.redirect_stdout(io.StringIO()):
                    getattr(b2_eval, mode if mode != "curve" else "tiny")(args)
                with open(args.out + ".json") as handle:
                    result = json.load(handle)
                metadata = result["inference_forgetting"]
                self.assertEqual(metadata["checkpoint_forget_rate"], 0.01)
                self.assertEqual(metadata["effective_forget_rate"], 0.05)
                self.assertEqual(metadata["override"], 0.05)
                self.assertEqual(set(metadata["effective_module_rates"].values()), {0.05})
                if "arm_config" in result:
                    self.assertEqual(result["arm_config"]["forget_rate"], 0.05)


class SimpleRNNProtocolTest(unittest.TestCase):
    def setUp(self):
        seed_everything(7, deterministic=True)
        self.model = SimpleRNN(4, 6, 4, 3, enable_recurrence=True, updater="bptt").eval()
        self.hot = onehot(TEXTS)

    def test_load_bptt_checkpoint_and_evaluate(self):
        from train import build_model, build_parser
        from utils import initialize_charset, save_checkpoint
        config = vars(build_parser().parse_args([]))
        config.update(model_type="rnn", updater="bptt", enable_recurrence=True, n_hidden=6,
                      n_layers=3, batch_size=2, input_mode="last_one", positional_encoding_dim=0)
        charset, c2i, _, size = initialize_charset(config["dataset"])
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            model = build_model(config, charset, size).eval()
            save_checkpoint({"config": config, "model_state_dict": model.state_dict(), "iter": 12},
                            directory, "bptt.pth")
            loaded, loaded_config, loaded_charset, iteration = b2_eval.load_model(
                os.path.join(directory, "bptt.pth"), "cpu")
            texts = [(charset * 11)[:11], (charset[::-1] * 11)[:11]]
            hot, _ = b2_eval.onehot_batch(texts, c2i, size, 11, 2, "cpu")
            expected = b2_eval.run_pass(model, hot, loaded_config, write=False)
            actual, keys = b2_eval.run_pass(loaded, hot, loaded_config, write=True, keys=True)
        self.assertIsInstance(loaded, SimpleRNN)
        self.assertTrue(loaded.enable_recurrence)
        self.assertEqual(loaded_charset, charset)
        self.assertEqual(iteration, 11)  # evaluator reports checkpoint's next iteration minus one
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(keys, [])

    def test_no_fast_protocols_or_keys_and_weights_unchanged(self):
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        expected = reference_pass(self.model, self.hot)
        for options in ({"write": False}, {"write": True}, {"write": True, "wipe_every": 1},
                        {"write": True, "wipe_at": 6}):
            actual, keys = b2_eval.run_pass(self.model, self.hot, CONFIG, keys=True, **options)
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(keys, [])
        for k, v in self.model.state_dict().items():
            self.assertTrue(torch.equal(v, before[k]))

    def test_hidden_is_recurrent_and_resets_between_stories_and_passes(self):
        expected = b2_eval.run_pass(self.model, self.hot, CONFIG, write=False)
        b2_eval.run_pass(self.model, self.hot.flip(1), CONFIG, write=True)
        self.assertTrue(torch.equal(expected, b2_eval.run_pass(self.model, self.hot, CONFIG, write=False)))
        # Permuting batch rows can change CPU GEMM rounding by one float32 ULP;
        # identical-order repeat passes above must still be bit-identical.
        torch.testing.assert_close(expected, b2_eval.run_pass(self.model, self.hot.flip(0), CONFIG,
                                                             write=True).flip(0), rtol=0, atol=1e-6)
        self.model.enable_recurrence = False
        self.assertFalse(torch.equal(expected, b2_eval.run_pass(self.model, self.hot, CONFIG, write=False)))

    def test_tiny_and_synth_report_empty_fast_key_pr(self):
        charset = "ab. "
        texts = ["ab. ab aa. ab", "ba. ba bb. ba"]
        config = dict(CONFIG, batch_size=2, dataset="unused")
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(device="cpu", checkpoint="unused", stories=2, length=len(texts[0]),
                                   val_file=None, names=None, mode="tiny", batches=1,
                                   out=os.path.join(directory, "tiny"))
            with mock.patch.object(b2_eval, "load_model", return_value=(self.model, config, charset, 0)), \
                    mock.patch.object(b2_eval, "load_validation", return_value=texts), \
                    contextlib.redirect_stdout(io.StringIO()):
                b2_eval.tiny(args)
            with open(args.out + ".json") as handle:
                result = json.load(handle)
            self.assertEqual(result["pr"], {"on": {}, "off": {}})
            self.assertEqual(result["ce"]["all"]["on"], result["ce"]["all"]["off"])
            args.out = os.path.join(directory, "synth")
            with mock.patch.object(b2_eval, "load_model", return_value=(self.model, config, charset, 0)), \
                    mock.patch("heldout.load_heldout_batches", return_value=[(TEXTS, self.hot)]), \
                    contextlib.redirect_stdout(io.StringIO()):
                b2_eval.synth(args)
            with open(args.out + ".json") as handle:
                self.assertEqual(json.load(handle)["pr"], {})


if __name__ == "__main__":
    unittest.main()
