"""--fast_layers, --decouple_fast_feedback and --fast_write_from_readout (ported 2026-10-15 from the
10-13 scratch patches; see tests/generate_scratch_flag_fixtures.py).

- Flags off, the model is the one main gave before the port (fixture "main/plain").
- Flags on, it reproduces the code_v2 snapshot the L01, DC2 and aimed-write runs used, including
  its held-out fast-only step (fixture "code_v2/*").
- The unfused step (and so train.py's print steps, which log update norms) and the eager fused step
  agree bit for bit for every flag; compiled, to rounding. Before the port the scratch flags
  existed only in the fused step, so every print step wrote with the plain DFA rule.
- The split (--fast_backward_per_forward K >= 2) and windowed (--slow_update_every) steps write the
  fast entries with the same error as the held-out fast-only step."""
import contextlib
import io
import unittest
from pathlib import Path

import torch

from ephemeral_model import EphemeralRNN, decoupled_fast_feedback
from reproducibility import seed_everything
from tests import scratch_flags_reference as ref
from train import build_model, parse_args, train

FIXTURE = Path(__file__).parent / "fixtures" / "scratch_flags.pt"


def build(**kwargs):
    return ref.build(EphemeralRNN, seed_everything, **kwargs)


def tolerance(fixture):
    """Exact on the Torch the fixture was generated with; otherwise to rounding."""
    if fixture["generated_with"]["torch"] == str(torch.__version__):
        return dict(rtol=0, atol=0)
    return dict(rtol=1e-5, atol=1e-6)


class FastLayerFlagsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.fixture = torch.load(FIXTURE)

    def assert_run(self, actual, expected, **tol):
        torch.testing.assert_close(torch.tensor(actual["losses"]), torch.tensor(expected["losses"]), **tol)
        self.assertEqual(actual["state"].keys(), expected["state"].keys())
        for name, tensor in expected["state"].items():
            torch.testing.assert_close(actual["state"][name], tensor, msg=name, **tol)

    def test_flags_off_matches_main_before_the_port(self):
        tol = tolerance(self.fixture)
        for fused in (False, True):
            with self.subTest(fused=fused):
                model = build()
                if fused:
                    model.enable_fused_update(compile=False)
                self.assert_run(ref.run(model, train), self.fixture["main"]["plain"], **tol)

    def test_flags_reproduce_the_code_v2_snapshot(self):
        tol = tolerance(self.fixture)
        for name, (kwargs, _, _) in ref.CASES.items():
            for fused in (False, True):
                with self.subTest(case=name, fused=fused):
                    model = build(**kwargs)
                    if fused:
                        model.enable_fused_update(compile=False)
                    self.assert_run(ref.run(model, train), self.fixture["code_v2"][name], **tol)

    def test_the_cases_differ(self):
        # Otherwise the reproduction above would not exercise the flags.
        plain = self.fixture["code_v2"]["plain"]["state"]
        for name in ("L01", "DC2", "RG2x0.1"):
            with self.subTest(case=name):
                state = self.fixture["code_v2"][name]["state"]
                self.assertFalse(torch.equal(state["2.weights"], plain["2.weights"]))

    def test_unfused_and_eager_fused_steps_are_bit_identical_with_norm_logging(self):
        for name, (kwargs, _, _) in ref.CASES.items():
            with self.subTest(case=name):
                unfused, fused = build(**kwargs), build(**kwargs)
                fused.enable_fused_update(compile=False)
                # every 4th batch logs norms: the fused model takes the unfused step there
                expected = ref.run(unfused, train)
                self.assert_run(ref.run(fused, train, log_norms_every=4), expected, rtol=0, atol=0)

    def test_compiled_fused_step_matches_to_rounding(self):
        for name in ("DC2", "RG2x0.1"):
            kwargs = ref.CASES[name][0]
            with self.subTest(case=name):
                unfused, fused = build(**kwargs), build(**kwargs)
                fused.enable_fused_update(compile=True)
                try:
                    actual = ref.run(fused, train)
                except Exception as exc:  # no C++ compiler for Inductor's CPU backend, for example
                    self.skipTest(f"torch.compile unavailable here: {type(exc).__name__}: {exc}")
                self.assert_run(actual, ref.run(unfused, train), rtol=1e-4, atol=1e-6)

    def test_fast_layers_only_zeroes_the_fraction(self):
        plain, l01 = build(), build(fast_layers=[0, 1])
        for index, (a, b) in enumerate(zip(plain.trained_layers(), l01.trained_layers())):
            if index == 2:
                self.assertFalse(b.ephemeral_mask.any())
                # the same random numbers were drawn: the slow init is the plain model's
                keep = ~a.ephemeral_mask
                self.assertTrue(torch.equal(a.per_sample_weights[:, keep], b.per_sample_weights[:, keep]))
            else:
                self.assertTrue(torch.equal(a.ephemeral_mask, b.ephemeral_mask))
                self.assertTrue(torch.equal(a.per_sample_weights, b.per_sample_weights))
        self.assertTrue(l01.i2h.ephemeral_mask.any())

    def test_decoupling_with_the_slow_matrix_is_the_plain_model(self):
        plain, same = build(), build(decouple_fast_feedback=[2])
        same.linear_layers[2].fast_feedback_weights = same.linear_layers[2].feedback_weights.data.clone()
        self.assert_run(ref.run(same, train), ref.run(plain, train), rtol=0, atol=0)

    def test_b_fast_is_deterministic_and_not_in_checkpoints(self):
        model = build(decouple_fast_feedback=[2])
        b_fast = model.linear_layers[2].fast_feedback_weights
        self.assertTrue(torch.equal(b_fast, decoupled_fast_feedback(*b_fast.shape, 2)))
        self.assertFalse(torch.equal(b_fast, model.linear_layers[2].feedback_weights))
        self.assertFalse(any("fast_feedback" in key for key in model.state_dict()))
        self.assertIsNone(model.linear_layers[1].fast_feedback_weights)

    def fast_step_from_same_state(self, kwargs, step):
        """Fast entries after `step` and after fast_only_dfa_step, from one trained state."""
        model = build(**kwargs)
        ref.run(model, train)
        output, _ = model(torch.eye(len(ref.CHARSET))[:ref.BATCH_SIZE], model.initHidden(ref.BATCH_SIZE))
        error = torch.softmax(output, 1) - torch.eye(len(ref.CHARSET))[1:ref.BATCH_SIZE + 1]
        saved = [layer.per_sample_weights.data.clone() for layer in model.trained_layers()]
        model.fast_only_dfa_step(error, ref.LEARNING_RATE, 0)
        expected = [layer.per_sample_weights.data.clone() for layer in model.trained_layers()]
        for layer, weights in zip(model.trained_layers(), saved):
            layer.per_sample_weights.data.copy_(weights)
        step(model, error)
        return model, expected

    def test_split_and_windowed_steps_write_the_same_fast_entries(self):
        steps = {
            "split": ({}, lambda model, error: model.split_dfa_step(error, ref.LEARNING_RATE, 0)),
            "windowed": ({"slow_update_every": 2}, lambda model, error: model.windowed_dfa_step(
                error, ref.LEARNING_RATE, 0, defer_apply=True)),
        }
        for name in ("DC2", "RG2x0.1"):
            for label, (extra, step) in steps.items():
                with self.subTest(case=name, step=label):
                    model, expected = self.fast_step_from_same_state(dict(ref.CASES[name][0], **extra), step)
                    for layer, weights in zip(model.trained_layers(), expected):
                        mask = layer.ephemeral_mask.unsqueeze(0).expand_as(weights)
                        torch.testing.assert_close(layer.per_sample_weights.data[mask], weights[mask],
                                                   rtol=0, atol=0)

    def test_invalid_combinations_are_rejected(self):
        for kwargs in (dict(fast_layers=[3]), dict(decouple_fast_feedback=[2], fast_write_from_readout=[2]),
                       dict(fast_layers=[0, 1], decouple_fast_feedback=[2]),
                       dict(fast_write_from_readout=[1]), dict(fast_write_from_readout=[2], layer_norm=True),
                       dict(decouple_fast_feedback=[2], updater="bptt")):
            with self.subTest(**{k: str(v) for k, v in kwargs.items()}), self.assertRaises(ValueError):
                build(**kwargs)
        for argv in (["--decouple_fast_feedback", "2", "--grad_norm_clip", "1"],
                     ["--decouple_fast_feedback", "2", "--updater", "backprop"],
                     ["--fast_write_from_readout", "2", "--updater", "backprop"],
                     ["--fast_layers", "0,1", "--model_type", "rnn"],
                     ["--fast_layers", "a"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(argv)

    def test_build_model_reads_the_flags_from_a_config(self):
        args = parse_args(["--num_layers", "3", "--hidden_size", "4", "--decouple_fast_feedback", "2",
                           "--fast_layers", "1,2", "--fast_write_from_readout_scale", "0.3"])
        config = dict(vars(args), n_hidden=4, n_layers=3, charset_size=4)
        with contextlib.redirect_stdout(io.StringIO()):
            model = build_model(config, ref.CHARSET, len(ref.CHARSET))
        self.assertEqual(model.fast_layers, [1, 2])
        self.assertFalse(model.linear_layers[0].ephemeral_mask.any())
        self.assertIsNotNone(model.linear_layers[2].fast_feedback_weights)
        self.assertEqual(model.fast_write_from_readout_scale, 0.3)
        # a config from before the port (no such keys) builds the plain model
        for key in ("fast_layers", "decouple_fast_feedback", "fast_write_from_readout",
                    "fast_write_from_readout_scale"):
            config.pop(key)
        with contextlib.redirect_stdout(io.StringIO()):
            model = build_model(config, ref.CHARSET, len(ref.CHARSET))
        self.assertIsNone(model.fast_layers)
        self.assertIsNone(model.linear_layers[2].fast_feedback_weights)


if __name__ == "__main__":
    unittest.main()
