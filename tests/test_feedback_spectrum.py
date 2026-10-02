"""Gain-matched aligned-feedback initializers and their checkpoint/CLI behavior."""
import contextlib
import io
import os
import tempfile
import unittest

import torch

import train as train_module
from ephemeral_model import EphemeralRNN
from reproducibility import seed_everything
from tests.test_feedback_init import BATCH, CHARSET, HIDDEN, LAYERS, V, alignment, build, feedback
import tests.test_heldout as heldout_tests
from utils import load_checkpoint


SPECTRUM_INITS = ("aligned_spectrum_matched", "random_spectrum_of_J")


class FeedbackSpectrumTest(unittest.TestCase):
    def test_aligned_spectrum_matched_has_random_singular_values_and_aligned_cosine(self):
        random = build("random")
        aligned = build("aligned")
        matched = build("aligned_spectrum_matched")
        jacobians = random.jacobian_feedback_matrices()
        for jac, random_b, matched_b in zip(jacobians, feedback(random), feedback(matched)):
            torch.testing.assert_close(torch.linalg.svdvals(matched_b), torch.linalg.svdvals(random_b),
                                       rtol=2e-5, atol=2e-6)
            u_j, _, vh_j = torch.linalg.svd(jac, full_matrices=False)
            u_m, _, vh_m = torch.linalg.svd(matched_b, full_matrices=False)
            # SVD signs are arbitrary, so compare absolute overlaps. Every matched singular
            # direction is J's corresponding left and right singular direction.
            torch.testing.assert_close((u_j.T @ u_m).abs(), torch.eye(u_j.shape[1]),
                                       rtol=0, atol=3e-5)
            torch.testing.assert_close((vh_j @ vh_m.T).abs(), torch.eye(vh_j.shape[0]),
                                       rtol=0, atol=3e-5)

        aligned_cos = alignment(aligned)
        matched_cos = alignment(matched)
        for layer in range(LAYERS):
            self.assertGreater(matched_cos[layer]["cos"], 0.3, layer)
            self.assertAlmostEqual(matched_cos[layer]["cos"], aligned_cos[layer]["cos"], delta=0.03)

    def test_random_spectrum_of_j_has_j_shape_random_alignment_and_random_norm(self):
        random = build("random")
        aligned = build("aligned")
        controlled = build("random_spectrum_of_J")
        for random_b, aligned_b, controlled_b in zip(feedback(random), feedback(aligned), feedback(controlled)):
            random_norm = random_b.norm()
            self.assertAlmostEqual(float(controlled_b.norm()), float(random_norm), places=5)
            aligned_shape = torch.linalg.svdvals(aligned_b) / aligned_b.norm()
            controlled_shape = torch.linalg.svdvals(controlled_b) / controlled_b.norm()
            torch.testing.assert_close(controlled_shape, aligned_shape, rtol=2e-5, atol=2e-6)
        for layer, result in alignment(controlled).items():
            if layer < LAYERS:
                self.assertLess(abs(result["cos"]), 0.2, layer)

    def test_only_feedback_matrices_differ_from_the_random_arm(self):
        random = build("random")
        for init in SPECTRUM_INITS:
            with self.subTest(init=init):
                model = build(init)
                for name, value in random.state_dict().items():
                    if "feedback_weights" not in name:
                        self.assertTrue(torch.equal(value, model.state_dict()[name]), name)

    def test_scales_apply_after_the_spectrum_transform(self):
        for init in SPECTRUM_INITS:
            with self.subTest(init=init):
                base, scaled = build(init), build(init, -0.5)
                for base_b, scaled_b in zip(feedback(base), feedback(scaled)):
                    torch.testing.assert_close(scaled_b, -0.5 * base_b, rtol=0, atol=0)


class SpectrumTrainingFlowTest(unittest.TestCase):
    def test_cli_choices_defaults_and_fprime_refusals(self):
        self.assertEqual(train_module.parse_args([]).feedback_init, "random")
        for init in SPECTRUM_INITS:
            self.assertEqual(train_module.parse_args(["--feedback_init", init]).feedback_init, init)
            with self.assertRaises(ValueError):
                build(init, dfa_fprime=True)
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                train_module.parse_args(["--feedback_init", init, "--dfa_fprime", "true"])
            for incompatible in (("--model_type", "rnn"), ("--updater", "backprop")):
                with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                    train_module.parse_args(["--feedback_init", init, *incompatible])

        with contextlib.redirect_stdout(io.StringIO()):
            unsupported = EphemeralRNN(V + 2, HIDDEN, V, LAYERS, CHARSET, updater="dfa",
                                       batch_size=BATCH)
        for init in SPECTRUM_INITS:
            with self.subTest(init=init), self.assertRaises(ValueError):
                unsupported.set_feedback(init)

    def test_checkpoint_round_trip_and_resume_flag_refusal_for_both_controls(self):
        for init, other in ((SPECTRUM_INITS[0], SPECTRUM_INITS[1]),
                            (SPECTRUM_INITS[1], SPECTRUM_INITS[0])):
            with self.subTest(init=init), tempfile.TemporaryDirectory() as directory:
                heldout_tests.run_main("--feedback_init", init, checkpoint_dir=directory)
                path = os.path.join(directory, "latest_checkpoint.pth")
                saved = torch.load(path, weights_only=False)
                self.assertEqual(saved["config"]["feedback_init"], init)

                config = {**saved["config"]}
                charset = train_module.initialize_charset(config["dataset"])
                seed_everything(99, deterministic=True)
                with contextlib.redirect_stdout(io.StringIO()):
                    fresh = train_module.build_model(config, charset[0], charset[3])
                    fresh, *_ = load_checkpoint(path, fresh, config)
                for name, value in saved["model_state_dict"].items():
                    if name.endswith(".feedback_weights"):
                        self.assertTrue(torch.equal(value, fresh.state_dict()[name]), name)

                with self.assertRaisesRegex(RuntimeError, "configuration mismatch"):
                    heldout_tests.run_main("--feedback_init", other, "--n_iters", "6",
                                           "--resume", "true", checkpoint_dir=directory)
                log = heldout_tests.run_main("--feedback_init", init, "--n_iters", "6",
                                             "--resume", "true", checkpoint_dir=directory)
                self.assertIn("resumed, starting from iter: 5", log)


if __name__ == "__main__":
    unittest.main()
