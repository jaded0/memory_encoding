"""--feedback_init / --feedback_scale / --alignment_log_every: aligned, scaled and anti-aligned DFA feedback."""
import contextlib
import io
import os
import tempfile
import unittest

import torch
import torch.nn.functional as F

import train as train_module
from ephemeral_model import EphemeralRNN
from feedback_alignment import measure_alignment
from reproducibility import seed_everything
from utils import load_checkpoint
import tests.test_heldout as heldout_tests

CHARSET = "abcdef"
V = len(CHARSET)
BATCH, HIDDEN, LAYERS = 4, 48, 3
LAYERS_IN_RUN = 1  # tests.test_heldout.run_main trains one layer


def build(init="random", scale=1.0, seed=7, recurrence=False, **options):
    seed_everything(seed, deterministic=True)
    with contextlib.redirect_stdout(io.StringIO()):
        model = EphemeralRNN(V, HIDDEN, V, LAYERS, CHARSET, updater="dfa", batch_size=BATCH, plasticity=50.0,
                             forget_rate=0.05, ephemeral_fraction=0.1, enable_recurrence=recurrence, **options)
    model.set_feedback(init, scale)
    return model


def episode(seed=1, steps=6):
    generator = torch.Generator().manual_seed(seed)
    onehot = F.one_hot(torch.randint(V, (BATCH, steps), generator=generator), V).float()
    inputs = [onehot[:, i] for i in range(steps - 1)]
    targets = [onehot[:, i + 1] for i in range(steps - 1)]
    return inputs, targets


def alignment(model, **kwargs):
    inputs, targets = episode(**kwargs)
    return measure_alignment(model, inputs, targets, 1e-3, 0)


def feedback(model):
    return [layer.feedback_weights.detach().clone() for layer in [*model.linear_layers, model.i2h]]


class DefaultsUnchangedTest(unittest.TestCase):
    def test_random_is_a_no_op_and_default_flags_are_random_scale_one(self):
        args = train_module.parse_args([])
        self.assertEqual((args.feedback_init, args.feedback_scale, args.alignment_log_every), ("random", 1.0, 0))
        seed_everything(7, deterministic=True)
        with contextlib.redirect_stdout(io.StringIO()):
            reference = EphemeralRNN(V, HIDDEN, V, LAYERS, CHARSET, updater="dfa", batch_size=BATCH,
                                     plasticity=50.0, forget_rate=0.05, ephemeral_fraction=0.1)
        model = build("random", recurrence=True)
        for name, value in reference.state_dict().items():
            self.assertTrue(torch.equal(value, model.state_dict()[name]), name)

    def test_random_with_a_scale_is_refused(self):
        with self.assertRaises(ValueError):
            build("random", 2.0)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            train_module.parse_args(["--feedback_scale", "2"])

    def test_the_random_matrices_do_not_depend_on_the_flag(self):
        random, scaled, aligned = build("random"), build("scaled", 3.0), build("aligned")
        for b0, b1 in zip(feedback(random), feedback(scaled)):
            torch.testing.assert_close(b1, 3.0 * b0, rtol=0, atol=0)
        # Every layer built after the feedback matrices (slow weights, masks) is identical too.
        for name, value in random.state_dict().items():
            if "feedback_weights" not in name:
                self.assertTrue(torch.equal(value, aligned.state_dict()[name]), name)


class AlignedFeedbackTest(unittest.TestCase):
    def test_aligned_has_the_random_norm_and_anti_is_its_negative(self):
        random, aligned, anti = build("random"), build("aligned"), build("aligned", -1.0)
        for br, ba, bn in zip(feedback(random), feedback(aligned), feedback(anti)):
            self.assertAlmostEqual(float(ba.norm()), float(br.norm()), places=4)
            torch.testing.assert_close(bn, -ba, rtol=0, atol=0)
        self.assertFalse(torch.allclose(feedback(random)[0], feedback(aligned)[0]))

    def test_aligned_feedback_is_the_mean_jacobian(self):
        model = build("random")
        jac = model.jacobian_feedback_matrices(n_rows=2 * BATCH)
        # Independent check for the last trunk layer: finite differences of the logits in a_l
        # for one calibration batch (J is exact for GELU + i2o; compare the mean over the batch).
        layer = model.linear_layers[-1]
        batch = model._calibration_inputs(BATCH)[0]
        model.start_sequence_wipe()
        hidden = model.initHidden(BATCH)
        outs = []
        hook = layer.register_forward_hook(lambda m, i, o: outs.append(o))
        with torch.no_grad():
            model(batch, hidden)
        hook.remove()
        pre = outs[0].detach().clone()
        w_out = model.i2o.per_sample_weights.detach()
        bias = model.i2o.bias.detach()

        def logits(a):
            return torch.bmm(w_out, F.gelu(a).unsqueeze(2)).squeeze(2) + bias

        expected = torch.autograd.functional.jacobian(lambda a: logits(a).sum(0), pre)  # [V, B, out]
        # per-row Jacobians: row b of the batch only affects row b of the logits
        per_row = torch.stack([expected[:, b] for b in range(BATCH)])
        self.assertEqual(per_row.shape, (BATCH, V, pre.shape[1]))
        # the calibration cycles the charset, so the batch mean over 2 batches of one-hots is a mean over
        # all 6 characters (not an exact match for one batch); the last layer's J is f'(a) * W_out, so
        # their cosine with the single-batch mean is high.
        single = per_row.mean(0)
        cos = F.cosine_similarity(single.flatten(), jac[-2].flatten(), dim=0)
        self.assertGreater(float(cos), 0.9)

    def test_aligned_projected_error_has_positive_cosine_to_the_true_gradient_at_init(self):
        random, aligned, anti = (alignment(build(i, s)) for i, s in (("random", 1.0), ("aligned", 1.0),
                                                                     ("aligned", -1.0)))
        for k in range(LAYERS):
            self.assertGreater(aligned[k]["cos"], 0.3, k)
            self.assertGreater(aligned[k]["cos"], random[k]["cos"] + 0.2, k)
            self.assertLess(anti[k]["cos"], -0.3, k)
            self.assertAlmostEqual(aligned[k]["cos"], -anti[k]["cos"], places=3)
        self.assertGreater(aligned[LAYERS - 1]["cos"], 0.8)  # last trunk layer: J = f'(a) W_out up to the mean

    def test_a_deeper_layer_matches_the_true_gradient_of_a_linear_readout_exactly_when_inputs_agree(self):
        # With one calibration character repeated, the mean Jacobian is the Jacobian of that input,
        # so at that input the projected error equals the true gradient (cosine 1) for every layer.
        model = build("random")
        batch = F.one_hot(torch.zeros(BATCH, dtype=torch.long), V).float()
        model._calibration_inputs = lambda n, batch=batch: [batch]
        model.set_feedback("aligned")
        model.start_sequence_wipe()
        with torch.enable_grad():
            outs = []
            hooks = [layer.register_forward_hook(lambda m, i, o: outs.append(o.requires_grad_(True)))
                     for layer in model.linear_layers]
            output, _ = model(batch, model.initHidden(BATCH))
            target = F.one_hot(torch.ones(BATCH, dtype=torch.long), V).float()
            loss = torch.nn.CrossEntropyLoss(reduction="none")(output, target)
            grads = torch.autograd.grad(loss.sum(), [output] + outs)
        for hook in hooks:
            hook.remove()
        projected, _ = model.dfa_step_errors(grads[0].detach(), 0)
        for k in range(LAYERS):
            torch.testing.assert_close(F.cosine_similarity(projected[k], grads[1 + k], dim=1),
                                       torch.ones(BATCH), atol=1e-4, rtol=0)

    def test_measure_alignment_leaves_the_model_untouched(self):
        model = build("aligned", recurrence=True)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        alignment(model)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)

    def test_aligned_with_fprime_and_unsupported_inputs_are_refused(self):
        with self.assertRaises(ValueError):
            build("aligned", dfa_fprime=True)
        with contextlib.redirect_stdout(io.StringIO()):
            model = EphemeralRNN(V + 2, HIDDEN, V, LAYERS, CHARSET, updater="dfa", batch_size=BATCH)
        with self.assertRaises(ValueError):
            model.set_feedback("aligned")


class TrainingFlowTest(unittest.TestCase):
    def test_cli_defaults_and_choices(self):
        args = train_module.parse_args(["--feedback_init", "aligned", "--feedback_scale", "-1"])
        self.assertEqual((args.feedback_init, args.feedback_scale), ("aligned", -1.0))
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            train_module.parse_args(["--feedback_init", "bogus"])

    def test_checkpoint_round_trip_and_alignment_log(self):
        with tempfile.TemporaryDirectory() as directory:
            log = heldout_tests.run_main("--feedback_init", "aligned", "--feedback_scale", "-1",
                                         "--alignment_log_every", "2", checkpoint_dir=directory)
            self.assertEqual(log.count("GAIN layer=0 "), 1)
            self.assertEqual(log.count("GAIN layer=i2h "), 1)
            self.assertIn("b_fro=", log)
            self.assertIn("b_top5=", log)
            self.assertIn("b_stable_rank=", log)
            self.assertIn("j_stable_rank=", log)
            self.assertIn("ALIGN iter 0 ", log)
            self.assertIn("ALIGN iter 2 ", log)
            path = os.path.join(directory, "latest_checkpoint.pth")
            saved = torch.load(path, weights_only=False)
            self.assertEqual(saved["config"]["feedback_init"], "aligned")
            self.assertEqual(saved["config"]["feedback_scale"], -1.0)
            # B is in the state dict and DFA never changes it: a model built with other random matrices
            # gets exactly the saved ones back from load_checkpoint.
            config = {**saved["config"]}
            charset = train_module.initialize_charset(config["dataset"])
            seed_everything(99, deterministic=True)
            with contextlib.redirect_stdout(io.StringIO()):
                fresh = train_module.build_model(config, charset[0], charset[3])
                fresh, *_ = load_checkpoint(path, fresh, config)
            names = [n for n in saved["model_state_dict"] if n.endswith(".feedback_weights")]
            self.assertEqual(len(names), LAYERS_IN_RUN + 2)  # trunk, i2h, i2o
            for name in names:
                self.assertTrue(torch.equal(saved["model_state_dict"][name], fresh.state_dict()[name]), name)
            alphas = [layer.feedback_weights for layer in fresh.linear_layers]
            self.assertTrue(all(float(a.norm()) > 0 for a in alphas))
            # a resume with a different flag is refused; the same flag continues
            with self.assertRaisesRegex(RuntimeError, "configuration mismatch"):
                heldout_tests.run_main("--feedback_init", "scaled", "--feedback_scale", "2", "--n_iters", "6",
                                       "--resume", "true", checkpoint_dir=directory)
            log = heldout_tests.run_main("--feedback_init", "aligned", "--feedback_scale", "-1", "--n_iters", "6",
                                         "--resume", "true", checkpoint_dir=directory)
            self.assertIn("resumed, starting from iter: 5", log)




class ReadoutScaleTest(unittest.TestCase):
    def test_scale_readout_scales_only_i2o(self):
        base, scaled = build("random"), build("random")
        scaled.scale_readout(0.25)
        for name, value in base.state_dict().items():
            expected = value * 0.25 if name in ("i2o.weight", "i2o.per_sample_weights") else value
            torch.testing.assert_close(scaled.state_dict()[name], expected, rtol=0, atol=0)
        self.assertEqual(train_module.parse_args([]).readout_init_scale, 1.0)


if __name__ == "__main__":
    unittest.main()
