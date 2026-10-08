"""Opt-in interventions for the slow-drift stability predictions."""
import contextlib
import io
import unittest

import torch
import torch.nn.functional as F

from ephemeral_model import (EphemeralRNN, dfa_output_error, normalize_slow_gradient,
                             normalized_readout_error)
from train import parse_args


class LabelSmoothingTest(unittest.TestCase):
    def test_error_is_softmax_minus_smoothed_target(self):
        logits = torch.tensor([[2.0, -1.0, 0.5], [1.0, 3.0, -2.0]])
        target = F.one_hot(torch.tensor([0, 2]), 3).float()
        criterion = torch.nn.CrossEntropyLoss(reduction="none")
        _, error = dfa_output_error(logits.clone(), target, criterion, label_smoothing=0.2)
        expected_target = target * 0.8 + 0.2 / 3
        torch.testing.assert_close(error, torch.softmax(logits, 1) - expected_target)

    def test_padding_row_stays_zero(self):
        logits = torch.randn(2, 3)
        target = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        criterion = torch.nn.CrossEntropyLoss(reduction="none")
        _, error = dfa_output_error(logits, target, criterion, label_smoothing=0.2)
        self.assertEqual(float(error[0].abs().max()), 0.0)


class NormalizedReadoutTest(unittest.TestCase):
    def test_scales_each_row_by_width_over_squared_norm(self):
        error = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        inputs = torch.tensor([[1.0, 1.0, 1.0], [2.0, 0.0, 0.0]])
        actual = normalized_readout_error(error, inputs, True, eps=0)
        expected = error * torch.tensor([[1 / 3], [1 / 4]])
        torch.testing.assert_close(actual, expected)
        self.assertIs(normalized_readout_error(error, inputs, False), error)

    def test_slow_normalization_leaves_fast_entries_unchanged(self):
        gradient = torch.arange(16, dtype=torch.float).reshape(2, 2, 4)
        inputs = torch.tensor([[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]])
        mask = torch.tensor([[True, False, True, False], [False, True, False, True]])
        actual = normalize_slow_gradient(gradient, inputs, mask, True, eps=0)
        expanded = mask.unsqueeze(0).expand_as(gradient)
        torch.testing.assert_close(actual[expanded], gradient[expanded])
        scale = torch.tensor([1.0, 0.25]).view(2, 1, 1).expand_as(gradient)
        torch.testing.assert_close(actual[~expanded], (gradient * scale)[~expanded])

    def test_fused_and_unfused_steps_match(self):
        def build(fused, **options):
            torch.manual_seed(4)
            with contextlib.redirect_stdout(io.StringIO()):
                model = EphemeralRNN(4, 5, 4, 2, list("abcd"), updater="dfa", plasticity=3,
                                     batch_size=2, ephemeral_fraction=0.5, **options)
            if fused:
                model.enable_fused_update(compile=False)
            return model

        for options in ({"readout_nlms": True}, {"slow_nlms": True},
                        {"readout_slow_lr_scale": 0.3}, {"trunk_slow_lr_scale": 0.3},
                        {"slow_nlms": True, "trunk_slow_lr_scale": 0.5}):
            with self.subTest(**options):
                unfused, fused = build(False, **options), build(True, **options)
                x, hidden = torch.eye(4)[:2], torch.zeros(2, 5)
                target = F.one_hot(torch.tensor([1, 2]), 4).float()
                criterion = torch.nn.CrossEntropyLoss(reduction="none")
                for model in (unfused, fused):
                    model.start_sequence_wipe()
                    output, _ = model(x, hidden)
                    _, error = dfa_output_error(output, target, criterion)
                    if model.fused_layer_step:
                        model.fused_dfa_step(error, 0.1, 0)
                    else:
                        model.clear_dfa_gradients()
                        for layer in model.trained_layers():
                            layer.populate_dfa_gradients(error)
                            layer.apply_update(0.1, 0, {})
                        model.apply_forget_step()
                for left, right in zip(unfused.state_dict().values(), fused.state_dict().values()):
                    if left.dtype.is_floating_point:
                        torch.testing.assert_close(left, right, rtol=0, atol=0)
                    else:
                        self.assertTrue(torch.equal(left, right))

    def test_full_nlms_changes_slow_but_not_fast_weight_step(self):
        def build(slow_nlms):
            torch.manual_seed(7)
            with contextlib.redirect_stdout(io.StringIO()):
                model = EphemeralRNN(4, 5, 4, 2, list("abcd"), updater="dfa", plasticity=3,
                                     batch_size=2, ephemeral_fraction=0.5,
                                     slow_nlms=slow_nlms)
            model.enable_fused_update(compile=False)
            return model

        baseline, normalized = build(False), build(True)
        x, hidden = torch.eye(4)[:2] * 2, torch.zeros(2, 5)
        target = F.one_hot(torch.tensor([1, 2]), 4).float()
        criterion = torch.nn.CrossEntropyLoss(reduction="none")
        for model in (baseline, normalized):
            model.start_sequence_wipe()
            output, _ = model(x, hidden)
            _, error = dfa_output_error(output, target, criterion)
            model.fused_dfa_step(error, 0.1, 0)

        saw_changed_slow = False
        for plain, nlms in zip(baseline.trained_layers(), normalized.trained_layers()):
            mask = plain.ephemeral_mask.unsqueeze(0).expand_as(plain.per_sample_weights)
            torch.testing.assert_close(
                plain.per_sample_weights[mask], nlms.per_sample_weights[mask], rtol=0, atol=0)
            if not torch.equal(plain.per_sample_weights[~mask], nlms.per_sample_weights[~mask]):
                saw_changed_slow = True
        self.assertTrue(saw_changed_slow)


class SlowLearningRateScaleTest(unittest.TestCase):
    """--readout_slow_lr_scale / --trunk_slow_lr_scale: the scaled group's slow step equals the
    step at learning_rate * scale; fast entries and the other group are exactly unchanged."""

    def step(self, learning_rate, **options):
        torch.manual_seed(11)
        with contextlib.redirect_stdout(io.StringIO()):
            model = EphemeralRNN(4, 5, 4, 2, list("abcd"), updater="dfa", plasticity=3,
                                 batch_size=2, ephemeral_fraction=0.5, **options)
        model.enable_fused_update(compile=False)
        x, hidden = torch.eye(4)[:2] * 2, torch.zeros(2, 5)
        target = F.one_hot(torch.tensor([1, 2]), 4).float()
        model.start_sequence_wipe()
        output, _ = model(x, hidden)
        _, error = dfa_output_error(output, target, torch.nn.CrossEntropyLoss(reduction="none"))
        model.fused_dfa_step(error, learning_rate, 0)
        return model

    def assert_layer(self, actual, expected, entries, exact):
        mask = actual.ephemeral_mask.unsqueeze(0).expand_as(actual.per_sample_weights)
        selected = {"fast": mask, "slow": ~mask}[entries]
        tolerance = {"rtol": 0, "atol": 0} if exact else {}
        torch.testing.assert_close(actual.per_sample_weights[selected],
                                   expected.per_sample_weights[selected], **tolerance)
        if entries == "slow" and actual.bias is not None:
            torch.testing.assert_close(actual.bias, expected.bias, **tolerance)

    def test_readout_scale_changes_only_the_readout(self):
        baseline, slow = self.step(0.1), self.step(0.03)
        scaled = self.step(0.1, readout_slow_lr_scale=0.3)
        self.assert_layer(scaled.i2o, slow.i2o, "slow", exact=False)
        self.assertFalse(torch.equal(scaled.i2o.per_sample_weights, baseline.i2o.per_sample_weights))
        for layer, plain in zip(scaled.trained_layers(), baseline.trained_layers()):
            if layer is not scaled.i2o:
                self.assert_layer(layer, plain, "slow", exact=True)
                self.assert_layer(layer, plain, "fast", exact=True)

    def test_trunk_scale_changes_only_trunk_slow_entries(self):
        baseline, slow = self.step(0.1), self.step(0.03)
        scaled = self.step(0.1, trunk_slow_lr_scale=0.3)
        self.assert_layer(scaled.i2o, baseline.i2o, "slow", exact=True)
        for layer, plain, low in zip(scaled.trained_layers(), baseline.trained_layers(),
                                     slow.trained_layers()):
            if layer is not scaled.i2o:
                self.assert_layer(layer, plain, "fast", exact=True)
                self.assert_layer(layer, low, "slow", exact=False)

    def test_scale_one_is_the_same_object(self):
        gradient = torch.randn(2, 3, 4)
        mask = torch.rand(3, 4) > 0.5
        self.assertIs(normalize_slow_gradient(gradient, torch.randn(2, 4), mask, False), gradient)


class CommandLineTest(unittest.TestCase):
    def test_defaults_and_validation(self):
        args = parse_args([])
        self.assertFalse(args.readout_nlms)
        self.assertFalse(args.slow_nlms)
        self.assertEqual(args.label_smoothing, 0.0)
        self.assertEqual((args.readout_slow_lr_scale, args.trunk_slow_lr_scale), (1.0, 1.0))
        for refused in (["--readout_slow_lr_scale", "0.3", "--model_type", "rnn"],
                        ["--trunk_slow_lr_scale", "-1"],
                        ["--readout_slow_lr_scale", "0.3", "--grad_norm_clip", "1"]):
            with self.subTest(refused=refused), self.assertRaises(SystemExit), \
                    contextlib.redirect_stderr(io.StringIO()):
                parse_args(refused)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            parse_args(["--label_smoothing", "1"])
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            parse_args(["--readout_nlms", "true", "--model_type", "rnn"])
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            parse_args(["--readout_nlms", "true", "--slow_nlms", "true"])
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            parse_args(["--slow_nlms", "true", "--grad_norm_clip", "1",
                        "--fused_update", "true"])
        for alternate in (["--slow_update_every", "2"],
                          ["--fast_backward_per_forward", "2"]):
            with self.subTest(alternate=alternate), self.assertRaises(SystemExit), \
                    contextlib.redirect_stderr(io.StringIO()):
                parse_args(["--slow_nlms", "true", "--grad_norm_clip", "1", *alternate])


if __name__ == "__main__":
    unittest.main()
