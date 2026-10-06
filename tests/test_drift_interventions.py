"""Opt-in interventions for the slow-drift stability predictions."""
import contextlib
import io
import unittest

import torch
import torch.nn.functional as F

from ephemeral_model import EphemeralRNN, dfa_output_error, normalized_readout_error
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

    def test_fused_and_unfused_steps_match(self):
        def build(fused):
            torch.manual_seed(4)
            with contextlib.redirect_stdout(io.StringIO()):
                model = EphemeralRNN(4, 5, 4, 2, list("abcd"), updater="dfa", plasticity=3,
                                     batch_size=2, ephemeral_fraction=0.5,
                                     readout_nlms=True)
            if fused:
                model.enable_fused_update(compile=False)
            return model

        unfused, fused = build(False), build(True)
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


class CommandLineTest(unittest.TestCase):
    def test_defaults_and_validation(self):
        args = parse_args([])
        self.assertFalse(args.readout_nlms)
        self.assertEqual(args.label_smoothing, 0.0)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            parse_args(["--label_smoothing", "1"])
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            parse_args(["--readout_nlms", "true", "--model_type", "rnn"])


if __name__ == "__main__":
    unittest.main()
