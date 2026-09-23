"""Unit checks of single training mechanics that the golden trace pins only indirectly."""
import contextlib
import io
import unittest

import torch

from ephemeral_model import EphemeralLinear, EphemeralRNN, SimpleRNN
from reproducibility import seed_everything
from train import train

CHARSET = list("abcd")
SEQUENCE = torch.tensor([[0, 1, 2, 3, 1], [3, 2, 0, 1, 2]])  # 4 steps per call
HIDDEN = 4


def build_rnn(model_type, updater, enable_recurrence=True):
    seed_everything(99, deterministic=True)
    with contextlib.redirect_stdout(io.StringIO()):
        if model_type == "rnn":
            return SimpleRNN(2 * len(CHARSET), HIDDEN, len(CHARSET), 1, enable_recurrence=enable_recurrence)
        return EphemeralRNN(2 * len(CHARSET), HIDDEN, len(CHARSET), 1, CHARSET, unit_norm_weights=False,
                            weight_clamp=0, updater=updater, plasticity=3.0, batch_size=2, forget_rate=0.25,
                            ephemeral_fraction=0.5, enable_recurrence=enable_recurrence)


def run_one_sequence(model, updater, optimizer=None):
    config = {"updater": updater, "criterion": torch.nn.CrossEntropyLoss(reduction="none"),
              "input_mode": "last_two", "pe_matrix": None, "self_grad": 0.0, "learning_rate": 0.1,
              "ephemeral_update_clamp": 0, "grad_norm_clip": 0, "plasticity": 3.0}
    onehot = torch.nn.functional.one_hot(SEQUENCE, len(CHARSET)).float()
    with contextlib.redirect_stdout(io.StringIO()):
        train(SEQUENCE, onehot, model, config, {"training_instance": 0}, optimizer=optimizer)


def build_layer(batch_size=3, unit_norm_weights=True):
    with contextlib.redirect_stdout(io.StringIO()):
        return EphemeralLinear(5, 4, list("abcd"), unit_norm_weights=unit_norm_weights, batch_size=batch_size,
                               ephemeral_fraction=0.5)


class UnitNormWeightsTest(unittest.TestCase):
    def test_each_sequence_is_normalised_on_its_own(self):
        torch.manual_seed(0)
        layer = build_layer()
        weights = torch.randn(3, 4, 5) * torch.tensor([0.1, 1.0, 300.0]).view(3, 1, 1)
        layer.per_sample_weights.data = weights.clone()
        layer._apply_regularization()
        normalised = layer.per_sample_weights.data
        for index in range(3):
            torch.testing.assert_close(torch.linalg.vector_norm(normalised[index]), torch.tensor(1.0),
                                       rtol=1e-5, atol=1e-5)
            # Only a rescaling of that sequence's own slice.
            torch.testing.assert_close(normalised[index], weights[index] / (weights[index].norm() + 1e-6))

        # One sequence's result does not depend on the others.
        other = build_layer()
        changed = weights.clone()
        changed[1:] *= 1e3
        other.per_sample_weights.data = changed
        other._apply_regularization()
        torch.testing.assert_close(other.per_sample_weights.data[0], normalised[0], rtol=0, atol=0)

    def test_off_leaves_weights_alone(self):
        layer = build_layer(unit_norm_weights=False)
        weights = torch.randn(3, 4, 5)
        layer.per_sample_weights.data = weights.clone()
        layer._apply_regularization()
        torch.testing.assert_close(layer.per_sample_weights.data, weights, rtol=0, atol=0)


class ElmanLayoutTest(unittest.TestCase):
    """h_t = tanh(i2h(combined)), y_t = i2o(h_t): the output reads this step's hidden state."""

    def test_i2h_learns_under_per_step_backprop_and_dfa(self):
        # Before the Elman layout, i2h fed only the next step, whose input is detached under
        # DFA and per-step backprop, so i2h never got a gradient or an update there.
        for updater in ("dfa", "backprop"):
            with self.subTest(model="ephemeral", updater=updater):
                model = build_rnn("ephemeral", updater)
                grads, update = [], model.i2h.apply_update

                def record(learning_rate, update_clamp, state):
                    grad = model.i2h.per_sample_weights.grad
                    grads.append(None if grad is None else grad.norm().item())
                    update(learning_rate, update_clamp, state)

                model.i2h.apply_update = record
                bias_before = model.i2h.bias.detach().clone()
                run_one_sequence(model, updater)
                self.assertEqual(len(grads), SEQUENCE.shape[1] - 1)  # updated every step
                self.assertTrue(all(g is not None for g in grads), grads)
                self.assertGreater(max(grads), 0.0)
                self.assertGreater(model.i2h.per_sample_weights.abs().sum().item(), 0.0)  # started at zero
                self.assertFalse(torch.equal(model.i2h.bias.detach(), bias_before))

        with self.subTest(model="rnn", updater="backprop"):
            model = build_rnn("rnn", "backprop")
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            grads, step = [], optimizer.step

            def record_step(*args, **kwargs):
                grad = model.i2h.weight.grad
                grads.append(None if grad is None else grad.norm().item())
                return step(*args, **kwargs)

            optimizer.step = record_step
            weight_before = model.i2h.weight.detach().clone()
            run_one_sequence(model, "backprop", optimizer=optimizer)
            self.assertEqual(len(grads), SEQUENCE.shape[1] - 1)
            self.assertTrue(all(g is not None and g > 0 for g in grads), grads)
            self.assertFalse(torch.equal(model.i2h.weight.detach(), weight_before))

    def test_dfa_leaves_the_rnn_baseline_untouched(self):
        # The DFA branch only updates an EphemeralRNN (README Updaters table); pinned here so the
        # i2h test above is not read as covering SimpleRNN under DFA.
        model = build_rnn("rnn", "dfa")
        before = {name: value.clone() for name, value in model.state_dict().items()}
        run_one_sequence(model, "dfa")
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0, msg=name)

    def test_output_heads_read_the_hidden_state(self):
        model = build_rnn("ephemeral", "dfa")
        for head in (model.i2o, model.self_grad):
            self.assertEqual(head.in_features, HIDDEN)
        self.assertEqual((model.i2h.in_features, model.i2h.out_features), (2 * len(CHARSET) + HIDDEN, HIDDEN))
        self.assertTrue(model.i2h.ephemeral_mask.any())  # a hidden layer, not a last layer
        self.assertFalse(model.i2h.is_last_layer)

    def test_recurrence_off_keeps_the_output_path_and_feeds_back_zeros(self):
        for model_type in ("ephemeral", "rnn"):
            with self.subTest(model=model_type):
                on, off = build_rnn(model_type, "dfa", True), build_rnn(model_type, "dfa", False)
                off.load_state_dict(on.state_dict())
                with torch.no_grad():
                    i2h = on.i2h.per_sample_weights if model_type == "ephemeral" else on.i2h.weight
                    i2h.normal_()  # so the output visibly depends on i2h
                    if model_type == "ephemeral":
                        on.i2o.per_sample_weights.normal_()  # starts at zero, which would hide i2h
                    off.load_state_dict(on.state_dict())
                    x, h = torch.randn(2, 2 * len(CHARSET)), torch.randn(2, HIDDEN)
                    out_on, hidden_on, _ = on(x, h)
                    out_off, hidden_off, _ = off(x, h)
                    torch.testing.assert_close(out_off, out_on, rtol=0, atol=0)
                    self.assertTrue(torch.equal(hidden_off, torch.zeros_like(h)))
                    self.assertGreater(hidden_on.abs().sum().item(), 0)
                    # y_t = i2o(tanh(i2h(combined))) with h_t not fed back.
                    torch.testing.assert_close(out_off, off.i2o(hidden_on), rtol=0, atol=0)
                    i2h.zero_()
                    self.assertFalse(torch.equal(on(x, h)[0], out_on))


if __name__ == "__main__":
    unittest.main()
