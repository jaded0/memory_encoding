"""Unit checks of single training mechanics that the golden trace pins only indirectly."""
import contextlib
import io
import unittest

import torch

from ephemeral_model import EphemeralLinear, EphemeralRNN, SimpleRNN, init_feedback_weights
from reproducibility import seed_everything
from train import train

CHARSET = list("abcd")
SEQUENCE = torch.tensor([[0, 1, 2, 3, 1], [3, 2, 0, 1, 2]])  # 4 steps per call
HIDDEN = 4
LEARNING_RATE = 0.1


def build_rnn(model_type, updater, enable_recurrence=True, num_layers=1):
    seed_everything(99, deterministic=True)
    with contextlib.redirect_stdout(io.StringIO()):
        if model_type == "rnn":
            return SimpleRNN(2 * len(CHARSET), HIDDEN, len(CHARSET), num_layers, enable_recurrence=enable_recurrence,
                             updater=updater)
        return EphemeralRNN(2 * len(CHARSET), HIDDEN, len(CHARSET), 1, CHARSET, unit_norm_weights=False,
                            weight_clamp=0, updater=updater, plasticity=3.0, batch_size=2, forget_rate=0.25,
                            ephemeral_fraction=0.5, enable_recurrence=enable_recurrence)


def run_one_sequence(model, updater, optimizer=None, grad_norm_clip=0):
    config = {"updater": updater, "criterion": torch.nn.CrossEntropyLoss(reduction="none"),
              "input_mode": "last_two", "pe_matrix": None, "learning_rate": LEARNING_RATE,
              "ephemeral_update_clamp": 0, "grad_norm_clip": grad_norm_clip, "plasticity": 3.0}
    onehot = torch.nn.functional.one_hot(SEQUENCE, len(CHARSET)).float()
    with contextlib.redirect_stdout(io.StringIO()):
        train(SEQUENCE, onehot, model, config, {"training_instance": 0}, optimizer=optimizer)


def build_layer(batch_size=3, unit_norm_weights=True):
    with contextlib.redirect_stdout(io.StringIO()):
        return EphemeralLinear(5, 4, list("abcd"), unit_norm_weights=unit_norm_weights, batch_size=batch_size,
                               ephemeral_fraction=0.5)


class WeightInitializationTest(unittest.TestCase):
    def test_slow_weights_use_default_init_and_fast_weights_start_at_zero(self):
        torch.manual_seed(0)
        layer = build_layer()
        mask = layer.ephemeral_mask

        self.assertTrue(mask.any())
        self.assertTrue((~mask).any())
        for weights in layer.per_sample_weights:
            torch.testing.assert_close(weights[~mask], layer.weight[~mask], rtol=0, atol=0)
            torch.testing.assert_close(weights[mask], torch.zeros_like(weights[mask]), rtol=0, atol=0)

        torch.testing.assert_close(layer.per_sample_weights[0], layer.per_sample_weights[1], rtol=0, atol=0)

    def test_last_layer_has_only_default_initialized_slow_weights(self):
        torch.manual_seed(0)
        with contextlib.redirect_stdout(io.StringIO()):
            layer = EphemeralLinear(5, 4, CHARSET, batch_size=2, is_last_layer=True)

        self.assertFalse(layer.ephemeral_mask.any())
        for weights in layer.per_sample_weights:
            torch.testing.assert_close(weights, layer.weight, rtol=0, atol=0)


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


class ForkedLayoutTest(unittest.TestCase):
    """The shared trunk forks into a recurrent state head and a current-emission head."""

    def test_i2h_gets_direct_dfa_but_not_per_step_backprop(self):
        for updater in ("dfa", "backprop"):
            with self.subTest(updater=updater):
                model = build_rnn("ephemeral", updater)
                grads, update = [], model.i2h.apply_update

                def record(learning_rate, update_clamp, state):
                    grad = model.i2h.per_sample_weights.grad
                    grads.append(None if grad is None else grad.norm().item())
                    update(learning_rate, update_clamp, state)

                model.i2h.apply_update = record
                weight_before = model.i2h.per_sample_weights.detach().clone()
                bias_before = model.i2h.bias.detach().clone()
                run_one_sequence(model, updater)
                self.assertEqual(len(grads), SEQUENCE.shape[1] - 1)
                if updater == "dfa":
                    self.assertTrue(all(g is not None and g > 0 for g in grads), grads)
                    self.assertFalse(torch.equal(model.i2h.per_sample_weights.detach(), weight_before))
                    self.assertFalse(torch.equal(model.i2h.bias.detach(), bias_before))
                else:
                    self.assertTrue(all(g is None for g in grads), grads)
                    self.assertTrue(torch.equal(model.i2h.per_sample_weights.detach(), weight_before))
                    self.assertTrue(torch.equal(model.i2h.bias.detach(), bias_before))

    def test_bptt_trains_the_state_head_through_future_steps(self):
        for model_type in ("ephemeral", "rnn"):
            with self.subTest(model=model_type):
                model = build_rnn(model_type, "bptt")
                optimizer = None if model_type == "ephemeral" else torch.optim.SGD(model.parameters(), lr=0.1)
                weights = model.i2h.per_sample_weights if model_type == "ephemeral" else model.i2h.weight
                before = weights.detach().clone()
                run_one_sequence(model, "bptt", optimizer=optimizer)
                self.assertFalse(torch.equal(weights.detach(), before))

    def test_output_and_state_heads_fork_from_the_shared_trunk(self):
        for model_type in ("ephemeral", "rnn"):
            with self.subTest(model=model_type):
                model = build_rnn(model_type, "dfa")
                if model_type == "ephemeral":
                    inner = 2 * len(CHARSET) + HIDDEN
                    self.assertEqual(model.i2o.in_features, inner)
                    self.assertEqual((model.i2h.in_features, model.i2h.out_features), (inner, HIDDEN))
                    self.assertTrue(model.i2h.ephemeral_mask.any())
                    self.assertFalse(model.i2h.is_last_layer)

                head_inputs = {}

                def record_head_input(name):
                    def record(_module, inputs):
                        head_inputs[name] = inputs[0].detach().clone()
                    return record

                state_hook = model.i2h.register_forward_pre_hook(record_head_input("state"))
                output_hook = model.i2o.register_forward_pre_hook(record_head_input("output"))
                model(torch.randn(2, 2 * len(CHARSET)), torch.randn(2, HIDDEN))
                state_hook.remove()
                output_hook.remove()
                torch.testing.assert_close(head_inputs["output"], head_inputs["state"], rtol=0, atol=0)

    def test_model_types_share_trunk_dimensions_activation_and_residual(self):
        input_size, hidden_size, layers = 11, 5, 2
        simple = SimpleRNN(input_size, hidden_size, 4, layers, residual_connection=True)
        with contextlib.redirect_stdout(io.StringIO()):
            ephemeral = EphemeralRNN(input_size, hidden_size, 4, layers, CHARSET,
                                     batch_size=2, residual_connection=True)
        inner = input_size + hidden_size
        for model in (simple, ephemeral):
            self.assertEqual([(layer.in_features, layer.out_features)
                              for layer in model.linear_layers], [(inner, inner)] * layers)
            self.assertEqual((model.i2h.in_features, model.i2o.in_features), (inner, inner))

        simple_no_residual = SimpleRNN(input_size, hidden_size, 4, layers)
        simple_no_residual.load_state_dict(simple.state_dict())
        x, h = torch.randn(2, input_size), torch.randn(2, hidden_size)
        trunk = torch.cat((x, h), dim=1)
        for layer in simple.linear_layers:
            trunk = torch.nn.functional.gelu(layer(trunk))
        output_without_residual, _ = simple_no_residual(x, h)
        expected_without_residual = torch.nn.functional.linear(
            trunk, simple_no_residual.i2o.weight, simple_no_residual.i2o.bias)
        expected_with_residual = torch.nn.functional.linear(
            trunk + torch.cat((x, h), dim=1), simple.i2o.weight, simple.i2o.bias)
        torch.testing.assert_close(output_without_residual, expected_without_residual)
        torch.testing.assert_close(simple(x, h)[0], expected_with_residual)

    def test_recurrence_off_keeps_the_output_path_and_feeds_back_zeros(self):
        for model_type in ("ephemeral", "rnn"):
            with self.subTest(model=model_type):
                on, off = build_rnn(model_type, "dfa", True), build_rnn(model_type, "dfa", False)
                off.load_state_dict(on.state_dict())
                with torch.no_grad():
                    i2h = on.i2h.per_sample_weights if model_type == "ephemeral" else on.i2h.weight
                    i2h.normal_()
                    if model_type == "ephemeral":
                        on.i2o.per_sample_weights.normal_()
                    off.load_state_dict(on.state_dict())
                    x, h = torch.randn(2, 2 * len(CHARSET)), torch.randn(2, HIDDEN)
                    out_on, hidden_on = on(x, h)
                    out_off, hidden_off = off(x, h)
                    torch.testing.assert_close(out_off, out_on, rtol=0, atol=0)
                    self.assertTrue(torch.equal(hidden_off, torch.zeros_like(h)))
                    self.assertGreater(hidden_on.abs().sum().item(), 0)
                    # The current output does not read the recurrent-state fork.
                    i2h.zero_()
                    torch.testing.assert_close(on(x, h)[0], out_on, rtol=0, atol=0)


class SimpleRnnDfaTest(unittest.TestCase):
    """--model_type rnn --updater dfa: the ephemeral model's DFA without ephemeral weights."""

    def instrument(self, model):
        """Records, per layer and step, what populate and update saw, and each step's output."""
        steps = []
        model.i2o.register_forward_hook(lambda _m, _i, output: steps.append({"output": output.detach().clone()}))
        for name, layer in zip(("linear_layers.0", "linear_layers.1", "i2h", "i2o"), model.dfa_layers()):
            populate, update = layer.populate_dfa_gradients, layer.apply_dfa_update

            def record_populate(error_signal, populate=populate, name=name, layer=layer):
                received = error_signal.clone()
                populate(error_signal)
                steps[-1][name] = {"received": received, "input": layer.in_traces.clone()}

            def record_update(learning_rate, update=update, name=name, layer=layer):
                record = steps[-1][name]
                record.update(weight_grad=layer.weight.grad.clone(), bias_grad=layer.bias.grad.clone(),
                              weight=layer.weight.detach().clone(), bias=layer.bias.detach().clone())
                update(learning_rate)
                record.update(weight_after=layer.weight.detach().clone(), bias_after=layer.bias.detach().clone())

            layer.populate_dfa_gradients, layer.apply_dfa_update = record_populate, record_update
        return steps

    def test_every_layer_including_i2h_is_updated_every_step(self):
        model = build_rnn("rnn", "dfa", num_layers=2)
        feedback = {name: layer.feedback_weights for name, layer in
                    zip(("linear_layers.0", "linear_layers.1", "i2h"), model.dfa_layers())}
        steps = self.instrument(model)
        run_one_sequence(model, "dfa")
        self.assertEqual(len(steps), SEQUENCE.shape[1] - 1)
        onehot = torch.nn.functional.one_hot(SEQUENCE, len(CHARSET)).float()
        for index, step in enumerate(steps):
            with self.subTest(step=index):
                self.assertEqual(set(step) - {"output"}, {"linear_layers.0", "linear_layers.1", "i2h", "i2o"})
                # Per sequence, d CrossEntropy/d output = softmax(output) - target (reduction 'none').
                error = torch.softmax(step["output"], dim=1) - onehot[:, index + 1]
                for name, record in ((k, v) for k, v in step.items() if k != "output"):
                    torch.testing.assert_close(record["received"], error, rtol=1e-6, atol=1e-7, msg=name)
                    projected = error if name == "i2o" else error @ feedback[name]
                    # EphemeralLinear's per-sequence outer product, averaged over the shared weight's batch.
                    gradient = torch.einsum("bo,bi->oi", projected, record["input"]) / SEQUENCE.shape[0]
                    torch.testing.assert_close(record["weight_grad"], gradient, rtol=1e-5, atol=1e-7, msg=name)
                    torch.testing.assert_close(record["bias_grad"], projected.mean(dim=0), rtol=1e-6, atol=1e-7, msg=name)
                    torch.testing.assert_close(record["weight_after"] - record["weight"], -LEARNING_RATE * gradient,
                                               rtol=1e-4, atol=1e-7, msg=name)
                    torch.testing.assert_close(record["bias_after"] - record["bias"], -LEARNING_RATE * projected.mean(dim=0),
                                               rtol=1e-4, atol=1e-7, msg=name)
                    self.assertGreater(record["weight_grad"].abs().sum().item(), 0, name)
                    self.assertFalse(torch.equal(record["weight_after"], record["weight"]), name)
                    self.assertFalse(torch.equal(record["bias_after"], record["bias"]), name)
        # The feedback matrices are fixed.
        for name, layer in zip(("linear_layers.0", "linear_layers.1", "i2h"), model.dfa_layers()):
            self.assertIs(layer.feedback_weights, feedback[name])

    def test_feedback_matrices_are_dfa_only_state_and_leave_the_init_alone(self):
        dfa, backprop = build_rnn("rnn", "dfa"), build_rnn("rnn", "backprop")
        extra = set(dfa.state_dict()) - set(backprop.state_dict())
        self.assertEqual(extra, {"linear_layers.0.feedback_weights", "i2h.feedback_weights"})
        for name, value in backprop.state_dict().items():
            torch.testing.assert_close(dfa.state_dict()[name], value, rtol=0, atol=0, msg=name)
        # Drawn like EphemeralLinear's feedback_weights, [vocab, out], after every layer.
        seed_everything(99, deterministic=True)
        with contextlib.redirect_stdout(io.StringIO()):
            SimpleRNN(2 * len(CHARSET), HIDDEN, len(CHARSET), 1)
        for layer in (dfa.linear_layers[0], dfa.i2h):
            torch.testing.assert_close(layer.feedback_weights,
                                       init_feedback_weights(len(CHARSET), layer.out_features),
                                       rtol=0, atol=0)
        self.assertIsNone(dfa.i2o.feedback_weights)

    def test_grad_norm_clip_clips_the_dfa_gradients(self):
        model = build_rnn("rnn", "dfa", num_layers=2)
        steps = self.instrument(model)
        run_one_sequence(model, "dfa", grad_norm_clip=0.01)
        for step in steps:
            grads = [g for key, record in step.items() if key != "output" for g in (record["weight_grad"], record["bias_grad"])]
            total = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads]))
            self.assertLessEqual(total.item(), 0.01 * (1 + 1e-5))
            self.assertGreater(total.item(), 0.01 * (1 - 1e-5))  # the clip binds
            for key, record in ((k, v) for k, v in step.items() if k != "output"):
                torch.testing.assert_close(record["weight_after"] - record["weight"], -LEARNING_RATE * record["weight_grad"],
                                           rtol=1e-4, atol=1e-7, msg=key)

    def test_normalization_and_weight_clamp_apply_under_every_updater(self):
        for updater in ("dfa", "backprop", "bptt"):
            with self.subTest(updater=updater):
                seed_everything(99, deterministic=True)
                model = SimpleRNN(2 * len(CHARSET), HIDDEN, len(CHARSET), 1,
                                  updater=updater, unit_norm_weights=True, weight_clamp=0.1)
                optimizer = (None if updater == "dfa" else
                             torch.optim.SGD(model.parameters(), lr=LEARNING_RATE))
                run_one_sequence(model, updater, optimizer=optimizer)
                for layer in model.dfa_layers():
                    self.assertLessEqual(layer.weight.norm().item(), 1.000001)
                    self.assertLessEqual(layer.weight.abs().max().item(), 0.100001)

    def test_rnn_dfa_gradient_is_the_batch_mean_of_per_sequence_outer_products(self):
        model = build_rnn("rnn", "dfa", num_layers=2)
        batch = SEQUENCE.shape[0]
        onehot = torch.nn.functional.one_hot(SEQUENCE, len(CHARSET)).float()
        output, _ = model(torch.cat([onehot[:, 1], onehot[:, 0]], dim=1), model.initHidden(batch))
        output_error = torch.softmax(output, 1) - onehot[:, 2]
        for layer in model.dfa_layers():
            layer.populate_dfa_gradients(output_error)
            projected = output_error if layer.is_last_layer else output_error @ layer.feedback_weights
            per_sequence = [torch.outer(projected[b], layer.in_traces[b]) for b in range(batch)]
            torch.testing.assert_close(layer.weight.grad, torch.stack(per_sequence).mean(0), rtol=1e-6, atol=1e-8)
            torch.testing.assert_close(layer.bias.grad, projected.mean(0), rtol=1e-6, atol=1e-8)

    def test_ephemeral_weight_clamp_applies_under_every_updater(self):
        for updater in ("dfa", "backprop", "bptt"):
            with self.subTest(updater=updater):
                seed_everything(99, deterministic=True)
                with contextlib.redirect_stdout(io.StringIO()):
                    model = EphemeralRNN(2 * len(CHARSET), HIDDEN, len(CHARSET), 1, CHARSET,
                                         unit_norm_weights=False, weight_clamp=0.05, updater=updater,
                                         plasticity=3.0, batch_size=2, forget_rate=0.25,
                                         ephemeral_fraction=0.5)
                initial = max(layer.per_sample_weights.abs().max().item() for layer in model.trained_layers())
                self.assertGreater(initial, 0.05)  # the clamp has something to bind
                run_one_sequence(model, updater)
                for layer in model.trained_layers():
                    self.assertLessEqual(layer.per_sample_weights.abs().max().item(), 0.050001)

    def test_rnn_without_dfa_state_refuses_dfa(self):
        model = build_rnn("rnn", "backprop")
        with self.assertRaises(ValueError):
            run_one_sequence(model, "dfa")


if __name__ == "__main__":
    unittest.main()
