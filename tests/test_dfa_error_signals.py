"""Pins the error tensors each layer uses on the DFA path, so an aliasing mistake fails a test.

Every layer receives one shared output_error. i2o and self_grad keep that same object for their
bias update, which runs after other layers' updates. An in-place change to it anywhere between
autograd.grad and the last bias update (for example a clamp_ or mul_ in one layer's populate or
update) would change what a later layer uses. For each step, these tests check the tensors at
populate time and again at update time against values computed independently from the step's
outputs. The golden trace has --self_grad 0, so the self_grad term is covered only here.
"""
import contextlib
import io
import unittest

import torch

from ephemeral_model import EphemeralRNN
from reproducibility import seed_everything
from train import train

LEARNING_RATE = 0.1
SEQUENCE = torch.tensor([[0, 1, 2, 3, 1], [3, 2, 0, 1, 2]])  # 4 DFA steps per call
CHARSET = list("abcd")


def build_model():
    with contextlib.redirect_stdout(io.StringIO()):
        return EphemeralRNN(8, 4, 4, 2, CHARSET, unit_norm_weights=False, weight_clamp=0, updater="dfa",
                            plasticity=3.0, batch_size=2, forget_rate=0.25, ephemeral_fraction=0.5,
                            enable_recurrence=True)


def named_layers(model):
    return {**{f"linear_layers.{i}": layer for i, layer in enumerate(model.linear_layers)},
            "i2h": model.i2h, "i2o": model.i2o, "self_grad": model.self_grad}


def instrument(model):
    """Records, per step, each layer's inputs to populate and to apply_update, and the model outputs."""
    steps = []

    def record_output(name):
        def hook(_module, _inputs, output):
            if name == "i2o":  # i2o runs before self_grad, once per step
                steps.append({"layers": {}})
            steps[-1][name] = output.detach().clone()  # returns None: the output is not replaced
        return hook

    output_hooks = [getattr(model, name).register_forward_hook(record_output(name)) for name in ("i2o", "self_grad")]
    for name, layer in named_layers(model).items():
        populate, update = layer.populate_dfa_gradients, layer.apply_update

        def record_populate(error_signal, populate=populate, name=name, layer=layer):
            received = error_signal.clone()
            populate(error_signal)
            steps[-1]["layers"][name] = {
                "received": received,
                "input": layer.in_traces.data.clone(),
                "projected": layer._last_projected_error.clone(),
                "grad": layer.per_sample_weights.grad.clone(),
                "shares_received_object": layer._last_projected_error is error_signal,
            }

        def record_update(learning_rate, update_clamp, state, update=update, name=name, layer=layer):
            record = steps[-1]["layers"][name]
            record["projected_at_update"] = layer._last_projected_error.clone()
            record["grad_at_update"] = layer.per_sample_weights.grad.clone()
            record["bias_before"] = layer.bias.detach().clone()
            update(learning_rate, update_clamp, state)
            record["bias_after"] = layer.bias.detach().clone()

        layer.populate_dfa_gradients, layer.apply_update = record_populate, record_update
    return steps, output_hooks


def run(self_grad):
    torch.set_num_threads(1)
    seed_everything(4242, deterministic=True)
    model = build_model()
    steps, _hooks = instrument(model)
    config = {"updater": "dfa", "criterion": torch.nn.CrossEntropyLoss(reduction="none"), "input_mode": "last_two",
              "pe_matrix": None, "self_grad": self_grad, "learning_rate": LEARNING_RATE,
              "ephemeral_update_clamp": 0, "plasticity": 3.0}
    onehot = torch.nn.functional.one_hot(SEQUENCE, len(CHARSET)).float()
    state = {"training_instance": 0}
    for _call in range(2):  # the second call starts from the first call's weights
        with contextlib.redirect_stdout(io.StringIO()):
            train(SEQUENCE, onehot, model, config, state)
    return model, steps, onehot


class DfaErrorSignalTest(unittest.TestCase):
    def check(self, self_grad):
        model, steps, onehot = run(self_grad)
        layers = named_layers(model)
        steps_per_call = SEQUENCE.shape[1] - 1
        self.assertEqual(len(steps), 2 * steps_per_call)
        exact = dict(rtol=0, atol=0)
        for index, step in enumerate(steps):
            with self.subTest(step=index):
                target = onehot[:, index % steps_per_call + 1]
                # d CrossEntropy(x, t)/dx = softmax(x) - t for a one-hot t, per sequence.
                expected = torch.softmax(step["i2o"], dim=1) - target
                if self_grad > 0:
                    expected = expected + torch.clamp(step["self_grad"], -self_grad, self_grad)

                # Every EphemeralLinear layer, i2h included, is populated each step.
                self.assertEqual(set(step["layers"]), set(layers))
                shared = step["layers"]["i2o"]["received"]
                torch.testing.assert_close(shared, expected, rtol=1e-6, atol=1e-7)
                for name, record in step["layers"].items():
                    layer = layers[name]
                    # Every layer received the identical tensor, so no populate call changed it.
                    torch.testing.assert_close(record["received"], shared, **exact, msg=name)
                    if layer.is_last_layer:
                        # Pins today's aliasing: the last layers keep the shared object itself.
                        # A copy would give the same values; if one is made on purpose, change this.
                        self.assertTrue(record["shares_received_object"], f"{name}: no longer the shared object")
                        projected = shared
                    else:
                        self.assertFalse(record["shares_received_object"], name)
                        projected = shared @ layer.feedback_weights.detach()
                    torch.testing.assert_close(record["projected"], projected, **exact, msg=name)
                    gradient = projected.unsqueeze(2) * record["input"].unsqueeze(1)
                    torch.testing.assert_close(record["grad"], gradient, **exact, msg=name)
                    # Unchanged by the time this layer's update runs, after earlier layers' updates.
                    torch.testing.assert_close(record["projected_at_update"], projected, **exact, msg=name)
                    torch.testing.assert_close(record["grad_at_update"], gradient, **exact, msg=name)
                    torch.testing.assert_close(record["bias_after"] - record["bias_before"],
                                               -LEARNING_RATE * projected.mean(dim=0), rtol=1e-5, atol=1e-7, msg=name)
        # The run did move the model, so the checks above are not on a trivial path.
        self.assertGreater(model.i2o.per_sample_weights.abs().sum().item(), 0)
        return steps

    def test_every_layer_uses_the_unmodified_output_error(self):
        self.check(self_grad=0.0)

    def test_self_grad_term_is_added_once_for_every_layer(self):
        steps = self.check(self_grad=0.05)
        # The clamp binds somewhere, so the self_grad term is not simply the raw output.
        self.assertTrue(any((step["self_grad"].abs() > 0.05).any() for step in steps))


if __name__ == "__main__":
    unittest.main()
