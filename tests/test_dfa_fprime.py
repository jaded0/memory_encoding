"""--dfa_fprime: each non-output layer's DFA error is (B_l e) * f'(a_l) (Nøkland 2016).

The reference is computed independently of the code under test: the pre-activations and layer
inputs are captured with forward hooks, the output error is softmax(logits) - target, and f' is
taken by autograd through F.gelu (the trunk) and torch.tanh (i2h). Off, the default, is pinned
byte for byte by the golden traces (test_characterization.py)."""
import contextlib
import copy
import io
import unittest

import torch
import torch.nn.functional as F

from ephemeral_model import EphemeralRNN, SimpleRNN, dfa_activation_derivative
from reproducibility import seed_everything
from train import parse_args, train

CHARSET = list("abcd")
BATCHES = (torch.tensor([[0, 1, 2, 3, 1], [3, 2, 0, 1, 2], [1, 1, 3, 0, 2]]),
           torch.tensor([[2, 0, 3, 1, 2], [1, 3, 2, 0, 1], [0, 0, 1, 2, 3]]))
LEARNING_RATE = 0.3


def build(model_type="ephemeral", dfa_fprime=True, **options):
    seed_everything(5, deterministic=True)
    with contextlib.redirect_stdout(io.StringIO()):
        if model_type == "rnn":
            return SimpleRNN(len(CHARSET), 4, len(CHARSET), 2, updater="dfa", dfa_fprime=dfa_fprime, **options)
        return EphemeralRNN(len(CHARSET), 4, len(CHARSET), 2, CHARSET, unit_norm_weights=False, weight_clamp=0,
                            updater="dfa", plasticity=3.0, batch_size=3, forget_rate=0.25,
                            ephemeral_fraction=0.5, enable_recurrence=True, dfa_fprime=dfa_fprime, **options)


def run(model, batches=BATCHES, grad_norm_clip=0):
    config = {"updater": "dfa", "criterion": torch.nn.CrossEntropyLoss(reduction="none"),
              "input_mode": "last_one", "pe_matrix": None, "learning_rate": LEARNING_RATE,
              "ephemeral_update_clamp": 0, "grad_norm_clip": grad_norm_clip}
    losses = []
    for batch in batches:
        with contextlib.redirect_stdout(io.StringIO()):
            _, loss, *_ = train(batch, F.one_hot(batch, len(CHARSET)).float(), model, config,
                                {"training_instance": 0, "log_norms_now": False})
        losses.append(loss)
    return losses


def layers(model):
    return {**{f"linear_layers.{i}": layer for i, layer in enumerate(model.linear_layers)},
            "i2h": model.i2h, "i2o": model.i2o}


def reference_error(name, feedback, output_error, pre_activation):
    """(B e) * f'(a) by autograd through the nonlinearity the model applies; e for i2o."""
    if name == "i2o":
        return output_error
    activation = torch.tanh if name == "i2h" else F.gelu
    a = pre_activation.clone().requires_grad_(True)
    return torch.autograd.grad(activation(a), a, grad_outputs=output_error @ feedback)[0]


def record(model, batch):
    """Per step: each layer's input, pre-activation, and (after populate) weight gradient and
    projected error, plus the logits."""
    steps = []
    handles = []
    for name, layer in layers(model).items():
        def hook(_module, inputs, output, name=name):
            if name == "linear_layers.0":
                steps.append({})
            steps[-1][name] = {"input": inputs[0].detach().clone(), "pre": output.detach().clone()}
        handles.append(layer.register_forward_hook(hook))

        def populate(error_signal, original=layer.populate_dfa_gradients, name=name, layer=layer):
            original(error_signal)
            grad = layer.per_sample_weights.grad if isinstance(model, EphemeralRNN) else layer.weight.grad
            steps[-1][name].update(grad=grad.clone(), projected=layer._last_projected_error.clone())
        layer.populate_dfa_gradients = populate
    run(model, [batch])
    for handle in handles:
        handle.remove()
    return steps


class ActivationDerivativeTest(unittest.TestCase):
    def test_matches_autograd(self):
        x = torch.linspace(-9, 9, 1001, dtype=torch.float64)
        for name, fn in (("gelu", F.gelu), ("tanh", torch.tanh)):
            with self.subTest(name):
                a = x.clone().requires_grad_(True)
                expected = torch.autograd.grad(fn(a).sum(), a)[0]
                torch.testing.assert_close(dfa_activation_derivative(x, name), expected, rtol=1e-12, atol=1e-12)
        with self.assertRaises(ValueError):
            dfa_activation_derivative(x, "relu")

    def test_models_apply_the_nonlinearities_the_derivatives_assume(self):
        for model_type in ("ephemeral", "rnn"):
            with self.subTest(model_type):
                model = build(model_type)
                self.assertEqual([layer.activation for layer in model.linear_layers], ["gelu", "gelu"])
                self.assertEqual((model.i2h.activation, model.i2o.activation), ("tanh", None))
                steps = record(model, BATCHES[0])
                for step in steps:
                    torch.testing.assert_close(step["linear_layers.1"]["input"], F.gelu(step["linear_layers.0"]["pre"]))
                    torch.testing.assert_close(step["i2h"]["input"], F.gelu(step["linear_layers.1"]["pre"]))
                # The next step's input hidden is tanh(i2h's pre-activation).
                torch.testing.assert_close(steps[1]["linear_layers.0"]["input"][:, len(CHARSET):],
                                           torch.tanh(steps[0]["i2h"]["pre"]))


class DfaFprimeGradientTest(unittest.TestCase):
    def check(self, model_type, **options):
        model = build(model_type, **options)
        feedback = {name: layer.feedback_weights.clone() if layer.feedback_weights is not None else None
                    for name, layer in layers(model).items()}
        onehot = F.one_hot(BATCHES[0], len(CHARSET)).float()
        # The i2o forward hook sees the logits; the DFA error for cross entropy is softmax - target.
        steps = record(model, BATCHES[0])
        self.assertEqual(len(steps), BATCHES[0].shape[1] - 1)
        for t, step in enumerate(steps):
            output_error = torch.softmax(step["i2o"]["pre"], 1) - onehot[:, t + 1]
            for name, values in step.items():
                expected = reference_error(name, feedback[name], output_error, values["pre"])
                torch.testing.assert_close(values["projected"], expected, rtol=1e-5, atol=1e-7, msg=f"{t} {name}")
                per_sequence = expected.unsqueeze(2) * values["input"].unsqueeze(1)
                if model_type == "rnn":
                    per_sequence = per_sequence.mean(0)
                torch.testing.assert_close(values["grad"], per_sequence, rtol=1e-5, atol=1e-7, msg=f"{t} {name}")
                if name != "i2o":  # not vacuous: f' changed the error
                    self.assertFalse(torch.allclose(values["projected"], output_error @ feedback[name]))

    def test_ephemeral(self):
        self.check("ephemeral")

    def test_ephemeral_with_residual_and_output_tanh(self):
        self.check("ephemeral", residual_connection=True, output_tanh=True)

    def test_simple_rnn(self):
        self.check("rnn")

    def test_simple_rnn_with_residual_and_output_tanh(self):
        self.check("rnn", residual_connection=True, output_tanh=True)

    def test_on_changes_training_and_off_matches_the_default(self):
        for model_type in ("ephemeral", "rnn"):
            with self.subTest(model_type):
                default, off, on = build(model_type, dfa_fprime=False), build(model_type, dfa_fprime=False), build(model_type)
                self.assertEqual(run(off), run(default))
                self.assertNotEqual(run(on), run(build(model_type, dfa_fprime=False)))
                for a, b in zip(off.state_dict().values(), default.state_dict().values()):
                    self.assertTrue(torch.equal(a, b))

    def test_rnn_records_pre_activations_only_with_the_flag(self):
        model = build("rnn", dfa_fprime=False)
        run(model, BATCHES[:1])
        self.assertTrue(all(layer.out_traces is None for layer in model.dfa_layers()))


class FusedDfaFprimeTest(unittest.TestCase):
    @staticmethod
    def state(model):
        return {f"{i}.{name}": tensor.detach().clone()
                for i, layer in enumerate(model.trained_layers())
                for name, tensor in (("weights", layer.per_sample_weights), ("bias", layer.bias))}

    def test_eager_fused_step_is_bit_identical_to_the_unfused_step(self):
        unfused, fused = build(), build()
        fused.enable_fused_update(compile=False)
        self.assertEqual(run(fused), run(unfused))
        for name, tensor in self.state(unfused).items():
            torch.testing.assert_close(self.state(fused)[name], tensor, rtol=0, atol=0, msg=name)

    def test_with_grad_norm_clip_matches_to_rounding(self):
        unfused, fused = build(), build()
        fused.enable_fused_update(compile=False)
        run(unfused, grad_norm_clip=0.05)
        run(fused, grad_norm_clip=0.05)
        for name, tensor in self.state(unfused).items():
            torch.testing.assert_close(self.state(fused)[name], tensor, rtol=1e-5, atol=1e-7, msg=name)

    def test_compiled_fused_step_matches_to_rounding(self):
        unfused, fused = build(), build()
        fused.enable_fused_update(compile=True)
        try:
            run(fused)
        except Exception as exc:  # no C++ compiler for Inductor's CPU backend, for example
            self.skipTest(f"torch.compile unavailable here: {type(exc).__name__}: {exc}")
        run(unfused)
        for name, tensor in self.state(unfused).items():
            torch.testing.assert_close(self.state(fused)[name], tensor, rtol=1e-5, atol=1e-6, msg=name)


class DfaFprimeFlagTest(unittest.TestCase):
    def test_default_off_and_dfa_only(self):
        self.assertFalse(parse_args([]).dfa_fprime)
        self.assertTrue(parse_args(["--dfa_fprime", "true"]).dfa_fprime)
        self.assertTrue(parse_args(["--dfa_fprime", "true", "--model_type", "rnn"]).dfa_fprime)
        for updater in ("backprop", "bptt"):
            with self.subTest(updater), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(["--dfa_fprime", "true", "--updater", updater])


if __name__ == "__main__":
    unittest.main()
