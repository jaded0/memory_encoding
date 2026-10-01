"""--layer_norm together with --dfa_fprime.

The f' term is gelu' at each trunk layer's pre-activation. --layer_norm sits after the GELU, so
the exact derivative would also carry the LayerNorm Jacobian; --dfa_fprime does not include it.
These tests pin what the pair does today: the error is (B e) * gelu'(a) exactly as without
LayerNorm (the LayerNorm Jacobian is deliberately not applied), the pair trains to finite
values, and each flag changes training. If the Jacobian is ever added, the pinned-error test
below is the one that should fail."""
import unittest

import torch
import torch.nn.functional as F

from ephemeral_model import trunk_layer_norm
from tests.test_dfa_fprime import BATCHES, CHARSET, build, layers, record, reference_error, run


class LayerNormWithFprimeTest(unittest.TestCase):
    def check_error_is_gelu_prime_only(self, model_type):
        model = build(model_type, layer_norm=True)
        feedback = {name: layer.feedback_weights.clone() if layer.feedback_weights is not None else None
                    for name, layer in layers(model).items()}
        onehot = F.one_hot(BATCHES[0], len(CHARSET)).float()
        steps = record(model, BATCHES[0])
        for t, step in enumerate(steps):
            output_error = torch.softmax(step["i2o"]["pre"], 1) - onehot[:, t + 1]
            # The next layer sees the LayerNormed activations.
            torch.testing.assert_close(step["linear_layers.1"]["input"],
                                       trunk_layer_norm(F.gelu(step["linear_layers.0"]["pre"])))
            for name, values in step.items():
                expected = reference_error(name, feedback[name], output_error, values["pre"])
                torch.testing.assert_close(values["projected"], expected, rtol=1e-5, atol=1e-7,
                                           msg=f"{model_type} step {t} {name}")

    def test_ephemeral_error_is_gelu_prime_without_the_layer_norm_jacobian(self):
        self.check_error_is_gelu_prime_only("ephemeral")

    def test_simple_rnn_error_is_gelu_prime_without_the_layer_norm_jacobian(self):
        self.check_error_is_gelu_prime_only("rnn")

    def test_the_pair_trains_finite_and_each_flag_changes_training(self):
        for model_type in ("ephemeral", "rnn"):
            with self.subTest(model_type):
                both = build(model_type, layer_norm=True)
                losses = run(both, BATCHES * 3)
                self.assertTrue(all(torch.isfinite(torch.as_tensor(loss)).all() for loss in losses))
                self.assertTrue(all(torch.isfinite(tensor).all() for tensor in both.state_dict().values()
                                    if tensor.is_floating_point()))
                self.assertNotEqual(losses, run(build(model_type, layer_norm=True, dfa_fprime=False), BATCHES * 3))
                self.assertNotEqual(losses, run(build(model_type, layer_norm=False), BATCHES * 3))

    def test_fused_step_matches_unfused_with_the_pair(self):
        unfused, fused = build(layer_norm=True), build(layer_norm=True)
        fused.enable_fused_update(compile=False)
        self.assertEqual(run(fused), run(unfused))
        for a, b in zip(fused.state_dict().values(), unfused.state_dict().values()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
