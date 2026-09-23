"""Unit checks of single training mechanics that the golden trace pins only indirectly."""
import contextlib
import io
import unittest

import torch

from ephemeral_model import EphemeralLinear


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


if __name__ == "__main__":
    unittest.main()
