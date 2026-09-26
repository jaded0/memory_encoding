"""--fused_update: EphemeralRNN.fused_dfa_step must be the unfused DFA step, only faster.

dfa_layer_step is built from the same helpers as apply_update and apply_forget_step, so run
eagerly it must reproduce the unfused step bit for bit. Compiled, only the rounding may differ."""
import contextlib
import io
import itertools
import unittest

import torch
import torch.nn.functional as F

from ephemeral_model import EphemeralRNN
from reproducibility import seed_everything
from train import parse_args, train

CHARSET = list("abcd")
BATCHES = (torch.tensor([[0, 1, 2, 3, 1], [3, 2, 0, 1, 2], [1, 1, 3, 0, 2]]),
           torch.tensor([[2, 0, 3, 1, 2], [1, 3, 2, 0, 1], [0, 0, 1, 2, 3]]))


def build(updater="dfa", unit_norm_weights=False, weight_clamp=0):
    seed_everything(11, deterministic=True)
    with contextlib.redirect_stdout(io.StringIO()):
        return EphemeralRNN(len(CHARSET), 4, len(CHARSET), 2, CHARSET, unit_norm_weights=unit_norm_weights,
                            weight_clamp=weight_clamp, updater=updater, plasticity=3.0, batch_size=3,
                            forget_rate=0.25, ephemeral_fraction=0.5, enable_recurrence=True)


def run(model, update_clamp=0, grad_norm_clip=0, log_norms_now=False):
    """Two consecutive batches through the real train(); returns their losses."""
    config = {"updater": "dfa", "criterion": torch.nn.CrossEntropyLoss(reduction="none"),
              "input_mode": "last_one", "pe_matrix": None, "learning_rate": 0.3,
              "ephemeral_update_clamp": update_clamp, "grad_norm_clip": grad_norm_clip}
    losses = []
    for batch in BATCHES:
        with contextlib.redirect_stdout(io.StringIO()):
            _, loss, *_ = train(batch, F.one_hot(batch, len(CHARSET)).float(), model, config,
                                {"training_instance": 0, "log_norms_now": log_norms_now})
        losses.append(loss)
    return losses


def state(model):
    return {f"{i}.{name}": tensor.detach().clone()
            for i, layer in enumerate(model.trained_layers())
            for name, tensor in (("weights", layer.per_sample_weights), ("bias", layer.bias))}


SETTINGS = [dict(update_clamp=clamp, weight_clamp=weight_clamp, unit_norm_weights=unit_norm)
            for clamp, weight_clamp, unit_norm in itertools.product((0, 0.05), (0, 0.2), (False, True))]


class FusedUpdateTest(unittest.TestCase):
    def assert_same(self, actual, expected, **tolerance):
        for name, tensor in expected.items():
            torch.testing.assert_close(actual[name], tensor, msg=name, **tolerance)

    def test_eager_fused_step_is_bit_identical_to_the_unfused_step(self):
        for settings in SETTINGS:
            with self.subTest(**settings):
                model_settings = {k: settings[k] for k in ("weight_clamp", "unit_norm_weights")}
                unfused, fused = build(**model_settings), build(**model_settings)
                fused.enable_fused_update(compile=False)
                expected_losses = run(unfused, update_clamp=settings["update_clamp"])
                actual_losses = run(fused, update_clamp=settings["update_clamp"])
                self.assertEqual(actual_losses, expected_losses)
                self.assert_same(state(fused), state(unfused), rtol=0, atol=0)
                self.assertIsNone(fused.i2h.per_sample_weights.grad)  # nothing materialized

    def test_the_settings_bind(self):
        # Otherwise the bit-identity above would not exercise the clamps.
        unfused = build(weight_clamp=0.2)
        run(unfused, update_clamp=0.05)
        weights = torch.cat([layer.per_sample_weights.flatten() for layer in unfused.trained_layers()])
        self.assertTrue((weights.abs() == 0.2).any())
        clamped, unclamped = build(), build()
        run(clamped, update_clamp=0.05)
        run(unclamped)
        self.assertFalse(all(torch.equal(a, b) for a, b in zip(state(clamped).values(), state(unclamped).values())))

    def test_eager_fused_step_with_grad_norm_clip_matches_to_rounding(self):
        unfused, fused = build(), build()
        fused.enable_fused_update(compile=False)
        run(unfused, grad_norm_clip=0.05)
        run(fused, grad_norm_clip=0.05)
        self.assertEqual(fused.grad_clip_stats.summary()["grad_norm_clip_fraction"], 1.0)  # binds
        self.assert_same(state(fused), state(unfused), rtol=1e-5, atol=1e-7)

    def test_norm_logging_steps_take_the_unfused_step(self):
        unfused, fused = build(), build()
        fused.enable_fused_update(compile=False)
        run(unfused, log_norms_now=True)
        run(fused, log_norms_now=True)
        self.assert_same(state(fused), state(unfused), rtol=0, atol=0)
        self.assertGreater(fused.linear_layers[0].last_ephemeral_step_norm.item(), 0)

    def test_compiled_fused_step_matches_to_rounding(self):
        unfused, fused = build(weight_clamp=0.2), build(weight_clamp=0.2)
        fused.enable_fused_update(compile=True)
        try:
            run(fused, update_clamp=0.05)
        except Exception as exc:  # no C++ compiler for Inductor's CPU backend, for example
            self.skipTest(f"torch.compile unavailable here: {type(exc).__name__}: {exc}")
        run(unfused, update_clamp=0.05)
        self.assert_same(state(fused), state(unfused), rtol=1e-5, atol=1e-6)

    def test_only_ephemeral_dfa_can_fuse(self):
        with self.assertRaises(ValueError):
            build(updater="backprop").enable_fused_update()
        for extra in (["--model_type", "rnn"], ["--updater", "bptt"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(["--fused_update", "true", *extra])
        self.assertTrue(parse_args(["--fused_update", "true"]).fused_update)


if __name__ == "__main__":
    unittest.main()
