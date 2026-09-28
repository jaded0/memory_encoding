"""--slow_weight_decay and --output_tanh in both models (tests/test_fused_update.py covers the fused step)."""
import contextlib
import io
import unittest

import torch
import torch.nn.functional as F

from ephemeral_model import EphemeralLinear, EphemeralRNN, SimpleRNN
from reproducibility import seed_everything
from train import train

CHARSET = list("abcd")
SEQUENCE = torch.tensor([[0, 1, 2, 3, 1], [3, 2, 0, 1, 2]])  # 4 steps


def run(model, updater, learning_rate):
    config = {"updater": updater, "criterion": torch.nn.CrossEntropyLoss(reduction="none"),
              "input_mode": "last_one", "pe_matrix": None, "learning_rate": learning_rate,
              "ephemeral_update_clamp": 0, "grad_norm_clip": 0, "plasticity": 3.0}
    optimizer = None if updater == "dfa" else torch.optim.SGD(model.parameters(), lr=learning_rate)
    with contextlib.redirect_stdout(io.StringIO()):
        train(SEQUENCE, F.one_hot(SEQUENCE, len(CHARSET)).float(), model, config, {"training_instance": 0},
              optimizer=optimizer)


def build(model_type, updater, **options):
    seed_everything(3, deterministic=True)
    with contextlib.redirect_stdout(io.StringIO()):
        if model_type == "rnn":
            return SimpleRNN(len(CHARSET), 4, len(CHARSET), 2, updater=updater, **options)
        return EphemeralRNN(len(CHARSET), 4, len(CHARSET), 2, CHARSET, unit_norm_weights=False, updater=updater,
                            plasticity=3.0, batch_size=2, forget_rate=0.25, ephemeral_fraction=0.5, **options)


class SlowWeightDecayTest(unittest.TestCase):
    def test_forget_step_decays_slow_entries_and_forgets_fast_ones(self):
        torch.manual_seed(0)
        with contextlib.redirect_stdout(io.StringIO()):
            layer = EphemeralLinear(5, 4, CHARSET, unit_norm_weights=False, batch_size=2, ephemeral_fraction=0.5,
                                    forget_rate=0.25, slow_weight_decay=0.1)
        layer.per_sample_weights.data.normal_()
        before = layer.per_sample_weights.detach().clone()
        layer.apply_forget_step()
        mask = layer.ephemeral_mask.expand_as(before)
        torch.testing.assert_close(layer.per_sample_weights[mask], 0.75 * before[mask], rtol=1e-6, atol=0)
        torch.testing.assert_close(layer.per_sample_weights[~mask], 0.9 * before[~mask], rtol=1e-6, atol=0)

    def test_every_updater_decays_slow_weights_once_per_update(self):
        # With learning rate 0 only the decay (and, for fast entries, forgetting) changes a weight.
        updates = {"dfa": 4, "backprop": 4, "bptt": 1}
        for model_type in ("ephemeral", "rnn"):
            for updater, count in updates.items():
                with self.subTest(model_type=model_type, updater=updater):
                    model = build(model_type, updater, slow_weight_decay=0.1)
                    if model_type == "ephemeral":
                        layers = model.trained_layers()
                        model.start_sequence_wipe()
                        before = [layer.per_sample_weights.detach().clone() for layer in layers]
                        run(model, updater, learning_rate=0.0)
                        for layer, old in zip(layers, before):
                            slow = ~layer.ephemeral_mask.expand_as(old)
                            torch.testing.assert_close(layer.per_sample_weights[slow], 0.9 ** count * old[slow],
                                                       rtol=1e-5, atol=0)
                    else:
                        before = [layer.weight.detach().clone() for layer in model.dfa_layers()]
                        run(model, updater, learning_rate=0.0)
                        for layer, old in zip(model.dfa_layers(), before):
                            torch.testing.assert_close(layer.weight, 0.9 ** count * old, rtol=1e-5, atol=0)

    def test_zero_decay_leaves_training_unchanged(self):
        for model_type, updater in (("ephemeral", "dfa"), ("rnn", "bptt")):
            with self.subTest(model_type=model_type):
                plain, zero = build(model_type, updater), build(model_type, updater, slow_weight_decay=0)
                run(plain, updater, 0.1)
                run(zero, updater, 0.1)
                for a, b in zip(plain.state_dict().values(), zero.state_dict().values()):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)


class OutputTanhTest(unittest.TestCase):
    def test_output_head_reads_tanh_of_the_trunk(self):
        for model_type in ("ephemeral", "rnn"):
            with self.subTest(model_type=model_type):
                plain, bounded = build(model_type, "dfa"), build(model_type, "dfa", output_tanh=True)
                x = torch.eye(len(CHARSET))[:2] * 50  # large input so the trunk leaves tanh's linear range
                hidden = torch.zeros(2, 4)
                plain_out, _ = plain(x, hidden)
                bounded_out, _ = bounded(x, hidden)
                trunk = plain.i2o.in_traces
                self.assertGreater(trunk.abs().max().item(), 1)
                torch.testing.assert_close(bounded.i2o.in_traces, torch.tanh(trunk), rtol=1e-6, atol=1e-7)
                self.assertFalse(torch.allclose(plain_out, bounded_out))


if __name__ == "__main__":
    unittest.main()
