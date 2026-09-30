"""--layer_norm: affine-free LayerNorm on each trunk layer's post-GELU activations, in both models."""
import contextlib
import io
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

import heldout
import train as train_module
from ephemeral_model import EphemeralRNN, SimpleRNN, trunk_layer_norm
from reproducibility import seed_everything
from utils import initialize_charset, load_checkpoint, save_checkpoint
import tests.test_heldout as heldout_tests

CHARSET = "abcd"
BATCH, HIDDEN, LAYERS = 3, 5, 2
INPUT = len(CHARSET)
WIDTH = INPUT + HIDDEN


def build(model_type="ephemeral", updater="dfa", seed=11, **options):
    seed_everything(seed, deterministic=True)
    with contextlib.redirect_stdout(io.StringIO()):
        if model_type == "ephemeral":
            return EphemeralRNN(INPUT, HIDDEN, len(CHARSET), LAYERS, CHARSET, updater=updater,
                                batch_size=BATCH, plasticity=20.0, forget_rate=0.2,
                                ephemeral_fraction=0.5, enable_recurrence=True, **options)
        return SimpleRNN(INPUT, HIDDEN, len(CHARSET), LAYERS, dropout_rate=0, enable_recurrence=True,
                         updater=updater, **options)


def onehot(seed=5, steps=6):
    generator = torch.Generator().manual_seed(seed)
    return F.one_hot(torch.randint(len(CHARSET), (BATCH, steps), generator=generator), len(CHARSET)).float()


def config(updater):
    return {"updater": updater, "criterion": torch.nn.CrossEntropyLoss(reduction="none"),
            "input_mode": "last_one", "pe_matrix": None, "learning_rate": 0.05, "plasticity": 20.0,
            "ephemeral_update_clamp": 0, "grad_norm_clip": 0}


def trunk_inputs(model):
    """What each layer after the first reads: the next trunk layers, then i2h and i2o."""
    return [*model.linear_layers[1:], model.i2h, model.i2o]


def input_trace(layer):
    return layer.in_traces.data if isinstance(layer.in_traces, torch.nn.Parameter) else layer.in_traces


class LayerNormPlacementTest(unittest.TestCase):
    def test_every_trunk_output_is_normalized_and_is_the_next_layers_input_trace(self):
        for model_type in ("ephemeral", "rnn"):
            with self.subTest(model_type=model_type):
                model = build(model_type, layer_norm=True)
                if model_type == "ephemeral":
                    model.start_sequence_wipe()
                model(torch.randn(BATCH, INPUT), torch.randn(BATCH, HIDDEN))
                for layer in trunk_inputs(model):
                    trace = input_trace(layer)
                    torch.testing.assert_close(trace.mean(1), torch.zeros(BATCH), atol=1e-6, rtol=0)
                    torch.testing.assert_close(trace.var(1, unbiased=False), torch.ones(BATCH), atol=1e-2, rtol=0)  # eps 1e-5
                if model_type == "ephemeral":
                    # Post-activation: layer 1 reads LN(gelu(layer 0's pre-activation)).
                    first, second = model.linear_layers[0], model.linear_layers[1]
                    torch.testing.assert_close(second.in_traces.data,
                                               trunk_layer_norm(F.gelu(first.out_traces.data)),
                                               rtol=0, atol=0)

    def test_first_layer_input_and_recurrent_state_are_not_normalized(self):
        model = build(layer_norm=True)
        model.start_sequence_wipe()
        x, h = torch.randn(BATCH, INPUT) * 3, torch.randn(BATCH, HIDDEN)
        _, next_hidden = model(x, h)
        torch.testing.assert_close(model.linear_layers[0].in_traces.data, torch.cat((x, h), 1), rtol=0, atol=0)
        torch.testing.assert_close(next_hidden, torch.tanh(model.i2h.out_traces.data), rtol=0, atol=0)

    def test_no_new_parameters_and_off_is_the_old_forward(self):
        for model_type in ("ephemeral", "rnn"):
            with self.subTest(model_type=model_type):
                plain, normed = build(model_type), build(model_type, layer_norm=True)
                self.assertEqual(list(plain.state_dict()), list(normed.state_dict()))
                for name, value in plain.state_dict().items():
                    self.assertTrue(torch.equal(value, normed.state_dict()[name]), name)
                x, h = torch.randn(BATCH, INPUT), torch.randn(BATCH, HIDDEN)
                off = build(model_type, layer_norm=False)
                torch.testing.assert_close(off(x, h), plain(x, h), rtol=0, atol=0)
                self.assertFalse(torch.allclose(normed(x, h)[0], plain(x, h)[0]))

    def test_dfa_gradient_is_the_outer_product_with_the_normalized_input(self):
        model = build(layer_norm=True)
        model.start_sequence_wipe()
        model(torch.randn(BATCH, INPUT), torch.randn(BATCH, HIDDEN))
        error = torch.randn(BATCH, len(CHARSET))
        for layer in model.trained_layers():
            layer.populate_dfa_gradients(error)
        expected = error.unsqueeze(2) * model.i2o.in_traces.data.unsqueeze(1)
        torch.testing.assert_close(model.i2o.per_sample_weights.grad, expected, rtol=0, atol=0)
        # |x| = sqrt(width) for every normalized input row (up to eps).
        norms = model.i2h.in_traces.data.norm(dim=1)
        torch.testing.assert_close(norms, torch.full((BATCH,), WIDTH ** 0.5), atol=1e-2, rtol=0)


class LayerNormTrainingTest(unittest.TestCase):
    def test_every_updater_and_model_trains_finite_and_differs_from_off(self):
        for model_type in ("ephemeral", "rnn"):
            for updater in ("dfa", "backprop", "bptt"):
                for residual in (False, True):
                    with self.subTest(model_type=model_type, updater=updater, residual=residual):
                        losses = []
                        for layer_norm in (False, True):
                            model = build(model_type, updater, layer_norm=layer_norm, residual_connection=residual)
                            optimizer = (torch.optim.SGD(model.parameters(), lr=0.05)
                                         if model_type == "rnn" and updater != "dfa" else None)
                            state = {"training_instance": 0, "log_norms_now": False}
                            with contextlib.redirect_stdout(io.StringIO()):
                                for batch in range(2):
                                    _, loss, _, _, _, _ = train_module.train(
                                        None, onehot(batch), model, config(updater), state, optimizer)
                            self.assertTrue(torch.isfinite(torch.tensor(loss)))
                            losses.append(loss)
                        self.assertNotEqual(losses[0], losses[1])

    def test_fused_update_matches_unfused_with_layer_norm(self):
        results = []
        for fused in (False, True):
            model = build(layer_norm=True)
            if fused:
                model.enable_fused_update(compile=False)
            state = {"training_instance": 0, "log_norms_now": False}
            train_module.train(None, onehot(), model, config("dfa"), state)
            results.append({k: v.clone() for k, v in model.state_dict().items()})
        for name, value in results[0].items():
            self.assertTrue(torch.equal(value, results[1][name]), name)


class HeldoutWithLayerNormTest(heldout_tests.EvaluatorMatchesTrainerTest):
    """The evaluator's fast entries match the training step bit for bit with --layer_norm too."""

    def test_fast_entries_match_the_training_step_bit_for_bit(self):
        self.check(exact=True, model_options={"fast_weight_clamp": 0.01, "layer_norm": True})

    def test_with_grad_norm_clip_they_match_to_rounding(self):
        self.check(exact=False, model_options={"fast_weight_clamp": 0.01, "layer_norm": True},
                   grad_norm_clip=0.05)

    def test_run_flag_and_heldout_cli_read_a_layer_norm_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            log = heldout_tests.run_main("--layer_norm", "true", "--heldout_eval_every", "2",
                                         checkpoint_dir=directory)
            self.assertEqual(log.count("heldout_strict/recall_acc:"), 2)
            path = os.path.join(directory, "latest_checkpoint.pth")
            self.assertTrue(torch.load(path, weights_only=False)["config"]["layer_norm"])
            output, built = io.StringIO(), []
            real_build = train_module.build_model
            with patch.object(heldout_tests.heldout, "load_heldout_batches",
                              side_effect=heldout_tests.fake_heldout_batches), \
                    patch.object(train_module, "build_model",
                                 side_effect=lambda *a: built.append(real_build(*a)) or built[-1]), \
                    contextlib.redirect_stdout(output):
                heldout.main(["--checkpoint", path, "--device", "cpu"])
            self.assertIn("heldout_strict/first_answer_acc", output.getvalue())
            self.assertTrue(built[0].layer_norm)  # rebuilt from the checkpoint's config


class LayerNormResumeTest(unittest.TestCase):
    CONFIG = {"n_hidden": HIDDEN, "n_layers": LAYERS, "updater": "dfa", "charset_size": 4,
              "model_type": "ephemeral"}

    def test_resume_refuses_a_changed_layer_norm_and_a_checkpoint_without_it_is_off(self):
        with tempfile.TemporaryDirectory() as directory:
            model = build()
            for saved, current, refused in (({}, {}, False), ({}, {"layer_norm": False}, False),
                                            ({}, {"layer_norm": True}, True),
                                            ({"layer_norm": True}, {"layer_norm": True}, False),
                                            ({"layer_norm": True}, {"layer_norm": False}, True)):
                with self.subTest(saved=saved, current=current), contextlib.redirect_stdout(io.StringIO()):
                    save_checkpoint({"config": {**self.CONFIG, **saved}, "model_state_dict": model.state_dict()},
                                    directory, "c.pth")
                    load = lambda: load_checkpoint(os.path.join(directory, "c.pth"), build(),
                                                   {**self.CONFIG, **current})
                    if refused:
                        with self.assertRaisesRegex(RuntimeError, "configuration mismatch"):
                            load()
                    else:
                        load()

    def test_cli_flag(self):
        self.assertFalse(train_module.parse_args([]).layer_norm)
        self.assertTrue(train_module.parse_args(["--layer_norm"]).layer_norm)
        self.assertTrue(train_module.parse_args(["--layer_norm", "true", "--model_type", "rnn"]).layer_norm)


if __name__ == "__main__":
    unittest.main()
