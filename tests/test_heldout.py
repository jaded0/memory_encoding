"""heldout.py: the held-out evaluator takes the training DFA step on fast entries only."""
import contextlib
import copy
import io
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

import heldout
import train as train_module
from ephemeral_model import EphemeralRNN, forget_keep
from reproducibility import seed_everything
from tests.test_seed_resume import DATASET, fake_loader, in_memory_items

CHARSET = "23. "  # DATASET's charset
BATCH = 2


def build(**options):
    settings = dict(updater="dfa", batch_size=BATCH, plasticity=50.0, forget_rate=0.2,
                    ephemeral_fraction=0.5, enable_recurrence=True,
                    weight_clamp=0.6, fast_weight_clamp=0.05, slow_weight_decay=0.1, output_tanh=True)
    settings.update(options)
    seed_everything(7, deterministic=True)
    with contextlib.redirect_stdout(io.StringIO()):
        return EphemeralRNN(10, 5, 4, 2, CHARSET, **settings)  # last_two (8) + PE (2)


def config(**options):
    return {"updater": "dfa", "criterion": torch.nn.CrossEntropyLoss(reduction="none"),
            "input_mode": "last_two", "pe_matrix": train_module.positional_encoding(2, "cpu"),
            "learning_rate": 0.3, "ephemeral_update_clamp": 0.02, "grad_norm_clip": 0, **options}


def episodes(texts):
    return F.one_hot(torch.tensor([[CHARSET.index(c) for c in text] for text in texts]), 4).float()


TEXTS = ["23.32  ", "322.223"]


def with_frozen_slow_training(model):
    """Makes train.train_batch restore every slow entry and bias after each step's forget, so the
    trainer's fast entries evolve against the same frozen slow weights as the evaluator's."""
    frozen = {}
    wipe, forget = model.start_sequence_wipe, model.apply_forget_step

    def wipe_then_snapshot():
        wipe()
        for layer in model.trained_layers():
            frozen[layer] = (layer.per_sample_weights.data.clone(), layer.bias.data.clone())

    def forget_then_restore():
        forget()
        for layer in model.trained_layers():
            weights, bias = frozen[layer]
            layer.per_sample_weights.data.copy_(torch.where(layer.ephemeral_mask, layer.per_sample_weights.data, weights))
            layer.bias.data.copy_(bias)

    model.start_sequence_wipe, model.apply_forget_step = wipe_then_snapshot, forget_then_restore
    return model


class EvaluatorMatchesTrainerTest(unittest.TestCase):
    def check(self, exact, model_options=None, **options):
        """model_options set clamps that bind here: without them the fast entries differ."""
        cfg = config(**options)
        trained = build(**(model_options or {}))
        evaluated = copy.deepcopy(trained)
        with_frozen_slow_training(trained)
        onehot = episodes(TEXTS)
        state = {"training_instance": 0, "log_norms_now": False}
        _, _, train_preds, train_losses, _, _ = train_module.train_batch(None, onehot, trained, cfg, state)
        preds, losses = heldout.evaluate_held_out(evaluated, onehot, torch.ones(BATCH, 6, dtype=torch.bool), cfg)
        tolerance = {"rtol": 0, "atol": 0} if exact else {}
        torch.testing.assert_close(preds, train_preds, rtol=0, atol=0)
        torch.testing.assert_close(losses, train_losses, **tolerance)
        for mine, theirs in zip(evaluated.trained_layers(), trained.trained_layers()):
            torch.testing.assert_close(mine.per_sample_weights, theirs.per_sample_weights, **tolerance)
        # Not vacuous: the fast entries were written, and without the clamps they would differ.
        fast = [layer.per_sample_weights[:, layer.ephemeral_mask] for layer in evaluated.linear_layers]
        self.assertTrue(all(entries.abs().sum() > 0 for entries in fast))
        for unclamped_model, unclamped_cfg in (({}, {"ephemeral_update_clamp": 0}),
                                               ({"weight_clamp": 0, "fast_weight_clamp": 0}, {})):
            unclamped = build(**{**(model_options or {}), **unclamped_model})
            heldout.evaluate_held_out(unclamped, onehot, torch.ones(BATCH, 6, dtype=torch.bool),
                                      {**cfg, **unclamped_cfg})
            self.assertFalse(torch.allclose(unclamped.linear_layers[0].per_sample_weights,
                                            evaluated.linear_layers[0].per_sample_weights))

    def test_fast_entries_match_the_training_step_bit_for_bit(self):
        # --ephemeral_update_clamp, --output_tanh, --slow_weight_decay and recurrence throughout;
        # --fast_weight_clamp binding, then --weight_clamp binding on the fast entries.
        self.check(exact=True, model_options={"fast_weight_clamp": 0.01})
        self.check(exact=True, model_options={"weight_clamp": 0.01, "fast_weight_clamp": 0})

    def test_with_grad_norm_clip_they_match_to_rounding(self):
        # Training clips the materialized gradient; the evaluator uses the closed form.
        self.check(exact=False, model_options={"fast_weight_clamp": 0.01}, grad_norm_clip=0.05)


class ProtocolTest(unittest.TestCase):
    def run_protocol(self, protocol, texts, model=None):
        model = model or build()
        onehot = episodes(texts)
        mask = heldout.update_mask(protocol, texts, DATASET, onehot)
        return model, heldout.evaluate_held_out(model, onehot, mask, config())

    def test_strict_mask_stops_writes_from_the_first_recall_prediction(self):
        mask = heldout.update_mask("strict", TEXTS, DATASET, episodes(TEXTS))
        # "23.32  ": first recall target is index 3, predicted at step 2; "322.223": index 4, step 3.
        self.assertEqual(mask.tolist(), [[True, True, False, False, False, False],
                                         [True, True, True, False, False, False]])

    def test_masked_steps_only_forget(self):
        model = build()
        onehot = episodes(TEXTS)
        mask = heldout.update_mask("strict", TEXTS, DATASET, onehot)
        states = []
        hook = model.register_forward_pre_hook(
            lambda *_: states.append([layer.per_sample_weights.detach().clone() for layer in model.trained_layers()]))
        heldout.evaluate_held_out(model, onehot, mask, config())
        hook.remove()
        states.append([layer.per_sample_weights.detach().clone() for layer in model.trained_layers()])
        for step in range(6):
            for layer, now, after in zip(model.trained_layers(), states[step], states[step + 1]):
                keep = forget_keep(layer.forget_rate, layer.ephemeral_mask)
                for row in range(BATCH):
                    if not mask[row, step]:
                        self.assertTrue(torch.equal(after[row], torch.where(layer.ephemeral_mask, now[row] * keep, now[row])))
                    elif step == 0 and layer.ephemeral_mask.any():
                        self.assertFalse(torch.equal(after[row], now[row] * keep))

    def test_no_fast_leaves_fast_entries_zero_and_slow_entries_frozen(self):
        for protocol in heldout.PROTOCOLS:
            with self.subTest(protocol=protocol):
                model = build()
                model.start_sequence_wipe()
                before = {name: value.clone() for name, value in model.state_dict().items()}
                self.run_protocol(protocol, TEXTS, model)
                for layer in model.trained_layers():
                    name = next(n for n, m in model.named_modules() if m is layer)
                    weights, mask = layer.per_sample_weights, layer.ephemeral_mask
                    self.assertTrue(torch.equal(weights[:, ~mask], before[f"{name}.per_sample_weights"][:, ~mask]))
                    self.assertTrue(torch.equal(layer.bias, before[f"{name}.bias"]))
                    if protocol == "no_fast":
                        self.assertEqual(weights[:, mask].abs().sum().item(), 0)
                    elif mask.any():
                        self.assertGreater(weights[:, mask].abs().sum().item(), 0)

    def test_evaluate_protocols_restores_state_and_reports_first_answer(self):
        model = build()
        before = {name: value.clone() for name, value in model.state_dict().items()}
        results = heldout.evaluate_protocols(model, [(TEXTS, episodes(TEXTS))], config(), DATASET)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)
        for protocol in heldout.PROTOCOLS:
            # In a palindrome the first answer is the lag-1 target.
            self.assertEqual(results[f"heldout_{protocol}/first_answer_acc"],
                             results[f"heldout_{protocol}/recall_acc_lag_1"])
            self.assertIn(f"heldout_{protocol}/recall_acc_lag_3", results)

    def trace(self, protocol, texts, model=None, oracle=False, generator=None):
        """Runs protocol on texts, recording each step's model input and output and the fast
        entries before each step (and at the end). oracle replaces each output with confident
        logits on the true next character: a perfect model."""
        model = model or build()
        onehot = episodes(texts)
        inputs, outputs, states = [], [], []
        fast = lambda: [layer.per_sample_weights[:, layer.ephemeral_mask].clone() for layer in model.linear_layers]

        def before(_, args):
            inputs.append(args[0].clone())
            states.append(fast())

        def after(_, __, out):
            if oracle:
                out = (20.0 * onehot[:, len(inputs)], out[1])
            outputs.append(out[0].clone())
            return out

        hooks = [model.register_forward_pre_hook(before), model.register_forward_hook(after)]
        steps = onehot.shape[1] - 1
        self_from = heldout.first_recall_steps(texts, DATASET, steps) if protocol == "free_running" else None
        preds, losses = heldout.evaluate_held_out(
            model, onehot, heldout.update_mask(protocol, texts, DATASET, onehot), config(), self_from, generator)
        for hook in hooks:
            hook.remove()
        states.append(fast())
        return preds, losses, inputs, outputs, states

    def test_free_running_is_observed_up_to_the_recall_boundary(self):
        free, observed = self.trace("free_running", TEXTS), self.trace("observed", TEXTS)
        firsts = heldout.first_recall_steps(TEXTS, DATASET, 6).tolist()
        self.assertEqual(firsts, [2, 3])
        for row, first in enumerate(firsts):
            # Up to and including the step that predicts the first recall target: the same inputs,
            # predictions, losses, and fast entries before each step.
            for step in range(first + 1):
                self.assertTrue(torch.equal(free[2][step][row], observed[2][step][row]))
                self.assertEqual(free[0][step, row], observed[0][step, row])
                self.assertEqual(free[1][step, row], observed[1][step, row])
                for mine, theirs in zip(free[4][step], observed[4][step]):
                    self.assertTrue(torch.equal(mine[row], theirs[row]))
        # Not vacuous: the untrained model's own predictions are not the truth, so they diverge.
        self.assertFalse(all(torch.equal(a, b) for a, b in zip(free[4][-1], observed[4][-1])))

    def test_inputs_after_the_boundary_are_the_models_own_predictions(self):
        preds, losses, inputs, outputs, _ = self.trace("free_running", TEXTS)
        onehot, vocab = episodes(TEXTS), len(CHARSET)
        wrong = 0
        for row, first in enumerate(heldout.first_recall_steps(TEXTS, DATASET, 6).tolist()):
            for step in range(first, 5):
                self.assertTrue(torch.equal(inputs[step + 1][row, :vocab], F.one_hot(preds[step, row], vocab).float()))
                if step > first:  # last_two: the previous character is also the model's own
                    self.assertTrue(torch.equal(inputs[step + 1][row, vocab:2 * vocab],
                                                F.one_hot(preds[step - 1, row], vocab).float()))
            for step in range(first, 6):
                # Scored against the true target, not the model's own.
                truth = onehot[row, step + 1].argmax()
                expected = F.cross_entropy(outputs[step][row:row + 1], truth[None])
                torch.testing.assert_close(losses[step, row], expected)
                wrong += int(preds[step, row] != truth)
        self.assertGreater(wrong, 0)

    def test_a_perfect_model_free_runs_exactly_as_observed(self):
        free, observed = self.trace("free_running", TEXTS, oracle=True), self.trace("observed", TEXTS, oracle=True)
        for mine, theirs in zip(free[:2], observed[:2]):
            self.assertTrue(torch.equal(mine, theirs))
        for step in range(len(free[2])):
            self.assertTrue(torch.equal(free[2][step], observed[2][step]))
        for mine, theirs in zip(free[4][-1], observed[4][-1]):
            self.assertTrue(torch.equal(mine, theirs))
        self.assertTrue(any(entries.abs().sum() > 0 for entries in free[4][-1]))

    def test_sampling_is_seeded_and_leaves_the_global_rng_alone(self):
        runs = []
        for _ in range(2):
            model = build()
            before = torch.get_rng_state()
            runs.append(self.trace("free_running", TEXTS, model, generator=torch.Generator().manual_seed(5))[0])
            self.assertTrue(torch.equal(torch.get_rng_state(), before))
        self.assertTrue(torch.equal(runs[0], runs[1]))
        results = heldout.evaluate_protocols(build(), [(TEXTS, episodes(TEXTS))], config(), DATASET,
                                             ["free_running"], free_running_sample=5)
        self.assertIn("heldout_free_running/recall_acc", results)

    def test_rejected_models(self):
        onehot = episodes(TEXTS)
        with self.assertRaisesRegex(ValueError, "dfa"):
            heldout.evaluate_held_out(build(updater="backprop"), onehot, None, config())
        with self.assertRaisesRegex(ValueError, "batch size"):
            heldout.evaluate_held_out(build(batch_size=3), onehot, None, config())
        for flags in (["--heldout_eval_every", "5", "--model_type", "rnn"],
                      ["--heldout_eval_every", "5", "--updater", "backprop"]):
            with self.subTest(flags=flags), self.assertRaises(SystemExit), \
                    contextlib.redirect_stderr(io.StringIO()):
                train_module.parse_args(flags)


def fake_heldout_batches(dataset, batch_size, n_batches, device, split="validation"):
    collate = __import__("preprocess").OneHotCollate(len(CHARSET))
    rows = in_memory_items()[:batch_size * 2]
    return [(texts, onehot) for texts, _, onehot in
            (collate(rows[i:i + batch_size]) for i in range(0, len(rows), batch_size))]


def run_main(*extra, checkpoint_dir):
    argv = ["train.py", "--dataset", DATASET, "--track", "False", "--n_iters", "4", "--print_freq", "2",
            "--checkpoint_save_freq", "4", "--checkpoint_dir", checkpoint_dir, "--batch_size", "2",
            "--hidden_size", "4", "--num_layers", "1", "--seed", "3", "--plasticity", "100", *extra]
    output = io.StringIO()
    with patch("sys.argv", argv), \
            patch.object(train_module, "load_and_preprocess_data", side_effect=fake_loader(0)), \
            patch.object(train_module, "load_heldout_batches", side_effect=fake_heldout_batches), \
            contextlib.redirect_stdout(output):
        train_module.main()
    return output.getvalue()


class RunFlagAndCheckpointTest(unittest.TestCase):
    def test_flag_logs_heldout_metrics_leaves_training_unchanged_and_cli_reads_checkpoint(self):
        with tempfile.TemporaryDirectory() as off, tempfile.TemporaryDirectory() as on:
            plain = run_main(checkpoint_dir=off)
            evaluated = run_main("--heldout_eval_every", "2", checkpoint_dir=on)
            self.assertNotIn("heldout_", plain)
            self.assertEqual(evaluated.count("heldout_strict/recall_acc:"), 2)
            first, second = (torch.load(os.path.join(d, "latest_checkpoint.pth"), weights_only=False)
                             for d in (off, on))
            for name, value in first["model_state_dict"].items():
                self.assertTrue(torch.equal(value, second["model_state_dict"][name]), name)

            output = io.StringIO()
            with patch.object(heldout, "load_heldout_batches", side_effect=fake_heldout_batches), \
                    contextlib.redirect_stdout(output):
                heldout.main(["--checkpoint", os.path.join(on, "latest_checkpoint.pth"), "--device", "cpu",
                              "--json", os.path.join(on, "heldout.json")])
            self.assertIn('"iteration": 4', output.getvalue())
            for protocol in heldout.PROTOCOLS:
                self.assertIn(f"heldout_{protocol}/first_answer_acc", output.getvalue())


if __name__ == "__main__":
    unittest.main()
