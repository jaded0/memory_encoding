"""--slow_update_every: how often the slow parameters take their DFA step.

1 is the per-step path (unchanged; the golden traces pin it). N > 1 accumulates each sequence's
slow gradients over N steps; 'sequence' freezes every slow parameter within a sequence and applies
the batch mean of the per-sequence sums at its end. The fast entries take the per-step update in
every setting."""
import contextlib
import io
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

import heldout
from ephemeral_model import EphemeralRNN, parse_slow_update_every
from reproducibility import seed_everything
from train import parse_args, train

CHARSET = list("abcd")
BATCH = 3
SEQUENCES = (torch.tensor([[0, 1, 2, 3, 1], [3, 2, 0, 1, 2], [1, 1, 3, 0, 2]]),   # 4 steps each
             torch.tensor([[2, 0, 3, 1, 2], [1, 3, 2, 0, 1], [0, 0, 1, 2, 3]]))
LEARNING_RATE = 0.3


def build(slow_update_every=1, **options):
    settings = dict(updater="dfa", plasticity=3.0, batch_size=BATCH,
                    forget_rate=0.25, ephemeral_fraction=0.5, enable_recurrence=True)
    settings.update(options)
    seed_everything(11, deterministic=True)
    with contextlib.redirect_stdout(io.StringIO()):
        return EphemeralRNN(len(CHARSET), 4, len(CHARSET), 2, CHARSET, slow_update_every=slow_update_every,
                            **settings)


def config(**options):
    return {"updater": "dfa", "criterion": torch.nn.CrossEntropyLoss(reduction="none"),
            "input_mode": "last_one", "pe_matrix": None, "learning_rate": LEARNING_RATE,
            "ephemeral_update_clamp": 0, "grad_norm_clip": 0, **options}


def run(model, batch, **options):
    with contextlib.redirect_stdout(io.StringIO()):
        _, loss, *_ = train(batch, F.one_hot(batch, len(CHARSET)).float(), model, config(**options),
                            {"training_instance": 0, "log_norms_now": False})
    return loss


def slow_state(model):
    """Every slow parameter: each layer's slow entries (all copies) and bias."""
    state = {}
    for i, layer in enumerate(model.trained_layers()):
        state[f"{i}.slow"] = layer.per_sample_weights.data[:, ~layer.ephemeral_mask].clone()
        state[f"{i}.bias"] = layer.bias.data.clone()
    return state


def fast_state(model):
    return {i: layer.per_sample_weights.data[:, layer.ephemeral_mask].clone()
            for i, layer in enumerate(model.trained_layers()) if not layer.is_last_layer}


def record_steps(model):
    """After each windowed step: the slow state, the fast state, and each layer's (projected
    error, input trace), as the step saw them."""
    steps = []
    original = model.windowed_dfa_step

    def recording(*args, **kwargs):
        original(*args, **kwargs)
        steps.append({"slow": slow_state(model), "fast": fast_state(model),
                      "errors": [layer._last_projected_error.clone() for layer in model.trained_layers()],
                      "inputs": [layer.in_traces.data.clone() for layer in model.trained_layers()]})
    model.windowed_dfa_step = recording
    return steps


class ParseTest(unittest.TestCase):
    def test_values(self):
        self.assertEqual(parse_slow_update_every("1"), 1)
        self.assertEqual(parse_slow_update_every(4), 4)
        self.assertEqual(parse_slow_update_every("Sequence"), "sequence")
        for bad in ("0", "-2", "1.5", "every", True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_slow_update_every(bad)

    def test_parser(self):
        self.assertEqual(parse_args([]).slow_update_every, 1)
        self.assertEqual(parse_args(["--slow_update_every", "sequence"]).slow_update_every, "sequence")
        self.assertEqual(parse_args(["--slow_update_every", "3"]).slow_update_every, 3)
        for extra in (["--updater", "backprop"], ["--updater", "bptt"], ["--model_type", "rnn"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(["--slow_update_every", "sequence", *extra])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--slow_update_every", "0"])
        with self.assertRaises(ValueError):
            build("sequence", updater="backprop")


class FusedPlasticityTest(unittest.TestCase):
    def test_alpha_from_the_mask_is_the_tensor(self):
        model = build()
        layer = model.linear_layers[0]
        self.assertEqual(layer.fused_plasticity(), 3.0)
        rebuilt = torch.where(layer.ephemeral_mask, layer.fused_plasticity(), 1.0)
        self.assertTrue(torch.equal(rebuilt, layer.plasticity.data))
        with contextlib.redirect_stdout(io.StringIO()):
            layer.set_plasticity(7.0)
        self.assertEqual(layer.fused_plasticity(), 7.0)
        layer.plasticity.data[0, 0] = 2.0
        layer.load_state_dict(layer.state_dict())  # a load invalidates the cache
        self.assertTrue(torch.is_tensor(layer.fused_plasticity()))  # not uniform: the tensor itself


class OneStepSequencesTest(unittest.TestCase):
    def test_every_setting_matches_the_per_step_update(self):
        # With one step per sequence the slow step of every setting is the same batch-mean step
        # (per step: each copy's step, then the wipe's mean; sequence: the mean step), to rounding.
        # A binding --weight_clamp is the exception under 'sequence': it clamps the shared matrix
        # after the mean step, where the per-step update clamps each copy before the wipe's mean.
        short = SEQUENCES[0][:, :2]
        stabilizers = {"slow_weight_decay": 0.05, "fast_weight_clamp": 0.1}
        for setting in ("sequence", 1, 3):
            clamp = {} if setting == "sequence" else {"weight_clamp": 0.2}
            for options in ({}, {**stabilizers, **clamp}, {"slow_nlms": True}):
                with self.subTest(setting=setting, **options):
                    reference, model = build(1, **options), build(setting, **options)
                    for batch in (short, SEQUENCES[1][:, :2], short):
                        self.assertAlmostEqual(run(model, batch), run(reference, batch), places=6)
                    reference.start_sequence_wipe()
                    model.start_sequence_wipe()
                    for mine, theirs in zip(model.trained_layers(), reference.trained_layers()):
                        torch.testing.assert_close(mine.per_sample_weights, theirs.per_sample_weights,
                                                   rtol=1e-5, atol=1e-7)
                        torch.testing.assert_close(mine.bias, theirs.bias, rtol=1e-5, atol=1e-7)

    def test_dfa_fprime_and_layer_norm_reach_the_windowed_step(self):
        # The windowed step shares the projected errors with the per-step path, so --dfa_fprime
        # (in the error) and --layer_norm (in the forward pass) must give the same one-step result
        # as the per-step update, and must actually change it (else this test pins nothing).
        short = SEQUENCES[0][:, :2]
        results = {}
        for options in ({}, {"dfa_fprime": True}, {"layer_norm": True},
                        {"dfa_fprime": True, "layer_norm": True}):
            for setting in (1, "sequence", 3):
                with self.subTest(setting=setting, **options):
                    model = build(setting, **options)
                    run(model, short)
                    model.start_sequence_wipe()
                    results[(tuple(options), setting)] = model.linear_layers[0].per_sample_weights.data.clone()
                    if setting != 1:
                        torch.testing.assert_close(results[(tuple(options), setting)],
                                                   results[(tuple(options), 1)], rtol=1e-5, atol=1e-7)
        for options in (("dfa_fprime",), ("layer_norm",)):
            self.assertFalse(torch.allclose(results[(options, 1)], results[((), 1)]), options)


class PerSequenceTest(unittest.TestCase):
    def test_slow_parameters_are_constant_within_a_sequence(self):
        model = build("sequence", weight_clamp=0.2, slow_weight_decay=0.05)
        run(model, SEQUENCES[0])
        model.start_sequence_wipe()
        start = slow_state(model)
        steps = record_steps(model)
        run(model, SEQUENCES[1])
        self.assertEqual(len(steps), 4)
        for step in steps:
            for key, value in start.items():
                self.assertTrue(torch.equal(step["slow"][key], value), key)
        for step, following in zip(steps, steps[1:]):  # the fast entries do change every step
            self.assertFalse(torch.equal(step["fast"][0], following["fast"][0]))
        after = slow_state(model)
        self.assertFalse(torch.equal(after["0.slow"], start["0.slow"]))
        for layer in model.trained_layers():  # still one shared matrix
            slow = layer.per_sample_weights.data[:, ~layer.ephemeral_mask]
            self.assertTrue(torch.equal(slow, slow[:1].expand_as(slow)))
        model.start_sequence_wipe()  # the wipe leaves them bit for bit
        for key, value in slow_state(model).items():
            self.assertTrue(torch.equal(value, after[key]), key)

    def test_wipe_every_keeps_the_fast_entries_and_the_shared_slow_matrix(self):
        # --wipe_every N > 1 calls start_sequence_wipe(wipe_fast=False) between wipes.
        for setting in ("sequence", 1, 3):
            with self.subTest(setting=setting):
                model = build(setting)
                run(model, SEQUENCES[0])
                model.start_sequence_wipe(wipe_fast=False)
                fast_before = fast_state(model)
                slow_before = slow_state(model)
                run(model, SEQUENCES[1])
                model.start_sequence_wipe(wipe_fast=False)
                # each row's fast entries carry into the next sequence (and were updated by it)
                carried = fast_state(model)
                self.assertFalse(torch.equal(carried[0], fast_before[0]))
                self.assertFalse(torch.equal(carried[0][:1].expand_as(carried[0]), carried[0]))  # rows differ
                model.start_sequence_wipe(wipe_fast=False)  # a second call changes nothing more
                for key, value in fast_state(model).items():
                    self.assertTrue(torch.equal(value, carried[key]), key)
                for layer in model.trained_layers():
                    slow = layer.per_sample_weights.data[:, ~layer.ephemeral_mask]
                    self.assertTrue(torch.equal(slow, slow[:1].expand_as(slow)))  # slow entries shared
                model.start_sequence_wipe()  # a full wipe zeroes the fast entries
                for value in fast_state(model).values():
                    self.assertEqual(value.abs().sum().item(), 0.0)

    def test_the_step_is_the_batch_mean_of_the_summed_per_sequence_gradients(self):
        for clip in (0, 0.05):
            with self.subTest(grad_norm_clip=clip):
                model = build("sequence")
                model.start_sequence_wipe()
                start = slow_state(model)
                steps = record_steps(model)
                run(model, SEQUENCES[0], grad_norm_clip=clip)
                for i, layer in enumerate(model.trained_layers()):
                    gradient = sum(torch.einsum("bo,bi->oi", step["errors"][i], step["inputs"][i]) for step in steps)
                    expected = start[f"{i}.slow"][0] - LEARNING_RATE * gradient[~layer.ephemeral_mask] / BATCH
                    actual = layer.per_sample_weights.data[:, ~layer.ephemeral_mask]
                    torch.testing.assert_close(actual, expected.expand_as(actual), rtol=1e-6, atol=1e-7)
                    bias = start[f"{i}.bias"] - LEARNING_RATE * sum(step["errors"][i].mean(0) for step in steps)
                    torch.testing.assert_close(layer.bias.data, bias, rtol=1e-6, atol=1e-7)
                if clip:
                    self.assertEqual(model.grad_clip_stats.summary()["grad_norm_clip_fraction"], 1.0)

    def test_clamp_and_decay_act_once_per_sequence(self):
        # lr 0: only the decay moves a slow entry, (1 - d) ** steps at the end of the sequence.
        model = build("sequence", slow_weight_decay=0.1)
        model.start_sequence_wipe()
        start = slow_state(model)
        steps = record_steps(model)
        run(model, SEQUENCES[0], learning_rate=0.0)
        self.assertTrue(all(torch.equal(step["slow"]["0.slow"], start["0.slow"]) for step in steps))
        torch.testing.assert_close(slow_state(model)["0.slow"], 0.9 ** 4 * start["0.slow"], rtol=1e-6, atol=0)
        clamped = build("sequence", weight_clamp=0.05)
        run(clamped, SEQUENCES[0])
        for layer in clamped.trained_layers():
            slow = layer.per_sample_weights.data[:, ~layer.ephemeral_mask]
            limit = torch.tensor(0.05)
            self.assertTrue((slow.abs() <= limit).all())
            self.assertTrue((slow.abs() == limit).any())  # it binds

    def test_fast_entries_take_the_per_step_update(self):
        # Against the per-step model with its slow entries and biases restored after every step
        # (heldout's reference trainer), the fast entries match bit for bit.
        from tests.test_heldout import with_frozen_slow_training
        for options in ({}, {"weight_clamp": 0.2, "fast_weight_clamp": 0.05, "slow_weight_decay": 0.1}):
            with self.subTest(**options):
                reference, model = build(1, **options), build("sequence", **options)
                for layer in reference.trained_layers():
                    # Only its wipe reads this: it then keeps the equal slow copies as they are
                    # instead of taking their mean, which can move them by an ulp.
                    layer.slow_update_every = "sequence"
                with_frozen_slow_training(reference)
                model.start_sequence_wipe()
                steps = record_steps(model)
                fast = []
                forget = reference.apply_forget_step
                reference.apply_forget_step = lambda: (forget(), fast.append(fast_state(reference)))
                self.assertEqual(run(model, SEQUENCES[0], ephemeral_update_clamp=0.02),
                                 run(reference, SEQUENCES[0], ephemeral_update_clamp=0.02))
                for mine, theirs in zip(steps, fast):
                    for key in theirs:
                        self.assertTrue(torch.equal(mine["fast"][key], theirs[key]))

    def test_heldout_evaluation(self):
        model = build("sequence")
        run(model, SEQUENCES[0])
        batches = [(["abcda", "dcabc", "bbdac"], F.one_hot(SEQUENCES[1], len(CHARSET)).float())]
        before = {key: value.clone() for key, value in model.state_dict().items()}
        with patch.object(heldout, "first_recall_steps", lambda texts, dataset, steps: torch.tensor([2, 2, 2])), \
                patch.object(heldout, "IntervalMetrics") as metrics:
            metrics.return_value.summary.return_value = {}
            heldout.evaluate_protocols(model, batches, config(), "3_palindrome_dataset_vary_length")
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)
        self.assertEqual(model.pending_slow_steps, 0)


class WindowTest(unittest.TestCase):
    def test_slow_parameters_change_every_n_steps_by_each_sequences_own_sum(self):
        model = build(2)
        model.start_sequence_wipe()
        start = slow_state(model)
        steps = record_steps(model)
        run(model, SEQUENCES[0])  # 4 steps: windows end after steps 2 and 4
        self.assertTrue(torch.equal(steps[0]["slow"]["0.slow"], start["0.slow"]))
        self.assertFalse(torch.equal(steps[1]["slow"]["0.slow"], start["0.slow"]))
        self.assertTrue(torch.equal(steps[2]["slow"]["0.slow"], steps[1]["slow"]["0.slow"]))
        for i, layer in enumerate(model.trained_layers()):
            slow = ~layer.ephemeral_mask
            gradient = sum(torch.einsum("bo,bi->boi", steps[t]["errors"][i], steps[t]["inputs"][i]) for t in (0, 1))
            expected = start[f"{i}.slow"] - LEARNING_RATE * gradient[:, slow]
            torch.testing.assert_close(steps[1]["slow"][f"{i}.slow"], expected, rtol=1e-6, atol=1e-7)
            bias = start[f"{i}.bias"] - LEARNING_RATE * sum(steps[t]["errors"][i].mean(0) for t in (0, 1))
            torch.testing.assert_close(steps[1]["slow"][f"{i}.bias"], bias, rtol=1e-6, atol=1e-7)
        copies = model.trained_layers()[0].per_sample_weights.data[:, ~model.trained_layers()[0].ephemeral_mask]
        self.assertFalse(torch.equal(copies[0], copies[1]))  # each copy took its own sum

    def test_a_partial_window_is_applied_at_the_end_of_the_sequence(self):
        model = build(3)
        steps = record_steps(model)
        run(model, SEQUENCES[0])  # 4 steps: windows of 3 and 1
        self.assertEqual(model.pending_slow_steps, 0)
        self.assertTrue(torch.equal(steps[3]["slow"]["0.slow"], steps[2]["slow"]["0.slow"]))
        self.assertFalse(torch.equal(slow_state(model)["0.slow"], steps[3]["slow"]["0.slow"]))

    def test_a_window_longer_than_the_sequence_is_the_sequence_setting(self):
        long, sequence = build(10), build("sequence")
        for batch in SEQUENCES:
            self.assertAlmostEqual(run(long, batch), run(sequence, batch), places=6)
        long.start_sequence_wipe()
        sequence.start_sequence_wipe()
        for mine, theirs in zip(long.trained_layers(), sequence.trained_layers()):
            torch.testing.assert_close(mine.per_sample_weights, theirs.per_sample_weights, rtol=1e-5, atol=1e-7)
            torch.testing.assert_close(mine.bias, theirs.bias, rtol=1e-5, atol=1e-7)


class CompiledTest(unittest.TestCase):
    def test_compiled_windowed_step_matches_to_rounding(self):
        for setting in ("sequence", 2):
            with self.subTest(setting=setting):
                eager, fused = build(setting, weight_clamp=0.2), build(setting, weight_clamp=0.2)
                fused.enable_fused_update(compile=True)
                try:
                    run(fused, SEQUENCES[0], ephemeral_update_clamp=0.05)
                except Exception as exc:  # no C++ compiler for Inductor's CPU backend, for example
                    self.skipTest(f"torch.compile unavailable here: {type(exc).__name__}: {exc}")
                run(eager, SEQUENCES[0], ephemeral_update_clamp=0.05)
                for mine, theirs in zip(fused.trained_layers(), eager.trained_layers()):
                    torch.testing.assert_close(mine.per_sample_weights, theirs.per_sample_weights, rtol=1e-5, atol=1e-6)


class CheckpointTest(unittest.TestCase):
    def test_resume_refuses_a_changed_setting(self):
        from utils import load_checkpoint, save_checkpoint
        model = build("sequence")
        cfg = {"n_hidden": 4, "n_layers": 2, "updater": "dfa", "model_type": "ephemeral", "slow_update_every": "sequence"}
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            save_checkpoint({"model_state_dict": model.state_dict(), "config": cfg, "iter": 2}, directory, "c.pth")
            path = os.path.join(directory, "c.pth")
            load_checkpoint(path, build("sequence"), cfg)
            for other in (1, 2):
                with self.subTest(other=other), self.assertRaises(RuntimeError):
                    load_checkpoint(path, build(other), {**cfg, "slow_update_every": other})
            old = {key: value for key, value in cfg.items() if key != "slow_update_every"}
            save_checkpoint({"model_state_dict": build().state_dict(), "config": old, "iter": 2}, directory, "old.pth")
            load_checkpoint(os.path.join(directory, "old.pth"), build(), {**old, "slow_update_every": 1})
            with self.assertRaises(RuntimeError):  # a checkpoint without the key was per-step
                load_checkpoint(os.path.join(directory, "old.pth"), build("sequence"), cfg)


if __name__ == "__main__":
    unittest.main()
