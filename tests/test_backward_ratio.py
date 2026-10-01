"""--fast_backward_per_forward: backward (DFA) passes per forward pass for the fast entries.

1 is today's one step per character (unchanged; the golden traces pin it). K >= 2 re-runs the
forward pass on the same character K - 1 more times and takes a fast-only step each time, without
forgetting; 1/N gives only every N-th character a fast update. The slow stream is the first pass's
in every setting."""
import contextlib
import io
import os
import tempfile
import unittest

import torch
import torch.nn.functional as F

from ephemeral_model import (EphemeralRNN, dfa_layer_step, dfa_output_error,
                             parse_fast_backward_per_forward)
from reproducibility import seed_everything
from train import parse_args, train

CHARSET = list("abcd")
BATCH = 3
SEQUENCES = (torch.tensor([[0, 1, 2, 3, 1], [3, 2, 0, 1, 2], [1, 1, 3, 0, 2]]),   # 4 steps each
             torch.tensor([[2, 0, 3, 1, 2], [1, 3, 2, 0, 1], [0, 0, 1, 2, 3]]))
LEARNING_RATE = 0.3
FORGET = 0.25
CRITERION = torch.nn.CrossEntropyLoss(reduction="none")


def build(ratio=1, slow_update_every=1, **options):
    settings = dict(updater="dfa", plasticity=3.0, batch_size=BATCH,
                    forget_rate=FORGET, ephemeral_fraction=0.5, enable_recurrence=True)
    settings.update(options)
    seed_everything(11, deterministic=True)
    with contextlib.redirect_stdout(io.StringIO()):
        return EphemeralRNN(len(CHARSET), 4, len(CHARSET), 2, CHARSET, slow_update_every=slow_update_every,
                            fast_backward_per_forward=ratio, **settings)


def config(**options):
    return {"updater": "dfa", "criterion": CRITERION, "input_mode": "last_one", "pe_matrix": None,
            "learning_rate": LEARNING_RATE, "ephemeral_update_clamp": 0, "grad_norm_clip": 0, **options}


def run(model, batch, state=None, **options):
    state = state if state is not None else {"training_instance": 0, "log_norms_now": False}
    with contextlib.redirect_stdout(io.StringIO()):
        _, loss, *_ = train(batch, F.one_hot(batch, len(CHARSET)).float(), model, config(**options), state)
    return loss


def weights(model):
    return [layer.per_sample_weights.data.clone() for layer in model.trained_layers()]


def biases(model):
    return [layer.bias.data.clone() for layer in model.trained_layers()]


def slow_state(model):
    state = []
    for layer in model.trained_layers():
        state += [layer.per_sample_weights.data[:, ~layer.ephemeral_mask].clone(), layer.bias.data.clone()]
    return state


def fast_state(model):
    return [layer.per_sample_weights.data[:, layer.ephemeral_mask].clone()
            for layer in model.trained_layers() if not layer.is_last_layer]


def slow_state_of(weights_list, biases_list, model):
    state = []
    for layer, w, b in zip(model.trained_layers(), weights_list, biases_list):
        state += [w[:, ~layer.ephemeral_mask].clone(), b.clone()]
    return state


def same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b))


def close(a, b):
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-6)


def default_step(model, error):
    """The per-step unfused DFA step of train.py's default branch."""
    state = {"training_instance": 0, "log_norms_now": False}
    model.clear_dfa_gradients()
    for layer in model.trained_layers():
        layer.populate_dfa_gradients(error)
    for layer in model.trained_layers():
        layer.apply_update(LEARNING_RATE, 0, state)
    model.apply_forget_step()
    model.clear_dfa_gradients()


def reference(model, batches, iterations=1, subsample=1, wipe_every=1):
    """train_batch written out by hand for slow_update_every 1: the default step on pass 1 (on a
    skipped character the same step with the fast entries put back to their forgotten old value),
    then iterations - 1 re-forwards, each with a fast-only step and no forgetting. The extra
    passes see the slow weights from before pass 1's step: the step's slow entries and biases are
    set aside, the old ones put back for the passes, and the step's ones restored after them."""
    count, losses = 0, []
    for batch in batches:
        model.start_sequence_wipe(wipe_fast=count % wipe_every == 0)
        count += 1
        onehot = F.one_hot(batch, len(CHARSET)).float()
        hidden = model.initHidden(batch.shape[0])
        for i in range(batch.shape[1] - 1):
            x, target, incoming = onehot[:, i, :], onehot[:, i + 1, :], hidden.detach()
            output, hidden = model(x, incoming)
            loss, error = dfa_output_error(output, target, CRITERION)
            losses.append(loss.detach())
            before, before_bias = weights(model), biases(model)
            default_step(model, error)
            stepped = weights(model), biases(model)
            for layer, old in zip(model.trained_layers(), before):  # pre-step slow weights for the passes
                slow = ~layer.ephemeral_mask
                layer.per_sample_weights.data[:, slow] = old[:, slow]
            for layer, bias in zip(model.trained_layers(), before_bias):
                layer.bias.data.copy_(bias)
            if i % subsample != 0:
                for layer, old in zip(model.trained_layers(), before):
                    layer.per_sample_weights.data[:, layer.ephemeral_mask] = \
                        old[:, layer.ephemeral_mask] * (1 - FORGET)
            with torch.no_grad():
                for _ in range(iterations - 1):
                    output, hidden = model(x, incoming)
                    _, error = dfa_output_error(output, target, CRITERION)
                    projected, _ = model.dfa_step_errors(error, 0)
                    for layer, p in zip(model.trained_layers(), projected):
                        if not layer.is_last_layer:
                            dfa_layer_step(layer.per_sample_weights.data, None, p, layer.in_traces.data,
                                           layer.fused_plasticity(), layer.ephemeral_mask, 0.0,
                                           LEARNING_RATE, 0, layer.weight_clamp, False, 0.0,
                                           layer.fast_weight_clamp, True)
            for layer, new, new_bias in zip(model.trained_layers(), *stepped):
                slow = ~layer.ephemeral_mask
                layer.per_sample_weights.data[:, slow] = new[:, slow]
                layer.bias.data.copy_(new_bias)
    return torch.stack(losses).mean().item()


class ParseTest(unittest.TestCase):
    def test_values(self):
        self.assertEqual(parse_fast_backward_per_forward("1"), 1)
        self.assertEqual(parse_fast_backward_per_forward(3), 3)
        self.assertEqual(parse_fast_backward_per_forward("1/4"), "1/4")
        for bad in ("0", "-2", "2.5", "1/1", "1/0", "2/3", "1/2.5", "x", "", True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_fast_backward_per_forward(bad)

    def test_parser(self):
        self.assertEqual(parse_args([]).fast_backward_per_forward, 1)
        self.assertEqual(parse_args(["--fast_backward_per_forward", "1/3"]).fast_backward_per_forward, "1/3")
        self.assertEqual(parse_args(["--fast_backward_per_forward", "2"]).fast_backward_per_forward, 2)
        for extra in (["--updater", "backprop"], ["--updater", "bptt"], ["--model_type", "rnn"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(["--fast_backward_per_forward", "2", *extra])
        for bad in ("0", "1/1", "1/0", "2.5"):
            with self.subTest(bad=bad), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--fast_backward_per_forward", bad])
        with self.assertRaises(ValueError):
            build(2, updater="backprop")


class DefaultIsUnchangedTest(unittest.TestCase):
    def test_ratio_one_is_bit_identical(self):
        for slow in (1, 3, "sequence"):
            with self.subTest(slow=slow):
                default, explicit = build(slow_update_every=slow), build("1", slow_update_every=slow)
                for batch in SEQUENCES:
                    self.assertEqual(run(default, batch), run(explicit, batch))
                self.assertTrue(same(weights(default), weights(explicit)))
                self.assertTrue(same(biases(default), biases(explicit)))


class IterationsTest(unittest.TestCase):
    def test_k2_and_k3_match_reference(self):
        for k in (2, 3):
            with self.subTest(k=k):
                mine, ref = build(k), build()
                self.assertEqual(run(mine, SEQUENCES[0]), reference(ref, SEQUENCES[:1], iterations=k))
                self.assertTrue(same(weights(mine), weights(ref)))
                self.assertTrue(same(biases(mine), biases(ref)))

    def test_extra_pass_changes_only_fast_entries_and_loss_is_first_pass(self):
        batch = SEQUENCES[0][:, :2]  # one character
        once, twice = build(1), build(2)
        self.assertEqual(run(once, batch), run(twice, batch))
        self.assertTrue(same(slow_state(once), slow_state(twice)))
        self.assertFalse(same(fast_state(once), fast_state(twice)))

    def test_forgetting_once_per_character(self):
        # lr 0: nothing is learned, so each character only forgets, once, however many passes it has.
        for k in (1, 3):
            with self.subTest(k=k):
                model = build(k)
                for layer in model.trained_layers():
                    layer.per_sample_weights.data[:, layer.ephemeral_mask] = 1.0
                # sequence_count 1 of --wipe_every 2: the fast entries carry into this sequence
                run(model, SEQUENCES[0], {"sequence_count": 1, "training_instance": 0, "log_norms_now": False},
                    wipe_every=2, learning_rate=0.0)
                for fast in fast_state(model):
                    torch.testing.assert_close(fast, torch.full_like(fast, (1 - FORGET) ** 4))

    def test_hidden_state_is_the_final_pass(self):
        model = build(2)
        calls = []
        original = model.forward

        def recording(x, h):
            result = original(x, h)
            calls.append((h.clone(), result[1].clone()))
            return result
        model.forward = recording
        run(model, SEQUENCES[0][:, :3])  # two characters: four forwards
        self.assertEqual(len(calls), 4)
        self.assertTrue(torch.equal(calls[0][0], calls[1][0]))   # same incoming state for both passes
        self.assertTrue(torch.equal(calls[2][0], calls[1][1]))   # the next character reads the last pass
        self.assertTrue(torch.equal(calls[2][0], calls[3][0]))

    def test_slow_entries_constant_within_a_sequence(self):
        model = build(3, slow_update_every="sequence")
        steps, accumulated = [], []
        original = model.windowed_dfa_step

        def recording(*args, **kwargs):
            original(*args, **kwargs)
            steps.append(slow_state(model))
        model.windowed_dfa_step = recording
        for layer in model.trained_layers():
            def counting(*args, _f=layer.accumulate_slow_gradient):
                accumulated.append(1)
                return _f(*args)
            layer.accumulate_slow_gradient = counting
        before = slow_state(model)
        run(model, SEQUENCES[0])
        self.assertEqual(len(steps), 4)
        for step in steps[:-1]:
            self.assertTrue(same(step, before))
        self.assertEqual(len(accumulated), 4 * len(model.trained_layers()))  # the first pass only
        self.assertFalse(same(slow_state(model), before))

    def test_one_character_slow_step_equals_k1(self):
        batch = SEQUENCES[0][:, :2]
        a, b = build(1, "sequence"), build(3, "sequence")
        run(a, batch), run(b, batch)
        self.assertTrue(same(slow_state(a), slow_state(b)))

    def test_slow_stream_does_not_depend_on_the_ratio(self):
        # alpha = 0: the fast entries stay zero whatever they are given, so any difference in the
        # slow state would come from the extra passes or skips touching slow parameters.
        for slow in (1, 2, "sequence"):
            base = build(1, slow, plasticity=0.0)
            for batch in SEQUENCES:
                run(base, batch)
            for ratio in (3, "1/3"):
                with self.subTest(slow=slow, ratio=ratio):
                    model = build(ratio, slow, plasticity=0.0)
                    for batch in SEQUENCES:
                        run(model, batch)
                    self.assertTrue(same(slow_state(model), slow_state(base)))

    def test_clip_statistics_record_the_first_pass_only(self):
        model = build(3)
        calls = []
        original = model.grad_clip_stats.record
        model.grad_clip_stats.record = lambda *a: (calls.append(1), original(*a))[1]
        run(model, SEQUENCES[0], grad_norm_clip=0.5)
        self.assertEqual(len(calls), 4)


class SubsampleTest(unittest.TestCase):
    def test_every_nth_character_matches_reference(self):
        for n in (2, 3):
            with self.subTest(n=n):
                mine, ref = build(f"1/{n}"), build()
                loss = run(mine, SEQUENCES[0])
                self.assertAlmostEqual(loss, reference(ref, SEQUENCES[:1], subsample=n), places=6)
                close(weights(mine), weights(ref))

    def test_skipped_characters_only_forget_fast_entries_under_windows(self):
        model = build("1/2", slow_update_every="sequence")
        fasts = []
        original = model.windowed_dfa_step

        def recording(*args, **kwargs):
            original(*args, **kwargs)
            fasts.append(fast_state(model))
        model.windowed_dfa_step = recording
        run(model, SEQUENCES[0])
        self.assertEqual(len(fasts), 4)
        for t in (1, 3):  # skipped: exactly the previous fast state, forgotten
            for now, before in zip(fasts[t], fasts[t - 1]):
                self.assertTrue(torch.equal(now, before * (1 - FORGET)))
        for now, before in zip(fasts[2], fasts[1]):  # updated: more than forgetting
            self.assertFalse(torch.equal(now, before * (1 - FORGET)))

    def test_counter_restarts_every_sequence(self):
        model = build("1/3")
        fired = []
        original = model.skip_fast_dfa_step
        model.skip_fast_dfa_step = lambda *a, **k: (fired.append(1), original(*a, **k))[1]
        run(model, SEQUENCES[0])  # 4 characters: 0 updated, 1 and 2 skipped, 3 updated
        run(model, SEQUENCES[1])
        self.assertEqual(len(fired), 4)


class SplitStepTest(unittest.TestCase):
    """K >= 2 splits a character's step: fast half in pass 1, slow half after the extra passes."""

    def forced(self, **options):
        model = build(**options)
        model._force_split_step = True
        return model

    def test_split_equals_fused_step_at_k1(self):
        cases = [
            ({}, {}),
            ({"slow_weight_decay": 0.1}, {}),
            ({"weight_clamp": 0.3}, {}),
            ({"fast_weight_clamp": 0.2}, {}),
            ({}, {"ephemeral_update_clamp": 0.05}),
            ({}, {"grad_norm_clip": 0.5}),
            ({"dfa_fprime": True}, {}),
            ({"layer_norm": True}, {}),
        ]
        for model_options, run_options in cases:
            for fused in (False, True):
                with self.subTest(model=model_options, run=run_options, fused=fused):
                    default, split = build(**model_options), self.forced(**model_options)
                    if fused:
                        default.enable_fused_update(compile=False)
                        split.enable_fused_update(compile=False)
                    for batch in SEQUENCES:
                        loss_default = run(default, batch, **run_options)
                        loss_split = run(split, batch, **run_options)
                        self.assertEqual(loss_default, loss_split)
                    check = same if (fused or "grad_norm_clip" not in run_options) else close
                    for a, b in ((weights(default), weights(split)), (biases(default), biases(split))):
                        check(a, b)

    def test_split_equals_fused_step_with_wipe_every(self):
        default, split = build(), self.forced()
        for model in (default, split):
            state = {"training_instance": 0, "log_norms_now": False}
            for batch in SEQUENCES:
                run(model, batch, state, wipe_every=4)
        self.assertTrue(same(weights(default), weights(split)))
        self.assertTrue(same(biases(default), biases(split)))

    def test_split_windowed_equals_windowed_at_k1(self):
        for slow in (2, "sequence"):
            with self.subTest(slow=slow):
                default, split = build(slow_update_every=slow), self.forced(slow_update_every=slow)
                for batch in SEQUENCES:
                    run(default, batch), run(split, batch)
                self.assertTrue(same(weights(default), weights(split)))
                self.assertTrue(same(biases(default), biases(split)))

    def instrument(self, model):
        """Records each forward's slow state, and each slow step's pre-state and saved pass-1 inputs."""
        log = {"forwards": [], "steps": []}
        original_forward, original_split, original_apply = model.forward, model.split_dfa_step, model.apply_slow_step
        pending = {}

        def forward(x, h):
            log["forwards"].append(slow_state(model))
            return original_forward(x, h)

        def split(*args, **kwargs):
            saved = original_split(*args, **kwargs)
            pending["pre"] = (weights(model), biases(model), saved)
            return saved

        def apply(saved, lr, clamp):
            original_apply(saved, lr, clamp)
            log["steps"].append((*pending["pre"], weights(model), biases(model)))
        model.forward, model.split_dfa_step, model.apply_slow_step = forward, split, apply
        return log

    def test_per_step_slow_step_is_pass_ones_and_all_passes_see_pre_step_weights(self):
        options = dict(slow_weight_decay=0.05, weight_clamp=0.9)
        model = build(3, **options)
        log = self.instrument(model)
        run(model, SEQUENCES[0], grad_norm_clip=0.5)
        self.assertEqual(len(log["forwards"]), 12)
        for c in range(4):  # the three passes of a character read identical slow weights
            self.assertTrue(same(log["forwards"][3 * c], log["forwards"][3 * c + 1]))
            self.assertTrue(same(log["forwards"][3 * c], log["forwards"][3 * c + 2]))
        self.assertEqual(len(log["steps"]), 4)
        for c, (w0, b0, saved, w1, b1) in enumerate(log["steps"]):
            self.assertTrue(same(slow_state_of(w0, b0, model), log["forwards"][3 * c]))
            oracle = build(**options)  # a K = 1 step from the same pass-1 errors and inputs
            for layer, w, b, (error, inputs) in zip(oracle.trained_layers(), w0, b0, saved):
                layer.per_sample_weights.data.copy_(w)
                layer.bias.data.copy_(b)
                dfa_layer_step(layer.per_sample_weights.data, layer.bias.data, error, inputs,
                               layer.fused_plasticity(), layer.ephemeral_mask, layer.forget_rate,
                               LEARNING_RATE, 0, layer.weight_clamp, layer.is_last_layer,
                               layer.slow_weight_decay, layer.fast_weight_clamp)
            self.assertTrue(same(slow_state_of(w1, b1, model), slow_state(oracle)))

    def test_window_slow_weights_change_only_between_characters(self):
        for slow in (2, 3):
            with self.subTest(slow=slow):
                model = build(3, slow_update_every=slow, slow_weight_decay=0.05)
                forwards = []
                original = model.forward
                model.forward = lambda x, h: (forwards.append(slow_state(model)), original(x, h))[1]
                accumulated = []
                for layer in model.trained_layers():
                    def counting(*args, _f=layer.accumulate_slow_gradient):
                        accumulated.append(1)
                        return _f(*args)
                    layer.accumulate_slow_gradient = counting
                run(model, SEQUENCES[0])
                for c in range(4):
                    self.assertTrue(same(forwards[3 * c], forwards[3 * c + 1]))
                    self.assertTrue(same(forwards[3 * c], forwards[3 * c + 2]))
                self.assertEqual(len(accumulated), 4 * len(model.trained_layers()))
                # the window closing at character slow - 1 is applied after that character's passes
                close = slow - 1
                self.assertFalse(same(forwards[3 * close], forwards[3 * (close + 1)]))


class CombinationsTest(unittest.TestCase):
    def check(self, **options):
        for ratio, iterations, subsample in ((2, 2, 1), ("1/2", 1, 2)):
            with self.subTest(options=options, ratio=ratio):
                mine, ref = build(ratio, **options), build(**options)
                run(mine, SEQUENCES[0])
                reference(ref, SEQUENCES[:1], iterations, subsample)
                close(weights(mine), weights(ref))

    def test_dfa_fprime(self):
        self.check(dfa_fprime=True)

    def test_layer_norm(self):
        self.check(layer_norm=True)

    def test_clamps(self):
        self.check(weight_clamp=0.8, fast_weight_clamp=0.5)

    def test_wipe_every(self):
        state = {"training_instance": 0, "log_norms_now": False}
        mine, ref = build(2), build()
        for batch in SEQUENCES:
            run(mine, batch, state, wipe_every=4)
        reference(ref, SEQUENCES, iterations=2, wipe_every=4)
        close(weights(mine), weights(ref))

    def test_compiled_matches_eager_for_k2(self):
        eager, fused = build(2), build(2)
        fused.enable_fused_update(compile=True)
        try:
            run(fused, SEQUENCES[0])
        except Exception as exc:  # no C++ compiler for Inductor's CPU backend, for example
            self.skipTest(f"torch.compile unavailable here: {type(exc).__name__}: {exc}")
        run(eager, SEQUENCES[0])
        close(weights(fused), weights(eager))

    def test_fused_eager_matches_unfused(self):
        for ratio in (2, "1/2"):
            with self.subTest(ratio=ratio):
                eager, fused = build(ratio), build(ratio)
                fused.enable_fused_update(compile=False)
                run(eager, SEQUENCES[0]), run(fused, SEQUENCES[0])
                close(weights(fused), weights(eager))


class CheckpointTest(unittest.TestCase):
    def test_resume_refuses_a_changed_setting(self):
        from utils import load_checkpoint, save_checkpoint
        cfg = {"n_hidden": 4, "n_layers": 2, "updater": "dfa", "model_type": "ephemeral",
               "fast_backward_per_forward": 2}
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            save_checkpoint({"model_state_dict": build(2).state_dict(), "config": cfg, "iter": 2}, directory, "c.pth")
            path = os.path.join(directory, "c.pth")
            load_checkpoint(path, build(2), cfg)
            for other in (1, 3, "1/2"):
                with self.subTest(other=other), self.assertRaises(RuntimeError):
                    load_checkpoint(path, build(other), {**cfg, "fast_backward_per_forward": other})
            old = {key: value for key, value in cfg.items() if key != "fast_backward_per_forward"}
            save_checkpoint({"model_state_dict": build().state_dict(), "config": old, "iter": 2}, directory, "old.pth")
            load_checkpoint(os.path.join(directory, "old.pth"), build(), {**old, "fast_backward_per_forward": 1})
            with self.assertRaises(RuntimeError):  # a checkpoint without the key was 1:1
                load_checkpoint(os.path.join(directory, "old.pth"), build(2), cfg)


if __name__ == "__main__":
    unittest.main()
