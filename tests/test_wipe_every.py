"""--wipe_every N: the fast entries are zeroed only on every N-th sequence; the slow entries are
averaged over the batch at every sequence start either way."""
import contextlib
import io
import unittest

import torch

import heldout
import train as train_module
from tests.test_heldout import TEXTS, build, config, episodes
from tests.test_seed_resume import DATASET


def fast_entries(model):
    return [layer.per_sample_weights.data[:, layer.ephemeral_mask].clone() for layer in model.trained_layers()]


def slow_entries(model):
    return [layer.per_sample_weights.data[:, ~layer.ephemeral_mask].clone() for layer in model.trained_layers()]


def has_fast(model):
    return [layer for layer in model.trained_layers() if layer.ephemeral_mask.any()]


class StartSequenceWipeTest(unittest.TestCase):
    def trained_model(self):
        model = build()
        train_module.train_batch(None, episodes(TEXTS), model, config(), {"training_instance": 0, "log_norms_now": False})
        return model

    def test_slow_average_is_the_same_with_or_without_the_fast_wipe(self):
        wiped, kept = self.trained_model(), self.trained_model()
        before_fast, before_slow = fast_entries(kept), slow_entries(kept)
        self.assertTrue(any(f.abs().sum() > 0 for f in before_fast))
        # The rows differ before the wipe, so averaging is a real change.
        self.assertTrue(any(not torch.equal(s[0], s[1]) for s in before_slow))
        wiped.start_sequence_wipe()
        kept.start_sequence_wipe(wipe_fast=False)
        for s_wiped, s_kept, s_before in zip(slow_entries(wiped), slow_entries(kept), before_slow):
            torch.testing.assert_close(s_kept, s_wiped, rtol=0, atol=0)
            torch.testing.assert_close(s_kept, s_before.mean(0, keepdim=True).expand_as(s_before), rtol=0, atol=0)
        for f_wiped, f_kept, f_before in zip(fast_entries(wiped), fast_entries(kept), before_fast):
            self.assertTrue(torch.equal(f_wiped, torch.zeros_like(f_wiped)))
            torch.testing.assert_close(f_kept, f_before, rtol=0, atol=0)  # per row, not averaged


class TrainBatchWipeEveryTest(unittest.TestCase):
    def run_batches(self, n_batches, wipe_every):
        """Returns, per batch, the fast entries just after its start_sequence_wipe and at its end,
        and the slow entries just after the wipe."""
        model = build()
        cfg = config() if wipe_every is None else config(wipe_every=wipe_every)
        state = {"training_instance": 0, "log_norms_now": False}
        starts, slow_starts, ends, calls = [], [], [], []
        original = model.start_sequence_wipe

        def spy(*args, **kwargs):
            calls.append((args, kwargs))
            original(*args, **kwargs)
            starts.append(fast_entries(model))
            slow_starts.append(slow_entries(model))

        model.start_sequence_wipe = spy
        for _ in range(n_batches):
            train_module.train_batch(None, episodes(TEXTS), model, cfg, state)
            ends.append(fast_entries(model))
        return starts, slow_starts, ends, calls, state

    def test_default_wipes_every_sequence_and_adds_no_state(self):
        for wipe_every in (None, 1):
            starts, _, _, calls, state = self.run_batches(3, wipe_every)
            self.assertEqual(calls, [((), {})] * 3)  # the unchanged call the golden trace records
            self.assertNotIn("sequence_count", state)
            for start in starts:
                self.assertTrue(all(torch.equal(f, torch.zeros_like(f)) for f in start))

    def test_fast_entries_carry_between_wipes_and_the_nth_sequence_wipes(self):
        starts, slow_starts, ends, calls, state = self.run_batches(5, wipe_every=3)
        self.assertEqual([kwargs["wipe_fast"] for _, kwargs in calls], [True, False, False, True, False])
        self.assertEqual(state["sequence_count"], 5)
        for batch in (0, 3):  # wiped
            self.assertTrue(all(torch.equal(f, torch.zeros_like(f)) for f in starts[batch]))
        for batch in (1, 2, 4):  # carried: exactly what the previous sequence's last forget left
            for f_start, f_prev_end in zip(starts[batch], ends[batch - 1]):
                torch.testing.assert_close(f_start, f_prev_end, rtol=0, atol=0)
            self.assertTrue(any(f.abs().sum() > 0 for f in starts[batch]))
        # Slow entries are averaged at every sequence start, wiped or not.
        for slow in slow_starts:
            for s in slow:
                torch.testing.assert_close(s, s[:1].expand_as(s), rtol=0, atol=0)

    def test_carried_fast_entries_decay_at_forget_rate(self):
        """With no error signal (zero learning rate), a carried fast entry only forgets: after a
        T-step sequence it is (1 - forget_rate)^T of its start value."""
        model = build()
        state = {"training_instance": 0, "log_norms_now": False}
        train_module.train_batch(None, episodes(TEXTS), model, config(wipe_every=2), state)
        carried = fast_entries(model)
        train_module.train_batch(None, episodes(TEXTS), model, config(wipe_every=2, learning_rate=0.0), state)
        steps = episodes(TEXTS).shape[1] - 1
        clamp = model.i2h.fast_weight_clamp
        for layer, before, after in zip(model.trained_layers(), carried, fast_entries(model)):
            expected = before.clamp(-clamp, clamp)  # the step's fast clamp; a no-op here after the first
            for _ in range(steps):
                expected = (expected * (1 - model.forget_rate)).clamp(-clamp, clamp)
            torch.testing.assert_close(after, expected)
        self.assertTrue(any(f.abs().sum() > 0 for f in fast_entries(model)))


class HeldoutStartsWipedTest(unittest.TestCase):
    def test_evaluation_starts_from_wiped_fast_entries_and_restores_the_carried_ones(self):
        model = build()
        state = {"training_instance": 0, "log_norms_now": False}
        train_module.train_batch(None, episodes(TEXTS), model, config(wipe_every=4), state)
        carried = fast_entries(model)
        self.assertTrue(any(f.abs().sum() > 0 for f in carried))
        results = heldout.evaluate_protocols(model, [(TEXTS, episodes(TEXTS))], config(), DATASET,
                                             protocols=("no_fast",))
        self.assertTrue(results)
        for before, after in zip(carried, fast_entries(model)):
            torch.testing.assert_close(after, before, rtol=0, atol=0)
        # no_fast leaves the evaluator's fast entries exactly where its wipe put them: zero.
        heldout.evaluate_held_out(model, episodes(TEXTS), None, config())
        self.assertTrue(all(torch.equal(f, torch.zeros_like(f)) for f in fast_entries(model)))


class WipeEveryArgumentsTest(unittest.TestCase):
    def parse_error(self, argv):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            train_module.parse_args(argv)

    def test_arguments(self):
        self.assertEqual(train_module.parse_args([]).wipe_every, 1)
        self.assertEqual(train_module.parse_args(["--wipe_every", "16"]).wipe_every, 16)
        self.parse_error(["--wipe_every", "0"])
        self.parse_error(["--model_type", "rnn", "--wipe_every", "2"])
        self.assertEqual(train_module.parse_args(["--model_type", "rnn", "--wipe_every", "1"]).wipe_every, 1)


if __name__ == "__main__":
    unittest.main()
