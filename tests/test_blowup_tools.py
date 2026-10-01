"""--early_stop_window, --plasticity_schedule and the tracer's h_sat / i2h_pre_norm."""
import contextlib
import io
import os
import tempfile
import unittest
import unittest.mock

import numpy as np
import torch
import torch.nn.functional as F

import train as train_module
from loop_trace import summarize
from tests.test_loop_trace import CHARSET, LEARNING_RATE, SEQUENCES, build, config, run, run_main, weights


class EarlyStopWindowTest(unittest.TestCase):
    def test_default_window_counts_consecutive_high_intervals(self):
        count, stop = 0, False
        for loss in [6.0] * 9:
            count, stop = train_module.high_loss_stop(count, loss, 10)
        self.assertEqual((count, stop), (9, False))
        self.assertEqual(train_module.high_loss_stop(count, 6.0, 10), (10, True))
        self.assertEqual(train_module.high_loss_stop(count, 5.0, 10), (0, False))   # not > 5: resets
        self.assertEqual(train_module.parse_args([]).early_stop_window, 10)

    def test_window_zero_never_stops_on_loss_and_nan_is_not_a_high_loss(self):
        count = 0
        for _ in range(1000):
            count, stop = train_module.high_loss_stop(count, 50.0, 0)
            self.assertFalse(stop)
        self.assertEqual(train_module.high_loss_stop(0, float("nan"), 10), (0, False))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            train_module.parse_args(["--early_stop_window", "-1"])

    def test_training_run_stops_at_the_window_and_continues_with_window_zero(self):
        real_summary = train_module.IntervalMetrics.summary
        high = lambda self: {**real_summary(self), "loss": 9.0}  # noqa: E731
        stopped_at = {}
        for window, n_iters in (("3", "12"), ("0", "12")):
            with tempfile.TemporaryDirectory() as directory, \
                    unittest.mock.patch.object(train_module.IntervalMetrics, "summary", high):
                try:
                    run_main("--early_stop_window", window, "--checkpoint_save_freq", "1",
                             checkpoint_dir=directory, n_iters=n_iters)
                except SystemExit:
                    pass
                stopped_at[window] = torch.load(os.path.join(directory, "latest_checkpoint.pth"),
                                                weights_only=False)["iter"]
        self.assertEqual(stopped_at["3"], 6)    # intervals end at 2, 4, 6; the third high one stops the run at 6 (last save: after 5)
        self.assertEqual(stopped_at["0"], 13)


class ScheduleParsingTest(unittest.TestCase):
    def test_parse_and_lookup(self):
        schedule = train_module.parse_plasticity_schedule("30000:5e3, 0:3e3,80000:7e3")
        self.assertEqual(schedule, [(0, 3e3), (30000, 5e3), (80000, 7e3)])
        self.assertEqual(train_module.parse_plasticity_schedule(""), [])
        at = lambda it: train_module.plasticity_at(schedule, it, 1e4)  # noqa: E731
        self.assertEqual([at(1), at(29999), at(30000), at(79999), at(80000), at(10 ** 6)],
                         [3e3, 3e3, 5e3, 5e3, 7e3, 7e3])
        self.assertEqual(train_module.plasticity_at([(5, 2.0)], 4, 9.0), 9.0)   # --plasticity before the first entry
        for bad in ("1:2,1:3", "-1:2", "x"):
            with self.assertRaises(ValueError):
                train_module.parse_plasticity_schedule(bad)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            train_module.parse_args(["--plasticity_schedule", "0:1,0:2"])


class ScheduledAlphaTest(unittest.TestCase):
    def test_a_changed_alpha_is_used_by_the_next_step_fused_eager_and_unfused(self):
        for fused in (False, True):
            with self.subTest(fused=fused):
                changed = build(fused=fused)
                run(changed, batches=SEQUENCES[:1], trace=False)   # fused_plasticity() is now cached at alpha
                with contextlib.redirect_stdout(io.StringIO()):
                    changed.set_plasticity(11.0)
                # a model constructed with alpha 11, holding the same weights, takes the same next step
                fresh = build(fused=fused, plasticity=11.0)
                fresh.load_state_dict(changed.state_dict())
                stayed = build(fused=fused)
                stayed.load_state_dict(changed.state_dict())
                with torch.no_grad():
                    for layer in (*stayed.linear_layers, stayed.i2h):   # back to the old alpha
                        layer.plasticity.data[layer.ephemeral_mask] = build().i2h.plasticity.data.max()
                        layer._fused_plasticity = None
                for model in (changed, fresh, stayed):
                    run(model, batches=SEQUENCES[1:], trace=False)
                for name, value in weights(changed).items():
                    torch.testing.assert_close(value, weights(fresh)[name], rtol=0, atol=0)
                self.assertFalse(torch.equal(weights(stayed)["0.w"], weights(changed)["0.w"]))
                for layer in (*changed.linear_layers, changed.i2h):
                    self.assertEqual(layer.fused_plasticity(), 11.0)   # still the float (bool-mask) fast path

    @staticmethod
    def read(directory):
        checkpoint = torch.load(os.path.join(directory, "latest_checkpoint.pth"), weights_only=False)
        return checkpoint["config"]["plasticity"], checkpoint["model_state_dict"]

    def test_schedule_in_training_equals_a_resume_with_the_new_plasticity_and_resume_applies_it(self):
        flags = ["--resume", "true"]
        with tempfile.TemporaryDirectory() as scheduled, tempfile.TemporaryDirectory() as manual, \
                tempfile.TemporaryDirectory() as resumed:
            out = run_main(*flags, "--plasticity_schedule", "0:100,3:7", checkpoint_dir=scheduled)
            self.assertIn("alpha 100.0 -> 7.0 at iter 3", out)
            self.assertIn("plasticity: 7.0000", out)
            # the same run by hand: alpha 100 for iterations 1-2, then 7
            run_main(*flags, "--checkpoint_save_freq", "2", checkpoint_dir=manual, n_iters="2")
            run_main(*flags, "--plasticity", "7", checkpoint_dir=manual)
            alpha_scheduled, state_scheduled = self.read(scheduled)
            alpha_manual, state_manual = self.read(manual)
            self.assertEqual((alpha_scheduled, alpha_manual), (7.0, 7.0))
            for name, value in state_scheduled.items():
                self.assertTrue(torch.equal(value, state_manual[name]), name)
            # resume at iteration 3 with a different schedule: the value in force there (7) is applied
            # before the first resumed step
            run_main(*flags, "--checkpoint_save_freq", "2", checkpoint_dir=resumed, n_iters="2")
            out = run_main(*flags, "--plasticity_schedule", "0:50,2:7", checkpoint_dir=resumed)
            self.assertIn("Plasticity schedule: alpha 100.0 -> 7.0 at iter 3", out)
            self.assertEqual(out.count("Plasticity schedule:"), 1)
            _, state_resumed = self.read(resumed)
            for name, value in state_manual.items():
                self.assertTrue(torch.equal(value, state_resumed[name]), name)

    def test_without_a_schedule_nothing_changes_or_prints(self):
        with tempfile.TemporaryDirectory() as directory:
            out = run_main(checkpoint_dir=directory)
        self.assertNotIn("Plasticity schedule", out)
        self.assertNotIn("  plasticity:", out)


def reference_saturation(model, batch):
    """h_sat and i2h_pre_norm for one batch in float64 numpy, advancing the model by its own step."""
    onehot = F.one_hot(batch, len(CHARSET)).float()
    criterion = config()["criterion"]
    model.start_sequence_wipe()
    hidden = model.initHidden(batch.shape[0])
    sat, norm, from_hidden = [], [], []
    for i in range(onehot.shape[1] - 1):
        logits, hidden_next = model(onehot[:, i], hidden.detach())
        x = model.i2h.in_traces.data.double().numpy()
        w = model.i2h.per_sample_weights.data.double().numpy()
        pre = np.einsum("boi,bi->bo", w, x) + model.i2h.bias.data.double().numpy()
        sat.append((np.abs(np.tanh(pre)) > 0.99).mean(1))
        norm.append(np.linalg.norm(pre, axis=1))
        from_hidden.append((hidden_next.detach().abs() > 0.99).float().mean(1).numpy())
        _, output_error = train_module.dfa_output_error(logits, onehot[:, i + 1], criterion)
        model.fused_dfa_step(output_error, LEARNING_RATE, 0)
        hidden = hidden_next
    return np.stack(sat), np.stack(norm), np.stack(from_hidden)


def saturated_build(**options):
    model = build(**options)
    with torch.no_grad():
        model.i2h.per_sample_weights.mul_(20.0)   # drive the tanh state into saturation for some units
        model.i2h.bias.mul_(20.0)
    return model


class SaturationTraceTest(unittest.TestCase):
    def test_h_sat_and_i2h_pre_norm_match_an_independent_computation(self):
        for recurrence in (True, False):
            with self.subTest(recurrence=recurrence):
                sat, norm, from_hidden = reference_saturation(
                    saturated_build(fused=True, enable_recurrence=recurrence), SEQUENCES[0])
                _, (traces, _) = run(saturated_build(enable_recurrence=recurrence))
                self.assertTrue(0 < sat.mean() < 1, "the test needs partial saturation")
                np.testing.assert_allclose(traces["h_sat"].numpy(), sat, atol=1e-6)
                np.testing.assert_allclose(traces["i2h_pre_norm"].numpy(), norm, rtol=2e-4, atol=1e-6)
                if recurrence:   # the state fed to the next step
                    np.testing.assert_allclose(traces["h_sat"].numpy(), from_hidden, atol=1e-6)
                else:            # still measured, from the pre-activation
                    self.assertTrue((traces["hidden_norm"] == 0).all())
                self.assertEqual(traces["h_sat"].shape, (4, 3))

    def test_summary_carries_them(self):
        _, (traces, _) = run(saturated_build())
        summary = summarize(traces)
        self.assertAlmostEqual(summary["trace/h_sat_mean"], float(traces["h_sat"].mean()), places=6)
        self.assertAlmostEqual(summary["trace/i2h_pre_norm_last"], float(traces["i2h_pre_norm"][-1].mean()), places=4)


if __name__ == "__main__":
    unittest.main()
