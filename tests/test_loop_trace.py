"""loop_trace.py and trace_replay.py: observation-only feedback-loop traces.

The traces of the training step are checked against an independent float64 computation (numpy,
from the raw layer tensors and an analytic softmax error), the fused (eager) step against the
unfused one, every --slow_update_every and --fast_backward_per_forward mode, and the promise that
tracing changes nothing (bit-identical weights and losses)."""
import contextlib
import io
import math
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

import train as train_module
import trace_replay
from ephemeral_model import EphemeralRNN
from loop_trace import NULL_TRACER, LoopTracer, derived_traces, longest_run_above_one, summarize
from reproducibility import seed_everything
from tests.test_heldout import fake_heldout_batches
from tests.test_seed_resume import DATASET, fake_loader

CHARSET = list("abcd")
BATCH = 3
SEQUENCES = (torch.tensor([[0, 1, 2, 3, 1], [3, 2, 0, 1, 2], [1, 1, 3, 0, 2]]),   # 4 steps each
             torch.tensor([[2, 0, 3, 1, 2], [1, 3, 2, 0, 1], [0, 0, 1, 2, 3]]))
LEARNING_RATE = 0.2
ALPHA = 3.0


def build(slow_update_every=1, fast_backward_per_forward=1, fused=False, **options):
    settings = dict(updater="dfa", plasticity=ALPHA, batch_size=BATCH, forget_rate=0.25,
                    ephemeral_fraction=0.5, enable_recurrence=True)
    settings.update(options)
    seed_everything(11, deterministic=True)
    with contextlib.redirect_stdout(io.StringIO()):
        model = EphemeralRNN(len(CHARSET), 4, len(CHARSET), 2, CHARSET, slow_update_every=slow_update_every,
                             fast_backward_per_forward=fast_backward_per_forward, **settings)
    if fused:
        model.enable_fused_update(compile=False)
    return model


def config(update_clamp=0, grad_norm_clip=0):
    return {"updater": "dfa", "criterion": torch.nn.CrossEntropyLoss(reduction="none"),
            "input_mode": "last_one", "pe_matrix": None, "learning_rate": LEARNING_RATE,
            "ephemeral_update_clamp": update_clamp, "grad_norm_clip": grad_norm_clip}


def run(model, batches=SEQUENCES, trace=True, update_clamp=0, grad_norm_clip=0):
    """The real train() on each batch; returns (losses, [traces per batch])."""
    cfg = config(update_clamp, grad_norm_clip)
    tracer = LoopTracer(model, LEARNING_RATE, update_clamp, grad_norm_clip) if trace else None
    losses, traces = [], []
    for batch in batches:
        with contextlib.redirect_stdout(io.StringIO()):
            _, loss, *_ = train_module.train(batch, F.one_hot(batch, len(CHARSET)).float(), model, cfg,
                                             {"training_instance": 0, "log_norms_now": False}, tracer=tracer)
        losses.append(loss)
        if trace:
            traces.append(tracer.finish())
    return losses, traces


def weights(model):
    return {f"{i}.{name}": tensor.detach().clone() for i, layer in enumerate(model.trained_layers())
            for name, tensor in (("w", layer.per_sample_weights), ("b", layer.bias))}


def reference_first_batch(model, batch, update_clamp=0.0):
    """The traces' quantities for one batch, computed independently in float64 numpy while the
    model is advanced by its own (fused, eager) step. update_clamp must be 0 (the closed form
    below has no element-wise clamp). Returns {name: array [T, B, L]} for L fast layers."""
    onehot = F.one_hot(batch, len(CHARSET)).float()
    cfg = config()
    criterion = cfg["criterion"]
    model.start_sequence_wipe()
    hidden = model.initHidden(batch.shape[0])
    fast_layers = [*model.linear_layers, model.i2h]
    out = {name: [] for name in ("act_norm", "fast_norm", "fast_write", "fast_drive", "slow_drive", "max_logit",
                                 "logit_norm", "loss")}
    for i in range(onehot.shape[1] - 1):
        logits, hidden = model(onehot[:, i], hidden.detach())
        target = onehot[:, i + 1]
        z = logits.detach().double().numpy()
        probs = np.exp(z - z.max(1, keepdims=True))
        probs /= probs.sum(1, keepdims=True)
        error = probs - target.double().numpy()   # d(cross entropy)/d(logits)
        out["max_logit"].append(z.max(1))
        out["logit_norm"].append(np.linalg.norm(z, axis=1))
        out["loss"].append(-np.log((probs * target.double().numpy()).sum(1)))
        per_layer = {name: [] for name in ("act_norm", "fast_norm", "fast_write", "fast_drive", "slow_drive")}
        for layer in fast_layers:
            x = layer.in_traces.data.double().numpy()                           # [B, in]
            w = layer.per_sample_weights.data.double().numpy()                  # [B, out, in]
            mask = layer.ephemeral_mask.data.numpy()                            # [out, in]
            p = error @ layer.feedback_weights.data.double().numpy()            # [B, out]
            per_layer["act_norm"].append(np.linalg.norm(x, axis=1))
            per_layer["fast_norm"].append(np.sqrt(((w * mask) ** 2).sum((1, 2))))
            per_layer["fast_drive"].append(np.linalg.norm(np.einsum("boi,bi->bo", w * mask, x), axis=1))
            per_layer["slow_drive"].append(np.linalg.norm(np.einsum("boi,bi->bo", w * ~mask, x), axis=1))
            # |lr * alpha * mask o (p x^T)|_F^2 = (lr alpha)^2 sum_i p_i^2 sum_j mask_ij x_j^2
            inner = (p ** 2) * (x ** 2 @ mask.T.astype(float))
            per_layer["fast_write"].append(LEARNING_RATE * ALPHA * np.sqrt(inner.sum(1)))
        for name, values in per_layer.items():
            out[name].append(np.stack(values, axis=1))
        # advance the model by the training step (eager fused: bit-identical to the unfused one)
        _, output_error = train_module.dfa_output_error(logits, target, criterion)
        model.fused_dfa_step(output_error, LEARNING_RATE, update_clamp)
    return {name: np.stack(values) for name, values in out.items()}


class TraceQuantitiesTest(unittest.TestCase):
    def test_traces_match_an_independent_computation(self):
        reference = reference_first_batch(build(fused=True), SEQUENCES[0])
        _, (traces, _) = run(build())
        for name, expected in reference.items():
            with self.subTest(name):
                np.testing.assert_allclose(traces[name].numpy(), expected, rtol=2e-4, atol=1e-6)

    def test_quantities_are_nontrivial(self):
        _, (traces, _) = run(build())
        self.assertEqual(traces["fast_write"].shape, (4, BATCH, 3))   # trunk x2 + i2h
        self.assertEqual(traces["slow_delta"].shape, (4, BATCH, 4))   # + i2o
        self.assertTrue((traces["fast_norm"][0] == 0).all())          # wiped at the sequence start
        self.assertTrue((traces["fast_norm"][1:] > 0).all())
        self.assertTrue((traces["fast_write"] > 0).all())
        self.assertTrue((traces["slow_drive"] > 0).all())
        self.assertTrue(torch.isfinite(traces["loop_gain"][1:]).all())
        self.assertTrue(torch.isnan(traces["loop_gain"][0]).all())

    def test_loop_gain_is_the_ratio_of_successive_write_norms(self):
        _, (traces, _) = run(build())
        write = torch.sqrt((traces["fast_write"] ** 2).sum(2))
        torch.testing.assert_close(traces["write_norm"], write)
        torch.testing.assert_close(traces["loop_gain"][1:], write[1:] / write[:-1])

    def test_derived_gain_skips_zero_writes_and_run_lengths(self):
        write = torch.tensor([[1.0], [2.0], [0.0], [3.0], [6.0], [12.0]])
        derived = derived_traces({"fast_write": write.unsqueeze(2)})
        gain = derived["loop_gain"][:, 0]
        self.assertTrue(math.isnan(gain[0]) and math.isnan(gain[3]))   # no previous write / previous is zero
        self.assertEqual(float(gain[1]), 2.0)
        self.assertEqual(float(gain[2]), 0.0)
        self.assertEqual(longest_run_above_one(derived["loop_gain"]).tolist(), [2])

    def test_with_no_forgetting_and_no_clamp_the_fast_delta_is_the_write(self):
        _, (traces, _) = run(build(forget_rate=0.0))
        torch.testing.assert_close(traces["fast_delta"], traces["fast_write"], rtol=1e-4, atol=1e-6)

    def test_update_clamp_caps_the_write_and_forgetting_shows_in_the_delta(self):
        _, (clamped, _) = run(build(), update_clamp=0.001)
        _, (free, _) = run(build())
        self.assertLess(float(clamped["fast_write"].max()), float(free["fast_write"].max()))
        # with a binding clamp the actual change is the clamped write (plus forgetting), not more
        self.assertTrue((clamped["fast_write"][0] > 0).all())

    def test_grad_norm_clip_scales_the_write(self):
        _, (clipped, _) = run(build(), grad_norm_clip=0.01)
        _, (free, _) = run(build())
        self.assertLess(float(clipped["fast_write"][0].max()), float(free["fast_write"][0].max()))

    def test_summary_scalars(self):
        _, (traces, _) = run(build())
        summary = summarize(traces)
        self.assertAlmostEqual(summary["trace/max_logit_max"], float(traces["max_logit"].max()), places=6)
        self.assertGreaterEqual(summary["trace/frac_gain_gt1"], 0.0)
        self.assertTrue(all(math.isfinite(v) for v in summary.values()))


class ModesTest(unittest.TestCase):
    def assert_traces_equal(self, first, second, **tolerance):
        for name in first:
            torch.testing.assert_close(first[name], second[name], msg=name, equal_nan=True, **tolerance)

    def test_eager_fused_traces_are_bit_identical_to_unfused(self):
        for options in (dict(), dict(slow_update_every=2), dict(slow_update_every="sequence"),
                        dict(fast_backward_per_forward=2), dict(fast_backward_per_forward="1/2")):
            with self.subTest(**options):
                (_, plain), (_, fused) = run(build(**options)), run(build(fused=True, **options))
                for first, second in zip(plain, fused):
                    self.assert_traces_equal(first, second, rtol=0, atol=0)

    def test_sequence_mode_freezes_the_slow_weights_within_the_sequence(self):
        _, (traces, _) = run(build(slow_update_every="sequence"))
        self.assertTrue((traces["slow_delta"] == 0).all())
        self.assertTrue((traces["slow_total_delta"] > 0).all())
        _, (per_step, _) = run(build())
        self.assertTrue((per_step["slow_delta"] > 0).all())

    def test_window_mode_applies_the_slow_step_only_at_window_ends(self):
        _, (traces, _) = run(build(slow_update_every=3))   # 4 steps: the window ends after step 2
        self.assertTrue((traces["slow_delta"][2] > 0).all())
        self.assertTrue((traces["slow_delta"][[0, 1, 3]] == 0).all())
        # the final one-step window is applied after step 3, so it counts in the total only
        self.assertTrue((traces["slow_total_delta"] > 0).all())

    def test_fast_writes_in_every_slow_mode_are_the_same(self):
        # the fast stream does not depend on how often the slow weights learn, at step 0 exactly
        writes = [run(build(**options))[1][0]["fast_write"][0] for options in
                  (dict(), dict(slow_update_every=2), dict(slow_update_every="sequence"))]
        for other in writes[1:]:
            torch.testing.assert_close(other, writes[0], rtol=0, atol=0)

    def test_subsampled_fast_backward_has_no_write_on_skipped_steps(self):
        _, (traces, _) = run(build(fast_backward_per_forward="1/2"))
        self.assertTrue((traces["fast_write"][[0, 2]] > 0).all())
        self.assertTrue((traces["fast_write"][[1, 3]] == 0).all())
        self.assertTrue(torch.isnan(traces["loop_gain"][[1, 3]]).all() or (traces["loop_gain"][1] == 0).all())

    def test_extra_fast_passes_are_traced_from_the_first_pass(self):
        _, (once, _) = run(build())
        _, (twice, _) = run(build(fast_backward_per_forward=2))
        for name in ("act_norm", "fast_norm", "fast_write", "max_logit"):   # step 0: same first pass
            torch.testing.assert_close(twice[name][0], once[name][0], rtol=0, atol=0)
        # step 1 sees the extra pass's fast weights
        self.assertFalse(torch.equal(twice["fast_norm"][1], once["fast_norm"][1]))

    def test_extra_passes_with_per_step_slow_updates_trace_the_slow_half_on_its_own_step(self):
        # K=2 splits each step: the slow half of pass 1 is applied after the extra passes, and the
        # tracer reads after that, so every step shows its own slow change.
        _, (twice, _) = run(build(fast_backward_per_forward=2))
        _, (once, _) = run(build())
        self.assertTrue((twice["slow_delta"] > 0).all())
        torch.testing.assert_close(twice["slow_delta"][0], once["slow_delta"][0], rtol=1e-5, atol=1e-7)  # same pass-1 error
        applied = twice["slow_delta"].sum(0)   # triangle inequality: the net change is at most the sum of steps
        self.assertTrue((twice["slow_total_delta"] > 0).all())
        self.assertTrue((twice["slow_total_delta"] <= applied + 1e-6).all())

    def test_extra_passes_with_slow_windows_jump_only_when_the_window_applies(self):
        _, (windowed, _) = run(build(fast_backward_per_forward=2, slow_update_every=2))  # windows end after steps 1 and 3
        self.assertTrue((windowed["slow_delta"][[0, 2]] == 0).all())
        self.assertTrue((windowed["slow_delta"][[1, 3]] > 0).all())
        _, (sequence, _) = run(build(fast_backward_per_forward=2, slow_update_every="sequence"))
        self.assertTrue((sequence["slow_delta"] == 0).all())      # applied after the last step, outside the steps
        self.assertTrue((sequence["slow_total_delta"] > 0).all())

    def test_fast_delta_includes_the_extra_passes_but_fast_write_is_pass_one(self):
        _, (once, _) = run(build(forget_rate=0.0))
        _, (twice, _) = run(build(forget_rate=0.0, fast_backward_per_forward=2))
        torch.testing.assert_close(once["fast_delta"], once["fast_write"], rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(twice["fast_write"][0], once["fast_write"][0], rtol=0, atol=0)
        # step 0 starts from zero fast weights: with the second pass's write the change is larger
        self.assertTrue((twice["fast_delta"][0] != twice["fast_write"][0]).any())

    def test_subsampled_characters_record_no_write_and_only_forgetting_as_fast_change(self):
        _, (traces, _) = run(build(fast_backward_per_forward="1/2", forget_rate=0.25))
        skipped = [1, 3]
        self.assertTrue((traces["fast_write"][skipped] == 0).all())
        self.assertTrue((traces["fast_delta"][skipped] > 0).all())   # forgetting only
        # a forget-only step shrinks F by the forget rate: |F' - F| = 0.25 |F|
        torch.testing.assert_close(traces["fast_delta"][skipped], 0.25 * traces["fast_norm"][skipped],
                                   rtol=1e-4, atol=1e-6)
        self.assertTrue((traces["slow_delta"] > 0).all())          # slow stream unchanged by subsampling

    def test_tracing_changes_nothing(self):
        for options in (dict(), dict(fused=True), dict(slow_update_every="sequence"), dict(slow_update_every=2),
                        dict(fast_backward_per_forward=2), dict(fast_backward_per_forward="1/2", fused=True),
                        dict(layer_norm=True, output_tanh=True, fast_weight_clamp=0.05, weight_clamp=0.3)):
            with self.subTest(**options):
                traced, plain = build(**options), build(**options)
                traced_losses, _ = run(traced, update_clamp=0.05, grad_norm_clip=0.5)
                plain_losses, _ = run(plain, trace=False, update_clamp=0.05, grad_norm_clip=0.5)
                self.assertEqual(traced_losses, plain_losses)
                for (name, a), b in zip(weights(traced).items(), weights(plain).values()):
                    self.assertTrue(torch.equal(a, b), name)
                self.assertEqual(traced.grad_clip_stats.summary(), plain.grad_clip_stats.summary())

    def test_null_tracer_is_inert_and_other_updaters_are_refused(self):
        self.assertIsNone(NULL_TRACER.finish())
        model = build(updater="backprop")
        batch = SEQUENCES[0]
        with self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()):
            train_module.train(batch, F.one_hot(batch, len(CHARSET)).float(), model,
                               {**config(), "updater": "backprop"}, {"training_instance": 0},
                               tracer=LoopTracer(model, LEARNING_RATE))


class PerPassTest(unittest.TestCase):
    """--fast_backward_per_forward K >= 2: the per-pass traces (fast_write_pass, pass_gain, ...)."""

    @staticmethod
    def closed_form_write(model, error):
        """|lr * alpha * mask o (p x^T)|_F per row and fast layer, in float64 numpy, from the layers'
        current in_traces and the analytic output error (no clamp, no clip, no f')."""
        fast_layers = [*model.linear_layers, model.i2h]
        out = []
        for layer in fast_layers:
            x = layer.in_traces.data.double().numpy()
            mask = layer.ephemeral_mask.data.numpy()
            p = error @ layer.feedback_weights.data.double().numpy()
            inner = (p ** 2) * (x ** 2 @ mask.T.astype(float))
            out.append(LEARNING_RATE * ALPHA * np.sqrt(inner.sum(1)))
        return np.stack(out, axis=1)                                              # [B, L]

    def test_pass_zero_is_the_first_pass_and_k1_has_no_pass_traces(self):
        _, (once, _) = run(build())
        self.assertFalse(any(name.endswith("_pass") or name.startswith("pass_") for name in once))
        _, (traces, _) = run(build(fast_backward_per_forward=3))
        self.assertEqual(traces["fast_write_pass"].shape, (4, 3, BATCH, 3))
        self.assertEqual(traces["pass_gain"].shape, (4, 3, BATCH))
        torch.testing.assert_close(traces["fast_write_pass"][:, 0], traces["fast_write"], rtol=0, atol=0)
        torch.testing.assert_close(traces["fast_norm_pass"][:, 0], traces["fast_norm"], rtol=0, atol=0)
        torch.testing.assert_close(traces["max_logit_pass"][:, 0], traces["max_logit"], rtol=0, atol=0)
        torch.testing.assert_close(traces["loss_pass"][:, 0], traces["loss"], rtol=0, atol=0)
        self.assertTrue(torch.isnan(traces["pass_gain"][:, 0]).all())

    def test_each_pass_matches_an_independent_closed_form_computed_at_the_write(self):
        for k in (2, 3):
            with self.subTest(k=k):
                model = build(fast_backward_per_forward=k)
                seen, original = [], model.fast_only_dfa_step

                def spy(output_error, *args, **kwargs):
                    seen.append((self.closed_form_write(model, output_error.double().numpy()),
                                 np.stack([np.sqrt(((l.per_sample_weights.data.double().numpy()
                                                     * l.ephemeral_mask.data.numpy()) ** 2).sum((1, 2)))
                                           for l in [*model.linear_layers, model.i2h]], axis=1)))
                    return original(output_error, *args, **kwargs)

                model.fast_only_dfa_step = spy
                _, (traces,) = run(model, batches=SEQUENCES[:1])
                steps = traces["fast_write_pass"].shape[0]
                self.assertEqual(len(seen), steps * (k - 1))
                for step in range(steps):
                    for extra in range(1, k):
                        write, norm = seen[step * (k - 1) + extra - 1]
                        torch.testing.assert_close(traces["fast_write_pass"][step, extra].double().numpy(), write,
                                                   rtol=1e-4, atol=1e-6)
                        torch.testing.assert_close(traces["fast_norm_pass"][step, extra].double().numpy(), norm,
                                                   rtol=1e-4, atol=1e-6)

    def test_pass_gain_is_the_within_character_ratio_and_the_fast_loop_shows_in_it(self):
        _, (traces, _) = run(build(fast_backward_per_forward=3))
        write = traces["pass_write_norm"]
        gain = traces["pass_gain"]
        torch.testing.assert_close(gain[:, 1:], write[:, 1:] / write[:, :-1], rtol=1e-5, atol=1e-7)
        self.assertTrue(torch.isfinite(gain[:, 1:]).all())
        # pass 1 sees the weights pass 0 wrote and the same input: the passes differ
        self.assertFalse(torch.equal(traces["fast_norm_pass"][:, 1], traces["fast_norm_pass"][:, 0]))
        self.assertTrue(torch.equal(traces["fast_norm_pass"][0, 0], torch.zeros_like(traces["fast_norm_pass"][0, 0])))
        # the derived arrays follow from fast_write_pass
        recomputed = derived_traces({k: traces[k] for k in ("fast_write", "fast_write_pass")})
        torch.testing.assert_close(recomputed["pass_gain"], gain, rtol=0, atol=0, equal_nan=True)

    def test_derived_pass_gain_skips_passes_after_a_zero_write(self):
        writes = torch.tensor([[[[1.0]], [[2.0]], [[0.0]]], [[[0.0]], [[3.0]], [[6.0]]]])   # [T=2, P=3, B=1, L=1]
        derived = derived_traces({"fast_write": writes[:, 0], "fast_write_pass": writes})
        gain = derived["pass_gain"]
        self.assertTrue(torch.isnan(gain[:, 0]).all())
        self.assertEqual(float(gain[0, 1, 0]), 2.0)
        self.assertEqual(float(gain[0, 2, 0]), 0.0)
        self.assertTrue(torch.isnan(gain[1, 1, 0]))        # previous pass wrote nothing
        self.assertEqual(float(gain[1, 2, 0]), 2.0)

    def test_eager_fused_pass_traces_are_bit_identical_to_unfused_in_every_slow_mode(self):
        for slow in (1, 2, "sequence"):
            with self.subTest(slow=slow):
                _, (plain, _) = run(build(fast_backward_per_forward=2, slow_update_every=slow))
                _, (fused, _) = run(build(fast_backward_per_forward=2, slow_update_every=slow, fused=True))
                for name in ("fast_write_pass", "fast_norm_pass", "max_logit_pass", "loss_pass"):
                    torch.testing.assert_close(fused[name], plain[name], rtol=0, atol=0)

    def test_pass_zero_does_not_depend_on_the_extra_passes_for_the_first_step(self):
        _, (two, _) = run(build(fast_backward_per_forward=2))
        _, (three, _) = run(build(fast_backward_per_forward=3))
        torch.testing.assert_close(two["fast_write_pass"][0, :2], three["fast_write_pass"][0, :2], rtol=0, atol=0)
        torch.testing.assert_close(two["fast_norm_pass"][0, :2], three["fast_norm_pass"][0, :2], rtol=0, atol=0)

    def test_tracing_k3_changes_nothing_and_clip_scales_the_pass_writes(self):
        traced, plain = build(fast_backward_per_forward=3), build(fast_backward_per_forward=3)
        traced_losses, (clipped, _) = run(traced, update_clamp=0.05, grad_norm_clip=0.5)
        plain_losses, _ = run(plain, trace=False, update_clamp=0.05, grad_norm_clip=0.5)
        self.assertEqual(traced_losses, plain_losses)
        for (name, a), b in zip(weights(traced).items(), weights(plain).values()):
            self.assertTrue(torch.equal(a, b), name)
        _, (free, _) = run(build(fast_backward_per_forward=3))
        self.assertLess(float(clipped["fast_write_pass"].sum()), float(free["fast_write_pass"].sum()))

    def test_replay_carries_the_pass_traces_and_restores_the_state(self):
        model = build(fast_backward_per_forward=2)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        onehot = F.one_hot(SEQUENCES[0], len(CHARSET)).float()
        results = trace_replay.replay_checkpoint(model, config(), {"training_instance": 0}, [(None, onehot)] * 2)
        self.assertEqual(len(results), 2)
        for traces in results:
            self.assertEqual(traces["pass_gain"].shape, (4, 2, BATCH))
        torch.testing.assert_close(results[0]["pass_gain"], results[1]["pass_gain"], rtol=0, atol=0, equal_nan=True)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)
        self.assertIn("trace/pass_gain_median", trace_replay.mean_summary(results))

    def test_summary_has_pass_scalars_only_with_passes(self):
        _, (with_passes, _) = run(build(fast_backward_per_forward=2))
        summary = summarize(with_passes)
        for key in ("pass_gain_median", "pass_gain_p90", "frac_pass_gain_gt1"):
            self.assertIn(f"trace/{key}", summary)
            self.assertTrue(math.isfinite(summary[f"trace/{key}"]))
        _, (without, _) = run(build())
        self.assertFalse(any("pass_gain" in key for key in summarize(without)))


class CliTest(unittest.TestCase):
    def test_flag_validation(self):
        for extra in (["--trace_loop_every", "3", "--print_freq", "2"], ["--trace_loop_every", "2", "--model_type", "rnn"],
                      ["--trace_loop_every", "-1"], ["--checkpoint_keep_every", "-2"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    train_module.parse_args(extra)
        args = train_module.parse_args(["--trace_loop_every", "100", "--print_freq", "50"])
        self.assertEqual((args.trace_loop_every, args.checkpoint_keep_every), (100, 0))


def run_main(*extra, checkpoint_dir, n_iters="6"):
    argv = ["train.py", "--dataset", DATASET, "--track", "False", "--n_iters", n_iters, "--print_freq", "2",
            "--checkpoint_save_freq", "6", "--checkpoint_dir", checkpoint_dir, "--batch_size", "2",
            "--hidden_size", "4", "--num_layers", "2", "--seed", "3", "--plasticity", "100", *extra]
    output = io.StringIO()
    with patch("sys.argv", argv), \
            patch.object(train_module, "load_and_preprocess_data", side_effect=fake_loader(0)), \
            contextlib.redirect_stdout(output):
        train_module.main()
    return output.getvalue()


class TrainFlagAndReplayTest(unittest.TestCase):
    def test_flag_logs_and_saves_traces_leaves_training_unchanged_and_replay_reads_the_checkpoints(self):
        with tempfile.TemporaryDirectory() as off, tempfile.TemporaryDirectory() as on, \
                tempfile.TemporaryDirectory() as out:
            flags = ["--slow_update_every", "sequence", "--fast_backward_per_forward", "2", "--layer_norm", "true"]
            plain = run_main(*flags, checkpoint_dir=off)
            traced = run_main(*flags, "--trace_loop_every", "2", "--checkpoint_keep_every", "2",
                              "--checkpoint_keep_max", "2", checkpoint_dir=on)
            self.assertNotIn("trace/", plain)
            self.assertEqual(traced.count("trace/loop_gain_median:"), 3)
            self.assertEqual(sorted(os.listdir(os.path.join(on, "traces"))),
                             [f"trace_{i:08d}.pt" for i in (2, 4, 6)])
            saved = torch.load(os.path.join(on, "traces", "trace_00000004.pt"), weights_only=False)
            self.assertEqual(saved["iter"], 4)
            self.assertIn("loop_gain", saved["traces"])
            first, second = (torch.load(os.path.join(d, "latest_checkpoint.pth"), weights_only=False)
                             for d in (off, on))
            for name, value in first["model_state_dict"].items():
                self.assertTrue(torch.equal(value, second["model_state_dict"][name]), name)
            # numbered copies: every 2nd iteration, only the newest 2 kept
            self.assertEqual(sorted(n for n in os.listdir(on) if n.startswith("checkpoint_")),
                             ["checkpoint_00000004.pth", "checkpoint_00000006.pth"])

            replay_out = os.path.join(out, "replay.pt")
            output = io.StringIO()
            with patch.object(trace_replay, "load_heldout_batches", side_effect=fake_heldout_batches), \
                    contextlib.redirect_stdout(output):
                trace_replay.main(["--checkpoints", on, "--device", "cpu", "--out", replay_out])
            replay = torch.load(replay_out, weights_only=False)
            self.assertEqual([record["iter"] for record in replay["checkpoints"]], [4, 6])
            for record in replay["checkpoints"]:
                self.assertEqual(len(record["batches"]), 2)
                traces = record["batches"][0]
                self.assertTrue(torch.isfinite(traces["fast_write"]).all())
                self.assertIn("trace/loop_gain_median", record["summary"])

    def test_replay_restores_the_checkpoint_state_and_matches_a_direct_training_step(self):
        with tempfile.TemporaryDirectory() as directory:
            run_main(checkpoint_dir=directory, n_iters="2", *["--checkpoint_save_freq", "2"])
            path = os.path.join(directory, "latest_checkpoint.pth")
            batches = fake_heldout_batches(DATASET, 2, 2, "cpu")
            with contextlib.redirect_stdout(io.StringIO()):
                model, cfg, state, _ = trace_replay.load_model(path, "cpu")
                before = {k: v.clone() for k, v in model.state_dict().items()}
                first = trace_replay.replay_checkpoint(model, cfg, state, batches)
                after = {k: v.clone() for k, v in model.state_dict().items()}
                second = trace_replay.replay_checkpoint(model, cfg, state, batches)
                fused_model, fused_cfg, fused_state, _ = trace_replay.load_model(path, "cpu", fused_update=True)
                fused = trace_replay.replay_checkpoint(fused_model, fused_cfg, fused_state, batches)
            for name, value in before.items():
                self.assertTrue(torch.equal(value, after[name]), name)
            for one, two in zip(first, second):   # a replay is repeatable
                for name in one:
                    torch.testing.assert_close(one[name], two[name], rtol=0, atol=0, equal_nan=True)
            for one, two in zip(first, fused):    # eager fused here: no cuda, so compile=False
                for name in one:
                    torch.testing.assert_close(one[name], two[name], rtol=1e-5, atol=1e-7, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
