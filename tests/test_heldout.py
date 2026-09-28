"""Held-out fast-memory evaluation has prequential, fast-only state semantics."""
import contextlib
import io
import unittest

import torch
import torch.nn.functional as F

from ephemeral_model import EphemeralRNN
from heldout import HeldOutBatch, evaluate_held_out
from reproducibility import seed_everything


CHARSET = list("abc")
BATCH, STEPS = 2, 3


def build(forget_rate=0.25, unit_norm_weights=False):
    seed_everything(123, deterministic=True)
    with contextlib.redirect_stdout(io.StringIO()):
        model = EphemeralRNN(3, 3, 3, 1, CHARSET, updater="dfa", batch_size=BATCH,
                             unit_norm_weights=unit_norm_weights, weight_clamp=0, plasticity=2,
                             forget_rate=forget_rate, ephemeral_fraction=0.6,
                             enable_recurrence=True, slow_weight_decay=0.9)
    # Avoid a vanishingly unlikely random mask making a test vacuous.
    for layer in (*model.linear_layers, model.i2h):
        layer.ephemeral_mask.data[0, 0] = True
        layer.per_sample_weights.data[:, 0, 0] = torch.tensor([0.4, -0.7])
    return model


def data(target_ids=None, score=None, update=None, reset=None):
    inputs = F.one_hot(torch.tensor([[0, 1, 2], [2, 0, 1]]), 3).float()
    ids = target_ids if target_ids is not None else torch.tensor([[1, 2, 0], [0, 1, 2]])
    return HeldOutBatch(
        inputs, F.one_hot(ids, 3).float(),
        torch.ones(BATCH, STEPS, dtype=torch.bool) if score is None else score,
        torch.zeros(BATCH, STEPS, dtype=torch.bool) if update is None else update,
        torch.zeros(BATCH, STEPS, dtype=torch.bool) if reset is None else reset)


def fast_state(model):
    return [layer.per_sample_weights.detach().clone() for layer in model.trained_layers()]


def continue_from_zero(model):
    return {"initial_state": "continue", "initial_hidden": model.initHidden(BATCH)}


class HeldOutEvaluationTest(unittest.TestCase):
    def test_fresh_entry_consolidates_slow_rows_and_wipes_fast_before_prediction(self):
        model = build(forget_rate=0)
        with torch.no_grad():
            for layer in model.trained_layers():
                layer.per_sample_weights[0].fill_(1.0)
                layer.per_sample_weights[1].fill_(3.0)
                if layer.ephemeral_mask.any():
                    layer.per_sample_weights[0][layer.ephemeral_mask] = 9.0
                    layer.per_sample_weights[1][layer.ephemeral_mask] = 7.0
        at_first_prediction = []
        hook = model.register_forward_pre_hook(
            lambda _model, _args: at_first_prediction.append(fast_state(model)))
        evaluate_held_out(model, data(), 0)
        hook.remove()

        first = at_first_prediction[0]
        for layer, weights in zip(model.trained_layers(), first):
            mask = layer.ephemeral_mask
            torch.testing.assert_close(weights[:, ~mask], torch.full_like(weights[:, ~mask], 2.0),
                                       rtol=0, atol=0)
            torch.testing.assert_close(weights[:, mask], torch.zeros_like(weights[:, mask]),
                                       rtol=0, atol=0)
            torch.testing.assert_close(weights[0], weights[1], rtol=0, atol=0)
        # The slow-only output head is consolidated too, not left row-specific.
        torch.testing.assert_close(first[-1], torch.full_like(first[-1], 2.0), rtol=0, atol=0)

    def test_continue_preserves_entry_state_and_uses_and_returns_hidden(self):
        model = build(forget_rate=0)
        with torch.no_grad():
            model.i2o.per_sample_weights[0].fill_(1.0)
            model.i2o.per_sample_weights[1].fill_(3.0)
        weights_before = fast_state(model)
        initial_hidden = torch.tensor([[0.1, 0.2, 0.3], [-0.4, 0.5, -0.6]])
        hidden_inputs, hidden_outputs, entry_weights = [], [], []

        def record_entry(_model, args):
            hidden_inputs.append(args[1].detach().clone())
            entry_weights.append(fast_state(model))

        pre = model.register_forward_pre_hook(record_entry)
        post = model.register_forward_hook(
            lambda _model, _args, output: hidden_outputs.append(output[1].detach().clone()))

        result = evaluate_held_out(model, data(), 0, initial_state="continue",
                                   initial_hidden=initial_hidden)
        pre.remove()
        post.remove()

        torch.testing.assert_close(hidden_inputs[0], initial_hidden, rtol=0, atol=0)
        for actual, expected in zip(entry_weights[0], weights_before):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(model.i2o.per_sample_weights, weights_before[-1], rtol=0, atol=0)
        torch.testing.assert_close(result.final_hidden, hidden_outputs[-1], rtol=0, atol=0)

    def test_inputs_and_targets_are_independent_and_prediction_precedes_write(self):
        first, second = build(), build()
        targets_a = torch.tensor([[0, 1, 2], [1, 2, 0]])
        targets_b = torch.tensor([[2, 1, 2], [0, 2, 0]])
        update = torch.zeros(BATCH, STEPS, dtype=torch.bool)
        update[:, 0] = True
        result_a = evaluate_held_out(first, data(targets_a, update=update), 0.2)
        result_b = evaluate_held_out(second, data(targets_b, update=update), 0.2)

        # Same input and pre-write state: target 0 cannot affect prediction 0.
        torch.testing.assert_close(result_a.logits[:, 0], result_b.logits[:, 0], rtol=0, atol=0)
        # Different independently supplied targets produce different writes and losses.
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(fast_state(first), fast_state(second))))
        expected_a = -(data(targets_a).targets[:, 0] * F.log_softmax(result_a.logits[:, 0], 1)).sum(1)
        expected_b = -(data(targets_b).targets[:, 0] * F.log_softmax(result_b.logits[:, 0], 1)).sum(1)
        torch.testing.assert_close(result_a.losses[:, 0], expected_a)
        torch.testing.assert_close(result_b.losses[:, 0], expected_b)

    def test_score_mask_only_changes_aggregates(self):
        left, right = build(), build()
        update = torch.ones(BATCH, STEPS, dtype=torch.bool)
        score_left = torch.tensor([[True, False, False], [False, False, False]])
        score_right = ~score_left
        a = evaluate_held_out(left, data(score=score_left, update=update), 0.1)
        b = evaluate_held_out(right, data(score=score_right, update=update), 0.1)
        torch.testing.assert_close(a.logits, b.logits, rtol=0, atol=0)
        torch.testing.assert_close(a.losses, b.losses, rtol=0, atol=0)
        for actual, expected in zip(fast_state(left), fast_state(right)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual((a.scored_count, b.scored_count), (1, 5))

    def test_update_mask_blocks_write_but_forgetting_advances(self):
        no_write, write = build(forget_rate=0.25), build(forget_rate=0.25)
        before = fast_state(no_write)
        updates = torch.zeros(BATCH, STEPS, dtype=torch.bool)
        written = updates.clone()
        written[:, 0] = True
        evaluate_held_out(no_write, data(update=updates), 0.2, **continue_from_zero(no_write))
        evaluate_held_out(write, data(update=written), 0.2, **continue_from_zero(write))
        for layer, initial, blocked, changed in zip(no_write.trained_layers(), before,
                                                     fast_state(no_write), fast_state(write)):
            mask = layer.ephemeral_mask.unsqueeze(0).expand_as(initial)
            torch.testing.assert_close(blocked[mask], initial[mask] * (0.75 ** STEPS), rtol=1e-6, atol=1e-7)
            if mask.any():
                self.assertFalse(torch.equal(blocked[mask], changed[mask]))

    def test_fast_write_matches_hand_computed_dfa_update_and_forgetting(self):
        model = build(forget_rate=0.25)
        layer = model.linear_layers[0]
        with torch.no_grad():
            for trained in model.trained_layers():
                trained.per_sample_weights.zero_()
                trained.bias.zero_()
                trained.ephemeral_mask.zero_()
                trained.feedback_weights.zero_()
            layer.ephemeral_mask[0, 0] = True
            layer.per_sample_weights[:, 0, 0] = 0.4
            layer.plasticity[0, 0] = 3.0
            # Zero logits and target class 0 give output error [-2/3, 1/3, 1/3]. This feedback
            # makes projected_error[0] = -1; input trace[0] = 1, so gradient = -1.
            layer.feedback_weights[0, 0] = 1.5
        inputs = F.one_hot(torch.tensor([[0], [0]]), 3).float()
        targets = F.one_hot(torch.tensor([[0], [0]]), 3).float()
        masks = torch.ones(BATCH, 1, dtype=torch.bool)
        batch = HeldOutBatch(inputs, targets, masks, masks, torch.zeros_like(masks))

        evaluate_held_out(model, batch, learning_rate=0.1, **continue_from_zero(model))

        # (0.4 - 0.1 * plasticity(3) * gradient(-1)) * (1 - forget_rate(.25))
        expected = torch.tensor(0.525)
        torch.testing.assert_close(layer.per_sample_weights[:, 0, 0], expected.expand(BATCH),
                                   rtol=0, atol=1e-7)

    def test_mixed_update_row_reset_predict_write_forget_order(self):
        model = build(forget_rate=0.25)
        layer = model.linear_layers[0]
        with torch.no_grad():
            for trained in model.trained_layers():
                trained.per_sample_weights.zero_()
                trained.bias.zero_()
                trained.ephemeral_mask.zero_()
                trained.feedback_weights.zero_()
            layer.ephemeral_mask[0, 0] = True
            layer.per_sample_weights[:, 0, 0] = torch.tensor([0.4, 0.8])
            layer.plasticity[0, 0] = 3.0
            layer.feedback_weights[0, 0] = 1.5
        inputs = F.one_hot(torch.tensor([[0], [0]]), 3).float()
        targets = F.one_hot(torch.tensor([[0], [0]]), 3).float()
        score = torch.ones(BATCH, 1, dtype=torch.bool)
        update = torch.tensor([[True], [False]])
        reset = torch.tensor([[True], [False]])

        result = evaluate_held_out(model, HeldOutBatch(inputs, targets, score, update, reset), 0.1,
                                   **continue_from_zero(model))

        # Both predictions happen with zero output logits. Row 0 resets, writes +0.3, then
        # forgets to .225. Row 1 receives no write and only forgets .8 to .6.
        torch.testing.assert_close(result.logits, torch.zeros_like(result.logits), rtol=0, atol=0)
        torch.testing.assert_close(layer.per_sample_weights[:, 0, 0], torch.tensor([0.225, 0.6]),
                                   rtol=0, atol=1e-7)

    def test_reset_is_per_row_and_preserves_slow_copies(self):
        model = build(forget_rate=0)
        before = fast_state(model)
        reset = torch.zeros(BATCH, STEPS, dtype=torch.bool)
        reset[0, 1] = True
        hidden_inputs = []
        hook = model.register_forward_pre_hook(
            lambda _model, args: hidden_inputs.append(args[1].detach().clone()))
        evaluate_held_out(model, data(reset=reset), 0, **continue_from_zero(model))
        hook.remove()
        # The selected recurrent row is also cleared before input 1; the other row is not.
        torch.testing.assert_close(hidden_inputs[1][0], torch.zeros_like(hidden_inputs[1][0]), rtol=0, atol=0)
        self.assertGreater(hidden_inputs[1][1].abs().sum().item(), 0)
        for layer, initial, final in zip(model.trained_layers(), before, fast_state(model)):
            mask = layer.ephemeral_mask
            torch.testing.assert_close(final[:, ~mask], initial[:, ~mask], rtol=0, atol=0)
            torch.testing.assert_close(final[1, mask], initial[1, mask], rtol=0, atol=0)
            torch.testing.assert_close(final[0, mask], torch.zeros_like(final[0, mask]), rtol=0, atol=0)

    def test_all_slow_parameters_biases_and_output_head_are_bitwise_frozen(self):
        model = build()
        # Traces are forward-pass scratch state, not learned weights. Everything else outside
        # per-sample weights must remain exact (base weights, biases, feedback and plasticity).
        parameters = {name: value.detach().clone() for name, value in model.named_parameters()
                      if "per_sample_weights" not in name and not any(
                          scratch in name for scratch in ("in_traces", "out_traces", "last_ephemeral", "last_slow"))}
        slow = [(layer.per_sample_weights.detach().clone(), layer.ephemeral_mask.detach().clone())
                for layer in model.trained_layers()]
        evaluate_held_out(model, data(update=torch.ones(BATCH, STEPS, dtype=torch.bool)), 0.3,
                          update_clamp=0.05, **continue_from_zero(model))
        for name, expected in parameters.items():
            self.assertTrue(torch.equal(dict(model.named_parameters())[name].detach(), expected), name)
        for layer, (expected, mask) in zip(model.trained_layers(), slow):
            self.assertTrue(torch.equal(layer.per_sample_weights.detach()[:, ~mask], expected[:, ~mask]))
        self.assertFalse(model.i2o.ephemeral_mask.any())

    def test_validation_rejects_non_boolean_or_misaligned_masks(self):
        inputs = torch.zeros(2, 3, 3)
        targets = torch.zeros(2, 3, 3)
        good = torch.zeros(2, 3, dtype=torch.bool)
        with self.assertRaises(ValueError):
            HeldOutBatch(inputs, targets, good.float(), good, good)
        with self.assertRaises(ValueError):
            HeldOutBatch(inputs, targets, good[:, :2], good, good)

    def test_validation_rejects_invalid_probability_targets(self):
        inputs = torch.zeros(1, 1, 3)
        mask = torch.zeros(1, 1, dtype=torch.bool)
        invalid = {
            "zero": [0.0, 0.0, 0.0],
            "negative": [1.1, -0.1, 0.0],
            "non_normalized": [0.2, 0.3, 0.4],
            "non_finite": [float("nan"), 0.0, 1.0],
        }
        for name, row in invalid.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                HeldOutBatch(inputs, torch.tensor([[row]]), mask, mask, mask)

    def test_validation_rejects_non_finite_inputs(self):
        inputs = torch.tensor([[[float("inf"), 0.0, 0.0]]])
        targets = torch.tensor([[[1.0, 0.0, 0.0]]])
        mask = torch.zeros(1, 1, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "inputs.*finite"):
            HeldOutBatch(inputs, targets, mask, mask, mask)

    def test_continuation_hidden_is_required_and_validated(self):
        model = build()
        batch = data()
        with self.assertRaisesRegex(ValueError, "required"):
            evaluate_held_out(model, batch, 0, initial_state="continue")
        with self.assertRaisesRegex(ValueError, "must not"):
            evaluate_held_out(model, batch, 0, initial_hidden=model.initHidden(BATCH))
        bad = {
            "shape": torch.zeros(BATCH, model.hidden_size + 1),
            "dtype": torch.zeros(BATCH, model.hidden_size, dtype=torch.float64),
            "device": torch.empty(BATCH, model.hidden_size, device="meta"),
        }
        for name, hidden in bad.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                evaluate_held_out(model, batch, 0, initial_state="continue",
                                  initial_hidden=hidden)
        for name, value in (("nan", float("nan")), ("inf", float("inf"))):
            hidden = model.initHidden(BATCH)
            hidden[0, 0] = value
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "finite"):
                evaluate_held_out(model, batch, 0, initial_state="continue",
                                  initial_hidden=hidden)

    def test_float64_model_data_hidden_and_results(self):
        model = build().double()
        source = data()
        batch = HeldOutBatch(source.inputs.double(), source.targets.double(), source.score_mask,
                             source.update_mask, source.reset_mask)
        result = evaluate_held_out(model, batch, 0)
        self.assertEqual(model.initHidden(BATCH).dtype, torch.float64)
        self.assertEqual(result.final_hidden.dtype, torch.float64)
        self.assertEqual(result.logits.dtype, torch.float64)
        self.assertEqual(result.losses.dtype, torch.float64)

    def test_unit_norm_weights_is_rejected(self):
        model = build(unit_norm_weights=True)
        with self.assertRaisesRegex(ValueError, "unit_norm_weights=True"):
            evaluate_held_out(model, data(), 0.1)

    def test_model_mode_is_restored_when_evaluation_raises(self):
        model = build()
        model.train()
        original_forward = model.forward

        def fail_forward(*_args, **_kwargs):
            self.assertFalse(model.training)
            raise RuntimeError("intentional forward failure")

        model.forward = fail_forward
        try:
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                evaluate_held_out(model, data(), 0.1)
        finally:
            model.forward = original_forward
        self.assertTrue(model.training)

        model.eval()
        evaluate_held_out(model, data(), 0.1)
        self.assertFalse(model.training)


if __name__ == "__main__":
    unittest.main()
