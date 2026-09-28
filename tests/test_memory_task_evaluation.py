"""Device transfer and diagnostic evaluation for key-value episodes."""
import contextlib
import copy
from dataclasses import replace
import io
import unittest

import torch
import torch.nn.functional as F

from ephemeral_model import EphemeralRNN
from heldout import HeldOutBatch, HeldOutResult, evaluate_held_out
from memory_tasks import (evaluate_key_value, generate_key_value_episodes,
                          summarize_key_value)


def episodes(batch_size=128):
    return generate_key_value_episodes(
        batch_size=batch_size, table_size=10, key_vocab_size=4, value_vocab_size=8,
        mode="replacement", query_key_assignments=4, distractor_count=2,
        distractor_vocab_size=3, query_mode="strict", seed=912)


def fake_result(task, query_predictions):
    batch_size, steps = task.batch.score_mask.shape
    vocab = task.layout.vocab_size
    predictions = torch.zeros(batch_size, steps, dtype=torch.long)
    predictions[:, -1] = query_predictions
    logits = torch.zeros(batch_size, steps, vocab)
    strengths = (1 + torch.arange(batch_size, dtype=torch.float32)[:, None] / 100 +
                 torch.arange(steps, dtype=torch.float32)[None, :] / 1000)
    logits.scatter_(2, predictions.unsqueeze(2), strengths.unsqueeze(2))
    losses = -(task.batch.targets * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    accuracy = (query_predictions == task.answers).float().mean().item()
    return HeldOutResult(logits, predictions, losses, losses[:, -1].mean().item(), accuracy,
                         batch_size, torch.zeros(batch_size, 1))


def audited_rows(task):
    """Find deterministic rows/tokens that exercise every value-error category and overlap."""
    found = {}
    used = set()
    value_tokens = set(task.layout.value_ids)
    for row in range(task.assignment_keys.shape[0]):
        keys, values, query = (task.assignment_keys[row], task.assignment_values[row],
                               task.query_keys[row])
        occurrences = torch.nonzero(keys == query).flatten()
        latest = task.answers[row].item()
        immediate = values[occurrences[-2]].item()
        older_values = [value.item() for value in values[occurrences[:-2]]
                        if value.item() not in (latest, immediate)]
        stale_values = set(values[occurrences[:-1]].tolist())
        wrong_values = set(values[keys != query].tolist()) - stale_values - {latest}
        unbound = value_tokens - set(values.tolist())
        overlap = stale_values & set(values[keys != query].tolist())
        candidates = {
            "immediate": immediate,
            "older": older_values[0] if older_values else None,
            "wrong": next(iter(wrong_values), None),
            "other": next(iter(unbound), None),
            "overlap": next(iter(overlap), None),
        }
        for name, token in candidates.items():
            if name not in found and token is not None and row not in used:
                found[name] = (row, token)
                used.add(row)
        if len(found) == len(candidates):
            break
    if set(found) != {"immediate", "older", "wrong", "other", "overlap"}:
        raise AssertionError(f"deterministic audit fixture lacks categories: {found}")
    available = [row for row in range(task.assignment_keys.shape[0]) if row not in used]
    found["correct"] = (available.pop(0), None)
    found["non_value"] = (available.pop(0), task.layout.query_marker)
    return found


class KeyValueEvaluationTest(unittest.TestCase):
    def test_episode_to_changes_only_floating_dtype_and_preserves_tensor_types(self):
        original = episodes(batch_size=3)
        moved = original.to(device=torch.device("cpu"), dtype=torch.float64)
        self.assertIsNot(moved, original)
        self.assertIsNot(moved.batch, original.batch)
        self.assertEqual(moved.batch.inputs.dtype, torch.float64)
        self.assertEqual(moved.batch.targets.dtype, torch.float64)
        for mask in (moved.batch.score_mask, moved.batch.update_mask, moved.batch.reset_mask):
            self.assertEqual(mask.dtype, torch.bool)
            self.assertEqual(mask.device.type, "cpu")
        for name in ("token_ids", "assignment_keys", "assignment_values", "query_keys",
                     "answers", "stale_values", "query_assignment_counts",
                     "latest_source_position", "source_to_answer_lag"):
            self.assertEqual(getattr(moved, name).dtype, torch.long, name)
        self.assertEqual(original.batch.inputs.dtype, torch.float32)
        with self.assertRaisesRegex(ValueError, "floating"):
            original.to(dtype=torch.int64)

    def test_categories_aggregates_and_exclusive_exhaustive_priority(self):
        task = episodes()
        rows = audited_rows(task)
        predictions = torch.full_like(task.answers, task.layout.query_marker)
        correct_row = rows["correct"][0]
        predictions[correct_row] = task.answers[correct_row]
        for name in ("immediate", "older", "wrong", "other"):
            row, token = rows[name]
            predictions[row] = token
        result = fake_result(task, predictions)
        diagnostic = summarize_key_value(result, task)

        self.assertTrue(diagnostic.correct[correct_row])
        self.assertTrue(diagnostic.immediate_stale[rows["immediate"][0]])
        self.assertTrue(diagnostic.older_stale[rows["older"][0]])
        self.assertTrue(diagnostic.wrong_key[rows["wrong"][0]])
        self.assertTrue(diagnostic.other_value[rows["other"][0]])
        self.assertTrue(diagnostic.non_value[rows["non_value"][0]])
        primary = torch.stack((diagnostic.correct, diagnostic.stale, diagnostic.wrong_key,
                               diagnostic.other_value, diagnostic.non_value))
        self.assertTrue((primary.sum(0) == 1).all())
        self.assertTrue((diagnostic.immediate_stale <= diagnostic.stale).all())
        self.assertTrue(torch.equal(diagnostic.older_stale,
                                    diagnostic.stale & ~diagnostic.immediate_stale))

        summary = diagnostic.summary
        self.assertEqual(summary.query_count, 128)
        self.assertEqual((summary.correct_count, summary.stale_count,
                          summary.immediate_stale_count, summary.older_stale_count,
                          summary.wrong_key_count, summary.other_value_count,
                          summary.non_value_count), (1, 2, 1, 1, 1, 1, 123))
        self.assertEqual(summary.stale_eligible_count, 128)
        self.assertAlmostEqual(summary.query_loss, result.losses[:, -1].mean().item())
        self.assertAlmostEqual(summary.query_accuracy, 1 / 128)
        self.assertAlmostEqual(summary.full_output_uniform_chance, 1 / task.layout.vocab_size)
        self.assertAlmostEqual(summary.value_restricted_chance, 1 / 8)
        self.assertAlmostEqual(summary.stale_rate, 2 / 128)
        self.assertAlmostEqual(summary.non_value_rate, 123 / 128)
        self.assertEqual(summary.stale_eligible_rate, 1.0)

        # A stale value that also appeared under another key remains stale by priority.
        overlap_predictions = torch.full_like(task.answers, task.layout.query_marker)
        overlap_row, overlap_token = rows["overlap"]
        overlap_predictions[overlap_row] = overlap_token
        overlap = summarize_key_value(fake_result(task, overlap_predictions), task)
        self.assertTrue(overlap.stale[overlap_row])
        self.assertFalse(overlap.wrong_key[overlap_row])

    def test_rejects_result_shape_count_and_episode_score_position_mismatches(self):
        task = episodes(batch_size=4)
        result = fake_result(task, task.answers)
        with self.assertRaisesRegex(ValueError, "predictions.*shape"):
            summarize_key_value(replace(result, predictions=result.predictions[:, :-1]), task)
        with self.assertRaisesRegex(ValueError, "scored_count"):
            summarize_key_value(replace(result, scored_count=3), task)

        two_scores = task.batch.score_mask.clone()
        two_scores[:, 0] = True
        bad_batch = replace(task.batch, score_mask=two_scores)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            summarize_key_value(result, replace(task, batch=bad_batch))

        wrong_position = torch.zeros_like(task.batch.score_mask)
        wrong_position[:, 0] = True
        bad_batch = replace(task.batch, score_mask=wrong_position)
        with self.assertRaisesRegex(ValueError, "final answer"):
            summarize_key_value(result, replace(task, batch=bad_batch))

    def test_rejects_inconsistent_result_and_episode_metadata(self):
        task = episodes(batch_size=4)
        result = fake_result(task, task.answers)
        with self.assertRaisesRegex(ValueError, "rank 2"):
            summarize_key_value(replace(result, final_hidden=torch.zeros(4)), task)
        with self.assertRaisesRegex(ValueError, "same device"):
            summarize_key_value(
                replace(result, final_hidden=torch.empty(4, 1, device="meta")), task)
        with self.assertRaisesRegex(ValueError, "token_ids.*shape"):
            summarize_key_value(result, replace(task, token_ids=task.token_ids[:, :-1]))

        bad_tokens = task.token_ids.clone()
        bad_tokens[:, -2] = task.layout.query_marker
        with self.assertRaisesRegex(ValueError, "queried key"):
            summarize_key_value(result, replace(task, token_ids=bad_tokens))
        with self.assertRaisesRegex(ValueError, "query_assignment_counts"):
            summarize_key_value(
                result, replace(task, query_assignment_counts=task.query_assignment_counts + 1))
        with self.assertRaisesRegex(ValueError, "source_to_answer_lag"):
            summarize_key_value(
                result, replace(task, source_to_answer_lag=task.source_to_answer_lag + 1))

        bad_predictions = result.predictions.clone()
        bad_predictions[:, -1] = task.layout.vocab_size
        with self.assertRaisesRegex(ValueError, "valid output token IDs"):
            summarize_key_value(replace(result, predictions=bad_predictions), task)

        inconsistent_predictions = result.predictions.clone()
        inconsistent_predictions[:, -1] = (inconsistent_predictions[:, -1] + 1) % task.layout.vocab_size
        with self.assertRaisesRegex(ValueError, "logits.argmax"):
            summarize_key_value(replace(result, predictions=inconsistent_predictions), task)
        inconsistent_losses = result.losses.clone()
        inconsistent_losses[0, 0] += 0.01
        with self.assertRaisesRegex(ValueError, "cross-entropy"):
            summarize_key_value(replace(result, losses=inconsistent_losses), task)

    def test_convenience_wrapper_matches_direct_evaluation_and_summary(self):
        task = generate_key_value_episodes(
            batch_size=2, table_size=3, key_vocab_size=4, value_vocab_size=3,
            mode="replacement", query_key_assignments=2, distractor_count=1,
            distractor_vocab_size=2, seed=18)
        with contextlib.redirect_stdout(io.StringIO()):
            direct_model = EphemeralRNN(task.layout.vocab_size, 4, task.layout.vocab_size, 1,
                                        list(range(task.layout.vocab_size)), updater="dfa",
                                        batch_size=2, unit_norm_weights=False, plasticity=2,
                                        forget_rate=0.1, ephemeral_fraction=0.5,
                                        enable_recurrence=True)
        wrapped_model = copy.deepcopy(direct_model)
        direct_result = evaluate_held_out(direct_model, task.batch, 0.01, update_clamp=0.05)
        direct = summarize_key_value(direct_result, task)
        wrapped = evaluate_key_value(wrapped_model, task, 0.01, update_clamp=0.05)
        torch.testing.assert_close(wrapped.result.logits, direct.result.logits, rtol=0, atol=0)
        torch.testing.assert_close(wrapped.query_predictions, direct.query_predictions, rtol=0, atol=0)
        self.assertEqual(wrapped.summary, direct.summary)


if __name__ == "__main__":
    unittest.main()
