"""Deterministic token-level key-value held-out episodes."""
import contextlib
import io
import random
import unittest

import numpy as np
import torch

from ephemeral_model import EphemeralRNN
from heldout import evaluate_held_out
from memory_tasks import STALE_SENTINEL, generate_key_value_episodes


def generate(**overrides):
    options = dict(batch_size=4, table_size=5, key_vocab_size=8, value_vocab_size=4,
                   mode="unique", distractor_count=0, distractor_vocab_size=3,
                   query_mode="strict", seed=123)
    options.update(overrides)
    return generate_key_value_episodes(**options)


class KeyValueEpisodeTest(unittest.TestCase):
    def test_unique_keys_query_and_answer(self):
        episodes = generate()
        for row in range(4):
            keys = episodes.assignment_keys[row]
            self.assertEqual(len(keys.unique()), 5)
            query = episodes.query_keys[row]
            self.assertTrue((keys == query).any())
            index = torch.nonzero(keys == query).item()
            self.assertEqual(episodes.answers[row], episodes.assignment_values[row, index])
            self.assertEqual(episodes.query_assignment_counts[row], 1)
            self.assertEqual(episodes.stale_values[row], STALE_SENTINEL)

    def test_replacement_latest_answer_and_controlled_overwrites(self):
        episodes = generate(mode="replacement", table_size=7, key_vocab_size=4,
                            query_key_assignments=3, seed=9)
        for row in range(4):
            occurrences = torch.nonzero(
                episodes.assignment_keys[row] == episodes.query_keys[row]).flatten()
            self.assertEqual(len(occurrences), 3)
            self.assertEqual(episodes.query_assignment_counts[row], 3)
            latest, previous = occurrences[-1], occurrences[-2]
            self.assertEqual(episodes.answers[row], episodes.assignment_values[row, latest])
            self.assertEqual(episodes.stale_values[row], episodes.assignment_values[row, previous])
            self.assertNotEqual(episodes.answers[row], episodes.stale_values[row])

    def test_answer_is_target_only_and_only_final_transition_is_scored(self):
        episodes = generate(mode="replacement", query_key_assignments=2)
        answer_position = episodes.token_ids.shape[1] - 1
        self.assertTrue(torch.equal(episodes.token_ids[:, -2], episodes.query_keys))
        self.assertTrue(torch.equal(episodes.token_ids[:, -1], episodes.answers))
        self.assertTrue(torch.equal(episodes.batch.inputs.argmax(2),
                                    episodes.token_ids[:, :answer_position]))
        self.assertTrue(torch.equal(episodes.batch.targets.argmax(2),
                                    episodes.token_ids[:, 1:]))
        self.assertEqual(episodes.batch.score_mask.sum().item(), 4)
        self.assertTrue(episodes.batch.score_mask[:, -1].all())
        self.assertFalse(episodes.batch.score_mask[:, :-1].any())
        self.assertTrue(episodes.batch.reset_mask[:, 0].all())
        self.assertFalse(episodes.batch.reset_mask[:, 1:].any())

    def test_strict_and_observed_update_boundaries(self):
        strict = generate(query_mode="strict", distractor_count=2)
        observed = generate(query_mode="observed", distractor_count=2)
        marker = 2 * 5 + 2
        self.assertTrue(strict.batch.update_mask[:, :marker].all())
        self.assertFalse(strict.batch.update_mask[:, marker:].any())
        self.assertTrue(observed.batch.update_mask.all())
        self.assertTrue(observed.batch.update_mask[:, -1].all())
        torch.testing.assert_close(strict.token_ids, observed.token_ids, rtol=0, atol=0)

    def test_count_only_lag_sweep_preserves_vocab_and_associative_content(self):
        plain = generate(distractor_count=0)
        distracted = generate(distractor_count=6)
        self.assertEqual(distracted.layout, plain.layout)
        self.assertEqual(distracted.layout.vocab_size, plain.layout.vocab_size)
        self.assertEqual(distracted.batch.inputs.shape[2], plain.batch.inputs.shape[2])
        self.assertEqual(distracted.token_ids.shape[1], plain.token_ids.shape[1] + 6)
        torch.testing.assert_close(distracted.source_to_answer_lag,
                                   plain.source_to_answer_lag + 6, rtol=0, atol=0)
        for name in ("assignment_keys", "assignment_values", "query_keys", "answers",
                     "stale_values", "query_assignment_counts", "latest_source_position"):
            torch.testing.assert_close(getattr(distracted, name), getattr(plain, name), rtol=0, atol=0)
        # Apart from inserted distractors, the emitted table/query/answer stream is identical.
        table_end = 2 * 5
        torch.testing.assert_close(distracted.token_ids[:, :table_end],
                                   plain.token_ids[:, :table_end], rtol=0, atol=0)
        torch.testing.assert_close(distracted.token_ids[:, table_end + 6:],
                                   plain.token_ids[:, table_end:], rtol=0, atol=0)

    def test_distractors_use_fixed_range_and_are_reproducible(self):
        first = generate(distractor_count=12, distractor_vocab_size=4, seed=87)
        second = generate(distractor_count=12, distractor_vocab_size=4, seed=87)
        distractors = first.token_ids[:, 2 * 5:2 * 5 + 12]
        repeated = second.token_ids[:, 2 * 5:2 * 5 + 12]
        torch.testing.assert_close(distractors, repeated, rtol=0, atol=0)
        self.assertTrue((distractors >= first.layout.distractor_start).all())
        self.assertTrue((distractors < first.layout.distractor_stop).all())
        # The private draws are per row/position, not one shared arange sequence.
        self.assertFalse(torch.equal(distractors[0], distractors[1]))

    def test_layout_ranges_ids_positions_and_one_hot_width_are_auditable(self):
        episodes = generate(mode="replacement", query_key_assignments=3,
                            distractor_count=4, distractor_vocab_size=5)
        layout = episodes.layout
        ranges = [set(layout.key_ids), set(layout.value_ids), set(layout.distractor_ids),
                  {layout.query_marker}]
        for index, left in enumerate(ranges):
            for right in ranges[index + 1:]:
                self.assertTrue(left.isdisjoint(right))
        self.assertGreaterEqual(episodes.token_ids.min().item(), 0)
        self.assertLess(episodes.token_ids.max().item(), layout.vocab_size)
        self.assertEqual(episodes.batch.inputs.shape[2], layout.vocab_size)
        self.assertEqual(episodes.batch.targets.shape[2], layout.vocab_size)
        answer_position = episodes.token_ids.shape[1] - 1
        for row in range(episodes.token_ids.shape[0]):
            source = episodes.latest_source_position[row].item()
            self.assertEqual(episodes.token_ids[row, source], episodes.answers[row])
            occurrences = torch.nonzero(
                episodes.assignment_keys[row] == episodes.query_keys[row]).flatten()
            self.assertEqual(source, 2 * occurrences[-1].item() + 1)
            self.assertEqual(episodes.source_to_answer_lag[row], answer_position - source)

    def test_uncontrolled_replacement_always_uses_latest_occurrence(self):
        # table_size=1 guarantees no repeat; one key with a larger table guarantees repeats.
        cases = [generate(mode="replacement", table_size=1, key_vocab_size=5,
                          query_key_assignments=None, batch_size=12, seed=4),
                 generate(mode="replacement", table_size=6, key_vocab_size=1,
                          query_key_assignments=None, batch_size=12, seed=5)]
        self.assertTrue((cases[0].query_assignment_counts == 1).all())
        self.assertTrue((cases[1].query_assignment_counts == 6).all())
        for episodes in cases:
            for row in range(episodes.token_ids.shape[0]):
                occurrences = torch.nonzero(
                    episodes.assignment_keys[row] == episodes.query_keys[row]).flatten()
                latest = occurrences[-1]
                self.assertEqual(episodes.answers[row], episodes.assignment_values[row, latest])

    def test_single_value_controlled_overwrite_can_equal_stale_answer(self):
        episodes = generate(mode="replacement", value_vocab_size=1,
                            query_key_assignments=3, table_size=5)
        self.assertTrue((episodes.query_assignment_counts == 3).all())
        self.assertTrue(torch.equal(episodes.stale_values, episodes.answers))

    def test_local_rng_is_reproducible_and_does_not_perturb_globals(self):
        random.seed(77)
        np.random.seed(77)
        torch.manual_seed(77)
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        first = generate(mode="replacement", query_key_assignments=2, seed=456)
        second = generate(mode="replacement", query_key_assignments=2, seed=456)
        self.assertEqual(random.getstate(), python_state)
        self.assertEqual(np.random.get_state()[0], numpy_state[0])
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
        self.assertEqual(np.random.get_state()[2:], numpy_state[2:])
        torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)
        for name in ("token_ids", "assignment_keys", "assignment_values", "query_keys",
                     "answers", "stale_values", "latest_source_position", "source_to_answer_lag"):
            torch.testing.assert_close(getattr(first, name), getattr(second, name), rtol=0, atol=0)

    def test_invalid_configs_fail_clearly(self):
        invalid = [
            ({"mode": "other"}, "mode"),
            ({"query_mode": "other"}, "query_mode"),
            ({"table_size": 9}, "table_size"),
            ({"distractor_count": -1}, "distractor_count"),
            ({"mode": "unique", "query_key_assignments": 1}, "replacement"),
            ({"mode": "replacement", "query_key_assignments": 6}, "cannot exceed"),
            ({"mode": "replacement", "key_vocab_size": 1,
              "query_key_assignments": 2}, "every table assignment"),
            ({"batch_size": 0}, "batch_size"),
            ({"table_size": False}, "table_size"),
            ({"key_vocab_size": True}, "key_vocab_size"),
            ({"value_vocab_size": 0}, "value_vocab_size"),
            ({"distractor_count": True}, "distractor_count"),
            ({"distractor_vocab_size": 0}, "distractor_vocab_size"),
            ({"seed": False}, "seed"),
            ({"seed": 1.5}, "seed"),
            ({"mode": "replacement", "query_key_assignments": 0}, "query_key_assignments"),
        ]
        for options, message in invalid:
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, message):
                generate(**options)

        edge = generate(mode="replacement", key_vocab_size=1, table_size=5,
                        query_key_assignments=5)
        self.assertTrue((edge.query_assignment_counts == 5).all())

    def test_generated_batch_runs_through_heldout_evaluator(self):
        episodes = generate(batch_size=2, table_size=3, key_vocab_size=4,
                            value_vocab_size=3, mode="replacement",
                            query_key_assignments=2, distractor_count=1)
        with contextlib.redirect_stdout(io.StringIO()):
            model = EphemeralRNN(episodes.layout.vocab_size, 4, episodes.layout.vocab_size, 1,
                                 list(range(episodes.layout.vocab_size)), updater="dfa", batch_size=2,
                                 unit_norm_weights=False, plasticity=2, forget_rate=0.1,
                                 ephemeral_fraction=0.5, enable_recurrence=True)
        result = evaluate_held_out(model, episodes.batch, learning_rate=0.01)
        self.assertEqual(result.logits.shape[:2], episodes.batch.score_mask.shape)
        self.assertEqual(result.scored_count, 2)
        self.assertEqual(result.final_hidden.shape, (2, 4))


if __name__ == "__main__":
    unittest.main()
