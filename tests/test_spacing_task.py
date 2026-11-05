"""Standalone spacing generator timing, independence and online resume tests."""
import copy
import random
import unittest
from collections import Counter

import spacing_task as spacing


class SpacingGeneratorTest(unittest.TestCase):
    def test_exact_schedule_and_constant_budget(self):
        for interval in (1, 2, 4, 16):
            for delay in (0, 1, 8):
                rows = spacing.generate_stream(random.Random(5), interval, delay, length=64)
                studies = [r for r in rows if r["kind"] == "study"]
                positions = [r["position"] for r in studies]
                self.assertEqual(positions, [63 - delay - 1 - 2 * interval,
                                             63 - delay - 1 - interval, 63 - delay - 1])
                self.assertEqual(len(studies), 3)
                self.assertEqual(sum(r["is_query"] for r in rows), 1)
                self.assertTrue(all(r["kind"] == "distractor"
                                    for r in rows[positions[-1] + 1:-1]))
                self.assertEqual(len(rows[positions[-1] + 1:-1]), delay)
                self.assertEqual(len(rows), 64)
                self.assertEqual({len(r["text"]) for r in rows}, {9})
                self.assertEqual(sum(len(r["text"]) - 1 for r in rows), 512)
                self.assertEqual([r["position"] for r in rows if r["wipe_fast"]], [0])
                if interval == 1:
                    self.assertEqual(positions, list(range(positions[0], positions[0] + 3)))

    def test_bindings_persist_and_only_query_is_recall_target(self):
        rows = spacing.generate_stream(random.Random(7), 4, 5)
        table = rows[0]["context"]
        context = dict(zip(table[::2], table[1::2]))
        self.assertEqual(len(context), 4)
        self.assertEqual(len(set(context.values())), 4)
        for row in rows:
            self.assertEqual(row["context"], table)
            self.assertEqual(row["answer"], context[row["query"]])
            self.assertTrue(set(row["text"]) <= set(spacing.CHARSET))
            if row["kind"] == "study":
                self.assertEqual(row["text"], "." + table)
                self.assertNotIn("?", row["text"])
            elif row["kind"] == "distractor":
                self.assertEqual(row["text"], "." * 9)
            else:
                self.assertEqual(row["text"], "." * 6 + "?" + row["query"] + row["answer"])
            self.assertEqual(row["answer_index"], 8 if row["is_query"] else -1)

    def test_prediction_step_timing_from_independent_flattened_targets(self):
        for n_keys in (1, 4, 10):
            rows = spacing.generate_stream(random.Random(2), 3, 2, n_keys=n_keys)
            query = rows[-1]
            # Reconstruct the clock train_batch uses: every row emits text[1:].
            targets = [(r["position"], char_index, char)
                       for r in rows for char_index, char in enumerate(r["text"][1:], 1)]
            answer_step = next(i for i, (p, c, _) in enumerate(targets)
                               if p == query["position"] and c == query["answer_index"])
            study_positions = [r["position"] for r in rows if r["kind"] == "study"]
            study = rows[study_positions[-1]]["text"]
            source_index = study.index(query["query"]) + 1
            source_step = next(i for i, (p, c, _) in enumerate(targets)
                               if p == study_positions[-1] and c == source_index)
            self.assertEqual(query["query_lag_steps"], answer_step - source_step)
            self.assertEqual(query["query_lag"], query["position"] - study_positions[-1])

    def test_deterministic_disjoint_seeds_splits_and_paired_eval(self):
        args = dict(intervals=(1,), delays=(0,), length=50)
        first = spacing.generate_split("test", 8, seed=3, **args)
        self.assertEqual(first, spacing.generate_split("test", 8, seed=3, **args))
        self.assertNotEqual(first, spacing.generate_split("validation", 8, seed=3, **args))
        self.assertNotEqual(first, spacing.generate_split("test", 8, seed=4, **args))
        other = spacing.generate_split("test", 8, seed=3, intervals=(16,), delays=(8,), length=50)
        self.assertEqual([(r["context"], r["query"]) for r in first],
                         [(r["context"], r["query"]) for r in other])
        self.assertEqual(first, spacing.generate_split("test", 3, seed=3, **args) +
                         spacing.generate_split("test", 5, seed=3, start_stream=3, **args))

    def test_stream_major_and_random_per_stream(self):
        length = 8
        rows = spacing.generate_split("test", 100, intervals=(1,), delays=(0,), length=length)
        for stream in range(100):
            block = rows[stream * length:(stream + 1) * length]
            self.assertEqual({r["stream"] for r in block}, {stream})
            self.assertEqual([r["position"] for r in block], list(range(length)))
        self.assertEqual(len({rows[s * length]["context"] for s in range(100)}), 100)

    def test_default_training_schedule_mixes_interval_and_delay(self):
        rows = spacing.generate_split("train", 300)
        cells = Counter((r["interval"], r["delay"]) for r in rows if r["is_query"])
        self.assertEqual(set(cells), {(i, d) for i in spacing.DEFAULT_INTERVALS
                                    for d in spacing.DEFAULT_DELAYS})
        self.assertTrue(all(count > 10 for count in cells.values()))

    def test_answer_classification(self):
        rows = spacing.generate_stream(random.Random(9), 1, 0)
        query = rows[-1]
        self.assertEqual(spacing.classify_answer(query, query["answer"]), "correct")
        wrong = next(v for v in query["context"][1::2] if v != query["answer"])
        self.assertEqual(spacing.classify_answer(query, wrong), "wrong_key")
        self.assertEqual(spacing.classify_answer(query, "?"), "other")
        with self.assertRaises(ValueError):
            spacing.classify_answer(rows[0], "0")

    def test_invalid_parameters(self):
        for interval, delay, repetitions in ((0, 0, 3), (1, -1, 3), (1, 0, 0),
                                              (1.5, 0, 3), (True, 0, 3)):
            with self.assertRaises(ValueError):
                spacing.minimum_length(interval, delay, repetitions)
        with self.assertRaises(ValueError):
            spacing.generate_stream(random.Random(0), 4, 2, length=8)
        for n_keys in (0, 11):
            with self.assertRaises(ValueError):
                spacing.generate_stream(random.Random(0), 1, 0, n_keys=n_keys)
        for kwargs in ({"intervals": ()}, {"delays": ()}, {"n_streams": -1}):
            args = {"n_streams": 1, **kwargs}
            with self.assertRaises(ValueError):
                spacing.generate_split("test", **args)

    def test_generator_does_not_touch_global_random_state(self):
        before = random.getstate()
        spacing.generate_split("test", 3)
        iterator = spacing.SpacingBatchStream(2)
        next(iterator)
        self.assertEqual(random.getstate(), before)


class SpacingBatchTest(unittest.TestCase):
    def test_existing_one_hot_collate_accepts_all_row_kinds(self):
        from preprocess import OneHotCollate

        iterator = spacing.SpacingBatchStream(2, intervals=(1,), delays=(1,), length=6)
        kinds = set()
        for _ in range(iterator.length):
            rows = next(iterator)
            for row in rows:
                kinds.add(row["kind"])
                row["tensor"] = [spacing.CHARSET.index(c) for c in row["text"]]
            texts, _, onehot = OneHotCollate(len(spacing.CHARSET))(rows)
            self.assertEqual(texts, [r["text"] for r in rows])
            self.assertEqual(tuple(onehot.shape), (2, 9, len(spacing.CHARSET)))
            self.assertEqual(onehot.argmax(-1).tolist(), [r["tensor"] for r in rows])
        self.assertEqual(kinds, {"study", "distractor", "query"})

    def test_batch_rows_keep_stream_identity_then_draw_fresh_bindings(self):
        iterator = spacing.SpacingBatchStream(3, intervals=(1,), delays=(0,), length=5)
        rows = [next(iterator) for _ in range(11)]
        for position, batch in enumerate(rows):
            self.assertEqual([r["stream"] for r in batch],
                             list(range((position // 5) * 3, (position // 5 + 1) * 3)))
            self.assertEqual({r["position"] for r in batch}, {position % 5})
            self.assertEqual({r["wipe_fast"] for r in batch}, {position % 5 == 0})
        for b in range(3):
            self.assertEqual(len({rows[t][b]["context"] for t in range(5)}), 1)
            self.assertNotEqual(rows[0][b]["context"], rows[5][b]["context"])
            self.assertNotEqual(rows[5][b]["context"], rows[10][b]["context"])

    def test_resume_exact_at_every_position_including_stream_boundaries(self):
        kwargs = dict(seed=12, intervals=(1, 3), delays=(0, 2), length=12)
        reference = spacing.SpacingBatchStream(2, **kwargs)
        batches = [next(reference) for _ in range(40)]
        for offset in range(25):
            first = spacing.SpacingBatchStream(2, **kwargs)
            for _ in range(offset):
                next(first)
            state = copy.deepcopy(first.state_dict())
            resumed = spacing.SpacingBatchStream(2, **kwargs)
            resumed.load_state_dict(state)
            self.assertEqual([next(resumed) for _ in range(15)], batches[offset:offset + 15])

    def test_reject_mismatched_resume_or_invalid_batch(self):
        source = spacing.SpacingBatchStream(2)
        state = source.state_dict()
        for target in (spacing.SpacingBatchStream(3), spacing.SpacingBatchStream(2, seed=1)):
            with self.assertRaises(ValueError):
                target.load_state_dict(state)
        for key, value in (("version", 2), ("sequence_count", -1)):
            bad = copy.deepcopy(state)
            bad[key] = value
            with self.assertRaises(ValueError):
                source.load_state_dict(bad)
        with self.assertRaises(ValueError):
            spacing.SpacingBatchStream(0)

    def test_returned_rows_cannot_mutate_cached_stream(self):
        iterator = spacing.SpacingBatchStream(2)
        row = next(iterator)[0]
        row["context"] = "corrupted"
        self.assertNotEqual(next(iterator)[0]["context"], "corrupted")


if __name__ == "__main__":
    unittest.main()
