"""kv_tasks.py: key-value episodes, their registration, recall targets and answer classes."""
import random
import tempfile
import unittest
from collections import Counter

import torch

import kv_tasks
import preprocess
import utils
from metrics import IntervalMetrics, recall_chance, recall_targets


class GeneratorTest(unittest.TestCase):
    def samples(self, name, n=2000, seed=0):
        return kv_tasks.generate_split(name, "train", n, seed)

    def test_names(self):
        self.assertEqual(kv_tasks.parse_kv_name("kv_unique_4"), ("unique", 4, 0))
        self.assertEqual(kv_tasks.parse_kv_name("kv_reassign_8_d16"), ("reassign", 8, 16))
        self.assertIsNone(kv_tasks.parse_kv_name("3_palindrome_dataset_vary_length"))
        with self.assertRaises(ValueError):
            kv_tasks.parse_kv_name("kv_unique_11")  # more distinct keys than letters

    def test_format_and_lengths(self):
        for name, (k, d) in {"kv_unique_4": (4, 0), "kv_reassign_2": (2, 0), "kv_unique_8_d8": (8, 8)}.items():
            for text in self.samples(name, 500):
                self.assertEqual(len(text), 2 * k + d + 3)
                self.assertTrue(set(text) <= set(kv_tasks.CHARSET))
                self.assertTrue(all(c in kv_tasks.KEYS for c in text[0:2 * k:2]))
                self.assertTrue(all(c in kv_tasks.VALUES for c in text[1:2 * k:2]))
                self.assertEqual(text[2 * k:2 * k + d], "." * d)
                self.assertEqual(text[2 * k + d], "?")

    def test_unique_answer_is_the_queried_keys_value(self):
        for text in self.samples("kv_unique_8"):
            table = text[:16]
            keys = table[::2]
            self.assertEqual(len(set(keys)), 8)
            query, answer = text[-2], text[-1]
            self.assertEqual(answer, table[table.index(query) + 1])

    def test_reassign_answer_is_the_latest_value_and_differs_from_the_previous(self):
        counts = Counter()
        for text in self.samples("kv_reassign_4"):
            pairs = [(text[i], text[i + 1]) for i in range(0, 8, 2)]
            query, answer = text[-2], text[-1]
            mine = [v for key, v in pairs if key == query]
            counts[len(mine)] += 1
            self.assertEqual(answer, mine[-1])
            if len(mine) >= 2:
                self.assertNotEqual(mine[-1], mine[-2])
        # c ~ U{1..4}: reassigned (c >= 2) about 3/4 of the time
        self.assertEqual(set(counts), {1, 2, 3, 4})
        self.assertAlmostEqual(sum(v for c, v in counts.items() if c >= 2) / 2000, 0.75, delta=0.04)

    def test_deterministic_per_seed_and_split(self):
        self.assertEqual(self.samples("kv_reassign_4", 50), self.samples("kv_reassign_4", 50))
        self.assertNotEqual(self.samples("kv_reassign_4", 50), self.samples("kv_reassign_4", 50, seed=1))
        self.assertNotEqual(kv_tasks.generate_split("kv_unique_4", "train", 50),
                            kv_tasks.generate_split("kv_unique_4", "validation", 50))

    def test_classify_answer(self):
        text = "a1b2a3b4a5?a5"  # a: 1, 3, then 5; b: 2, 4
        self.assertEqual([kv_tasks.classify_answer(text, c) for c in "53124.?9"],
                         ["correct", "stale", "stale", "wrong_key", "wrong_key", "other", "other", "other"])
        self.assertEqual(kv_tasks.classify_answer("a1b1?b1", "1"), "correct")  # correct wins over overlap


class PipelineTest(unittest.TestCase):
    def test_registered_as_synthetic_with_their_charset(self):
        for name in ("kv_unique_2", "kv_unique_8", "kv_reassign_4", "kv_reassign_16_d8"):
            self.assertTrue(preprocess.is_synthetic(name))
            self.assertEqual(preprocess.dataset_keys[name], "train")
            self.assertEqual(utils.dataset_keys[name], "text")
            self.assertEqual(utils.get_charset(name), kv_tasks.CHARSET)
        self.assertEqual(len(set(kv_tasks.CHARSET)), len(kv_tasks.CHARSET))

    def test_recall_target_is_the_answer_with_its_lag(self):
        for name in ("kv_unique_4", "kv_reassign_4", "kv_unique_2_d4"):
            for text in kv_tasks.generate_split(name, "test", 300):
                targets, end = recall_targets(text, name)
                self.assertIsNone(end)
                (target, lag), = targets.items()
                self.assertEqual(target, len(text) - 1)
                self.assertEqual(text[target], text[target - lag - 1])  # the latest value of the key
                self.assertEqual(text[target - lag - 2], text[-2])  # ... bound to the queried key
        self.assertEqual(recall_targets("a1b2a3b4a5?a5", "kv_reassign_5"), ({12: 2}, None))
        self.assertAlmostEqual(recall_chance("kv_unique_4"), 0.1)

    def test_rows_load_through_preprocess_and_collate(self):
        name = "kv_reassign_2"
        with tempfile.TemporaryDirectory() as out:
            kv_tasks.generate_dataset(name, sizes={"train": 40, "validation": 8, "test": 8}, out_dir=out)
            from datasets import load_from_disk
            rows = preprocess.preprocess_rows(load_from_disk(f"{out}/{name}")["validation"], name)
        texts, indices, onehot = preprocess.OneHotCollate(len(kv_tasks.CHARSET))(list(rows))
        self.assertEqual(onehot.shape, (8, 7, len(kv_tasks.CHARSET)))
        self.assertEqual("".join(kv_tasks.CHARSET[i] for i in indices[0]), texts[0])


class AnswerClassMetricsTest(unittest.TestCase):
    def test_interval_metrics_classify_answers(self):
        name = "kv_reassign_5"
        charset = kv_tasks.CHARSET
        texts = ["a1b2a3b4a5?a5"] * 4
        onehot = torch.nn.functional.one_hot(
            torch.tensor([[charset.index(c) for c in t] for t in texts]), len(charset)).float()
        preds = onehot[:, 1:].argmax(-1).t().clone()  # [T-1, B], all correct
        for b, c in zip(range(1, 4), "34."):  # stale, wrong key, other
            preds[-1, b] = charset.index(c)
        metrics = IntervalMetrics(name)
        metrics.update(texts, onehot, preds, torch.ones_like(preds, dtype=torch.float))
        summary = metrics.summary()
        self.assertEqual([summary[f"kv_{c}"] for c in ("correct", "stale", "wrong_key", "other")], [0.25] * 4)
        self.assertEqual(summary["recall_acc"], 0.25)
        self.assertEqual(summary["recall_acc_lag_2"], 0.25)

    def test_other_datasets_report_no_answer_classes(self):
        metrics = IntervalMetrics("long_range_memory_dataset")
        onehot = torch.nn.functional.one_hot(torch.tensor([["0?!123,. ".index(c) for c in "0?1!1"]]), 9).float()
        metrics.update(["0?1!1"], onehot, onehot[:, 1:].argmax(-1).t(), torch.ones(4, 1))
        self.assertFalse(any(k.startswith("kv_") for k in metrics.summary()))


if __name__ == "__main__":
    unittest.main()
