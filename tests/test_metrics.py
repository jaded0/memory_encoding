import unittest

import torch

from metrics import IntervalMetrics, recall_chance, recall_targets

PALINDROME = "3_palindrome_dataset_vary_length"


class RecallTargetsTest(unittest.TestCase):
    def assert_recalls_earlier_character(self, text, dataset_name):
        targets, _ = recall_targets(text, dataset_name)
        self.assertTrue(targets)
        for target, lag in targets.items():
            self.assertEqual(text[target], text[target - lag - 1], (text, target, lag))

    def test_each_task_marks_positions_determined_by_an_earlier_character(self):
        cases = [
            (",!.!,  ", PALINDROME), ("12,.,21", PALINDROME), ("32.23", "2_small_palindrome_dataset_vary_length"),
            ("1.2.1", "palindrome_dataset"), ("0000?.0!.", "long_range_memory_dataset"), ("3.23.", "4_resequence"),
        ]
        for text, dataset_name in cases:
            self.assert_recalls_earlier_character(text, dataset_name)

    def test_palindrome_lags_and_end_position(self):
        self.assertEqual(recall_targets(",!.!,  ", PALINDROME), ({3: 1, 4: 3}, 5))
        self.assertEqual(recall_targets("12,.,21", PALINDROME), ({4: 1, 5: 3, 6: 5}, None))

    def test_tasks_without_memory_targets(self):
        self.assertEqual(recall_targets("!!!!!", "1_resequence"), ({}, None))
        self.assertEqual(recall_targets("once upon a time", "roneneldan/tinystories"), ({}, None))
        self.assertIsNone(recall_chance("roneneldan/tinystories"))

    def test_chance_levels(self):
        self.assertAlmostEqual(recall_chance(PALINDROME), 1 / 7)
        self.assertAlmostEqual(recall_chance("2_small_palindrome_dataset_vary_length"), 1 / 2)
        self.assertAlmostEqual(recall_chance("long_range_memory_dataset"), 1 / 6)


class IntervalMetricsTest(unittest.TestCase):
    def batch(self, texts, charset="0?!123,. "):
        index = {c: i for i, c in enumerate(charset)}
        length = max(map(len, texts))
        onehot = torch.zeros(len(texts), length, len(charset))
        for b, text in enumerate(texts):
            for t, c in enumerate(text):
                onehot[b, t, index[c]] = 1
        return onehot

    def test_perfect_and_wrong_recall(self):
        texts = [",!.!,  ", "1.1    "]
        onehot = self.batch(texts)
        targets = onehot[:, 1:].argmax(-1).t()  # [T-1, B]
        wrong = targets.clone()
        wrong[2, 0] = (wrong[2, 0] + 1) % 9  # break text 0's lag-1 recall (target index 3)
        losses = torch.ones_like(targets, dtype=torch.float)

        metrics = IntervalMetrics(PALINDROME)
        metrics.update(texts, onehot, targets, losses)
        perfect = metrics.summary()
        self.assertEqual(perfect["recall_acc"], 1.0)
        self.assertEqual(perfect["recall_seq_exact"], 1.0)
        self.assertEqual(perfect["final_char_acc"], 1.0)

        metrics.reset()
        metrics.update(texts, onehot, wrong, losses)
        broken = metrics.summary()
        self.assertAlmostEqual(broken["recall_acc"], 2 / 3)  # 3 recall targets, one wrong
        self.assertAlmostEqual(broken["recall_acc_lag_1"], 1 / 2)
        self.assertEqual(broken["recall_acc_lag_3"], 1.0)
        self.assertAlmostEqual(broken["recall_seq_exact"], 1 / 2)

    def test_padding_is_excluded(self):
        texts = ["0?1!1", "0?1!"]  # second sequence is padded by one all-zero step
        onehot = self.batch(texts)
        targets = onehot[:, 1:].argmax(-1).t()
        metrics = IntervalMetrics("long_range_memory_dataset")
        metrics.update(texts, onehot, targets, torch.zeros_like(targets, dtype=torch.float))
        summary = metrics.summary()
        self.assertEqual(summary["token_acc"], 1.0)
        self.assertEqual(summary["final_char_acc"], 1.0)


if __name__ == "__main__":
    unittest.main()
