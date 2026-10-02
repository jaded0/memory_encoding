"""bench_tasks.py: mqar / parity / mod-m / selective-copy episodes, held-out disjointness, registration."""
import tempfile
import unittest

import bench_tasks
import kv_tasks
import preprocess
import utils
from metrics import IntervalMetrics, recall_chance, recall_targets


def reference_answers(text, name):
    """Independent re-derivation of every answer (index, character, source index) from the text."""
    family, params = bench_tasks.parse_name(name)
    out = []
    if family == "mqar":
        k, q = params
        table = {text[2 * i]: (2 * i + 1, text[2 * i + 1]) for i in range(k)}
        for j in range(q):
            at = 2 * k + 3 * j
            assert text[at] == "?"
            source, value = table[text[at + 1]]
            out.append((at + 2, value, source))
    elif family == "count":
        modulus, length = params
        total = sum(int(c) for c in text[:length])
        out.append((length + 1, str(total % modulus), 0))
    else:
        n, t = params
        body, tail = text[:t], text[t + 1:]
        content = [(i, c) for i, c in enumerate(body) if c != "."]
        for j, (i, c) in enumerate(content):
            out.append((t + 1 + j, tail[j], i))
            assert c == tail[j]
    return out


class GeneratorTest(unittest.TestCase):
    NAMES = ("mqar_4", "mqar_6_q2", "mod3_8", "parity_8", "mod5_6", "selcopy_3_9", "selcopy_4_12")

    def samples(self, name, split="train", n=1500, seed=0):
        return bench_tasks.generate_split(name, split, n, seed)

    def test_names(self):
        self.assertEqual(bench_tasks.parse_name("mqar_4"), ("mqar", (4, 4)))
        self.assertEqual(bench_tasks.parse_name("mqar_8_q3"), ("mqar", (8, 3)))
        self.assertEqual(bench_tasks.parse_name("parity_8"), ("count", (2, 8)))
        self.assertEqual(bench_tasks.parse_name("mod3_12"), ("count", (3, 12)))
        self.assertEqual(bench_tasks.parse_name("selcopy_3_9"), ("selcopy", (3, 9)))
        for other in ("kv_unique_4", "3_palindrome_dataset_vary_length", "long_range_memory_dataset"):
            self.assertIsNone(bench_tasks.parse_name(other))
            self.assertFalse(bench_tasks.is_bench(other))
        for bad in ("mqar_11", "mqar_4_q5", "selcopy_5_3", "parity_0"):
            with self.assertRaises(ValueError):
                bench_tasks.parse_name(bad)

    def test_format_and_lengths(self):
        lengths = {"mqar_4": 20, "mqar_6_q2": 12 + 6, "mod3_8": 10, "parity_8": 10, "mod5_6": 8,
                   "selcopy_3_9": 13, "selcopy_4_12": 17}
        for name in self.NAMES:
            for text in self.samples(name, n=300):
                self.assertEqual(len(text), lengths[name], text)
                self.assertTrue(set(text) <= set(kv_tasks.CHARSET))
        for text in self.samples("mqar_4", n=300):
            self.assertEqual(len(set(text[0:8:2])), 4)  # distinct keys
            self.assertEqual(len({text[9 + 3 * j] for j in range(4)}), 4)  # distinct queries
        for text in self.samples("selcopy_3_9", n=300):
            self.assertEqual(len(set(text[10:])), 3)  # distinct digits

    def test_answers_match_an_independent_reference(self):
        for name in self.NAMES:
            for text in self.samples(name, n=400):
                ref = reference_answers(text, name)
                targets, end = recall_targets(text, name)
                self.assertIsNone(end)
                self.assertEqual(sorted(targets), [a for a, _, _ in ref])
                for answer, char, source in ref:
                    self.assertEqual(text[answer], char)
                    self.assertEqual(targets[answer], answer - source - 1)

    def test_mqar_lag_is_distance_to_the_stored_value(self):
        text = "a1b2c3d4?c3?a1?d4?b2"
        self.assertEqual(recall_targets(text, "mqar_4"), ({10: 10 - 5 - 1, 13: 13 - 1 - 1, 16: 16 - 7 - 1, 19: 19 - 3 - 1}, None))

    def test_state_tracking_values(self):
        self.assertEqual(recall_targets("01101001?0", "parity_8"), ({9: 8}, None))
        counts = {"mod3_8": set(), "parity_8": set()}
        for name in counts:
            for text in self.samples(name, n=500):
                counts[name].add(text[-1])
        self.assertEqual(counts["mod3_8"], set("012"))
        self.assertEqual(counts["parity_8"], set("01"))
        # an explicit case: 5 ones -> parity 1, mod 3 = 2
        self.assertEqual(reference_answers("01101101?1", "parity_8")[0][1], "1")
        self.assertEqual(reference_answers("01101101?2", "mod3_8")[0][1], "2")

    def test_answer_balance(self):
        for name, classes in (("mod3_12", 3), ("parity_12", 2)):
            hist = {}
            for text in self.samples(name, n=3000):
                hist[text[-1]] = hist.get(text[-1], 0) + 1
            self.assertEqual(len(hist), classes)
            self.assertLess(max(hist.values()) / 3000, 0.45 if classes == 3 else 0.55)

    def test_selcopy_known_episode(self):
        text = "..4.7..2.?472"
        self.assertEqual(recall_targets(text, "selcopy_3_9"), ({10: 10 - 2 - 1, 11: 11 - 4 - 1, 12: 12 - 7 - 1}, None))

    def test_deterministic_per_seed_and_split(self):
        for name in self.NAMES:
            self.assertEqual(self.samples(name, n=40), self.samples(name, n=40))
            self.assertNotEqual(self.samples(name, n=40), self.samples(name, n=40, seed=1))
            self.assertNotEqual(self.samples(name, n=40), self.samples(name, split="validation", n=40))

    def test_held_out_cores_are_disjoint_from_training(self):
        for name in ("mqar_4", "mod3_8", "parity_8", "selcopy_3_9"):
            train = {bench_tasks.core(t) for t in self.samples(name, "train", 20000)}
            val = {bench_tasks.core(t) for t in self.samples(name, "validation", 3000)}
            test = {bench_tasks.core(t) for t in self.samples(name, "test", 3000)}
            self.assertFalse(train & val, name)
            self.assertFalse(train & test, name)
            self.assertFalse(val & test, name)
            self.assertTrue(val and test)
        # the split is a function of the core alone: a train text never hashes elsewhere
        for text in self.samples("mqar_4", "validation", 200):
            self.assertEqual(bench_tasks.split_of(text), "validation")

    def test_held_out_has_enough_distinct_strings(self):
        self.assertGreater(len({bench_tasks.core(t) for t in self.samples("parity_8", "validation", 3000)}), 15)
        self.assertGreater(len({bench_tasks.core(t) for t in self.samples("mod3_12", "validation", 3000)}), 200)


class PipelineTest(unittest.TestCase):
    def test_registered_as_synthetic_with_the_kv_charset(self):
        for name in ("mqar_4", "mod3_8", "parity_12", "selcopy_3_9", "mqar_5_q2"):
            self.assertTrue(preprocess.is_synthetic(name))
            self.assertEqual(utils.get_charset(name), kv_tasks.CHARSET)
        for name in bench_tasks.registered_names():
            self.assertEqual(preprocess.dataset_keys[name], "train")
            self.assertEqual(utils.dataset_keys[name], "text")

    def test_recall_chance(self):
        self.assertAlmostEqual(recall_chance("mqar_4"), 0.1)
        self.assertAlmostEqual(recall_chance("parity_8"), 0.5)
        self.assertAlmostEqual(recall_chance("mod3_8"), 1 / 3)
        self.assertAlmostEqual(recall_chance("selcopy_3_9"), 0.1)

    def test_interval_metrics_score_every_answer(self):
        import torch
        name, charset = "mqar_2", kv_tasks.CHARSET
        text = "a1b2?b2?a1"
        onehot = torch.nn.functional.one_hot(torch.tensor([[charset.index(c) for c in text]]), len(charset)).float()
        preds = onehot[:, 1:].argmax(-1).t().clone()  # all correct
        preds[8, 0] = charset.index("9")              # the last answer wrong
        metrics = IntervalMetrics(name)
        metrics.update([text], onehot, preds, torch.ones_like(preds, dtype=torch.float))
        summary = metrics.summary()
        self.assertEqual(summary["recall_acc"], 0.5)
        self.assertEqual(summary["recall_seq_exact"], 0.0)
        self.assertNotIn("kv_correct", summary)

    def test_rows_load_through_preprocess_and_collate(self):
        name = "selcopy_3_9"
        with tempfile.TemporaryDirectory() as out:
            bench_tasks.generate_dataset(name, sizes={"train": 40, "validation": 8, "test": 8}, out_dir=out)
            from datasets import load_from_disk
            rows = preprocess.preprocess_rows(load_from_disk(f"{out}/{name}")["validation"], name)
        texts, indices, onehot = preprocess.OneHotCollate(len(kv_tasks.CHARSET))(list(rows))
        self.assertEqual(onehot.shape, (8, 13, len(kv_tasks.CHARSET)))
        self.assertEqual("".join(kv_tasks.CHARSET[i] for i in indices[0]), texts[0])


if __name__ == "__main__":
    unittest.main()
