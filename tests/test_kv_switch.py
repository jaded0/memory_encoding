"""kv_switch.py: switching key-value streams, their stream-ordered loading, metrics and stream_eval."""
import argparse
import contextlib
import copy
import io
import random
import tempfile
import unittest
from collections import Counter

import torch
import torch.nn.functional as F

import heldout
import kv_switch
import kv_tasks
import preprocess
import stream_eval
import train as train_module
import utils
from ephemeral_model import EphemeralRNN
from metrics import CARRIED_LAG, IntervalMetrics, recall_chance, recall_targets
from reproducibility import DataStream, StreamSampler, seed_everything

CHARSET = kv_tasks.CHARSET


def reference_metadata(texts, s):
    """Re-derives every row's metadata from the texts and the switch period alone, without the
    generator's code: the bindings are read off the shown pairs and answers, a new context starts
    at every multiple of s, and a binding is 'last shown' when its key appears in a pair or the query."""
    rows, context, previous, last_shown = [], {}, {}, {}
    for position, text in enumerate(texts):
        if position % s == 0:
            previous, context, last_shown = context, {}, {}
        pairs = [(text[i], text[i + 1]) for i in range(0, 2 * kv_switch.PAIRS, 2)]
        query, answer = text[-2], text[-1]
        shown = [key for key, _ in pairs]
        for key, value in pairs + [(query, answer)]:
            assert context.get(key, value) == value, "a key changed value within a context"
            context[key] = value
        lag = 0 if query in shown else position - last_shown[query]
        rows.append({"segment": position // s, "since_switch": position % s, "query_lag": lag,
                     "carried": lag > 0,
                     # None: the previous context exists but never showed this key (unknowable here)
                     "stale": "" if position < s else previous.get(query)})
        for key in shown + [query]:
            last_shown[key] = position
    return rows


def stream(name="kvswitch_s4", seed=0, index=0, split="train"):
    s, length = kv_switch.parse_switch_name(name)
    rows = kv_switch.generate_split(name, split, index + 1, seed)
    return rows[index * length:(index + 1) * length]


class GeneratorTest(unittest.TestCase):
    def test_names(self):
        self.assertEqual(kv_switch.parse_switch_name("kvswitch_s16"), (16, 1024))
        self.assertEqual(kv_switch.parse_switch_name("kvswitch_s4_l64"), (4, 64))
        self.assertIsNone(kv_switch.parse_switch_name("kv_unique_4"))
        self.assertFalse(kv_tasks.is_kv("kvswitch_s4"))
        self.assertEqual(kv_switch.switch_name(4, 64), "kvswitch_s4_l64")
        self.assertEqual(kv_switch.switch_name(256), "kvswitch_s256")
        with self.assertRaises(ValueError):
            kv_switch.parse_switch_name("kvswitch_s0")

    def test_format(self):
        for row in stream("kvswitch_s16", index=1):
            text = row["text"]
            self.assertEqual(len(text), kv_switch.SEQUENCE_LENGTH)
            self.assertTrue(all(c in kv_tasks.KEYS for c in text[0:4:2] + text[5]))
            self.assertTrue(all(c in kv_tasks.VALUES for c in text[1:4:2] + text[6]))
            self.assertEqual(text[4], "?")
            self.assertNotEqual(text[0], text[2])  # two different keys per sequence

    def test_metadata_matches_an_independent_reference(self):
        for name in ("kvswitch_s1_l64", "kvswitch_s4_l64", "kvswitch_s16", "kvswitch_s256"):
            s = kv_switch.parse_switch_name(name)[0]
            for index in (0, 2):
                rows = stream(name, index=index, seed=3)
                reference = reference_metadata([row["text"] for row in rows], s)
                for row, expected in zip(rows, reference):
                    if expected["stale"] is None:
                        expected["stale"] = row["stale"]
                        self.assertIn(row["stale"], kv_tasks.VALUES)
                        self.assertNotEqual(row["stale"], row["text"][-1])
                    self.assertEqual({k: row[k] for k in expected}, expected)
                self.assertEqual([row["position"] for row in rows], list(range(len(rows))))
                self.assertTrue(all(row["stream"] == index for row in rows))

    def test_switch_schedule(self):
        s = 16
        rows = stream("kvswitch_s16")
        keys = set(kv_switch.parse_bindings(rows[0]["context"]))
        self.assertEqual(len(keys), kv_switch.N_KEYS)
        for position, row in enumerate(rows):
            context = kv_switch.parse_bindings(row["context"])
            self.assertEqual(set(context), keys)  # the stream keeps its keys
            self.assertEqual(len(set(context.values())), kv_switch.N_KEYS)  # distinct values
            if position % s:
                self.assertEqual(row["context"], rows[position - 1]["context"])
                self.assertEqual(row["previous"], rows[position - 1]["previous"])
            elif position:
                previous = kv_switch.parse_bindings(rows[position - 1]["context"])
                self.assertEqual(kv_switch.parse_bindings(row["previous"]), previous)
                self.assertFalse(set(previous.values()) & set(context.values()))  # disjoint values
            else:
                self.assertEqual((row["previous"], row["stale"]), ("", ""))
            text = row["text"]
            self.assertEqual(text[-1], context[text[-2]])  # the answer is the current binding
        self.assertEqual(sum(row["since_switch"] == 0 for row in rows), 1024 // s)

    def test_carried_queries(self):
        # S = 1: nothing to carry. Large S: carried about CARRY_PROB of the time, lag >= 1.
        self.assertFalse(any(row["carried"] for row in stream("kvswitch_s1")))
        rows = [row for i in range(4) for row in stream("kvswitch_s256", index=i)]
        later = [row for row in rows if row["since_switch"] >= 8]
        self.assertAlmostEqual(sum(row["carried"] for row in later) / len(later), kv_switch.CARRY_PROB, delta=0.03)
        lags = Counter(row["query_lag"] for row in rows if row["carried"])
        self.assertEqual(min(lags), 1)
        self.assertGreater(lags[1], lags[3])
        self.assertTrue(all(not row["carried"] for row in rows if row["since_switch"] == 0))

    def test_deterministic_per_seed_and_split(self):
        name = "kvswitch_s4_l64"
        self.assertEqual(kv_switch.generate_split(name, "train", 2), kv_switch.generate_split(name, "train", 2))
        self.assertNotEqual(kv_switch.generate_split(name, "train", 2), kv_switch.generate_split(name, "train", 2, seed=1))
        self.assertNotEqual(kv_switch.generate_split(name, "train", 2), kv_switch.generate_split(name, "validation", 2))

    def test_classify_answer(self):
        row = {"text": "c7a2?a2", "stale": "5", "previous": "a5c6d8b9", "context": "a2c7d0b1"}
        self.assertEqual([kv_switch.classify_answer(row, c) for c in "2567013?"],
                         ["correct", "stale", "stale_other", "wrong_key", "wrong_key", "wrong_key",
                          "other", "other"])
        first = {"text": "c7a2?a2", "stale": "", "previous": "", "context": "a2c7d0b1"}
        self.assertEqual(kv_switch.classify_answer(first, "5"), "other")


class PipelineTest(unittest.TestCase):
    def test_registered_as_synthetic_with_the_kv_charset(self):
        for name in ("kvswitch_s1", "kvswitch_s256", "kvswitch_s4_l64"):
            self.assertTrue(preprocess.is_synthetic(name))
            self.assertEqual(preprocess.dataset_keys[name], "train")
            self.assertEqual(utils.dataset_keys[name], "text")
            self.assertEqual(utils.get_charset(name), CHARSET)
        self.assertNotIn("kvswitch_s256_l64", preprocess.dataset_keys)  # S > L is not registered

    def test_recall_targets_and_chance(self):
        self.assertEqual(recall_targets("c7a2?a2", "kvswitch_s4"), ({6: 2}, None))
        self.assertEqual(recall_targets("c7a2?c7", "kvswitch_s4"), ({6: 4}, None))
        self.assertEqual(recall_targets("d5c7?a2", "kvswitch_s4"), ({6: CARRIED_LAG}, None))  # carried
        for row in stream("kvswitch_s16"):
            (target, lag), = recall_targets(row["text"], "kvswitch_s16")[0].items()
            self.assertEqual(lag == CARRIED_LAG, row["carried"])
            if not row["carried"]:
                self.assertEqual(row["text"][target], row["text"][target - lag - 1])
        self.assertAlmostEqual(recall_chance("kvswitch_s4"), 0.1)

    def test_interval_metrics_split_carried_and_in_sequence(self):
        texts = ["c7a2?a2", "c7a2?a2", "d5c7?a2", "d5c7?a2"]
        onehot = F.one_hot(torch.tensor([[CHARSET.index(c) for c in t] for t in texts]), len(CHARSET)).float()
        preds = onehot[:, 1:].argmax(-1).t().clone()
        preds[-1, 1] = CHARSET.index("7")  # one in-sequence miss
        metrics = IntervalMetrics("kvswitch_s4")
        metrics.update(texts, onehot, preds, torch.ones_like(preds, dtype=torch.float))
        summary = metrics.summary()
        self.assertEqual(summary["recall_acc_in_sequence"], 0.5)
        self.assertEqual(summary["recall_acc_carried"], 1.0)
        self.assertEqual(summary["recall_acc"], 0.75)

    def test_wipe_every_must_divide_the_stream_length(self):
        base = ["--dataset", "kvswitch_s4_l64"]
        for ok in ("1", "8", "64"):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(train_module.parse_args(base + ["--wipe_every", ok]).wipe_every, int(ok))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            train_module.parse_args(base + ["--wipe_every", "48"])


class StreamSamplerTest(unittest.TestCase):
    def order(self, sampler):
        return list(iter(sampler))

    def test_each_batch_row_follows_one_stream_in_order(self):
        n_streams, length, batch = 7, 5, 3
        sampler = StreamSampler(n_streams * length, length, batch, torch.Generator().manual_seed(0))
        order = self.order(sampler)
        self.assertEqual(len(order), len(sampler))
        self.assertEqual(len(order), 2 * batch * length)  # 7 streams -> 2 whole groups
        batches = [order[i:i + batch] for i in range(0, len(order), batch)]
        for g in range(2):
            group = batches[g * length:(g + 1) * length]
            for row in range(batch):
                indices = [b[row] for b in group]
                stream_id = indices[0] // length
                self.assertEqual(indices, [stream_id * length + t for t in range(length)])
        streams = [i // length for i in order]
        self.assertEqual(len(set(streams)), 2 * batch)  # each stream used once per epoch

    def test_seeded_resume_continues_the_order(self):
        def stream_of(seed):
            generator = torch.Generator().manual_seed(seed)
            sampler = StreamSampler(40, 4, 2, generator)
            loader = torch.utils.data.DataLoader(list(range(40)), batch_size=2, sampler=sampler,
                                                 generator=generator, drop_last=True)
            return DataStream(loader)

        full = stream_of(5)
        reference = [next(full).tolist() for _ in range(30)]  # crosses an epoch (20 batches)
        first = stream_of(5)
        for _ in range(13):
            next(first)
        state = first.state_dict()
        resumed = stream_of(5)
        resumed.load_state_dict(state)
        self.assertEqual([next(resumed).tolist() for _ in range(17)], reference[13:])
        self.assertNotEqual(reference[:20], reference[20:])  # a new stream permutation per epoch

    def test_rejects_partial_streams(self):
        with self.assertRaises(ValueError):
            StreamSampler(10, 4, 1, torch.Generator())
        with self.assertRaises(ValueError):
            StreamSampler(8, 4, 3, torch.Generator())

    def test_load_and_preprocess_data_keeps_streams(self):
        name = "kvswitch_s4_l64"
        with tempfile.TemporaryDirectory() as out:
            kv_switch.generate_dataset(name, streams={"train": 4, "validation": 2, "test": 2}, out_dir=out)
            from datasets import load_from_disk
            raw = load_from_disk(f"{out}/{name}")
            original = preprocess.load_from_disk
            preprocess.load_from_disk = lambda path: raw
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    loader = preprocess.load_and_preprocess_data(name, batch_size=2, seed=1)
                loader.num_workers = 0
                batches = [texts for texts, _, _ in loader]
            finally:
                preprocess.load_from_disk = original
        self.assertEqual(len(batches), 2 * 64)
        train = raw["train"]
        by_text = {}
        for g in range(2):
            for row in range(2):
                texts = [batches[g * 64 + t][row] for t in range(64)]
                matches = [s for s in range(4) if train[s * 64:(s + 1) * 64]["text"] == texts]
                self.assertEqual(len(matches), 1)
                by_text[(g, row)] = matches[0]
        self.assertEqual(sorted(by_text.values()), [0, 1, 2, 3])


def build(batch=2, **options):
    settings = dict(updater="dfa", batch_size=batch, plasticity=50.0, forget_rate=0.1,
                    ephemeral_fraction=0.5, enable_recurrence=False, fast_weight_clamp=0.05)
    settings.update(options)
    seed_everything(11, deterministic=True)
    with contextlib.redirect_stdout(io.StringIO()):
        return EphemeralRNN(len(CHARSET), 8, len(CHARSET), 2, CHARSET, **settings)


def config(**options):
    return {"updater": "dfa", "criterion": torch.nn.CrossEntropyLoss(reduction="none"),
            "input_mode": "last_one", "pe_matrix": None, "learning_rate": 0.3,
            "ephemeral_update_clamp": 0.02, "grad_norm_clip": 0, **options}


def onehot_of(texts):
    return F.one_hot(torch.tensor([[CHARSET.index(c) for c in t] for t in texts]), len(CHARSET)).float()


class StreamEvalTest(unittest.TestCase):
    def setUp(self):
        self.name = "kvswitch_s4_l16"
        rows = kv_switch.generate_split(self.name, "validation", 2, seed=4)
        self.streams = [rows[:16], rows[16:]]
        for s in self.streams:
            for row in s:
                row["tensor"] = [CHARSET.index(c) for c in row["text"]]

    def test_carry_matches_training_with_wipe_every(self):
        # train_batch with --wipe_every 8 and frozen slow weights writes and carries the fast
        # entries exactly as consecutive evaluate_held_out(wipe_fast=False) episodes do.
        trained = build()
        evaluated = copy.deepcopy(trained)
        frozen = {}
        wipe, forget = trained.start_sequence_wipe, trained.apply_forget_step

        def wipe_then_snapshot(*args, **kwargs):
            wipe(*args, **kwargs)
            for layer in trained.trained_layers():
                frozen[layer] = (layer.per_sample_weights.data.clone(), layer.bias.data.clone())

        def forget_then_restore():
            forget()
            for layer in trained.trained_layers():
                weights, bias = frozen[layer]
                layer.per_sample_weights.data.copy_(torch.where(layer.ephemeral_mask, layer.per_sample_weights.data, weights))
                layer.bias.data.copy_(bias)

        trained.start_sequence_wipe, trained.apply_forget_step = wipe_then_snapshot, forget_then_restore
        cfg = config(wipe_every=8)
        state = {"training_instance": 0, "log_norms_now": False}
        carried_changed = False
        for position in range(12):
            onehot = onehot_of([s[position]["text"] for s in self.streams])
            _, _, train_preds, train_losses, _, _ = train_module.train_batch(None, onehot, trained, cfg, state)
            preds, losses = heldout.evaluate_held_out(evaluated, onehot, torch.ones(2, 6, dtype=torch.bool), cfg,
                                                      wipe_fast=position % 8 == 0)
            torch.testing.assert_close(preds, train_preds, rtol=0, atol=0)
            torch.testing.assert_close(losses, train_losses)
            for mine, theirs in zip(evaluated.trained_layers(), trained.trained_layers()):
                torch.testing.assert_close(mine.per_sample_weights, theirs.per_sample_weights)
            if position == 1:
                wiped = copy.deepcopy(evaluated)
                heldout.evaluate_held_out(wiped, onehot, torch.ones(2, 6, dtype=torch.bool), cfg)
                carried_changed = not torch.allclose(wiped.linear_layers[0].per_sample_weights,
                                                     evaluated.linear_layers[0].per_sample_weights)
        self.assertTrue(carried_changed)  # not vacuous: carrying differs from wiping

    def test_protocols_and_counts(self):
        model = build()
        groups = [self.streams]
        cfg = config()
        calls = []
        wipe = model.start_sequence_wipe
        model.start_sequence_wipe = lambda wipe_fast=True: (calls.append(wipe_fast), wipe(wipe_fast))
        results = stream_eval.evaluate_checkpoint_streams(model, groups, cfg, self.name,
                                                          stream_eval.PROTOCOLS, 8, CHARSET)
        # carry wipes at positions 0 and 8 of the 16; wiped and no_fast wipe at every sequence.
        self.assertEqual(calls, [t % 8 == 0 for t in range(16)] + [True] * 32)
        wiped_like = stream_eval.evaluate_checkpoint_streams(model, groups, cfg, self.name, ["carry"], 1, CHARSET)
        self.assertEqual(wiped_like["carry"], results["wiped"])  # wipe_every 1 is the wiped protocol
        for summary in results.values():
            self.assertEqual(summary["by"]["segment"]["first"]["n"] + summary["by"]["segment"]["later"]["n"], 32)
            self.assertEqual(sum(b["n"] for b in summary["by"]["since_switch"].values()), 32)
            self.assertEqual(sum(b["n"] for b in summary["by"]["query_lag"].values()), 32)
            self.assertTrue(0 <= summary["acc"] <= 1)
        # The model's state is restored after evaluation.
        fresh = build()
        for mine, theirs in zip(model.trained_layers(), fresh.trained_layers()):
            torch.testing.assert_close(mine.per_sample_weights, theirs.per_sample_weights, rtol=0, atol=0)

    def test_counts_classify_with_metadata(self):
        counts = stream_eval.StreamCounts()
        row = {"text": "c7a2?a2", "stale": "5", "previous": "a5c6d8b9", "context": "a2c7d0b1",
               "since_switch": 5, "query_lag": 2, "segment": 1}
        for predicted in "2255":
            counts.add(row, kv_switch.classify_answer(row, predicted))
        summary = counts.summary()
        self.assertEqual(summary["acc"], 0.5)
        self.assertEqual(summary["stale_rate_later"], 0.5)
        self.assertEqual(summary["acc_carried"], 0.5)
        self.assertIsNone(summary["acc_in_sequence"])
        self.assertEqual(summary["by"]["since_switch_later"]["4-7"]["n"], 4)
        self.assertEqual(stream_eval.bucket_label(70, stream_eval.SINCE_BUCKETS), "64+")
        self.assertEqual(stream_eval.bucket_label(3, stream_eval.LAG_BUCKETS), "3+")


if __name__ == "__main__":
    unittest.main()


class UnregisteredNameKeyTest(unittest.TestCase):
    """kvswitch_s<S>_l4096 names are not registered but must resolve in both dataset_key tables."""

    def test_l4096_names_resolve(self):
        import preprocess
        import utils
        for s in (4, 64):
            name = f"kvswitch_s{s}_l4096"
            self.assertEqual(utils.dataset_keys[name], "text")
            self.assertEqual(preprocess.dataset_keys[name], "train")
            self.assertEqual(utils.get_charset(name), kv_tasks.CHARSET)
        with self.assertRaises(KeyError):
            utils.dataset_keys["not_a_dataset"]
