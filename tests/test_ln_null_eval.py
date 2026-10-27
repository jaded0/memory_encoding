"""S12 follow-up 2 (scratch/s12_2026-10-27/ln_null_eval.py): the valid no_fast baseline for --fast_read_norm ln.

- With the donor set to the episode itself (row for row, under the same protocol), swap_raw and swap
  reproduce heldout.py's strict exactly: the patched read is the layer's own ln read, and the donor
  runs the same episode in lockstep.
- noise has exactly the own read's norm at every read (zero where the own read is zero), is
  reproducible by seed and changes the result; swap with a real donor (another batch) changes it too
  and keeps the own read's norm.
- The model's state and methods are restored, the donor of a single batch is its rows rolled by one,
  a shorter donor is padded, and a model without ln reads is refused."""
import contextlib
import copy
import importlib.util
import io
import json
import os
import tempfile
import unittest

import torch
import torch.nn.functional as F

import heldout
import train as train_module
from ephemeral_model import EphemeralLinear, EphemeralRNN
from reproducibility import seed_everything
from tests.test_seed_resume import DATASET

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = importlib.util.spec_from_file_location(
    "ln_null_eval", os.path.join(HERE, "scratch", "s12_2026-10-27", "ln_null_eval.py"))
ln_null_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ln_null_eval)

CHARSET = "23. "  # DATASET's charset
BATCH = 2
TEXTS = [["23.32  ", "322.223"], ["2.2    ", "33.33  "]]


def build(**options):
    settings = dict(updater="dfa", batch_size=BATCH, plasticity=50.0, forget_rate=0.2, ephemeral_fraction=0.5,
                    enable_recurrence=False, weight_clamp=0.6, fast_read_norm="ln")
    settings.update(options)
    seed_everything(7, deterministic=True)
    with contextlib.redirect_stdout(io.StringIO()):
        model = EphemeralRNN(4, 16, 4, 3, CHARSET, **settings)
    with torch.no_grad():  # a trained-looking model: nonzero slow entries everywhere
        for layer in model.trained_layers():
            layer.per_sample_weights.add_(0.3 * torch.randn(layer.per_sample_weights.shape[1:]))
    return model


def config():
    return {"updater": "dfa", "input_mode": "last_one", "pe_matrix": None, "learning_rate": 0.3,
            "ephemeral_update_clamp": 0, "grad_norm_clip": 0}


def batches(n=2):
    return [(texts, F.one_hot(torch.tensor([[CHARSET.index(c) for c in t] for t in texts]), 4).float())
            for texts in TEXTS[:n]]


class LnNullEvalTest(unittest.TestCase):
    def test_self_donor_reproduces_strict(self):
        model = build()
        data = batches()
        strict = heldout.evaluate_protocols(model, data, config(), DATASET, ["strict"])
        for kind in ("swap_raw", "swap"):
            summary, stats = ln_null_eval.evaluate_null(model, data, config(), DATASET, kind, self_donor=True)
            for key, value in summary.items():
                self.assertAlmostEqual(value, strict[f"heldout_strict/{key}"], places=5, msg=(kind, key))
            for name in ("L0", "L1", "L2"):
                self.assertGreater(stats[f"{name}_own_nonzero_reads"], 0)
                self.assertAlmostEqual(stats[f"{name}_null_norm_mean"], stats[f"{name}_own_norm_mean"], places=4)
                self.assertEqual(stats[f"{name}_null_nonzero_where_own_zero"], 0)

    def test_noise_is_norm_matched_reproducible_and_changes_the_result(self):
        model = build()
        data = batches()
        strict = heldout.evaluate_protocols(model, data, config(), DATASET, ["strict"])
        a, stats = ln_null_eval.evaluate_null(model, data, config(), DATASET, "noise", seed=3)
        b, _ = ln_null_eval.evaluate_null(model, data, config(), DATASET, "noise", seed=3)
        c, _ = ln_null_eval.evaluate_null(model, data, config(), DATASET, "noise", seed=4)
        self.assertEqual(a, b)
        self.assertNotEqual(a["loss"], c["loss"])
        self.assertNotAlmostEqual(a["loss"], strict["heldout_strict/loss"], places=4)
        for name in ("L0", "L1", "L2"):
            self.assertAlmostEqual(stats[f"{name}_null_norm_mean"], stats[f"{name}_own_norm_mean"], places=4)
            self.assertEqual(stats[f"{name}_null_nonzero_where_own_zero"], 0)
            self.assertEqual(stats[f"{name}_null_zero_where_own_nonzero"], 0)

    def test_noise_read_is_centred_and_scaled_to_the_own_read(self):
        model = build()
        layer = model.linear_layers[1]
        with torch.no_grad():
            layer.per_sample_weights.add_(torch.randn(layer.per_sample_weights.shape) * layer.ephemeral_mask)
        read = ln_null_eval.NullRead(layer, "noise", generator=torch.Generator().manual_seed(0))
        x = torch.randn(BATCH, layer.in_features)
        out = read(x, x.unsqueeze(2))
        w, m = layer.per_sample_weights, layer.ephemeral_mask
        slow = torch.einsum('boi,bi->bo', w * ~m, x)
        own = F.layer_norm(torch.einsum('boi,bi->bo', w * m, x), (layer.out_features,), eps=1e-5)
        noise = out - slow
        torch.testing.assert_close(noise.mean(1), torch.zeros(BATCH), atol=1e-5, rtol=0)
        torch.testing.assert_close(noise.norm(dim=1), own.norm(dim=1), atol=1e-4, rtol=1e-5)
        self.assertLess(float(F.cosine_similarity(noise, own, dim=1).abs().max()), 0.9)

    def test_swap_with_a_real_donor_keeps_the_norm_and_changes_the_result(self):
        model = build()
        data = batches()
        strict = heldout.evaluate_protocols(model, data, config(), DATASET, ["strict"])
        summary, stats = ln_null_eval.evaluate_null(model, data, config(), DATASET, "swap")
        self.assertNotAlmostEqual(summary["loss"], strict["heldout_strict/loss"], places=4)
        for name in ("L0", "L1", "L2"):
            self.assertAlmostEqual(stats[f"{name}_null_norm_mean"], stats[f"{name}_own_norm_mean"], places=4)
            self.assertEqual(stats[f"{name}_null_nonzero_where_own_zero"], 0)
            self.assertEqual(stats[f"{name}_null_zero_where_own_nonzero"], 0)
        raw, _ = ln_null_eval.evaluate_null(model, data, config(), DATASET, "swap_raw")
        self.assertTrue(all(torch.isfinite(torch.tensor(list(raw.values())))))

    def test_swap_zero_donor_l0_uses_seeded_centred_norm_matched_fallback(self):
        model = build()
        model.start_sequence_wipe()
        layer = model.linear_layers[0]
        donor = copy.deepcopy(layer)  # no donor writes: every fast read is zero
        with torch.no_grad():
            layer.per_sample_weights[0].add_(torch.randn_like(layer.per_sample_weights[0]) * layer.ephemeral_mask)
        x = torch.ones(BATCH, layer.in_features)
        w, m = layer.per_sample_weights, layer.ephemeral_mask
        slow = torch.einsum('boi,bi->bo', w * ~m, x)
        own = F.layer_norm(torch.einsum('boi,bi->bo', w * m, x), (layer.out_features,), eps=1e-5)

        def run(seed):
            read = ln_null_eval.NullRead(layer, "swap", donor, torch.Generator().manual_seed(seed))
            return read(x, x.unsqueeze(2)) - slow, read.stats

        a, stats = run(3)
        b, _ = run(3)
        c, _ = run(4)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertFalse(torch.equal(a[0], c[0]))
        self.assertGreater(float(own[0].norm()), 0)
        torch.testing.assert_close(a.mean(1), torch.zeros(BATCH), atol=1e-6, rtol=0)
        torch.testing.assert_close(a.norm(dim=1), own.norm(dim=1), atol=1e-5, rtol=1e-5)
        self.assertEqual(float(a[1].abs().sum()), 0)  # own-zero rows stay zero
        self.assertEqual(float(stats[4]), 0)  # no missing read remains
        self.assertEqual(float(stats[5]), 1)  # exactly one fallback

    def test_null_scoring_retains_norm_and_fallback_diagnostics(self):
        spec = importlib.util.spec_from_file_location(
            "s12_score", os.path.join(HERE, "scratch", "s12_2026-10-27", "s12_score.py"))
        scorer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scorer)
        diagnostics = {"null_swap/L0_swap_noise_fallback_reads": 7,
                       "null_swap/L0_null_zero_where_own_nonzero": 0,
                       "null_swap/L0_own_norm_mean": 3.0}
        with tempfile.TemporaryDirectory() as root:
            with open(os.path.join(root, "null_test_30k.json"), "w") as f:
                json.dump(dict(iteration=30000, episodes=2, **diagnostics), f)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                scorer.score_null([root])
            row = json.loads(output.getvalue())
        for key, value in diagnostics.items():
            self.assertEqual(row[key], value)

    def test_state_and_methods_are_restored(self):
        model = build()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        for kind in ln_null_eval.NULLS:
            ln_null_eval.evaluate_null(model, batches(), config(), DATASET, kind)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)
        for layer in model.linear_layers:
            self.assertNotIn("_normalized_read", layer.__dict__)
            self.assertEqual(type(layer)._normalized_read, EphemeralLinear._normalized_read)

    def test_donor_choice_and_padding(self):
        data = batches()
        self.assertIs(ln_null_eval.donor_batch(data, 0), data[1])
        self.assertIs(ln_null_eval.donor_batch(data, 1), data[0])
        texts, onehot = ln_null_eval.donor_batch(data[:1], 0)
        self.assertEqual(texts, TEXTS[0][1:] + TEXTS[0][:1])
        self.assertTrue(torch.equal(onehot[0], data[0][1][1]))
        short = data[0][1][:, :4]
        mask = torch.ones(BATCH, 3, dtype=torch.bool)
        padded, pmask = ln_null_eval.fit_steps(short, mask, 6)
        self.assertEqual(tuple(padded.shape), (BATCH, 7, 4))
        self.assertEqual(float(padded[:, 4:].abs().sum()), 0.0)
        self.assertEqual(pmask[:, 3:].any().item(), False)
        cut, cmask = ln_null_eval.fit_steps(data[0][1], torch.ones(BATCH, 6, dtype=torch.bool), 3)
        self.assertEqual((cut.shape[1], cmask.shape[1]), (4, 3))
        # A single batch evaluates against its rolled rows.
        ln_null_eval.evaluate_null(build(), data[:1], config(), DATASET, "swap")

    def test_layer_subset_and_refusals(self):
        model = build()
        self.assertEqual([n for n, _ in ln_null_eval.ln_layers(model, [2])], ["L2"])
        _, stats = ln_null_eval.evaluate_null(model, batches(), config(), DATASET, "noise", layers=[2])
        self.assertEqual(sorted({k.split("_")[0] for k in stats}), ["L2"])
        with self.assertRaises(ValueError):
            ln_null_eval.evaluate_null(build(fast_read_norm="off"), batches(), config(), DATASET, "noise")
        with self.assertRaises(ValueError):
            ln_null_eval.evaluate_null(model, batches(), config(), DATASET, "bogus")


if __name__ == "__main__":
    unittest.main()
