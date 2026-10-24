"""B2 round 2 (scratch/b2_tinystories_2026-10-22/b2_eval.py): the WIPE R protocol of the horizon mode."""
import contextlib
import importlib.util
import io
import os
import unittest

import torch
import torch.nn.functional as F

from ephemeral_model import EphemeralRNN
from reproducibility import seed_everything

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = importlib.util.spec_from_file_location(
    "b2_eval", os.path.join(HERE, "scratch", "b2_tinystories_2026-10-22", "b2_eval.py"))
b2_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(b2_eval)

CHARSET = "23. "
TEXTS = ["23.32  3.22 3.2 2", "322.223 .3 2..3 3"]  # 17 characters, 16 predicted steps


def build(**options):
    # The B2 setting: recurrence off, last_one input, no positional encoding, so nothing but fast entries
    # carries context; a nonzero fast clamp and forget so the wipe has state to clear.
    settings = dict(updater="dfa", batch_size=len(TEXTS), plasticity=50.0, forget_rate=0.05,
                    ephemeral_fraction=0.5, enable_recurrence=False, weight_clamp=0.6, fast_weight_clamp=0.5)
    settings.update(options)
    seed_everything(7, deterministic=True)
    with contextlib.redirect_stdout(io.StringIO()):
        model = EphemeralRNN(4, 6, 4, 3, CHARSET, **settings)
    model.eval()
    return model


CONFIG = {"input_mode": "last_one", "pe_matrix": None, "learning_rate": 0.3, "ephemeral_update_clamp": 0.05,
          "grad_norm_clip": 0}


def onehot(texts):
    return F.one_hot(torch.tensor([[CHARSET.index(c) for c in text] for text in texts]), 4).float()


class WipeProtocolTest(unittest.TestCase):
    def setUp(self):
        self.model = build()
        self.hot = onehot(TEXTS)
        self.on = b2_eval.run_pass(self.model, self.hot, CONFIG, write=True)
        self.off = b2_eval.run_pass(self.model, self.hot, CONFIG, write=False)

    def test_fast_weights_matter(self):
        self.assertFalse(torch.equal(self.on, self.off))

    def test_wipe_1_is_off(self):
        self.assertTrue(torch.equal(b2_eval.run_pass(self.model, self.hot, CONFIG, write=True, wipe_every=1), self.off))

    def test_wipe_at_or_past_length_is_on(self):
        for r in (16, 17, 100):
            self.assertTrue(torch.equal(b2_eval.run_pass(self.model, self.hot, CONFIG, write=True, wipe_every=r),
                                        self.on))

    def test_each_block_equals_a_fresh_pass(self):
        r = 5
        wiped = b2_eval.run_pass(self.model, self.hot, CONFIG, write=True, wipe_every=r)
        self.assertFalse(torch.equal(wiped, self.on))
        steps = self.hot.shape[1] - 1
        for lo in range(0, steps, r):
            hi = min(lo + r, steps)
            fresh = b2_eval.run_pass(self.model, self.hot[:, lo:hi + 1], CONFIG, write=True)
            torch.testing.assert_close(wiped[:, lo:hi], fresh, rtol=0, atol=1e-6)

    def test_slow_entries_untouched(self):
        before = [layer.per_sample_weights.detach().clone() for layer in self.model.linear_layers]
        masks = [layer.ephemeral_mask for layer in self.model.linear_layers]
        b2_eval.run_pass(self.model, self.hot, CONFIG, write=True, wipe_every=3)
        for layer, old, mask in zip(self.model.linear_layers, before, masks):
            self.assertTrue(torch.equal(layer.per_sample_weights[:, ~mask], old[:, ~mask]))


if __name__ == "__main__":
    unittest.main()
