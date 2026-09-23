"""Checkpoints saved before the naming cleanup (old state-dict names) still load and resume.

The fixtures in tests/fixtures/legacy_names were written by the code at commit 1775121; see
make_legacy_checkpoints.py there.
"""
import contextlib
import io
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from ephemeral_model import EphemeralRNN
from tests.test_seed_resume import run_main
from utils import load_checkpoint

LEGACY_DIR = Path(__file__).parent / "fixtures" / "legacy_names"
# The flags the fixtures were made with (old names, as in a frozen run_used.sh).
COMMON = ["--seed", "11", "--deterministic", "True", "--learning_rate", "0.05",
          "--plast_clip", "3", "--plast_proportion", "0.5", "--grad_clip", "0.2",
          "--clip_weights", "0.5", "--forget_rate", "0.25",
          "--plast_learning_rate", "0.005", "--imprint_rate", "0"]
CASES = {
    "dfa": ["--updater", "dfa", "--normalize", "True"],
    "backprop": ["--updater", "backprop"],
    "rnn_backprop": ["--model_type", "rnn", "--updater", "backprop"],
}
FORGET_RATE = 0.25
# Written out here rather than imported from utils, so the test does not trust the shim.
OLD_TO_NEW = {
    "candidate_weights": "per_sample_weights",
    "mask": "ephemeral_mask",
    "last_high_plast_update_norm": "last_ephemeral_step_norm",
    "last_low_plast_update_norm": "last_slow_step_norm",
}


def legacy(case, name):
    return torch.load(LEGACY_DIR / case / name, map_location="cpu", weights_only=False)


def rename_old_state_dict(state_dict):
    """Old names -> new names; checks and drops forgetting_factor. Returns (dict, dropped keys)."""
    renamed, dropped = {}, []
    for key, value in state_dict.items():
        prefix, _, name = key.rpartition(".")
        if name == "forgetting_factor":
            mask = state_dict[f"{prefix}.mask"]
            torch.testing.assert_close(value, FORGET_RATE * mask, rtol=0, atol=0)
            dropped.append(key)
        else:
            renamed[f"{prefix}.{OLD_TO_NEW.get(name, name)}"] = value
    return renamed, dropped


def resume_legacy(case, directory):
    """Resumes <case>/iter3.pth with today's code exactly as the old code made iter5.pth."""
    shutil.copy(LEGACY_DIR / case / "iter3.pth", os.path.join(directory, "latest_checkpoint.pth"))
    run_main(*COMMON, *CASES[case], "--resume", "--n_iters", "5", "--checkpoint_save_freq", "5",
             checkpoint_dir=directory)
    return torch.load(os.path.join(directory, "latest_checkpoint.pth"), weights_only=False)


class LegacyCheckpointResumeTest(unittest.TestCase):
    def test_resuming_an_old_checkpoint_matches_the_old_code(self):
        for case in CASES:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                new = resume_legacy(case, directory)
                old = legacy(case, "iter5.pth")

                expected, dropped = rename_old_state_dict(old["model_state_dict"])
                self.assertEqual(list(new["model_state_dict"]), list(expected))
                for key, value in expected.items():
                    torch.testing.assert_close(new["model_state_dict"][key], value, rtol=0, atol=0, msg=key)
                if case == "rnn_backprop":
                    self.assertEqual(dropped, [])
                else:
                    self.assertEqual(len(dropped), 4)  # one per EphemeralLinear layer

                old_optimizer, new_optimizer = old["optimizer_state_dict"], new["optimizer_state_dict"]
                if old_optimizer is None:
                    self.assertIsNone(new_optimizer)
                else:
                    self.assertEqual(new_optimizer["state"], old_optimizer["state"])
                    for old_group, new_group in zip(old_optimizer["param_groups"], new_optimizer["param_groups"]):
                        self.assertEqual({k: v for k, v in old_group.items() if k != "params"},
                                         {k: v for k, v in new_group.items() if k != "params"})
                        self.assertEqual(len(new_group["params"]), len(old_group["params"]) - len(dropped))

                for key in ("iter", "main_program_state", "python_rng_state"):
                    self.assertEqual(new[key], old[key], key)
                torch.testing.assert_close(new["torch_rng_state"], old["torch_rng_state"], rtol=0, atol=0)
                self.assertEqual(str(new["numpy_rng_state"]), str(old["numpy_rng_state"]))
                self.assertEqual(set(new["data_stream_state"]), set(old["data_stream_state"]))
                for key, value in old["data_stream_state"].items():
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(new["data_stream_state"][key], value, rtol=0, atol=0)
                    else:
                        self.assertEqual(new["data_stream_state"][key], value, key)


def build_model(updater="dfa"):
    with contextlib.redirect_stdout(io.StringIO()):
        return EphemeralRNN(4, 4, 4, 1, "23. ", normalize=True, clip_weights=0.5, updater=updater,
                            plast_clip=3.0, batch_size=2, forget_rate=FORGET_RATE, plast_proportion=0.5,
                            enable_recurrence=False)


class LegacyStateDictLoadTest(unittest.TestCase):
    CONFIG = {"n_hidden": 4, "n_layers": 1, "updater": "dfa", "charset_size": 4,
              "model_type": "ephemeral", "forget_rate": FORGET_RATE}

    def load(self, checkpoint, model=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return load_checkpoint("<memory>", model or build_model(), self.CONFIG, checkpoint=checkpoint)

    def old_checkpoint(self):
        checkpoint = legacy("dfa", "iter3.pth")
        checkpoint["config"] = dict(self.CONFIG)  # the resume checks are tested elsewhere
        return checkpoint

    def test_old_names_load_into_the_new_tensors(self):
        checkpoint = self.old_checkpoint()
        old = dict(checkpoint["model_state_dict"])
        model = self.load(checkpoint)[0]
        for prefix in ("linear_layers.0", "i2h", "i2o", "self_grad"):
            layer = model.get_submodule(prefix)
            for old_name, new_name in OLD_TO_NEW.items():
                torch.testing.assert_close(getattr(layer, new_name).data, old[f"{prefix}.{old_name}"], rtol=0, atol=0)
            self.assertFalse(hasattr(layer, "forgetting_factor"))
        # The ephemeral weights were trained, so the check above is not comparing zeros.
        self.assertGreater(model.linear_layers[0].per_sample_weights.abs().sum().item(), 0)

    def test_new_checkpoint_round_trips(self):
        model = build_model()
        self.load({"config": self.CONFIG, "model_state_dict": model.state_dict()})

    def test_forgetting_factor_other_than_forget_rate_on_the_mask_raises(self):
        for change in ("scaled", "off_mask"):
            with self.subTest(change=change):
                checkpoint = self.old_checkpoint()
                state = dict(checkpoint["model_state_dict"])
                if change == "scaled":  # as --normalize did before 2026-09
                    state["linear_layers.0.forgetting_factor"] = state["linear_layers.0.forgetting_factor"] * 0.5
                else:  # as mask_tier_two did with plast_proportion < 0.01
                    factor = state["i2h.forgetting_factor"].clone()
                    factor[~state["i2h.mask"]] = FORGET_RATE
                    state["i2h.forgetting_factor"] = factor
                checkpoint["model_state_dict"] = state
                with self.assertRaisesRegex(RuntimeError, "forgetting_factor is not forget_rate"):
                    self.load(checkpoint)

    def test_forgetting_factor_with_a_different_forget_rate_raises(self):
        checkpoint = self.old_checkpoint()
        del checkpoint["config"]["forget_rate"]  # an old run that did not record it
        with contextlib.redirect_stdout(io.StringIO()), \
                self.assertRaisesRegex(RuntimeError, "forgetting_factor is not forget_rate"):
            model = EphemeralRNN(4, 4, 4, 1, "23. ", batch_size=2, forget_rate=0.1, plast_proportion=0.5)
            load_checkpoint("<memory>", model, {k: v for k, v in self.CONFIG.items() if k != "forget_rate"},
                            checkpoint=checkpoint)

    def test_missing_or_unexpected_keys_raise(self):
        for change in ("missing", "unexpected"):
            with self.subTest(change=change):
                checkpoint = self.old_checkpoint()
                state = dict(checkpoint["model_state_dict"])
                if change == "missing":
                    del state["i2h.plasticity"]
                else:
                    state["i2h.mask_tier_two"] = state["i2h.mask"]
                checkpoint["model_state_dict"] = state
                with self.assertRaisesRegex(RuntimeError, f"{change} keys \\['i2h"):
                    self.load(checkpoint)


if __name__ == "__main__":
    unittest.main()
