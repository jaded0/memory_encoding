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

from unittest.mock import patch

import train as train_module
from ephemeral_model import EphemeralRNN
from tests.test_seed_resume import DATASET, fake_loader
from utils import load_checkpoint, upgrade_legacy_config

LEGACY_DIR = Path(__file__).parent / "fixtures" / "legacy_names"
# The argv the fixtures were made with (old names, as in a frozen run_used.sh): the base argv of
# tests/test_seed_resume.run_main at 1775121, then COMMON, then the case's flags.
OLD_BASE_ARGV = ["--dataset", DATASET, "--track", "False", "--n_iters", "3", "--print_freq", "1",
                 "--checkpoint_save_freq", "1", "--batch_size", "2", "--hidden_size", "4",
                 "--num_layers", "1", "--normalize", "False", "--input_mode", "last_one"]
COMMON = ["--seed", "11", "--deterministic", "True", "--learning_rate", "0.05",
          "--plast_clip", "3", "--plast_proportion", "0.5", "--grad_clip", "0.2",
          "--clip_weights", "0.5", "--forget_rate", "0.25",
          "--plast_learning_rate", "0.005", "--imprint_rate", "0"]
CASES = {
    "dfa": ["--updater", "dfa", "--normalize", "True"],
    "backprop": ["--updater", "backprop"],
    "rnn_backprop": ["--model_type", "rnn", "--updater", "backprop"],
}
# The same settings under today's names.
NEW_BASE_ARGV = [flag.replace("--normalize", "--unit_norm_weights") for flag in OLD_BASE_ARGV]
NEW_COMMON = ["--seed", "11", "--deterministic", "True", "--learning_rate", "0.05",
              "--plasticity", "3", "--ephemeral_fraction", "0.5",
              "--weight_clamp", "0.5", "--forget_rate", "0.25"]
NEW_CASES = {
    "dfa": ["--updater", "dfa", "--unit_norm_weights", "True", "--ephemeral_update_clamp", "0.2"],
    "backprop": ["--updater", "backprop", "--ephemeral_update_clamp", "0.2"],
    "rnn_backprop": ["--model_type", "rnn", "--updater", "backprop", "--grad_norm_clip", "0.2"],
}
# Config entries that legitimately differ between the fixture's run and a rerun here.
UNCOMPARED_CONFIG_KEYS = {"checkpoint_dir", "criterion"}
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


def run_train_main(argv, checkpoint_dir):
    """Runs train.main() with argv on the in-memory rows of test_seed_resume; returns stdout."""
    output = io.StringIO()
    with patch("sys.argv", ["train.py", *argv, "--checkpoint_dir", checkpoint_dir]), \
            patch.object(train_module, "load_and_preprocess_data", side_effect=fake_loader(0)), \
            contextlib.redirect_stdout(output):
        train_module.main()
    return output.getvalue()


def resume_legacy(case, directory, names="old", extra=()):
    """Resumes <case>/iter3.pth with today's code as the old code made iter5.pth, passing the
    flags under their old names (like a frozen run_used.sh) or today's. Returns (checkpoint, stdout)."""
    shutil.copy(LEGACY_DIR / case / "iter3.pth", os.path.join(directory, "latest_checkpoint.pth"))
    flags = [*OLD_BASE_ARGV, *COMMON, *CASES[case]] if names == "old" else [*NEW_BASE_ARGV, *NEW_COMMON, *NEW_CASES[case]]
    stdout = run_train_main([*flags, "--resume", "--n_iters", "5", "--checkpoint_save_freq", "5", *extra], directory)
    return torch.load(os.path.join(directory, "latest_checkpoint.pth"), weights_only=False), stdout


class LegacyCheckpointResumeTest(unittest.TestCase):
    def test_resuming_an_old_checkpoint_matches_the_old_code(self):
        for case in CASES:
            for names in ("old", "new"):
                with self.subTest(case=case, flag_names=names), tempfile.TemporaryDirectory() as directory:
                    self.check_matches_old_continuation(case, *resume_legacy(case, directory, names))

    def check_matches_old_continuation(self, case, new, stdout):
        old = legacy(case, "iter5.pth")

        # Same settings under today's keys; the old names never reach the config.
        upgraded = upgrade_legacy_config(old["config"])
        self.assertEqual({k: v for k, v in new["config"].items() if k not in UNCOMPARED_CONFIG_KEYS},
                         {k: v for k, v in upgraded.items() if k not in UNCOMPARED_CONFIG_KEYS})
        # The resume diff lists only settings that really changed, none of the renamed ones.
        diff_keys = set()
        for line in stdout.split("Config differences (checkpoint -> this run):\n")[1].splitlines():
            if not (line.startswith("  ") and " -> " in line):
                break
            diff_keys.add(line.split(":")[0].strip())
        self.assertEqual(diff_keys, {"checkpoint_dir", "checkpoint_save_freq", "n_iters", "resume"})
        self.assertNotIn("Plasticity changed", stdout)

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
        return EphemeralRNN(4, 4, 4, 1, "23. ", unit_norm_weights=True, weight_clamp=0.5, updater=updater,
                            plasticity=3.0, batch_size=2, forget_rate=FORGET_RATE, ephemeral_fraction=0.5,
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
            model = EphemeralRNN(4, 4, 4, 1, "23. ", batch_size=2, forget_rate=0.1, ephemeral_fraction=0.5)
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
