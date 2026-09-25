"""Checkpoints saved before the naming cleanup (old state-dict names).

They predate CHECKPOINT_CODE_VERSION, so a resume refuses them. The old-name mapping in
utils (upgrade_legacy_state_dict, upgrade_legacy_config) is still checked here at the unit level.

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
from ephemeral_model import EphemeralRNN, SimpleRNN
from tests.test_seed_resume import DATASET, fake_loader
from utils import CHECKPOINT_CODE_VERSION, load_checkpoint, upgrade_legacy_config, upgrade_legacy_state_dict

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
    """Runs train.main() with argv on the in-memory rows of test_seed_resume; returns stdout.

    The fixtures are CPU runs, compared bit for bit, so this always runs on the CPU with one
    thread. Without that, train.main() moves the model to cuda:0 when a GPU is visible: the
    saved tensors are then on another device, and the GPU arithmetic differs by up to ~1e-7.
    """
    output = io.StringIO()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with patch("sys.argv", ["train.py", *argv, "--checkpoint_dir", checkpoint_dir]), \
            patch("torch.cuda.is_available", return_value=False), \
            patch.object(train_module, "load_and_preprocess_data", side_effect=fake_loader(0)), \
            contextlib.redirect_stdout(output):
        try:
            train_module.main()
        finally:
            torch.set_num_threads(threads)
    return output.getvalue()


def resume_legacy(case, directory, names="old", extra=()):
    """Resumes <case>/iter3.pth with today's code as the old code made iter5.pth, passing the
    flags under their old names (like a frozen run_used.sh) or today's. Returns (checkpoint, stdout)
    if the resume runs; today's version check refuses it, so the tests expect it to raise."""
    shutil.copy(LEGACY_DIR / case / "iter3.pth", os.path.join(directory, "latest_checkpoint.pth"))
    flags = [*OLD_BASE_ARGV, *COMMON, *CASES[case]] if names == "old" else [*NEW_BASE_ARGV, *NEW_COMMON, *NEW_CASES[case]]
    stdout = run_train_main([*flags, "--resume", "--n_iters", "5", "--checkpoint_save_freq", "5", *extra], directory)
    return torch.load(os.path.join(directory, "latest_checkpoint.pth"), weights_only=False), stdout


class LegacyCheckpointResumeTest(unittest.TestCase):
    # These checkpoints predate CHECKPOINT_CODE_VERSION, so a resume refuses them (the run must
    # start fresh) instead of continuing them. Until code_version existed, this test resumed them
    # and matched the old code's own continuation; the name mapping that made that possible is
    # now checked at the unit level below (LegacyStateDictLoadTest).
    def test_resuming_an_old_checkpoint_is_refused_by_the_version_check(self):
        for case in CASES:
            for names in ("old", "new"):
                with self.subTest(case=case, flag_names=names), tempfile.TemporaryDirectory() as directory:
                    with self.assertRaisesRegex(RuntimeError, "has no code_version.*start the run fresh"):
                        resume_legacy(case, directory, names)
                    # Refused before training: the checkpoint is left exactly as it was.
                    resumed = torch.load(os.path.join(directory, "latest_checkpoint.pth"), weights_only=False)
                    self.assertEqual(resumed["iter"], legacy(case, "iter3.pth")["iter"])
                    self.assertNotIn("code_version", resumed)


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
        # The version check refuses this checkpoint (see LegacyCheckpointResumeTest); stamp it so
        # the tests below reach the state-dict mapping behind it.
        checkpoint["code_version"] = CHECKPOINT_CODE_VERSION
        return checkpoint

    def test_shim_maps_old_names_like_the_independent_renamer(self):
        # Unit level, without loading into a model, so it does not depend on today's layer
        # shapes: the old fixtures' state dicts and configs map to today's names.
        for case in CASES:
            for name in ("iter3.pth", "iter5.pth"):
                with self.subTest(case=case, checkpoint=name):
                    old = legacy(case, name)
                    with contextlib.redirect_stdout(io.StringIO()):
                        model = SimpleRNN(4, 4, 4, 1) if case == "rnn_backprop" else build_model()
                    # The fixtures predate the removal of the self_grad head; today's model has none.
                    state = {k: v for k, v in old["model_state_dict"].items() if not k.startswith("self_grad.")}
                    upgraded, dropped = upgrade_legacy_state_dict(state, model)
                    expected, expected_dropped = rename_old_state_dict(state)
                    self.assertEqual(list(upgraded), list(expected))
                    for key, value in expected.items():
                        torch.testing.assert_close(upgraded[key], value, rtol=0, atol=0, msg=key)
                    self.assertEqual(dropped, expected_dropped)
                    self.assertEqual(len(dropped), 0 if case == "rnn_backprop" else 3)  # one per EphemeralLinear
                    if case != "rnn_backprop":
                        # The ephemeral weights were trained, so the check above is not comparing zeros.
                        self.assertGreater(upgraded["linear_layers.0.per_sample_weights"].abs().sum().item(), 0)

                    config = upgrade_legacy_config(old["config"])
                    for old_key in ("plast_clip", "plast_proportion", "grad_clip", "clip_weights", "normalize",
                                    "plast_learning_rate", "imprint_rate"):
                        self.assertNotIn(old_key, config)
                    self.assertEqual((config["plasticity"], config["ephemeral_fraction"], config["weight_clamp"]),
                                     (3.0, 0.5, 0.5))
                    self.assertEqual(config["unit_norm_weights"], case == "dfa")
                    clamp = ("grad_norm_clip", "ephemeral_update_clamp")[case != "rnn_backprop"]
                    other = ("grad_norm_clip", "ephemeral_update_clamp")[case == "rnn_backprop"]
                    self.assertEqual((config[clamp], config[other]), (0.2, 0))

    def test_new_checkpoint_round_trips(self):
        model = build_model()
        self.load({"config": self.CONFIG, "model_state_dict": model.state_dict(),
                   "code_version": CHECKPOINT_CODE_VERSION})

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
        # On a current-format state dict, so the check does not depend on old layer shapes.
        for change in ("missing", "unexpected"):
            with self.subTest(change=change):
                checkpoint = {"config": dict(self.CONFIG), "model_state_dict": build_model().state_dict(),
                              "code_version": CHECKPOINT_CODE_VERSION}
                state = dict(checkpoint["model_state_dict"])
                if change == "missing":
                    del state["i2h.plasticity"]
                else:
                    state["i2h.mask_tier_two"] = state["i2h.ephemeral_mask"]
                checkpoint["model_state_dict"] = state
                with self.assertRaisesRegex(RuntimeError, f"{change} keys \\['i2h"):
                    self.load(checkpoint)


if __name__ == "__main__":
    unittest.main()
