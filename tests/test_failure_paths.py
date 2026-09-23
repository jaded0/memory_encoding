import contextlib
import io
import os
import signal
import tempfile
import unittest
from unittest.mock import patch

import torch

import train as train_module
from ephemeral_model import EphemeralRNN
from utils import CHECKPOINT_CODE_VERSION, load_checkpoint, save_checkpoint

DATASET = "2_small_palindrome_dataset_vary_length"  # charset "23. " (4 symbols)
CONFIG = {"n_hidden": 4, "n_layers": 1, "updater": "dfa", "charset_size": 4, "model_type": "ephemeral"}


def tiny_batches():
    """One in-memory batch in the (texts, index tensor, one-hot tensor) collate format."""
    indices = torch.tensor([[0, 1, 2, 1, 0], [1, 0, 2, 0, 1]])
    return [(["23.32", "32.23"], indices, torch.nn.functional.one_hot(indices, 4).float())]


def build_model():
    with contextlib.redirect_stdout(io.StringIO()):
        return EphemeralRNN(8, 4, 4, 1, "23. ", unit_norm_weights=False, weight_clamp=0, batch_size=2)


def run_main(*extra_args, checkpoint_dir):
    argv = [
        "train.py", "--dataset", DATASET, "--track", "False", "--n_iters", "3", "--print_freq", "1",
        "--checkpoint_save_freq", "0", "--checkpoint_dir", checkpoint_dir, "--batch_size", "2",
        "--hidden_size", "4", "--num_layers", "1", "--unit_norm_weights", "False", "--input_mode", "last_one",
        *extra_args,
    ]
    with patch("sys.argv", argv), \
            patch.object(train_module, "load_and_preprocess_data", return_value=tiny_batches()), \
            contextlib.redirect_stdout(io.StringIO()):
        train_module.main()


class CheckpointCompatibilityTest(unittest.TestCase):
    def save(self, directory, config):
        with contextlib.redirect_stdout(io.StringIO()):
            save_checkpoint({"config": config, "model_state_dict": build_model().state_dict()}, directory, "c.pth")
        return os.path.join(directory, "c.pth")

    def load(self, path, config):
        with contextlib.redirect_stdout(io.StringIO()):
            return load_checkpoint(path, build_model(), config)

    def test_updater_mismatch_raises_configuration_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.save(directory, {**CONFIG, "updater": "backprop"})
            with self.assertRaisesRegex(RuntimeError, "configuration mismatch"):
                self.load(path, CONFIG)

    def test_model_type_mismatch_raises_configuration_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.save(directory, {**CONFIG, "model_type": "rnn"})
            with self.assertRaisesRegex(RuntimeError, "configuration mismatch"):
                self.load(path, CONFIG)

    def test_forget_rate_change_raises_configuration_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.save(directory, {**CONFIG, "forget_rate": 0.01})
            with self.assertRaisesRegex(RuntimeError, "configuration mismatch"):
                self.load(path, {**CONFIG, "forget_rate": 0.3})
            self.load(path, {**CONFIG, "forget_rate": 0.01})

    def test_dataset_or_learning_rate_change_raises_configuration_error(self):
        saved = {**CONFIG, "dataset": DATASET, "learning_rate": 1e-4}
        with tempfile.TemporaryDirectory() as directory:
            path = self.save(directory, saved)
            for change in ({"dataset": "palindrome_dataset"}, {"learning_rate": 2e-4}):
                with self.subTest(change=change), self.assertRaisesRegex(RuntimeError, "configuration mismatch"):
                    self.load(path, {**saved, **change})

    def test_other_config_differences_print_but_load(self):
        # The checkpoint uses the old key plast_clip; load_checkpoint maps it to plasticity.
        saved = {**CONFIG, "print_freq": 50, "n_iters": 100, "plast_clip": 10.0}
        with tempfile.TemporaryDirectory() as directory:
            path = self.save(directory, saved)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                load_checkpoint(path, build_model(), {**CONFIG, "print_freq": 5, "n_iters": 100, "plasticity": 20.0, "notes": "x"})
        printed = output.getvalue()
        self.assertIn("print_freq: 50 -> 5", printed)
        self.assertIn("plasticity: 10.0 -> 20.0", printed)
        self.assertNotIn("plast_clip", printed)
        self.assertIn("notes: (not in checkpoint) -> 'x'", printed)
        self.assertNotIn("n_iters", printed)

    def test_other_or_missing_code_version_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.save(directory, CONFIG)
            self.assertEqual(torch.load(path, weights_only=False)["code_version"], CHECKPOINT_CODE_VERSION)
            self.load(path, CONFIG)
            for version in (CHECKPOINT_CODE_VERSION - 1, CHECKPOINT_CODE_VERSION + 1, None):
                with self.subTest(code_version=version):
                    checkpoint = torch.load(path, weights_only=False)
                    if version is None:
                        del checkpoint["code_version"]
                        expected = "has no code_version"
                    else:
                        checkpoint["code_version"] = version
                        expected = f"has code_version {version}"
                    torch.save(checkpoint, path)
                    with self.assertRaisesRegex(RuntimeError, f"{expected}.*start the run fresh"):
                        self.load(path, CONFIG)

    def test_legacy_ethereal_checkpoint_loads_as_ephemeral(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.save(directory, {**CONFIG, "model_type": "ethereal"})
            self.load(path, CONFIG)


class MainFailurePathTest(unittest.TestCase):
    def test_training_exception_propagates(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(train_module, "train", side_effect=RuntimeError("boom")):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                run_main(checkpoint_dir=directory)

    def test_infinite_loss_exits_nonzero(self):
        infinite_step = (None, float("inf"), 0, 0, [], [])
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(train_module, "train", return_value=infinite_step):
            with self.assertRaises(SystemExit) as raised:
                run_main(checkpoint_dir=directory)
        self.assertEqual(raised.exception.code, 1)

    def test_missing_explicit_checkpoint_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = os.path.join(directory, "missing.pth")
            with self.assertRaises(FileNotFoundError):
                run_main("--resume_checkpoint", missing, checkpoint_dir=directory)

    def test_unreadable_checkpoint_raises_instead_of_restarting(self):
        with tempfile.TemporaryDirectory() as directory:
            corrupt = os.path.join(directory, "latest_checkpoint.pth")
            with open(corrupt, "wb") as handle:
                handle.write(b"not a checkpoint")
            with self.assertRaises(Exception):
                run_main("--resume", checkpoint_dir=directory)

    def test_explicit_checkpoint_resumes_without_extra_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            run_main("--checkpoint_save_freq", "3", checkpoint_dir=directory)
            latest = os.path.join(directory, "latest_checkpoint.pth")
            with patch.object(train_module, "load_checkpoint", wraps=train_module.load_checkpoint) as loader:
                run_main("--resume_checkpoint", latest, "--n_iters", "5", checkpoint_dir=directory)
            loader.assert_called_once()

    def test_resume_with_new_learning_rate_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            run_main("--checkpoint_save_freq", "3", checkpoint_dir=directory)
            with self.assertRaisesRegex(RuntimeError, "configuration mismatch"):
                run_main("--resume", "--learning_rate", "0.5", checkpoint_dir=directory)

    def test_resume_of_another_code_version_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            run_main("--checkpoint_save_freq", "3", checkpoint_dir=directory)
            latest = os.path.join(directory, "latest_checkpoint.pth")
            checkpoint = torch.load(latest, weights_only=False)
            self.assertEqual(checkpoint["code_version"], CHECKPOINT_CODE_VERSION)
            checkpoint["code_version"] = CHECKPOINT_CODE_VERSION - 1
            torch.save(checkpoint, latest)
            with patch.object(train_module, "train") as train:
                with self.assertRaisesRegex(RuntimeError, "start the run fresh"):
                    run_main("--resume", "--n_iters", "5", checkpoint_dir=directory)
            train.assert_not_called()

    def run_with_signal(self, signum, directory, *extra_args):
        real_train, calls = train_module.train, []

        def train_then_signal(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                os.kill(os.getpid(), signum)  # arrives mid-run, like SLURM's warning
            return real_train(*args, **kwargs)

        with patch.object(train_module, "train", side_effect=train_then_signal):
            with self.assertRaises(SystemExit) as raised:
                run_main("--n_iters", "10", *extra_args, checkpoint_dir=directory)
        return raised.exception.code, len(calls)

    def test_time_limit_signal_checkpoints_and_exits_124(self):
        with tempfile.TemporaryDirectory() as directory:
            code, calls = self.run_with_signal(signal.SIGUSR1, directory, "--checkpoint_save_freq", "1000")
            self.assertEqual(code, 124)
            self.assertEqual(calls, 2)  # stops at the next iteration boundary
            checkpoint = torch.load(os.path.join(directory, "latest_checkpoint.pth"), weights_only=False)
            self.assertEqual(checkpoint["iter"], 3)  # resumes at the first iteration not yet run
        self.assertIs(signal.getsignal(signal.SIGUSR1), signal.SIG_DFL)  # handlers restored

    def test_sigterm_exits_143(self):
        with tempfile.TemporaryDirectory() as directory:
            code, _ = self.run_with_signal(signal.SIGTERM, directory)
        self.assertEqual(code, 143)
        self.assertIs(signal.getsignal(signal.SIGTERM), signal.SIG_DFL)

    def test_clean_run_completes(self):
        with tempfile.TemporaryDirectory() as directory:
            run_main(checkpoint_dir=directory)


if __name__ == "__main__":
    unittest.main()
