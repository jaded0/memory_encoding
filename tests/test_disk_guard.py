"""--min_free_disk_gb: a full disk pauses training instead of killing it mid-save (K13, 2026-10-31)."""
import collections
import os
import tempfile
import unittest
from unittest.mock import patch

import torch

import utils
from tests.test_seed_resume import latest, run_main

Usage = collections.namedtuple("Usage", "total used free")


def disk(*free_values):
    """shutil.disk_usage stand-in returning these free byte counts in turn, then the last one."""
    values = list(free_values)
    def usage(path):
        return Usage(0, 0, values.pop(0) if len(values) > 1 else values[0])
    return usage


class WaitForFreeSpaceTest(unittest.TestCase):
    def test_off_by_default_and_when_space_is_enough(self):
        sleeps = []
        with patch("utils.shutil.disk_usage", side_effect=disk(10)):
            self.assertEqual(utils.wait_for_free_space(".", 0, sleep=sleeps.append), 0)
            self.assertEqual(utils.wait_for_free_space(".", 10, sleep=sleeps.append), 0)
        self.assertEqual(sleeps, [])

    def test_pauses_until_space_returns_and_notifies_once(self):
        sleeps = []
        with tempfile.TemporaryDirectory() as directory:
            marker = os.path.join(directory, "pings")
            with patch("utils.shutil.disk_usage", side_effect=disk(1, 2, 3, 50)), \
                    patch.dict(os.environ, {"DISK_WAIT_NOTIFY": f"echo ping >> {marker}"}):
                waited = utils.wait_for_free_space(directory, 50, poll_seconds=7, sleep=sleeps.append)
            pings = open(marker).read().split()
        self.assertEqual((waited, sleeps, pings), (21, [7, 7, 7], ["ping"]))


class SaveRetryTest(unittest.TestCase):
    def test_failed_write_is_retried_and_keeps_the_previous_checkpoint(self):
        sleeps = []
        real_save = torch.save
        failures = [RuntimeError("PytorchStreamWriter failed writing file data/0: file write failed")]

        def flaky_save(obj, path):
            if failures:
                with open(path, "wb") as handle:
                    handle.write(b"partial")
                raise failures.pop()
            real_save(obj, path)

        with tempfile.TemporaryDirectory() as directory:
            utils.save_checkpoint({"iter": 1}, directory, "latest_checkpoint.pth")
            with patch("utils.torch.save", side_effect=flaky_save):
                utils.save_checkpoint_retrying({"iter": 2}, directory, "latest_checkpoint.pth",
                                               sleep=sleeps.append, poll_seconds=5)
            saved = torch.load(os.path.join(directory, "latest_checkpoint.pth"), weights_only=False)
            leftovers = [name for name in os.listdir(directory) if ".tmp." in name]
        self.assertEqual((saved["iter"], sleeps, leftovers), (2, [5], []))

    def test_gives_up_after_the_last_attempt(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch("utils.torch.save", side_effect=OSError(28, "No space left on device")):
            with self.assertRaises(OSError):
                utils.save_checkpoint_retrying({}, directory, "x.pth", attempts=3, sleep=lambda s: None)


class TrainingPausesTest(unittest.TestCase):
    def test_run_waits_for_disk_then_finishes_like_an_unguarded_run(self):
        flags = ("--seed", "1729", "--deterministic", "True", "--n_iters", "4")
        with tempfile.TemporaryDirectory() as plain_dir, tempfile.TemporaryDirectory() as guarded_dir:
            run_main(*flags, checkpoint_dir=plain_dir)
            sleeps = []
            with patch("utils.shutil.disk_usage", side_effect=disk(10**12, 1, 1, 10**12)), \
                    patch("utils.time.sleep", side_effect=sleeps.append):
                output = run_main(*flags, "--min_free_disk_gb", "1", checkpoint_dir=guarded_dir)
            plain, guarded = latest(plain_dir), latest(guarded_dir)
        self.assertIn("pausing until space is freed", output)
        self.assertEqual(sleeps, [300, 300])
        for key, value in plain["model_state_dict"].items():
            torch.testing.assert_close(value, guarded["model_state_dict"][key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
