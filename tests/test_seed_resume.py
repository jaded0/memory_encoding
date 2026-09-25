import contextlib
import io
import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch

import train as train_module
from preprocess import make_dataloader
from reproducibility import make_torch_generator, record_seed_in_slurm, resolve_seed
from utils import collate_fn

DATASET = "2_small_palindrome_dataset_vary_length"  # charset "23. " (4 symbols)


def in_memory_items():
    """Ten distinct padded palindromes in the preprocessed-row format, so data order is visible."""
    halves = [format(index, "03b") for index in range(8)] + ["01", "10"]
    items = []
    for half in halves:
        half = half.replace("0", "2").replace("1", "3")
        text = (half + "." + half[::-1]).ljust(7)
        indices = ["23. ".index(char) for char in text]
        onehot = torch.nn.functional.one_hot(torch.tensor(indices), 4).float().tolist()
        items.append({"text": text, "tensor": indices, "onehot_tensor": onehot})
    return items


def fake_loader(num_workers):
    def load(dataset, batch_size, drop_last=True, seed=None):
        return make_dataloader(in_memory_items(), batch_size, drop_last=drop_last, seed=seed, num_workers=num_workers)
    return load


def run_main(*extra_args, checkpoint_dir, num_workers=0, seen_batches=None):
    """Run train.main() on 10 in-memory rows (5 batches per epoch); returns its stdout."""
    argv = [
        "train.py", "--dataset", DATASET, "--track", "False", "--n_iters", "3", "--print_freq", "1",
        "--checkpoint_save_freq", "1", "--checkpoint_dir", checkpoint_dir, "--batch_size", "2",
        "--hidden_size", "4", "--num_layers", "1", "--unit_norm_weights", "False", "--input_mode", "last_one",
        *extra_args,
    ]
    real_train = train_module.train

    def recording_train(line_tensor, *args, **kwargs):
        if seen_batches is not None:
            seen_batches.append(line_tensor.clone())
        return real_train(line_tensor, *args, **kwargs)

    output = io.StringIO()
    with patch("sys.argv", argv), \
            patch.object(train_module, "load_and_preprocess_data", side_effect=fake_loader(num_workers)), \
            patch.object(train_module, "train", side_effect=recording_train), \
            contextlib.redirect_stdout(output):
        train_module.main()
    return output.getvalue()


def latest(directory):
    return torch.load(os.path.join(directory, "latest_checkpoint.pth"), weights_only=False)


def assert_tensors_equal(test, first, second):
    test.assertEqual(len(first), len(second))
    for first_value, second_value in zip(first, second):
        torch.testing.assert_close(first_value, second_value, rtol=0, atol=0)


class SeedResolutionTest(unittest.TestCase):
    def test_fresh_unseeded_runs_record_distinct_seeds(self):
        seeds = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as directory:
                output = run_main(checkpoint_dir=directory)
                seed = latest(directory)["config"]["seed"]
            self.assertIsNotNone(seed)
            self.assertIn(f"Seed: {seed} (generated)", output)
            seeds.append(seed)
        self.assertNotEqual(seeds[0], seeds[1])

    def test_resume_uses_checkpoint_seed_without_seed_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            run_main(checkpoint_dir=directory)
            original = latest(directory)["config"]["seed"]
            with patch.object(train_module, "seed_everything", wraps=train_module.seed_everything) as seeder:
                output = run_main("--resume", "--n_iters", "5", checkpoint_dir=directory)
            seeder.assert_called_once_with(original, deterministic=False)
            self.assertIn(f"Seed: {original} (from checkpoint)", output)
            self.assertEqual(latest(directory)["config"]["seed"], original)

    def test_resume_accepts_matching_seed_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            run_main("--seed", "11", checkpoint_dir=directory)
            run_main("--resume", "--seed", "11", "--n_iters", "4", checkpoint_dir=directory)
            with self.assertRaises(SystemExit) as raised, contextlib.redirect_stderr(io.StringIO()):
                run_main("--resume", "--seed", "12", "--n_iters", "5", checkpoint_dir=directory)
            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(latest(directory)["iter"], 5)  # the rejected resume trained nothing

    def test_legacy_unseeded_checkpoint_resumes_without_inventing_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            run_main(checkpoint_dir=directory)
            legacy = latest(directory)
            legacy["config"]["seed"] = None
            del legacy["data_stream_state"]
            torch.save(legacy, os.path.join(directory, "latest_checkpoint.pth"))
            with patch.object(train_module, "seed_everything", wraps=train_module.seed_everything) as seeder:
                output = run_main("--resume", "--n_iters", "5", checkpoint_dir=directory)
            seeder.assert_called_once_with(None, deterministic=False)
            self.assertIn("resumed legacy unseeded run", output)
            self.assertIsNone(latest(directory)["config"]["seed"])

    def test_deterministic_survives_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            run_main("--seed", "5", "--deterministic", "True", checkpoint_dir=directory)
            torch.use_deterministic_algorithms(False)
            run_main("--resume", "--n_iters", "5", checkpoint_dir=directory)
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertTrue(latest(directory)["config"]["deterministic"])
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                run_main("--resume", "--deterministic", "False", "--n_iters", "6", checkpoint_dir=directory)

    def test_resolve_seed_rejects_changes_on_resume(self):
        checkpoint = {"config": {"seed": 3, "deterministic": True}}
        self.assertEqual(resolve_seed(None, None, checkpoint), (3, True, "from checkpoint"))
        self.assertEqual(resolve_seed(3, True, checkpoint), (3, True, "from checkpoint"))
        with self.assertRaisesRegex(ValueError, "conflicts"):
            resolve_seed(4, None, checkpoint)
        with self.assertRaisesRegex(ValueError, "conflicts"):
            resolve_seed(None, False, checkpoint)
        with self.assertRaisesRegex(ValueError, "conflicts"):
            resolve_seed(3, None, {"config": {"seed": None}})  # no seeding a legacy run part-way


class ResumeContinuityTest(unittest.TestCase):
    def test_resume_continues_exactly_like_an_uninterrupted_run(self):
        # 7 + 5 steps over 5-batch epochs: the checkpoint lands mid-epoch and both runs cross epochs.
        flags = ("--seed", "1729", "--deterministic", "True")
        with tempfile.TemporaryDirectory() as straight_dir, tempfile.TemporaryDirectory() as split_dir:
            straight_batches, split_batches = [], []
            run_main(*flags, "--n_iters", "12", checkpoint_dir=straight_dir,
                     num_workers=2, seen_batches=straight_batches)
            run_main(*flags, "--n_iters", "7", checkpoint_dir=split_dir,
                     num_workers=2, seen_batches=split_batches)
            run_main("--resume", "--n_iters", "12", checkpoint_dir=split_dir,
                     num_workers=2, seen_batches=split_batches)
            straight, split = latest(straight_dir), latest(split_dir)

        self.assertEqual(len(straight_batches), 12)
        assert_tensors_equal(self, straight_batches, split_batches)  # data order continues, not replays
        self.assertEqual(straight["model_state_dict"].keys(), split["model_state_dict"].keys())
        assert_tensors_equal(self, list(straight["model_state_dict"].values()),
                             list(split["model_state_dict"].values()))
        torch.testing.assert_close(straight["torch_rng_state"], split["torch_rng_state"], rtol=0, atol=0)
        self.assertEqual(straight["data_stream_state"]["batches_into_epoch"],
                         split["data_stream_state"]["batches_into_epoch"])
        self.assertEqual(straight["main_program_state"], split["main_program_state"])

    def test_resumable_sampler_matches_plain_shuffle_order(self):
        # Explicitly seeded runs must see the same batches as before the resumable sampler existed.
        def epochs(loader):
            return [batch[1] for _ in range(3) for batch in loader]

        plain = torch.utils.data.DataLoader(
            in_memory_items(), batch_size=3, shuffle=True, drop_last=True,
            collate_fn=collate_fn, generator=make_torch_generator(42),
        )
        resumable = make_dataloader(in_memory_items(), 3, seed=42, num_workers=0)
        assert_tensors_equal(self, epochs(plain), epochs(resumable))


class SlurmSeedCommentTest(unittest.TestCase):
    def test_sets_job_comment_when_under_slurm(self):
        with patch.dict(os.environ, {"SLURM_JOB_ID": "123"}), patch("subprocess.run") as run:
            record_seed_in_slurm(7)
        self.assertEqual(run.call_args.args[0], ["scontrol", "update", "JobId=123", "Comment=seed=7"])

    def test_failures_are_ignored(self):
        for error in (FileNotFoundError("scontrol"), subprocess.TimeoutExpired("scontrol", 5), PermissionError()):
            with patch.dict(os.environ, {"SLURM_JOB_ID": "123"}), patch("subprocess.run", side_effect=error):
                record_seed_in_slurm(7)

    def test_no_op_outside_slurm(self):
        with patch.dict(os.environ, {}, clear=True), patch("subprocess.run") as run:
            record_seed_in_slurm(7)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
