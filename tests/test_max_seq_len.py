"""--max_seq_len: crop every training sequence to its first N characters, in the collate."""
import contextlib
import io
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
from datasets import Dataset

import preprocess
import train as train_module
from preprocess import OneHotCollate, make_dataloader
from utils import load_checkpoint, save_checkpoint
import tests.test_layer_norm as layer_norm_tests

DATASET = "2_small_palindrome_dataset_vary_length"  # charset "23. " (4 symbols)


def rows():
    """Ten rows of different lengths in the stored {'text', 'tensor'} format."""
    out = []
    for index in range(10):
        text = ("23." * 4)[: 3 + index % 7]
        out.append({"text": text, "tensor": ["23. ".index(char) for char in text]})
    return out


def fake_loader(seen_kwargs):
    def load(dataset, batch_size, drop_last=True, seed=None, **kwargs):
        seen_kwargs.append(kwargs)
        collate = OneHotCollate(4, kwargs.get("max_seq_len", 0))
        return make_dataloader(rows(), batch_size, drop_last=drop_last, seed=seed, num_workers=0,
                               collate_fn=collate)
    return load


def run_main(*extra_args, checkpoint_dir):
    """train.main() on the in-memory rows; returns (loader kwargs, the batches train() saw, stdout)."""
    argv = ["train.py", "--dataset", DATASET, "--track", "False", "--n_iters", "4", "--print_freq", "2",
            "--checkpoint_save_freq", "2", "--checkpoint_dir", checkpoint_dir, "--batch_size", "2",
            "--hidden_size", "4", "--num_layers", "1", "--seed", "3", *extra_args]
    seen_kwargs, seen_batches = [], []
    real_train = train_module.train

    def recording_train(line_tensor, onehot, *args, **kwargs):
        seen_batches.append((line_tensor.clone(), onehot.clone()))
        return real_train(line_tensor, onehot, *args, **kwargs)

    output = io.StringIO()
    with patch("sys.argv", argv), \
            patch.object(train_module, "load_and_preprocess_data", side_effect=fake_loader(seen_kwargs)), \
            patch.object(train_module, "train", side_effect=recording_train), \
            contextlib.redirect_stdout(output):
        train_module.main()
    return seen_kwargs, seen_batches, output.getvalue()


class CollateCropTest(unittest.TestCase):
    def test_zero_keeps_whole_rows_and_matches_the_old_collate(self):
        batch = rows()[:4]
        old = OneHotCollate(4)(batch)
        new = OneHotCollate(4, 0)(batch)
        self.assertEqual(old[0], new[0])
        self.assertTrue(torch.equal(old[1], new[1]) and torch.equal(old[2], new[2]))

    def test_crop_keeps_the_first_characters_of_text_and_tensor(self):
        batch = rows()[:6]
        whole = OneHotCollate(4)(batch)
        texts, indices, onehot = OneHotCollate(4, 5)(batch)
        self.assertEqual(texts, [row["text"][:5] for row in batch])
        self.assertEqual(indices.shape[1], 5)
        self.assertTrue(torch.equal(indices, whole[1][:, :5]))
        self.assertTrue(torch.equal(onehot, whole[2][:, :5]))
        # A row shorter than the crop is padded with all-zero one-hot rows, as before.
        self.assertEqual(onehot[0, 3:].abs().sum().item(), 0)

    def test_negative_is_refused(self):
        with self.assertRaises(ValueError):
            OneHotCollate(4, -1)

    def test_load_and_preprocess_data_passes_the_crop_to_its_collate(self):
        with patch.object(preprocess, "is_synthetic", return_value=False), \
                patch.object(preprocess, "load_processed_dataset",
                             return_value=Dataset.from_list(rows())), \
                contextlib.redirect_stdout(io.StringIO()):
            whole = preprocess.load_and_preprocess_data(DATASET, batch_size=2, seed=1)
            cropped = preprocess.load_and_preprocess_data(DATASET, batch_size=2, seed=1, max_seq_len=4)
        whole.num_workers = cropped.num_workers = 0
        batches = 0
        for (whole_texts, whole_idx, _), (texts, idx, _) in zip(whole, cropped):
            # Same rows in the same order: the crop acts after the sampler.
            self.assertEqual([text[:4] for text in whole_texts], texts)
            self.assertTrue(torch.equal(whole_idx[:, :4], idx))
            batches += 1
        self.assertEqual(batches, 5)


class TrainFlagTest(unittest.TestCase):
    def test_default_calls_the_loader_as_before(self):
        with tempfile.TemporaryDirectory() as directory:
            kwargs, _, _ = run_main(checkpoint_dir=directory)
        self.assertEqual(kwargs, [{}])

    def test_crop_reaches_training_with_the_same_data_order(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            _, whole, _ = run_main(checkpoint_dir=first)
            kwargs, cropped, _ = run_main("--max_seq_len", "4", checkpoint_dir=second)
        self.assertEqual(kwargs, [{"max_seq_len": 4}])
        self.assertEqual(len(whole), len(cropped))
        for (whole_idx, whole_hot), (idx, hot) in zip(whole, cropped):
            self.assertLessEqual(idx.shape[1], 4)
            self.assertTrue(torch.equal(whole_idx[:, :idx.shape[1]], idx))
            self.assertTrue(torch.equal(whole_hot[:, :hot.shape[1]], hot))

    def test_cli_values(self):
        self.assertEqual(train_module.parse_args([]).max_seq_len, 0)
        self.assertEqual(train_module.parse_args(["--max_seq_len", "512"]).max_seq_len, 512)
        for bad in ("-1", "1"):
            with self.subTest(bad=bad), self.assertRaises(SystemExit), \
                    contextlib.redirect_stderr(io.StringIO()):
                train_module.parse_args(["--max_seq_len", bad])

    def test_resume_refuses_a_changed_crop_and_a_checkpoint_without_it_is_whole(self):
        config = layer_norm_tests.LayerNormResumeTest.CONFIG
        build = layer_norm_tests.build
        with tempfile.TemporaryDirectory() as directory:
            model = build()
            for saved, current, refused in (({}, {}, False), ({}, {"max_seq_len": 0}, False),
                                            ({}, {"max_seq_len": 512}, True),
                                            ({"max_seq_len": 512}, {"max_seq_len": 512}, False),
                                            ({"max_seq_len": 512}, {"max_seq_len": 256}, True)):
                with self.subTest(saved=saved, current=current), contextlib.redirect_stdout(io.StringIO()):
                    save_checkpoint({"config": {**config, **saved}, "model_state_dict": model.state_dict()},
                                    directory, "c.pth")
                    load = lambda: load_checkpoint(os.path.join(directory, "c.pth"), build(),
                                                   {**config, **current})
                    if refused:
                        with self.assertRaisesRegex(RuntimeError, "configuration mismatch"):
                            load()
                    else:
                        load()


if __name__ == "__main__":
    unittest.main()
