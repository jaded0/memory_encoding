import os
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from reproducibility import (
    capture_rng_state,
    make_torch_generator,
    restore_rng_state,
    seed_data_worker,
    seed_everything,
)


class RandomValueDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, index):
        return index, random.random(), np.random.random(), torch.rand(()).item()


class ReproducibilityTest(unittest.TestCase):
    def test_unseeded_configuration_does_not_reset_rng(self):
        seed_everything(99)
        expected_first = torch.rand(4)
        expected_second = torch.rand(4)

        seed_everything(99)
        actual_first = torch.rand(4)
        seed_everything()
        actual_second = torch.rand(4)

        torch.testing.assert_close(expected_first, actual_first, rtol=0, atol=0)
        torch.testing.assert_close(expected_second, actual_second, rtol=0, atol=0)

    def test_seed_repeats_supported_rngs(self):
        seed_everything(1234, deterministic=True)
        expected = (
            random.random(),
            np.random.random(4),
            torch.rand(4),
        )

        seed_everything(1234, deterministic=True)
        actual = (
            random.random(),
            np.random.random(4),
            torch.rand(4),
        )

        self.assertEqual(expected[0], actual[0])
        np.testing.assert_array_equal(expected[1], actual[1])
        torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)

    def test_capture_and_restore_rng_state(self):
        seed_everything(5678)
        state = capture_rng_state()
        expected = (
            random.random(),
            np.random.random(4),
            torch.rand(4),
        )

        random.random()
        np.random.random(4)
        torch.rand(4)
        restore_rng_state(state)
        actual = (
            random.random(),
            np.random.random(4),
            torch.rand(4),
        )

        self.assertEqual(expected[0], actual[0])
        np.testing.assert_array_equal(expected[1], actual[1])
        torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)

    def test_seeded_generators_repeat_data_order(self):
        first = torch.randperm(20, generator=make_torch_generator(42))
        second = torch.randperm(20, generator=make_torch_generator(42))
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_data_workers_repeat_order_and_rng_streams(self):
        def collect_batches():
            loader = torch.utils.data.DataLoader(
                RandomValueDataset(),
                batch_size=3,
                shuffle=True,
                num_workers=2,
                generator=make_torch_generator(42),
                worker_init_fn=seed_data_worker,
            )
            return [tuple(value.clone() for value in batch) for batch in loader]

        first = collect_batches()
        second = collect_batches()
        self.assertEqual(len(first), len(second))
        for first_batch, second_batch in zip(first, second):
            for first_values, second_values in zip(first_batch, second_batch):
                torch.testing.assert_close(first_values, second_values, rtol=0, atol=0)

    def test_deterministic_mode_requires_seed(self):
        with self.assertRaisesRegex(ValueError, "requires --seed"):
            seed_everything(deterministic=True)

    def test_deterministic_mode_rejects_invalid_cublas_configuration(self):
        with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": "invalid"}):
            with self.assertRaisesRegex(ValueError, "CUBLAS_WORKSPACE_CONFIG"):
                seed_everything(42, deterministic=True)

    def test_deterministic_mode_enables_strict_torch_operations(self):
        torch.use_deterministic_algorithms(False)
        seed_everything(42, deterministic=True)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertTrue(torch.backends.cudnn.deterministic)
        self.assertFalse(torch.backends.cudnn.benchmark)
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(torch.backends.cudnn.allow_tf32)
        self.assertIn(
            os.environ["CUBLAS_WORKSPACE_CONFIG"], (":4096:8", ":16:8")
        )


if __name__ == "__main__":
    unittest.main()
