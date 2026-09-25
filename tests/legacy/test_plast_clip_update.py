#!/usr/bin/env python3
import os
import random
import tempfile
import unittest

import numpy as np
import torch

from ephemeral_model import EphemeralRNN
from reproducibility import capture_rng_state, seed_everything
from utils import initialize_charset, load_checkpoint, save_checkpoint


class CheckpointBehaviorTest(unittest.TestCase):
    def test_plasticity_update_and_rng_round_trip(self):
        seed_everything(31415, deterministic=True)
        charset, _char_to_idx, _idx_to_char, n_characters = initialize_charset(
            "palindrome_dataset"
        )
        hidden_size = 16
        num_layers = 1
        batch_size = 2
        initial_plasticity = 10.0
        new_plasticity = 50.0

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "test_checkpoint.pth")
            original = EphemeralRNN(
                n_characters,
                hidden_size,
                n_characters,
                num_layers,
                charset,
                updater="dfa",
                plasticity=initial_plasticity,
                batch_size=batch_size,
                ephemeral_fraction=0.5,
            )

            config = {
                "plasticity": initial_plasticity,
                "n_hidden": hidden_size,
                "n_layers": num_layers,
                "charset_size": n_characters,
                "updater": "dfa",
            }
            checkpoint_state = {
                "iter": 1000,
                "model_state_dict": original.state_dict(),
                "optimizer_state_dict": None,
                "main_program_state": {"training_instance": 500},
                "config": config,
                **capture_rng_state(),
            }
            save_checkpoint(checkpoint_state, temp_dir, "test_checkpoint.pth")

            expected_python = random.random()
            expected_numpy = np.random.random(4)
            expected_torch = torch.rand(4)

            restored = EphemeralRNN(
                n_characters,
                hidden_size,
                n_characters,
                num_layers,
                charset,
                updater="dfa",
                plasticity=new_plasticity,
                batch_size=batch_size,
                ephemeral_fraction=0.5,
            )
            new_config = {**config, "plasticity": new_plasticity}
            restored, _, start_iter, loaded_state, loaded_config = load_checkpoint(
                checkpoint_path, restored, new_config, device="cpu"
            )

            self.assertEqual(start_iter, 1000)
            self.assertEqual(loaded_state["training_instance"], 500)
            self.assertEqual(random.random(), expected_python)
            np.testing.assert_array_equal(np.random.random(4), expected_numpy)
            torch.testing.assert_close(torch.rand(4), expected_torch, rtol=0, atol=0)

            for mismatch in ({"seed": 1}, {"deterministic": True}):
                with self.subTest(mismatch=mismatch), self.assertRaisesRegex(
                    RuntimeError, "configuration mismatch"
                ):
                    load_checkpoint(
                        checkpoint_path,
                        restored,
                        {**new_config, **mismatch},
                        device="cpu",
                    )

            if loaded_config["plasticity"] != new_config["plasticity"]:
                restored.set_plasticity(new_config["plasticity"])

            ephemeral_plasticity_values = []
            slow_plasticity_values = []
            for layer in restored.linear_layers:
                ephemeral_plasticity_values.extend(layer.plasticity[layer.ephemeral_mask].tolist())
                slow_plasticity_values.extend(layer.plasticity[~layer.ephemeral_mask].tolist())

            self.assertTrue(ephemeral_plasticity_values)
            self.assertTrue(slow_plasticity_values)
            self.assertTrue(
                all(value == new_plasticity for value in ephemeral_plasticity_values)
            )
            self.assertTrue(all(value == 1.0 for value in slow_plasticity_values))


if __name__ == "__main__":
    unittest.main()
