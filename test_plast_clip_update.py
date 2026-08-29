#!/usr/bin/env python3
import os
import random
import tempfile
import unittest

import numpy as np
import torch

from hebbian_model import EtherealRNN
from reproducibility import capture_rng_state, seed_everything
from utils import initialize_charset, load_checkpoint, save_checkpoint


class CheckpointBehaviorTest(unittest.TestCase):
    def test_plast_clip_update_and_rng_round_trip(self):
        seed_everything(31415, deterministic=True)
        charset, _char_to_idx, _idx_to_char, n_characters = initialize_charset(
            "palindrome_dataset"
        )
        hidden_size = 16
        num_layers = 1
        batch_size = 2
        initial_plast_clip = 10.0
        new_plast_clip = 50.0

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "test_checkpoint.pth")
            original = EtherealRNN(
                n_characters,
                hidden_size,
                n_characters,
                num_layers,
                charset,
                updater="dfa",
                plast_clip=initial_plast_clip,
                batch_size=batch_size,
                plast_proportion=0.5,
            )

            config = {
                "plast_clip": initial_plast_clip,
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

            restored = EtherealRNN(
                n_characters,
                hidden_size,
                n_characters,
                num_layers,
                charset,
                updater="dfa",
                plast_clip=new_plast_clip,
                batch_size=batch_size,
                plast_proportion=0.5,
            )
            new_config = {**config, "plast_clip": new_plast_clip}
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

            if loaded_config["plast_clip"] != new_config["plast_clip"]:
                restored.update_plasticity_clip(new_config["plast_clip"])

            high_plasticity_values = []
            low_plasticity_values = []
            for layer in restored.linear_layers:
                high_plasticity_values.extend(layer.plasticity[layer.mask].tolist())
                low_plasticity_values.extend(layer.plasticity[~layer.mask].tolist())

            self.assertTrue(high_plasticity_values)
            self.assertTrue(low_plasticity_values)
            self.assertTrue(
                all(value == new_plast_clip for value in high_plasticity_values)
            )
            self.assertTrue(all(value == 1.0 for value in low_plasticity_values))


if __name__ == "__main__":
    unittest.main()
