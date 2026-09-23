import json
import math
import unittest
from pathlib import Path

import torch

from tests.characterization import all_trace_keys, run_characterization


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "training_traces.json"


class TrainingCharacterizationTest(unittest.TestCase):
    def assert_nested_close(self, expected, actual, path="trace"):
        if isinstance(expected, dict):
            self.assertIsInstance(actual, dict, path)
            self.assertEqual(set(expected), set(actual), path)
            for key in expected:
                self.assert_nested_close(expected[key], actual[key], f"{path}.{key}")
            return

        if isinstance(expected, list):
            self.assertIsInstance(actual, list, path)
            self.assertEqual(len(expected), len(actual), path)
            for index, (expected_value, actual_value) in enumerate(zip(expected, actual)):
                self.assert_nested_close(
                    expected_value, actual_value, f"{path}[{index}]"
                )
            return

        if isinstance(expected, float):
            self.assertTrue(
                math.isclose(expected, actual, rel_tol=1e-6, abs_tol=1e-7),
                f"{path}: expected {expected!r}, got {actual!r}",
            )
            return

        self.assertEqual(expected, actual, path)

    def test_real_training_paths_match_golden_traces(self):
        fixture = json.loads(FIXTURE_PATH.read_text())
        self.assertEqual(fixture["schema_version"], 1)
        self.assertEqual(
            fixture["generated_with"]["torch"],
            torch.__version__,
            "golden traces must be regenerated after changing Torch versions",
        )

        self.assertEqual(
            set(fixture["traces"]), {key for key, _updater, _case in all_trace_keys()}
        )
        for key, updater, case in all_trace_keys():
            with self.subTest(trace=key):
                actual = run_characterization(updater, case=case)
                self.assert_nested_close(fixture["traces"][key], actual, key)

    def test_seed_changes_model_mask(self):
        first = run_characterization("dfa", seed=1729)
        second = run_characterization("dfa", seed=1730)
        self.assertNotEqual(
            first["modules"]["linear_layers.0"]["ephemeral_mask"]["values"],
            second["modules"]["linear_layers.0"]["ephemeral_mask"]["values"],
        )


if __name__ == "__main__":
    unittest.main()
