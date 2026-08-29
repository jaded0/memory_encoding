#!/usr/bin/env python3
import math
import unittest

from tests.characterization import UPDATERS, run_characterization


class UnifiedUpdatesSmokeTest(unittest.TestCase):
    def test_real_training_paths_produce_finite_updates(self):
        for updater in UPDATERS:
            with self.subTest(updater=updater):
                trace = run_characterization(updater)
                self.assertTrue(math.isfinite(trace["loss"]))
                self.assertEqual(len(trace["step_outputs"]), 4)

                output_values = trace["final_output"]["values"]
                self.assertTrue(all(math.isfinite(value) for value in output_values))

                candidate_values = trace["modules"]["i2o"]["candidate_weights"]["values"]
                self.assertTrue(all(math.isfinite(value) for value in candidate_values))
                self.assertGreater(sum(abs(value) for value in candidate_values), 0.0)


if __name__ == "__main__":
    unittest.main()
