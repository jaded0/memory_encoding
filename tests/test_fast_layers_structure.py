"""--fast_layers composed with --fast_structure (C6 follow-up 2, 2026-10-27: fast weights in trunk
layer 0 only, the deep trunk layers slow-only, for exclusive, additive_masked and additive_dense).

No code change was needed for this: EphemeralRNN gives a trunk layer outside --fast_layers
ephemeral_fraction 0, and EphemeralLinear makes any layer with fraction 0 'exclusive' (all-slow)
whatever --fast_structure says. These tests pin that down for all three structures on a 3-layer
trunk, through the paths the follow-up uses: the unfused and the fused (eager) step, the CLI with a
checkpoint and a resume, the loop tracer, and load_eval.py's read-time ablation."""
import contextlib
import importlib.util
import io
import os
import tempfile
import unittest

import torch

import train as train_module
from ephemeral_model import EphemeralRNN
from loop_trace import LoopTracer
from reproducibility import seed_everything
from tests.test_fast_structure import assert_same, hconfig, train_once
from tests.test_heldout import CHARSET, TEXTS, episodes
from tests.test_seed_resume import latest, run_main

STRUCTURES = ('exclusive', 'additive_masked', 'additive_dense')
LOAD_EVAL = os.path.join(os.path.dirname(__file__), '..', 'scratch', 'fast_structure_2026-10-22', 'load_eval.py')


def build_rnn(structure, fast_layers=(0,), **options):
    seed_everything(7, deterministic=True)
    settings = dict(updater="dfa", batch_size=2, plasticity=50.0, forget_rate=0.2, ephemeral_fraction=0.5,
                    enable_recurrence=False, fast_structure=structure,
                    fast_layers=None if fast_layers is None else list(fast_layers))
    settings.update(options)
    with contextlib.redirect_stdout(io.StringIO()):
        return EphemeralRNN(10, 5, 4, 3, CHARSET, **settings)


class LayerZeroOnlyTest(unittest.TestCase):
    def test_layer_zero_and_i2h_follow_the_flag_and_the_deep_trunk_is_all_slow(self):
        for structure in STRUCTURES:
            with self.subTest(structure):
                model = build_rnn(structure)
                self.assertEqual([l.fast_structure for l in model.trained_layers()],
                                 [structure, 'exclusive', 'exclusive', structure, 'exclusive'])
                for index in (1, 2):
                    deep = model.linear_layers[index]
                    self.assertFalse(deep.ephemeral_mask.any())
                    self.assertNotIn('fast_state', dict(deep.named_parameters()))
                self.assertTrue(model.linear_layers[0].fast_mask.any())
                self.assertTrue(model.i2h.fast_mask.any())

    def test_training_writes_fast_weights_only_in_layer_zero_and_i2h(self):
        for structure in STRUCTURES:
            with self.subTest(structure):
                model = build_rnn(structure)
                train_once(model, batches=2)
                self.assertGreater(model.linear_layers[0].fast_weights().abs().sum().item(), 0)
                self.assertGreater(model.i2h.fast_weights().abs().sum().item(), 0)
                for index in (1, 2):
                    self.assertEqual(model.linear_layers[index].fast_weights().abs().sum().item(), 0.0)

    def test_deep_layers_match_a_slow_layer_of_an_all_slow_trunk(self):
        """A slow-only deep layer is the same kind of layer in every structure: built from the same
        seed, its slow weights start identical to those of an exclusive model with the same
        --fast_layers (the additive layers draw the same ephemeral mask, so the RNG stream agrees)."""
        reference = build_rnn('exclusive')
        for structure in ('additive_masked', 'additive_dense'):
            with self.subTest(structure):
                model = build_rnn(structure)
                for index in (1, 2):
                    assert_same(self, model.linear_layers[index].per_sample_weights.data,
                                reference.linear_layers[index].per_sample_weights.data)

    def test_fused_eager_step_equals_unfused_step(self):
        for structure in STRUCTURES:
            with self.subTest(structure):
                unfused, fused = build_rnn(structure), build_rnn(structure)
                fused.enable_fused_update(compile=False)
                for model in (unfused, fused):
                    train_once(model, hconfig(), batches=2)
                for key, value in unfused.state_dict().items():
                    assert_same(self, fused.state_dict()[key], value)

    def test_loop_tracer_reports_zero_fast_drive_on_the_slow_layers(self):
        for structure in STRUCTURES:
            with self.subTest(structure):
                model = build_rnn(structure)
                tracer = LoopTracer(model, hconfig()["learning_rate"])
                with contextlib.redirect_stdout(io.StringIO()):
                    train_module.train_batch(None, episodes(TEXTS), model, hconfig(),
                                             {"training_instance": 0, "log_norms_now": False}, tracer=tracer)
                traces = tracer.finish()
                drive = traces["fast_drive"]  # [T, B, L]: trunk layers 0-2, then i2h
                self.assertEqual(drive[..., 1:3].abs().sum().item(), 0.0)
                self.assertGreater(drive[1:, :, 0].abs().sum().item(), 0.0)
                self.assertGreater(drive[1:, :, 3].abs().sum().item(), 0.0)

    def test_load_eval_ablation_is_a_no_op_on_slow_layers(self):
        spec = importlib.util.spec_from_file_location("load_eval", LOAD_EVAL)
        load_eval = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(load_eval)
        for structure in STRUCTURES:
            with self.subTest(structure):
                model = build_rnn(structure)
                train_once(model)
                before = {k: v.clone() for k, v in model.state_dict().items()}
                for index in (1, 2):
                    load_eval.zero_fast(model.linear_layers[index])
                for key, value in model.state_dict().items():
                    assert_same(self, value, before[key])
                load_eval.zero_fast(model.linear_layers[0])
                self.assertEqual(model.linear_layers[0].fast_weights().abs().sum().item(), 0.0)


class CliTest(unittest.TestCase):
    def test_cli_trains_checkpoints_and_resumes_with_fast_layers_0(self):
        for structure in STRUCTURES:
            with self.subTest(structure), tempfile.TemporaryDirectory() as directory:
                flags = ("--fast_structure", structure, "--fast_layers", "0", "--num_layers", "3",
                         "--ephemeral_fraction", "0.5", "--plasticity", "5", "--enable_recurrence", "False")
                run_main(*flags, checkpoint_dir=directory)
                checkpoint = latest(directory)
                self.assertEqual(checkpoint["config"]["fast_layers"], "0")
                keys = sorted(k for k in checkpoint["model_state_dict"] if k.endswith("fast_state"))
                if structure == 'exclusive':
                    self.assertEqual(keys, [])
                else:
                    self.assertEqual(keys, ["i2h.fast_state", "linear_layers.0.fast_state"])
                run_main(*flags, "--resume", "--n_iters", "5", checkpoint_dir=directory)
                self.assertEqual(latest(directory)["iter"], 6)
                self.assertEqual(latest(directory)["config"]["fast_layers"], "0")


if __name__ == "__main__":
    unittest.main()
