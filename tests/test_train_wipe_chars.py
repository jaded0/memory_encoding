"""Character wipes must not consolidate slow weights or reset recurrent state."""
import copy
import contextlib
import io
import tempfile
import unittest
from unittest.mock import patch

import torch

import train
from tests.test_heldout import build, config, episodes, TEXTS
from tests.test_seed_resume import run_main, latest, assert_tensors_equal


class TrainWipeCharsTest(unittest.TestCase):
    def test_default_is_exactly_old_path(self):
        a, b = build(), build()
        hot = episodes(TEXTS)
        state = {"training_instance": 0, "log_norms_now": False}
        ra = train.train_batch(None, hot, a, config(), state.copy())
        rb = train.train_batch(None, hot, b, config(train_wipe_chars=0), state.copy())
        for key in a.state_dict():
            self.assertTrue(torch.equal(a.state_dict()[key], b.state_dict()[key]), key)
        torch.testing.assert_close(ra[3], rb[3], rtol=0, atol=0)

    def test_boundaries_are_fast_only_and_preserve_slow_row_differences(self):
        model = build()
        snapshots = []
        original = model.wipe_fast_state

        def spy():
            old = [l.per_sample_weights.detach().clone() for l in model.trained_layers()]
            times = [l.t.clone() for l in model.trained_layers()]
            original()
            for layer, before, t in zip(model.trained_layers(), old, times):
                torch.testing.assert_close(layer.per_sample_weights[:, ~layer.ephemeral_mask],
                                           before[:, ~layer.ephemeral_mask], rtol=0, atol=0)
                self.assertEqual(int(torch.count_nonzero(
                    layer.per_sample_weights[:, layer.ephemeral_mask])), 0)
                torch.testing.assert_close(layer.t, t, rtol=0, atol=0)
            snapshots.append(old)

        with patch.object(model, 'wipe_fast_state', side_effect=spy) as wipe, \
                patch.object(model, 'start_sequence_wipe', wraps=model.start_sequence_wipe) as start:
            train.train_batch(None, episodes(TEXTS), model, config(train_wipe_chars=2),
                              {"training_instance": 0, "log_norms_now": False})
        self.assertEqual(wipe.call_count, 2)  # 6 steps: before i=2,4, never i=0
        self.assertEqual(start.call_count, 1)
        self.assertTrue(any(not torch.equal(w[0], w[1]) for s in snapshots for w in s))

    def test_wipe_helper_matches_evaluator_including_additive(self):
        from tests.test_b2_horizon import b2_eval
        for structure in ('exclusive', 'additive_dense', 'additive_masked'):
            a = build(fast_structure=structure)
            train.train_batch(None, episodes(TEXTS), a, config(),
                              {"training_instance": 0, "log_norms_now": False})
            b = copy.deepcopy(a)
            a.wipe_fast_state()
            b2_eval.wipe_fast(b)
            for k in a.state_dict():
                self.assertTrue(torch.equal(a.state_dict()[k], b.state_dict()[k]), k)

    def test_arguments(self):
        self.assertEqual(train.parse_args([]).train_wipe_chars, 0)
        self.assertEqual(train.parse_args(['--train_wipe_chars', '16']).train_wipe_chars, 16)
        for flags in (['--train_wipe_chars', '-1'],
                      ['--train_wipe_chars', '16', '--model_type', 'rnn'],
                      ['--train_wipe_chars', '16', '--updater', 'bptt']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                train.parse_args(flags)

    def test_resume_refuses_changed_wipe_period(self):
        with tempfile.TemporaryDirectory() as d:
            run_main('--train_wipe_chars', '2', checkpoint_dir=d)
            with self.assertRaisesRegex(RuntimeError, 'configuration mismatch'):
                run_main('--train_wipe_chars', '3', '--resume', 'true', checkpoint_dir=d)

    def test_same_period_resume_is_bit_identical(self):
        flags = ('--train_wipe_chars', '2', '--seed', '1729', '--deterministic', 'true')
        with tempfile.TemporaryDirectory() as straight, tempfile.TemporaryDirectory() as split:
            run_main(*flags, '--n_iters', '8', checkpoint_dir=straight)
            run_main(*flags, '--n_iters', '3', checkpoint_dir=split)
            run_main(*flags, '--resume', 'true', '--n_iters', '8', checkpoint_dir=split)
            a, b = latest(straight), latest(split)
        self.assertEqual(a['config']['train_wipe_chars'], 2)
        assert_tensors_equal(self, list(a['model_state_dict'].values()), list(b['model_state_dict'].values()))
        torch.testing.assert_close(a['torch_rng_state'], b['torch_rng_state'], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
