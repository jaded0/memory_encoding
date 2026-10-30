"""Every launcher arm parses, is fixed at B2 size, and production stays gated."""
import importlib.util
import json
from pathlib import Path
import subprocess
import unittest

import train

HERE = Path(__file__).resolve().parents[1] / 'scratch/b2_followups_2026-10-27'


class B2FollowupLaunchersTest(unittest.TestCase):
    def test_all_arms_parse_and_have_expected_recipe(self):
        expected = {'DEEP-W16': (False, 'dfa', .01, 16),
                    'DEEP-F03': (False, 'dfa', .03, 0),
                    'DEEP-F10': (False, 'dfa', .1, 0),
                    'REC-NOFAST': (True, 'dfa', .01, 0),
                    'REC-DEEP': (True, 'dfa', .01, 0),
                    'RNN-BPTT': (True, 'bptt', .01, 0)}
        for arm, values in expected.items():
            result = subprocess.check_output(['bash', '-c',
                'source "$1/arms.sh"; arm_args "$2"; printf "%s\\0" "${FLAGS[@]}"',
                'bash', str(HERE), arm])
            args = train.parse_args(result.decode().rstrip('\0').split('\0'))
            self.assertEqual((args.enable_recurrence, args.updater, args.forget_rate,
                              args.train_wipe_chars), values)
            self.assertEqual((args.hidden_size, args.num_layers, args.batch_size,
                              args.max_seq_len, args.n_iters), (1024, 3, 16, 512, 20000))
            if arm == 'RNN-BPTT':
                self.assertEqual(args.optimizer, 'adam')
                self.assertEqual(args.grad_norm_clip, 1)
                self.assertEqual(args.learning_rate, .0001)
                self.assertTrue(args.residual_connection)
                self.assertFalse(args.layer_norm)  # fixed README baseline, not B2's DFA stabilizer

    def test_rnn_lr_override_only_touches_rnn(self):
        import os
        def flags(arm, lr):
            env = dict(os.environ, RNN_LR=lr)
            return subprocess.run(['bash', '-c',
                'source "$1/arms.sh"; arm_args "$2" && printf "%s\\0" "${FLAGS[@]}"',
                'bash', str(HERE), arm], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for lr in ('3e-5', '1e-4', '3e-4', '1e-3'):
            p = flags('RNN-BPTT', lr)
            args = train.parse_args(p.stdout.decode().rstrip('\0').split('\0'))
            self.assertAlmostEqual(args.learning_rate, float(lr))
            self.assertEqual((args.optimizer, args.grad_norm_clip, args.n_iters), ('adam', 1, 20000))
        p = flags('REC-DEEP', '1e-3')
        self.assertEqual(p.returncode, 2)
        self.assertIn(b'RNN-BPTT only', p.stderr)

    def test_production_launcher_requires_approval(self):
        for arm in ('DEEP-W16', 'REC-DEEP'):
            import os
            env = dict(os.environ)
            env.pop('B2FU_APPROVED', None)
            p = subprocess.run(['bash', str(HERE / 'run.sh'), arm, '7201'], env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(p.returncode, 2)
            self.assertIn(b'Launch gated', p.stderr)

    def test_score_requires_complete_inputs_and_uses_seed_as_unit(self):
        spec = importlib.util.spec_from_file_location('b2fu_score', HERE / 'score.py')
        score = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(score)
        self.assertEqual(score.summary([1, 1, 1, 1, 1])['ci95_t'], [1, 1])
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                score.score(Path(d), 'A')

    def test_score_pairs_the_correct_control_and_validation_count(self):
        import tempfile
        spec = importlib.util.spec_from_file_location('b2fu_score', HERE / 'score.py')
        score = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(score)
        def dump(root, sub, arm, seed, stories, ce):
            d = root / sub
            d.mkdir(parents=True, exist_ok=True)
            (d / f'{arm}-s{seed}.json').write_text(json.dumps({
                'stories': stories, 'length': 512, 'iteration': 20000,
                'arm_config': {'seed': seed}, 'ce': ce}))
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for i, seed in enumerate(score.SEEDS):
                base = 2 + i * .1
                for arm, on, off in [('NOFAST', base, base), ('DEEP', base - .2, base + .21)]:
                    ce = {c: {'on': on, 'off': off} for c in score.CLASSES}
                    dump(root, 'controls/final', arm, seed, 4096, ce)
                dump(root, 'controls/horizon4096', 'DEEP', seed, 4096, {'wipe16': {'all': base - .3}})
                for arm in score.ARMS['A']:
                    ce = {c: {'on': base - .25, 'off': base + .1} for c in score.CLASSES}
                    dump(root, 'eval_out/final', arm, seed, 4096, ce)
                    dump(root, 'eval_out/horizon', arm, seed, 1024,
                         {'on': {'all': base - .25}, 'wipe16': {'all': base - .26}})
            result = score.score(root, 'A')['arms']['DEEP-W16']
            self.assertAlmostEqual(result['tax']['all']['mean'], .1)
            self.assertAlmostEqual(result['old_tax_minus_new_tax']['mean'], .11)
            self.assertAlmostEqual(result['old_wipe16_minus_new_on']['mean'], -.05)


if __name__ == '__main__':
    unittest.main()
