"""--write_gain_power (2026-10-27, proposal P3, the expansive write-gain test).

- Off (the default), the model is the one main gave: the same run, bit for bit (fixture "main/plain"),
  and no checkpoint key is added.
- The gain is (|e| / m)^p per gained layer and batch row, m the lower median of the window's raw
  norms (NaN-free rows only); no history gives gain 1, a zero row gain 0, --write_gain_max caps it.
- Only the fast entries change: one step from the wipe writes gain_b times the ungained write, and
  the slow entries, biases and i2o are bit-identical to the ungained step.
- The median is causal and kept by the training steps only: the first step equals the ungained
  step; fast_write_projected, the held-out fast-only step and evaluate_protocols leave the history
  alone; the ring buffer wraps; padding rows stay out of it.
- The unfused step, the eager fused step and the norm-logging steps agree bit for bit.
- Unsupported combinations and a resume with different settings are refused; the history is saved."""
import contextlib
import io
import math
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from ephemeral_model import EphemeralRNN, nan_lower_median, write_gain_factor
from reproducibility import seed_everything
from tests import scratch_flags_reference as ref
from train import build_model, parse_args, train
from utils import CHECKPOINT_CODE_VERSION, load_checkpoint

FIXTURE = Path(__file__).parent / "fixtures" / "scratch_flags.pt"


def build(**kwargs):
    return ref.build(EphemeralRNN, seed_everything, **kwargs)


def build_unclamped(**kwargs):
    """ref.build without --weight_clamp, so a write is linear in its error."""
    seed_everything(11, deterministic=True)
    with contextlib.redirect_stdout(io.StringIO()):
        return EphemeralRNN(len(ref.CHARSET), 4, len(ref.CHARSET), 3, ref.CHARSET, plasticity=3.0,
                            batch_size=ref.BATCH_SIZE, forget_rate=0.25, ephemeral_fraction=0.5,
                            enable_recurrence=False, **kwargs)


def same_state(test, first, second):
    test.assertEqual(first.keys(), second.keys())
    for key in first:
        test.assertTrue(torch.equal(first[key], second[key]), key)


def one_step(model, error_seed=4, fused=True):
    """A forward from the wipe on a fixed input and one training DFA step with a random output error."""
    model.start_sequence_wipe()
    torch.manual_seed(3)
    onehot = F.one_hot(torch.randint(0, 4, (ref.BATCH_SIZE,)), 4).float()
    out, _ = model(onehot, model.initHidden(ref.BATCH_SIZE))
    torch.manual_seed(error_seed)
    error = torch.randn_like(out)
    if fused:
        model.fused_dfa_step(error, ref.LEARNING_RATE, 0)
    return error


class WriteGainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_off_matches_main_and_the_default(self):
        fixture = torch.load(FIXTURE, weights_only=False)
        exact = fixture["generated_with"]["torch"] == str(torch.__version__)
        tol = dict(rtol=0, atol=0) if exact else dict(rtol=1e-5, atol=1e-6)
        default = ref.run(build(), train)
        explicit = ref.run(build(write_gain_power=0.0, write_gain_window=7, write_gain_max=3.0), train)
        self.assertEqual(default["losses"], explicit["losses"])
        same_state(self, default["state"], explicit["state"])
        torch.testing.assert_close(torch.tensor(default["losses"]),
                                   torch.tensor(fixture["main"]["plain"]["losses"]), **tol)
        self.assertFalse(any("write_gain" in key for key in build().state_dict()))
        self.assertEqual(build().write_gain_summary(), {})

    def test_median_and_gain_formula(self):
        torch.manual_seed(0)
        values = torch.rand(3, 20)
        values[0, 5:] = float('nan')
        values[2] = float('nan')
        med = nan_lower_median(values)
        self.assertEqual(float(med[0]), float(torch.median(values[0, :5])))
        self.assertEqual(float(med[1]), float(torch.median(values[1])))  # even count: the lower middle
        self.assertTrue(math.isnan(float(med[2])))
        norms = torch.tensor([[0.0, 1.0, 2.0], [0.5, 1.0, 4.0], [3.0, 1.0, 0.0]])
        medians = torch.tensor([1.0, 0.5, float('nan')])
        gain = write_gain_factor(norms, medians, 2.0)
        torch.testing.assert_close(gain, torch.tensor([[0.0, 1.0, 4.0], [1.0, 4.0, 64.0], [1.0, 1.0, 1.0]]))
        capped = write_gain_factor(norms, medians, 2.0, max_gain=5.0)
        torch.testing.assert_close(capped, torch.tensor([[0.0, 1.0, 4.0], [1.0, 4.0, 5.0], [1.0, 1.0, 1.0]]))

    def test_gain_scales_only_the_fast_write(self):
        power = 2.0
        off, on = build_unclamped(), build_unclamped(write_gain_power=power, write_gain_window=4)
        for model in (off, on):
            model.enable_fused_update(compile=False)
        self.assertEqual(on.write_gain_layers, [0, 1, 2, 3])  # the three trunk layers and i2h
        torch.manual_seed(7)
        on.write_gain_history.uniform_(0.05, 0.3)  # a history: the gains are far from 1
        medians = on.write_gain_medians().clone()
        error = one_step(off)
        projected, _ = off.dfa_step_errors(error)
        self.assertTrue(torch.equal(one_step(on), error))
        norms = torch.stack([projected[i].norm(dim=1) for i in on.write_gain_layers])
        gains = write_gain_factor(norms, medians, power)
        self.assertGreater(float(gains.max()), 2.0)
        for index, (a, b) in enumerate(zip(off.trained_layers(), on.trained_layers())):
            mask = a.ephemeral_mask.unsqueeze(0).expand_as(a.per_sample_weights)
            # slow entries, biases and i2o: the ungained step
            self.assertTrue(torch.equal(a.per_sample_weights[~mask], b.per_sample_weights[~mask]), index)
            if a.bias is not None:
                self.assertTrue(torch.equal(a.bias, b.bias), index)
            if index in on.write_gain_layers:
                g = gains[on.write_gain_layers.index(index)].view(-1, 1, 1)
                expected = torch.where(mask, a.per_sample_weights * g, a.per_sample_weights)
                torch.testing.assert_close(b.per_sample_weights, expected, rtol=1e-5, atol=1e-7)
                self.assertGreater(float(a.per_sample_weights[mask].abs().sum()), 0)

    def test_first_step_is_ungained_and_the_history_is_causal(self):
        off, on = build_unclamped(), build_unclamped(write_gain_power=4.0, write_gain_window=3)
        for model in (off, on):
            model.enable_fused_update(compile=False)
        error = one_step(on, fused=False)
        projected, _ = on.dfa_step_errors(error)
        # inspecting the write does not touch the history
        inspected = on.fast_write_projected(error)
        self.assertTrue(torch.isnan(on.write_gain_history).all())
        for index in on.write_gain_layers:
            self.assertTrue(torch.equal(inspected[index], projected[index]))  # no history: gain 1
        on.fused_dfa_step(error, ref.LEARNING_RATE, 0)
        one_step(off)
        same_state(self, off.state_dict(), {k: v for k, v in on.state_dict().items() if "write_gain" not in k})
        self.assertEqual(int(on.write_gain_steps), 1)
        expected = torch.stack([projected[i].norm(dim=1) for i in on.write_gain_layers])
        self.assertTrue(torch.equal(on.write_gain_history[:, :ref.BATCH_SIZE], expected))
        self.assertTrue(torch.isnan(on.write_gain_history[:, ref.BATCH_SIZE:]).all())
        # the held-out fast-only step reads the median but does not add to it
        history = on.write_gain_history.clone()
        on.fast_only_dfa_step(error, ref.LEARNING_RATE, 0)
        self.assertTrue(torch.equal(on.write_gain_history.nan_to_num(-1), history.nan_to_num(-1)))
        self.assertEqual(int(on.write_gain_steps), 1)

    def test_ring_buffer_wraps_and_skips_padding(self):
        model = build_unclamped(write_gain_power=1.0, write_gain_window=2)
        model.enable_fused_update(compile=False)
        pushed = []
        for step in range(3):
            model.start_sequence_wipe()
            onehot = F.one_hot(torch.tensor([0, 1, 2]), 4).float()
            out, _ = model(onehot, model.initHidden(ref.BATCH_SIZE))
            torch.manual_seed(10 + step)
            error = torch.randn_like(out)
            error[1] = 0  # a padding row
            projected, _ = model.dfa_step_errors(error)
            pushed.append(torch.stack([projected[i].norm(dim=1) for i in model.write_gain_layers]))
            model.fused_dfa_step(error, ref.LEARNING_RATE, 0)
        history = model.write_gain_history.view(len(model.write_gain_layers), 2, ref.BATCH_SIZE)
        for slot, step in ((0, 2), (1, 1)):
            torch.testing.assert_close(history[:, slot, 0], pushed[step][:, 0], rtol=0, atol=0)
            torch.testing.assert_close(history[:, slot, 2], pushed[step][:, 2], rtol=0, atol=0)
            self.assertTrue(torch.isnan(history[:, slot, 1]).all())
        self.assertEqual(int(model.write_gain_steps), 3)
        stats = model.write_gain_summary()
        self.assertEqual(set(stats), {"write_gain_mean", "write_gain_max", "write_gain_frac_above_1",
                                      *(f"write_gain_median_{i}" for i in model.write_gain_layers)})

    def test_unfused_fused_and_norm_logging_paths_agree(self):
        for kwargs in ({"write_gain_power": 2.0, "write_gain_window": 5},
                       {"write_gain_power": 1.0, "write_gain_window": 64, "write_gain_max": 3.0},
                       {"write_gain_power": 4.0, "write_gain_window": 8, "decouple_fast_feedback": [2]},
                       {"write_gain_power": 2.0, "write_gain_window": 8, "fast_layers": [1]}):
            with self.subTest(**{k: str(v) for k, v in kwargs.items()}):
                unfused, fused, logged = build(**kwargs), build(**kwargs), build(**kwargs)
                fused.enable_fused_update(compile=False)
                logged.enable_fused_update(compile=False)
                a, b, c = ref.run(unfused, train), ref.run(fused, train), ref.run(logged, train, log_norms_every=3)
                self.assertEqual(a["losses"], b["losses"])
                self.assertEqual(a["losses"], c["losses"])
                same_state(self, a["state"], b["state"])
                same_state(self, a["state"], c["state"])
                for m in (fused, logged):
                    self.assertTrue(torch.equal(unfused.write_gain_history.nan_to_num(-1),
                                                m.write_gain_history.nan_to_num(-1)))
                    self.assertEqual(int(unfused.write_gain_steps), int(m.write_gain_steps))
                self.assertEqual(int(unfused.write_gain_steps), ref.N_BATCHES * (ref.SEQUENCE_LENGTH - 1))

    def test_on_changes_the_run_and_stays_finite(self):
        default = ref.run(build(), train)
        for power in (1.0, 2.0, 4.0):
            run = ref.run(build(write_gain_power=power, write_gain_window=16), train)
            self.assertNotEqual(run["losses"], default["losses"], power)
            self.assertTrue(all(math.isfinite(v) for v in run["losses"]), power)

    def test_heldout_leaves_the_history_alone(self):
        from heldout import evaluate_protocols
        model = build(write_gain_power=2.0, write_gain_window=8)
        ref.run(model, train)
        history, steps = model.write_gain_history.clone(), int(model.write_gain_steps)
        texts = ["abcdab", "bcdabc", "cdabcd"]
        onehot = F.one_hot(torch.tensor([[ref.CHARSET.index(ch) for ch in t] for t in texts]), 4).float()
        config = {"input_mode": "last_one", "pe_matrix": None, "learning_rate": 0.1,
                  "ephemeral_update_clamp": 0, "grad_norm_clip": 0}
        with contextlib.redirect_stdout(io.StringIO()):
            evaluate_protocols(model, [(texts, onehot)], config, "palindrome_dataset",
                               protocols=("observed", "strict", "no_fast"))
        self.assertTrue(torch.equal(model.write_gain_history.nan_to_num(-1), history.nan_to_num(-1)))
        self.assertEqual(int(model.write_gain_steps), steps)

    def test_invalid_combinations_are_refused(self):
        for kwargs in ({"write_gain_power": -1.0}, {"write_gain_power": 1.0, "write_gain_window": 0},
                       {"write_gain_power": 1.0, "write_gain_max": -1.0},
                       {"write_gain_power": 1.0, "updater": "backprop"},
                       {"write_gain_power": 1.0, "slow_update_every": 2},
                       {"write_gain_power": 1.0, "fast_backward_per_forward": 2},
                       {"write_gain_power": 1.0, "fast_backward_per_forward": "1/2"},
                       {"write_gain_power": 1.0, "order2_layers": [1]},
                       {"write_gain_power": 1.0, "fast_structure": "additive_masked"}):
            with self.subTest(**{k: str(v) for k, v in kwargs.items()}), self.assertRaises(ValueError):
                build(**kwargs)
        base = ['--model_type', 'ephemeral', '--updater', 'dfa', '--write_gain_power', '2']
        for argv in (['--grad_norm_clip', '1'], ['--updater', 'backprop'], ['--model_type', 'rnn'],
                     ['--slow_update_every', '2'], ['--fast_backward_per_forward', '2'],
                     ['--order2_layers', '1'], ['--fast_structure', 'additive_dense'],
                     ['--write_gain_window', '0'], ['--write_gain_max', '-1'], ['--write_gain_power', '-1']):
            with self.subTest(argv=argv), self.assertRaises(SystemExit), \
                    contextlib.redirect_stderr(io.StringIO()):
                parse_args(base + argv)

    def test_build_model_reads_the_flags_and_resume_keeps_or_refuses(self):
        args = parse_args(['--model_type', 'ephemeral', '--updater', 'dfa', '--write_gain_power', '4',
                           '--write_gain_window', '9', '--write_gain_max', '50', '--hidden_size', '8',
                           '--num_layers', '3', '--batch_size', '2', '--seed', '9'])
        config = dict(vars(args), n_hidden=8, n_layers=3, pe_matrix=None)
        with contextlib.redirect_stdout(io.StringIO()):
            model = build_model(config, ref.CHARSET, len(ref.CHARSET))
            fresh = build_model(config, ref.CHARSET, len(ref.CHARSET))
        self.assertEqual((model.write_gain_power, model.write_gain_window, model.write_gain_max), (4.0, 9, 50.0))
        self.assertEqual(model.write_gain_history.shape, (4, 18))
        model.write_gain_history[:, :5] = 0.7
        model.write_gain_steps.fill_(5)
        checkpoint = {"code_version": CHECKPOINT_CODE_VERSION, "config": dict(config),
                      "model_state_dict": model.state_dict(), "iter": 1}
        with contextlib.redirect_stdout(io.StringIO()):
            load_checkpoint("x.pth", fresh, config, checkpoint=checkpoint)
        self.assertEqual(int(fresh.write_gain_steps), 5)
        torch.testing.assert_close(fresh.write_gain_medians(), torch.full((4,), 0.7))
        for change in ({"write_gain_power": 2.0}, {"write_gain_window": 8}, {"write_gain_max": 0.0},
                       {"write_gain_power": 0.0}):
            checkpoint["config"] = dict(config, **change)
            with self.subTest(**change), contextlib.redirect_stdout(io.StringIO()), \
                    self.assertRaises(RuntimeError):
                load_checkpoint("x.pth", fresh, config, checkpoint=checkpoint)


if __name__ == "__main__":
    unittest.main()
