"""--label_smoothing, --entropy_penalty, --readout_lr_scale, --log_margin: opt-in margin interventions."""
import contextlib
import copy
import io
import tempfile
import unittest

import torch
import torch.nn.functional as F

import train as train_module
from ephemeral_model import EphemeralRNN, dfa_output_error
from reproducibility import seed_everything
from tests.test_feedback_init import BATCH, CHARSET, V, build, episode
from tests.test_heldout import run_main

CE = torch.nn.CrossEntropyLoss(reduction="none")


def logits_and_target(seed=0):
    generator = torch.Generator().manual_seed(seed)
    output = torch.randn(BATCH, V, generator=generator)
    target = F.one_hot(torch.randint(V, (BATCH,), generator=generator), V).float()
    return output, target


class OutputErrorTest(unittest.TestCase):
    def test_defaults_are_the_plain_error(self):
        output, target = logits_and_target()
        loss, error = dfa_output_error(output.clone(), target, CE)
        loss0, error0 = dfa_output_error(output.clone(), target, CE, 0.0, 0.0, None)
        self.assertTrue(torch.equal(error, error0) and torch.equal(loss, loss0))
        torch.testing.assert_close(error, torch.softmax(output, 1) - target)

    def test_label_smoothing_error_is_softmax_minus_smoothed_target(self):
        output, target = logits_and_target()
        _, error = dfa_output_error(output.clone(), target, CE, 0.3)
        smoothed = target * 0.7 + 0.3 / V
        torch.testing.assert_close(error, torch.softmax(output, 1) - smoothed)
        torch.testing.assert_close(error.sum(1), torch.zeros(BATCH), atol=1e-6, rtol=0)

    def test_padding_rows_stay_at_zero_error_and_rows_mask_limits_the_scope(self):
        output, target = logits_and_target()
        target[0] = 0  # padding
        _, error = dfa_output_error(output.clone(), target, CE, 0.3, 0.5)
        self.assertEqual(float(error[0].abs().max()), 0.0)
        self.assertGreater(float(error[1].abs().max()), 0.0)
        _, plain = dfa_output_error(output.clone(), target, CE)
        _, masked = dfa_output_error(output.clone(), target, CE, 0.3, 0.5, torch.tensor([True, False, False, False]))
        torch.testing.assert_close(masked[1:], plain[1:])

    def test_entropy_penalty_matches_autograd_and_pushes_toward_less_confidence(self):
        output, target = logits_and_target()
        beta = 0.4
        _, error = dfa_output_error(output.clone(), target, CE, 0.0, beta)
        z = output.clone().requires_grad_(True)
        logp = torch.log_softmax(z, 1)
        reference = (CE(z, target) + beta * (logp.exp() * logp).sum(1)).sum()
        torch.testing.assert_close(error, torch.autograd.grad(reference, z)[0])
        # a very confident wrong guess: the penalty adds a gradient that lowers the top logit
        confident = torch.tensor([[6.0, 0, 0, 0, 0, 0]])
        t = F.one_hot(torch.tensor([3]), V).float()
        _, with_penalty = dfa_output_error(confident.clone(), t, CE, 0.0, beta)
        _, without = dfa_output_error(confident.clone(), t, CE)
        self.assertGreater(float(with_penalty[0, 0]), float(without[0, 0]))


class ReadoutLearningRateTest(unittest.TestCase):
    def step(self, scale, fused):
        model = build("random", recurrence=False)
        model.i2o.lr_scale = scale
        if fused:
            model.enable_fused_update(compile=False)
        inputs, targets = episode()
        model.start_sequence_wipe()
        hidden = model.initHidden(BATCH)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        with torch.no_grad():
            out, hidden = model(inputs[0], hidden)
        _, error = dfa_output_error(out, targets[0], CE)
        if fused:
            model.fused_dfa_step(error, 0.5, 0)
        else:
            model.clear_dfa_gradients()
            for layer in model.linear_layers:
                layer.populate_dfa_gradients(error)
            model.i2h.populate_dfa_gradients(error)
            model.i2o.populate_dfa_gradients(error)
            for layer in model.trained_layers():
                layer.apply_update(0.5, 0, {})
        return model, before

    def test_scale_multiplies_only_the_i2o_step_fused_and_unfused(self):
        for fused in (False, True):
            full, before = self.step(1.0, fused)
            half, _ = self.step(0.5, fused)
            changed = False
            for name, p in full.named_parameters():
                q = dict(half.named_parameters())[name]
                if name in ("i2o.bias", "i2o.per_sample_weights"):
                    delta, delta_half = p.detach() - before[name], q.detach() - before[name]
                    changed |= float(delta.abs().max()) > 0
                    torch.testing.assert_close(delta_half, 0.5 * delta, rtol=1e-4, atol=1e-8)
                elif not name.startswith("i2o") and p.dtype.is_floating_point and "traces" not in name:
                    self.assertTrue(torch.equal(p, q), (fused, name))
            self.assertTrue(changed)

    def test_scale_one_keeps_the_shared_error_object(self):
        model = build("random")
        error = torch.randn(BATCH, V)
        model.i2o.populate_dfa_gradients(error)
        self.assertIs(model.i2o._last_projected_error, error)


class MarginStatsTest(unittest.TestCase):
    def test_margin_is_correct_minus_best_other_at_the_marked_rows(self):
        stats = train_module.MarginStats()
        logits = torch.tensor([[2.0, 0.0, 1.0], [0.0, 3.0, 1.0], [5.0, 5.0, 5.0]])
        target = F.one_hot(torch.tensor([2, 1, 0]), 3).float()
        stats.add(logits, target, torch.tensor([True, True, False]))
        summary = stats.summary()
        self.assertAlmostEqual(summary["answer_margin"], (-1.0 + 2.0) / 2)
        p = torch.softmax(logits, 1)
        self.assertAlmostEqual(summary["answer_p_correct"], float((p[0, 2] + p[1, 1]) / 2), places=5)
        stats.reset()
        self.assertEqual(stats.summary(), {})


class PalindromeMarginTest(unittest.TestCase):
    CHARS = train_module.get_charset("3_palindrome_dataset_vary_length")

    def onehot(self, strings):
        idx = torch.tensor([[self.CHARS.index(c) for c in text] for text in strings])
        return F.one_hot(idx, len(self.CHARS)).float()

    def test_answer_groups_mark_the_reversed_half_only(self):
        seqs = self.onehot(["20!.!02", ",!.!,  ", "1.1    "])
        dot = self.CHARS.index(".")
        groups = torch.stack([train_module.palindrome_answer_groups(seqs, i, dot, 3) for i in range(6)], 1)
        # step i has input char i and target char i+1: the answers are the targets after the '.'
        self.assertEqual(groups.tolist(), [[0, 0, 0, 1, 2, 3], [0, 0, 1, 2, 0, 0], [0, 1, 0, 0, 0, 0]])

    def test_grouped_stats_are_per_position_and_whole_sequence(self):
        stats = train_module.MarginStats()
        V = 4
        target = F.one_hot(torch.tensor([0, 1]), V).float()
        # step 1: both rows answer position 1; row 0 right with margin 2, row 1 wrong with margin -1
        stats.add(torch.tensor([[2.0, 0, 0, 0], [0, 0, 1.0, 0]]), target, torch.tensor([True, True]),
                  torch.tensor([1, 1]), 2)
        # step 2: only row 0 answers position 2; wrong with margin -3
        stats.add(torch.tensor([[0, 3.0, 0, 0], [5.0, 0, 0, 0]]), F.one_hot(torch.tensor([2, 0]), V).float(), torch.tensor([True, False]),
                  torch.tensor([2, 0]), 2)
        stats.end_batch()
        out = stats.summary()
        self.assertAlmostEqual(out["answer_margin_pos1"], 0.5)
        self.assertAlmostEqual(out["answer_acc_pos1"], 0.5)
        self.assertAlmostEqual(out["answer_margin_pos2"], -3.0)
        self.assertAlmostEqual(out["answer_acc_pos2"], 0.0)
        self.assertAlmostEqual(out["answer_margin"], (2 - 1 - 3) / 3)
        self.assertAlmostEqual(out["answer_seq_acc"], 0.0)  # row 0 misses pos 2; row 1 misses pos 1
        stats.reset()
        self.assertEqual(stats.summary(), {})

    def test_ungrouped_summary_has_only_the_original_keys(self):
        stats = train_module.MarginStats()
        stats.add(torch.eye(3), torch.eye(3), torch.tensor([True, True, False]))
        stats.end_batch()
        self.assertEqual(set(stats.summary()), {"answer_margin", "answer_p_correct", "answer_p_max"})

    def test_cli_accepts_the_palindrome_task_only_for_log_margin(self):
        ok = train_module.parse_args(["--dataset", "3_palindrome_dataset_vary_length", "--log_margin", "true"])
        self.assertTrue(ok.log_margin)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            train_module.parse_args(["--dataset", "3_palindrome_dataset_vary_length", "--shaping_scope", "answer",
                                     "--label_smoothing", "0.1"])
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            train_module.parse_args(["--dataset", "palindrome_dataset_vary_length", "--log_margin", "true"])

    def test_logging_does_not_change_training(self):
        from tests.test_heldout import TEXTS, build as build_small, config as config_small, episodes
        outcome = {}
        for flag in (False, True):
            model = build_small()
            train_module.MARGIN_STATS.reset()
            cfg = config_small(dataset="3_small_palindrome_dataset_vary_length", log_margin=flag)
            train_module.train_batch(None, episodes(TEXTS), model, cfg, {"training_instance": 0, "log_norms_now": False})
            outcome[flag] = ({k: v.clone() for k, v in model.state_dict().items()}, train_module.MARGIN_STATS.summary())
        for name, value in outcome[False][0].items():
            self.assertTrue(torch.equal(value, outcome[True][0][name]), name)
        self.assertEqual(outcome[False][1], {})
        summary = outcome[True][1]
        # "23.32  " answers 2 characters, "322.223" answers 3: position 3 has one row per step it exists
        for pos in (1, 2, 3):
            self.assertIn(f"answer_margin_pos{pos}", summary)
        self.assertIn("answer_seq_acc", summary)
        train_module.MARGIN_STATS.reset()


class CommandLineTest(unittest.TestCase):
    def refused(self, *argv):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            train_module.parse_args(list(argv))

    def test_defaults_and_refusals(self):
        args = train_module.parse_args([])
        self.assertEqual((args.label_smoothing, args.entropy_penalty, args.readout_lr_scale, args.log_margin,
                          args.shaping_scope), (0.0, 0.0, 1.0, False, "all"))
        self.refused("--label_smoothing", "1.0")
        self.refused("--label_smoothing", "0.1", "--updater", "backprop")
        self.refused("--readout_lr_scale", "0.1", "--model_type", "rnn")
        self.refused("--entropy_penalty", "-1")
        self.refused("--shaping_scope", "answer")
        self.refused("--label_smoothing", "0.1", "--shaping_scope", "answer")  # dataset is not long_range
        self.refused("--dataset", "roneneldan/tinystories", "--log_margin", "true")
        self.refused("--label_smoothing", "0.1", "--fast_backward_per_forward", "2")
        self.refused("--label_smoothing", "0.1", "--heldout_eval_every", "2")
        ok = train_module.parse_args(["--dataset", "long_range_memory_dataset", "--label_smoothing", "0.1",
                                      "--shaping_scope", "answer", "--log_margin", "true"])
        self.assertEqual((ok.label_smoothing, ok.shaping_scope, ok.log_margin), (0.1, "answer", True))

    def test_main_runs_with_the_levers_and_defaults_are_unchanged(self):
        def weights(*extra):
            with tempfile.TemporaryDirectory() as d:
                run_main(*extra, checkpoint_dir=d)
                return torch.load(f"{d}/latest_checkpoint.pth", weights_only=False)["model_state_dict"]
        base, explicit = weights(), weights("--label_smoothing", "0", "--readout_lr_scale", "1")
        for name, value in base.items():
            self.assertTrue(torch.equal(value, explicit[name]), name)
        levers = weights("--label_smoothing", "0.2", "--readout_lr_scale", "0.5", "--entropy_penalty", "0.1")
        self.assertTrue(any(not torch.equal(base[n], levers[n]) for n in base))
        for value in levers.values():
            self.assertTrue(torch.isfinite(value.float()).all())

    def test_resume_with_a_different_lever_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            run_main("--label_smoothing", "0.2", checkpoint_dir=d)
            with self.assertRaises(RuntimeError), contextlib.redirect_stdout(io.StringIO()):
                run_main("--label_smoothing", "0.3", "--n_iters", "8", "--resume", "true", checkpoint_dir=d)


if __name__ == "__main__":
    unittest.main()
