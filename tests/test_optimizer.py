"""--optimizer: SGD stays the default everywhere; Adam steps the SimpleRNN baseline under backprop and
BPTT, is refused where updates are manual (the ephemeral model, DFA), and is part of checkpoint
compatibility."""
import contextlib
import copy
import io
import os
import tempfile
import unittest

import torch

import train as train_module
from ephemeral_model import SimpleRNN
from tests.test_cli_aliases import parse, parse_error
from tests.test_failure_paths import run_main
from utils import load_checkpoint, save_checkpoint


def tiny_rnn(updater):
    torch.manual_seed(0)
    return SimpleRNN(4, 4, 4, 1, dropout_rate=0, enable_recurrence=True, updater=updater)


def tiny_config(updater, optimizer):
    return {"learning_rate": 1e-2, "optimizer": optimizer, "updater": updater, "grad_norm_clip": 0,
            "input_mode": "last_one", "pe_matrix": None, "criterion": torch.nn.CrossEntropyLoss(reduction="none")}


def batch():
    indices = torch.tensor([[0, 1, 2, 1, 0], [1, 0, 2, 0, 1]])
    return indices, torch.nn.functional.one_hot(indices, 4).float()


class OptimizerFlagTest(unittest.TestCase):
    def test_default_is_sgd_for_every_model_and_updater(self):
        for model_type in ("rnn", "ephemeral"):
            for updater in ("dfa", "backprop", "bptt"):
                args, _ = parse("--model_type", model_type, "--updater", updater)
                self.assertEqual(args["optimizer"], "sgd")
                config = {"optimizer": args["optimizer"], "learning_rate": 1e-3}
                self.assertIs(type(train_module.build_optimizer(tiny_rnn("bptt"), config)), torch.optim.SGD)

    def test_adam_only_for_the_rnn_baseline_under_autograd_updaters(self):
        for updater in ("backprop", "bptt"):
            args, _ = parse("--model_type", "rnn", "--updater", updater, "--optimizer", "adam")
            self.assertEqual(args["optimizer"], "adam")
        for model_type, updater in (("rnn", "dfa"), ("ephemeral", "dfa"), ("ephemeral", "backprop"),
                                    ("ephemeral", "bptt")):
            with self.subTest(model_type=model_type, updater=updater):
                self.assertEqual(parse_error("--model_type", model_type, "--updater", updater,
                                             "--optimizer", "adam"), 2)

    def test_adam_first_bptt_step_moves_each_weight_by_lr_times_sign_of_its_gradient(self):
        # Adam's first step from zero moments is -lr * g / (|g| + eps): -lr * sign(g) where |g| >> eps.
        model = tiny_rnn("bptt")
        reference = copy.deepcopy(model)
        config = tiny_config("bptt", "adam")
        optimizer = train_module.build_optimizer(model, config)
        self.assertIs(type(optimizer), torch.optim.Adam)
        indices, onehot = batch()
        # The gradient, independently: the summed per-step loss over the whole sequence, batch mean.
        hidden, total = reference.initHidden(2), 0
        for i in range(onehot.shape[1] - 1):
            output, hidden = reference(onehot[:, i], hidden)
            total = total + torch.nn.functional.cross_entropy(output, onehot[:, i + 1], reduction="none")
        total.mean().backward()
        with contextlib.redirect_stdout(io.StringIO()):
            train_module.train(indices, onehot, model, config, {"training_instance": 0}, optimizer)
        for (name, after), before in zip(model.named_parameters(), reference.parameters()):
            grad = before.grad
            expected = before.detach() - 1e-2 * grad / (grad.abs() + 1e-8)
            with self.subTest(name=name):
                self.assertTrue(torch.allclose(after.detach(), expected, atol=1e-6))
                self.assertGreater((after.detach() - before.detach()).abs().max().item(), 5e-3)

    def test_changed_optimizer_is_a_checkpoint_mismatch_and_old_checkpoints_mean_sgd(self):
        base = {"n_hidden": 4, "n_layers": 1, "updater": "bptt", "charset_size": 4, "model_type": "rnn"}
        with tempfile.TemporaryDirectory() as directory:
            for saved, current, refused in (({"optimizer": "adam"}, {"optimizer": "sgd"}, True),
                                            ({}, {"optimizer": "adam"}, True),
                                            ({}, {"optimizer": "sgd"}, False),
                                            ({"optimizer": "adam"}, {"optimizer": "adam"}, False)):
                with self.subTest(saved=saved, current=current), contextlib.redirect_stdout(io.StringIO()):
                    save_checkpoint({"config": {**base, **saved}, "model_state_dict": tiny_rnn("bptt").state_dict()},
                                    directory, "c.pth")
                    load = lambda: load_checkpoint(os.path.join(directory, "c.pth"), tiny_rnn("bptt"),
                                                   {**base, **current})
                    if refused:
                        with self.assertRaisesRegex(RuntimeError, "configuration mismatch"):
                            load()
                    else:
                        load()

    def test_adam_run_saves_and_resumes_its_moments(self):
        with tempfile.TemporaryDirectory() as directory:
            args = ("--model_type", "rnn", "--updater", "bptt", "--enable_recurrence", "True",
                    "--optimizer", "adam", "--seed", "3", "--checkpoint_save_freq", "1")
            run_main(*args, checkpoint_dir=directory)
            checkpoint = torch.load(os.path.join(directory, "latest_checkpoint.pth"), weights_only=False)
            self.assertEqual(checkpoint["config"]["optimizer"], "adam")
            first = next(iter(checkpoint["optimizer_state_dict"]["state"].values()))
            self.assertIn("exp_avg", first)
            run_main(*args, "--resume", "True", "--n_iters", "5", checkpoint_dir=directory)
            resumed = torch.load(os.path.join(directory, "latest_checkpoint.pth"), weights_only=False)
            later = next(iter(resumed["optimizer_state_dict"]["state"].values()))
            # The moments were restored and continued, not restarted: Adam's step count carries on.
            self.assertEqual(float(later["step"]) - float(first["step"]), 2)


if __name__ == "__main__":
    unittest.main()
