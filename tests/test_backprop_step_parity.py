"""--backprop_step_parity and --decouple_fast_feedback under per-step backprop (C2, 2026-10-27).

- Off (the default), the backprop step is the one main has always taken: lr*alpha^2/B * g on the
  fast entries and lr/B * g on the slow ones, g each sequence's own loss gradient. Pinned here so
  the derivation in README "Backprop step parity" stays true; the golden traces
  (tests/test_characterization.py) pin it bit for bit.
- On, it is lr*alpha*g and lr*g, DFA's form. On a tiny model, i2o (whose DFA gradient is the exact
  gradient) and a layer whose fast entries take the B_fast write step exactly as under DFA.
- With --decouple_fast_feedback, the listed layers' fast entries take DC2's B_fast write while
  every slow entry and bias takes the same backprop step as without it, and the fast state is a
  constant to the slow gradient (no gradient through the write).
- The CLI refuses the combinations that are not defined."""
import contextlib
import copy
import io
import unittest

import torch
import torch.nn.functional as F

from ephemeral_model import EphemeralRNN, decoupled_fast_feedback
from reproducibility import seed_everything
from train import parse_args, train

CHARSET = list("abcd")
BATCH = 3
ALPHA = 7.0
LR = 0.05
BASE_ARGS = ["--model_type", "ephemeral", "--updater", "backprop", "--track", "false"]


def build(updater="backprop", **kwargs):
    seed_everything(5, deterministic=True)
    with contextlib.redirect_stdout(io.StringIO()):
        return EphemeralRNN(len(CHARSET), 6, len(CHARSET), 3, CHARSET, plasticity=ALPHA,
                            batch_size=BATCH, forget_rate=0.0, ephemeral_fraction=0.5,
                            enable_recurrence=False, updater=updater, **kwargs)


def config(updater="backprop", parity=False):
    return {"updater": updater, "criterion": torch.nn.CrossEntropyLoss(reduction="none"),
            "input_mode": "last_one", "pe_matrix": None, "learning_rate": LR, "plasticity": ALPHA,
            "ephemeral_update_clamp": 0, "grad_norm_clip": 0, "backprop_step_parity": parity}


def batch(length=2, seed=0):
    generator = torch.Generator().manual_seed(seed)
    line = torch.randint(0, len(CHARSET), (BATCH, length), generator=generator)
    return line, F.one_hot(line, len(CHARSET)).float()


def run(model, cfg, line, onehot):
    with contextlib.redirect_stdout(io.StringIO()):
        return train(line, onehot, model, cfg, {"training_instance": 0})


def weights(model):
    return [layer.per_sample_weights.detach().clone() for layer in model.trained_layers()]


def biases(model):
    return [None if layer.bias is None else layer.bias.detach().clone() for layer in model.trained_layers()]


def own_gradients(model, onehot):
    """Each sequence's own loss gradient for one prediction step from the model's current
    (just-wiped) state, per trained layer [B, out, in]. Copy b only affects sequence b's loss, so
    the gradient of the summed loss is each sequence's own gradient in its own slice."""
    model = copy.deepcopy(model)
    model.start_sequence_wipe()
    output, _ = model(onehot[:, 0], model.initHidden(BATCH))
    loss = F.cross_entropy(output, onehot[:, 1], reduction="none").sum()
    params = [layer.per_sample_weights for layer in model.trained_layers()]
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    return [torch.zeros_like(p) if g is None else g for p, g in zip(params, grads)]


class BackpropStepSizeTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def step(self, parity):
        model = build()
        line, onehot = batch()
        grads = own_gradients(model, onehot)
        before = weights(model)
        run(model, config(parity=parity), line, onehot)
        return model, grads, before, weights(model)

    def assert_step(self, parity, fast_factor, slow_factor):
        model, grads, before, after = self.step(parity)
        checked_fast = False
        for layer, g, w0, w1 in zip(model.trained_layers(), grads, before, after):
            mask = layer.ephemeral_mask.unsqueeze(0).expand_as(g)
            # Starting copies: wiped fast entries, batch-mean slow ones (all equal at init).
            w0 = torch.where(mask, torch.zeros_like(w0), w0)
            expected = torch.where(mask, w0 - LR * fast_factor * g, w0 - LR * slow_factor * g)
            torch.testing.assert_close(w1, expected, rtol=1e-5, atol=1e-7)
            checked_fast |= bool((mask & (g != 0)).any())
        self.assertTrue(checked_fast, "no fast entry had a gradient: the test checks nothing")

    def test_default_is_alpha_squared_over_batch(self):
        self.assert_step(False, ALPHA ** 2 / BATCH, 1 / BATCH)

    def test_parity_is_lr_alpha_g_and_lr_g(self):
        self.assert_step(True, ALPHA, 1.0)

    def test_parity_bias_step_is_the_batch_mean(self):
        # DFA's bias step is -lr * mean_b(share_b); under the summed loss the bias gradient is the
        # sum, so parity divides it by B. The default's mean loss gives the same step.
        steps = []
        for parity in (False, True):
            model = build()
            line, onehot = batch()
            b0 = biases(model)
            run(model, config(parity=parity), line, onehot)
            steps.append([b1 - b for b, b1 in zip(b0, biases(model))])
        for off, on in zip(*steps):
            torch.testing.assert_close(on, off, rtol=1e-5, atol=1e-8)

    def test_grad_norm_clip_sees_each_sequences_own_gradient(self):
        # The clip norm is each sequence's own gradient (weights and bias share): B times the
        # default's. A clip that does not bind leaves the parity step exactly as without it.
        norms = {}
        for parity in (False, True):
            model = build(retain_sequence_bias_grads=True)
            seen = []
            original = model.clip_grad_norm_per_sequence
            model.clip_grad_norm_per_sequence = lambda m: seen.append(original(m)) or seen[-1]
            line, onehot = batch()
            run(model, {**config(parity=parity), "grad_norm_clip": 1e9}, line, onehot)
            norms[parity] = seen[0]
            if parity:
                plain = build(retain_sequence_bias_grads=True)
                run(plain, config(parity=True), line, onehot)
                for a, b in zip(weights(model) + biases(model), weights(plain) + biases(plain)):
                    if a is not None:
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(norms[True], BATCH * norms[False], rtol=1e-5, atol=0)

    def test_explicit_false_is_bit_identical_to_absent(self):
        line, onehot = batch(length=5, seed=3)
        results = []
        for cfg in (config(), {k: v for k, v in config().items() if k != "backprop_step_parity"}):
            model = build()
            for _ in range(3):
                _, loss, *_ = run(model, cfg, line, onehot)
            results.append((loss, weights(model), biases(model)))
        self.assertEqual(results[0][0], results[1][0])
        for a, b in zip(results[0][1] + results[0][2], results[1][1] + results[1][2]):
            if a is not None:
                self.assertTrue(torch.equal(a, b))


class ParityMatchesDfaTest(unittest.TestCase):
    """The same tiny model, one step from the same weights, under DFA and under parity backprop.
    i2o's DFA gradient is its exact gradient, so its step (weights and bias) must agree; layer 2's
    fast entries take the B_fast write under both (--decouple_fast_feedback 2), so they must agree
    too. Without parity, i2o's backprop step is 1/B of DFA's."""

    def setUp(self):
        torch.set_num_threads(1)

    def one_step(self, updater, parity):
        model = build(updater=updater, decouple_fast_feedback=[2])
        line, onehot = batch()
        w0, b0 = weights(model), biases(model)
        run(model, config(updater, parity), line, onehot)
        return model, w0, b0, weights(model), biases(model)

    def test_readout_and_fast_write_match_dfa(self):
        dfa = self.one_step("dfa", False)
        bp = self.one_step("backprop", True)
        i2o = len(dfa[0].trained_layers()) - 1
        torch.testing.assert_close(bp[3][i2o], dfa[3][i2o], rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(bp[4][i2o], dfa[4][i2o], rtol=1e-5, atol=1e-7)
        mask = dfa[0].linear_layers[2].ephemeral_mask.unsqueeze(0).expand_as(dfa[3][2])
        torch.testing.assert_close(bp[3][2][mask], dfa[3][2][mask], rtol=1e-5, atol=1e-7)
        self.assertGreater(float(dfa[3][2][mask].abs().max()), 0)

    def test_fast_step_magnitude_matches_dfa_only_with_parity(self):
        dfa = self.one_step("dfa", False)
        layer = dfa[0].linear_layers[2]
        mask = layer.ephemeral_mask.unsqueeze(0).expand_as(dfa[3][2])
        dfa_norm = float(dfa[3][2][mask].norm())
        # Parity's B_fast write has exactly DFA's size.
        bp = self.one_step("backprop", True)
        self.assertAlmostEqual(float(bp[3][2][mask].norm()) / dfa_norm, 1.0, places=5)
        # The default's readout step is 1/B of DFA's (the same exact gradient, mean loss).
        i2o = len(dfa[0].trained_layers()) - 1
        model = build()
        line, onehot = batch()
        w0 = weights(model)
        run(model, config("backprop", False), line, onehot)
        ratio = float((weights(model)[i2o] - w0[i2o]).norm() / (dfa[3][i2o] - dfa[1][i2o]).norm())
        self.assertAlmostEqual(ratio, 1 / BATCH, places=5)


class HostAgnosticArmTest(unittest.TestCase):
    """--updater backprop --backprop_step_parity --decouple_fast_feedback L: backprop for the slow
    weights, DC2's B_fast write for layer L's fast entries."""

    def setUp(self):
        torch.set_num_threads(1)

    def test_slow_steps_unchanged_and_fast_write_is_b_fast(self):
        line, onehot = batch()
        plain, decoupled = build(), build(decouple_fast_feedback=[1, 2])
        w0 = weights(plain)
        run(plain, config(parity=True), line, onehot)
        # The fast write's error, from the output error at the pre-step weights.
        reference = copy.deepcopy(decoupled)
        reference.start_sequence_wipe()
        output, _ = reference(onehot[:, 0], reference.initHidden(BATCH))
        error = (torch.softmax(output, 1) - onehot[:, 1]).detach()
        run(decoupled, config(parity=True), line, onehot)
        for index, layer in enumerate(decoupled.trained_layers()):
            mask = layer.ephemeral_mask.unsqueeze(0).expand_as(layer.per_sample_weights)
            a, b = weights(plain)[index], weights(decoupled)[index]
            # Every slow entry and bias takes plain parity backprop's step.
            torch.testing.assert_close(b[~mask], a[~mask], rtol=0, atol=0)
            torch.testing.assert_close(biases(decoupled)[index], biases(plain)[index], rtol=0, atol=0)
            if index in (1, 2):
                b_fast = decoupled_fast_feedback(len(CHARSET), layer.out_features, index)
                self.assertTrue(torch.equal(layer.fast_feedback_weights, b_fast))
                x = reference.trained_layers()[index].in_traces.data
                write = -LR * ALPHA * (error @ b_fast).unsqueeze(2) * x.unsqueeze(1)
                torch.testing.assert_close(b[mask], write[mask], rtol=1e-5, atol=1e-7)
                self.assertFalse(torch.allclose(b[mask], a[mask]))
            else:
                # Layer 0 (and i2h) keep their backprop writes.
                torch.testing.assert_close(b[mask], a[mask], rtol=0, atol=0)
        self.assertFalse(torch.equal(weights(decoupled)[2], w0[2]))

    def test_fast_state_is_a_constant_to_the_slow_gradient(self):
        # Over a 3-step sequence, each step's slow step is the gradient of that step's loss with
        # the weights as they stand (the earlier fast writes included) as constants: the weights
        # are leaves written outside autograd, so nothing differentiates through a write.
        line, onehot = batch(length=4, seed=1)
        model = build(decouple_fast_feedback=[0, 1, 2])
        run(model, config(parity=True), line, onehot)
        for layer in model.trained_layers():
            self.assertTrue(layer.per_sample_weights.is_leaf)
            self.assertIsNone(layer.per_sample_weights.grad_fn)
        # Replay by hand: the step-3 slow gradient from the step-3 weights alone.
        replay = build(decouple_fast_feedback=[0, 1, 2])
        cfg = config(parity=True)
        run(replay, cfg, line[:, :3], onehot[:, :3])  # steps 1-2 of the same sequence
        snapshot = copy.deepcopy(replay)
        # Step 3 by hand, without a new wipe: the gradient at the weights as they stand.
        hidden = snapshot.initHidden(BATCH)
        snapshot.zero_grad()
        output, _ = snapshot(onehot[:, 2], hidden)
        loss = F.cross_entropy(output, onehot[:, 3], reduction="none").sum()
        g = torch.autograd.grad(loss, snapshot.linear_layers[1].per_sample_weights)[0]
        layer = model.linear_layers[1]
        mask = layer.ephemeral_mask.unsqueeze(0).expand_as(g)
        expected_slow = (replay.linear_layers[1].per_sample_weights.detach() - LR * g)[~mask]
        torch.testing.assert_close(layer.per_sample_weights.detach()[~mask], expected_slow,
                                   rtol=1e-5, atol=1e-7)


class ParityCliTest(unittest.TestCase):
    def parse(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return parse_args(BASE_ARGS + list(extra))

    def refused(self, *argv):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(list(argv))

    def test_default_off(self):
        self.assertFalse(self.parse().backprop_step_parity)

    def test_accepted(self):
        args = self.parse("--backprop_step_parity", "true", "--decouple_fast_feedback", "0,1,2")
        self.assertTrue(args.backprop_step_parity)

    def test_refused_combinations(self):
        ephemeral = ["--model_type", "ephemeral", "--track", "false"]
        self.refused(*ephemeral, "--updater", "dfa", "--backprop_step_parity", "true")
        self.refused(*ephemeral, "--updater", "bptt", "--backprop_step_parity", "true")
        self.refused("--model_type", "rnn", "--updater", "backprop", "--backprop_step_parity", "true",
                     "--track", "false")
        # B_fast writes in a backprop host need the parity step (else alpha^2/B on every write).
        self.refused(*BASE_ARGS[2:], *ephemeral, "--decouple_fast_feedback", "2")
        self.refused(*ephemeral, "--updater", "bptt", "--decouple_fast_feedback", "2")
        self.refused(*BASE_ARGS, "--backprop_step_parity", "true", "--decouple_fast_feedback", "2",
                     "--grad_norm_clip", "1")
        self.refused(*BASE_ARGS, "--backprop_step_parity", "true", "--fast_write_from_readout", "2")

    def test_model_refuses_bptt_decoupling(self):
        with self.assertRaises(ValueError):
            build(updater="bptt", decouple_fast_feedback=[2])
        with self.assertRaises(ValueError):
            build(updater="backprop", fast_write_from_readout=[2])


if __name__ == "__main__":
    unittest.main()
