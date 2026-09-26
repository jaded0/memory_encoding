"""--grad_norm_clip on the ephemeral model: conventional gradient-norm clipping applied to each
sequence's raw gradient, before plasticity and the two clamps."""
import contextlib
import copy
import io
import unittest

import torch
import torch.nn.functional as F

from ephemeral_model import EphemeralRNN
from reproducibility import seed_everything
from train import train

CHARSET = list("abcd")
HIDDEN = 4
LEARNING_RATE = 0.1
PLASTICITY = 3.0
SEQUENCES = torch.tensor([[0, 1, 2, 3, 1], [3, 2, 0, 1, 2], [1, 1, 3, 0, 2]])


def build(updater, batch_size=3, plasticity=PLASTICITY, retain=False, weight_clamp=0,
          enable_recurrence=True):
    seed_everything(7, deterministic=True)
    with contextlib.redirect_stdout(io.StringIO()):
        return EphemeralRNN(len(CHARSET), HIDDEN, len(CHARSET), 2, CHARSET, unit_norm_weights=False,
                            weight_clamp=weight_clamp, updater=updater, plasticity=plasticity,
                            batch_size=batch_size, forget_rate=0.25, ephemeral_fraction=0.5,
                            enable_recurrence=enable_recurrence, retain_sequence_bias_grads=retain)


def run(model, updater, sequences, grad_norm_clip=0, update_clamp=0):
    config = {"updater": updater, "criterion": torch.nn.CrossEntropyLoss(reduction="none"),
              "input_mode": "last_one", "pe_matrix": None, "learning_rate": LEARNING_RATE,
              "ephemeral_update_clamp": update_clamp, "grad_norm_clip": grad_norm_clip,
              "plasticity": PLASTICITY}
    onehot = F.one_hot(sequences, len(CHARSET)).float()
    with contextlib.redirect_stdout(io.StringIO()):
        train(sequences, onehot, model, config, {"training_instance": 0})


def trained_tensors(model):
    return {f"{index}.{name}": tensor.detach().clone()
            for index, layer in enumerate(model.trained_layers())
            for name, tensor in (("weights", layer.per_sample_weights), ("bias", layer.bias))}


def backward_one_step(model, sequences, steps=1):
    """Forward `steps` tokens (hidden not detached) and backprop the batch-mean loss."""
    model.start_sequence_wipe()
    hidden = model.initHidden(sequences.shape[0])
    onehot = F.one_hot(sequences, len(CHARSET)).float()
    loss = 0
    for step in range(steps):
        output, hidden = model(onehot[:, step], hidden)
        loss = loss + F.cross_entropy(output, sequences[:, step + 1], reduction="none")
    model.zero_grad()
    loss.mean().backward()


def dfa_populate(model, sequences):
    """One DFA step's forward and populated gradients; returns nothing, leaves them on the model."""
    model.start_sequence_wipe()
    onehot = F.one_hot(sequences, len(CHARSET)).float()
    output, _ = model(onehot[:, 0], model.initHidden(sequences.shape[0]))
    output_error = torch.softmax(output, 1) - onehot[:, 1]
    model.clear_dfa_gradients()
    for layer in model.trained_layers():
        layer.populate_dfa_gradients(output_error)


def per_sequence_gradients(model):
    """[(weight grad [B, out, in], bias share [B, out])] for every layer, cloned."""
    return [(layer.per_sample_weights.grad.clone(), layer.sequence_bias_grads().clone())
            for layer in model.trained_layers() if layer.per_sample_weights.grad is not None]


def torch_clip_reference(tensors, max_norm):
    references = [torch.nn.Parameter(torch.zeros_like(tensor)) for tensor in tensors]
    for reference, tensor in zip(references, tensors):
        reference.grad = tensor.clone()
    total = torch.nn.utils.clip_grad_norm_(references, max_norm)
    return total, [reference.grad for reference in references]


class GradNormClipTest(unittest.TestCase):
    def test_a_threshold_that_never_binds_is_bit_identical_to_no_clip(self):
        for updater in ("dfa", "backprop", "bptt"):
            with self.subTest(updater=updater):
                off = build(updater)
                on = build(updater, retain=updater != "dfa")
                run(off, updater, SEQUENCES)
                run(on, updater, SEQUENCES, grad_norm_clip=1e30)
                for name, tensor in trained_tensors(off).items():
                    torch.testing.assert_close(trained_tensors(on)[name], tensor, rtol=0, atol=0, msg=name)
                self.assertEqual(on.grad_clip_stats.summary()["grad_norm_clip_fraction"], 0)

    def test_batch_of_one_is_torch_clip_grad_norm_over_every_trained_gradient(self):
        # Per-step backprop never trains the forked i2h; BPTT with recurrence does.
        cases = (("backprop", 1, False, 3), ("bptt", 3, True, 4))
        for updater, steps, recurrence, layers_with_grads in cases:
            with self.subTest(updater=updater):
                model = build(updater, batch_size=1, retain=True, enable_recurrence=recurrence)
                backward_one_step(model, SEQUENCES[:1], steps)
                grads = [p.grad for layer in model.trained_layers()
                         for p in (layer.per_sample_weights, layer.bias) if p.grad is not None]
                self.assertEqual(len(grads), 2 * layers_with_grads)
                total, expected = torch_clip_reference(grads, 0.05)
                self.assertGreater(total.item(), 0.05)  # the clip binds
                norms = model.clip_grad_norm_per_sequence(0.05)
                torch.testing.assert_close(norms, total.reshape(1), rtol=1e-6, atol=0)
                clipped = [p.grad for layer in model.trained_layers()
                           for p in (layer.per_sample_weights, layer.bias) if p.grad is not None]
                for actual, reference in zip(clipped, expected):
                    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-8)

    def test_dfa_batch_of_one_clips_weight_gradients_and_bias_errors_together(self):
        model = build("dfa", batch_size=1)
        dfa_populate(model, SEQUENCES[:1])
        tensors = [tensor for pair in per_sequence_gradients(model) for tensor in pair]
        total, expected = torch_clip_reference(tensors, 0.05)
        self.assertGreater(total.item(), 0.05)
        model.clip_grad_norm_per_sequence(0.05)
        actual = [tensor for pair in per_sequence_gradients(model) for tensor in pair]
        for got, reference in zip(actual, expected):
            torch.testing.assert_close(got, reference, rtol=1e-5, atol=1e-8)

    def test_each_sequence_is_clipped_to_the_threshold_independently(self):
        model = build("dfa")
        dfa_populate(model, SEQUENCES)
        before = per_sequence_gradients(model)
        norms = torch.stack([sum(weight[b].square().sum() + bias[b].square().sum() for weight, bias in before)
                             for b in range(SEQUENCES.shape[0])]).sqrt()
        threshold = norms.median().item()  # binds for one sequence, not for another
        self.assertTrue((norms > threshold).any() and (norms < threshold).any())
        output_error = model.i2o.sequence_bias_grads()
        returned = model.clip_grad_norm_per_sequence(threshold)
        torch.testing.assert_close(returned, norms, rtol=1e-6, atol=0)
        after = per_sequence_gradients(model)
        for b, norm in enumerate(norms):
            scale = min(1.0, threshold / (norm.item() + 1e-6))
            clipped_norm = sum(weight[b].square().sum() + bias[b].square().sum() for weight, bias in after).sqrt()
            if norm > threshold:
                self.assertAlmostEqual(clipped_norm.item(), threshold, places=5)
            for (weight, bias), (old_weight, old_bias) in zip(after, before):
                # Direction and the ratios between layers are preserved; unclipped sequences are untouched.
                tolerance = 0 if norm <= threshold else 1e-6
                torch.testing.assert_close(weight[b], scale * old_weight[b], rtol=tolerance, atol=0)
                torch.testing.assert_close(bias[b], scale * old_bias[b], rtol=tolerance, atol=0)
        # i2o's projected error was replaced, not modified in place (it is train.py's output_error).
        torch.testing.assert_close(output_error, before[-1][1], rtol=0, atol=0)
        stats = model.grad_clip_stats.summary()
        self.assertAlmostEqual(stats["grad_norm_clip_fraction"], (norms > threshold).float().mean().item())
        self.assertAlmostEqual(stats["grad_norm_max"], norms.max().item(), places=5)

    def test_the_norm_is_of_the_raw_gradient_not_the_plasticity_scaled_one(self):
        norms = []
        for plasticity in (1.0, 1000.0):
            model = build("backprop", plasticity=plasticity, retain=True)
            backward_one_step(model, SEQUENCES)
            norms.append(model.clip_grad_norm_per_sequence(1e30))
        torch.testing.assert_close(norms[0], norms[1], rtol=0, atol=0)

    def test_dfa_order_is_clip_then_plasticity_then_update_clamp_then_weight_clamp_then_forget(self):
        grad_clip, update_clamp, weight_clamp = 0.05, 0.004, 0.3
        model = build("dfa", weight_clamp=weight_clamp)
        reference = copy.deepcopy(model)
        run(model, "dfa", SEQUENCES[:, :2], grad_norm_clip=grad_clip, update_clamp=update_clamp)

        reference.start_sequence_wipe()
        start = {name: tensor for name, tensor in trained_tensors(reference).items()}
        dfa_populate(reference, SEQUENCES[:, :2])
        gradients = per_sequence_gradients(reference)
        norms = torch.stack([sum(w[b].square().sum() + e[b].square().sum() for w, e in gradients)
                             for b in range(SEQUENCES.shape[0])]).sqrt()
        scale = (grad_clip / (norms + 1e-6)).clamp(max=1)
        self.assertTrue((scale < 1).any())
        for index, (layer, (weight_grad, error)) in enumerate(zip(reference.trained_layers(), gradients)):
            update = -scale.view(-1, 1, 1) * weight_grad
            if not layer.is_last_layer:
                update = update * layer.plasticity
                update = torch.where(layer.ephemeral_mask, update.clamp(-update_clamp, update_clamp), update)
            weights = (start[f"{index}.weights"] + LEARNING_RATE * update).clamp(-weight_clamp, weight_clamp)
            weights = weights * (1 - 0.25 * layer.ephemeral_mask)
            bias = start[f"{index}.bias"] - LEARNING_RATE * (scale.unsqueeze(1) * error).mean(0)
            torch.testing.assert_close(model.trained_layers()[index].per_sample_weights, weights,
                                       rtol=1e-5, atol=1e-7, msg=f"layer {index} weights")
            torch.testing.assert_close(model.trained_layers()[index].bias, bias,
                                       rtol=1e-5, atol=1e-7, msg=f"layer {index} bias")

    def test_backprop_and_bptt_refuse_to_clip_without_retained_bias_shares(self):
        for updater in ("backprop", "bptt"):
            with self.subTest(updater=updater):
                model = build(updater)
                with self.assertRaises(RuntimeError):
                    run(model, updater, SEQUENCES, grad_norm_clip=1.0)


if __name__ == "__main__":
    unittest.main()
