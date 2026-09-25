#!/usr/bin/env python3
"""Compare CPU throughput of the main and standalone DFA implementations.

The core paths consume pre-generated batches and perform the same sequence wipe,
forward passes, DFA outer products, updates, and forgetting. ``main_native`` also
includes train_batch's loss/autograd and metric bookkeeping.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ephemeral_model import EphemeralRNN
from scratch.dfa_forked_output_activation.experiment import (
    Config as ScratchConfig,
    ForkedEphemeralRNN,
    make_batch,
)
from train import train_batch


PATHS = ("main_native", "main_core", "scratch_core")


def build_models(cfg: ScratchConfig, seed: int):
    torch.manual_seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        main = EphemeralRNN(
            input_size=cfg.vocab_size,
            hidden_size=cfg.hidden_size,
            output_size=cfg.vocab_size,
            num_layers=cfg.num_layers,
            charset=list(range(cfg.vocab_size)),
            residual_connection=False,
            unit_norm_weights=cfg.unit_norm_weights,
            weight_clamp=cfg.weight_clamp,
            updater="dfa",
            plasticity=cfg.plasticity,
            batch_size=cfg.batch_size,
            forget_rate=cfg.forget_rate,
            ephemeral_fraction=cfg.ephemeral_fraction,
            enable_recurrence=False,
        )
    torch.manual_seed(seed + 1)
    scratch = ForkedEphemeralRNN("identity", cfg)
    main_layers = [*main.linear_layers, main.i2h, main.i2o]
    scratch_layers = [*scratch.trunk, scratch.i2h, scratch.i2o]
    with torch.no_grad():
        for source, target in zip(main_layers, scratch_layers, strict=True):
            for name in ("weight", "bias", "feedback_weights", "per_sample_weights",
                         "ephemeral_mask", "plasticity"):
                getattr(target, name).copy_(getattr(source, name))
    return main, scratch


def generate_batches(task: str, cfg: ScratchConfig, count: int, seed: int):
    rng = np.random.default_rng(seed)
    batches = []
    for _ in range(count):
        x, targets, _ = make_batch(task, cfg, rng)
        sequence = torch.cat((x[:, :1], targets), dim=1)
        batches.append((sequence, F.one_hot(sequence, cfg.vocab_size).float()))
    return batches


def main_native_batch(model, batch, cfg, state):
    sequence, onehot = batch
    train_batch(sequence, onehot, model, {
        "updater": "dfa",
        "criterion": torch.nn.CrossEntropyLoss(reduction="none"),
        "input_mode": "last_one",
        "pe_matrix": None,
        "learning_rate": cfg.learning_rate,
        "ephemeral_update_clamp": cfg.ephemeral_update_clamp,
    }, state)


@torch.no_grad()
def main_core_batch(model, batch, cfg, state):
    _, onehot = batch
    model.start_sequence_wipe()
    hidden = model.initHidden(cfg.batch_size)
    layers = [*model.linear_layers, model.i2h, model.i2o]
    for step in range(onehot.shape[1] - 1):
        logits, hidden = model(onehot[:, step], hidden)
        output_error = torch.softmax(logits, 1) - onehot[:, step + 1]
        model.clear_dfa_gradients()
        for layer in layers:
            layer.populate_dfa_gradients(output_error)
        for layer in layers:
            layer.apply_update(cfg.learning_rate, cfg.ephemeral_update_clamp, state)
        model.apply_forget_step()
        model.clear_dfa_gradients()


@torch.no_grad()
def scratch_core_batch(model, batch, cfg, _state):
    _, onehot = batch
    model.start_sequence_wipe()
    hidden = torch.zeros(cfg.batch_size, cfg.hidden_size)
    layers = [*model.trunk, model.i2h, model.i2o]
    for step in range(onehot.shape[1] - 1):
        logits, hidden = model(onehot[:, step], hidden)
        output_error = torch.softmax(logits, 1) - onehot[:, step + 1]
        for layer in layers:
            layer.populate_dfa_gradients(output_error)
        for layer in layers:
            gradient = -layer._gradient * layer.plasticity.unsqueeze(0)
            if layer.update_clamp > 0 and not layer.is_last_layer:
                mask = layer.ephemeral_mask.unsqueeze(0)
                gradient = torch.where(mask, gradient.clamp(-layer.update_clamp,
                                                             layer.update_clamp), gradient)
            layer.per_sample_weights.add_(gradient, alpha=cfg.learning_rate)
            layer.bias.add_(-cfg.learning_rate * layer._projected_error.mean(0))
            if layer.unit_norm_weights:
                norms = torch.linalg.vector_norm(layer.per_sample_weights,
                                                   dim=(1, 2), keepdim=True)
                layer.per_sample_weights.div_(norms + 1e-6)
            if layer.weight_clamp:
                layer.per_sample_weights.clamp_(-layer.weight_clamp, layer.weight_clamp)
            layer.per_sample_weights.mul_(1 - layer.forget_rate * layer.ephemeral_mask)


RUNNERS = {
    "main_native": lambda pair: (pair[0], main_native_batch),
    "main_core": lambda pair: (pair[0], main_core_batch),
    "scratch_core": lambda pair: (pair[1], scratch_core_batch),
}


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def benchmark(path, batches, cfg, warmup, iterations, repeats, seed):
    elapsed = []
    state = {"training_instance": 0, "log_norms_now": False}
    for repeat in range(repeats):
        pair = build_models(cfg, seed + repeat)
        model, runner = RUNNERS[path](pair)
        for index in range(warmup):
            runner(model, batches[index % len(batches)], cfg, state)
        gc.collect()
        gc.disable()
        started = time.perf_counter()
        for index in range(iterations):
            runner(model, batches[(warmup + index) % len(batches)], cfg, state)
        elapsed.append(time.perf_counter() - started)
        gc.enable()
        if not torch.isfinite(model.i2o.per_sample_weights).all():
            raise FloatingPointError(f"{path} produced non-finite weights")
    steps = batches[0][0].shape[1] - 1
    rates = [iterations * cfg.batch_size / value for value in elapsed]
    token_rates = [rate * steps for rate in rates]
    return {
        "path": path,
        "elapsed_seconds": elapsed,
        "sequences_per_second_median": statistics.median(rates),
        "sequences_per_second_iqr": percentile(rates, .75) - percentile(rates, .25),
        "tokens_per_second_median": statistics.median(token_rates),
        "tokens_per_second_iqr": percentile(token_rates, .75) - percentile(token_rates, .25),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("repeated_copy", "palindrome2"),
                        default="repeated_copy")
    parser.add_argument("--paths", nargs="+", choices=PATHS, default=PATHS)
    parser.add_argument("--warmup-batches", type=int, default=50)
    parser.add_argument("--timed-batches", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cfg = replace(ScratchConfig(), train_batches=args.timed_batches)
    batches = generate_batches(args.task, cfg, max(args.warmup_batches,
                                                    args.timed_batches), args.seed)
    results = [benchmark(path, batches, cfg, args.warmup_batches,
                         args.timed_batches, args.repeats, args.seed)
               for path in args.paths]
    payload = {
        "configuration": {
            "task": args.task,
            "warmup_batches": args.warmup_batches,
            "timed_batches": args.timed_batches,
            "repeats": args.repeats,
            "batch_size": cfg.batch_size,
            "hidden_size": cfg.hidden_size,
            "num_layers": cfg.num_layers,
            "vocab_size": cfg.vocab_size,
            "threads": torch.get_num_threads(),
        },
        "results": results,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
