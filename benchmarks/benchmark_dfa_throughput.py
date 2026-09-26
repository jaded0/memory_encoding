#!/usr/bin/env python3
"""Compare throughput of the main and standalone DFA implementations.

The core paths consume pre-generated batches and perform the same sequence wipe,
forward passes, DFA outer products, updates, and forgetting. ``main_native`` also
includes train_batch's loss/autograd and metric bookkeeping.

Defaults reproduce the committed single-thread CPU runs. ``--device cuda`` with the
model-size flags times the same paths at production scale, and ``--profile`` prints
a torch.profiler operator table for each path.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import os
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


PATHS = ("main_native", "main_core", "scratch_core", "fused_core", "fused_native")
DEFAULT_PATHS = PATHS[:3]  # the fused paths need torch.compile (Triton on CUDA: sm_70 or newer)


def build_models(cfg: ScratchConfig, seed: int, device: torch.device):
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
    return main.to(device), scratch.to(device)


def palindrome3_sequences(cfg: ScratchConfig, rng: np.random.Generator):
    """Three symbols, a fixed middle token, then the reverse: seven tokens, as in the
    project's 3_palindrome_dataset_vary_length (whose nine-character charset is
    matched by --symbols 7)."""
    half = rng.integers(0, cfg.symbols, (cfg.batch_size, 3))
    middle = np.full((cfg.batch_size, 1), cfg.symbols + 1)
    return torch.tensor(np.concatenate((half, middle, half[:, ::-1]), axis=1))


def generate_batches(task: str, cfg: ScratchConfig, count: int, seed: int, device: torch.device):
    rng = np.random.default_rng(seed)
    batches = []
    for _ in range(count):
        if task == "palindrome3":
            sequence = palindrome3_sequences(cfg, rng)
        else:
            x, targets, _ = make_batch(task, cfg, rng)
            sequence = torch.cat((x[:, :1], targets), dim=1)
        sequence = sequence.to(device)
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
    hidden = torch.zeros(cfg.batch_size, cfg.hidden_size, device=onehot.device)
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


@torch.no_grad()
def fused_core_batch(model, batch, cfg, _state):
    """main_core through the production --fused_update step (EphemeralRNN.fused_dfa_step)."""
    _, onehot = batch
    model.start_sequence_wipe()
    hidden = model.initHidden(cfg.batch_size)
    for step in range(onehot.shape[1] - 1):
        logits, hidden = model(onehot[:, step], hidden)
        output_error = torch.softmax(logits, 1) - onehot[:, step + 1]
        model.fused_dfa_step(output_error, cfg.learning_rate, cfg.ephemeral_update_clamp)


def fused(model):
    model.enable_fused_update()
    return model


RUNNERS = {
    "main_native": lambda pair: (pair[0], main_native_batch),
    "main_core": lambda pair: (pair[0], main_core_batch),
    "scratch_core": lambda pair: (pair[1], scratch_core_batch),
    "fused_core": lambda pair: (fused(pair[0]), fused_core_batch),
    "fused_native": lambda pair: (fused(pair[0]), main_native_batch),
}
UNFUSED = {"fused_core": "main_core", "fused_native": "main_native"}


def fused_difference(path, batches, cfg, seed, device, count):
    """Largest relative difference in any layer's per_sample_weights between a fused path and its
    unfused counterpart after `count` batches from the same initialization."""
    state = {"training_instance": 0, "log_norms_now": False}
    reference, reference_runner = RUNNERS[UNFUSED[path]](build_models(cfg, seed, device))
    candidate, runner = RUNNERS[path](build_models(cfg, seed, device))
    for index in range(count):
        reference_runner(reference, batches[index % len(batches)], cfg, state)
        runner(candidate, batches[index % len(batches)], cfg, state)
    pairs = zip([*reference.linear_layers, reference.i2h, reference.i2o],
                [*candidate.linear_layers, candidate.i2h, candidate.i2o])
    return max(((a.per_sample_weights - b.per_sample_weights).abs().max()
                / a.per_sample_weights.abs().max()).item() for a, b in pairs)


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(path, batches, cfg, warmup, iterations, repeats, seed, device):
    elapsed = []
    state = {"training_instance": 0, "log_norms_now": False}
    for repeat in range(repeats):
        pair = build_models(cfg, seed + repeat, device)
        model, runner = RUNNERS[path](pair)
        for index in range(warmup):
            runner(model, batches[index % len(batches)], cfg, state)
        gc.collect()
        gc.disable()
        synchronize(device)
        started = time.perf_counter()
        for index in range(iterations):
            runner(model, batches[(warmup + index) % len(batches)], cfg, state)
        synchronize(device)
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
        "batches_per_second_median": statistics.median(rates) / cfg.batch_size,
        "sequences_per_second_median": statistics.median(rates),
        "sequences_per_second_iqr": percentile(rates, .75) - percentile(rates, .25),
        "tokens_per_second_median": statistics.median(token_rates),
        "tokens_per_second_iqr": percentile(token_rates, .75) - percentile(token_rates, .25),
    }


def measure_copy_bandwidth(device, megabytes=512, repeats=20):
    """Device memory bandwidth in bytes/s from a large tensor copy (read + write)."""
    source = torch.empty(megabytes * 2**20 // 4, device=device)
    target = torch.empty_like(source)
    target.copy_(source)
    synchronize(device)
    started = time.perf_counter()
    for _ in range(repeats):
        target.copy_(source)
    synchronize(device)
    return 2 * source.numel() * 4 * repeats / (time.perf_counter() - started)


def bandwidth_floor(cfg, steps, bandwidth):
    """Upper bound on batches/s if each step only read every per-sequence weight tensor for
    the forward pass and read and wrote it once for a fused update, and each batch's wipe
    read and wrote it once: 5 passes per step plus 2 per batch."""
    inner = cfg.vocab_size + cfg.hidden_size
    entries = cfg.batch_size * (cfg.num_layers * inner * inner + inner * cfg.hidden_size
                                + inner * cfg.vocab_size)
    bytes_per_batch = 4 * entries * (3 * steps + 2)
    return bandwidth / bytes_per_batch


def profile_path(path, batches, cfg, seed, device, count):
    from torch.profiler import ProfilerActivity, profile
    state = {"training_instance": 0, "log_norms_now": False}
    model, runner = RUNNERS[path](build_models(cfg, seed, device))
    for index in range(5):
        runner(model, batches[index % len(batches)], cfg, state)
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == "cuda" else [])
    synchronize(device)
    with profile(activities=activities) as profiler:
        for index in range(count):
            runner(model, batches[index % len(batches)], cfg, state)
        synchronize(device)
    sort_by = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    print(f"=== profile: {path}, {count} batches, sorted by {sort_by} ===")
    print(profiler.key_averages().table(sort_by=sort_by, row_limit=25))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("repeated_copy", "palindrome2", "palindrome3"),
                        default="repeated_copy")
    parser.add_argument("--paths", nargs="+", choices=PATHS, default=DEFAULT_PATHS)
    parser.add_argument("--warmup-batches", type=int, default=50)
    parser.add_argument("--timed-batches", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1, help="CPU threads (default 1)")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--symbols", type=int, help="Task symbols; vocab is symbols + 2")
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--plasticity", type=float)
    parser.add_argument("--forget-rate", type=float)
    parser.add_argument("--weight-clamp", type=float)
    parser.add_argument("--check-batches", type=int, default=0, metavar="BATCHES",
                        help="Report each fused path's weight difference from its unfused path after this many batches")
    parser.add_argument("--profile", type=int, default=0, metavar="BATCHES",
                        help="Also print a torch.profiler table over this many batches per path")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(args.deterministic)
    overrides = {name: value for name, value in {
        "symbols": args.symbols, "hidden_size": args.hidden_size,
        "num_layers": args.num_layers, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "plasticity": args.plasticity,
        "forget_rate": args.forget_rate, "weight_clamp": args.weight_clamp,
    }.items() if value is not None}
    cfg = replace(ScratchConfig(), train_batches=args.timed_batches, **overrides)
    batches = generate_batches(args.task, cfg, max(args.warmup_batches,
                                                    args.timed_batches), args.seed, device)
    results = [benchmark(path, batches, cfg, args.warmup_batches,
                         args.timed_batches, args.repeats, args.seed, device)
               for path in args.paths]
    steps = batches[0][0].shape[1] - 1
    bandwidth = measure_copy_bandwidth(device)
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
            "steps_per_sequence": steps,
            "learning_rate": cfg.learning_rate,
            "plasticity": cfg.plasticity,
            "forget_rate": cfg.forget_rate,
            "weight_clamp": cfg.weight_clamp,
            "threads": torch.get_num_threads(),
            "deterministic": args.deterministic,
            "device": str(device),
            "device_name": (torch.cuda.get_device_name(device) if device.type == "cuda"
                            else "cpu"),
            "torch": torch.__version__,
        },
        "copy_bandwidth_gb_per_second": bandwidth / 1e9,
        "bandwidth_floor_batches_per_second": bandwidth_floor(cfg, steps, bandwidth),
        "results": results,
    }
    for path in [path for path in args.paths if path in UNFUSED] if args.check_batches else ():
        payload[f"{path}_max_relative_weight_difference"] = fused_difference(
            path, batches, cfg, args.seed, device, args.check_batches)
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n")
    for path in args.paths if args.profile else ():
        profile_path(path, batches, cfg, args.seed, device, args.profile)


if __name__ == "__main__":
    main()
