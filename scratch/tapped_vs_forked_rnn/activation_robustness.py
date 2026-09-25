#!/usr/bin/env python3
"""Factorial robustness panel for tanh versus identity in a locked forked RNN."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import platform
import random
import shlex
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


ACTIVATIONS = ("tanh", "identity")
CLIP_MODES = ("clip_1.0", "no_clip")
INIT_MODES = ("identity", "orthogonal")
TASKS = ("delayed_copy", "two_palindrome")
METRICS = ("token_accuracy", "exact_sequence_accuracy", "recurrent_state_rms",
           "recurrent_state_max_abs", "logit_rms", "logit_max_abs")


@dataclass(frozen=True)
class Config:
    symbols: int = 4
    embed_dim: int = 16
    hidden_dim: int = 40
    batch_size: int = 64
    learning_rate: float = 0.003
    copy_steps: int = 1800
    palindrome_steps: int = 2600
    eval_every: int = 25
    eval_batches: int = 4
    fixed_eval_batches: int = 16
    stress_eval_batches: int = 8
    train_copy_lengths: tuple[int, int] = (2, 4)
    long_copy_lengths: tuple[int, int] = (5, 6)
    stress_copy_lengths: tuple[int, int] = (8, 10)
    train_pal_lengths: tuple[int, int] = (2, 3)
    long_pal_lengths: tuple[int, int] = (4, 5)
    stress_pal_lengths: tuple[int, int] = (6, 6)
    delay_range: tuple[int, int] = (1, 3)
    gradient_clip_norm: float = 1.0
    orthogonal_gain: float = 1.0


class Tokens:
    def __init__(self, symbols: int):
        self.symbols = symbols
        self.PAD = symbols
        self.BLANK = symbols + 1
        self.START = symbols + 2
        self.SEP = symbols + 3
        self.QUERY1 = symbols + 4
        self.QUERY2 = symbols + 5
        self.vocab_in = symbols + 6


class MemoryRNN(nn.Module):
    """Equal-module forked models; activation and W_state initialization vary."""

    def __init__(self, activation: str, init_mode: str, vocab_in: int,
                 outputs: int, cfg: Config):
        super().__init__()
        if activation not in ACTIVATIONS or init_mode not in INIT_MODES:
            raise ValueError((activation, init_mode))
        self.activation = activation
        self.init_mode = init_mode
        self.embedding = nn.Embedding(vocab_in, cfg.embed_dim)
        self.trunk1 = nn.Linear(cfg.embed_dim + cfg.hidden_dim, cfg.hidden_dim)
        self.trunk2 = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.W_state = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.W_out = nn.Linear(cfg.hidden_dim, outputs)
        self.hidden_dim = cfg.hidden_dim
        with torch.no_grad():
            if init_mode == "identity":
                self.W_state.weight.copy_(torch.eye(cfg.hidden_dim))
            else:
                # Gain 1 preserves Euclidean norm at initialization, the standard
                # neutral choice for an ungated recurrent transition.
                nn.init.orthogonal_(self.W_state.weight, gain=cfg.orthogonal_gain)
            self.W_state.bias.zero_()
            self.trunk2.weight.mul_(0.1)
            self.trunk2.bias.zero_()

    def forward(self, x: torch.Tensor, diagnostics: bool = False):
        batch, steps = x.shape
        h = torch.zeros(batch, self.hidden_dim, device=x.device)
        logits_by_step, states = [], []
        for t in range(steps):
            transformed = torch.relu(
                self.trunk1(torch.cat((self.embedding(x[:, t]), h), dim=-1)))
            z = h + torch.relu(self.trunk2(transformed))
            u = self.W_state(z)
            h = torch.tanh(u) if self.activation == "tanh" else u
            # Topology is locked: output reads z, not u or h_next.
            logits_by_step.append(self.W_out(z))
            if diagnostics:
                states.append(h)
        logits = torch.stack(logits_by_step, dim=1)
        return (logits, torch.stack(states, dim=1)) if diagnostics else logits


def make_batch(task: str, batch_size: int, length_range: tuple[int, int],
               cfg: Config, tok: Tokens, rng: np.random.Generator):
    sequences, answers = [], []
    for _ in range(batch_size):
        delay = int(rng.integers(cfg.delay_range[0], cfg.delay_range[1] + 1))
        if task == "delayed_copy":
            n = int(rng.integers(length_range[0], length_range[1] + 1))
            block = rng.integers(0, cfg.symbols, n).tolist()
            seq = [tok.START] + block + [tok.SEP] + [tok.BLANK] * delay + [tok.QUERY1]
            start = len(seq)
            seq += [tok.BLANK] * n
            ans = [(start + j, value) for j, value in enumerate(block)]
        else:
            n1 = int(rng.integers(length_range[0], length_range[1] + 1))
            n2 = int(rng.integers(length_range[0], length_range[1] + 1))
            b1 = rng.integers(0, cfg.symbols, n1).tolist()
            b2 = rng.integers(0, cfg.symbols, n2).tolist()
            seq = [tok.START] + b1 + [tok.SEP] + b2 + [tok.SEP]
            seq += [tok.BLANK] * delay + [tok.QUERY1]
            start1 = len(seq)
            seq += [tok.BLANK] * n1 + [tok.QUERY2]
            start2 = len(seq)
            seq += [tok.BLANK] * n2
            ans = ([(start1 + j, v) for j, v in enumerate(reversed(b1))]
                   + [(start2 + j, v) for j, v in enumerate(reversed(b2))])
        sequences.append(seq)
        answers.append(ans)
    max_len = max(map(len, sequences))
    x = torch.full((batch_size, max_len), tok.PAD, dtype=torch.long)
    y = torch.zeros((batch_size, max_len), dtype=torch.long)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    for i, (seq, ans) in enumerate(zip(sequences, answers)):
        x[i, :len(seq)] = torch.tensor(seq)
        for position, value in ans:
            y[i, position], mask[i, position] = value, True
    return x, y, mask


def masked_metrics(logits, y, mask):
    selected, targets = logits[mask], y[mask]
    return (nn.functional.cross_entropy(selected, targets),
            (selected.argmax(-1) == targets).float().mean().item())


class Magnitudes:
    def __init__(self):
        self.ssq = {"recurrent_state": 0.0, "logit": 0.0}
        self.count = {"recurrent_state": 0, "logit": 0}
        self.maximum = {"recurrent_state": 0.0, "logit": 0.0}

    def add(self, logits, states):
        for name, tensor in (("recurrent_state", states), ("logit", logits)):
            data = tensor.detach().double()
            self.ssq[name] += float((data * data).sum())
            self.count[name] += data.numel()
            self.maximum[name] = max(self.maximum[name], float(data.abs().max()))

    def result(self):
        out = {}
        for name in self.ssq:
            out[name + "_rms"] = math.sqrt(self.ssq[name] / self.count[name])
            out[name + "_max_abs"] = self.maximum[name]
        return out


@torch.no_grad()
def evaluate(model, task, lengths, cfg, tok, seed, batches):
    model.eval()
    rng = np.random.default_rng(seed)
    losses, correct, total, exact, samples = [], 0, 0, 0, 0
    magnitudes = Magnitudes()
    for batch_index in range(batches):
        x, y, mask = make_batch(task, cfg.batch_size, lengths, cfg, tok, rng)
        logits, states = model(x, diagnostics=True)
        if not torch.isfinite(logits).all() or not torch.isfinite(states).all():
            model.train()
            return {"status": "nonfinite", "nonfinite_batch": batch_index,
                    "loss": None, "token_accuracy": None,
                    "exact_sequence_accuracy": None, "batches_completed": batch_index,
                    **{metric: None for metric in METRICS[2:]}}
        loss, _ = masked_metrics(logits, y, mask)
        pred, matches = logits.argmax(-1), logits.argmax(-1) == y
        losses.append(loss.item())
        correct += int(matches[mask].sum())
        total += int(mask.sum())
        exact += int((matches | ~mask).all(dim=1).sum())
        samples += x.shape[0]
        magnitudes.add(logits, states)
    model.train()
    return {"status": "ok", "nonfinite_batch": None, "loss": float(np.mean(losses)),
            "token_accuracy": correct / total, "exact_sequence_accuracy": exact / samples,
            "batches_completed": batches, "samples": samples, "target_tokens": total,
            **magnitudes.result()}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def failed_run(activation, clip_mode, init_mode, task, seed, cfg, tok, reason,
               step, started, traceback_text=None):
    return {"activation": activation, "clip_mode": clip_mode, "init_mode": init_mode,
            "task": task, "seed": seed,
            "parameter_count": count_parameters(MemoryRNN(
                activation, init_mode, tok.vocab_in, cfg.symbols, cfg)),
            "steps_budget": cfg.copy_steps if task == "delayed_copy" else cfg.palindrome_steps,
            "steps_completed": 0 if step is None else max(0, step - 1), "status": "failed",
            "nonfinite_failure": "nonfinite" in reason.lower(), "failure_reason": reason,
            "failure_step": step, "traceback": traceback_text,
            "thresholds": {"0.90": None, "0.99": None}, "evaluations": {},
            "optimization": None, "runtime_seconds": time.perf_counter() - started,
            "curve": []}


def gradient_norm(parameters):
    norms = [torch.linalg.vector_norm(p.grad.detach().double()) for p in parameters
             if p.grad is not None]
    return float(torch.linalg.vector_norm(torch.stack(norms))) if norms else 0.0


def panel_specs(task, cfg):
    if task == "delayed_copy":
        return (("in_distribution", cfg.train_copy_lengths, cfg.eval_batches),
                ("longer_lengths", cfg.long_copy_lengths, cfg.eval_batches),
                ("stress_lengths_8_10", cfg.stress_copy_lengths, cfg.stress_eval_batches))
    return (("in_distribution", cfg.train_pal_lengths, cfg.eval_batches),
            ("longer_lengths", cfg.long_pal_lengths, cfg.eval_batches),
            ("fixed_block_length_3", (3, 3), cfg.fixed_eval_batches),
            ("fixed_block_length_4", (4, 4), cfg.fixed_eval_batches),
            ("stress_fixed_block_length_6", cfg.stress_pal_lengths, cfg.stress_eval_batches))


def train_one(activation, clip_mode, init_mode, task, seed, cfg, tok):
    started = time.perf_counter()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = MemoryRNN(activation, init_mode, tok.vocab_in, cfg.symbols, cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    offset = 0 if task == "delayed_copy" else 1_000_000
    rng = np.random.default_rng(10_000 + seed + offset)
    lengths = cfg.train_copy_lengths if task == "delayed_copy" else cfg.train_pal_lengths
    steps = cfg.copy_steps if task == "delayed_copy" else cfg.palindrome_steps
    curves, grad_norms, thresholds, clip_binds = [], [], {"0.90": None, "0.99": None}, 0
    for step in range(1, steps + 1):
        upper = min(lengths[1], 1 + (step - 1) // 200)
        batch_lengths = (min(lengths[0], upper), upper)
        x, y, mask = make_batch(task, cfg.batch_size, batch_lengths, cfg, tok, rng)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        if not torch.isfinite(logits).all():
            return failed_run(activation, clip_mode, init_mode, task, seed, cfg, tok,
                              "nonfinite training logits", step, started)
        loss, train_accuracy = masked_metrics(logits, y, mask)
        if not torch.isfinite(loss):
            return failed_run(activation, clip_mode, init_mode, task, seed, cfg, tok,
                              "nonfinite training loss", step, started)
        loss.backward()
        norm = gradient_norm(model.parameters())
        if not math.isfinite(norm):
            return failed_run(activation, clip_mode, init_mode, task, seed, cfg, tok,
                              "nonfinite gradient norm", step, started)
        grad_norms.append(norm)
        if clip_mode == "clip_1.0":
            clip_binds += norm > cfg.gradient_clip_norm
            nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            return failed_run(activation, clip_mode, init_mode, task, seed, cfg, tok,
                              "nonfinite parameter after optimizer step", step, started)
        if step == 1 or step % cfg.eval_every == 0 or step == steps:
            val = evaluate(model, task, lengths, cfg, tok, 20_000 + seed + offset,
                           cfg.eval_batches)
            if val["status"] != "ok":
                return failed_run(activation, clip_mode, init_mode, task, seed, cfg, tok,
                                  "nonfinite validation state/logit", step, started)
            curves.append({"step": step, "examples": step * cfg.batch_size,
                           "train_loss": loss.item(), "train_accuracy": train_accuracy,
                           "validation_loss": val["loss"],
                           "validation_accuracy": val["token_accuracy"],
                           "validation_exact_sequence_accuracy": val["exact_sequence_accuracy"]})
            for label, threshold in (("0.90", .90), ("0.99", .99)):
                if thresholds[label] is None and val["token_accuracy"] >= threshold:
                    thresholds[label] = {"step": step, "examples": step * cfg.batch_size}
    evaluations = {}
    for panel_index, (name, eval_lengths, batches) in enumerate(panel_specs(task, cfg)):
        evaluations[name] = evaluate(model, task, eval_lengths, cfg, tok,
                                     30_000 + panel_index * 100_000 + seed + offset, batches)
    optimization = {"gradient_norm_mean": statistics.mean(grad_norms),
                    "gradient_norm_std": statistics.stdev(grad_norms),
                    "gradient_norm_max": max(grad_norms), "clip_bind_steps": clip_binds,
                    "optimizer_steps": len(grad_norms),
                    "clip_bind_fraction": clip_binds / len(grad_norms)
                    if clip_mode == "clip_1.0" else None}
    return {"activation": activation, "clip_mode": clip_mode, "init_mode": init_mode,
            "task": task, "seed": seed, "parameter_count": count_parameters(model),
            "steps_budget": steps, "steps_completed": steps, "status": "ok",
            "nonfinite_failure": False, "failure_reason": None, "failure_step": None,
            "traceback": None, "thresholds": thresholds, "evaluations": evaluations,
            "optimization": optimization, "runtime_seconds": time.perf_counter() - started,
            "curve": curves}


def ci95(values):
    mean = statistics.mean(values)
    if len(values) < 2:
        return [mean, mean]
    critical = {11: 2.2010, 19: 2.0930}.get(len(values) - 1, 1.96)
    half = critical * statistics.stdev(values) / math.sqrt(len(values))
    return [mean - half, mean + half]


def summarize(runs, cfg):
    arm_rows, panel_rows, contrast_rows = [], [], []
    by_key = {(r["task"], r["init_mode"], r["clip_mode"], r["seed"],
               r["activation"]): r for r in runs}
    for task in TASKS:
        for init_mode in INIT_MODES:
            for clip_mode in CLIP_MODES:
                for activation in ACTIVATIONS:
                    group = [r for r in runs if r["task"] == task
                             and r["init_mode"] == init_mode and r["clip_mode"] == clip_mode
                             and r["activation"] == activation]
                    good = [r for r in group if r["status"] == "ok"]
                    arm_rows.append({"task": task, "init_mode": init_mode,
                                     "clip_mode": clip_mode, "activation": activation,
                                     "runs": len(group), "successful_runs": len(good),
                                     "failed_runs": len(group) - len(good),
                                     "nonfinite_failures": sum(r["nonfinite_failure"] for r in group),
                                     "failure_steps": [r["failure_step"] for r in group
                                                       if r["status"] != "ok"],
                                     "gradient_norm_mean": statistics.mean(
                                         r["optimization"]["gradient_norm_mean"] for r in good)
                                         if good else None,
                                     "gradient_norm_max_mean": statistics.mean(
                                         r["optimization"]["gradient_norm_max"] for r in good)
                                         if good else None,
                                     "clip_bind_fraction_mean": statistics.mean(
                                         r["optimization"]["clip_bind_fraction"] for r in good)
                                         if good and clip_mode == "clip_1.0" else None})
                    for name, _, _ in panel_specs(task, cfg):
                        finite = [r for r in good if r["evaluations"][name]["status"] == "ok"]
                        row = {"task": task, "init_mode": init_mode, "clip_mode": clip_mode,
                               "activation": activation, "panel": name, "runs": len(group),
                               "finite_evaluations": len(finite),
                               "nonfinite_evaluations": len(good) - len(finite)}
                        for metric in METRICS:
                            vals = [r["evaluations"][name][metric] for r in finite]
                            row[metric + "_mean"] = statistics.mean(vals) if vals else None
                            row[metric + "_std"] = (statistics.stdev(vals) if len(vals) > 1
                                                    else 0.0 if vals else None)
                        panel_rows.append(row)
                seeds = sorted({r["seed"] for r in runs if r["task"] == task})
                for name, _, _ in panel_specs(task, cfg):
                    for metric in METRICS:
                        diffs, used = [], []
                        for seed in seeds:
                            identity = by_key.get((task, init_mode, clip_mode, seed, "identity"))
                            tanh = by_key.get((task, init_mode, clip_mode, seed, "tanh"))
                            if (identity and tanh and identity["status"] == tanh["status"] == "ok"
                                    and identity["evaluations"][name]["status"] == "ok"
                                    and tanh["evaluations"][name]["status"] == "ok"):
                                diffs.append(identity["evaluations"][name][metric]
                                             - tanh["evaluations"][name][metric])
                                used.append(seed)
                        contrast_rows.append({"contrast": "identity_minus_tanh",
                                              "task": task, "init_mode": init_mode,
                                              "clip_mode": clip_mode, "panel": name,
                                              "metric": metric, "paired_seeds": len(diffs),
                                              "seed_ids": used, "differences_by_seed": diffs,
                                              "mean_difference": statistics.mean(diffs) if diffs else None,
                                              "std_difference": statistics.stdev(diffs)
                                              if len(diffs) > 1 else 0.0 if diffs else None,
                                              "ci95": ci95(diffs) if diffs else None,
                                              "identity_wins": sum(v > 0 for v in diffs),
                                              "ties": sum(v == 0 for v in diffs),
                                              "tanh_wins": sum(v < 0 for v in diffs)})
    return arm_rows, panel_rows, contrast_rows


def write_csv(path, rows):
    rows = [{k: json.dumps(v) if isinstance(v, (list, dict)) else v
             for k, v in row.items()} for row in rows]
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def validate(runs, seeds, cfg, paired_checks, parameter_counts):
    expected = {(task, init_mode, clip_mode, seed, activation) for task in TASKS
                for init_mode in INIT_MODES for clip_mode in CLIP_MODES
                for seed in range(seeds) for activation in ACTIVATIONS}
    observed = {(r["task"], r["init_mode"], r["clip_mode"], r["seed"], r["activation"])
                for r in runs}
    complete = all(set(r["evaluations"]) == {p[0] for p in panel_specs(r["task"], cfg)}
                   and all(all(metric in evaluation for metric in METRICS)
                           for evaluation in r["evaluations"].values())
                   for r in runs if r["status"] == "ok")
    checks = {"expected_run_count": len(expected), "observed_run_count": len(runs),
              "run_count_ok": len(runs) == len(expected),
              "duplicate_keys": len(runs) - len(observed),
              "missing_keys": [list(x) for x in sorted(expected - observed)],
              "unexpected_keys": [list(x) for x in sorted(observed - expected)],
              "parameter_counts_equal": len(set(parameter_counts.values())) == 1,
              "all_run_parameter_counts_match": all(
                  r["parameter_count"] == parameter_counts[(r["activation"], r["init_mode"])]
                  for r in runs), "paired_initializations_equal": all(paired_checks),
              "successful_outputs_complete": complete}
    checks["passed"] = (checks["run_count_ok"] and checks["duplicate_keys"] == 0
                        and not checks["missing_keys"] and not checks["unexpected_keys"]
                        and checks["parameter_counts_equal"]
                        and checks["all_run_parameter_counts_match"]
                        and checks["paired_initializations_equal"] and complete)
    return checks


def make_plot(panel_rows, out_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    rows = [r for r in panel_rows if r["panel"] == "in_distribution"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for axis, task in zip(axes, TASKS):
        task_rows = [r for r in rows if r["task"] == task]
        labels, values, colors = [], [], []
        for row in task_rows:
            labels.append(f"{row['init_mode'][:4]}\n{row['clip_mode']}\n{row['activation']}")
            values.append(row["token_accuracy_mean"])
            colors.append("#e1812c" if row["activation"] == "identity" else "#4c72b0")
        axis.bar(range(len(values)), values, color=colors)
        axis.set_xticks(range(len(labels)), labels, fontsize=7)
        axis.set_title(task.replace("_", " "))
        axis.set_ylim(0, 1.02)
    axes[0].set_ylabel("mean token accuracy")
    fig.tight_layout()
    fig.savefig(out_dir / "activation_robustness.png", dpi=150)
    plt.close(fig)


def worker(job):
    activation, clip_mode, init_mode, task, seed, cfg = job
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    tok = Tokens(cfg.symbols)
    started = time.perf_counter()
    try:
        return train_one(activation, clip_mode, init_mode, task, seed, cfg, tok)
    except Exception as exc:
        import traceback
        return failed_run(activation, clip_mode, init_mode, task, seed, cfg, tok,
                          f"worker exception: {type(exc).__name__}: {exc}", None, started,
                          traceback.format_exc())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("activation_robustness_results.json"))
    parser.add_argument("--copy-steps", type=int)
    parser.add_argument("--palindrome-steps", type=int)
    args = parser.parse_args()
    cfg = Config(**{k: v for k, v in (("copy_steps", args.copy_steps),
                                      ("palindrome_steps", args.palindrome_steps)) if v is not None})
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    tok = Tokens(cfg.symbols)
    parameter_counts = {(a, i): count_parameters(MemoryRNN(
        a, i, tok.vocab_in, cfg.symbols, cfg)) for a in ACTIVATIONS for i in INIT_MODES}
    paired_checks = []
    # Validate activation and clip cells start from byte-identical parameters for each init/seed.
    for seed in range(args.seeds):
        for init_mode in INIT_MODES:
            states = []
            for activation in ACTIVATIONS:
                torch.manual_seed(seed)
                states.append(MemoryRNN(activation, init_mode, tok.vocab_in,
                                        cfg.symbols, cfg).state_dict())
            paired_checks.append(all(torch.equal(states[0][key], states[1][key])
                                     for key in states[0]))
    jobs = [(activation, clip_mode, init_mode, task, seed, cfg) for task in TASKS
            for init_mode in INIT_MODES for clip_mode in CLIP_MODES
            for seed in range(args.seeds) for activation in ACTIVATIONS]
    started = time.perf_counter()
    if args.workers == 1:
        runs = [worker(job) for job in jobs]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            runs = []
            for run in executor.map(worker, jobs):
                runs.append(run)
                print(f"finished {run['task']} seed={run['seed']} {run['init_mode']} "
                      f"{run['clip_mode']} {run['activation']} status={run['status']} "
                      f"time={run['runtime_seconds']:.1f}s", flush=True)
    runs.sort(key=lambda r: (TASKS.index(r["task"]), INIT_MODES.index(r["init_mode"]),
                             CLIP_MODES.index(r["clip_mode"]), r["seed"],
                             ACTIVATIONS.index(r["activation"])))
    arm_rows, panel_rows, contrasts = summarize(runs, cfg)
    validation = validate(runs, args.seeds, cfg, paired_checks, parameter_counts)
    runtime = time.perf_counter() - started
    payload = {"metadata": {"command": shlex.join([sys.executable, *sys.argv]),
                             "python_version": platform.python_version(),
                             "torch_version": torch.__version__, "device": "cpu",
                             "deterministic_algorithms": True, "workers": args.workers,
                             "paired_batch_streams": True,
                             "paired_initializations_equal_within_init_seed": all(paired_checks),
                             "parameter_counts": {"/".join(k): v for k, v in parameter_counts.items()},
                             "locked_readout": "logits = W_out(z)", "full_bptt": True,
                             "optimizer": "Adam", "orthogonal_gain": cfg.orthogonal_gain,
                             "orthogonal_gain_rationale": "gain 1 preserves Euclidean norm at initialization; neutral standard choice for an ungated recurrent transition",
                             "config": asdict(cfg), "total_runtime_seconds": runtime},
               "validation": validation, "arm_summaries": arm_rows,
               "panel_summaries": panel_rows, "factorial_contrasts": contrasts, "runs": runs}
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    csv_path = args.output.with_name("activation_robustness_summary.csv")
    write_csv(csv_path, ([{"row_type": "arm_summary", **r} for r in arm_rows]
                         + [{"row_type": "evaluation_panel", **r} for r in panel_rows]
                         + [{"row_type": "factorial_contrast", **r} for r in contrasts]))
    make_plot(panel_rows, args.output.parent)
    print(json.dumps({"validation": validation, "arm_summaries": arm_rows}, indent=2))
    print(f"wrote {args.output} and {csv_path}; total runtime {runtime:.1f}s")
    if not validation["passed"]:
        raise SystemExit("validation failed")


if __name__ == "__main__":
    main()
