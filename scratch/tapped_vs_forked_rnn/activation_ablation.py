#!/usr/bin/env python3
"""Forked-readout recurrent-carrier activation ablation on standalone tasks."""

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


ACTIVATIONS = ("tanh", "relu", "identity", "softsign")
TASKS = ("delayed_copy", "two_palindrome")
METRICS = ("token_accuracy", "exact_sequence_accuracy")


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
    train_copy_lengths: tuple[int, int] = (2, 4)
    long_copy_lengths: tuple[int, int] = (5, 6)
    train_pal_lengths: tuple[int, int] = (2, 3)
    long_pal_lengths: tuple[int, int] = (4, 5)
    delay_range: tuple[int, int] = (1, 3)
    gradient_clip_norm: float = 1.0


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
    """Identical forked models; only the parameter-free state activation varies."""

    def __init__(self, activation: str, vocab_in: int, outputs: int, cfg: Config):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(activation)
        self.activation = activation
        self.embedding = nn.Embedding(vocab_in, cfg.embed_dim)
        self.trunk1 = nn.Linear(cfg.embed_dim + cfg.hidden_dim, cfg.hidden_dim)
        self.trunk2 = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.W_state = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.W_out = nn.Linear(cfg.hidden_dim, outputs)
        self.hidden_dim = cfg.hidden_dim
        with torch.no_grad():
            self.W_state.weight.copy_(torch.eye(cfg.hidden_dim))
            self.W_state.bias.zero_()
            self.trunk2.weight.mul_(0.1)
            self.trunk2.bias.zero_()

    def activate(self, u: torch.Tensor) -> torch.Tensor:
        if self.activation == "tanh":
            return torch.tanh(u)
        if self.activation == "relu":
            return torch.relu(u)
        if self.activation == "softsign":
            return nn.functional.softsign(u)
        return u

    def forward(self, x: torch.Tensor, diagnostics: bool = False):
        batch, steps = x.shape
        h = torch.zeros(batch, self.hidden_dim, device=x.device)
        logits_by_step = []
        h_by_step = [] if diagnostics else None
        for t in range(steps):
            transformed = torch.relu(
                self.trunk1(torch.cat((self.embedding(x[:, t]), h), dim=-1))
            )
            z = h + torch.relu(self.trunk2(transformed))
            u = self.W_state(z)
            h = self.activate(u)
            # Locked fork: immediate output reads z, never u or h_next.
            logits_by_step.append(self.W_out(z))
            if diagnostics:
                h_by_step.append(h)
        logits = torch.stack(logits_by_step, dim=1)
        if diagnostics:
            return logits, torch.stack(h_by_step, dim=1)
        return logits


def make_batch(task: str, batch_size: int, length_range: tuple[int, int],
               cfg: Config, tok: Tokens, rng: np.random.Generator):
    sequences: list[list[int]] = []
    answers: list[list[tuple[int, int]]] = []
    for _ in range(batch_size):
        delay = int(rng.integers(cfg.delay_range[0], cfg.delay_range[1] + 1))
        if task == "delayed_copy":
            n = int(rng.integers(length_range[0], length_range[1] + 1))
            block = rng.integers(0, cfg.symbols, n).tolist()
            seq = [tok.START] + block + [tok.SEP] + [tok.BLANK] * delay + [tok.QUERY1]
            start = len(seq)
            seq += [tok.BLANK] * n
            ans = [(start + j, value) for j, value in enumerate(block)]
        elif task == "two_palindrome":
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
            ans = ([(start1 + j, value) for j, value in enumerate(reversed(b1))]
                   + [(start2 + j, value) for j, value in enumerate(reversed(b2))])
        else:
            raise ValueError(task)
        sequences.append(seq)
        answers.append(ans)
    max_len = max(map(len, sequences))
    x = torch.full((batch_size, max_len), tok.PAD, dtype=torch.long)
    y = torch.zeros((batch_size, max_len), dtype=torch.long)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    for i, (seq, ans) in enumerate(zip(sequences, answers)):
        x[i, :len(seq)] = torch.tensor(seq)
        for position, value in ans:
            y[i, position] = value
            mask[i, position] = True
    return x, y, mask


def masked_metrics(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
    selected, targets = logits[mask], y[mask]
    loss = nn.functional.cross_entropy(selected, targets)
    return loss, (selected.argmax(-1) == targets).float().mean().item()


class Magnitudes:
    def __init__(self):
        self.state_sum_sq = 0.0
        self.state_count = 0
        self.state_max = 0.0
        self.logit_sum_sq = 0.0
        self.logit_count = 0
        self.logit_max = 0.0

    def add(self, logits: torch.Tensor, states: torch.Tensor) -> None:
        for tensor, prefix in ((states, "state"), (logits, "logit")):
            data = tensor.detach().double()
            setattr(self, prefix + "_sum_sq", getattr(self, prefix + "_sum_sq")
                    + float((data * data).sum()))
            setattr(self, prefix + "_count", getattr(self, prefix + "_count") + data.numel())
            setattr(self, prefix + "_max", max(getattr(self, prefix + "_max"),
                                                float(data.abs().max())))

    def result(self) -> dict[str, float]:
        return {
            "recurrent_state_rms": math.sqrt(self.state_sum_sq / self.state_count),
            "recurrent_state_max_abs": self.state_max,
            "logit_rms": math.sqrt(self.logit_sum_sq / self.logit_count),
            "logit_max_abs": self.logit_max,
        }


@torch.no_grad()
def evaluate(model: MemoryRNN, task: str, lengths: tuple[int, int], cfg: Config,
             tok: Tokens, seed: int, batches: int | None = None) -> dict:
    model.eval()
    rng = np.random.default_rng(seed)
    losses, correct, total, exact, samples = [], 0, 0, 0, 0
    magnitudes = Magnitudes()
    for _ in range(cfg.eval_batches if batches is None else batches):
        x, y, mask = make_batch(task, cfg.batch_size, lengths, cfg, tok, rng)
        logits, states = model(x, diagnostics=True)
        if not torch.isfinite(logits).all() or not torch.isfinite(states).all():
            raise FloatingPointError("nonfinite deterministic evaluation state/logit")
        loss, _ = masked_metrics(logits, y, mask)
        losses.append(loss.item())
        predictions = logits.argmax(-1)
        matches = predictions == y
        correct += int(matches[mask].sum())
        total += int(mask.sum())
        exact += int((matches | ~mask).all(dim=1).sum())
        samples += x.shape[0]
        magnitudes.add(logits, states)
    model.train()
    return {"loss": float(np.mean(losses)), "token_accuracy": correct / total,
            "exact_sequence_accuracy": exact / samples, "batches": len(losses),
            "samples": samples, "target_tokens": total, **magnitudes.result()}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def failed_run(activation: str, task: str, seed: int, cfg: Config, tok: Tokens,
               reason: str, step: int | None, started: float, traceback_text: str | None = None):
    return {
        "activation": activation, "task": task, "seed": seed,
        "parameter_count": count_parameters(MemoryRNN(activation, tok.vocab_in, cfg.symbols, cfg)),
        "steps_budget": cfg.copy_steps if task == "delayed_copy" else cfg.palindrome_steps,
        "steps_completed": 0 if step is None else max(0, step - 1),
        "status": "failed", "nonfinite_failure": "nonfinite" in reason.lower(),
        "failure_reason": reason, "failure_step": step, "traceback": traceback_text,
        "thresholds": {"0.90": None, "0.99": None}, "evaluations": {},
        "optimization": None, "runtime_seconds": time.perf_counter() - started,
        "curve": [],
    }


def train_one(activation: str, task: str, seed: int, cfg: Config, tok: Tokens) -> dict:
    started = time.perf_counter()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = MemoryRNN(activation, tok.vocab_in, cfg.symbols, cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    task_offset = 0 if task == "delayed_copy" else 1_000_000
    train_rng = np.random.default_rng(10_000 + seed + task_offset)
    lengths = cfg.train_copy_lengths if task == "delayed_copy" else cfg.train_pal_lengths
    long_lengths = cfg.long_copy_lengths if task == "delayed_copy" else cfg.long_pal_lengths
    steps = cfg.copy_steps if task == "delayed_copy" else cfg.palindrome_steps
    curves, grad_norms = [], []
    clip_bind_count = 0
    thresholds: dict[str, dict | None] = {"0.90": None, "0.99": None}

    for step in range(1, steps + 1):
        curriculum_upper = min(lengths[1], 1 + (step - 1) // 200)
        curriculum_lengths = (min(lengths[0], curriculum_upper), curriculum_upper)
        x, y, mask = make_batch(task, cfg.batch_size, curriculum_lengths, cfg, tok, train_rng)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        if not torch.isfinite(logits).all():
            return failed_run(activation, task, seed, cfg, tok,
                              "nonfinite training logits", step, started)
        loss, train_acc = masked_metrics(logits, y, mask)
        if not torch.isfinite(loss):
            return failed_run(activation, task, seed, cfg, tok,
                              "nonfinite training loss", step, started)
        loss.backward()
        grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm))
        if not math.isfinite(grad_norm):
            return failed_run(activation, task, seed, cfg, tok,
                              "nonfinite pre-clipping gradient norm", step, started)
        grad_norms.append(grad_norm)
        clip_bind_count += grad_norm > cfg.gradient_clip_norm
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            return failed_run(activation, task, seed, cfg, tok,
                              "nonfinite parameter after optimizer step", step, started)
        if step == 1 or step % cfg.eval_every == 0 or step == steps:
            try:
                validation = evaluate(model, task, lengths, cfg, tok,
                                      20_000 + seed + task_offset)
            except FloatingPointError as exc:
                return failed_run(activation, task, seed, cfg, tok, str(exc), step, started)
            curves.append({"step": step, "examples": step * cfg.batch_size,
                           "train_loss": loss.item(), "train_accuracy": train_acc,
                           "validation_loss": validation["loss"],
                           "validation_accuracy": validation["token_accuracy"],
                           "validation_exact_sequence_accuracy": validation["exact_sequence_accuracy"]})
            for label, threshold in (("0.90", 0.90), ("0.99", 0.99)):
                if thresholds[label] is None and validation["token_accuracy"] >= threshold:
                    thresholds[label] = {"step": step, "examples": step * cfg.batch_size}

    try:
        evaluations = {
            "in_distribution": evaluate(model, task, lengths, cfg, tok,
                                        30_000 + seed + task_offset),
            "longer_lengths": evaluate(model, task, long_lengths, cfg, tok,
                                       40_000 + seed + task_offset),
        }
        if task == "two_palindrome":
            for n in (3, 4):
                evaluations[f"fixed_block_length_{n}"] = evaluate(
                    model, task, (n, n), cfg, tok,
                    50_000 + n * 10_000 + seed + task_offset,
                    batches=cfg.fixed_eval_batches)
    except FloatingPointError as exc:
        return failed_run(activation, task, seed, cfg, tok, str(exc), steps, started)
    optimization = {
        "pre_clip_gradient_norm_mean": statistics.mean(grad_norms),
        "pre_clip_gradient_norm_std": statistics.stdev(grad_norms),
        "pre_clip_gradient_norm_max": max(grad_norms),
        "clip_bind_steps": clip_bind_count, "optimizer_steps": len(grad_norms),
        "clip_bind_fraction": clip_bind_count / len(grad_norms),
    }
    return {
        "activation": activation, "task": task, "seed": seed,
        "parameter_count": count_parameters(model), "steps_budget": steps,
        "steps_completed": steps, "status": "ok", "nonfinite_failure": False,
        "failure_reason": None, "failure_step": None, "traceback": None,
        "thresholds": thresholds, "evaluations": evaluations,
        "optimization": optimization, "runtime_seconds": time.perf_counter() - started,
        "curve": curves,
    }


def _ci95(values: list[float]) -> list[float]:
    mean = statistics.mean(values)
    if len(values) < 2:
        return [mean, mean]
    # Two-sided .975 Student-t quantiles (all sample sizes possible here).
    critical = {
        1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706,
        6: 2.4469, 7: 2.3646, 8: 2.3060, 9: 2.2622, 10: 2.2281,
        11: 2.2010, 12: 2.1788, 13: 2.1604, 14: 2.1448, 15: 2.1314,
        16: 2.1199, 17: 2.1098, 18: 2.1009, 19: 2.0930,
    }.get(len(values) - 1, 1.96)
    half = critical * statistics.stdev(values) / math.sqrt(len(values))
    return [mean - half, mean + half]


def panel_names(task: str) -> list[str]:
    panels = ["in_distribution", "longer_lengths"]
    if task == "two_palindrome":
        panels += ["fixed_block_length_3", "fixed_block_length_4"]
    return panels


def summarize(runs: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    arm_rows, panel_rows, paired_rows = [], [], []
    by_key = {(r["task"], r["seed"], r["activation"]): r for r in runs}
    for task in TASKS:
        seeds = sorted({r["seed"] for r in runs if r["task"] == task})
        for activation in ACTIVATIONS:
            all_group = [r for r in runs if r["task"] == task and r["activation"] == activation]
            good = [r for r in all_group if r["status"] == "ok"]
            row = {"task": task, "activation": activation, "runs": len(all_group),
                   "successful_runs": len(good), "failed_runs": len(all_group) - len(good),
                   "nonfinite_failures": sum(r["nonfinite_failure"] for r in all_group)}
            for label in ("0.90", "0.99"):
                reached = [r["thresholds"][label]["step"] for r in good
                           if r["thresholds"][label] is not None]
                prefix = "threshold_" + label.replace(".", "_")
                row[prefix + "_successes"] = len(reached)
                row[prefix + "_step_mean_reached"] = statistics.mean(reached) if reached else None
            for key in ("pre_clip_gradient_norm_mean", "pre_clip_gradient_norm_max",
                        "clip_bind_fraction"):
                vals = [r["optimization"][key] for r in good]
                row[key + "_mean"] = statistics.mean(vals) if vals else None
                row[key + "_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0 if vals else None
            row["runtime_seconds_mean"] = statistics.mean(r["runtime_seconds"] for r in all_group)
            arm_rows.append(row)
            for panel in panel_names(task):
                prow = {"task": task, "panel": panel, "activation": activation,
                        "runs": len(all_group), "successful_runs": len(good)}
                for metric in (*METRICS, "recurrent_state_rms", "recurrent_state_max_abs",
                               "logit_rms", "logit_max_abs"):
                    vals = [r["evaluations"][panel][metric] for r in good]
                    prow[metric + "_mean"] = statistics.mean(vals) if vals else None
                    prow[metric + "_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0 if vals else None
                    prow[metric + "_ci95"] = _ci95(vals) if vals else None
                panel_rows.append(prow)
        for activation in ACTIVATIONS[1:]:
            for panel in panel_names(task):
                for metric in (*METRICS, "recurrent_state_rms", "recurrent_state_max_abs",
                               "logit_rms", "logit_max_abs"):
                    diffs, used_seeds = [], []
                    for seed in seeds:
                        first, base = by_key[task, seed, activation], by_key[task, seed, "tanh"]
                        if first["status"] == base["status"] == "ok":
                            diffs.append(first["evaluations"][panel][metric]
                                         - base["evaluations"][panel][metric])
                            used_seeds.append(seed)
                    paired_rows.append({
                        "comparison_type": "evaluation", "task": task, "panel": panel,
                        "metric": metric, "activation": activation, "baseline": "tanh",
                        "paired_seeds": len(diffs), "seed_ids": used_seeds,
                        "differences_by_seed": diffs,
                        "mean_difference": statistics.mean(diffs) if diffs else None,
                        "std_difference": statistics.stdev(diffs) if len(diffs) > 1 else 0.0 if diffs else None,
                        "ci95": _ci95(diffs) if diffs else None,
                        "activation_wins": sum(d > 0 for d in diffs),
                        "ties": sum(d == 0 for d in diffs), "tanh_wins": sum(d < 0 for d in diffs),
                    })
            for label in ("0.90", "0.99"):
                both_steps, both_seeds = [], []
                activation_only = tanh_only = both_success = neither = 0
                for seed in seeds:
                    first, base = by_key[task, seed, activation], by_key[task, seed, "tanh"]
                    a = first["status"] == "ok" and first["thresholds"][label] is not None
                    b = base["status"] == "ok" and base["thresholds"][label] is not None
                    if a and b:
                        both_success += 1
                        both_steps.append(first["thresholds"][label]["step"]
                                          - base["thresholds"][label]["step"])
                        both_seeds.append(seed)
                    elif a:
                        activation_only += 1
                    elif b:
                        tanh_only += 1
                    else:
                        neither += 1
                paired_rows.append({
                    "comparison_type": "threshold", "task": task, "panel": None,
                    "metric": "threshold_" + label.replace(".", "_") + "_step",
                    "activation": activation, "baseline": "tanh", "paired_seeds": len(seeds),
                    "both_success": both_success, "activation_only_success": activation_only,
                    "tanh_only_success": tanh_only, "neither_success": neither,
                    "joint_success_seed_ids": both_seeds, "differences_by_seed": both_steps,
                    "mean_difference": statistics.mean(both_steps) if both_steps else None,
                    "std_difference": statistics.stdev(both_steps) if len(both_steps) > 1 else 0.0 if both_steps else None,
                    "ci95": _ci95(both_steps) if both_steps else None,
                })
    return arm_rows, panel_rows, paired_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    serialized = []
    for row in rows:
        serialized.append({k: json.dumps(v) if isinstance(v, (list, dict)) else v
                           for k, v in row.items()})
    keys = list(dict.fromkeys(key for row in serialized for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(serialized)


def make_plots(runs: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    good = [r for r in runs if r["status"] == "ok"]
    for task in TASKS:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for activation in ACTIVATIONS:
            group = [r for r in good if r["task"] == task and r["activation"] == activation]
            if not group:
                continue
            steps = [p["step"] for p in group[0]["curve"]]
            vals = np.array([[p["validation_accuracy"] for p in r["curve"]] for r in group])
            mean = vals.mean(0)
            axes[0].plot(steps, mean, label=activation)
            final = [r["evaluations"]["longer_lengths"]["token_accuracy"] for r in group]
            x = ACTIVATIONS.index(activation)
            axes[1].scatter([x] * len(final), final, alpha=.6, s=14)
            axes[1].bar(x, np.mean(final), alpha=.25)
        axes[0].set(title="Validation token accuracy", xlabel="optimizer step",
                    ylabel="accuracy", ylim=(0, 1.03))
        axes[0].legend(fontsize=8)
        axes[1].set(title="Held-out longer lengths", ylabel="token accuracy",
                    ylim=(0, 1.03), xticks=range(4), xticklabels=ACTIVATIONS)
        fig.suptitle(task.replace("_", " "))
        fig.tight_layout()
        fig.savefig(out_dir / f"activation_ablation_{task}.png", dpi=150)
        plt.close(fig)


def _train_worker(job):
    activation, task, seed, cfg = job
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    tok = Tokens(cfg.symbols)
    started = time.perf_counter()
    try:
        return train_one(activation, task, seed, cfg, tok)
    except Exception as exc:  # Preserve the rest of the panel on any divergent run.
        import traceback
        return failed_run(activation, task, seed, cfg, tok,
                          f"worker exception: {type(exc).__name__}: {exc}", None, started,
                          traceback.format_exc())


def validate(runs: list[dict], seeds: int, tasks: list[str], activations: list[str],
             parameter_counts: dict[str, int], paired_initializations_equal: bool) -> dict:
    expected = {(task, seed, activation) for task in tasks for seed in range(seeds)
                for activation in activations}
    observed = {(r["task"], r["seed"], r["activation"]) for r in runs}
    duplicates = len(runs) - len(observed)
    checks = {
        "expected_run_count": len(expected), "observed_run_count": len(runs),
        "run_count_ok": len(runs) == len(expected), "duplicate_keys": duplicates,
        "missing_keys": [list(x) for x in sorted(expected - observed)],
        "unexpected_keys": [list(x) for x in sorted(observed - expected)],
        "parameter_counts_equal": len(set(parameter_counts.values())) == 1,
        "paired_initializations_equal": paired_initializations_equal,
        "all_run_parameter_counts_match": all(
            r["parameter_count"] == parameter_counts[r["activation"]] for r in runs),
        "all_successful_outputs_complete": all(
            set(r["evaluations"]) == set(panel_names(r["task"]))
            and all(all(metric in r["evaluations"][p]
                        for metric in (*METRICS, "recurrent_state_rms",
                                       "recurrent_state_max_abs", "logit_rms", "logit_max_abs"))
                    for p in panel_names(r["task"]))
            for r in runs if r["status"] == "ok"),
    }
    checks["passed"] = (checks["run_count_ok"] and duplicates == 0
                        and not checks["missing_keys"] and not checks["unexpected_keys"]
            and checks["parameter_counts_equal"] and checks["paired_initializations_equal"]
                        and checks["all_run_parameter_counts_match"]
                        and checks["all_successful_outputs_complete"])
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--copy-steps", type=int, default=None)
    parser.add_argument("--palindrome-steps", type=int, default=None)
    parser.add_argument("--activations", nargs="+", choices=ACTIVATIONS, default=list(ACTIVATIONS))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--output", type=Path, default=Path("activation_ablation_results.json"))
    args = parser.parse_args()
    cfg = Config(**{k: v for k, v in {"copy_steps": args.copy_steps,
                                      "palindrome_steps": args.palindrome_steps}.items()
                    if v is not None})
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    tok = Tokens(cfg.symbols)
    parameter_counts = {activation: count_parameters(
        MemoryRNN(activation, tok.vocab_in, cfg.symbols, cfg)) for activation in ACTIVATIONS}
    assert len(set(parameter_counts.values())) == 1, parameter_counts
    initial_states = {}
    for activation in ACTIVATIONS:
        torch.manual_seed(123_456)
        initial_states[activation] = MemoryRNN(
            activation, tok.vocab_in, cfg.symbols, cfg).state_dict()
    paired_initializations_equal = all(
        initial_states[activation].keys() == initial_states["tanh"].keys()
        and all(torch.equal(initial_states[activation][key], initial_states["tanh"][key])
                for key in initial_states["tanh"])
        for activation in ACTIVATIONS[1:])
    assert paired_initializations_equal
    jobs = [(activation, task, seed, cfg) for task in args.tasks
            for seed in range(args.seeds) for activation in args.activations]
    total_started = time.perf_counter()
    if args.workers == 1:
        runs = [_train_worker(job) for job in jobs]
    else:
        runs = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            for run in executor.map(_train_worker, jobs):
                runs.append(run)
                print(f"finished task={run['task']} seed={run['seed']} "
                      f"activation={run['activation']} status={run['status']} "
                      f"time={run['runtime_seconds']:.1f}s", flush=True)
    runs.sort(key=lambda r: (TASKS.index(r["task"]), r["seed"],
                             ACTIVATIONS.index(r["activation"])))
    arm_rows, panel_rows, paired_rows = summarize(runs)
    validation = validate(runs, args.seeds, args.tasks, args.activations, parameter_counts,
                          paired_initializations_equal)
    total_runtime = time.perf_counter() - total_started
    payload = {
        "metadata": {
            "command": shlex.join([sys.executable, *sys.argv]),
            "torch_version": torch.__version__, "python_version": platform.python_version(),
            "device": "cpu", "deterministic_algorithms": True,
            "parameter_counts": parameter_counts, "config": asdict(cfg),
            "workers": args.workers, "paired_batch_streams": True,
            "paired_initializations_equal": paired_initializations_equal,
            "locked_readout": "logits = W_out(z)",
            "varied_component": "h_next = activation(W_state(z))",
            "baseline": "tanh",
            "confidence_interval": "two-sided 95% Student t interval over paired seed differences",
            "magnitude_aggregation": "RMS over all elements and max absolute element across deterministic evaluation batches, summarized across runs",
            "total_runtime_seconds": total_runtime,
        },
        "validation": validation, "arm_summaries": arm_rows,
        "panel_summaries": panel_rows, "paired_comparisons": paired_rows, "runs": runs,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    csv_rows = ([{"row_type": "arm_summary", **r} for r in arm_rows]
                + [{"row_type": "evaluation_panel", **r} for r in panel_rows]
                + [{"row_type": "paired_comparison", **r} for r in paired_rows])
    csv_path = args.output.with_name("activation_ablation_summary.csv")
    write_csv(csv_path, csv_rows)
    make_plots(runs, args.output.parent)
    print(json.dumps({"validation": validation, "arm_summaries": arm_rows}, indent=2))
    print(f"wrote {args.output} and {csv_path}; total runtime {total_runtime:.1f}s")
    if not validation["passed"]:
        raise SystemExit("validation failed")


if __name__ == "__main__":
    main()
