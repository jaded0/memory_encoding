#!/usr/bin/env python3
"""CPU ablation of the forked EphemeralRNN output-feature activation.

This is a standalone, close transcription of the project's DFA EphemeralLinear
mechanics.  It deliberately has no imports from the worktree so the experiment
remains reproducible if project modules later change.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
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
from torch.nn import functional as F


ACTIVATIONS = ("identity", "tanh", "relu")
TASKS = ("repeated_copy", "palindrome2")
LAYERS = ("trunk.0", "trunk.1", "i2h", "i2o")


@dataclass(frozen=True)
class Config:
    symbols: int = 4
    hidden_size: int = 64
    num_layers: int = 2
    batch_size: int = 16
    train_batches: int = 600
    eval_every: int = 40
    eval_batches: int = 8
    learning_rate: float = 1e-4
    plasticity: float = 1e5
    forget_rate: float = 0.01
    ephemeral_fraction: float = 0.20
    ephemeral_update_clamp: float = 0.0
    unit_norm_weights: bool = False
    weight_clamp: float = 0.0
    copy_length: int = 4

    @property
    def vocab_size(self) -> int:
        return self.symbols + 2  # SEP and fixed palindrome middle token


class EphemeralLinear(nn.Linear):
    """Project-style per-sequence fast/slow layer, with explicit diagnostics."""

    def __init__(self, in_features: int, out_features: int, vocab_size: int,
                 cfg: Config, is_last_layer: bool = False):
        super().__init__(in_features, out_features, bias=True)
        self.weight.requires_grad_(False)
        self.bias.requires_grad_(False)
        self.is_last_layer = is_last_layer
        self.batch_size = cfg.batch_size
        self.forget_rate = cfg.forget_rate
        self.unit_norm_weights = cfg.unit_norm_weights
        self.weight_clamp = cfg.weight_clamp
        self.update_clamp = cfg.ephemeral_update_clamp
        self.feedback_weights = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(vocab_size, out_features)),
            requires_grad=False)
        # Match the project RNG order: feedback, zero per-sample tensor, then mask draw.
        self.per_sample_weights = nn.Parameter(
            torch.zeros(cfg.batch_size, out_features, in_features), requires_grad=False)
        rand_vals = torch.rand_like(self.weight)
        mask = torch.zeros_like(self.weight, dtype=torch.bool) if is_last_layer else (
            rand_vals < cfg.ephemeral_fraction)
        self.ephemeral_mask = nn.Parameter(mask, requires_grad=False)
        plasticity = torch.ones_like(self.weight)
        plasticity[mask] = cfg.plasticity
        self.plasticity = nn.Parameter(plasticity, requires_grad=False)
        # Retain nn.Linear's already-drawn default values on slow entries; fast starts zero.
        with torch.no_grad():
            self.per_sample_weights.copy_(self.weight.unsqueeze(0).expand_as(self.per_sample_weights))
            self.per_sample_weights.masked_fill_(mask.unsqueeze(0), 0.0)
        self.in_traces = torch.zeros(cfg.batch_size, in_features)
        self._projected_error: torch.Tensor | None = None
        self._gradient: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.in_traces = x.detach()
        return torch.bmm(self.per_sample_weights, x.unsqueeze(2)).squeeze(2) + self.bias

    @torch.no_grad()
    def start_sequence_wipe(self) -> None:
        aggregate = self.per_sample_weights.mean(0, keepdim=True)
        self.per_sample_weights.copy_(aggregate.expand_as(self.per_sample_weights))
        self.per_sample_weights.masked_fill_(self.ephemeral_mask.unsqueeze(0), 0.0)
        self.weight[self.ephemeral_mask] = 0.0  # unused base weight; project does this too

    def populate_dfa_gradients(self, output_error: torch.Tensor) -> None:
        projected = output_error if self.is_last_layer else output_error @ self.feedback_weights
        self._projected_error = projected
        self._gradient = projected.unsqueeze(2) * self.in_traces.unsqueeze(1)

    @torch.no_grad()
    def apply_update(self, learning_rate: float) -> dict[str, float]:
        assert self._gradient is not None and self._projected_error is not None
        update = -self._gradient * self.plasticity.unsqueeze(0)
        mask = self.ephemeral_mask.unsqueeze(0).expand_as(update)
        if self.update_clamp > 0 and not self.is_last_layer:
            update = torch.where(mask, update.clamp(-self.update_clamp, self.update_clamp), update)
        fast_update = update[mask]
        slow_update = update[~mask]
        stats = {
            "fast_update_norm": float(fast_update.norm()) if fast_update.numel() else 0.0,
            "slow_update_norm": float(slow_update.norm()) if slow_update.numel() else 0.0,
        }
        self.per_sample_weights.add_(update, alpha=learning_rate)
        self.bias.add_(-learning_rate * self._projected_error.mean(0))
        if self.unit_norm_weights:
            norms = torch.linalg.vector_norm(self.per_sample_weights, dim=(1, 2), keepdim=True)
            self.per_sample_weights.div_(norms + 1e-6)
        if self.weight_clamp:
            self.per_sample_weights.clamp_(-self.weight_clamp, self.weight_clamp)
        # Project semantics: forgetting is after update and regularization.
        self.per_sample_weights.mul_(1 - self.forget_rate * self.ephemeral_mask)
        fast_weight = self.per_sample_weights[mask]
        slow_weight = self.per_sample_weights[~mask]
        stats.update(
            fast_weight_norm=float(fast_weight.norm()) if fast_weight.numel() else 0.0,
            slow_weight_norm=float(slow_weight.norm()) if slow_weight.numel() else 0.0,
        )
        return stats


class ForkedEphemeralRNN(nn.Module):
    def __init__(self, activation: str, cfg: Config):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(activation)
        self.activation = activation
        self.cfg = cfg
        inner = cfg.vocab_size + cfg.hidden_size
        self.trunk = nn.ModuleList([
            EphemeralLinear(inner, inner, cfg.vocab_size, cfg),
            *[EphemeralLinear(inner, inner, cfg.vocab_size, cfg)
              for _ in range(1, cfg.num_layers)],
        ])
        # Historical aef0697 fork: both heads read the deep trunk representation.
        self.i2h = EphemeralLinear(inner, cfg.hidden_size, cfg.vocab_size, cfg)
        self.i2o = EphemeralLinear(inner, cfg.vocab_size, cfg.vocab_size, cfg,
                                   is_last_layer=True)
        self.last_z: torch.Tensor | None = None
        self.last_output_features: torch.Tensor | None = None
        self.last_hidden_candidate: torch.Tensor | None = None

    def named_dfa_layers(self):
        return [(f"trunk.{i}", layer) for i, layer in enumerate(self.trunk)] + [
            ("i2h", self.i2h), ("i2o", self.i2o)]

    def start_sequence_wipe(self) -> None:
        for _, layer in self.named_dfa_layers():
            layer.start_sequence_wipe()

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.cat((x, hidden), dim=1)
        for layer in self.trunk:
            z = F.gelu(layer(z))
        hidden_candidate = torch.tanh(self.i2h(z))
        # Clipped recurrence is unconditional: candidate is formed, then discarded.
        next_hidden = torch.zeros_like(hidden)
        if self.activation == "identity":
            features = z
        elif self.activation == "tanh":
            features = torch.tanh(z)
        else:
            features = torch.relu(z)
        logits = self.i2o(features)  # activation is on features, never logits
        self.last_z, self.last_output_features = z.detach(), features.detach()
        self.last_hidden_candidate = hidden_candidate.detach()
        return logits, next_hidden

    def dfa_step(self, output_error: torch.Tensor) -> dict[str, dict[str, float]]:
        # Populate every layer from the same, unmodified output error before any update.
        for _, layer in self.named_dfa_layers():
            layer.populate_dfa_gradients(output_error)
        return {name: layer.apply_update(self.cfg.learning_rate)
                for name, layer in self.named_dfa_layers()}


def make_batch(task: str, cfg: Config, rng: np.random.Generator):
    seqs, recall_starts = [], []
    for _ in range(cfg.batch_size):
        if task == "repeated_copy":
            block = rng.integers(0, cfg.symbols, cfg.copy_length).tolist()
            seq = block + [cfg.symbols] + block
            recall_start = cfg.copy_length  # target-step index of first copied symbol
        elif task == "palindrome2":
            half = rng.integers(0, cfg.symbols, 2).tolist()
            seq = half + [cfg.symbols + 1] + half[::-1]
            recall_start = 2  # target-step index immediately after fixed middle token
        else:
            raise ValueError(task)
        seqs.append(seq)
        recall_starts.append(recall_start)
    seq = torch.tensor(seqs, dtype=torch.long)
    x, targets = seq[:, :-1], seq[:, 1:]
    mask = torch.zeros_like(targets, dtype=torch.bool)
    for row, start in enumerate(recall_starts):
        mask[row, start:] = True
    return x, targets, mask


class Accumulator:
    def __init__(self):
        self.correct = self.total = self.exact = self.sequences = 0
        self.loss_sum = self.loss_count = 0
        self.feature_sq = self.feature_count = 0
        self.feature_max = 0.0
        self.saturated = self.saturation_count = 0
        self.norm_sums = {layer: {key: 0.0 for key in (
            "fast_update_norm", "slow_update_norm", "fast_weight_norm", "slow_weight_norm")}
            for layer in LAYERS}
        self.norm_steps = 0

    def add_features(self, features: torch.Tensor) -> None:
        values = features.double()
        self.feature_sq += float((values * values).sum())
        self.feature_count += values.numel()
        self.feature_max = max(self.feature_max, float(values.abs().max()))
        self.saturated += int((values.abs() >= 0.99).sum())
        self.saturation_count += values.numel()

    def add_norms(self, norms: dict[str, dict[str, float]]) -> None:
        for layer, values in norms.items():
            for key, value in values.items():
                self.norm_sums[layer][key] += value
        self.norm_steps += 1

    def result(self) -> dict:
        return {
            "loss": self.loss_sum / self.loss_count,
            "recall_token_accuracy": self.correct / self.total,
            "exact_sequence_accuracy": self.exact / self.sequences,
            "recall_tokens": self.total, "sequences": self.sequences,
            "output_feature_rms": math.sqrt(self.feature_sq / self.feature_count),
            "output_feature_max_abs": self.feature_max,
            # For tanh this is the requested saturation diagnostic; for the other arms it is
            # the directly comparable fraction of output features outside tanh's linear regime.
            "output_feature_abs_ge_0.99_fraction": self.saturated / self.saturation_count,
            "layer_norms": {layer: {key: value / self.norm_steps for key, value in vals.items()}
                            for layer, vals in self.norm_sums.items()},
        }


def process_batch(model: ForkedEphemeralRNN, task: str, rng: np.random.Generator) -> dict:
    cfg = model.cfg
    x, targets, recall_mask = make_batch(task, cfg, rng)
    model.start_sequence_wipe()
    hidden = torch.zeros(cfg.batch_size, cfg.hidden_size)
    acc = Accumulator()
    all_match = torch.ones(cfg.batch_size, dtype=torch.bool)
    with torch.no_grad():
        for step in range(x.shape[1]):
            onehot = F.one_hot(x[:, step], cfg.vocab_size).float()
            logits, hidden = model(onehot, hidden)
            target = targets[:, step]
            losses = F.cross_entropy(logits, target, reduction="none")
            output_error = torch.softmax(logits, 1) - F.one_hot(target, cfg.vocab_size)
            norms = model.dfa_step(output_error)
            if not torch.isfinite(logits).all() or not torch.isfinite(model.i2o.per_sample_weights).all():
                raise FloatingPointError(f"nonfinite at sequence step {step}")
            active = recall_mask[:, step]
            if active.any():
                match = logits.argmax(1) == target
                acc.correct += int(match[active].sum())
                acc.total += int(active.sum())
                all_match &= (match | ~active)
                acc.loss_sum += float(losses[active].sum())
                acc.loss_count += int(active.sum())
            acc.add_features(model.last_output_features)
            acc.add_norms(norms)
    acc.exact = int(all_match.sum())
    acc.sequences = cfg.batch_size
    return acc.result()


def merge_batch_results(results: list[dict]) -> dict:
    # Equal batch sizes and task lengths make arithmetic means exact for scalar rates.
    keys = ("loss", "recall_token_accuracy", "exact_sequence_accuracy",
            "output_feature_rms", "output_feature_abs_ge_0.99_fraction")
    merged = {key: statistics.mean(r[key] for r in results) for key in keys}
    merged["output_feature_max_abs"] = max(r["output_feature_max_abs"] for r in results)
    merged["recall_tokens"] = sum(r["recall_tokens"] for r in results)
    merged["sequences"] = sum(r["sequences"] for r in results)
    merged["layer_norms"] = {layer: {key: statistics.mean(
        r["layer_norms"][layer][key] for r in results)
        for key in results[0]["layer_norms"][layer]} for layer in LAYERS}
    return merged


def evaluate(model: ForkedEphemeralRNN, task: str, cfg: Config, seed: int) -> dict:
    probe = copy.deepcopy(model)
    rng = np.random.default_rng(seed)
    result = merge_batch_results([process_batch(probe, task, rng) for _ in range(cfg.eval_batches)])
    result["tanh_saturation_fraction"] = (result["output_feature_abs_ge_0.99_fraction"]
                                          if model.activation == "tanh" else None)
    return result


def train_one(activation: str, task: str, seed: int, cfg: Config) -> dict:
    started = time.perf_counter()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = ForkedEphemeralRNN(activation, cfg)
    task_offset = 0 if task == "repeated_copy" else 1_000_000
    train_rng = np.random.default_rng(10_000 + seed + task_offset)
    curve, status, failure = [], "ok", None
    thresholds = {"0.75": None, "0.90": None}
    try:
        for batch in range(1, cfg.train_batches + 1):
            train_metrics = process_batch(model, task, train_rng)
            if batch == 1 or batch % cfg.eval_every == 0 or batch == cfg.train_batches:
                val = evaluate(model, task, cfg, 20_000 + seed + task_offset)
                point = {"batch": batch, "sequences_seen": batch * cfg.batch_size,
                         "train_recall_token_accuracy": train_metrics["recall_token_accuracy"],
                         **{f"validation_{key}": val[key] for key in (
                             "loss", "recall_token_accuracy", "exact_sequence_accuracy")}}
                curve.append(point)
                for label, threshold in (("0.75", .75), ("0.90", .90)):
                    if thresholds[label] is None and val["recall_token_accuracy"] >= threshold:
                        thresholds[label] = batch
        final = evaluate(model, task, cfg, 30_000 + seed + task_offset)
    except (FloatingPointError, RuntimeError) as exc:
        status, failure, final = "failed", f"{type(exc).__name__}: {exc}", None
    auc = None
    if curve:
        xs = np.array([p["batch"] for p in curve], dtype=float)
        ys = np.array([p["validation_recall_token_accuracy"] for p in curve])
        # Include chance at batch zero, then normalize by total training budget.
        xs = np.r_[0.0, xs]; ys = np.r_[1 / cfg.symbols, ys]
        auc = float(np.trapezoid(ys, xs) / cfg.train_batches)
    return {"activation": activation, "task": task, "seed": seed, "status": status,
            "failure": failure, "nonfinite": bool(failure and "nonfinite" in failure),
            "runtime_seconds": time.perf_counter() - started, "threshold_batches": thresholds,
            "learning_auc": auc, "final": final, "curve": curve}


def ci95(values: list[float]) -> list[float]:
    mean = statistics.mean(values)
    if len(values) < 2:
        return [mean, mean]
    critical = {11: 2.201, 19: 2.093}.get(len(values) - 1, 1.96)
    half = critical * statistics.stdev(values) / math.sqrt(len(values))
    return [mean - half, mean + half]


def summarize(runs: list[dict], activations: list[str], tasks: list[str]):
    arms, pairs = [], []
    scalar_metrics = ("recall_token_accuracy", "exact_sequence_accuracy", "output_feature_rms",
                      "output_feature_max_abs", "output_feature_abs_ge_0.99_fraction")
    by_key = {(r["task"], r["seed"], r["activation"]): r for r in runs}
    for task in tasks:
        seeds = sorted({r["seed"] for r in runs if r["task"] == task})
        for activation in activations:
            group = [r for r in runs if r["task"] == task and r["activation"] == activation]
            good = [r for r in group if r["status"] == "ok"]
            row = {"task": task, "activation": activation, "runs": len(group),
                   "successful": len(good), "failed": len(group) - len(good),
                   "nonfinite": sum(r["nonfinite"] for r in group)}
            for metric in scalar_metrics:
                vals = [r["final"][metric] for r in good]
                row[f"{metric}_mean"] = statistics.mean(vals) if vals else None
                row[f"{metric}_ci95"] = ci95(vals) if vals else None
            tanh_saturation = [r["final"]["tanh_saturation_fraction"] for r in good
                               if r["final"]["tanh_saturation_fraction"] is not None]
            row["tanh_saturation_fraction_mean"] = (statistics.mean(tanh_saturation)
                                                      if tanh_saturation else None)
            aucs = [r["learning_auc"] for r in good]
            row["learning_auc_mean"] = statistics.mean(aucs) if aucs else None
            row["learning_auc_ci95"] = ci95(aucs) if aucs else None
            for threshold in ("0.75", "0.90"):
                reached = [r["threshold_batches"][threshold] for r in good
                           if r["threshold_batches"][threshold] is not None]
                row[f"threshold_{threshold}_successes"] = len(reached)
                row[f"threshold_{threshold}_mean_batch"] = statistics.mean(reached) if reached else None
            for layer in LAYERS:
                for metric in ("fast_update_norm", "slow_update_norm", "fast_weight_norm", "slow_weight_norm"):
                    vals = [r["final"]["layer_norms"][layer][metric] for r in good]
                    row[f"{layer}_{metric}_mean"] = statistics.mean(vals) if vals else None
            arms.append(row)
        if "identity" in activations and "tanh" in activations:
            for metric in (*scalar_metrics, "learning_auc"):
                diffs, used = [], []
                for seed in seeds:
                    identity, tanh = by_key[task, seed, "identity"], by_key[task, seed, "tanh"]
                    if identity["status"] == tanh["status"] == "ok":
                        a = identity["learning_auc"] if metric == "learning_auc" else identity["final"][metric]
                        b = tanh["learning_auc"] if metric == "learning_auc" else tanh["final"][metric]
                        diffs.append(a - b); used.append(seed)
                pairs.append({"task": task, "comparison": "identity_minus_tanh", "metric": metric,
                              "paired_seeds": len(diffs), "seed_ids": used,
                              "differences": diffs, "mean_difference": statistics.mean(diffs),
                              "ci95": ci95(diffs), "identity_wins": sum(x > 0 for x in diffs),
                              "ties": sum(x == 0 for x in diffs), "tanh_wins": sum(x < 0 for x in diffs)})
    return arms, pairs


def validation(cfg: Config, activations: list[str]) -> dict:
    states, models = {}, {}
    for activation in activations:
        torch.manual_seed(123456)
        model = ForkedEphemeralRNN(activation, cfg)
        models[activation] = model
        states[activation] = model.state_dict()
    base = states[activations[0]]
    init_equal = all(all(torch.equal(base[k], states[a][k]) for k in base) for a in activations[1:])
    probe = models[activations[0]]
    fast_zero = all(torch.count_nonzero(layer.per_sample_weights[:, layer.ephemeral_mask]) == 0
                    for _, layer in probe.named_dfa_layers())
    slow_default = all(all(torch.equal(sample[~layer.ephemeral_mask],
                                       layer.weight[~layer.ephemeral_mask])
                           for sample in layer.per_sample_weights)
                       for _, layer in probe.named_dfa_layers())
    last_no_fast = not probe.i2o.ephemeral_mask.any().item()
    x = F.one_hot(torch.arange(cfg.batch_size) % cfg.vocab_size, cfg.vocab_size).float()
    hidden = torch.randn(cfg.batch_size, cfg.hidden_size)
    _, next_hidden = probe(x, hidden)
    z_equal_i2o_trace = torch.equal(probe.i2o.in_traces, probe.last_output_features)
    # Isolate and verify the sole arm difference on an identical state and input.
    torch.manual_seed(123456)
    identity_probe = ForkedEphemeralRNN("identity", cfg)
    tanh_probe = ForkedEphemeralRNN("tanh", cfg)
    tanh_probe.load_state_dict(identity_probe.state_dict())
    identity_logits, _ = identity_probe(x, hidden)
    tanh_logits, _ = tanh_probe(x, hidden)
    output_only_difference = (torch.equal(identity_probe.last_z, tanh_probe.last_z)
                              and torch.equal(identity_probe.last_hidden_candidate,
                                              tanh_probe.last_hidden_candidate)
                              and torch.equal(identity_probe.i2o.in_traces,
                                              identity_probe.last_z)
                              and torch.equal(tanh_probe.i2o.in_traces,
                                              torch.tanh(tanh_probe.last_z))
                              and not torch.equal(identity_logits, tanh_logits))
    # Pin project wipe semantics: batch-mean slow weights are copied; fast entries become zero.
    wipe_layer = copy.deepcopy(identity_probe.trunk[0])
    with torch.no_grad():
        wipe_layer.per_sample_weights.add_(torch.arange(cfg.batch_size).view(-1, 1, 1))
        expected_slow = wipe_layer.per_sample_weights.mean(0)[~wipe_layer.ephemeral_mask].clone()
    wipe_layer.start_sequence_wipe()
    wipe_ok = (torch.count_nonzero(
        wipe_layer.per_sample_weights[:, wipe_layer.ephemeral_mask]).item() == 0
        and all(torch.equal(sample[~wipe_layer.ephemeral_mask], expected_slow)
                for sample in wipe_layer.per_sample_weights))
    checks = {"paired_initializations_equal": init_equal, "fast_entries_initially_zero": fast_zero,
              "slow_entries_equal_nn_linear_default": slow_default, "i2o_has_no_fast_entries": last_no_fast,
              "recurrence_output_is_zero": torch.count_nonzero(next_hidden).item() == 0,
              "i2h_candidate_was_formed": probe.last_hidden_candidate is not None,
              "i2o_trace_equals_activated_deep_trunk": z_equal_i2o_trace,
              "output_head_in_features_equals_deep_trunk": probe.i2o.in_features == cfg.vocab_size + cfg.hidden_size,
              "activation_is_only_forward_difference": output_only_difference,
              "wipe_zeros_fast_and_averages_slow": wipe_ok}
    checks["passed"] = all(checks.values())
    return checks


def write_csv(path: Path, rows: list[dict]) -> None:
    serial = [{k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in rows]
    fields = list(dict.fromkeys(k for row in serial for k in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(serial)


def plot(runs: list[dict], path: Path, activations: list[str], tasks: list[str]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        for activation in activations:
            group = [r for r in runs if r["task"] == task and r["activation"] == activation and r["status"] == "ok"]
            if not group: continue
            xs = [p["batch"] for p in group[0]["curve"]]
            ys = np.array([[p["validation_recall_token_accuracy"] for p in r["curve"]] for r in group])
            ax.plot(xs, ys.mean(0), label=activation)
            ax.fill_between(xs, ys.mean(0) - ys.std(0), ys.mean(0) + ys.std(0), alpha=.15)
        ax.set(title=task, xlabel="training batch", ylabel="recall token accuracy", ylim=(0, 1.02))
        ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def worker(job):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    return train_one(*job)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--train-batches", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--plasticity", type=float)
    parser.add_argument("--weight-clamp", type=float)
    parser.add_argument("--activations", nargs="+", choices=ACTIVATIONS, default=list(ACTIVATIONS))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    args = parser.parse_args()
    overrides = {
        "train_batches": args.train_batches,
        "learning_rate": args.learning_rate,
        "plasticity": args.plasticity,
        "weight_clamp": args.weight_clamp,
    }
    cfg = Config(**{key: value for key, value in overrides.items() if value is not None})
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    checks = validation(cfg, args.activations)
    if not checks["passed"]: raise RuntimeError(checks)
    jobs = [(activation, task, seed, cfg) for task in args.tasks for seed in range(args.seeds)
            for activation in args.activations]
    started = time.perf_counter()
    if args.workers == 1:
        runs = [worker(job) for job in jobs]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            runs = list(executor.map(worker, jobs))
    runs.sort(key=lambda r: (TASKS.index(r["task"]), r["seed"], ACTIVATIONS.index(r["activation"])))
    arms, pairs = summarize(runs, args.activations, args.tasks)
    payload = {"metadata": {"command": shlex.join([sys.executable, *sys.argv]),
                "runtime_seconds": time.perf_counter() - started, "python": platform.python_version(),
                "torch": torch.__version__, "device": "cpu", "deterministic_algorithms": True,
                "config": asdict(cfg), "paired_seeds_batches_masks_parameters_initialization": True,
                "recurrence": False, "topology": "forked: i2o(activation(deep_trunk_z)); tanh(i2h(z)) discarded",
                "dfa_activation_derivatives": False,
                "confidence_intervals": "two-sided 95% Student-t interval over seeds/paired differences"},
               "validation": checks, "arm_summaries": arms, "paired_comparisons": pairs, "runs": runs}
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    write_csv(args.output.with_name("summary.csv"),
              [{"row_type": "arm", **r} for r in arms] + [{"row_type": "paired", **r} for r in pairs])
    plot(runs, args.output.with_name("learning_curves.png"), args.activations, args.tasks)
    print(json.dumps({"runtime_seconds": payload["metadata"]["runtime_seconds"],
                      "validation": checks, "arm_summaries": arms, "paired_comparisons": pairs}, indent=2))


if __name__ == "__main__":
    main()
