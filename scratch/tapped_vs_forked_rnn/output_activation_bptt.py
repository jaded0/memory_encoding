#!/usr/bin/env python3
"""Paired forked-output activation ablation with locked tanh recurrence."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
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


READOUTS = ("identity", "tanh", "relu")
TASKS = ("delayed_copy", "two_palindrome")
ACCURACY_METRICS = ("token_accuracy", "exact_sequence_accuracy")
DIAGNOSTIC_METRICS = ("readout_feature_rms", "readout_feature_max_abs",
                      "recurrent_state_rms", "recurrent_state_max_abs",
                      "logit_rms", "logit_max_abs")


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


class Tokens:
    def __init__(self, symbols: int):
        self.PAD = symbols
        self.BLANK = symbols + 1
        self.START = symbols + 2
        self.SEP = symbols + 3
        self.QUERY1 = symbols + 4
        self.QUERY2 = symbols + 5
        self.vocab_in = symbols + 6


class MemoryRNN(nn.Module):
    """Identical forked models; only a parameter-free output transform varies."""

    def __init__(self, readout: str, vocab_in: int, outputs: int, cfg: Config):
        super().__init__()
        if readout not in READOUTS:
            raise ValueError(readout)
        self.readout = readout
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

    def output_features(self, z: torch.Tensor) -> torch.Tensor:
        if self.readout == "tanh":
            return torch.tanh(z)
        if self.readout == "relu":
            return torch.relu(z)
        return z

    def forward(self, x: torch.Tensor, diagnostics: bool = False):
        h = torch.zeros(x.shape[0], self.hidden_dim, device=x.device)
        logits_by_step, states, features, carriers = [], [], [], []
        for t in range(x.shape[1]):
            transformed = torch.relu(self.trunk1(
                torch.cat((self.embedding(x[:, t]), h), dim=-1)))
            z = h + torch.relu(self.trunk2(transformed))
            # Locked recurrence and topology in every arm.
            h = torch.tanh(self.W_state(z))
            readout_features = self.output_features(z)
            logits_by_step.append(self.W_out(readout_features))
            if diagnostics:
                states.append(h)
                features.append(readout_features)
                carriers.append(z)
        logits = torch.stack(logits_by_step, dim=1)
        if diagnostics:
            return (logits, torch.stack(states, dim=1),
                    torch.stack(features, dim=1), torch.stack(carriers, dim=1))
        return logits


def make_batch(task, batch_size, length_range, cfg, tok, rng):
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
            y[i, position], mask[i, position] = value, True
    return x, y, mask


def masked_metrics(logits, y, mask):
    selected, targets = logits[mask], y[mask]
    return (nn.functional.cross_entropy(selected, targets),
            (selected.argmax(-1) == targets).float().mean().item())


class Diagnostics:
    def __init__(self):
        names = ("readout_feature", "recurrent_state", "logit")
        self.ssq = {name: 0.0 for name in names}
        self.count = {name: 0 for name in names}
        self.maximum = {name: 0.0 for name in names}
        self.saturated = 0
        self.saturation_count = 0

    def add(self, logits, states, features, carriers, is_tanh):
        for name, tensor in (("readout_feature", features),
                             ("recurrent_state", states), ("logit", logits)):
            data = tensor.detach().double()
            self.ssq[name] += float((data * data).sum())
            self.count[name] += data.numel()
            self.maximum[name] = max(self.maximum[name], float(data.abs().max()))
        if is_tanh:
            # Compute from z explicitly; equivalent to abs(readout feature) > .95.
            transformed = torch.tanh(carriers.detach())
            self.saturated += int((transformed.abs() > .95).sum())
            self.saturation_count += transformed.numel()

    def result(self, is_tanh):
        out = {}
        for name in self.ssq:
            out[name + "_rms"] = math.sqrt(self.ssq[name] / self.count[name])
            out[name + "_max_abs"] = self.maximum[name]
        out["tanh_readout_saturation_fraction"] = (
            self.saturated / self.saturation_count if is_tanh else None)
        return out


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


@torch.no_grad()
def evaluate(model, task, lengths, cfg, tok, seed, batches):
    model.eval()
    rng = np.random.default_rng(seed)
    losses, correct, total, exact, samples = [], 0, 0, 0, 0
    diagnostics = Diagnostics()
    for batch_index in range(batches):
        x, y, mask = make_batch(task, cfg.batch_size, lengths, cfg, tok, rng)
        logits, states, features, carriers = model(x, diagnostics=True)
        tensors = (logits, states, features, carriers)
        if not all(torch.isfinite(tensor).all() for tensor in tensors):
            model.train()
            return {"status": "nonfinite", "nonfinite_batch": batch_index}
        loss, _ = masked_metrics(logits, y, mask)
        matches = logits.argmax(-1) == y
        losses.append(loss.item())
        correct += int(matches[mask].sum())
        total += int(mask.sum())
        exact += int((matches | ~mask).all(dim=1).sum())
        samples += x.shape[0]
        diagnostics.add(logits, states, features, carriers, model.readout == "tanh")
    model.train()
    return {"status": "ok", "nonfinite_batch": None, "loss": float(np.mean(losses)),
            "token_accuracy": correct / total, "exact_sequence_accuracy": exact / samples,
            "batches": batches, "samples": samples, "target_tokens": total,
            **diagnostics.result(model.readout == "tanh")}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def failed_run(readout, task, seed, cfg, tok, reason, step, started, traceback_text=None):
    return {"readout": readout, "task": task, "seed": seed,
            "parameter_count": count_parameters(MemoryRNN(
                readout, tok.vocab_in, cfg.symbols, cfg)),
            "steps_budget": cfg.copy_steps if task == "delayed_copy" else cfg.palindrome_steps,
            "steps_completed": 0 if step is None else max(0, step - 1), "status": "failed",
            "nonfinite_failure": "nonfinite" in reason.lower(), "failure_reason": reason,
            "failure_step": step, "traceback": traceback_text, "evaluations": {},
            "optimization": None, "runtime_seconds": time.perf_counter() - started,
            "curve": []}


def train_one(readout, task, seed, cfg, tok):
    started = time.perf_counter()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = MemoryRNN(readout, tok.vocab_in, cfg.symbols, cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    offset = 0 if task == "delayed_copy" else 1_000_000
    rng = np.random.default_rng(10_000 + seed + offset)
    lengths = cfg.train_copy_lengths if task == "delayed_copy" else cfg.train_pal_lengths
    steps = cfg.copy_steps if task == "delayed_copy" else cfg.palindrome_steps
    curves, grad_norms, clip_binds = [], [], 0
    for step in range(1, steps + 1):
        upper = min(lengths[1], 1 + (step - 1) // 200)
        batch_lengths = (min(lengths[0], upper), upper)
        x, y, mask = make_batch(task, cfg.batch_size, batch_lengths, cfg, tok, rng)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        if not torch.isfinite(logits).all():
            return failed_run(readout, task, seed, cfg, tok,
                              "nonfinite training logits", step, started)
        loss, train_accuracy = masked_metrics(logits, y, mask)
        if not torch.isfinite(loss):
            return failed_run(readout, task, seed, cfg, tok,
                              "nonfinite training loss", step, started)
        loss.backward()  # Full BPTT: recurrent state is never detached.
        norm = float(nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm))
        if not math.isfinite(norm):
            return failed_run(readout, task, seed, cfg, tok,
                              "nonfinite pre-clipping gradient norm", step, started)
        grad_norms.append(norm)
        clip_binds += norm > cfg.gradient_clip_norm
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            return failed_run(readout, task, seed, cfg, tok,
                              "nonfinite parameter after optimizer step", step, started)
        if step == 1 or step % cfg.eval_every == 0 or step == steps:
            val = evaluate(model, task, lengths, cfg, tok, 20_000 + seed + offset,
                           cfg.eval_batches)
            if val["status"] != "ok":
                return failed_run(readout, task, seed, cfg, tok,
                                  "nonfinite validation tensor", step, started)
            curves.append({"step": step, "examples": step * cfg.batch_size,
                           "train_loss": loss.item(), "train_accuracy": train_accuracy,
                           "validation_loss": val["loss"],
                           "validation_accuracy": val["token_accuracy"],
                           "validation_exact_sequence_accuracy": val["exact_sequence_accuracy"]})
    evaluations = {}
    for panel_index, (name, panel_lengths, batches) in enumerate(panel_specs(task, cfg)):
        evaluations[name] = evaluate(model, task, panel_lengths, cfg, tok,
                                     30_000 + panel_index * 100_000 + seed + offset, batches)
    optimization = {"pre_clip_gradient_norm_mean": statistics.mean(grad_norms),
                    "pre_clip_gradient_norm_std": statistics.stdev(grad_norms),
                    "pre_clip_gradient_norm_max": max(grad_norms),
                    "clip_bind_steps": clip_binds, "optimizer_steps": len(grad_norms),
                    "clip_bind_fraction": clip_binds / len(grad_norms)}
    return {"readout": readout, "task": task, "seed": seed,
            "parameter_count": count_parameters(model), "steps_budget": steps,
            "steps_completed": steps, "status": "ok", "nonfinite_failure": False,
            "failure_reason": None, "failure_step": None, "traceback": None,
            "evaluations": evaluations, "optimization": optimization,
            "runtime_seconds": time.perf_counter() - started, "curve": curves}


def ci95(values):
    mean = statistics.mean(values)
    if len(values) < 2:
        return [mean, mean]
    critical = {19: 2.0930240544}.get(len(values) - 1, 1.96)
    half = critical * statistics.stdev(values) / math.sqrt(len(values))
    return [mean - half, mean + half]


def summarize(runs, cfg):
    arm_rows, panel_rows, contrasts = [], [], []
    by_key = {(r["task"], r["seed"], r["readout"]): r for r in runs}
    for task in TASKS:
        seeds = sorted({r["seed"] for r in runs if r["task"] == task})
        for readout in READOUTS:
            group = [r for r in runs if r["task"] == task and r["readout"] == readout]
            good = [r for r in group if r["status"] == "ok"]
            arm_rows.append({"task": task, "readout": readout, "runs": len(group),
                             "successful_runs": len(good), "failed_runs": len(group)-len(good),
                             "nonfinite_failures": sum(r["nonfinite_failure"] for r in group),
                             **{key + "_mean": statistics.mean(r["optimization"][key] for r in good)
                                if good else None for key in
                                ("pre_clip_gradient_norm_mean", "pre_clip_gradient_norm_max",
                                 "clip_bind_fraction")}})
            for panel, _, _ in panel_specs(task, cfg):
                finite = [r for r in good if r["evaluations"][panel]["status"] == "ok"]
                row = {"task": task, "panel": panel, "readout": readout,
                       "runs": len(group), "finite_evaluations": len(finite),
                       "nonfinite_evaluations": len(good) - len(finite)}
                metrics = (*ACCURACY_METRICS, *DIAGNOSTIC_METRICS,
                           "tanh_readout_saturation_fraction")
                for metric in metrics:
                    vals = [r["evaluations"][panel][metric] for r in finite
                            if r["evaluations"][panel][metric] is not None]
                    row[metric + "_mean"] = statistics.mean(vals) if vals else None
                    row[metric + "_std"] = (statistics.stdev(vals) if len(vals) > 1
                                             else 0.0 if vals else None)
                    row[metric + "_ci95"] = ci95(vals) if vals else None
                panel_rows.append(row)
        for first in ("identity", "relu"):
            for panel, _, _ in panel_specs(task, cfg):
                for metric in (*ACCURACY_METRICS, *DIAGNOSTIC_METRICS):
                    diffs, used = [], []
                    for seed in seeds:
                        a, b = by_key[task, seed, first], by_key[task, seed, "tanh"]
                        if (a["status"] == b["status"] == "ok"
                                and a["evaluations"][panel]["status"] == "ok"
                                and b["evaluations"][panel]["status"] == "ok"):
                            diffs.append(a["evaluations"][panel][metric]
                                         - b["evaluations"][panel][metric])
                            used.append(seed)
                    contrasts.append({"contrast": first + "_minus_tanh", "task": task,
                                      "panel": panel, "metric": metric,
                                      "paired_seeds": len(diffs), "seed_ids": used,
                                      "differences_by_seed": diffs,
                                      "mean_difference": statistics.mean(diffs) if diffs else None,
                                      "std_difference": statistics.stdev(diffs) if len(diffs)>1 else 0.0,
                                      "ci95": ci95(diffs) if diffs else None,
                                      "first_wins": sum(v > 0 for v in diffs),
                                      "ties": sum(v == 0 for v in diffs),
                                      "tanh_wins": sum(v < 0 for v in diffs)})
    return arm_rows, panel_rows, contrasts


def write_csv(path, rows):
    rows = [{k: json.dumps(v) if isinstance(v, (list, dict)) else v
             for k, v in row.items()} for row in rows]
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def batch_stream_fingerprint(task, seed, cfg, tok):
    offset = 0 if task == "delayed_copy" else 1_000_000
    lengths = cfg.train_copy_lengths if task == "delayed_copy" else cfg.train_pal_lengths
    rng = np.random.default_rng(10_000 + seed + offset)
    digest = hashlib.sha256()
    for step in range(1, 4):
        upper = min(lengths[1], 1 + (step - 1) // 200)
        batch = make_batch(task, cfg.batch_size, (min(lengths[0], upper), upper), cfg, tok, rng)
        for tensor in batch:
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def validate(runs, seeds, cfg, paired_initializations, counts, batch_checks):
    expected = {(task, seed, readout) for task in TASKS for seed in range(seeds)
                for readout in READOUTS}
    observed = {(r["task"], r["seed"], r["readout"]) for r in runs}
    complete = all(set(r["evaluations"]) == {p[0] for p in panel_specs(r["task"], cfg)}
                   and all(e["status"] == "ok" and all(metric in e for metric in
                           (*ACCURACY_METRICS, *DIAGNOSTIC_METRICS,
                            "tanh_readout_saturation_fraction"))
                           for e in r["evaluations"].values())
                   for r in runs if r["status"] == "ok")
    checks = {"expected_run_count": len(expected), "observed_run_count": len(runs),
              "run_count_ok": len(runs) == len(expected),
              "duplicate_keys": len(runs) - len(observed),
              "missing_keys": [list(x) for x in sorted(expected - observed)],
              "unexpected_keys": [list(x) for x in sorted(observed - expected)],
              "parameter_counts_equal": len(set(counts.values())) == 1,
              "all_run_parameter_counts_match": all(r["parameter_count"] == counts[r["readout"]]
                                                      for r in runs),
              "paired_initializations_equal": all(paired_initializations),
              "paired_batch_stream_checks_passed": all(batch_checks),
              "successful_outputs_complete_and_finite": complete}
    checks["passed"] = all((checks["run_count_ok"], checks["duplicate_keys"] == 0,
                            not checks["missing_keys"], not checks["unexpected_keys"],
                            checks["parameter_counts_equal"], checks["all_run_parameter_counts_match"],
                            checks["paired_initializations_equal"],
                            checks["paired_batch_stream_checks_passed"], complete))
    return checks


def make_plot(panel_rows, out_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, task in zip(axes, TASKS):
        rows = [r for r in panel_rows if r["task"] == task]
        panels = [p[0] for p in panel_specs(task, Config())]
        width = .25
        x = np.arange(len(panels))
        for index, readout in enumerate(READOUTS):
            values = [next(r for r in rows if r["panel"] == p and r["readout"] == readout)
                      ["token_accuracy_mean"] for p in panels]
            axis.bar(x + (index - 1) * width, values, width, label=readout)
        axis.set_xticks(x, [p.replace("_", "\n") for p in panels], fontsize=7)
        axis.set_title(task.replace("_", " "))
        axis.set_ylim(0, 1.02)
    axes[0].set_ylabel("mean token accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "output_activation_bptt.png", dpi=150)
    plt.close(fig)


def worker(job):
    readout, task, seed, cfg = job
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    tok = Tokens(cfg.symbols)
    started = time.perf_counter()
    try:
        return train_one(readout, task, seed, cfg, tok)
    except Exception as exc:
        import traceback
        return failed_run(readout, task, seed, cfg, tok,
                          f"worker exception: {type(exc).__name__}: {exc}", None, started,
                          traceback.format_exc())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--copy-steps", type=int)
    parser.add_argument("--palindrome-steps", type=int)
    parser.add_argument("--output", type=Path, default=Path("output_activation_bptt_results.json"))
    args = parser.parse_args()
    cfg = Config(**{k: v for k, v in (("copy_steps", args.copy_steps),
                                      ("palindrome_steps", args.palindrome_steps)) if v is not None})
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    tok = Tokens(cfg.symbols)
    counts = {r: count_parameters(MemoryRNN(r, tok.vocab_in, cfg.symbols, cfg)) for r in READOUTS}
    initialization_checks, batch_checks = [], []
    for task in TASKS:
        for seed in range(args.seeds):
            states = []
            for readout in READOUTS:
                torch.manual_seed(seed)
                states.append(MemoryRNN(readout, tok.vocab_in, cfg.symbols, cfg).state_dict())
            initialization_checks.append(all(torch.equal(states[0][key], state[key])
                                             for state in states[1:] for key in states[0]))
            # Each arm reconstructs this same task/seed stream; compare explicit fingerprints.
            fingerprints = [batch_stream_fingerprint(task, seed, cfg, tok) for _ in READOUTS]
            batch_checks.append(len(set(fingerprints)) == 1)
    jobs = [(readout, task, seed, cfg) for task in TASKS for seed in range(args.seeds)
            for readout in READOUTS]
    started = time.perf_counter()
    if args.workers == 1:
        runs = [worker(job) for job in jobs]
    else:
        runs = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            for run in executor.map(worker, jobs):
                runs.append(run)
                print(f"finished {run['task']} seed={run['seed']} readout={run['readout']} "
                      f"status={run['status']} time={run['runtime_seconds']:.1f}s", flush=True)
    runs.sort(key=lambda r: (TASKS.index(r["task"]), r["seed"], READOUTS.index(r["readout"])))
    arm_rows, panel_rows, contrasts = summarize(runs, cfg)
    validation = validate(runs, args.seeds, cfg, initialization_checks, counts, batch_checks)
    runtime = time.perf_counter() - started
    payload = {"metadata": {"command": shlex.join([sys.executable, *sys.argv]),
                             "python_version": platform.python_version(),
                             "torch_version": torch.__version__, "device": "cpu",
                             "workers": args.workers, "deterministic_algorithms": True,
                             "full_bptt": True, "optimizer": "Adam",
                             "paired_batch_streams": True,
                             "parameter_counts": counts,
                             "locked_topology": "forked",
                             "locked_recurrence": "h_next = tanh(W_state(z))",
                             "varied_component": "logits = W_out(f(z)); f in identity,tanh,relu",
                             "confidence_interval": "two-sided 95% Student t interval over paired seed differences",
                             "config": asdict(cfg), "total_runtime_seconds": runtime},
               "validation": validation, "arm_summaries": arm_rows,
               "panel_summaries": panel_rows, "paired_comparisons": contrasts, "runs": runs}
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    csv_path = args.output.with_name("output_activation_bptt_summary.csv")
    write_csv(csv_path, ([{"row_type": "arm_summary", **r} for r in arm_rows]
                         + [{"row_type": "evaluation_panel", **r} for r in panel_rows]
                         + [{"row_type": "paired_comparison", **r} for r in contrasts]))
    make_plot(panel_rows, args.output.parent)
    print(json.dumps({"validation": validation, "arm_summaries": arm_rows}, indent=2))
    print(f"wrote {args.output} and {csv_path}; total runtime {runtime:.1f}s")
    if not validation["passed"]:
        raise SystemExit("validation failed")


if __name__ == "__main__":
    main()
