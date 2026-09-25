#!/usr/bin/env python3
"""Standalone CPU experiment comparing three vanilla-BPTT RNN readout taps."""

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


ARMS = ("tapped_post_tanh", "tapped_pre_tanh_relu", "forked")
TASKS = ("delayed_copy", "two_palindrome")


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
    """All arms own and execute the exact same modules; only readout input differs."""

    def __init__(self, arm: str, vocab_in: int, outputs: int, cfg: Config):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm = arm
        self.embedding = nn.Embedding(vocab_in, cfg.embed_dim)
        self.trunk1 = nn.Linear(cfg.embed_dim + cfg.hidden_dim, cfg.hidden_dim)
        self.trunk2 = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.W_state = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.W_out = nn.Linear(cfg.hidden_dim, outputs)
        self.hidden_dim = cfg.hidden_dim
        # Identity-centered state dynamics keep this vanilla RNN trainable over
        # delays without gates; all arms receive precisely the same setup.
        with torch.no_grad():
            self.W_state.weight.copy_(torch.eye(cfg.hidden_dim))
            self.W_state.bias.zero_()
            self.trunk2.weight.mul_(0.1)
            self.trunk2.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps = x.shape
        h = torch.zeros(batch, self.hidden_dim, device=x.device)
        logits = []
        for t in range(steps):
            transformed = torch.relu(self.trunk1(torch.cat((self.embedding(x[:, t]), h), dim=-1)))
            # A parameter-free residual still leaves two Linear+ReLU trunk
            # transformations, while giving the ungated carrier a stable path.
            z = h + torch.relu(self.trunk2(transformed))
            u = self.W_state(z)
            h = torch.tanh(u)
            # W_out executes exactly once in every arm.
            if self.arm == "tapped_post_tanh":
                readout = h
            elif self.arm == "tapped_pre_tanh_relu":
                readout = torch.relu(u)
            else:
                readout = z
            logits.append(self.W_out(readout))
        return torch.stack(logits, dim=1)


def make_batch(
    task: str,
    batch_size: int,
    length_range: tuple[int, int],
    cfg: Config,
    tok: Tokens,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def masked_metrics(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, float]:
    selected = logits[mask]
    targets = y[mask]
    loss = nn.functional.cross_entropy(selected, targets)
    accuracy = (selected.argmax(-1) == targets).float().mean().item()
    return loss, accuracy


@torch.no_grad()
def evaluate(model: nn.Module, task: str, lengths: tuple[int, int], cfg: Config,
             tok: Tokens, seed: int, batches: int | None = None) -> dict[str, float]:
    model.eval()
    rng = np.random.default_rng(seed)
    losses, correct, total, exact, samples = [], 0, 0, 0, 0
    for _ in range(cfg.eval_batches if batches is None else batches):
        x, y, mask = make_batch(task, cfg.batch_size, lengths, cfg, tok, rng)
        logits = model(x)
        loss, _ = masked_metrics(logits, y, mask)
        losses.append(loss.item())
        predictions = logits.argmax(-1)
        matches = predictions == y
        correct += int(matches[mask].sum())
        total += int(mask.sum())
        exact += int((matches | ~mask).all(dim=1).sum())
        samples += x.shape[0]
    model.train()
    return {"loss": float(np.mean(losses)), "token_accuracy": correct / total,
            "exact_sequence_accuracy": exact / samples, "batches": len(losses),
            "samples": samples, "target_tokens": total}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one(arm: str, task: str, seed: int, cfg: Config, tok: Tokens) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = MemoryRNN(arm, tok.vocab_in, cfg.symbols, cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    # This stream depends only on task+seed, so every arm sees byte-identical batches.
    task_offset = 0 if task == "delayed_copy" else 1_000_000
    train_rng = np.random.default_rng(10_000 + seed + task_offset)
    lengths = cfg.train_copy_lengths if task == "delayed_copy" else cfg.train_pal_lengths
    long_lengths = cfg.long_copy_lengths if task == "delayed_copy" else cfg.long_pal_lengths
    steps = cfg.copy_steps if task == "delayed_copy" else cfg.palindrome_steps
    curves = []
    thresholds: dict[str, dict | None] = {"0.90": None, "0.99": None}
    started = time.perf_counter()

    for step in range(1, steps + 1):
        model.train()
        # A shared deterministic curriculum avoids the well-known cold-start
        # failure of plain tanh RNNs on copy tasks. Most optimization still
        # uses the complete variable-length training distribution.
        curriculum_upper = min(lengths[1], 1 + (step - 1) // 200)
        curriculum_lengths = (min(lengths[0], curriculum_upper), curriculum_upper)
        x, y, mask = make_batch(task, cfg.batch_size, curriculum_lengths, cfg, tok, train_rng)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss, train_acc = masked_metrics(logits, y, mask)
        loss.backward()  # Full BPTT: recurrent state is never detached.
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % cfg.eval_every == 0 or step == steps:
            evaluation = evaluate(
                model, task, lengths, cfg, tok, seed=20_000 + seed + task_offset
            )
            point = {"step": step, "examples": step * cfg.batch_size,
                     "train_loss": loss.item(), "train_accuracy": train_acc,
                     "validation_loss": evaluation["loss"],
                     "validation_accuracy": evaluation["token_accuracy"],
                     "validation_exact_sequence_accuracy": evaluation["exact_sequence_accuracy"]}
            curves.append(point)
            for label, threshold in (("0.90", 0.90), ("0.99", 0.99)):
                if thresholds[label] is None and evaluation["token_accuracy"] >= threshold:
                    thresholds[label] = {"step": step, "examples": step * cfg.batch_size}

    evaluations = {
        "in_distribution": evaluate(model, task, lengths, cfg, tok, 30_000 + seed + task_offset),
        "longer_lengths": evaluate(model, task, long_lengths, cfg, tok, 40_000 + seed + task_offset),
    }
    if task == "two_palindrome":
        for n in (3, 4):
            evaluations[f"fixed_block_length_{n}"] = evaluate(
                model, task, (n, n), cfg, tok,
                50_000 + n * 10_000 + seed + task_offset,
                batches=cfg.fixed_eval_batches,
            )
    return {
        "arm": arm, "task": task, "seed": seed,
        "parameter_count": count_parameters(model), "steps": steps,
        "thresholds": thresholds, "evaluations": evaluations,
        "final_in_distribution_loss": evaluations["in_distribution"]["loss"],
        "final_in_distribution_accuracy": evaluations["in_distribution"]["token_accuracy"],
        "final_in_distribution_exact_sequence_accuracy": evaluations["in_distribution"]["exact_sequence_accuracy"],
        "longer_length_loss": evaluations["longer_lengths"]["loss"],
        "longer_length_accuracy": evaluations["longer_lengths"]["token_accuracy"],
        "longer_length_exact_sequence_accuracy": evaluations["longer_lengths"]["exact_sequence_accuracy"],
        "runtime_seconds": time.perf_counter() - started, "curve": curves,
    }


def _ci95(values: list[float]) -> list[float]:
    """Two-sided 95% t interval (critical values for the experiment's n)."""
    mean = statistics.mean(values)
    if len(values) < 2:
        return [mean, mean]
    critical = {14: 2.1447866879, 19: 2.0930240544}.get(len(values) - 1, 1.96)
    half = critical * statistics.stdev(values) / math.sqrt(len(values))
    return [mean - half, mean + half]


def summarize(runs: list[dict]) -> list[dict]:
    summaries = []
    for task in TASKS:
        for arm in ARMS:
            group = [r for r in runs if r["task"] == task and r["arm"] == arm]
            if not group:
                continue
            row = {"task": task, "arm": arm, "seeds": len(group)}
            for key in ("final_in_distribution_accuracy", "final_in_distribution_exact_sequence_accuracy",
                        "longer_length_accuracy", "longer_length_exact_sequence_accuracy", "runtime_seconds"):
                values = [r[key] for r in group]
                row[key + "_mean"] = statistics.mean(values)
                row[key + "_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            for label in ("0.90", "0.99"):
                reached = [r["thresholds"][label]["step"] for r in group if r["thresholds"][label]]
                prefix = "threshold_" + label.replace(".", "_")
                row[prefix + "_successes"] = len(reached)
                row[prefix + "_step_mean_reached"] = statistics.mean(reached) if reached else None
                row[prefix + "_step_std_reached"] = statistics.stdev(reached) if len(reached) > 1 else (0.0 if reached else None)
            summaries.append(row)
    return summaries


def paired_comparisons(runs: list[dict]) -> list[dict]:
    """Paired differences are first arm minus second arm on identical data streams."""
    comparisons = []
    pairs = (("forked", "tapped_post_tanh", "primary"),
             ("tapped_pre_tanh_relu", "tapped_post_tanh", "secondary"),
             ("forked", "tapped_pre_tanh_relu", "secondary"))
    by_key = {(r["task"], r["seed"], r["arm"]): r for r in runs}
    for task in TASKS:
        panels = ["in_distribution", "longer_lengths"]
        if task == "two_palindrome":
            panels += ["fixed_block_length_3", "fixed_block_length_4"]
        seeds = sorted({r["seed"] for r in runs if r["task"] == task})
        for panel in panels:
            for metric in ("token_accuracy", "exact_sequence_accuracy"):
                for first, second, role in pairs:
                    differences = [by_key[task, s, first]["evaluations"][panel][metric]
                                   - by_key[task, s, second]["evaluations"][panel][metric]
                                   for s in seeds]
                    comparisons.append({
                        "task": task, "panel": panel, "metric": metric,
                        "first_arm": first, "second_arm": second, "role": role,
                        "seeds": len(seeds), "differences_by_seed": differences,
                        "mean_difference": statistics.mean(differences),
                        "std_difference": statistics.stdev(differences),
                        "ci95": _ci95(differences),
                        "first_wins": sum(d > 0 for d in differences),
                        "ties": sum(d == 0 for d in differences),
                        "second_wins": sum(d < 0 for d in differences),
                    })
    return comparisons


def panel_summaries(runs: list[dict]) -> list[dict]:
    rows = []
    for task in TASKS:
        panels = ["in_distribution", "longer_lengths"]
        if task == "two_palindrome":
            panels += ["fixed_block_length_3", "fixed_block_length_4"]
        for panel in panels:
            for arm in ARMS:
                group = [r for r in runs if r["task"] == task and r["arm"] == arm]
                row = {"task": task, "panel": panel, "arm": arm, "seeds": len(group)}
                for metric in ("token_accuracy", "exact_sequence_accuracy"):
                    values = [r["evaluations"][panel][metric] for r in group]
                    row[metric + "_mean"] = statistics.mean(values)
                    row[metric + "_std"] = statistics.stdev(values)
                    row[metric + "_ci95"] = _ci95(values)
                rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(runs: list[dict], out_dir: Path, prefix: str = "expanded_") -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    for task in TASKS:
        if not any(r["task"] == task for r in runs):
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for arm in ARMS:
            group = [r for r in runs if r["task"] == task and r["arm"] == arm]
            if not group:
                continue
            steps = [p["step"] for p in group[0]["curve"]]
            vals = np.array([[p["validation_accuracy"] for p in r["curve"]] for r in group])
            mean, std = vals.mean(0), vals.std(0, ddof=1) if len(group) > 1 else np.zeros_like(vals[0])
            axes[0].plot(steps, mean, label=arm)
            axes[0].fill_between(steps, mean - std, mean + std, alpha=.15)
            final = [r["longer_length_accuracy"] for r in group]
            axes[1].scatter([ARMS.index(arm)] * len(final), final, alpha=.75)
            axes[1].bar(ARMS.index(arm), np.mean(final), alpha=.3)
        axes[0].set(title="Validation learning curve (mean ± SD)", xlabel="optimizer step", ylabel="recall accuracy", ylim=(0, 1.03))
        axes[0].legend(fontsize=8)
        axes[1].set(title="Held-out longer lengths", ylabel="recall accuracy", ylim=(0, 1.03), xticks=range(3), xticklabels=["post tanh", "pre tanh\nReLU", "forked"])
        fig.suptitle(task.replace("_", " "))
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}{task}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--copy-steps", type=int, default=None)
    parser.add_argument("--palindrome-steps", type=int, default=None)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--output", type=Path, default=Path("expanded_results.json"))
    args = parser.parse_args()
    cfg = Config(**{k: v for k, v in {"copy_steps": args.copy_steps,
                                      "palindrome_steps": args.palindrome_steps}.items() if v is not None})
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    tok = Tokens(cfg.symbols)
    probe_counts = {arm: count_parameters(MemoryRNN(arm, tok.vocab_in, cfg.symbols, cfg)) for arm in ARMS}
    assert len(set(probe_counts.values())) == 1, probe_counts

    total_started = time.perf_counter()
    jobs = [(arm, task, seed, cfg) for task in args.tasks for seed in range(args.seeds) for arm in args.arms]
    runs = []
    if args.workers == 1:
        runs = [_train_worker(job) for job in jobs]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            for run in executor.map(_train_worker, jobs):
                runs.append(run)
                print(f"finished task={run['task']} seed={run['seed']} arm={run['arm']} "
                      f"in={run['final_in_distribution_accuracy']:.3f} "
                      f"long={run['longer_length_accuracy']:.3f} time={run['runtime_seconds']:.1f}s", flush=True)
    runs.sort(key=lambda r: (TASKS.index(r["task"]), r["seed"], ARMS.index(r["arm"])))
    summaries = summarize(runs)
    panels = panel_summaries(runs)
    comparisons = paired_comparisons(runs)
    payload = {
        "metadata": {
            "command": shlex.join([sys.executable, *sys.argv]),
            "torch_version": torch.__version__, "python_version": platform.python_version(),
            "device": "cpu", "deterministic_algorithms": True,
            "parameter_counts": probe_counts, "config": asdict(cfg),
            "workers": args.workers, "paired_batch_streams": True,
            "primary_comparison": "forked minus tapped_post_tanh",
            "confidence_interval": "two-sided 95% Student t interval over paired seed differences",
            "total_runtime_seconds": time.perf_counter() - total_started,
        },
        "summaries": summaries, "panel_summaries": panels,
        "paired_comparisons": comparisons, "runs": runs,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    csv_rows = [{"row_type": "arm_summary", **row} for row in summaries]
    csv_rows += [{"row_type": "evaluation_panel", **row,
                  "token_accuracy_ci95": json.dumps(row["token_accuracy_ci95"]),
                  "exact_sequence_accuracy_ci95": json.dumps(row["exact_sequence_accuracy_ci95"])}
                 for row in panels]
    csv_rows += [{"row_type": "paired_comparison", **{k: v for k, v in row.items()
                  if k != "differences_by_seed"}, "ci95": json.dumps(row["ci95"])} for row in comparisons]
    write_csv(args.output.with_name("expanded_summary.csv"), csv_rows)
    make_plots(runs, args.output.parent)
    print(json.dumps(summaries, indent=2))
    print(f"wrote {args.output}; total runtime {payload['metadata']['total_runtime_seconds']:.1f}s")


def _train_worker(job: tuple[str, str, int, Config]) -> dict:
    arm, task, seed, cfg = job
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return train_one(arm, task, seed, cfg, Tokens(cfg.symbols))


if __name__ == "__main__":
    main()
