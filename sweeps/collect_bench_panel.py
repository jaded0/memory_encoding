#!/usr/bin/env python
"""Tabulate the benchmark panel (sweeps/orc_bench_panel.sbatch) from its job logs.

For each run (task x arm x seed) it reads the interval-metrics blocks of the SLURM log
("--- Interval metrics (ending @ iter N, whole batch) ---" followed by "  key: value" lines, with the
held-out lines heldout_<protocol>/<metric> when an evaluation ran in that interval) and reports:
final and best recall_acc, the first iteration reaching recall_acc >= --threshold, the latest
held-out recall_acc under each protocol (observed / strict / no_fast / free_running), the final
answer-class split for kv tasks, the last iteration reached, and a status (done, collapsed, running).
Then a task x arm table of seed means (final recall, held-out strict / no-fast).

    python sweeps/collect_bench_panel.py [LOG_OR_DIR ...] [--root DIR] [--csv out.csv] [--threshold 0.5]

Default: every benchpanel_*.out under /home/jaded79/memory_encoding_benchmarks/bench_panel/logs.
A requeued run appends to its log; the last block for an iteration wins.
"""
import argparse
import csv
import glob
import os
import re
import statistics
from collections import defaultdict

DEFAULT_ROOT = "/home/jaded79/memory_encoding_benchmarks/bench_panel"
HEADER = re.compile(r"^--- Interval metrics \(ending @ iter (\d+)")
VALUE = re.compile(r"^\s+([A-Za-z_][\w/]*): (-?[\d.]+(?:e[-+]?\d+)?|nan|inf)\s*$")
RUN = re.compile(r"^Run: (\S+) arm (\S+), (\S+), seed (\d+), (\d+) iterations")
PROTOCOLS = ("observed", "strict", "no_fast", "free_running")


def parse_log(path):
    """{'run', 'arm', 'task', 'seed', 'n_iters', 'blocks': {iter: {key: value}}, 'status'}."""
    info = {"run": os.path.basename(path), "arm": "?", "task": "?", "seed": "?", "n_iters": None}
    blocks, current, text = {}, None, []
    with open(path, errors="replace") as handle:
        for line in handle:
            text.append(line)
            match = RUN.match(line)
            if match:
                info.update(run=match.group(1), arm=match.group(2), task=match.group(3),
                            seed=match.group(4), n_iters=int(match.group(5)))
                continue
            match = HEADER.match(line)
            if match:
                current = blocks.setdefault(int(match.group(1)), {})
                current.clear()
                continue
            if current is not None:
                match = VALUE.match(line)
                if match:
                    current[match.group(1)] = float(match.group(2))
                elif not line.startswith("  ") or line.strip() == "":
                    current = None
    joined = "".join(text)
    last = max(blocks, default=0)
    if "--- Finished" in joined:
        status = "done"
    elif "--- Collapsed" in joined or re.search(r"^Early stopping", joined, re.M):
        status = "collapsed"
    elif "failed with exit" in joined or "Traceback" in joined:
        status = "FAILED"
    else:
        status = "running"
    info.update(blocks=blocks, status=status, last_iter=last)
    return info


def summarize(info, threshold):
    blocks = info["blocks"]
    row = {k: info[k] for k in ("run", "task", "arm", "seed", "status", "last_iter", "n_iters")}
    iters = sorted(i for i, b in blocks.items() if "recall_acc" in b)
    recalls = [blocks[i]["recall_acc"] for i in iters]
    row["final_recall"] = recalls[-1] if recalls else None
    row["best_recall"] = max(recalls) if recalls else None
    row["iter_ge_threshold"] = next((i for i, r in zip(iters, recalls) if r >= threshold), None)
    row["final_loss"] = blocks[iters[-1]].get("loss") if iters else None
    for protocol in PROTOCOLS:  # latest evaluation that has this protocol
        key = f"heldout_{protocol}/recall_acc"
        held = [i for i in sorted(blocks) if key in blocks[i]]
        row[f"heldout_{protocol}"] = blocks[held[-1]][key] if held else None
        row[f"heldout_{protocol}_iter"] = held[-1] if held else None
    held = [i for i in sorted(blocks) if "heldout_strict/first_answer_acc" in blocks[i]]
    row["heldout_strict_first_answer"] = blocks[held[-1]]["heldout_strict/first_answer_acc"] if held else None
    for cls in ("correct", "stale", "wrong_key", "other"):
        row[f"kv_{cls}"] = blocks[iters[-1]].get(f"kv_{cls}") if iters else None
    return row


def fmt(value, digits=3):
    if value is None:
        return "-"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", help="log files or directories of *.out logs")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--threshold", type=float, default=0.5, help="recall_acc for the speed column")
    args = parser.parse_args(argv)
    paths = []
    for path in args.paths or [os.path.join(args.root, "logs")]:
        paths += sorted(glob.glob(os.path.join(path, "benchpanel_*.out"))) if os.path.isdir(path) else [path]
    rows = [summarize(parse_log(path), args.threshold) for path in paths]
    rows.sort(key=lambda r: (r["task"], r["arm"], str(r["seed"])))
    if not rows:
        print("no logs found")
        return
    cols = ["task", "arm", "seed", "status", "last_iter", "final_recall", "best_recall", "iter_ge_threshold",
            "heldout_observed", "heldout_strict", "heldout_no_fast", "heldout_free_running", "heldout_strict_first_answer"]
    print("| " + " | ".join(c.replace("iter_ge_threshold", f"iter>={args.threshold}") for c in cols) + " |")
    print("|" + "---|" * len(cols))
    for row in rows:
        print("| " + " | ".join(fmt(row[c]) for c in cols) + " |")

    groups = defaultdict(list)
    for row in rows:
        groups[(row["task"], row["arm"])].append(row)
    mean = lambda rs, key: (fmt(statistics.mean(r[key] for r in rs if r[key] is not None))
                            if any(r[key] is not None for r in rs) else "-")
    print("\nSeed means (final recall_acc; held-out strict / no_fast at the latest evaluation; n seeds, statuses):\n")
    print("| task | arm | final recall | best recall | held-out strict | held-out no-fast | seeds | status |")
    print("|---|---|---|---|---|---|---|---|")
    for (task, arm), rs in sorted(groups.items()):
        print(f"| {task} | {arm} | {mean(rs, 'final_recall')} | {mean(rs, 'best_recall')} | "
              f"{mean(rs, 'heldout_strict')} | {mean(rs, 'heldout_no_fast')} | {len(rs)} | "
              f"{','.join(sorted({r['status'] for r in rs}))} |")
    if args.csv:
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
