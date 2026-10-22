"""Batch FS part L (mask load test): one row per run and checkpoint from load_<it>k.json, train.log and the
loop traces. AREA = mean strict recall over the evaluated loads; K50 = first K (linear interpolation
between grid loads) where strict recall falls below 0.5 (None: never within the grid, "52+").
Usage: python collect_L.py RUNS_DIR OUT.jsonl"""
import glob
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from logparse import blowup_onset, intervals, memory_onset  # noqa: E402


def k50(curve):
    """curve: [(K, recall)] sorted by K."""
    if not curve or curve[0][1] < 0.5:
        return curve[0][0] if curve else None
    for (k0, r0), (k1, r1) in zip(curve, curve[1:]):
        if r1 < 0.5 <= r0:
            return k0 + (r0 - 0.5) / (r0 - r1) * (k1 - k0)
    return None


def trace_ratios(d):
    """{iteration: [median over steps and rows of fast_drive / slow_drive, per trunk layer]} from traces/."""
    out = {}
    for path in sorted(glob.glob(os.path.join(d, "traces", "trace_*.pt"))):
        it = int(os.path.basename(path)[6:-3])
        tr = torch.load(path, weights_only=False)["traces"]
        fast, slow = tr["fast_drive"], tr["slow_drive"]  # [T, B, L] (trunk layers then i2h)
        ratio = fast / slow.clamp_min(1e-12)
        valid = slow > 0
        out[it] = [float(ratio[..., l][valid[..., l]].median()) if valid[..., l].any() else None
                   for l in range(ratio.shape[-1] - 1)]
    return out


runs, out = sys.argv[1], sys.argv[2]
rows = []
for d in sorted(glob.glob(os.path.join(runs, "L-*"))):
    name = os.path.basename(d)
    arm, seed = name[2:].rsplit("-s", 1)
    log = intervals(os.path.join(d, "train.log")) if os.path.exists(os.path.join(d, "train.log")) else []
    rates = [m["iters_per_sec"] for _, m in log[1:] if "iters_per_sec" in m]
    base = {"run": name, "arm": arm, "seed": int(seed), "last_iter": log[-1][0] if log else 0,
            "blowup_onset": blowup_onset(log), "memory_onset": memory_onset(log),
            "train_recall_last": log[-1][1].get("recall_acc") if log else None,
            "train_loss_last": log[-1][1].get("loss") if log else None,
            "median_its": statistics.median(rates) if rates else None,
            "train_recall_traj": {i: m.get("recall_acc") for i, m in log if i % 25000 == 0},
            "fast_slow_drive": trace_ratios(d)}
    evals = sorted(glob.glob(os.path.join(d, "load_*.json")))
    if not evals:
        rows.append({**base, "checkpoint": None})
    for path in evals:
        ev = json.load(open(path))
        loads = {int(k): v for k, v in ev["loads"].items()}
        curve = [(k, loads[k]["heldout_strict/recall_acc"]) for k in sorted(loads)]
        row = {**base, "checkpoint": os.path.basename(path)[5:-5], "iteration": ev["iteration"],
               "AREA": sum(r for _, r in curve) / len(curve), "K50": k50(curve),
               "strict": dict(curve)}
        for key, label in (("heldout_no_fast/recall_acc", "no_fast"), ("heldout_observed/recall_acc", "observed")):
            row[label] = {k: loads[k].get(key) for k in sorted(loads)}
        for ab in ("all", "only0", "only1", "only2", "drop0", "drop2", "none"):
            key = f"read_{ab}/recall_acc"
            if key in loads[curve[0][0]]:
                row[f"read_{ab}"] = {k: loads[k].get(key) for k in sorted(loads)}
        row["strict_by_query"] = {k: loads[k].get("strict_by_query") for k in sorted(loads)}
        row["strict_by_lag"] = {k: loads[k].get("strict_by_lag") for k in sorted(loads)}
        row["read_all_matches_strict"] = all(loads[k].get("read_all_matches_strict", True) for k in loads)
        rows.append(row)
with open(out, "w") as handle:
    for row in rows:
        handle.write(json.dumps(row) + "\n")
grid = sorted({k for r in rows if r.get("strict") for k in r["strict"]})
print(f"{'run':16s} {'ck':>5s} {'onset':>6s} {'mem':>6s} {'it/s':>5s} {'AREA':>6s} {'K50':>5s} " + " ".join(f"{k:>5d}" for k in grid))
for r in rows:
    if r.get("strict") is None:
        print(f"{r['run']:16s} (no eval) last {r['last_iter']} onset {r['blowup_onset']} mem {r['memory_onset']}")
        continue
    k50s = "52+" if r["K50"] is None else f"{r['K50']:.1f}"
    print(f"{r['run']:16s} {r['checkpoint']:>5s} {str(r['blowup_onset']):>6s} {str(r['memory_onset']):>6s} "
          f"{(r['median_its'] or 0):5.2f} {r['AREA']:6.3f} {k50s:>5s} " + " ".join(f"{r['strict'][k]:5.2f}" for k in grid))
