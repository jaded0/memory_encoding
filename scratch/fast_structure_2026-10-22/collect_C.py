"""Batch FS part C (C6 on 3pal): one row per run from train.log and final_eval.json, V2/battery scoring.
Usage: python collect_C.py RUNS_DIR OUT.jsonl"""
import glob
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(__file__))
from logparse import blowup_onset, intervals, memory_onset  # noqa: E402

runs, out = sys.argv[1], sys.argv[2]
rows = []
for d in sorted(glob.glob(os.path.join(runs, "C-*"))):
    name = os.path.basename(d)
    arm, seed = name[2:].rsplit("-s", 1)
    log = intervals(os.path.join(d, "train.log")) if os.path.exists(os.path.join(d, "train.log")) else []
    last_iter = log[-1][0] if log else 0
    onset = blowup_onset(log)
    final_loss = log[-1][1].get("loss", float("nan")) if log else float("nan")
    rates = [m["iters_per_sec"] for _, m in log[1:] if "iters_per_sec" in m]
    row = {"run": name, "arm": arm, "seed": int(seed), "last_iter": last_iter, "blowup_onset": onset,
           "memory_onset": memory_onset(log), "final_train_loss": final_loss,
           "max_loss_after_30k": max([m.get("loss", float("nan")) for i, m in log if i > 30000] or [float("nan")]),
           "median_its": statistics.median(rates) if rates else None,
           "train_recall_last": log[-1][1].get("recall_acc") if log else None}
    path = os.path.join(d, "final_eval.json")
    if os.path.exists(path):
        ev = json.load(open(path))
        for p in ("strict", "observed", "no_fast"):
            row[f"{p}_recall"] = ev.get(f"heldout_{p}/recall_acc")
            row[f"{p}_loss"] = ev.get(f"heldout_{p}/loss")
        row["eval_iteration"] = ev.get("iteration")
    complete = last_iter >= 600000
    row["complete"] = complete
    strict = row.get("strict_recall")
    row["useful_healthy"] = bool(complete and onset is None and math.isfinite(final_loss) and final_loss < 5
                                 and strict is not None and strict >= 0.30)
    rows.append(row)
with open(out, "w") as handle:
    for row in rows:
        handle.write(json.dumps(row) + "\n")
print(f"{'run':22s} {'last':>7s} {'onset':>7s} {'mem':>7s} {'loss':>9s} {'strict':>6s} {'obs':>6s} {'nofast':>6s} {'hloss':>8s} {'it/s':>6s} healthy")
for r in rows:
    f = lambda v, fmt: (fmt % v) if isinstance(v, (int, float)) and v is not None else "-"
    print(f"{r['run']:22s} {r['last_iter']:7d} {f(r['blowup_onset'], '%d'):>7s} {f(r['memory_onset'], '%d'):>7s} "
          f"{f(r['final_train_loss'], '%.3g'):>9s} {f(r.get('strict_recall'), '%.3f'):>6s} {f(r.get('observed_recall'), '%.3f'):>6s} "
          f"{f(r.get('no_fast_recall'), '%.3f'):>6s} {f(r.get('strict_loss'), '%.3g'):>8s} {f(r['median_its'], '%.1f'):>6s} {r['useful_healthy']}")
