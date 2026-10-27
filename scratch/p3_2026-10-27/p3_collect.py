"""P3 collector (stdlib only): one row per run directory RUNS/KR-<arm>-s<seed>, written as JSON and a markdown table.

usage: python scratch/p3_2026-10-27/p3_collect.py RUNS [--out RUNS/p3_summary]
Reads what p3_train.sh writes: train.log (interval metrics, "Non-finite loss"), final_eval.json (heldout.py) and
sg/*.json (p3_sg.py). Status (pre-registered in P3.md):
  nan        training stopped on a non-finite loss
  collapsed  training recall_acc reached >= 0.9 in some interval, and final held-out strict recall < 0.5
  no_learn   training recall_acc never reached 0.9 and final strict recall < 0.5
  ok         otherwise
"""
import argparse
import glob
import json
import os
import re

ARMS = ("ctl", "g1", "g2", "g4", "g2w64", "g4cap10")
SG_KEYS = ("ratio_p", "ratio_p_raw", "ratio_gain", "ratio_g", "value_share_of_lead", "gain_share_of_lead",
           "store_largest", "store_largest_equal_p", "store_largest_equal_key", "store_largest_no_gain",
           "signal_share", "cos_store_query")


def parse_log(path):
    intervals, it, cur, nonfinite = [], None, None, False
    with open(path, errors="replace") as f:
        for line in f:
            m = re.match(r"--- Interval metrics \(ending @ iter (\d+)", line)
            if m:
                cur = {"iter": int(m.group(1))}
                intervals.append(cur)
                continue
            if line.startswith("Non-finite loss"):
                nonfinite = True
            m = re.match(r"  ([A-Za-z0-9_/]+): (\S+)$", line)
            if m and cur is not None:
                try:
                    cur[m.group(1)] = float(m.group(2))
                except ValueError:
                    pass
            elif line.startswith("---") and not line.startswith("--- Interval"):
                cur = None
    return intervals, nonfinite


def row(d):
    name = os.path.basename(d)
    m = re.match(r"KR-(.+)-s(\d+)$", name)
    r = {"run": name, "arm": m.group(1) if m else "?", "seed": int(m.group(2)) if m else None}
    intervals, nonfinite = parse_log(os.path.join(d, "train.log")) if os.path.exists(os.path.join(d, "train.log")) else ([], False)
    rec = [(x["iter"], x["recall_acc"]) for x in intervals if "recall_acc" in x]
    r["last_iter"] = intervals[-1]["iter"] if intervals else None
    r["first_iter_recall_ge_0.9"] = next((i for i, v in rec if v >= 0.9), None)
    r["train_recall_last"] = rec[-1][1] if rec else None
    for k in ("write_gain_mean", "write_gain_max", "write_gain_frac_above_1"):
        r[k + "_last"] = intervals[-1].get(k) if intervals else None
    r["write_gain_mean_max_over_run"] = max((x["write_gain_mean"] for x in intervals if "write_gain_mean" in x), default=None)
    try:
        with open(os.path.join(d, "final_eval.json")) as f:
            ev = json.load(f)
        for p in ("observed", "strict", "no_fast"):
            r[f"{p}_recall"] = ev.get(f"heldout_{p}/recall_acc")
        r["eval_iter"] = ev.get("iteration")
    except (OSError, ValueError):
        pass
    strict = r.get("strict_recall")
    if nonfinite:
        r["status"] = "nan"
    elif strict is not None and strict < 0.5:
        r["status"] = "collapsed" if r["first_iter_recall_ge_0.9"] is not None else "no_learn"
    else:
        r["status"] = "ok" if strict is not None else "unfinished"
    r["sg"] = {}
    for path in sorted(glob.glob(os.path.join(d, "sg", "*.json"))):
        with open(path) as f:
            sg = json.load(f)
        r["sg"][str(sg["iteration"])] = {"recall": sg["recall"], "medians": sg.get("write_gain", {}).get("medians"),
                                          **{layer: {k: v.get(k) for k in SG_KEYS}
                                             for layer, v in sg["layers"].items() if layer in ("L1", "L2")}}
    return r


def fmt(v):
    return "" if v is None else (f"{v:.3g}" if isinstance(v, float) else str(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.runs, "p3_summary")
    dirs = sorted(glob.glob(os.path.join(a.runs, "KR-*-s*")),
                  key=lambda d: (ARMS.index(row_arm(d)) if row_arm(d) in ARMS else 99, d))
    rows = [row(d) for d in dirs]
    with open(out + ".json", "w") as f:
        json.dump(rows, f, indent=1)
    lines = ["| run | status | last iter | it recall>=.9 | obs | strict | no_fast | gain mean (last) | gain max (last) |"
             " SG iter | layer | ratio_p | raw | gain | key ratio | value share | equal-p | equal-key | no-gain |",
             "|" + "---|" * 19]
    for r in rows:
        head = [r["run"], r["status"], fmt(r["last_iter"]), fmt(r["first_iter_recall_ge_0.9"]), fmt(r.get("observed_recall")),
                fmt(r.get("strict_recall")), fmt(r.get("no_fast_recall")), fmt(r["write_gain_mean_last"]),
                fmt(r["write_gain_max_last"])]
        sgs = [(it, layer, v) for it, s in sorted(r["sg"].items(), key=lambda x: int(x[0]))
               for layer, v in s.items() if layer in ("L1", "L2")]
        if not sgs:
            lines.append("| " + " | ".join(head + [""] * 10) + " |")
        for k, (it, layer, v) in enumerate(sgs):
            cells = (head if k == 0 else [""] * 9) + [it, layer] + [fmt(v.get(x)) for x in (
                "ratio_p", "ratio_p_raw", "ratio_gain", "ratio_g", "value_share_of_lead", "store_largest_equal_p",
                "store_largest_equal_key", "store_largest_no_gain")]
            lines.append("| " + " | ".join(cells) + " |")
    with open(out + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def row_arm(d):
    m = re.match(r"KR-(.+)-s\d+$", os.path.basename(d))
    return m.group(1) if m else "?"


if __name__ == "__main__":
    main()
