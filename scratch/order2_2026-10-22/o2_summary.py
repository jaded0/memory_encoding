"""Batch O2: markdown tables from o2_collect.py output.  usage: python o2_summary.py o2_runs.jsonl"""
import json
import sys
from collections import defaultdict

import numpy as np

SEEDS = [2718, 3141, 1618, 1414, 1732]
ARMS = ["lin", "par2", "par1", "rep2", "rep1", "rep2t1", "rep2t16", "par2t16"]


def fmt(v, nd=2):
    if v is None:
        return "–"
    if isinstance(v, float) and np.isnan(v):
        return "nan"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def load(path):
    runs = defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        runs[(r["task"], r["arm"])][r["seed"]] = r
    return runs


def k(v):
    return None if v is None else f"{v / 1000:g}k"


def outcome_table(runs, task):
    print(f"\n### {task}: held-out strict / no-fast recall (final), onset (train recall >= 0.5)\n")
    print("| arm | " + " | ".join(f"s{s}" for s in SEEDS) + " | mean strict | median onset |")
    print("|---|" + "---|" * (len(SEEDS) + 2))
    for arm in ARMS:
        rs = runs.get((task, arm))
        if not rs:
            continue
        cells, st, on = [], [], []
        for s in SEEDS:
            r = rs.get(s)
            if not r or r.get("strict") is None:
                cells.append("running" if r else "–")
                continue
            cells.append(f"{r['strict']:.3f} / {r['no_fast']:.2f} / {k(r['onset50']) or 'never'}")
            st.append(r["strict"])
            on.append(r["onset50"] if r["onset50"] is not None else np.inf)
        print(f"| {arm} | " + " | ".join(cells) + f" | {fmt(float(np.mean(st)), 3) if st else '–'} | "
              f"{k(float(np.median(on))) if on and np.isfinite(np.median(on)) else ('never' if on else '–')} |")


def key_table(runs, task, arms, it, cols, layers=("L0", "L1", "L2")):
    print(f"\n### {task}: key measures at {it // 1000}k (per seed, median over recall targets)\n")
    print("| arm | layer | " + " | ".join(cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    for arm in arms:
        rs = runs.get((task, arm), {})
        for L in layers:
            cells = []
            for c in cols:
                vals = []
                for s in SEEDS:
                    r = rs.get(s)
                    kk = r and r["keys"].get(str(it))
                    v = kk["layers"].get(L, {}).get(c) if kk else None
                    vals.append(fmt(v))
                cells.append(" / ".join(vals))
            print(f"| {arm} | {L} | " + " | ".join(cells) + " |")


def ablation_table(runs, task, arms, it):
    print(f"\n### {task}: answer accuracy at {it // 1000}k with fast reads removed at the answer step\n")
    names = ("all", "noG", "nolin_o2layers", "onlyG", "none")
    print("| arm | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for arm in arms:
        rs = runs.get((task, arm), {})
        cells = []
        for n in names:
            vals = []
            for s in SEEDS:
                r = rs.get(s)
                kk = r and r["keys"].get(str(it))
                vals.append(fmt(kk["acc"][n]) if kk else "–")
            cells.append(" / ".join(vals))
        print(f"| {arm} | " + " | ".join(cells) + " |")


def main(path):
    runs = load(path)
    tasks = sorted({t for t, _ in runs})
    for task in tasks:
        outcome_table(runs, task)
        final = 100000 if task == "KR" else 200000
        if task.startswith("long_range_memory_dataset_"):
            final = max((int(it) for (t, _), seeds in runs.items() if t == task
                         for row in seeds.values() for it in row.get("keys", {})), default=100000)
        geo = ["cos_sq", "cos_oth", "pcos_oth", "read_lin_key_ratio", "lin_ratio", "o1_t4_ratio", "o2_t4_ratio",
               "o2_t16_ratio", "lin_sel", "o1_t4_sel", "o2_t4_sel"]
        if task.startswith("long_range_memory_dataset_"):
            geo += ["key_pr", "cos_oth_abs", "cos_oth_p95"]
        key_table(runs, task, ["lin"], final, geo)
        key_table(runs, task, [a for a in ARMS if a != "lin"], final,
                  ["cos_oth", "read_lin_key_ratio", "read_G_key_ratio", "read_G_share", "o2_t4_ratio", "o1_t4_ratio"],
                  layers=("L1", "L2"))
        ablation_table(runs, task, ARMS, final)


if __name__ == "__main__":
    main(sys.argv[1])
