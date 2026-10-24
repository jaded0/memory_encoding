"""Batch O2: one JSON line per run directory (KR-<arm>-s<seed>, mqar_K-<arm>-s<seed>).

usage: python o2_collect.py RUN_DIR [RUN_DIR ...] > o2_runs.jsonl
Per run: train-log intervals (500 its), onset = first interval with train recall_acc >= 0.5 (and >= 0.9),
the in-run held-out strict recall every 25k, the last interval loss, exits, final heldout.py numbers
(observed / strict / no_fast recall and loss), and the o2_keys.py summaries (keys/*.json) by iteration."""
import glob
import json
import math
import os
import re
import sys


def parse_log(path):
    blocks, cur = [], None
    exits = {}
    with open(path, errors="replace") as f:
        for line in f:
            m = re.match(r"--- Interval metrics \(ending @ iter (\d+)", line)
            if m:
                cur = {"iter": int(m.group(1))}
                blocks.append(cur)
                continue
            m = re.match(r"\s+([\w/]+): (\S+)$", line)
            if m and cur is not None:
                try:
                    cur[m.group(1)] = float(m.group(2))
                except ValueError:
                    pass
                continue
            m = re.match(r"(TRAIN|EVAL|KEYS) EXIT (\d+)", line)
            if m:
                exits[m.group(1)] = int(m.group(2))
            if line.strip() and not line.startswith("  "):
                cur = None if not line.startswith("---") else cur
    return blocks, exits


def onset(blocks, thr):
    for b in blocks:
        if b.get("recall_acc", 0) >= thr:
            return b["iter"]
    return None


def main(dirs):
    for d in dirs:
        name = os.path.basename(d.rstrip("/"))
        m = re.match(r"(KR|mqar_\d+)-(\w+)-s(\d+)$", name)
        if not m:
            continue
        row = {"run": name, "task": m.group(1), "arm": m.group(2), "seed": int(m.group(3))}
        log = os.path.join(d, "train.log")
        if os.path.exists(log):
            blocks, exits = parse_log(log)
            row["exits"] = exits
            row["last_iter"] = blocks[-1]["iter"] if blocks else 0
            row["onset50"], row["onset90"] = onset(blocks, 0.5), onset(blocks, 0.9)
            row["nonfinite"] = any(not math.isfinite(b.get("loss", 0)) for b in blocks)
            row["max_loss"] = max((b.get("loss", 0) for b in blocks), default=None)
            row["last_loss"] = blocks[-1].get("loss") if blocks else None
            row["last_train_recall"] = blocks[-1].get("recall_acc") if blocks else None
            row["heldout_traj"] = {b["iter"]: b["heldout_strict/recall_acc"] for b in blocks
                                   if "heldout_strict/recall_acc" in b}
            row["train_recall_traj"] = {b["iter"]: b.get("recall_acc") for b in blocks if b["iter"] % 5000 == 0}
        fe = os.path.join(d, "final_eval.json")
        if os.path.exists(fe):
            e = json.load(open(fe))
            row["final_iter"] = e.get("iteration")
            for p in ("observed", "strict", "no_fast"):
                row[p] = e.get(f"heldout_{p}/recall_acc")
                row[p + "_loss"] = e.get(f"heldout_{p}/loss")
        keys = {}
        for kf in sorted(glob.glob(os.path.join(d, "keys", "*.json"))):
            k = json.load(open(kf))
            keys[str(k["iteration"])] = {"acc": k["acc"], "targets": k["targets"], "layers": k["layers"]}
        row["keys"] = keys
        print(json.dumps(row))


if __name__ == "__main__":
    main(sys.argv[1:])
