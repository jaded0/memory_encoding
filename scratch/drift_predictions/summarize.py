"""Summarize drift-prediction run logs and checkpoint weight trajectories."""
import argparse
import glob
import json
import os
import re

import torch


INTERVAL = re.compile(r"Interval metrics \(ending @ iter (\d+), whole batch\)")
METRIC = re.compile(r"  ([\w/]+): ([-+\w.]+)")
CHECKPOINT = re.compile(r"checkpoint_(\d+)\.pth$")
LAYERS = [f"linear_layers.{i}" for i in range(3)] + ["i2h", "i2o"]


def read_log(path):
    records = {}
    current = None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            match = INTERVAL.search(line)
            if match:
                current = {"iter": int(match.group(1))}
                records[current["iter"]] = current
                continue
            match = METRIC.match(line)
            if current is not None and match:
                try:
                    current[match.group(1)] = float(match.group(2))
                except ValueError:
                    pass
    ordered = [records[key] for key in sorted(records)]
    high = 0
    onset = None
    for row in ordered:
        high = high + 1 if row.get("loss", 0) > 5 else 0
        if high == 5:
            onset = row["iter"] - 4 * 500
            break
    return {
        "last_iter": ordered[-1]["iter"] if ordered else None,
        "max_loss_after_30k": max((row.get("loss", float("-inf")) for row in ordered
                                   if row["iter"] >= 30000), default=None),
        "collapse_onset": onset,
        "at_50k": [row for row in ordered if row["iter"] % 50000 == 0],
    }


def slow_matrix(state, prefix):
    weights = state[f"{prefix}.per_sample_weights"].float().mean(0)
    mask = state[f"{prefix}.ephemeral_mask"]
    return weights.masked_fill(mask, 0)


def read_checkpoints(run_dir):
    paths = sorted(glob.glob(os.path.join(run_dir, "checkpoint_*.pth")))
    first = None
    rows = []
    for path in paths:
        match = CHECKPOINT.search(path)
        if not match:
            continue
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint["model_state_dict"]
        matrices = {layer: slow_matrix(state, layer) for layer in LAYERS}
        if first is None:
            first = {layer: matrix.clone() for layer, matrix in matrices.items()}
        rows.append({
            "iter": int(match.group(1)),
            "slow_fro": {layer: torch.linalg.vector_norm(matrix).item()
                         for layer, matrix in matrices.items()},
            "drift_from_first_fro": {
                layer: torch.linalg.vector_norm(matrix - first[layer]).item()
                for layer, matrix in matrices.items()
            },
        })
        del checkpoint, state, matrices
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = {}
    for run_dir in args.run_dirs:
        name = os.path.basename(run_dir.rstrip("/"))
        result[name] = {"log": read_log(os.path.join(run_dir, "train.log")),
                        "checkpoints": read_checkpoints(run_dir)}
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
