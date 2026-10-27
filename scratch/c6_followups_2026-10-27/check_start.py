"""Refuse to start a C6 follow-up run from a checkpoint of another arm. train.py's resume checks the dataset, learning rate
and --fast_structure, but not --fast_layers, --plasticity or the seed. Usage:
  python check_start.py CHECKPOINT --fast_structure S --plasticity A --seed N --fast_layers L ('' = all layers)"""
import argparse
import sys

import torch

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
parser.add_argument("--fast_structure", required=True)
parser.add_argument("--plasticity", type=float, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--fast_layers", default="")
args = parser.parse_args()
config = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)["config"]
found = {"fast_structure": config.get("fast_structure", "exclusive"), "plasticity": float(config.get("plasticity")),
         "seed": config.get("seed"), "fast_layers": config.get("fast_layers") or ""}
wanted = {"fast_structure": args.fast_structure, "plasticity": args.plasticity, "seed": args.seed,
          "fast_layers": args.fast_layers}
bad = {k: (found[k], wanted[k]) for k in wanted if found[k] != wanted[k]}
if bad:
    print(f"check_start: {args.checkpoint} does not match this arm (found, wanted): {bad}")
    sys.exit(1)
print(f"check_start: ok {found}")
