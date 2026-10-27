"""Seeded KR interference/gap datasets, generated separately from synth_datasets.py.

long_range_memory_dataset_load_D: 0?V + D random non-marker characters + !V.
long_range_memory_dataset_gap_D: the same, with D zeroes instead (delay control).
The original generator and dataset are unchanged. No extra '?' or '!' is allowed.
"""
import argparse
import random
import re

CHARSET = "0?!123,. "
PAYLOAD = "123,."  # no padding character in distractors or answers
LOADS = (0, 4, 8, 16, 32, 48)  # recall lag D+1 must stay below metrics.MAX_LAG=60
PREFIX = "long_range_memory_dataset_"
PATTERN = re.compile(r"^long_range_memory_dataset_(load|gap)_(\d+)$")


def parse_name(name):
    match = PATTERN.fullmatch(name)
    if not match:
        raise ValueError(f"invalid KR load dataset: {name}")
    return match[1], int(match[2])


def registered_names():
    return [f"{PREFIX}{mode}_{d}" for mode in ("load", "gap") for d in LOADS]


def generate_sample(rng, mode, d):
    if mode not in ("load", "gap") or d < 0:
        raise ValueError("mode must be load/gap and D nonnegative")
    value = rng.choice(PAYLOAD)
    gap = "".join(rng.choice(PAYLOAD) for _ in range(d)) if mode == "load" else "0" * d
    return "0?" + value + gap + "!" + value


def main(argv=None):
    from datasets import Dataset, DatasetDict

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("names", nargs="+", choices=registered_names())
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--train-size", type=int, default=100000)
    ap.add_argument("--validation-size", type=int, default=5000)
    ap.add_argument("--test-size", type=int, default=20000)
    ap.add_argument("--root", default="synth_datasets")
    args = ap.parse_args(argv)
    for name in args.names:
        mode, d = parse_name(name)
        splits = {}
        for split, size in (("train", args.train_size), ("validation", args.validation_size),
                            ("test", args.test_size)):
            rng = random.Random(f"{args.seed}:{name}:{split}")
            splits[split] = Dataset.from_dict({"text": [generate_sample(rng, mode, d) for _ in range(size)]})
        DatasetDict(splits).save_to_disk(f"{args.root}/{name}")


if __name__ == "__main__":
    main()
