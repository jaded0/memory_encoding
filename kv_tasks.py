"""Character-level key-value memory tasks (associative recall), as synth_datasets datasets.

An episode is K key-value pairs, D distractors, a query marker, a queried key, and its value:

    kv_unique_4      c7a2h2e9?a2
    kv_reassign_4    d5d1j0d6?d6        d assigned three times: the answer is its latest value
    kv_unique_4_d8   c7a2h2e9........?a2

Keys are the letters a-j and values the digits 0-9; '?' marks the query, '.' is the distractor
filler and ' ' the (unused) padding character. The answer is the only recall target
(metrics.recall_targets); with --input_mode last_one and no recurrence, the step whose input is
the queried key can only predict it from what the fast weights stored at that key's pair.

- kv_unique_<K>: K distinct keys (without replacement), values independent (they may repeat),
  one of the K keys queried, as in Ba et al. 2016's associative retrieval (letters and digits,
  there with '??' before the query).
- kv_reassign_<K>: the queried key is assigned c times, c uniform in 1..K, at random pairs; the
  other pairs take any other key, with replacement (so they may repeat too). The answer is the
  queried key's latest value, which differs from its previous one whenever c >= 2, so a stale
  answer is unambiguous. Queries hit a reassigned key (c >= 2) with probability (K - 1) / K.
- _d<D> (optional): D distractor characters between the table and the query. Every sequence
  of a dataset has the same length, 2K + D + 3.

Generate: python kv_tasks.py [names ...] [--seed 0] (default: K in 2, 4, 8, both modes). Each
split has its own random stream derived from (seed, name, split), so a dataset is reproducible
and independent of which others are generated with it. Sizes follow the palindrome datasets
(1,000,000 / 5,000 / 20,000 train / validation / test). Small tables have few distinct
episodes (kv_unique_2: 18,000), so held-out episodes also occur in training there.
"""
import argparse
import random
import re

KEYS = "abcdefghij"
VALUES = "0123456789"
QUERY = "?"
DISTRACTOR = "."
PAD = " "
CHARSET = PAD + QUERY + DISTRACTOR + KEYS + VALUES

MODES = ("unique", "reassign")
SPLIT_SIZES = {"train": 1_000_000, "validation": 5_000, "test": 20_000}
DEFAULT_SEED = 0
# Names registered in the dataset tables (utils/preprocess dataset_keys); any name matching the
# pattern works with the generator and metrics.
REGISTERED_K = (1, 2, 4, 8, 16)
REGISTERED_D = (0, 4, 8, 16)
_NAME = re.compile(r"^kv_(unique|reassign)_(\d+)(?:_d(\d+))?$")


def parse_kv_name(name):
    """(mode, K, D) for a key-value dataset name, else None."""
    match = _NAME.match(name)
    if not match:
        return None
    mode, k, d = match.group(1), int(match.group(2)), int(match.group(3) or 0)
    if k < 1 or (mode == "unique" and k > len(KEYS)):
        raise ValueError(f"{name}: K must be 1..{len(KEYS)} for unique keys, >= 1 for reassign")
    return mode, k, d


def is_kv(name):
    return _NAME.match(name) is not None


def kv_name(mode, k, d=0):
    return f"kv_{mode}_{k}" + (f"_d{d}" if d else "")


def registered_names():
    return [kv_name(mode, k, d) for mode in MODES for k in REGISTERED_K for d in REGISTERED_D
            if not (mode == "unique" and k > len(KEYS))]


def generate_sample(rng, mode, k, d=0):
    """One episode string, drawn with rng (a random.Random)."""
    values = [rng.choice(VALUES) for _ in range(k)]
    if mode == "unique":
        keys = rng.sample(KEYS, k)
        query = rng.choice(keys)
    elif mode == "reassign":
        query = rng.choice(KEYS)
        count = rng.randint(1, k)
        slots = set(rng.sample(range(k), count))
        others = KEYS.replace(query, "")
        keys = [query if i in slots else rng.choice(others) for i in range(k)]
        occurrences = sorted(slots)
        if count >= 2:  # the latest value differs from the previous one: stale is unambiguous
            previous = values[occurrences[-2]]
            values[occurrences[-1]] = rng.choice(VALUES.replace(previous, ""))
    else:
        raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
    latest = max(i for i in range(k) if keys[i] == query)
    table = "".join(key + value for key, value in zip(keys, values))
    return table + DISTRACTOR * d + QUERY + query + values[latest]


def episode(text):
    """Parse an episode: {'answer': index of the answer, 'source': index of the latest value of
    the queried key, 'stale': earlier values of the queried key, 'other_keys': values bound to
    other keys}."""
    query_at = text.index(QUERY)
    answer = query_at + 2
    table = text[:query_at].rstrip(DISTRACTOR)
    pairs = [(2 * i, table[2 * i], table[2 * i + 1]) for i in range(len(table) // 2)]
    query = text[query_at + 1]
    mine = [(pos + 1, value) for pos, key, value in pairs if key == query]
    return {
        "answer": answer,
        "source": mine[-1][0],
        "stale": {value for _, value in mine[:-1]},
        "other_keys": {value for _, key, value in pairs if key != query},
    }


def classify_answer(text, predicted):
    """'correct', 'stale' (an earlier value of the queried key), 'wrong_key' (a value bound to
    another key in this episode) or 'other', in that priority order."""
    info = episode(text)
    if predicted == text[info["answer"]]:
        return "correct"
    if predicted in info["stale"]:
        return "stale"
    if predicted in info["other_keys"]:
        return "wrong_key"
    return "other"


def generate_split(name, split, size, seed=DEFAULT_SEED):
    mode, k, d = parse_kv_name(name)
    rng = random.Random(f"{seed}/{name}/{split}")
    return [generate_sample(rng, mode, k, d) for _ in range(size)]


def generate_dataset(name, seed=DEFAULT_SEED, sizes=None, out_dir="synth_datasets"):
    from datasets import Dataset, DatasetDict
    sizes = sizes or SPLIT_SIZES
    splits = DatasetDict({split: Dataset.from_dict({"text": generate_split(name, split, size, seed)}, split=split)
                          for split, size in sizes.items()})
    path = f"{out_dir}/{name}"
    splits.save_to_disk(path)
    print(f"{name}: {', '.join(f'{s} {len(splits[s])}' for s in splits)} -> {path}; e.g. {splits['train'][:3]['text']}")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate key-value synth_datasets.")
    parser.add_argument("names", nargs="*", default=[kv_name(mode, k) for mode in MODES for k in (2, 4, 8)])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out_dir", default="synth_datasets")
    args = parser.parse_args(argv)
    for name in args.names:
        if not is_kv(name):
            parser.error(f"{name} is not a key-value dataset name (kv_<unique|reassign>_<K>[_d<D>])")
        generate_dataset(name, args.seed, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
