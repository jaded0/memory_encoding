"""Benchmark tasks beyond key-value binding (see kv_tasks.py), as synth_datasets datasets.

Three families, all in kv_tasks' character set (so vocabulary is the same as the kv_* bridge):

    mqar_4         a1b2c3d4?c3?a1?d4?b2     multi-query associative recall: K pairs, then Q queries
    mod3_8         01101001?1               state tracking: bits, '?', (number of ones) mod 3
    parity_8       01101001?0               the same with modulus 2
    selcopy_3_9    ..4.7..2.?472            selective copy: N digits among blanks, then in order

- mqar_<K>[_q<Q>] (default Q = K): K distinct keys a-j with independent values 0-9, then Q distinct
  queried keys in random order, each written '?' key value. Every value is an answer (Q recall
  targets); the table-to-answer lags span 2 to 2K + 3Q. Zoology's MQAR (Arora et al. 2023) at
  character scale with keys sampled without replacement (a scaled-down adaptation, not a
  comparable MQAR score). Under the strict held-out protocol no answer writes before the first one,
  so later answers see the table only through the fast weights and the earlier answers' queries.
- parity_<L>, mod<m>_<L> (m = 3..9): L random bits, '?', and the count of ones mod m. The single
  answer depends on every bit (its "lag" is L, counted from the first bit, so metrics bucket it
  like a copy of the first bit). Not associative: the state must be updated at every bit.
- selcopy_<N>_<T> (N <= 10, T >= N): T slots, N of them (random positions, sorted) hold distinct
  random digits and the rest are '.', then '?' and the N digits again in order, as in MAD's
  selective copying (Poli et al. 2024) but teacher forced: the model's input is always the true text,
  so each answer step sees the previous answer as input. The digits are distinct so the
  answer's own input never names the next digit through an earlier bigram (body digits are
  separated by blanks): ordered storage, selectively ignoring the blanks.

Held-out splits hold unseen strings. A hash of the episode's core, the text before the first '?'
(mqar: the table, parity/mod: the bits, selcopy: the slots), sends 10% of cores to validation, 10%
to test, and 80% to train, so no validation or test core occurs in training. Small spaces
(parity_8: 256 bit strings) have few distinct held-out cores, so held-out accuracy there counts
distinct strings, not 5,000 / 20,000 independent ones; larger names have more.

Generate: python bench_tasks.py [names ...] [--seed 0] (default: mqar_4, mod3_8, parity_8, selcopy_3_9).
Each split has its own random stream from (seed, name, split), like kv_tasks.
"""
import argparse
import hashlib
import random
import re

import kv_tasks
from kv_tasks import DISTRACTOR, KEYS, QUERY, SPLIT_SIZES, VALUES

CHARSET = kv_tasks.CHARSET
DEFAULT_SEED = kv_tasks.DEFAULT_SEED
BLANK = DISTRACTOR
FAMILIES = ("mqar", "count", "selcopy")
_NAMES = {
    "mqar": re.compile(r"^mqar_(\d+)(?:_q(\d+))?$"),
    "count": re.compile(r"^(?:parity|mod([3-9]))_(\d+)$"),
    "selcopy": re.compile(r"^selcopy_(\d+)_(\d+)$"),
}
REGISTERED = ("mqar_2", "mqar_4", "mqar_8", "mod3_4", "mod3_8", "mod3_12", "parity_4", "parity_8",
              "parity_12", "selcopy_3_9", "selcopy_4_12")


def parse_name(name):
    """(family, params) for a benchmark task name, else None. mqar: (K, Q); count: (modulus, L);
    selcopy: (N, T)."""
    match = _NAMES["mqar"].match(name)
    if match:
        k = int(match.group(1))
        q = int(match.group(2) or k)
        if not 1 <= k <= len(KEYS) or not 1 <= q <= k:
            raise ValueError(f"{name}: need 1 <= Q <= K <= {len(KEYS)}")
        return "mqar", (k, q)
    match = _NAMES["count"].match(name)
    if match:
        modulus, length = int(match.group(1) or 2), int(match.group(2))
        if not 1 <= length <= 50:
            raise ValueError(f"{name}: L must be 1..50")
        return "count", (modulus, length)
    match = _NAMES["selcopy"].match(name)
    if match:
        n, t = int(match.group(1)), int(match.group(2))
        if not 1 <= n <= len(VALUES) or t < n or t + n > 50:
            raise ValueError(f"{name}: need 1 <= N <= {len(VALUES)}, N <= T, T + N <= 50")
        return "selcopy", (n, t)
    return None


def is_bench(name):
    return any(pattern.match(name) for pattern in _NAMES.values())


def registered_names():
    return list(REGISTERED)


def generate_sample(rng, family, params):
    """One episode string, drawn with rng (a random.Random)."""
    if family == "mqar":
        k, q = params
        keys = rng.sample(KEYS, k)
        values = [rng.choice(VALUES) for _ in range(k)]
        table = "".join(key + value for key, value in zip(keys, values))
        return table + "".join(QUERY + keys[i] + values[i] for i in rng.sample(range(k), q))
    if family == "count":
        modulus, length = params
        bits = "".join(rng.choice("01") for _ in range(length))
        return bits + QUERY + str(bits.count("1") % modulus)
    if family == "selcopy":
        n, t = params
        positions = sorted(rng.sample(range(t), n))
        digits = rng.sample(VALUES, n)
        body = [BLANK] * t
        for position, digit in zip(positions, digits):
            body[position] = digit
        return "".join(body) + QUERY + "".join(digits)
    raise ValueError(f"unknown family {family!r}; choose from {FAMILIES}")


def core(text):
    """The part of an episode before the first query marker (what the held-out split hashes)."""
    return text[:text.index(QUERY)]


def split_of(text):
    """'validation', 'test' or 'train' for an episode, by a stable hash of its core."""
    digest = hashlib.md5(core(text).encode()).digest()[0] % 10
    return {0: "validation", 1: "test"}.get(digest, "train")


def generate_split(name, split, size, seed=DEFAULT_SEED):
    family, params = parse_name(name)
    rng = random.Random(f"{seed}/{name}/{split}")
    texts = []
    while len(texts) < size:  # rejection: the cores of a split are its hash class's
        text = generate_sample(rng, family, params)
        if split_of(text) == split:
            texts.append(text)
    return texts


def episode(text, name):
    """{'answers': {answer index: source index}} for an episode: the source is the character the
    answer is stored at (parity/mod: the first bit, which only sets the lag bucket)."""
    family, params = parse_name(name)
    query_at = text.index(QUERY)
    if family == "mqar":
        k, q = params
        return {"answers": {2 * k + 3 * j + 2: _mqar_source(text, k, j) for j in range(q)}}
    if family == "count":
        return {"answers": {query_at + 1: 0}}
    n, t = params
    sources = [i for i in range(query_at) if text[i] != BLANK]
    return {"answers": {query_at + 1 + j: sources[j] for j in range(n)}}


def _mqar_source(text, k, j):
    """Index of the value bound to query j's key in an mqar episode's table."""
    key = text[2 * k + 3 * j + 1]
    return text.index(key, 0, 2 * k) + 1


def recall_targets(text, name):
    """({target index: lag}, None), lag = target - source - 1, as metrics.recall_targets."""
    return {answer: answer - source - 1 for answer, source in episode(text, name)["answers"].items()}, None


def recall_chance(name):
    family, params = parse_name(name)
    return 1 / params[0] if family == "count" else 1 / len(VALUES)


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
    parser = argparse.ArgumentParser(description="Generate benchmark synth_datasets (mqar, parity/mod, selcopy).")
    parser.add_argument("names", nargs="*", default=["mqar_4", "mod3_8", "parity_8", "selcopy_3_9"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out_dir", default="synth_datasets")
    args = parser.parse_args(argv)
    for name in args.names:
        if not is_bench(name):
            parser.error(f"{name} is not a benchmark task name (mqar_<K>[_q<Q>], parity_<L>, mod<m>_<L>, "
                         "selcopy_<N>_<T>)")
        generate_dataset(name, args.seed, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
