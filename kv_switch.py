"""Key-value streams with context switches: kvswitch_s<S>[_l<L>], a synth_datasets task family
whose best memory timescale is set by the switch period S.

A *stream* is L consecutive sequences (default 1024). It owns N_KEYS = 4 keys (letters drawn
once per stream). A *context* binds each key to a value (distinct digits). The context is
re-drawn every S sequences (a switch at positions 0, S, 2S, ...); the new values are drawn from
the digits the previous context did not use, so every key's new value differs from its old one
and the old values never reappear in the new context (a stale answer is unambiguous). Each
sequence shows PAIRS = 2 of the current bindings, then queries one key:

    c7a2?a2     shows c->7 and a->2, queries a (in-sequence query, lag 0)
    d5c7?a2     shows d and c, queries a, last shown 1 sequence ago (carried query, lag 1)

Only keys already shown since the last switch are queried. With probability CARRY_PROB the
query is a key shown in an earlier sequence of the current context but not in this one (a
*carried* query: it can only be answered by memory that survived the sequence boundary), if
there is one; otherwise one of the two keys of this sequence (an *in-sequence* query, the same
kind of recall as kv_unique_2). The answer (teacher-forced in training) counts as a showing.

Training needs --wipe_every to keep the fast weights across sequences within a stream: with
--wipe_every L the fast entries are zeroed exactly at each stream start; --wipe_every 1 is the
no-carry control. Each batch row is one stream, in order (preprocess.load_and_preprocess_data
uses reproducibility.StreamSampler instead of shuffling sequences), so batch t holds position t
of B streams; streams are shuffled, sequences are not. S = 1 re-draws the context every sequence,
so no query is carried (the control with nothing to remember across sequences).

The design target: memory that survives across sequences is useful (carried queries) and
harmful (bindings from before the last switch interfere), and the balance moves with S, so
the forget rate that maximises accuracy should move with S (vault note "ephemeral weights
forget-rate switching streams 2026-10-02").

Metadata (stored next to 'text' in every split; training ignores it, stream_eval.py scores with
it): stream, position, segment (position // S), since_switch (position % S), query_lag
(sequences since the queried binding was last shown, 0 = in this sequence), carried, stale (the
queried key's value in the previous context, '' in a stream's first context), previous and
context (the previous and current bindings, as 'a3c5...' in the stream's key order).

Generate: python kv_switch.py [names ...] [--seed 0]. Each split has its own random stream
derived from (seed, name, split). Split sizes are in streams (train 1024, validation and test
32), so a split has streams x L rows, stored stream-major (row = stream * L + position).
"""
import argparse
import random
import re

from kv_tasks import CHARSET, KEYS, QUERY, VALUES  # same characters as the kv_* tasks

N_KEYS = 4
PAIRS = 2
CARRY_PROB = 0.5
DEFAULT_LENGTH = 1024
SEQUENCE_LENGTH = 2 * PAIRS + 3
SPLIT_STREAMS = {"train": 1024, "validation": 32, "test": 32}
DEFAULT_SEED = 0
REGISTERED_S = (1, 4, 16, 64, 256)
REGISTERED_L = (DEFAULT_LENGTH, 64)  # 64: short streams for smoke tests
CLASSES = ("correct", "stale", "stale_other", "wrong_key", "other")
_NAME = re.compile(r"^kvswitch_s(\d+)(?:_l(\d+))?$")

assert 2 * N_KEYS <= len(VALUES), "a new context needs N_KEYS values the previous one did not use"


def parse_switch_name(name):
    """(S, L) for a kvswitch dataset name, else None."""
    match = _NAME.match(name)
    if not match:
        return None
    s, length = int(match.group(1)), int(match.group(2) or DEFAULT_LENGTH)
    if s < 1 or length < 1:
        raise ValueError(f"{name}: S and L must be at least 1")
    return s, length


def is_switch(name):
    return _NAME.match(name) is not None


def switch_name(s, length=DEFAULT_LENGTH):
    return f"kvswitch_s{s}" + (f"_l{length}" if length != DEFAULT_LENGTH else "")


def registered_names():
    """Names in the dataset tables (utils/preprocess dataset_keys); any name matching the
    pattern works with the generator and metrics."""
    return [switch_name(s, length) for length in REGISTERED_L for s in REGISTERED_S if s <= length]


def stream_length(name):
    return parse_switch_name(name)[1]


def _bindings(keys, context):
    return "".join(key + context[key] for key in keys) if context else ""


def generate_stream(rng, s, length, stream=0):
    """One stream: a list of length rows {'text', metadata...}, drawn with rng (random.Random)."""
    keys = rng.sample(KEYS, N_KEYS)
    rows, context, previous, last_shown = [], None, None, {}
    for position in range(length):
        if position % s == 0:  # switch: new values, all different from the previous context's
            pool = [v for v in VALUES if context is None or v not in context.values()]
            previous, context = context, dict(zip(keys, rng.sample(pool, N_KEYS)))
            last_shown = {}
        shown = rng.sample(keys, PAIRS)
        carriable = [key for key in keys if key in last_shown and key not in shown]
        if carriable and rng.random() < CARRY_PROB:
            query = rng.choice(carriable)
        else:
            query = rng.choice(shown)
        lag = 0 if query in shown else position - last_shown[query]
        rows.append({
            "text": "".join(key + context[key] for key in shown) + QUERY + query + context[query],
            "stream": stream,
            "position": position,
            "segment": position // s,
            "since_switch": position % s,
            "query_lag": lag,
            "carried": lag > 0,
            "stale": previous[query] if previous else "",
            "previous": _bindings(keys, previous),
            "context": _bindings(keys, context),
        })
        for key in shown + [query]:
            last_shown[key] = position
    return rows


def generate_split(name, split, n_streams, seed=DEFAULT_SEED):
    """n_streams streams of the named dataset, stream-major."""
    s, length = parse_switch_name(name)
    rng = random.Random(f"{seed}/{name}/{split}")
    return [row for stream in range(n_streams) for row in generate_stream(rng, s, length, stream)]


def parse_bindings(text):
    return {text[i]: text[i + 1] for i in range(0, len(text), 2)}


def answer_index(text):
    return text.index(QUERY) + 2


def classify_answer(row, predicted):
    """One of CLASSES for a predicted answer character, from the row's metadata: 'stale' is the
    queried key's value before the last switch, 'stale_other' another key's value then,
    'wrong_key' another key's value now. The classes are disjoint by construction."""
    text = row["text"]
    query = text[answer_index(text) - 1]
    if predicted == text[answer_index(text)]:
        return "correct"
    if predicted == row["stale"]:
        return "stale"
    if predicted in parse_bindings(row["previous"]).values():
        return "stale_other"
    current = parse_bindings(row["context"])
    if predicted in [value for key, value in current.items() if key != query]:
        return "wrong_key"
    return "other"


def in_sequence_source(text):
    """Index of the queried key's value among this sequence's pairs, or None (a carried query)."""
    query_at = text.index(QUERY)
    query = text[query_at + 1]
    sources = [i + 1 for i in range(0, query_at, 2) if text[i] == query]
    return sources[-1] if sources else None


def generate_dataset(name, seed=DEFAULT_SEED, streams=None, out_dir="synth_datasets"):
    from datasets import Dataset, DatasetDict
    streams = streams or SPLIT_STREAMS
    splits = {}
    for split, n in streams.items():
        rows = generate_split(name, split, n, seed)
        splits[split] = Dataset.from_dict({key: [row[key] for row in rows] for key in rows[0]}, split=split)
    splits = DatasetDict(splits)
    path = f"{out_dir}/{name}"
    splits.save_to_disk(path)
    print(f"{name}: {', '.join(f'{s} {len(splits[s])}' for s in splits)} -> {path}; "
          f"e.g. {splits['train'][:4]['text']}")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate key-value switching-stream synth_datasets.")
    parser.add_argument("names", nargs="*", default=registered_names())
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out_dir", default="synth_datasets")
    args = parser.parse_args(argv)
    for name in args.names:
        if not is_switch(name):
            parser.error(f"{name} is not a switching-stream dataset name (kvswitch_s<S>[_l<L>])")
        generate_dataset(name, args.seed, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
