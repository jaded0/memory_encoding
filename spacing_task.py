"""Standalone spaced persistent-binding streams (no model or training flags).

I is the onset-to-onset interval in *sequences*: I=1 means consecutive
presentations (massed). D is the number of complete distractor sequences after
the last presentation, before the query. R defaults to 3. Every presentation
shows the same randomly drawn table; only the final row queries it, so answers
cannot introduce extra study exposures. A stream owns its bindings, never the
whole dataset or a batch slot across streams.

Rows have fixed character length 2*K+1: '.a0b1c2d3' for study (K=4), all dots
for distractors, and '......?a0' for query. Each row performs 2*K next-character
prediction steps under train.py; hidden state resets between rows. The query
answer must be scored BEFORE its teacher-forced update. Filler targets still
cause ordinary updates/forgetting; they are not blank time or no-write masks.

Streams are left-padded to a common L and queried at L-1, allowing the same
compute budget, probe position and wipe period across I,D cells. Study onsets
are q-D-1-(R-1)*I, ..., q-D-1. Carry fast weights across the L rows, wipe at
stream starts; freeze slow weights during evaluation. Fixed I,D evaluation
streams use the same bindings/query for the same seed, split and stream ID,
independent of schedule, for paired comparisons.

Use SpacingBatchStream for training: it never cycles a finite cached dataset,
so slow weights cannot memorize a replayed stream's random table. It does NOT
prevent within-stream slow adaptation (evaluation must freeze slow weights).
generate_split is for finite held-out data/inspection, NOT epoch-replayed
training. Integration with train/preprocess/metrics is intentionally external.
"""
import random

from kv_tasks import CHARSET, DISTRACTOR, KEYS, QUERY, VALUES

DEFAULT_REPETITIONS = 3
DEFAULT_INTERVALS = (1, 4, 16)
DEFAULT_DELAYS = (0, 4, 16)
N_KEYS = 4
CLASSES = ("correct", "wrong_key", "other")


def _integer(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def minimum_length(interval, delay, repetitions=DEFAULT_REPETITIONS):
    """Minimum sequences including R presentations, gaps, D fillers and query."""
    _integer("interval", interval, 1)
    _integer("delay", delay, 0)
    _integer("repetitions", repetitions, 1)
    return (repetitions - 1) * interval + delay + 2


def presentation_positions(interval, delay, length=None, repetitions=DEFAULT_REPETITIONS):
    needed = minimum_length(interval, delay, repetitions)
    length = needed if length is None else _integer("length", length, needed)
    first = length - needed
    return tuple(first + r * interval for r in range(repetitions))


def generate_stream(rng, interval, delay, length=None, stream=0,
                    repetitions=DEFAULT_REPETITIONS, n_keys=N_KEYS):
    """One stream as rows {'text', metadata...}; rng is random.Random.

    Metadata-only context/query/answer MUST NOT be fed as model inputs. query_lag
    counts sequence onsets; query_lag_steps counts actual prediction-step
    distances from the queried value's last study target to its query target.
    """
    positions = presentation_positions(interval, delay, length, repetitions)
    length = minimum_length(interval, delay, repetitions) if length is None else length
    _integer("n_keys", n_keys, 1)
    if n_keys > min(len(KEYS), len(VALUES)):
        raise ValueError("n_keys exceeds the key/value alphabet")
    _integer("stream", stream, 0)
    keys = rng.sample(KEYS, n_keys)
    context = dict(zip(keys, rng.sample(VALUES, n_keys)))
    query = rng.choice(keys)
    table = "".join(key + context[key] for key in keys)
    width = 2 * n_keys + 1
    steps = width - 1
    source = 2 + 2 * keys.index(query)  # character index of last study's value
    target = width - 1
    lag_steps = (length - 1 - positions[-1]) * steps + target - source
    rows = []
    for position in range(length):
        study = position in positions
        probe = position == length - 1
        text = (DISTRACTOR + table if study else
                DISTRACTOR * (width - 3) + QUERY + query + context[query] if probe else
                DISTRACTOR * width)
        rows.append({
            "text": text, "stream": stream, "position": position,
            "kind": "study" if study else "query" if probe else "distractor",
            "interval": interval, "delay": delay, "repetitions": repetitions,
            "context": table, "query": query, "answer": context[query],
            "is_query": probe, "answer_index": target if probe else -1,
            "query_lag": delay + 1 if probe else -1,
            "query_lag_steps": lag_steps if probe else -1,
            "steps_per_sequence": steps, "wipe_fast": position == 0,
        })
    return rows


def classify_answer(row, predicted):
    if not row["is_query"]:
        raise ValueError("only query rows have recall targets")
    if predicted == row["answer"]:
        return "correct"
    if predicted in row["context"][1::2]:
        return "wrong_key"
    return "other"


def _schedule(intervals, delays, length, repetitions):
    intervals, delays = tuple(intervals), tuple(delays)
    if not intervals or not delays:
        raise ValueError("intervals and delays must be nonempty")
    needed = max(minimum_length(i, d, repetitions) for i in intervals for d in delays)
    length = needed if length is None else _integer("length", length, needed)
    return intervals, delays, length


def _stream(seed, split, stream, intervals, delays, length, repetitions, n_keys):
    # Schedule draws are separate so changing I,D never changes paired bindings.
    schedule = random.Random(f"spacing/schedule/{seed}/{split}/{stream}")
    bindings = random.Random(f"spacing/bindings/{seed}/{split}/{stream}")
    return generate_stream(bindings, schedule.choice(intervals), schedule.choice(delays),
                           length, stream, repetitions, n_keys)


def generate_split(split, n_streams, seed=0, intervals=DEFAULT_INTERVALS,
                   delays=DEFAULT_DELAYS, length=None,
                   repetitions=DEFAULT_REPETITIONS, n_keys=N_KEYS, start_stream=0):
    """Finite stream-major rows. For controlled eval pass intervals=(I,), delays=(D,).

    Supply a common length across evaluation cells (at least their maximum
    minimum_length). split seeds are disjoint. start_stream supports chunking.
    """
    _integer("n_streams", n_streams, 0)
    _integer("start_stream", start_stream, 0)
    intervals, delays, length = _schedule(intervals, delays, length, repetitions)
    return [row for stream in range(start_stream, start_stream + n_streams)
            for row in _stream(seed, split, stream, intervals, delays, length, repetitions, n_keys)]


class SpacingBatchStream:
    """Infinite fresh streams, one independent persistent stream per batch row.

    next(iterator) returns B row dictionaries at the same position. Use
    OneHotCollate(len(CHARSET)) after adding each row's 'tensor' character IDs.
    Save state_dict alongside model checkpoints; no worker RNG/prefetch state
    is needed. This iterator is single-consumer (not a DataLoader dataset).
    """

    def __init__(self, batch_size, seed=0, split="train", intervals=DEFAULT_INTERVALS,
                 delays=DEFAULT_DELAYS, length=None, repetitions=DEFAULT_REPETITIONS,
                 n_keys=N_KEYS):
        self.batch_size = _integer("batch_size", batch_size, 1)
        self.intervals, self.delays, self.length = _schedule(intervals, delays, length, repetitions)
        _integer("n_keys", n_keys, 1)
        if n_keys > min(len(KEYS), len(VALUES)):
            raise ValueError("n_keys exceeds the key/value alphabet")
        self.seed, self.split = seed, split
        self.repetitions, self.n_keys = repetitions, n_keys
        self.sequence_count = 0
        self._group = None

    def __iter__(self):
        return self

    def __next__(self):
        group, position = divmod(self.sequence_count, self.length)
        if self._group is None or position == 0:
            self._group = [_stream(self.seed, self.split, group * self.batch_size + b,
                                   self.intervals, self.delays, self.length,
                                   self.repetitions, self.n_keys) for b in range(self.batch_size)]
        rows = [dict(stream[position]) for stream in self._group]
        self.sequence_count += 1
        return rows

    def state_dict(self):
        return {"version": 1, "sequence_count": self.sequence_count,
                "config": {key: getattr(self, key) for key in
                           ("batch_size", "seed", "split", "intervals", "delays", "length",
                            "repetitions", "n_keys")}}

    def load_state_dict(self, state):
        if state.get("version") != 1 or state.get("config") != self.state_dict()["config"]:
            raise ValueError("spacing stream state version/config mismatch")
        self.sequence_count = _integer("sequence_count", state["sequence_count"], 0)
        self._group = None
