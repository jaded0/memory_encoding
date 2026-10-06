"""Stream evaluation of a checkpoint on a kvswitch dataset (kv_switch.py), with frozen slow weights.

Each batch row follows one held-out stream, sequence by sequence, as training does: the fast
entries are carried from one sequence to the next and zeroed only every wipe_every sequences (the
checkpoint's --wipe_every, aligned with stream starts), and every target writes them with the
training DFA step (heldout.py's "observed" protocol; slow entries, biases and i2o stay frozen).
Protocols:

  carry     the fast entries carry across sequences (wiped at stream starts: the training setting)
  wiped     the fast entries are wiped at every sequence (what --wipe_every 1 would see)
  no_fast   no fast weights at all (the slow scaffold alone)

Each query's answer is classified with the row's metadata (kv_switch.classify_answer: correct,
stale = the queried key's value before the last switch, stale_other, wrong_key, other) and
counted by protocol and by

  since_switch   sequences since the last switch, bucketed 0, 1, 2-3, 4-7, 8-15, 16-63, 64+
  query_lag      sequences since the queried binding was last shown: 0 (in this sequence), 1, 2, 3+
  first_segment  the stream's first context (no earlier context, so no stale answer exists)

The headline numbers are acc (all queries), acc_in_sequence, acc_carried, and stale_rate over the
queries after a stream's first context. --forget_rate overrides the fast forget rate at evaluation
(a stress test of a model trained at another rate, as in ephemeral-lowrank/wipe_forget), and
--fast_weight_clamp the fast-only clamp (0 = off: does the clamp bind on this checkpoint?).

    python stream_eval.py --checkpoint PATH [--split validation] [--streams N] [--positions T]
                          [--protocols carry wiped no_fast] [--forget_rate F] [--json OUT]
"""
import argparse
import json
from collections import defaultdict

import torch
from datasets import load_from_disk

import kv_switch
from heldout import evaluate_held_out
from preprocess import OneHotCollate, preprocess_rows
from reproducibility import capture_rng_state, restore_rng_state
from utils import initialize_charset, load_checkpoint, read_checkpoint, upgrade_legacy_config

PROTOCOLS = ("carry", "wiped", "no_fast")
SINCE_BUCKETS = ((0, 0), (1, 1), (2, 3), (4, 7), (8, 15), (16, 63), (64, None))
LAG_BUCKETS = ((0, 0), (1, 1), (2, 2), (3, None))


def bucket_label(value, buckets):
    for low, high in buckets:
        if value >= low and (high is None or value <= high):
            return f"{low}" if low == high else (f"{low}+" if high is None else f"{low}-{high}")
    raise ValueError(value)


def load_streams(dataset, split, batch_size, n_streams=0, positions=0):
    """Rows of the split's first n_streams streams (0 = all, rounded down to whole batches),
    with metadata and character tensors, as groups of batch_size streams: [[stream rows] * B]."""
    length = kv_switch.stream_length(dataset)
    rng = capture_rng_state()
    raw = load_from_disk(f"synth_datasets/{dataset}")[split]
    tensors = preprocess_rows(raw, dataset)["tensor"]
    raw = raw.to_list()
    restore_rng_state(rng)
    total = len(raw) // length
    n_streams = total if n_streams <= 0 else min(n_streams, total)
    n_streams -= n_streams % batch_size
    if n_streams == 0:
        raise ValueError(f"{dataset} {split} has fewer than {batch_size} streams")
    positions = length if positions <= 0 else min(positions, length)
    streams = []
    for stream in range(n_streams):
        rows = []
        for position in range(positions):
            index = stream * length + position
            row = dict(raw[index], tensor=tensors[index])
            assert row["stream"] == stream and row["position"] == position, "rows are not stream-major"
            rows.append(row)
        streams.append(rows)
    return [streams[g:g + batch_size] for g in range(0, n_streams, batch_size)]


class StreamCounts:
    """Counts of answer classes per (group name, bucket)."""

    def __init__(self):
        self.counts = defaultdict(lambda: defaultdict(int))

    def add(self, row, answer_class):
        keys = [("all", "all"),
                ("since_switch", bucket_label(row["since_switch"], SINCE_BUCKETS)),
                ("query_lag", bucket_label(row["query_lag"], LAG_BUCKETS)),
                ("segment", "first" if row["segment"] == 0 else "later")]
        if row["segment"] > 0:
            keys.append(("since_switch_later", bucket_label(row["since_switch"], SINCE_BUCKETS)))
            keys.append(("query_lag_later", bucket_label(row["query_lag"], LAG_BUCKETS)))
        for key in keys:
            self.counts[key][answer_class] += 1
            self.counts[key]["n"] += 1

    def summary(self):
        """{'acc', 'acc_in_sequence', 'acc_carried', 'stale_rate', ..., 'by': {group: {bucket: {...}}}}"""
        def rates(c):
            n = c["n"]
            return {"n": n, **{name: c[name] / n for name in kv_switch.CLASSES}} if n else {"n": 0}
        by = defaultdict(dict)
        for (group, bucket), c in self.counts.items():
            by[group][bucket] = rates(c)
        out = {"acc": by["all"]["all"]["correct"]}
        lag0 = by["query_lag"].get("0")
        carried = defaultdict(int)
        for bucket, c in self.counts.items():
            if bucket[0] == "query_lag" and bucket[1] != "0":
                for name, value in c.items():
                    carried[name] += value
        out["acc_in_sequence"] = lag0["correct"] if lag0 else None
        out["acc_carried"] = carried["correct"] / carried["n"] if carried["n"] else None
        later = by["segment"].get("later")
        out["stale_rate_later"] = later["stale"] if later else None
        out["acc_later"] = later["correct"] if later else None
        out["by"] = {group: dict(sorted(buckets.items())) for group, buckets in by.items() if group != "all"}
        return out


@torch.no_grad()
def evaluate_streams(model, groups, config, dataset, protocol, wipe_every, charset):
    """One protocol over groups of streams; returns StreamCounts. Changes the model's state."""
    collate = OneHotCollate(len(charset))
    counts = StreamCounts()
    device = next(model.parameters()).device
    answer = kv_switch.answer_index("a0b1?a0")
    for streams in groups:
        for position in range(len(streams[0])):
            rows = [stream[position] for stream in streams]
            texts, _, onehot = collate(rows)
            onehot = onehot.to(device)
            batch, steps = onehot.shape[0], onehot.shape[1] - 1
            mask = None if protocol == "no_fast" else torch.ones(batch, steps, dtype=torch.bool, device=device)
            wipe = protocol != "carry" or position % wipe_every == 0
            preds, _ = evaluate_held_out(model, onehot, mask, config, wipe_fast=wipe)
            predicted = preds[answer - 1].tolist()
            for row, index in zip(rows, predicted):
                counts.add(row, kv_switch.classify_answer(row, charset[index]))
    return counts


def evaluate_checkpoint_streams(model, groups, config, dataset, protocols, wipe_every, charset):
    saved = {key: value.clone() for key, value in model.state_dict().items()}
    was_training = model.training
    model.eval()
    try:
        results = {}
        for protocol in protocols:
            results[protocol] = evaluate_streams(model, groups, config, dataset, protocol, wipe_every,
                                                 charset).summary()
            model.load_state_dict(saved)
    finally:
        model.load_state_dict(saved)
        model.train(was_training)
    return results


def format_table(results):
    lines = []
    for protocol, summary in results.items():
        head = {k: v for k, v in summary.items() if k != "by"}
        lines.append(f"{protocol}: " + ", ".join(f"{k} {v:.3f}" for k, v in head.items() if v is not None))
        for group in ("since_switch_later", "query_lag_later", "since_switch", "query_lag"):
            buckets = summary["by"].get(group, {})
            if buckets:
                cells = "  ".join(f"{b}: {r['correct']:.2f}/{r['stale']:.2f} (n={r['n']})"
                                  for b, r in buckets.items() if r["n"])
                lines.append(f"  {group} [acc/stale]: {cells}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stream evaluation of a kvswitch checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default=None, help="default: the checkpoint's")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--streams", type=int, default=0, help="streams to evaluate (0 = all whole batches)")
    parser.add_argument("--positions", type=int, default=0, help="sequences per stream (0 = the whole stream)")
    parser.add_argument("--protocols", nargs="+", default=list(PROTOCOLS), choices=PROTOCOLS)
    parser.add_argument("--wipe_every", type=int, default=None, help="default: the checkpoint's")
    parser.add_argument("--forget_rate", type=float, default=None, help="override the fast forget rate")
    parser.add_argument("--fast_weight_clamp", type=float, default=None,
                        help="override the fast-only clamp (0 = off) at evaluation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--json", default=None, help="also write the results here")
    args = parser.parse_args(argv)

    from train import build_model, build_parser, positional_encoding  # train.py imports heldout
    checkpoint = read_checkpoint(args.checkpoint)
    defaults = {key: value for key, value in vars(build_parser().parse_args([])).items()
                if not key.startswith("_")}
    config = {**defaults, **upgrade_legacy_config(checkpoint.get("config", {}))}
    dataset = args.dataset or config["dataset"]
    if not kv_switch.is_switch(dataset):
        parser.error(f"{dataset} is not a kvswitch dataset")
    charset, _, _, n_characters = initialize_charset(config["dataset"])
    model = build_model(config, charset, n_characters)
    model, _, next_iter, _, _ = load_checkpoint(args.checkpoint, model, config, device=args.device,
                                                checkpoint=checkpoint)
    config["pe_matrix"] = positional_encoding(config["positional_encoding_dim"], args.device)
    if args.forget_rate is not None:
        for layer in model.trained_layers():
            layer.forget_rate = args.forget_rate
    if args.fast_weight_clamp is not None:
        for layer in model.trained_layers():
            layer.fast_weight_clamp = args.fast_weight_clamp
    wipe_every = args.wipe_every or config.get("wipe_every", 1)
    groups = load_streams(dataset, args.split, config["batch_size"], args.streams, args.positions)
    results = evaluate_checkpoint_streams(model, groups, config, dataset, args.protocols, wipe_every, charset)
    header = {"checkpoint": args.checkpoint, "iteration": next_iter - 1, "dataset": dataset,
              "split": args.split, "streams": sum(len(g) for g in groups),
              "positions": len(groups[0][0]), "wipe_every": wipe_every,
              "forget_rate": args.forget_rate if args.forget_rate is not None else config.get("forget_rate"),
              "fast_weight_clamp": (args.fast_weight_clamp if args.fast_weight_clamp is not None
                                    else config.get("fast_weight_clamp"))}
    print(json.dumps(header))
    print(format_table(results))
    if args.json:
        with open(args.json, "w") as handle:
            json.dump({**header, "results": results}, handle, indent=2)


if __name__ == "__main__":
    main()
