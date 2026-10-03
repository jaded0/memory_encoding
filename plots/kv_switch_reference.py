"""Reference learners on the kvswitch streams: what a linear fast-weight bigram memory predicts for
accuracy as a function of the forget rate f and the switch period S, before any model is trained.

The real model with --input_mode last_one and no recurrence is a next-character predictor whose
fast weights store "after c comes 7"; the key->value binding is that bigram. The reference is the
same thing reduced to one linear layer: a memory M [V, d] read as logits = M e(c) for the input
character c, written at every step (every bigram of the stream, as training writes every target)
and decayed by (1 - f) per step, zeroed at each stream start. Two write rules:

  hebb   M += eta * onehot(next) e(c)^T                          (outer product, no error term)
  delta  M += eta * (onehot(next) - M e(c)) e(c)^T               (LMS: error-driven, like DFA's
                                                                  softmax - onehot once aligned)

Character embeddings e have unit norm and pairwise cosine rho (rho about 0.5-0.7 mimics the
measured effective key dimension of about 2; ephemeral-lowrank/wipe_forget). The answer is
the argmax over the ten digits (the slow weights learn the answer is a digit). Accuracy, stale
rate and carried accuracy are scored on the validation streams' answers after the first context.

    python plots/kv_switch_reference.py [--streams 32] [--json OUT]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kv_switch  # noqa: E402
from kv_tasks import CHARSET, VALUES  # noqa: E402

FORGET = (0.0, 0.003, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5)
SWITCH = (1, 4, 16, 64, 256)


def embeddings(rho, d=64, seed=0):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((d, len(CHARSET) + 1)))
    shared, own = q[:, 0], q[:, 1:].T  # orthonormal
    e = np.sqrt(1 - rho) * own + np.sqrt(rho) * shared
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def run(rows_by_stream, rule, eta, forget, rho):
    """Scores one learner on [streams][positions] rows; returns accuracy summaries."""
    e = embeddings(rho)
    index = {c: i for i, c in enumerate(CHARSET)}
    digits = np.array([index[v] for v in VALUES])
    n, length = len(rows_by_stream), len(rows_by_stream[0])
    memory = np.zeros((n, len(CHARSET), e.shape[1]))
    counts = {"n": 0, "correct": 0, "stale": 0, "carried_n": 0, "carried_correct": 0,
              "insq_n": 0, "insq_correct": 0}
    for position in range(length):
        rows = [stream[position] for stream in rows_by_stream]
        chars = np.array([[index[c] for c in row["text"]] for row in rows])  # [n, T]
        for step in range(chars.shape[1] - 1):
            x = e[chars[:, step]]  # [n, d]
            target = np.eye(len(CHARSET))[chars[:, step + 1]]
            logits = np.einsum("nvd,nd->nv", memory, x)
            if step == chars.shape[1] - 2:  # the answer step
                predicted = np.array(list(VALUES))[logits[:, digits].argmax(1)]
                for row, p in zip(rows, predicted):
                    if row["segment"] == 0:
                        continue
                    counts["n"] += 1
                    hit = p == row["text"][-1]
                    counts["correct"] += hit
                    counts["stale"] += p == row["stale"]
                    key = "carried" if row["carried"] else "insq"
                    counts[f"{key}_n"] += 1
                    counts[f"{key}_correct"] += hit
            error = target if rule == "hebb" else target - logits
            memory += eta * np.einsum("nv,nd->nvd", error, x)
            memory *= 1 - forget
    ratio = lambda a, b: counts[a] / counts[b] if counts[b] else None
    return {"acc": ratio("correct", "n"), "stale": ratio("stale", "n"),
            "acc_carried": ratio("carried_correct", "carried_n"), "acc_in_sequence": ratio("insq_correct", "insq_n")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--streams", type=int, default=32)
    parser.add_argument("--positions", type=int, default=1024)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    learners = [("hebb", 1.0, 0.0), ("hebb", 1.0, 0.6), ("delta", 1.0, 0.0), ("delta", 1.0, 0.6),
                ("delta", 0.3, 0.6)]
    results = []
    for s in SWITCH:
        rows = kv_switch.generate_split(kv_switch.switch_name(s), "validation", args.streams)
        streams = [rows[i * 1024:i * 1024 + args.positions] for i in range(args.streams)]
        for rule, eta, rho in learners:
            for f in FORGET:
                r = run(streams, rule, eta, f, rho)
                results.append({"S": s, "rule": rule, "eta": eta, "rho": rho, "f": f, **r})
                print(json.dumps(results[-1]), flush=True)
    print("\nbest f per S (accuracy after the first context):")
    for rule, eta, rho in learners:
        cells = []
        for s in SWITCH:
            rs = [r for r in results if (r["S"], r["rule"], r["eta"], r["rho"]) == (s, rule, eta, rho)]
            best = max(rs, key=lambda r: r["acc"])
            cells.append(f"S={s}: f*={best['f']} acc {best['acc']:.3f}")
        print(f"  {rule} eta={eta} rho={rho}: " + "; ".join(cells))
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
