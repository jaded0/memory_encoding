"""Batch C2 scoring (PREREG_fastlayers.md "Batch C2"). Reads what the earlier batches' scoring read:
each run's train.log interval blocks (scratch/fast_structure_2026-10-22/logparse.py) and its kept
checkpoints (the readout-share measure of align_layers.py / V2 / DC2 scoring, sweep_2026-10-13).

usage: python scratch/c2_2026-10-27/c2_score.py RUNS_DIR [--json out.jsonl]
  RUNS_DIR holds one directory per run, named <arm>-s<seed> (c2_arms.sh), each with train.log and
  checkpoint_<iter>.pth files (--checkpoint_keep_every 50000).

Per run: the outcome (completed / collapsed at the V2 blowup onset / early-stopped), best and final
recall_acc, final loss, mean iters_per_sec, and for every kept checkpoint the readout's share in
range(B_fast,2) and in range(B_2) (random = vocab / hidden), and the fraction of written fast entries
of trunk 1 and 2 at the weight clamp. Per arm: the means over seeds."""
import argparse
import json
import math
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fast_structure_2026-10-22"))
from logparse import blowup_onset, intervals  # noqa: E402

FINAL_ITER = 400_000


def b_fast(vocab, out, layer):
    """ephemeral_model.decoupled_fast_feedback, redrawn as align_layers.py does."""
    generator = torch.Generator().manual_seed(918273 + layer)
    return torch.randn(vocab, out, generator=generator) * math.sqrt(2.0 / (vocab + out))


def readout_shares(path, layer=2):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    readout = sd["i2o.per_sample_weights"].double().mean(0)
    vocab, hidden = readout.shape

    def share(matrix):
        q, _ = torch.linalg.qr(matrix.T.double())
        return float((readout @ q).norm() ** 2 / readout.norm() ** 2)

    row = {"iter": int(ck["iter"]), "random": vocab / hidden,
           f"B{layer}_share": share(sd[f"linear_layers.{layer}.feedback_weights"]),
           f"Bfast{layer}_share": share(b_fast(vocab, hidden, layer))}
    # Fast entries at the --weight_clamp (1) in trunk 1 and 2, as the checkpoint holds them (the state after the
    # last step of the last sequence, after its forget step: a pinned entry reads (1 - 0.01) * 1 = 0.99, hence 0.989).
    for l in (1, 2):
        w = sd[f"linear_layers.{l}.per_sample_weights"]
        m = sd[f"linear_layers.{l}.ephemeral_mask"].unsqueeze(0).expand_as(w)
        fast = w[m]
        written = fast[fast != 0]
        row[f"fast_at_clamp_L{l}"] = float((written.abs() >= 0.989).float().mean()) if written.numel() else 0.0
    return row


def score_run(directory):
    log = directory / "train.log"
    rows = intervals(log) if log.exists() else []
    text = log.read_text(errors="replace") if log.exists() else ""
    last = rows[-1] if rows else (0, {})
    recalls = [m["recall_acc"] for _, m in rows if "recall_acc" in m]
    speeds = [m["iters_per_sec"] for _, m in rows if "iters_per_sec" in m]
    onset = blowup_onset(rows)
    if "Early stopping: Loss has been" in text:
        outcome = "collapsed"
    elif last[0] >= FINAL_ITER:
        outcome = "completed"
    else:
        outcome = f"running/stopped at {last[0]}"
    checkpoints = sorted(directory.glob("checkpoint_*.pth"), key=lambda p: int(re.findall(r"\d+", p.name)[-1]))
    return {"run": directory.name, "arm": directory.name.rsplit("-s", 1)[0],
            "outcome": outcome, "last_iter": last[0], "blowup_onset": onset,
            "best_recall": max(recalls) if recalls else None,
            "final_recall": last[1].get("recall_acc"), "final_loss": last[1].get("loss"),
            "iters_per_sec": sum(speeds) / len(speeds) if speeds else None,
            "shares": [readout_shares(p) for p in checkpoints]}


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs")
    parser.add_argument("--json")
    args = parser.parse_args()
    runs = [score_run(d) for d in sorted(Path(args.runs).iterdir()) if (d / "train.log").exists()]
    if args.json:
        with open(args.json, "w") as handle:
            for run in runs:
                handle.write(json.dumps(run) + "\n")
    fmt = lambda v: "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
    print("| Run | Outcome | Blowup onset | Best recall | Final recall | Final loss | it/s | Bfast2 share (last ckpt) | B2 share | random | L2 fast at clamp |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for run in runs:
        s = run["shares"][-1] if run["shares"] else {}
        print(f"| {run['run']} | {run['outcome']} | {fmt(run['blowup_onset'])} | {fmt(run['best_recall'])} | "
              f"{fmt(run['final_recall'])} | {fmt(run['final_loss'])} | {fmt(run['iters_per_sec'])} | "
              f"{fmt(s.get('Bfast2_share'))} | {fmt(s.get('B2_share'))} | {fmt(s.get('random'))} | "
              f"{fmt(s.get('fast_at_clamp_L2'))} |")
    print()
    print("| Arm | Seeds | Collapsed | Mean best recall | Mean final recall | Mean Bfast2 share |")
    print("|---|---:|---:|---:|---:|---:|")
    for arm in sorted({r["arm"] for r in runs}):
        group = [r for r in runs if r["arm"] == arm]
        print(f"| {arm} | {len(group)} | {sum(r['outcome'] == 'collapsed' for r in group)} | "
              f"{fmt(mean([r['best_recall'] for r in group]))} | {fmt(mean([r['final_recall'] for r in group]))} | "
              f"{fmt(mean([r['shares'][-1]['Bfast2_share'] if r['shares'] else None for r in group]))} |")


if __name__ == "__main__":
    main()
