"""B2 round 2: summarize b2_eval.py horizon outputs (PREREG_fastlayers.md, "Batch B2 ROUND 2"). Standard library only.

    python3 b2_horizon_aggregate.py FINAL_DIR [--curves CURVES_DIR] [--round1 ROUND1_FINAL_DIR]

FINAL_DIR holds <ARM>-s<SEED>.json (1024 stories), CURVES_DIR <ARM>-s7201-it<N>.json (256 stories), ROUND1_FINAL_DIR
the round-1 tiny jsons (for H2-0's ON comparison).
"""
import argparse
import glob
import json
import os
import re

CLASSES = ("all", "novel_word_letter", "repeat_word_letter", "nonletter")


def load(directory):
    out = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        out[os.path.basename(path)[:-5]] = json.load(open(path))
    return out


def best(res, cls="all"):
    """(R, CE) of the lowest WIPE R CE for a class."""
    return min(((r, res["ce"][f"wipe{r}"][cls]) for r in res["wipe_every"]), key=lambda x: x[1])


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    for rank, i in enumerate(order):
        r[i] = float(rank)
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("final")
    parser.add_argument("--curves", default=None)
    parser.add_argument("--round1", default=None)
    args = parser.parse_args()
    final = load(args.final)
    round1 = load(args.round1) if args.round1 else {}
    for arm in ("DEEP", "L0"):
        runs = {k: v for k, v in final.items() if k.startswith(arm + "-s")}
        if not runs:
            continue
        periods = next(iter(runs.values()))["wipe_every"]
        print(f"== {arm} ({len(runs)} seeds: {', '.join(sorted(runs))})")
        for cls in CLASSES:
            line = [f"  {cls:19s}"]
            for p in ["on", "off"] + [f"wipe{r}" for r in periods]:
                vals = [v["ce"][p][cls] for v in runs.values()]
                line.append(f"{p} {sum(vals) / len(vals):.3f}")
            print(" | ".join(line))
        for name, v in sorted(runs.items()):
            r, ce = best(v)
            h = " ".join(f"{k}:{x:.2f}" for k, x in v["H"].items())
            gains = " ".join(f"{c[:6]} {v['ce']['on'][c] - best(v, c)[1]:+.3f}(R{best(v, c)[0]})" for c in CLASSES)
            r1 = round1.get(name)
            r1s = f" | round-1 ON {r1['ce']['all']['on']:.3f} (diff {v['ce']['on']['all'] - r1['ce']['all']['on']:+.3f})" if r1 else ""
            print(f"  {name}: best R {r} CE {ce:.3f}, ON - best {v['ce']['on']['all'] - ce:+.3f}{r1s}")
            print(f"     H {h}")
            print(f"     ON - best by class: {gains}")
        h32 = [v["H"]["32"] for v in runs.values() if "32" in v["H"]]
        if h32:
            print(f"  H(32) mean {sum(h32) / len(h32):.3f}")
    if args.curves:
        curves = load(args.curves)
        for arm in ("DEEP", "L0"):
            pts = sorted(((int(re.search(r"-it(\d+)$", k).group(1)), v) for k, v in curves.items()
                          if k.startswith(arm + "-s7201-it")), key=lambda x: x[0])
            if not pts:
                continue
            print(f"== {arm} s7201 curve (256 stories): it ON OFF bestR best ON-best")
            gaps = []
            for it, v in pts:
                r, ce = best(v)
                gaps.append(v["ce"]["on"]["all"] - ce)
                print(f"  {it:6d} {v['ce']['on']['all']:.3f} {v['ce']['off']['all']:.3f} R{r} {ce:.3f} {gaps[-1]:+.3f}")
            print(f"  Spearman(ON - best, iteration) = {spearman([p[0] for p in pts], gaps):.2f} over {len(pts)} checkpoints")


if __name__ == "__main__":
    main()
