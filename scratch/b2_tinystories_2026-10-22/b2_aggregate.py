"""B2 (2026-10-22): tables from b2_eval.py outputs.

    python b2_aggregate.py EVAL_DIR [--synth SYNTH_DIR] [--curves CURVE_DIR] > tables.txt

EVAL_DIR holds <ARM>-s<SEED>.json from `b2_eval.py tiny`. NOFAST is the reference arm for the slow-weight tax;
seeds are paired by their position in the sorted seed list.
"""
import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

import numpy as np

ARMS = ["NOFAST", "L0", "DEEP", "DEEPS"]
BIGRAM = 2.287


def load(directory):
    runs = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        match = re.match(r"(\w+)-s(\d+)\.json$", os.path.basename(path))
        if match:
            runs[match.group(1)][int(match.group(2))] = json.load(open(path))
    return runs


def fmt(values, digits=3):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not values:
        return "-"
    mean = np.mean(values)
    per = " / ".join(f"{v:.{digits}f}" for v in values)
    return f"{mean:.{digits}f} ({per})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_dir")
    parser.add_argument("--synth", default=None)
    parser.add_argument("--curves", default=None)
    args = parser.parse_args()
    runs = load(args.eval_dir)
    arms = [a for a in ARMS if a in runs] + sorted(a for a in runs if a not in ARMS)

    print("== (d) held-out CE, nats/char: mean (per seed), ON and OFF; bits/char = nats / ln 2")
    for arm in arms:
        seeds = sorted(runs[arm])
        on = [runs[arm][s]["ce"]["all"]["on"] for s in seeds]
        off = [runs[arm][s]["ce"]["all"]["off"] for s in seeds]
        print(f"{arm:7s} seeds {seeds}\n   ON  {fmt(on)}  bits {np.mean(on) / math.log(2):.3f}"
              f"\n   OFF {fmt(off)}  bits {np.mean(off) / math.log(2):.3f}")
    print(f"(empirical bigram on the same text: {BIGRAM})")

    if "NOFAST" in runs:
        ref_seeds = sorted(runs["NOFAST"])
        print("\n== (c) slow-weight tax = OFF(arm) - OFF(NOFAST), paired by seed index; and ON gain = OFF(NOFAST) - ON(arm)")
        for cls in ("all", "novel_word_letter", "repeat_word_letter", "nonletter"):
            print(f" class {cls} (n chars per run {runs['NOFAST'][ref_seeds[0]]['ce'][cls]['n']})")
            for arm in arms:
                if arm == "NOFAST":
                    continue
                seeds = sorted(runs[arm])
                pairs = list(zip(seeds, ref_seeds))
                tax = [runs[arm][s]["ce"][cls]["off"] - runs["NOFAST"][r]["ce"][cls]["off"] for s, r in pairs]
                gain = [runs["NOFAST"][r]["ce"][cls]["off"] - runs[arm][s]["ce"][cls]["on"] for s, r in pairs]
                print(f"   {arm:6s} tax {fmt(tax)}   ON gain {fmt(gain)}")

    print("\n== (a) key participation ratio, ON protocol (keys while the memory is written); OFF in brackets")
    for layer in ("layer0", "layer1", "layer2"):
        print(f" {layer}")
        for arm in arms:
            seeds = sorted(runs[arm])
            get = lambda p, k: [runs[arm][s]["pr"][p][layer].get(k) for s in seeds]
            line = (f"   {arm:6s} corpus unc {fmt(get('on', 'pr_uncentered'), 2)} [off {np.mean(get('off', 'pr_uncentered')):.2f}]"
                    f" | cen {fmt(get('on', 'pr_centered'), 2)}"
                    f" | win9 {np.mean(get('on', 'win9_pr_uncentered')):.2f} win64 {np.mean(get('on', 'win64_pr_uncentered')):.2f}"
                    f" win511 {np.mean(get('on', 'win511_pr_uncentered')):.2f}"
                    f" | top1 {np.mean(get('on', 'top1_share')):.2f}")
            if layer != "layer0":
                line += (f" | rho64 vs L0 {fmt(get('on', 'win64_spearman_vs_layer0'), 2)}"
                         f" rho9 {np.mean(get('on', 'win9_spearman_vs_layer0')):.2f}")
            print(line)

    print("\n== (b) re-mention surprisal (nats per mention; continuation = characters 2..n)")
    for kind in ("name", "control"):
        print(f" {kind}")
        for arm in arms:
            seeds = sorted(runs[arm])
            rm = [runs[arm][s].get("remention", {}).get(kind) for s in seeds]
            if not all(rm):
                continue
            g = lambda k: [r[k] for r in rm]
            print(f"   {arm:6s} pairs {g('pairs')[0]} cont chars {rm[0]['mean_cont_chars']:.2f}")
            print(f"      cont ON 1st {fmt(g('cont_on1'), 2)} 2nd {fmt(g('cont_on2'), 2)} | OFF {np.mean(g('cont_off1')):.2f}/{np.mean(g('cont_off2')):.2f}"
                  f" | RESET {np.mean(g('cont_reset1')):.2f}/{np.mean(g('cont_reset2')):.2f}")
            print(f"      cont M {fmt(g('cont_M'), 3)} (sem {np.mean(g('cont_M_sem')):.3f}) per char {fmt(g('cont_M_per_char'), 3)}"
                  f" | Armeni drop {fmt(g('cont_armeni_drop'), 3)} | OFF drop {fmt(g('cont_off_drop'), 4)}")
            print(f"      first-char M {fmt(g('first_M'), 3)} | first Armeni drop {fmt(g('first_armeni_drop'), 3)}"
                  f" | reset2-on2 cont {fmt(g('cont_reset2_minus_on2'), 3)}")

    if args.synth:
        print("\n== key PR on the synthetic tasks (same script, observed protocol, 1024 episodes)")
        for path in sorted(glob.glob(os.path.join(args.synth, "*.json"))):
            r = json.load(open(path))
            row = " | ".join(f"L{l}: corpus {r['pr'][f'layer{l}']['pr_uncentered']:.2f} (cen {r['pr'][f'layer{l}']['pr_centered']:.2f})"
                             f" episode {r['pr'][f'layer{l}']['episode_pr_uncentered']:.2f}" for l in range(3))
            print(f" {os.path.basename(path)[:-5]:24s} {row}")

    if args.curves:
        print("\n== held-out curves (256 stories): ON / OFF CE by iteration, mean over seeds")
        curves = defaultdict(lambda: defaultdict(list))
        for path in glob.glob(os.path.join(args.curves, "*.json")):
            match = re.match(r"(\w+)-s(\d+)-it(\d+)\.json$", os.path.basename(path))
            if match:
                r = json.load(open(path))
                curves[match.group(1)][int(match.group(3))].append((r["ce"]["all"]["on"], r["ce"]["all"]["off"]))
        for arm in arms:
            if arm not in curves:
                continue
            pts = " ".join(f"{it // 1000}k:{np.mean([v[0] for v in vals]):.3f}/{np.mean([v[1] for v in vals]):.3f}(n{len(vals)})"
                           for it, vals in sorted(curves[arm].items()))
            print(f" {arm:6s} {pts}")


if __name__ == "__main__":
    main()
