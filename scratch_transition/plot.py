"""Aggregate meas/*.json (both seeds) into results.json + figures.  python plot.py MEAS_DIR OUT_DIR"""
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

meas, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
LN = ["L0", "L1", "L2"]
SEEDS = [2718, 3141]
COL = {"L0": "#7a7a7a", "L1": "#2a7ab8", "L2": "#c4452d"}
data = {}
for s in SEEDS:
    rows = []
    for f in glob.glob(f"{meas}/s{s}_ckpt_*.json"):
        r = json.load(open(f))
        rows.append(r)
    rows.sort(key=lambda r: r["iteration"])
    data[s] = rows
json.dump({str(s): data[s] for s in SEEDS}, open(f"{out}/results.json", "w"), indent=1)


def series(s, fn):
    return np.array([r["iteration"] for r in data[s]]), np.array([fn(r) for r in data[s]], dtype=float)


def onset(s, thr=0.5):
    it, rc = series(s, lambda r: r["ablation"]["full"]["recall"])
    k = np.where(rc >= thr)[0]
    return it[k[0]] if len(k) else None


def grid(title, panels, fname, sharex=True):
    fig, axes = plt.subplots(len(SEEDS), len(panels), figsize=(4.2 * len(panels), 3.4 * len(SEEDS)), squeeze=False)
    for i, s in enumerate(SEEDS):
        for j, (ttl, draw) in enumerate(panels):
            ax = axes[i][j]
            draw(ax, s)
            o = onset(s)
            if o is not None:
                ax.axvspan(max(o - 2500, 0), o, color="gold", alpha=0.25, lw=0)
            ax.set_title(f"seed {s}: {ttl}", fontsize=9)
            ax.set_xlabel("iteration")
            ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{out}/{fname}", dpi=130)
    plt.close(fig)


def lines(items, ylabel=None, ylim=None, hline=None, logy=False):
    def draw(ax, s):
        for lab, fn, col, ls in items:
            x, y = series(s, fn)
            ax.plot(x, y, ls, color=col, label=lab, marker="o", ms=2.5, lw=1.2)
        if hline is not None:
            ax.axhline(hline[0], color="k", ls=":", lw=1, label=hline[1])
        if ylim:
            ax.set_ylim(*ylim)
        if logy:
            ax.set_yscale("log")
        ax.legend(fontsize=7)
    return draw


A = lambda l, key, sub="mean": (lambda r: r["align"][l][key][sub] if isinstance(r["align"][l][key], dict) else r["align"][l][key])
G = lambda l, key: (lambda r: r["geom"][l][key]["mean"] if isinstance(r["geom"][l][key], dict) else r["geom"][l][key])
# fig1: recall + ablations
abl = lambda name: (lambda r: r["ablation"][name]["recall"])
grid("Held-out recall at the answer step (gold band = the 2500 iterations before recall >= 0.5)", [
    ("recall: full / no fast / ablations", lines([
        ("full", abl("full"), "k", "-"), ("no F at all", abl("no_fast@all"), "#888", "--"),
        ("no F in L2", abl("no_fast@L2"), COL["L2"], "-"), ("no F in L1", abl("no_fast@L1"), COL["L1"], "-"),
        ("no F in L0", abl("no_fast@L0"), COL["L0"], "-"),
        ("L2 store-step term only", abl("signal_only@L2"), "#e08a2d", ":"), ("store term only, all layers", abl("signal_only@all"), "#2d9e5a", ":")], ylim=(-0.03, 1.05))),
    ("answer-step CE loss (full vs no fast)", lines([
        ("full", lambda r: r["answer_loss"]["full"]["mean"], "k", "-"), ("no fast", lambda r: r["answer_loss"]["no_fast"]["mean"], "#888", "--")])),
    ("logit margin (target - best other)", lines([("full", lambda r: r["answer_margin"]["mean"], "k", "-")])),
], "fig1_recall_ablations.png")
# fig2: alignment
grid("Alignment of DFA feedback B_l with the true backprop Jacobian (answer step)", [
    ("cos theta = <J_l,B_l>_F/(|J||B|)", lines([(l, A(l, "cosF"), COL[l], "-") for l in LN] + [("toy all-error threshold", lambda r: r["align"]["L2"]["toy_threshold_cos"], "k", ":")])),
    ("cos(true grad g_q, DFA p_q)", lines([(l, A(l, "cos_gq_pq"), COL[l], "-") for l in LN])),
    ("frac. episodes with store write lowering loss", lines([(l, A(l, "frac_store_write_descends"), COL[l], "-") for l in LN[1:]] +
                                                                  [(l + " toy Phi(sqrt d cot th)", A(l, "pred_phi"), COL[l], ":") for l in LN[1:]], ylim=(-0.03, 1.05))),
    ("min eig sym(J B^T) on 1-perp", lines([(l, A(l, "min_eig"), COL[l], "-") for l in LN], hline=(0, "0"))),
], "fig2_alignment.png")
# fig3: key geometry
grid("Key geometry", [
    ("within-sequence off-diagonal cosine", lines([(l, G(l, "off_diag_cos"), COL[l], "-") for l in LN], ylim=(-0.05, 1.05))),
    ("participation-ratio dimension of keys", lines([(l, G(l, "PR_dim"), COL[l], "-") for l in LN])),
    ("cos(store key, query key)", lines([(l, G(l, "cos_store_query"), COL[l], "-") for l in LN], ylim=(-0.05, 1.05))),
], "fig3_key_geometry.png")
# fig4: first-order effect of writes and fast fraction
grid("Effect of the fast weights at the answer step", [
    ("1st-order dL, store write (<0 helps)", lines([(l, A(l, "dL_store"), COL[l], "-") for l in LN], hline=(0, "0"))),
    ("1st-order dL, all of F", lines([(l, A(l, "dL_fast_total"), COL[l], "-") for l in LN], hline=(0, "0"))),
    ("|F x_q| / |pre-activation|", lines([(l, A(l, "Fq_over_pre"), COL[l], "-") for l in LN])),
    ("|F|_F at answer step", lines([(l, A(l, "Fnorm"), COL[l], "-") for l in LN])),
], "fig4_fast_effect.png")
# fig5: slow weights and readout
grid("Slow weights", [
    ("relative change |W-W0|/|W0|", lines([(l, (lambda l: lambda r: r["slow"][l]["rel_change"])(l), COL[l], "-") for l in LN] +
                                           [("i2o", lambda r: r["slow"]["i2o"]["rel_change"], "#8a3fb8", "-")], logy=True)),
    ("|J_l|_F (logit sensitivity to z_l)", lines([(l, A(l, "J_norm"), COL[l], "-") for l in LN])),
    ("|g_q| true grad / |p_q| DFA signal", lines([(l, (lambda l: lambda r: r["align"][l]["g_norm"]["mean"] / r["align"][l]["p_norm"]["mean"])(l), COL[l], "-") for l in LN])),
    ("PR of singular values^2 of dW", lines([(l, (lambda l: lambda r: r["slow"][l]["dW_PR_sv2"])(l), COL[l], "-") for l in LN])),
], "fig5_slow_readout.png")

EX = lambda n, key, sub="mean": (lambda r: r["answer_extra"][n][key][sub] if isinstance(r["answer_extra"][n][key], dict) else r["answer_extra"][n][key])
if all("answer_extra" in r for s_ in SEEDS for r in data[s_]):
    grid("Answer-step logit margin and prediction sharpness", [
        ("logit margin target - best other", lines([("full", EX("full", "margin"), "k", "-"), ("no fast", EX("no_fast@all", "margin"), "#888", "--"),
                                                    ("store term only (all layers)", EX("signal_only@all", "margin"), "#2d9e5a", ":")], hline=(0, "0"))),
        ("mean max-prob of prediction", lines([("full", EX("full", "pmax"), "k", "-"), ("no fast", EX("no_fast@all", "pmax"), "#888", "--")])),
        ("fraction of episodes with margin > 0", lines([("full", EX("full", "frac_margin_pos"), "k", "-"), ("no fast", EX("no_fast@all", "frac_margin_pos"), "#888", "--")])),
        ("share of the modal prediction (no fast)", lines([("no fast", EX("no_fast@all", "modal_pred_share"), "#888", "--"), ("full", EX("full", "modal_pred_share"), "k", "-")])),
    ], "fig6_margin.png")

# compact table
with open(f"{out}/table.md", "w") as f:
    for s in SEEDS:
        f.write(f"\n### seed {s}\n\n| iter | recall | noF | noF L2 | L2 store only | cosθ L0/L1/L2 | descend L2 | PR L2 | cos(s,q) L2 | Fq/pre L2 | dW L2 | |J2| |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in data[s]:
            a = r["align"]
            f.write(f"| {r['iteration']} | {r['ablation']['full']['recall']:.3f} | {r['ablation']['no_fast@all']['recall']:.3f} | {r['ablation']['no_fast@L2']['recall']:.3f} | "
                    f"{r['ablation']['signal_only@L2']['recall']:.3f} | {a['L0']['cosF']['mean']:.3f}/{a['L1']['cosF']['mean']:.3f}/{a['L2']['cosF']['mean']:.3f} | "
                    f"{a['L2']['frac_store_write_descends']:.2f} | {r['geom']['L2']['PR_dim']:.2f} | {r['geom']['L2']['cos_store_query']['mean']:.2f} | "
                    f"{a['L2']['Fq_over_pre']['mean']:.3f} | {r['slow']['L2']['rel_change']:.3f} | {a['L2']['J_norm']['mean']:.2f} |\n")
print("onsets", {s: onset(s) for s in SEEDS})
