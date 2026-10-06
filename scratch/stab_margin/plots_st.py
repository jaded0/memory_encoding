#!/usr/bin/env python3
"""Plot stabilizer-arm kick traces and sampled edges."""
import argparse, json, math
from pathlib import Path
import matplotlib.pyplot as plt
from edge_st import infer

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cells", default="cells_st.json")
    ap.add_argument("--health", default="data/health_rows.json"); a = ap.parse_args()
    cells = json.loads(Path(a.cells).read_text()); arms = sorted({c["arm"] for c in cells.values()})
    stages = sorted({c["stage"] for c in cells.values()}); colors = {}
    ms = sorted({c["m"] for c in cells.values()}); cmap = plt.get_cmap("viridis")
    for j, m in enumerate(ms): colors[m] = cmap(j / max(1, len(ms) - 1))
    fig, axes = plt.subplots(max(1, len(arms)), max(1, len(stages)), squeeze=False,
                             figsize=(4 * max(1, len(stages)), 2.8 * max(1, len(arms))), sharex="col", sharey=True)
    for ri, arm in enumerate(arms):
        for ci, stage in enumerate(stages):
            ax = axes[ri][ci]
            for name, c in sorted(cells.items()):
                if c["arm"] != arm or c["stage"] != stage: continue
                pairs = [(i - stage, y) for i, y in zip(c["its"], c["series"]["loss"])
                         if y is not None and y > 0 and math.isfinite(y)]
                if pairs:
                    x, y = zip(*pairs); label = f"m={c['m']:g}" + (f" r{c['reseed']}" if c["reseed"] is not None else "")
                    ax.plot(x, y, color=colors[c["m"]], alpha=.8, lw=1.2, label=label)
            ax.set_yscale("log"); ax.grid(alpha=.25); ax.set_title(f"{arm}, stage {stage:,}")
            if ri == len(arms) - 1: ax.set_xlabel("iterations after kick")
            if ci == 0: ax.set_ylabel("interval loss")
            if ax.lines: ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig("fig_b_traces.png", dpi=180); plt.close(fig)

    edges = infer(cells); fig, ax = plt.subplots(figsize=(6.5, 4))
    for arm in arms:
        rr = [r for r in edges if r["arm"] == arm and r["edge"] is not None]
        if not rr: continue
        x = [r["stage"] for r in rr]; y = [r["edge"] for r in rr]
        low = [r["edge"] - r["bracket_low"] if r["bracket_low"] is not None else 0 for r in rr]
        ax.errorbar(x, y, yerr=[low, [0] * len(low)], marker="o", capsize=4, label=arm)
    health = json.loads(Path(a.health).read_text()) if Path(a.health).exists() else []
    for lin, label, style in (("B", "plain-recipe B", "--"), ("L4241", "4241", ":")):
        rr = sorted((r for r in health if r.get("lin") == lin and r.get("edge") is not None), key=lambda r: r["stage"])
        if rr: ax.plot([r["stage"] for r in rr], [r["edge"] for r in rr], style, lw=1.5, label=label)
    ax.set(xlabel="stage", ylabel="sampled edge m*", title="Stabilizer-arm alpha-kick stability edge")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig("fig_b_edge.png", dpi=180); plt.close(fig)
    print("wrote fig_b_traces.png and fig_b_edge.png")

if __name__ == "__main__": main()
