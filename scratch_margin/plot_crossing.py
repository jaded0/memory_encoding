"""usage: plot_crossing.py OUT_DIR  (reads OUT_DIR/results.json from analyze.py)"""
import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1]
res = json.load(open(f"{out}/results.json"))
groups = {}
for r in res.values():
    groups.setdefault((r["hidden"], r["arm"]), []).append(r)
cmap = plt.get_cmap("tab10")
fig, ax = plt.subplots(figsize=(6.4, 5.4))
for ci, ((h, a), rs) in enumerate(sorted(groups.items())):
    pts = [(r["t_margin_gt_-0.1"], r["t05"]) for r in rs if r["t05"] and r["t_margin_gt_-0.1"]]
    ax.scatter([p[0] / 1000 for p in pts], [p[1] / 1000 for p in pts], color=cmap(ci % 10),
               marker="o" if h == 256 else "^", label=f"{a} (h{h})", s=34)
ax.plot([15, 70], [15, 70], "k--", lw=0.7)
ax.set_xlabel("iteration where answer-step margin first > -0.1 (k)")
ax.set_ylabel("iteration of recall >= 0.5 (k)")
ax.set_title("recall flips when the margin closes, in every arm", fontsize=10)
ax.legend(fontsize=7)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{out}/fig_onset_vs_margin_crossing.png", dpi=120)
