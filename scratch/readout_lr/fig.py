import csv, json, collections, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
C = sys.argv[1]; out = sys.argv[2]
ARMS = [("CTRL", "control", "#2a78d6"), ("RO3", "readout-only ×0.3", "#eb6834"),
        ("TR3", "trunk-only ×0.3", "#1baf7a"), ("GLOBAL3", "global ×0.3", "#eda100")]
rows = collections.defaultdict(list)
for r in csv.DictReader(open(f"{C}/metrics.csv")):
    rows[r["run"]].append((int(r["iter"]), float(r["loss"])))
def binned(points, width=2500):  # 500-it windows -> 2500-it means, matching GLOBAL3's print_freq
    acc = collections.defaultdict(list)
    for it, v in points:
        acc[(it - 1) // width * width + width].append(v)
    return sorted((k, sum(v) / len(v)) for k, v in acc.items())
summ = json.load(open(f"{C}/summary.json"))
b150 = summ["B150"]["checkpoints"][0]["slow_fro"]
def norms(name, layer):
    keys = {"GLOBAL3": ["INT2_R2", "INT3_R2"]}.get(name, [name])
    pts = [(150000, b150[layer])]
    for k in keys:
        pts += [(c["iter"], c["slow_fro"][layer]) for c in summ[k]["checkpoints"]]
    return sorted(pts)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e"})
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor="#fcfcfb")
panels = [("Training loss (2.5k-iteration means, log)", None),
          ("Readout (i2o) slow-weight norm", "i2o"), ("Trunk layer 2 slow-weight norm", "linear_layers.2")]
for ax, (title, layer) in zip(axes, panels):
    ax.set_facecolor("#fcfcfb"); ax.grid(axis="y", color="#e6e5e0", lw=0.8); ax.set_axisbelow(True)
    for name, label, color in ARMS:
        if layer is None:
            pts = binned(rows[name]) if name != "GLOBAL3" else rows[name]
            x, y = zip(*pts); ax.plot([i / 1000 for i in x], y, color=color, lw=2, label=label)
        else:
            x, y = zip(*norms(name, layer))
            ax.plot([i / 1000 for i in x], y, color=color, lw=2, marker="o", ms=5,
                    markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=label)
    ax.set_title(title, loc="left", fontsize=11, color="#0b0b0b")
    ax.set_xlabel("iteration (k)"); ax.set_xlim(145, 455)
    if layer is None:
        ax.set_yscale("log"); ax.axhline(5, color="#8a8984", lw=1, ls="--")
        ax.text(452, 5.6, "collapse threshold (5)", ha="right", fontsize=8, color="#52514e")
axes[0].legend(frameon=False, loc="upper left")
fig.suptitle("Slow learning rate ×0.3 from B 150k: readout only, trunk only, or both (seed 3141)",
             x=0.01, ha="left", fontsize=12, color="#0b0b0b")
fig.tight_layout(); fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
