"""Parse margin-study train logs: onset iterations, margin trajectories, tables and figures.
usage: analyze.py RUNS_DIR OUT_DIR"""
import json, re, sys, os, glob, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

runs_dir, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
blk = re.compile(r"--- Interval metrics \(ending @ iter (\d+)")


def parse(path):
    parts = blk.split(open(path).read())
    rows = []
    for i in range(1, len(parts), 2):
        it, body = int(parts[i]), parts[i + 1].split("-------------------------------------------")[0]
        d = {k: float(v) for k, v in re.findall(r"^  (\w+): (-?[0-9.]+(?:e-?\d+)?|nan)", body, re.M)}
        d["iter"] = it
        rows.append(d)
    return rows


def first(rows, key, thr):
    for r in rows:
        if r.get(key, 0) >= thr:
            return r["iter"]
    return None


res = {}
for d in sorted(glob.glob(f"{runs_dir}/h*_s*")):
    name = os.path.basename(d)
    m = re.match(r"h(\d+)_(.+)_s(\d+)$", name)
    if not m or not os.path.exists(f"{d}/train.log"):
        continue
    rows = parse(f"{d}/train.log")
    if not rows:
        continue
    hidden, arm, seed = int(m[1]), m[2], int(m[3])
    last = rows[-1]["iter"]
    early = [r.get("recall_acc", 0) for r in rows if r["iter"] <= 25000]
    mt = [(r["iter"], r.get("answer_margin")) for r in rows]
    res[name] = dict(
        hidden=hidden, arm=arm, seed=seed, last_iter=last,
        finished=os.path.exists(f"{d}/EARLYSTOP") or last >= 80000,
        t_first_nonzero=first(rows, "recall_acc", 1e-4), t05=first(rows, "recall_acc", 0.5),
        t099=first(rows, "recall_acc", 0.99),
        zero_first_25k=all(x == 0 for x in early), max_recall_le25k=max(early or [0]),
        margin_traj=mt, recall_traj=[(r["iter"], r.get("recall_acc")) for r in rows],
        pmax_traj=[(r["iter"], r.get("answer_p_max")) for r in rows])
    dm = {it: v for it, v in mt if v is not None}
    res[name]["margin_at"] = {k: dm.get(k) for k in (1000, 5000, 10000, 20000, 30000)}
    res[name]["t_margin_gt_-0.1"] = next((it for it, v in mt if v is not None and it > 3000 and v > -0.1), None)
    res[name]["min_margin"] = min([v for _, v in mt if v is not None] or [None])
json.dump(res, open(f"{out}/results.json", "w"), indent=1)


def fmt(x):
    return "never" if x is None else f"{x/1000:g}k"


def med(v):
    v = [x for x in v if x is not None]
    return f"{st.median(v)/1000:g}k" if v else "-"


lines = ["| width | arm | n (done/started) | median first>0 | median to 0.5 | median to 0.99 | to 0.5 per seed | to 0.99 per seed | recall 0 through 25k | margin@10k | min margin |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
groups = {}
for n, r in res.items():
    groups.setdefault((r["hidden"], r["arm"]), []).append(r)
for (h, a), rs in sorted(groups.items()):
    rs.sort(key=lambda r: r["seed"])
    done = [r for r in rs if r["finished"]]
    m10 = [r["margin_at"][10000] for r in rs if r["margin_at"].get(10000) is not None]
    mm = [r["min_margin"] for r in rs if r["min_margin"] is not None]
    lines.append(f"| {h} | {a} | {len(done)}/{len(rs)} | {med([r['t_first_nonzero'] for r in done])} | "
                 f"{med([r['t05'] for r in done])} | {med([r['t099'] for r in done])} | "
                 + ", ".join(f"{r['seed']}:{fmt(r['t05'])}" for r in rs) + " | "
                 + ", ".join(f"{r['seed']}:{fmt(r['t099'])}" for r in rs)
                 + f" | {sum(r['zero_first_25k'] for r in rs)}/{len(rs)} | "
                 + (f"{np.mean(m10):.2f}" if m10 else "-") + " | " + (f"{np.mean(mm):.2f}" if mm else "-") + " |")
open(f"{out}/table.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

cmap = plt.get_cmap("tab10")
for h in sorted({r["hidden"] for r in res.values()}):
    arms = sorted(a for (hh, a) in groups if hh == h)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for ci, a in enumerate(arms):
        for k, r in enumerate(groups[(h, a)]):
            for ax, key in zip(axes, ("recall_traj", "margin_traj", "pmax_traj")):
                xs = [x for x, y in r[key] if y is not None]
                ys = [y for x, y in r[key] if y is not None]
                ax.plot(xs, ys, color=cmap(ci), alpha=0.7, lw=1.2, label=a if k == 0 else None)
    for ax, t in zip(axes, ("training recall", "answer-step margin (correct - best other logit)", "answer-step max softmax prob")):
        ax.set_title(f"{t} (hidden {h})", fontsize=10)
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.3)
    axes[1].axhline(0, color="k", lw=0.6)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out}/fig_traj_h{h}.png", dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for ci, a in enumerate(arms):
        vals = [(r["t05"] or 82000) / 1000 for r in groups[(h, a)] if r["finished"]]
        ax.scatter([ci] * len(vals), vals, color=cmap(ci), s=40)
        if vals:
            ax.hlines(np.median(vals), ci - 0.3, ci + 0.3, color=cmap(ci))
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms, rotation=30)
    ax.set_ylabel("iteration of recall>=0.5 (k)")
    ax.set_title(f"hidden {h}: onset per seed (82 = never)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out}/fig_onset_h{h}.png", dpi=120)
    plt.close(fig)
