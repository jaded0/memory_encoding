"""usage: analyze_floor.py RUNS_DIR OUT_DIR   (per-arm table, margin@10k, plot of recall/margin)"""
import json, re, sys, os, glob, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
runs, out = sys.argv[1:3]; os.makedirs(out, exist_ok=True)
blk = re.compile(r"--- Interval metrics \(ending @ iter (\d+)")
def parse(path):
    parts = blk.split(open(path).read()); rows = []
    for i in range(1, len(parts), 2):
        body = parts[i + 1].split("-------------------------------------------")[0]
        d = {k: float(v) for k, v in re.findall(r"^  (\w+): (-?[0-9.]+(?:e[-+]?\d+)?|nan|inf)", body, re.M)}
        d["iter"] = int(parts[i]); rows.append(d)
    return rows
first = lambda rows, t: next((r["iter"] for r in rows if r.get("recall_acc", 0) >= t), None)
ARMS = {}
for d in sorted(glob.glob(f"{runs}/*_s*")):
    n = os.path.basename(d); arm, s = n.rsplit("_s", 1)
    if not os.path.exists(f"{d}/train.log"): continue
    rows = parse(f"{d}/train.log")
    if not rows: continue
    mt = {r["iter"]: r.get("answer_margin") for r in rows}
    bad = any(r.get("loss", 0) != r.get("loss", 0) or r.get("loss", 0) > 1e3 for r in rows)
    ARMS.setdefault(arm, {})[int(s)] = dict(t05=first(rows, .5), t099=first(rows, .99), m10=mt.get(10000),
        last=rows[-1]["iter"], done=os.path.exists(f"{d}/EARLYSTOP") or rows[-1]["iter"] >= 60000, diverged=bad,
        maxrec=max(r.get("recall_acc", 0) for r in rows), recall=[(r["iter"], r.get("recall_acc")) for r in rows], margin=list(mt.items()),
        loss=[(r["iter"], r.get("loss")) for r in rows])
json.dump(ARMS, open(f"{out}/results.json", "w"), indent=1)
f = lambda x: "never" if x is None else f"{x/1000:g}k"
def med(v):
    v = [x for x in v if x is not None]; return f"{st.median(v)/1000:g}k" if v else "-"
L = ["| arm | n (done) | median to 0.5 | median to 0.99 | to 0.5 per seed | to 0.99 per seed | margin@10k per seed | diverged/maxrecall |", "|---|---|---|---|---|---|---|---|"]
for a, v in ARMS.items():
    ss = sorted(v); m10 = [v[s]["m10"] for s in ss if v[s]["m10"] is not None]
    L.append(f"| {a} | {len(ss)} ({sum(v[s]['done'] for s in ss)}) | {med([v[s]['t05'] for s in ss])} | {med([v[s]['t099'] for s in ss])} | "
      + ", ".join(f"{s}:{f(v[s]['t05'])}" for s in ss) + " | " + ", ".join(f"{s}:{f(v[s]['t099'])}" for s in ss) + " | "
      + (", ".join(f"{x:.2f}" for x in m10) + f" (med {st.median(m10):.2f})" if m10 else "-")
      + " | " + ", ".join(f"{s}:{'DIV' if v[s]['diverged'] else 'ok'}/{v[s]['maxrec']:.2f}" for s in ss) + " |")
open(f"{out}/table.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5)); cm = plt.get_cmap("tab10")
for i, (a, v) in enumerate(ARMS.items()):
    for s, r in v.items():
        rc = [(x, y) for x, y in r["recall"] if y is not None]
        if rc: ax[0].plot(*zip(*rc), color=cm(i), alpha=.7, label=a if s == sorted(v)[0] else None)
        mm = [(x, y) for x, y in r["margin"] if y is not None]
        if mm: ax[1].plot(*zip(*mm), color=cm(i), alpha=.7)
ax[0].set_xlabel("iter"); ax[0].set_ylabel("train recall"); ax[0].legend(fontsize=7)
ax[1].set_xlabel("iter"); ax[1].set_ylabel("answer margin"); ax[1].axhline(0, color="k", lw=.5)
plt.tight_layout(); plt.savefig(f"{out}/fig_floor.png", dpi=130)
