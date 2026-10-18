"""Re-forward drift test scoring (PREREG_fastlayers.md, "Re-forward drift test (2026-10-16)", RF0-RF5 and ADDENDUM 1 RW1-RW5).
usage: rf_score.py DATA_DIR  (rf_losses.jsonl, rf_collect.jsonl, rf_dx.jsonl, rf_sg.jsonl, rf_lv.jsonl)"""
import sys, os, json, math, statistics as st
D = sys.argv[1]
rd = lambda f: [json.loads(l) for l in open(os.path.join(D, f)) if l.startswith("{")]
loss = {r["run"]: r["loss"] for r in rd("rf_losses.jsonl")}
col = {r["run"]: r for r in rd("rf_collect.jsonl")}
dx = {(r["run"], r["iter"]): r for r in rd("rf_dx.jsonl")}
sg = {(r["ckpt"].split("/")[0], r["iter"]): r for r in rd("rf_sg.jsonl")}
lv = {(r["run"], r["iter"]): r for r in rd("rf_lv.jsonl")}
seeds = ["620101", "620203", "620307"]
med = lambda xs: st.median(xs) if xs else float("nan")

def onset(L, after, thr=5.0, k=5):
    v = [(i, x) for i, x in L if i > after]
    for j in range(len(v) - k + 1):
        if all((not math.isfinite(x)) or x > thr for _, x in v[j:j + k]): return v[j][0]
    return None
def first_above(L, thr, after=0):
    return next((i for i, x in L if i > after and ((not math.isfinite(x)) or x > thr)), None)
def held(run, kind, it):
    return dict(map(tuple, col[run]["held"].get(kind, []))).get(it)
def sv(run, it):
    r = dx.get((run, it))
    if r is None and run.startswith("W") and it == 50000: r = dx.get(("K1-" + run.split("-")[1], 50000))
    return r["W2_top_sv"] if r else None

rows = {}
for run, L in sorted(loss.items()):
    after = 60000 if run.startswith("W") else 30000
    on = onset(L, after)
    rows[run] = dict(run=run, last=L[-1][0], onset=on, blowup_1e3=first_above(L, 1e3), min_loss_after_50k=round(min(x for i, x in L if i > 50000), 2) if L[-1][0] > 50000 else None,
                     loss_50_60k=round(med([x for i, x in L if 50000 < i <= 60000]), 2) if L[-1][0] > 60000 else None,
                     sv_ratio_150_50=(round(sv(run, 150000) / sv(run, 50000), 3) if sv(run, 150000) and sv(run, 50000) else None),
                     sv_ratio_100_50=(round(sv(run, 100000) / sv(run, 50000), 3) if sv(run, 100000) and sv(run, 50000) else None),
                     shareB2_50k=sg.get((run, 50000), {}).get("share_B2"), shareB2_100k=sg.get((run, 100000), {}).get("share_B2"),
                     strict_100k=held(run, "strict", 100000), strict_150k=held(run, "strict", 150000),
                     nofast_100k=held(run, "no_fast", 100000), S2p90_100k=lv.get((run, 100000), {}).get("S2_p90"),
                     I2p90_100k=lv.get((run, 100000), {}).get("I2_p90"), final=col[run]["final"])
for r in rows.values(): print(json.dumps(r))
print()
K1 = [rows[f"K1-s{s}"] for s in seeds]
cens = lambda r: r["onset"] if r["onset"] else 350000
# RF0
n1 = sum(r["onset"] is not None for r in K1)
print(f"RF0 K=1 collapses (onset after 30k) {n1}/3: onsets {[r['onset'] for r in K1]} -> {'PASS' if n1 >= 2 else 'FAIL'}")
# RF1 / RF4 (from scratch): K>=2 never healthy
for K, runs in (("2", ["K2-s620203"]), ("4", [f"K4-s{s}" for s in seeds])):
    rs = [rows[r] for r in runs]
    print(f"RF1 K={K} from scratch: onset(after 30k) {[r['onset'] for r in rs]}, never below loss 3 after the early peak: "
          f"{[col[r['run']]['first_below3_after_peak'] is None for r in rs]} -> scored as 'never healthy' (harm)")
    s100 = [r["strict_100k"] for r in rs]
    print(f"RF4 K={K}: strict@100k {s100} vs K=1 median {med([r['strict_100k'] for r in K1]):.3f}; strict-no_fast {[round(r['strict_100k']-r['nofast_100k'],3) for r in rs]}")
# RF2/RW3 drift
k1sv = med([r["sv_ratio_150_50"] for r in K1])
print(f"RF2/RW3 trunk-2 slow top SV 150k/50k: K=1 {[r['sv_ratio_150_50'] for r in K1]} (median {k1sv:.3f})")
for p in ("W2", "W4"):
    xs = [rows[f'{p}-s{s}']['sv_ratio_150_50'] for s in seeds]
    print(f"   {p}: {xs} (median {med(xs):.3f}; within x1.5 of K=1: {k1sv/1.5 <= med(xs) <= k1sv*1.5})")
print(f"   from scratch 100k/50k: K=1 {[r['sv_ratio_100_50'] for r in K1]}; K2 {rows['K2-s620203']['sv_ratio_100_50']}; K4 {[rows[f'K4-s{s}']['sv_ratio_100_50'] for s in seeds]}")
# RF3
sh = {r: rows[r]["shareB2_50k"] for r in rows if rows[r]["shareB2_50k"] is not None}
print(f"RF3 share in range(B2) at 50k (random 0.0087), {len(sh)} runs: {sh} -> {'PASS' if sh and all(v < 0.0087 for v in sh.values()) else 'FAIL'}")
print(f"   share at 100k: { {r: rows[r]['shareB2_100k'] for r in rows if rows[r]['shareB2_100k'] is not None} }")
# RF5/RW5
k1s = med([r["S2p90_100k"] for r in K1])
print(f"RF5/RW5 S2 p90 @100k: K=1 {[round(r['S2p90_100k'],3) for r in K1]} (median {k1s:.3f}); K4 scratch {[round(rows[f'K4-s{s}']['S2p90_100k'],3) for s in seeds]}; "
      f"W4 {[round(rows[f'W4-s{s}']['S2p90_100k'],3) for s in seeds]}; W2 {[round(rows[f'W2-s{s}']['S2p90_100k'],3) for s in seeds]}")
# RW1
for p in ("W2", "W4"):
    print(f"RW1 {p}: interval-loss median 50-60k {[rows[f'{p}-s{s}']['loss_50_60k'] for s in seeds]} (K=1 {[r['loss_50_60k'] for r in K1]})")
# RW2
for p in ("W2", "W4"):
    rs = [rows[f"{p}-s{s}"] for s in seeds]
    print(f"RW2 {p}: onset(after 60k) {[r['onset'] for r in rs]} vs K=1 {[r['onset'] for r in K1]}; loss>1e3 first at {[r['blowup_1e3'] for r in rs]} vs K=1 {[r['blowup_1e3'] for r in K1]}; "
          f"median onset {med([cens(r) for r in rs])} vs {med([cens(r) for r in K1])}")
# RW4
for p in ("W2", "W4"):
    for it in (100000, 150000):
        k = "strict_100k" if it == 100000 else "strict_150k"
        print(f"RW4 {p} strict@{it//1000}k {[rows[f'{p}-s{s}'][k] for s in seeds]} vs same-seed K=1 {[r[k] for r in K1]}")
