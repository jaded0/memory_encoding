"""Scores LV1-LV5 (PREREG_fastlayers.md, "Large and visible", 2026-10-16). usage: lv_score.py lv_all.jsonl [tn_all.jsonl]"""
import sys, json, statistics as stx

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
R = {(r["run"], r["iter"]): r for r in rows}
tn = {}
if len(sys.argv) > 2:
    for l in open(sys.argv[2]):
        if l.strip():
            r = json.loads(l); tn[(r["run"], r["iter"])] = r


def get(run, it, k):
    r = R.get((run, it))
    return None if r is None else r.get(k)


X01 = {"M1-RGs0.1-s620203": 200000, "M1-RGs0.1-s620307": 175000, "M1-RGs0.1-s620411": 200000, "RGs0.1-s620101": 200000}
M2 = {"M2-T1": 200000, "M2-T03": 225000, "M2-T0": 250000}
COLL = {**X01, **M2}
X003 = ["M1-RGs0.03-s620203", "M1-RGs0.03-s620307", "M1-RGs0.03-s620411", "RGs0.03-s620101"]
DC2 = [f"Rbreak-DC2-s{s}" for s in (620101, 620203, 620307, 620411, 620509)]
L01 = [f"Rbreak-L01-s{s}" for s in (620101, 620203, 620307, 620411, 620509)]
PLAIN = [f"A1-R-break-plain-s{s}" for s in (620101, 620203, 620307)]
surv = [(r["run"], r["iter"]) for r in rows if r["run"] in X003 or r["run"] in DC2]
out = []
p = out.append


def fmt(v):
    return "NA" if v is None else (f"{v:.3g}")


p(f"rows: {len(rows)}; reconstruction max |z - model output| over all rows: {max(r['recon_maxabs'] for r in rows):.2e}")
p("\nPer-run trunk-2 trajectory (iter: S2_p90 / V2_p50 / I2_p90):")
for run in sorted({r["run"] for r in rows}):
    its = sorted(it for (rn, it) in R if rn == run)
    p(f"  {run}: " + "  ".join(f"{it // 1000}k {fmt(get(run, it, 'S2_p90'))}/{fmt(get(run, it, 'V2_p50'))}/{fmt(get(run, it, 'I2_p90'))}" for it in its))
p("\nPer-run trunk-1 trajectory (iter: S1_p90 / V1_p50 / I1_p90):")
for run in sorted({r["run"] for r in rows}):
    its = sorted(it for (rn, it) in R if rn == run)
    p(f"  {run}: " + "  ".join(f"{it // 1000}k {fmt(get(run, it, 'S1_p90'))}/{fmt(get(run, it, 'V1_p50'))}/{fmt(get(run, it, 'I1_p90'))}" for it in its))

# LV1
x01_S = [get(r, it, "S2_p90") for r, it in X01.items()]
x01_I = [get(r, it, "I2_p90") for r, it in X01.items()]
medS, minI = stx.median(x01_S), min(x01_I)
ok = 0
p(f"\nLV1: x0.1 pre-collapse S2_p90 {[round(v, 3) for v in x01_S]} (median {medS:.3f}); I2_p90 {[round(v, 3) for v in x01_I]} (min {minI:.3f})")
for r in DC2:
    S, V, I = get(r, 600000, "S2_p90"), get(r, 600000, "V2_p50"), get(r, 600000, "I2_p90")
    good = S >= medS and V < 0.5 and I < minI
    ok += good
    p(f"  {r} 600k: S2_p90 {S:.3f} (>= {medS:.3f}: {S >= medS})  V2_p50 {V:.3f} (< 0.5: {V < 0.5})  I2_p90 {I:.3f} (< {minI:.3f}: {I < minI})")
p(f"  LV1 {'PASS' if ok == 5 else 'FAIL'} ({ok}/5)")

# LV2
a = [(r, get(r, it, "V2_p50")) for r, it in X01.items()]
b = [(r, get(r, 300000, "V2_p50")) for r in X003]
na, nb_ = sum(v > 1 for _, v in a), sum(v is not None and v > 1 for _, v in b)
p(f"\nLV2: x0.1 pre-collapse V2_p50 {[round(v, 3) for _, v in a]} ({na}/4 > 1); x0.03 300k V2_p50 {[fmt(v) for _, v in b]} ({nb_}/4 > 1)")
p(f"  LV2 {'PASS' if na == 4 and nb_ == 4 else 'FAIL'}")


# LV3 and the comparison statistics
def sep(key, source):
    c = [(r, it, source(r, it, key)) for r, it in COLL.items()]
    s = [(r, it, source(r, it, key)) for r, it in surv if source(r, it, key) is not None]
    cmin, smax = min(v for *_, v in c), max(v for *_, v in s)
    overlap_s = [(r, it, round(v, 3)) for r, it, v in s if v >= cmin]
    overlap_c = [(r, it, round(v, 3)) for r, it, v in c if v <= smax]
    return c, s, cmin, smax, overlap_s, overlap_c


def tnget(r, it, k):
    t = tn.get((r, it))
    return None if t is None else t.get(k)


p("\nLV3: separation of the 7 collapse pre-checkpoints from every surviving checkpoint (x0.03 all, DC2 150k/600k)")
for label, key, src in (("I2_p90 (impact)", "I2_p90", get), ("S2_p90 (size, real writes)", "S2_p90", get),
                        ("V2_p50 (visibility)", "V2_p50", get), ("TN ratio_2_p90 (no-write proxy)", "ratio_2_p90", tnget)):
    try:
        c, s, cmin, smax, os_, oc = sep(key, src)
    except (ValueError, TypeError):
        p(f"  {label}: not available"); continue
    p(f"  {label}: collapse min {cmin:.3f}, survivor max {smax:.3f} -> {'separates' if cmin > smax else 'OVERLAP'};"
      f" survivors >= collapse min: {len(os_)}/{len(s)} {os_[:6]}; collapse <= survivor max: {len(oc)}/{len(c)} {oc}")
c, s, cmin, smax, *_ = sep("I2_p90", get)
p(f"  LV3 {'PASS' if cmin > smax else 'FAIL'}")

# LV4
k = 0
p("\nLV4: x0.1 I2_p90 100k -> pre-collapse")
for r, it in X01.items():
    a0, a1 = get(r, 100000, "I2_p90"), get(r, it, "I2_p90")
    k += a1 / a0 >= 2
    p(f"  {r}: {a0:.3f} -> {a1:.3f} ({a1 / a0:.2f}x)")
p(f"  LV4 {'PASS' if k >= 3 else 'FAIL'} ({k}/4 >= 2x)")

# LV5
p("\nLV5: plain V2_p50 at 50k-110k (all < 0.5?) and s620101 I2_p90 at 110k vs survivors' max")
allv = True
for r in PLAIN:
    vs = [(it // 1000, get(r, it, "V2_p50")) for it in range(50000, 120000, 10000)]
    allv &= all(v is not None and v < 0.5 for _, v in vs)
    p(f"  {r}: " + " ".join(f"{i}k {fmt(v)}" for i, v in vs))
i110 = get(PLAIN[0], 110000, "I2_p90")
p(f"  s620101 110k I2_p90 {fmt(i110)} vs survivors' max {smax:.3f}: {'below (a)' if i110 < smax else 'above (b)'}")
p(f"  LV5 {'PASS' if allv and i110 < smax else 'FAIL'} (V part {'pass' if allv else 'fail'}; impact part {'(a)' if i110 < smax else '(b)'})")
print("\n".join(out))
