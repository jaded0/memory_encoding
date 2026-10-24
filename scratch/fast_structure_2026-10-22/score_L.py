"""Score Batch FS part L predictions L1-L7 (PREREG_fastlayers.md "Batch FS") at one checkpoint.
Usage: python score_L.py collect_L.jsonl CHECKPOINT (e.g. 150k, 300k)"""
import json, math, statistics as S, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
ck = sys.argv[2]
def arm(a, c=ck):
    return {r["seed"]: r for r in rows if r["arm"] == a and r.get("checkpoint") == c}
def final(a):  # diverged arms: load_final
    return arm(a, ck) or arm(a, "final")
A3, D3, M3, A10, D10 = arm("A3"), arm("D3"), arm("M3"), final("A10"), final("D10")
m = lambda d, f: S.mean(f(r) for r in d.values())
area = lambda d: m(d, lambda r: r["AREA"])
def R(r, k): return r["strict"][str(k)]
print(f"checkpoint {ck}: n A3 {len(A3)} D3 {len(D3)} M3 {len(M3)} A10 {len(A10)} D10 {len(D10)}")
for n, d in (("A3", A3), ("D3", D3), ("M3", M3), ("A10", A10), ("D10", D10)):
    print(f"  {n:4s} AREA mean {area(d):.4f} [{min(r['AREA'] for r in d.values()):.3f}-{max(r['AREA'] for r in d.values()):.3f}]"
          f"  K50 mean {m(d, lambda r: r['K50'] or 52):.2f}  R@4 {m(d, lambda r: R(r,4)):.3f} R@8 {m(d, lambda r: R(r,8)):.3f} R@52 {m(d, lambda r: R(r,52)):.3f}")
# L1
r4 = {f"{n}-{s}": R(r, 4) for n, d in (("A3", A3), ("D3", D3)) for s, r in d.items()}
nf = max(v for d in (A3, D3) for r in d.values() for v in r["no_fast"].values())
nf_all = max(v for d in (A3, D3, M3, A10, D10) for r in d.values() for v in r["no_fast"].values())
print(f"L1: min R@4 (A3,D3) {min(r4.values()):.3f}; max no_fast A3/D3 {nf:.3f}, all arms {nf_all:.3f} ->",
      "PASS" if min(r4.values()) >= 0.5 and nf <= 0.15 else "FAIL")
# L2
seeds = sorted(set(A3) & set(D3))
gap = area(D3) - area(A3); wins = sum(D3[s]["AREA"] > A3[s]["AREA"] for s in seeds)
fals = sum(A3[s]["AREA"] > D3[s]["AREA"] + 0.05 for s in seeds)
print(f"L2a: AREA D3-A3 {gap:+.4f} (>= -0.03?) -> {'PASS' if gap >= -0.03 else 'FAIL'}; L2b: D3>A3 in {wins}/{len(seeds)} pairs ->",
      "PASS" if wins >= 4 else "FAIL", f"; falsifier A3>D3+0.05 in {fals}/{len(seeds)}")
print("   pairs:", " ".join(f"{s}:{D3[s]['AREA']-A3[s]['AREA']:+.3f}" for s in seeds))
# L3
def brk(d):
    ks = sorted(int(k) for k in next(iter(d.values()))["strict"])
    mc = {k: m(d, lambda r: R(r, k)) for k in ks}
    return max(mc.values()) - mc[52]
k50 = lambda d: m(d, lambda r: r["K50"] or 52)
lr = abs(math.log2(k50(A3) / k50(D3)))
print(f"L3a: drop max->K52 A3 {brk(A3):.3f} D3 {brk(D3):.3f} (>=0.15 both?) -> {'PASS' if min(brk(A3), brk(D3)) >= 0.15 else 'FAIL'};"
      f" L3b: K50 A3 {k50(A3):.2f} D3 {k50(D3):.2f} |log2| {lr:.3f} -> {'PASS' if lr <= 0.6 else 'FAIL'}")
# L4
dec = lambda d: m(d, lambda r: R(r, 8) - R(r, 52))
print(f"L4: decline K8->52 A3 {dec(A3):.3f} D3 {dec(D3):.3f}; A3 shallower by {dec(D3)-dec(A3):+.3f} (>0.05 = FAIL) ->",
      "PASS" if dec(D3) - dec(A3) <= 0.05 else "FAIL")
# L5b, L5c
g10 = area(D10) - area(A10)
same = (math.copysign(1, g10) == math.copysign(1, gap)) or (abs(g10) <= 0.03 and abs(gap) <= 0.03)
print(f"L5b: D10-A10 {g10:+.4f}, D3-A3 {gap:+.4f} -> {'PASS' if same else 'FAIL'}")
print(f"L5c: D10 {area(D10):.4f} vs D3 {area(D3):.4f}; A10 {area(A10):.4f} vs A3 {area(A3):.4f} ->",
      "PASS" if area(D10) > area(D3) and area(A10) > area(A3) else "FAIL")
# L6
if M3:
    d6 = area(M3) - area(A3); d6m = area(M3) - S.mean(A3[s]["AREA"] for s in M3)
    print(f"L6: AREA M3-A3 {d6:+.4f} (3-seed matched {d6m:+.4f}) -> {'PASS' if abs(d6) <= 0.05 else 'FAIL'}")
# L7
for n, d, in (("A3", A3), ("D3", D3)):
    ok = []
    for s, r in d.items():
        if "read_only0" not in r: ok.append(None); continue
        ok.append(all(r["read_only0"][k] >= r["read_all"][k] for k in r["read_all"] if int(k) >= 4))
    marg = [min(r["read_only0"][k] - r["read_all"][k] for k in r["read_all"] if int(k) >= 4) for r in d.values() if "read_only0" in r]
    print(f"L7 {n}: only0>=all at all K>=4 in {sum(bool(x) for x in ok)}/{len(ok)} seeds (min margin per seed {' '.join(f'{x:+.3f}' for x in marg)}) ->",
          "PASS" if sum(bool(x) for x in ok) >= 4 else "FAIL")
    print(f"   mean only0/all at K=8: {m(d, lambda r: r['read_only0']['8']):.3f}/{m(d, lambda r: r['read_all']['8']):.3f}, K=52: {m(d, lambda r: r['read_only0']['52']):.3f}/{m(d, lambda r: r['read_all']['52']):.3f}")
