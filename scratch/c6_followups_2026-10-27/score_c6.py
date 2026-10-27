"""Score the C6 follow-ups (PREREG_fastlayers.md "Batch C6F"; draft queue_2026-10-27/prereg_drafts/C6.md): part Z (follow-up 2,
fast weights in trunk layer 0 only) and part X (follow-up 6, M3 vs A3 at 5 seeds to 600k). Metrics are Batch FS part L's, from
collect_L.py rows (load_eval.py at every 50k): AREA (mean held-out strict recall over the 11 loads), K50, strict per load, read-time
ablations, blowup onset (five adjacent 2.5k prints with loss > 5 or nonfinite) and memory onset (first print with train recall > 0.5).

Usage: python score_c6.py FS.jsonl L0.jsonl F6.jsonl
  FS.jsonl  Batch FS part L rows (ORC ~/fslt_2026-10-22/collect_L_orc_300k.jsonl): the full-model references and the 50k-300k
            history of the extended A3/M3 runs.
  L0.jsonl  collect_L.py ~/c6fu_2026-10-27/runs_L0 (arms A3f0, D3f0, M3f0).
  F6.jsonl  collect_L.py ~/c6fu_2026-10-27/runs_F6 (A3, M3: 350k-600k of the extended runs, 50k-600k of the new M3 seeds).
Rows of a later file replace rows of an earlier one with the same (arm, seed, checkpoint). A missing part is reported, not scored."""
import json
import statistics as S
import sys

Z_SEEDS = (2718, 3141, 1618)
X_SEEDS = (2718, 3141, 1618, 4241, 8128)
LOADS = ("1", "2", "4", "8", "12", "16", "24", "32", "40", "48", "52")

rows, onset, memory = {}, {}, {}
for path in sys.argv[1:]:
    for line in open(path):
        r = json.loads(line)
        key = (r["arm"], r["seed"])
        # onsets: a run's history is split across files (an extension's train.log starts at 300k): the first one found wins
        for table, field in ((onset, "blowup_onset"), (memory, "memory_onset")):
            if r.get(field) is not None:
                table[key] = min(table.get(key, r[field]), r[field])
        if r.get("checkpoint"):
            rows[(r["arm"], r["seed"], r["checkpoint"])] = r


def at(arm, ck, seeds):
    return {s: rows[(arm, s, ck)] for s in seeds if (arm, s, ck) in rows}


def mean(d, f):
    return S.mean(f(r) for r in d.values()) if d else float("nan")


def area(d):
    return mean(d, lambda r: r["AREA"])


def k50(d):
    return mean(d, lambda r: r["K50"] or 52)


def verdict(ok):
    return "PASS" if ok else "FAIL"


def table(arms, cks, seeds):
    for arm in arms:
        for ck in cks:
            d = at(arm, ck, seeds)
            if not d:
                continue
            print(f"  {arm:5s} {ck:>5s} n {len(d)}  AREA {area(d):.4f} [{min(r['AREA'] for r in d.values()):.3f}-"
                  f"{max(r['AREA'] for r in d.values()):.3f}]  K50 {k50(d):5.2f}  R@8 {mean(d, lambda r: r['strict']['8']):.3f}"
                  f"  R@52 {mean(d, lambda r: r['strict']['52']):.3f}  it/s {mean(d, lambda r: r['median_its'] or 0):.2f}")


def onsets(arm, seeds, before):
    return {s: onset.get((arm, s)) for s in seeds if onset.get((arm, s)) is not None and onset[(arm, s)] <= before}


# ---------------------------------------------------------------- part Z: fast weights in trunk layer 0 only (follow-up 2)
print("== Part Z (follow-up 2): --fast_layers 0, 3 seeds x 300k; references = Batch FS full-model arms, same seeds ==")
table(("A3f0", "A3", "D3f0", "D3", "M3f0", "M3"), ("50k", "150k", "300k"), Z_SEEDS)
pairs = {x: (at(x + "f0", "300k", Z_SEEDS), at(x, "300k", Z_SEEDS)) for x in ("A3", "D3", "M3")}
if all(len(z) == 3 for z, _ in pairs.values()):
    wins = {x: sum(z[s]["AREA"] > f[s]["AREA"] for s in Z_SEEDS) for x, (z, f) in pairs.items()}
    better = {x: area(z) > area(f) for x, (z, f) in pairs.items()}
    print(f"Z1 PRIMARY: A3f0 > A3 in {wins['A3']}/3 seed pairs -> {verdict(wins['A3'] == 3)}; mean AREA f0 > full in "
          f"{sum(better.values())}/3 structures ({', '.join(f'{x} {area(z) - area(f):+.4f}' for x, (z, f) in pairs.items())}) -> "
          f"{verdict(all(better.values()))}")
    losing = sum(sum(z[s]["AREA"] < f[s]["AREA"] for s in Z_SEEDS) >= 2 for z, f in pairs.values())
    print(f"   falsifier (f0 < full in >= 2/3 pairs for >= 2 structures): {losing} structures -> {'MET' if losing >= 2 else 'not met'}")
    a3f0, d3f0, m3f0 = (pairs[x][0] for x in ("A3", "D3", "M3"))
    print(f"Z2: mean AREA A3f0 {area(a3f0):.4f} (>= 0.85?) -> {verdict(area(a3f0) >= 0.85)}; best f0 arm mean "
          f"{max(area(a3f0), area(d3f0), area(m3f0)):.4f} -> {verdict(max(area(a3f0), area(d3f0), area(m3f0)) >= 0.85)}")
    e_z, e_f = at("A3f0", "50k", Z_SEEDS), at("A3", "50k", Z_SEEDS)
    if e_z and e_f:
        print(f"Z3: 50k AREA A3f0 {area(e_z):.4f} vs A3 {area(e_f):.4f} (A3f0 lower?) -> {verdict(area(e_z) < area(e_f))}")
    gap_f0 = area(d3f0) - area(a3f0)
    gap_full = area(pairs["D3"][1]) - area(pairs["A3"][1])
    print(f"Z4a: D3f0 - A3f0 {gap_f0:+.4f} (>= -0.03?) -> {verdict(gap_f0 >= -0.03)}; Z4b: M3f0 - A3f0 "
          f"{area(m3f0) - area(a3f0):+.4f} (>= -0.03?) -> {verdict(area(m3f0) - area(a3f0) >= -0.03)}; Z4c: |D3f0 - A3f0| "
          f"{abs(gap_f0):.4f} < full |D3 - A3| {abs(gap_full):.4f} -> {verdict(abs(gap_f0) < abs(gap_full))}")
    blown = {x: onsets(x + "f0", Z_SEEDS, 300000) for x in ("A3", "D3", "M3")}
    print(f"Z5: blowup onsets in the 9 f0 runs: {blown} -> {verdict(not any(blown.values()))}")
    drop0 = {f"{x}f0-{s}": r["read_drop0"]["8"] for x, (z, _) in pairs.items() for s, r in z.items() if "read_drop0" in r}
    print(f"Z6: read-time drop0 at K 8 (layer 0's fast read zeroed, i2h kept): "
          f"{' '.join(f'{k}:{v:.3f}' for k, v in drop0.items())} -> {verdict(len(drop0) == 9 and max(drop0.values()) <= 0.3)}")
    ratio = {x: mean(pairs[x][0], lambda r: r["median_its"] or 0) / mean(pairs[x][1], lambda r: r["median_its"] or 0)
             for x in ("D3", "M3")}
    print(f"Z7 (cost): it/s f0 / full: D3 {ratio['D3']:.2f} M3 {ratio['M3']:.2f} (>= 1.15 both?) -> "
          f"{verdict(min(ratio.values()) >= 1.15)}  (different A100 nodes: rough)")
    for x, (z, _) in pairs.items():
        ok = all(abs(r["read_only0"][k] - r["read_all"][k]) < 1e-9 for r in z.values() if "read_only0" in r for k in r["read_all"])
        print(f"   sanity {x}f0: read-time only0 == all (the deep layers have no fast part) -> {ok}")
else:
    print("  part Z incomplete: n per arm at 300k", {x + "f0": len(z) for x, (z, _) in pairs.items()})

# ---------------------------------------------------------------- part X: M3 vs A3, 5 seeds, 600k (follow-up 6)
print("\n== Part X (follow-up 6): M3 vs A3, 5 seeds each, to 600k ==")
table(("A3", "M3"), ("300k", "350k", "400k", "450k", "500k", "550k", "600k"), X_SEEDS)
a6, m6 = at("A3", "600k", X_SEEDS), at("M3", "600k", X_SEEDS)
if len(a6) == 5 and len(m6) == 5:
    wins = sum(m6[s]["AREA"] > a6[s]["AREA"] for s in X_SEEDS)
    gap = area(m6) - area(a6)
    print(f"X1 PRIMARY: mean AREA M3 - A3 {gap:+.4f} (> 0?), M3 > A3 in {wins}/5 pairs (>= 4?) -> {verdict(gap > 0 and wins >= 4)};"
          f" falsifier (M3 <= A3 in >= 3/5 pairs): {'MET' if 5 - wins >= 3 else 'not met'}")
    print("   pairs:", " ".join(f"{s}:{m6[s]['AREA'] - a6[s]['AREA']:+.3f}" for s in X_SEEDS))
    print(f"X2 (the L6 band): |M3 - A3| {abs(gap):.4f} <= 0.05 -> {verdict(abs(gap) <= 0.05)}")
    late = {arm: (area(at(arm, "500k", X_SEEDS)), area(at(arm, "600k", X_SEEDS))) for arm in ("A3", "M3")}
    print(f"X4: AREA 500k -> 600k: A3 {late['A3'][0]:.4f} -> {late['A3'][1]:.4f}, M3 {late['M3'][0]:.4f} -> {late['M3'][1]:.4f}"
          f" (both rise >= 0.02?) -> {verdict(all(b - a >= 0.02 for a, b in late.values()))}")
    blown = {arm: onsets(arm, X_SEEDS, 600000) for arm in ("A3", "M3")}
    worse = [f"{arm}-{s}" for arm, d in (("A3", a6), ("M3", m6)) for s in X_SEEDS
             if (arm, s, "300k") in rows and d[s]["AREA"] < rows[(arm, s, "300k")]["AREA"]]
    print(f"X5: blowup onsets {blown}; runs with AREA(600k) < AREA(300k): {worse} -> {verdict(not any(blown.values()) and not worse)}")
    print(f"X6: K50 M3 {k50(m6):.2f} vs A3 {k50(a6):.2f} (M3 larger?) -> {verdict(k50(m6) > k50(a6))}")
else:
    print(f"  600k incomplete: A3 {len(a6)}/5, M3 {len(m6)}/5 (onsets so far: "
          f"{ {arm: onsets(arm, X_SEEDS, 600000) for arm in ('A3', 'M3')} })")
new = at("M3", "300k", (4241, 8128))
if new:
    print(f"X3: new M3 seeds at 300k: {' '.join(f'{s}:{r['AREA']:.3f}' for s, r in new.items())} (in [0.61, 0.68]?) -> "
          f"{verdict(len(new) == 2 and all(0.61 <= r['AREA'] <= 0.68 for r in new.values()))}")
cont = [(arm, s, rows[(arm, s, "350k")]["AREA"] - rows[(arm, s, "300k")]["AREA"]) for arm in ("A3", "M3") for s in X_SEEDS[:5]
        if (arm, s, "350k") in rows and (arm, s, "300k") in rows and not (arm == "M3" and s in (4241, 8128))]
if cont:
    print(f"X7 (continuity of the extended runs): AREA 350k - 300k {' '.join(f'{a}-{s}:{d:+.3f}' for a, s, d in cont)} "
          f"(all >= -0.02?) -> {verdict(all(d >= -0.02 for *_, d in cont))}")
for arm in ("A3", "M3"):
    for ck in ("300k", "600k"):
        d = at(arm, ck, X_SEEDS)
        if d and all("read_only0" in r for r in d.values()):
            print(f"   exploratory {arm} {ck}: only0 - all at K 52 = {mean(d, lambda r: r['read_only0']['52'] - r['read_all']['52']):+.3f}")
