"""Re-forward drift test collector (PREREG_fastlayers.md, "Re-forward drift test (2026-10-16)").
usage: rf_collect.py RUN_DIR [RUN_DIR ...] -> one JSON line per run: late onset (5 consecutive 500-step interval losses > 5,
starting after 30k, the batch M rule restricted past the early excursion), early-excursion peak and first iteration after
it with loss < 3, and the held-out traces (every 50k, 8 batches) from train.log."""
import sys, os, re, json, math

def parse(path):
    L, held, it = [], {}, None
    cur = None
    for line in open(path, errors="replace"):
        s = line.strip()
        m = re.match(r"--- Interval metrics \(ending @ iter (\d+)", s)
        if m:
            it = int(m.group(1)); continue
        if s.startswith("loss:") and it is not None:
            try: v = float(s.split()[1])
            except ValueError: v = float("nan")
            L.append((it, v)); cur = it; it = None; continue
        m = re.match(r"heldout_(strict|no_fast|observed)/recall_acc: ([\d.na]+)", s)
        if m and cur is not None:
            held.setdefault(m.group(1), []).append((cur, float(m.group(2))))
    return L, held

def onset(L, after=30000, thr=5.0, k=5):
    vals = [(i, v) for i, v in L if i > after]
    for j in range(len(vals) - k + 1):
        if all((not math.isfinite(v)) or v > thr for _, v in vals[j:j + k]):
            return vals[j][0]
    return None

for run in sys.argv[1:]:
    L, held = parse(os.path.join(run, "train.log"))
    early = [(i, v) for i, v in L if i <= 30000]
    peak_i, peak_v = max(early, key=lambda x: x[1] if math.isfinite(x[1]) else float("inf")) if early else (None, None)
    rec = next((i for i, v in L if peak_i is not None and i > peak_i and math.isfinite(v) and v < 3), None)
    final = None
    fe = os.path.join(run, "final_eval.json")
    if os.path.exists(fe):
        j = json.load(open(fe))
        final = {k.split("/")[0].replace("heldout_", ""): round(j[k], 4) for k in j if k.endswith("/recall_acc")}
    print(json.dumps(dict(run=os.path.basename(os.path.normpath(run)), last_iter=L[-1][0] if L else None,
                          onset=onset(L), early_peak_iter=peak_i, early_peak_loss=peak_v, first_below3_after_peak=rec,
                          nonfinite=any(not math.isfinite(v) for _, v in L), held=held, final=final)))
