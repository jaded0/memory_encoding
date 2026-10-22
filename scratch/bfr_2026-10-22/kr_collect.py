"""Batch BFR KR: one JSON line per key-recall run (agent A, 2026-10-22). Usage: python kr_collect.py RUNS_DIR [RUNS_DIR ...] > kr_runs.jsonl
From train.log (print_freq 100): blowup onset = first of five adjacent 500-it windows (5 prints each) with mean loss > 5 (RN/V2 definition);
memory onset = first print with train recall_acc > 0.5; mean loss over iterations 10-1000; max loss in the first 100; train loss/recall at the
last print. From heldout.json (128 batches, last checkpoint): observed/strict/no_fast recall and loss."""
import glob, json, math, os, re, sys

for root in sys.argv[1:]:
    for d in sorted(glob.glob(os.path.join(root, "kr_*"))):
        name = os.path.basename(d)
        m = re.match(r"kr_(plain|DC2)_(off|rms|ln|absrun)_s(\d+)$", name)
        if not m:
            continue
        try:
            txt = open(os.path.join(d, "train.log"), errors="replace").read()
        except OSError:
            continue
        it, loss, rec = [], [], []
        for mm in re.finditer(r"ending @ iter (\d+).*?\n  loss: ([^\n]+)\n.*?recall_acc: ([^\n]+)\n", txt, re.S):
            it.append(int(mm.group(1)))
            try:
                loss.append(float(mm.group(2)))
            except ValueError:
                loss.append(float("nan"))
            rec.append(float(mm.group(3)))
        win = [sum(loss[i:i + 5]) / 5 for i in range(0, len(loss) - len(loss) % 5, 5)]
        onset = next(((i + 1) * 500 for i in range(len(win) - 4)
                      if all((not math.isfinite(w)) or w > 5 for w in win[i:i + 5])), None)
        early = [l for i_, l in zip(it, loss) if 10 <= i_ <= 1000]
        first100 = [l for i_, l in zip(it, loss) if i_ <= 100]
        row = dict(run=name, arm=m.group(1), mode=m.group(2), seed=int(m.group(3)), host=root,
                   last_iter=it[-1] if it else None, done="EXIT 0" in txt,
                   mean_loss_10_1000=sum(early) / len(early) if early else None,
                   max_loss_first100=max(first100) if first100 else None, blowup_onset=onset,
                   memory_onset=next((i_ for i_, r in zip(it, rec) if r > 0.5), None),
                   last_loss=loss[-1] if loss else None, last_recall=rec[-1] if rec else None,
                   max_window=max(win) if win else None)
        hj = os.path.join(d, "heldout.json")
        if os.path.exists(hj):
            h = json.load(open(hj))
            row["heldout_iter"] = h["iteration"]
            for p in ("observed", "strict", "no_fast"):
                row[f"{p}_recall"] = h[f"heldout_{p}/recall_acc"]
                row[f"{p}_loss"] = h[f"heldout_{p}/loss"]
        print(json.dumps(row))
