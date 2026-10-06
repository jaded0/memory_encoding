#!/usr/bin/env python3
"""Collect and classify stabilizer-arm alpha-kick cells."""
import argparse, glob, json, math, os, re
from pathlib import Path
from statistics import median

METRICS = ("loss", "recall_acc", "trunk_act_norm_last", "loop_gain_median", "max_logit_max")
TRACE = {k: "trace/" + k for k in METRICS[2:]}


def parse_log(path):
    """The interval parser from scratch/two_modes/collect.py, retaining all metrics."""
    text = re.sub(r"\x1b\[[0-9;]*m", "", path.read_text(errors="replace"))
    blocks = re.split(r"--- Interval metrics \(ending @ iter (\d+)[^\n]*\n", text)[1:]
    out = {}
    for i in range(0, len(blocks) - 1, 2):
        vals = {}
        for key, value in re.findall(
            r"^\s*([\w/]+): (-?[\d.eE+-]+|nan|inf)\s*$",
            blocks[i + 1].split("-----")[0], re.M):
            try: vals[key] = float(value)
            except ValueError: pass
        out[int(blocks[i])] = vals
    return text, out


def finite(xs):
    return [x for x in xs if x is not None and math.isfinite(x)]


def med(xs):
    xs = finite(xs)
    return float(median(xs)) if xs else None


def runs(flag, n=5):
    return any(all(flag[i:i + n]) for i in range(len(flag) - n + 1))


def loss_class(its, loss, kick):
    """Classification from two_modes/classify.py, mapped to S/T/N/R."""
    pairs = [(i, x) for i, x in zip(its, loss)
             if i > kick and x is not None and math.isfinite(x)]
    if not pairs:
        return dict(cls=None, peak=None, onset=None, final=None)
    ii, ll = map(list, zip(*pairs)); n = len(ll)
    peak = max(ll); ipk = ll.index(peak); tail = ll[int(n * .8):]
    first6 = next((j for j, x in enumerate(ll) if x > 6), None)
    onset = next((ii[j] - kick for j in range(n)
                  if ll[j] > 20 or (j + 5 <= n and all(x > 5 for x in ll[j:j + 5]))), None)
    if peak <= 6:
        cls = "S"
    elif all(x < 4 for x in tail):
        cls = "T"
    elif peak > 100 and not any(x < 5 for x in ll[ipk:]):
        cls = "R"
    elif first6 is not None and ii[first6] - kick > 10000 and onset is not None and not all(x < 4 for x in tail):
        cls = "N"                         # delayed collapse
    elif ((onset is not None and first6 is not None and ii[first6] - kick <= 10000
           and min(ll[ipk:]) >= 4) or med(tail) > 6):
        cls = "N"                         # non-recovering
    else:
        cls = "T"                         # includes classify.py's transient?
    return dict(cls=cls, peak=float(peak), onset=onset, final=float(ll[-1]))


classify = loss_class


def read_specs(paths):
    specs = {}
    for path in paths:
        for raw in Path(path).read_text().splitlines():
            p = raw.split()
            if not p or p[0].startswith("#") or len(p) < 6: continue
            specs[p[0]] = dict(arm=p[1], stage=int(p[2]), n_iters=int(p[3]),
                               m=float(p[4]), reseed=int(p[6]) if len(p) > 6 else None)
    return specs


def metadata(name, text, spec):
    head = re.search(r"=== cell \S+ .*? m=(\S+) arm=(\S+) stage=(\d+)", text)
    ck = re.search(r"resume_checkpoint \S*checkpoint_(\d+)\.pth", text)
    ni = re.search(r"--n_iters (\d+)", text)
    pl = re.search(r"--plasticity (\S+)", text)
    rs = re.search(r"--resume_reseed (\d+)", text)
    suffix = re.search(r"_r(\d+)$", name)
    arm = spec.get("arm") or (head.group(2) if head else name.split("_")[0])
    stage = spec.get("stage") or (int(head.group(3)) if head else int(ck.group(1)))
    m = spec.get("m")
    if m is None: m = float(head.group(1)) if head else float(pl.group(1)) / 1e4
    return dict(arm=arm, stage=stage, m=m,
                n_iters=spec.get("n_iters") or (int(ni.group(1)) if ni else None),
                reseed=spec.get("reseed") if spec.get("reseed") is not None else
                       (int(rs.group(1)) if rs else (int(suffix.group(1)) if suffix else None)))


def recall_key(rows):
    keys = {k for row in rows.values() for k in row}
    for key in ("recall_acc", "recall/acc", "metrics/recall_acc"):
        if key in keys: return key
    return next((k for k in sorted(keys) if "recall" in k.lower() and "acc" in k.lower()), "recall_acc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=os.path.expanduser("~/stab_margin_2026-10-06/logs"))
    ap.add_argument("--cells", nargs="*", default=sorted(glob.glob("cells*.txt")))
    ap.add_argument("--json", default="cells_st.json")
    ap.add_argument("--table", default="cells_st_table.md")
    args = ap.parse_args(); specs = read_specs(args.cells); cells = {}
    for path_s in sorted(glob.glob(os.path.join(args.logs, "*", "train.log"))):
        path = Path(path_s); name = path.parent.name; text, rows = parse_log(path)
        meta = metadata(name, text, specs.get(name, {})); rkey = recall_key(rows)
        its = sorted(i for i in rows if i > meta["stage"])
        series = {"iteration": its}
        for key in METRICS:
            source = rkey if key == "recall_acc" else TRACE.get(key, key)
            series[key] = [rows[i].get(source) for i in its]
        horizon = meta["n_iters"] - meta["stage"] if meta["n_iters"] else None
        progress = (its[-1] - meta["stage"]) if its else 0
        done = "=== exit 0" in text
        info = loss_class(its, series["loss"], meta["stage"])
        cells[name] = dict(**{k: meta[k] for k in ("arm", "stage", "m", "reseed")},
            horizon=horizon, done_fraction=min(1., max(0., progress / horizon)) if horizon else None,
            its=series.pop("iteration"), series=series, recall_metric=rkey,
            **{"class": info["cls"], "peak_loss": info["peak"],
               "onset_iter": info["onset"], "final_loss": info["final"]}, incomplete=not done)

    # Activation baseline: true pre-kick last five; otherwise m=1 same arm/stage first three; otherwise own first three.
    for name, cell in cells.items():
        own = cell["series"]["trunk_act_norm_last"]
        # Current cell logs normally start after the kick; this supports concatenated logs if encountered.
        text, all_rows = parse_log(Path(args.logs) / name / "train.log")
        pre = finite([all_rows[i].get(TRACE["trunk_act_norm_last"])
                      for i in sorted(all_rows) if i <= cell["stage"]])[-5:]
        source = "cell pre-kick last 5"
        baseline = med(pre) if len(pre) == 5 else None
        if baseline is None:
            candidates = [(n, c) for n, c in cells.items() if c["arm"] == cell["arm"]
                          and c["stage"] == cell["stage"] and c["m"] == 1]
            candidates.sort(key=lambda z: (z[1]["reseed"] is not None, z[1]["reseed"] or -1, z[0]))
            proxy = finite(candidates[0][1]["series"]["trunk_act_norm_last"])[:3] if candidates else []
            if len(proxy) == 3:
                base_name, base_cell = candidates[0]
                baseline = med(proxy)
                source = f"{base_name} first 3 post-kick (m=1 proxy; parent trace unavailable)"
            else:
                own_first = finite(own)[:3]
                baseline = med(own_first) if len(own_first) == 3 else None
                source = f"{name} first 3 post-kick (fallback; parent trace unavailable)"
        act_hit = runs([x is not None and baseline is not None and x > 10 * baseline for x in own])
        first_recall = finite(cell["series"]["recall_acc"])[:3]
        rb = med(first_recall) if len(first_recall) == 3 else None
        recall_hit = runs([x is not None and rb is not None and x < .5 * rb
                           for x in cell["series"]["recall_acc"]])
        cell.update(A=act_hit or recall_hit, activation_baseline=baseline,
                    activation_baseline_source=source, recall_baseline=rb,
                    activation_runaway_reason=("act" if act_hit else "recall" if recall_hit else None))

    Path(args.json).write_text(json.dumps(cells, indent=2, allow_nan=False) + "\n")
    def fmt(x, digits=4): return "—" if x is None else f"{x:.{digits}g}"
    def onset(x): return "—" if x is None else str(x)
    lines = ["# Stabilizer-arm alpha-kick cells", "",
             "Onset is iterations after the kick. `act/base` uses the baseline source shown in the last column.", "",
             "| cell | arm | stage | m | reseed | class | A | peak | onset Δiter | final loss | final recall | final act/base | incomplete | activation baseline source |",
             "|---|---:|---:|---:|---:|:---:|:---:|---:|---:|---:|---:|---:|:---:|---|"]
    for name, c in sorted(cells.items(), key=lambda z: (z[1]["arm"], z[1]["stage"], z[1]["m"], z[1]["reseed"] or -1)):
        last = lambda k: next((x for x in reversed(c["series"][k]) if x is not None), None)
        ratio = last("trunk_act_norm_last") / c["activation_baseline"] if last("trunk_act_norm_last") is not None and c["activation_baseline"] else None
        lines.append(f"| {name} | {c['arm']} | {c['stage']} | {c['m']:g} | {c['reseed'] if c['reseed'] is not None else 'base'} | {c['class'] or '—'} | {'yes' if c['A'] else 'no'} | {fmt(c['peak_loss'])} | {onset(c['onset_iter'])} | {fmt(c['final_loss'])} | {fmt(last('recall_acc'))} | {fmt(ratio)} | {'yes' if c['incomplete'] else 'no'} | {c['activation_baseline_source']} |")
    Path(args.table).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.json} and {args.table}: {len(cells)} cells, {sum(not c['incomplete'] for c in cells.values())} complete")


if __name__ == "__main__": main()
