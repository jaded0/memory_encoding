#!/usr/bin/env python3
"""Infer sampled stability edges from cells_st.json."""
import argparse, json
from collections import defaultdict
from pathlib import Path

def infer(cells):
    groups = defaultdict(list)
    for name, c in cells.items(): groups[(c["arm"], c["stage"])].append((name, c))
    out = []
    for (arm, stage), rows in sorted(groups.items()):
        incomplete = sorted(n for n, c in rows if c["incomplete"])
        by_m = defaultdict(list)
        for _, c in rows:
            if not c["incomplete"] and c["class"]: by_m[c["m"]].append(c)
        unstable = [m for m, cs in by_m.items() if any(c["class"] != "S" or c["A"] for c in cs)]
        edge = min(unstable) if unstable else None
        stable = [m for m, cs in by_m.items() if all(c["class"] == "S" and not c["A"] for c in cs) and (edge is None or m < edge)]
        lo = max(stable) if stable else None
        status = ("no complete cells" if not by_m else
                  f"stable up to max m={max(by_m):g}" if edge is None else
                  f"edge m*={edge:g}; bracket {lo:g}–{edge:g}" if lo is not None else
                  f"edge m*={edge:g}; no all-stable lower sample")
        out.append(dict(arm=arm, stage=stage, edge=edge, bracket_low=lo,
                        max_complete_m=max(by_m) if by_m else None,
                        status=status, incomplete=incomplete))
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cells", default="cells_st.json")
    ap.add_argument("--json", default="edges_st.json"); ap.add_argument("--table", default="edges_st_table.md")
    a = ap.parse_args(); rows = infer(json.loads(Path(a.cells).read_text()))
    Path(a.json).write_text(json.dumps(rows, indent=2) + "\n")
    lines = ["# Stabilizer-arm sampled edges", "", "Incomplete cells are excluded from inference and listed explicitly.", "",
             "| arm | stage | result | incomplete cells |", "|---|---:|---|---|"]
    lines += [f"| {r['arm']} | {r['stage']} | {r['status']} | {', '.join(r['incomplete']) or '—'} |" for r in rows]
    Path(a.table).write_text("\n".join(lines) + "\n")
    print(f"wrote {a.json} and {a.table}")

if __name__ == "__main__": main()
