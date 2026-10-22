"""Median iters_per_sec over the intervals after the first (compile/warm-up) of each timing log."""
import glob
import os
import re
import statistics
import sys

out = sys.argv[1]
rows = []
for path in sorted(glob.glob(os.path.join(out, "timing_*.log"))):
    rates = [float(x) for x in re.findall(r"iters_per_sec: ([0-9.]+)", open(path).read())]
    name = os.path.basename(path)[len("timing_"):-len(".log")]
    if len(rates) >= 2:
        rows.append((name, statistics.median(rates[1:]), len(rates)))
    else:
        rows.append((name, float("nan"), len(rates)))
for name, rate, n in rows:
    print(f"{name:40s} {rate:8.2f} it/s  ({1000 / rate if rate == rate else float('nan'):7.1f} ms/it, {n} intervals)")
