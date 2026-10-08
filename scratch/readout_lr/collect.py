"""Interval metrics of the readout-only test arms and the global slow-lr x0.3 lineage, as CSV on stdout."""
import re
INTERVAL = re.compile(r"Interval metrics \(ending @ iter (\d+)")
METRIC = re.compile(r"  ([\w/]+): ([-+\w.]+)")
KEYS = ["loss", "recall_acc", "recall_acc_lag_3", "trace/trunk_act_norm_last", "trace/fast_norm_last",
        "trace/slow_total_delta", "trace/max_logit_max"]
H = "/home/jaden"
RUNS = {
    "CTRL": [f"{H}/readout_lr_2026-10-07/runs/CTRL/train.log"],
    "RO3": [f"{H}/readout_lr_2026-10-07/runs/RO3/train.log"],
    "TR3": [f"{H}/readout_lr_2026-10-07/runs/TR3/train.log"],
    "GLOBAL3": [f"{H}/overnight_2026-10-01/runs/{r}/train.log" for r in ("INT_R2", "INT2_R2", "INT3_R2")],
    "INT_R0": [f"{H}/overnight_2026-10-01/runs/INT_R0/train.log"],
}
print("run,iter," + ",".join(KEYS))
for name, paths in RUNS.items():
    rows = {}
    for path in paths:
        cur = None
        for line in open(path, errors="replace"):
            m = INTERVAL.search(line)
            if m:
                cur = rows.setdefault(int(m.group(1)), {})
                continue
            m = METRIC.match(line)
            if cur is not None and m:
                try:
                    cur[m.group(1)] = float(m.group(2))
                except ValueError:
                    pass
    for it in sorted(rows):
        print(f"{name},{it}," + ",".join(str(rows[it].get(k, "")) for k in KEYS))
