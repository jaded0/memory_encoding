"""Parse arm logs, classify onset (amended section-12 definition), summarise. usage: analyze.py RUNS_DIR OUT_JSON [arm ...]"""
import json
import os
import re
import sys

BLOCK = re.compile(r"--- Interval metrics \(ending @ iter (\d+)")
KV = re.compile(r"^\s+([\w/]+): (-?[\d.]+(?:e[+-]?\d+)?|nan|inf)\s*$")


def parse_log(path):
    rows, cur = [], None
    for line in open(path, errors='replace'):
        m = BLOCK.search(line)
        if m:
            cur = {'iter': int(m.group(1))}
            rows.append(cur)
            continue
        if cur is not None:
            kv = KV.match(line)
            if kv:
                cur[kv.group(1)] = float(kv.group(2))
            elif line.startswith('---') or line.startswith('Checkpoint'):
                cur = None
    # restarts re-print iterations: keep the last occurrence
    by = {r['iter']: r for r in rows}
    return [by[k] for k in sorted(by)]


def onset(rows, after=30000, thr=5.0, run=5, single=20.0):
    """First window start (iteration at the window's start = previous print) of >= `run` consecutive windows
    with loss > thr, or one window > single, restricted to windows ending after `after`. None if none."""
    its = [r['iter'] for r in rows]
    loss = [r['loss'] for r in rows]
    for i, (it, l) in enumerate(zip(its, loss)):
        if it <= after:
            continue
        start = its[i - 1] if i > 0 else it
        if l > single:
            return start
        if all(x > thr for x in loss[i:i + run]) and len(loss[i:i + run]) == run:
            return start
    return None


if __name__ == '__main__':
    runs, out = sys.argv[1], sys.argv[2]
    arms = sys.argv[3:] or sorted(d for d in os.listdir(runs) if os.path.exists(os.path.join(runs, d, 'train.log')))
    res = {}
    for a in arms:
        rows = parse_log(os.path.join(runs, a, 'train.log'))
        late = [r['loss'] for r in rows if r['iter'] > 160000]
        res[a] = {'rows': rows, 'onset': onset(rows), 'n': len(rows), 'last_iter': rows[-1]['iter'] if rows else None,
                  'median_after_160k': sorted(late)[len(late) // 2] if late else None, 'max': max(late) if late else None}
        print(a, 'last', res[a]['last_iter'], 'onset', res[a]['onset'], 'median>160k', res[a]['median_after_160k'], 'max', res[a]['max'])
    json.dump(res, open(out, 'w'))
