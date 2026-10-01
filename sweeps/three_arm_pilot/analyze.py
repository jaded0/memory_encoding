import glob, json, os, re, statistics, sys

ROOT = os.path.expanduser('~/lowrank_deep/three_arm/runs')


def parse(path):
    rows, cur = [], None
    for line in open(path):
        m = re.match(r'--- Interval metrics \(ending @ iter (\d+)', line)
        if m:
            cur = {'iter': int(m.group(1))}
            rows.append(cur)
            continue
        m = re.match(r'\s+(loss|recall_acc|recall_loss|iters_per_sec): (\S+)', line)
        if m and cur is not None:
            cur[m.group(1)] = float(m.group(2))
    return [r for r in rows if 'recall_acc' in r]


def first(rows, thr):
    for r in rows:
        if r['recall_acc'] >= thr:
            return r['iter']
    return None


out = []
for d in sorted(glob.glob(f'{ROOT}/[ABC]_a*_s*')):
    name = os.path.basename(d)
    path = f'{d}/train.log'
    if not os.path.exists(path):
        continue
    rows = parse(path)
    if not rows:
        continue
    txt = open(path).read()
    flag = ''
    if 'Non-finite loss' in txt:
        flag = 'NaN'
    elif 'Early stopping: Loss' in txt:
        flag = 'loss>5 stop'
    losses = [r['loss'] for r in rows if 'loss' in r]
    if losses and max(losses) > 50:
        flag += ' loss>50 seen'
    speeds = [r['iters_per_sec'] for r in rows[2:] if 'iters_per_sec' in r]
    best = max(r['recall_acc'] for r in rows)
    ho = {}
    try:
        h = json.load(open(f'{d}/heldout.json'))
        ho = {k.split('/')[0].replace('heldout_', ''): v for k, v in h.items() if k.endswith('/recall_acc')}
    except Exception:
        pass
    out.append((name, rows[-1]['iter'], first(rows, 0.9), first(rows, 0.99), rows[-1]['recall_acc'], best,
                rows[-1].get('loss'), flag.strip(), statistics.median(speeds) if speeds else None, ho))
print('| run | iters run | it to 0.9 | it to 0.99 | final recall | best | final loss | flags | it/s (median) | heldout strict / no_fast / observed |')
print('|---|---|---|---|---|---|---|---|---|---|')
for n, it, a, b, f, best, l, flag, sp, ho in out:
    hs = ' / '.join(f"{ho[k]:.3f}" if k in ho else '-' for k in ('strict', 'no_fast', 'observed'))
    print(f"| {n} | {it} | {a or '-'} | {b or '-'} | {f:.3f} | {best:.3f} | {l if l is None else round(l, 3)} | {flag or '-'} | "
          f"{'-' if sp is None else round(sp, 1)} | {hs} |")
