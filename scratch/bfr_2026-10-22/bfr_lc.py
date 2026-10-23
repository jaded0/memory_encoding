"""Batch BFR LC/SIX scoring on ORC (agent A, 2026-10-22), adapted from f8ln_2026-10-19/qsum.py (Batch Q).
Usage (on ORC): python bfr_lc.py JOBID [TASK=RESUMEJOB[,RESUMEJOB2...] ...] > bfr_lc.jsonl
  e.g. python bfr_lc.py 14613310 1=14648594 4=14654958   (resume logs of timed-out tasks; later jobs override earlier
  ones from their first printed iteration on, since a resume restarts from the last 25k checkpoint and re-prints that stretch)
Per run: 500-it train loss trajectory from the slurm log, onset = first of five adjacent 500-it windows > 5 (any time; early if <= 30k),
first window > 1e3; final_eval.json (heldout.py, 128 batches) observed/strict/no_fast; for 3-layer runs, from checkpoint_00150000 and
_00600000: the readout's squared-norm share in range(B_2) (B_2 = linear_layers.2.feedback_weights) and trunk-2 slow top SV."""
import json, math, os, re, sys
import torch
torch.set_num_threads(4)
base = os.path.expanduser('~/bfr_2026-10-22')
job = sys.argv[1]
resumes = {int(a.split('=')[0]): a.split('=')[1].split(',') for a in sys.argv[2:]}
seeds5 = (620101, 620203, 620307, 620411, 620509)
tasks = [(t, f'BFR-LC-{m}-s{s}', 3) for t, (m, s) in enumerate([(m, s) for m in ('ln', 'absrun') for s in seeds5])]
tasks += [(10 + i, f'BFR-SIX-{m}-s{s}', 6) for i, (m, s) in enumerate([(m, s) for m in ('ln', 'absrun') for s in seeds5[:3]])]


def losses(p):
    L = []; it = None
    for ln in open(p, errors='replace'):
        m = re.match(r'^(\d+) [\d.]+% \(', ln) or re.search(r'Interval metrics \(ending @ iter (\d+)', ln)
        if m: it = int(m.group(1)); continue
        m = re.match(r'^\s+loss:\s+(\S+)', ln)
        if m and it is not None:
            try: v = float(m.group(1))
            except ValueError: v = float('nan')
            L.append((it, v)); it = None
    return L


def onset(L, after=0):
    for i in range(len(L) - 4):
        w = L[i:i + 5]
        if w[0][0] >= after and all((not math.isfinite(v)) or v > 5 for _, v in w): return w[0][0]


def al(path):
    sd = torch.load(path, map_location='cpu', weights_only=False)['model_state_dict']
    R = sd['i2o.per_sample_weights'].double().mean(0)
    B = sd['linear_layers.2.feedback_weights']; Q, _ = torch.linalg.qr(B.T.double())
    W = sd['linear_layers.2.per_sample_weights'].double().mean(0); m = sd['linear_layers.2.ephemeral_mask']
    return float((R @ Q).norm() ** 2 / R.norm() ** 2), float(torch.linalg.matrix_norm(torch.where(m, torch.zeros_like(W), W), 2))


for t, n, layers in tasks:
    log = f'{base}/logs/bfr_{job}_{t}.out'
    if not os.path.exists(log): continue
    L = losses(log); logs = [log]; resumed_from = []
    for rj in resumes.get(t, []):
        rlog = f'{base}/logs/bfr_{rj}_{t}.out'
        if not os.path.exists(rlog): continue
        R = losses(rlog)
        if R: L = [x for x in L if x[0] < R[0][0]] + R; resumed_from.append(R[0][0])
        logs.append(rlog)
    d = dict(L)
    row = dict(run=n, task=t, prints=len(L), last_iter=L[-1][0] if L else None, onset=onset(L), late_onset=onset(L, 30001),
               first_gt1e3=next((i for i, v in L if not math.isfinite(v) or v > 1e3), None),
               early_windows_gt5=sum(1 for i, v in L if i <= 30000 and (not math.isfinite(v) or v > 5)),
               maxloss=max((v for _, v in L if math.isfinite(v)), default=None),
               loss_at={k: d.get(k) for k in (5000, 30000, 100000, 150000, 200000, 300000, 400000, 500000, 600000) if k in d},
               train_exit=[e for lg in logs for e in re.findall(r'TRAIN EXIT (\d+)', open(lg, errors='replace').read())],
               logs=[os.path.basename(lg) for lg in logs], resumed_from=resumed_from)
    fe = f'{base}/runs/{n}/final_eval.json'
    if os.path.exists(fe):
        h = json.load(open(fe)); row['eval_iter'] = h['iteration']
        for p in ('observed', 'strict', 'no_fast'):
            row[f'{p}_recall'] = h[f'heldout_{p}/recall_acc']; row[f'{p}_loss'] = h[f'heldout_{p}/loss']
    if layers == 3:
        for k in (150000, 600000):
            p = f'{base}/runs/{n}/checkpoint_{k:08d}.pth'
            if os.path.exists(p):
                row[f'share_{k // 1000}k'], row[f'sv_{k // 1000}k'] = al(p)
        if 'sv_150k' in row and 'sv_600k' in row: row['sv_ratio'] = row['sv_600k'] / row['sv_150k']
    print(json.dumps(row), flush=True)
