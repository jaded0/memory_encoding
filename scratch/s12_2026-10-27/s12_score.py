"""Batch S12F scoring (S12 follow-ups 1-3, prep 2026-10-27; PREREG "Batch S12F"). Run on ORC (conda hebby) from anywhere:
  python s12_score.py lc   > s12_lc.jsonl     ln + DC2 LC (s12_lndc2_lc.sbatch) and absrun_floor LC (s12_absfloor_lc.sbatch)
  python s12_score.py kr   > s12_kr.jsonl     absrun_floor KR (s12_absfloor_kr.sbatch)
  python s12_score.py null [DIR ...] > s12_null.jsonl   ln_null_eval.py JSONs (default $BASE/null; add manticore's ~/s12_null copy)
Same definitions as Batch BFR / BFR2 (bfr_lc.py, bfr2_score.py, kr_collect.py), whose outputs these rows are compared with:
- LC train loss from the slurm logs (500-it prints), every log of a task spliced in job order (a resubmission replaces the earlier log
  from its first printed iteration on). onset = first of five adjacent 500-it windows > 5; useful healthy = no onset, final loss < 5
  and final strict >= 0.30. share = the readout's squared-norm share in range(B_2), sv = trunk-2 slow top singular value, at 150k,
  300k, 600k, 900k, 1.2M where those checkpoints exist; last10 = mean of the last ten 500-it windows of the run.
- KR: train.log (100-it prints) averaged into 500-it windows, the same onset rule; memory onset = first print with recall_acc > 0.5.
- ratios: bfr_ratio.py JSONs in $BASE/ratios (eff/slow and raw/slow medians per trunk layer).
BASE defaults to ~/s12_2026-10-27 (env S12_BASE)."""
import glob
import json
import math
import os
import re
import sys

base = os.path.expanduser(os.environ.get('S12_BASE', '~/s12_2026-10-27'))
LC_TASKS = ([('s12_lndc2', t, f'S12-LC-lnDC2-s{s}') for t, s in enumerate((620101, 620203, 620307, 620411, 620509))]
            + [('s12_afl', t, f'S12-LC-absrunC{c}-s{s}') for t, (c, s) in
               enumerate([(c, s) for c in (10, 100) for s in (620101, 620203, 620307)])])
KR_RUNS = [f'kr_plain_absrunC{c}_s{s}' for c in (10, 100) for s in (2718, 3141, 1618)]


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


def splice(paths):
    L = []
    for p in paths:
        R = losses(p)
        if R: L = [x for x in L if x[0] < R[0][0]] + R
    return L


def onset(L, after=0):
    for i in range(len(L) - 4):
        w = L[i:i + 5]
        if w[0][0] >= after and all((not math.isfinite(v)) or v > 5 for _, v in w): return w[0][0]


def al(path):
    import torch
    torch.set_num_threads(4)
    sd = torch.load(path, map_location='cpu', weights_only=False)['model_state_dict']
    R = sd['i2o.per_sample_weights'].double().mean(0)
    B = sd['linear_layers.2.feedback_weights']; Q, _ = torch.linalg.qr(B.T.double())
    W = sd['linear_layers.2.per_sample_weights'].double().mean(0); m = sd['linear_layers.2.ephemeral_mask']
    return float((R @ Q).norm() ** 2 / R.norm() ** 2), float(torch.linalg.matrix_norm(torch.where(m, torch.zeros_like(W), W), 2))


def evals(row, path, tag):
    if os.path.exists(path):
        h = json.load(open(path)); row[f'{tag}iter'] = h['iteration']
        for p in ('observed', 'strict', 'no_fast', 'swap', 'swap_raw', 'noise'):
            if f'heldout_{p}/recall_acc' in h:
                row[f'{tag}{p}_recall'] = h[f'heldout_{p}/recall_acc']; row[f'{tag}{p}_loss'] = h[f'heldout_{p}/loss']


def ratios(row, name):
    for f in sorted(glob.glob(f'{base}/ratios/ratio_{name}_*k.json')):
        k = f.rsplit('_', 1)[1][:-5]
        for layer, v in json.load(open(f))['layers'].items():
            row[f'effslow_L{layer}_{k}'] = v['eff_over_slow_med']
            row[f'effslow_rec_L{layer}_{k}'] = v['eff_over_slow_recall_med']


def score_lc():
    for job, t, n in LC_TASKS:
        logs = sorted(glob.glob(f'{base}/logs/{job}_*_{t}.out'), key=lambda p: int(re.search(rf'{job}_(\d+)_', p).group(1)))
        if not logs: continue
        L = splice(logs); d = dict(L)
        fin = [v for _, v in L[-10:]]
        row = dict(run=n, task=t, logs=[os.path.basename(p) for p in logs], last_iter=L[-1][0] if L else None,
                   onset=onset(L), late_onset=onset(L, 30001),
                   first_gt1e3=next((i for i, v in L if not math.isfinite(v) or v > 1e3), None),
                   early_windows_gt5=sum(1 for i, v in L if i <= 30000 and (not math.isfinite(v) or v > 5)),
                   maxloss=max((v for _, v in L if math.isfinite(v)), default=None),
                   last10=sum(fin) / len(fin) if fin else None,
                   loss_at={k: d.get(k) for k in (5000, 30000, 150000, 300000, 600000, 900000, 1200000) if k in d},
                   train_exit=[e for p in logs for e in re.findall(r'TRAIN EXIT (\d+)', open(p, errors='replace').read())])
        evals(row, f'{base}/runs/{n}/final_eval.json', '')
        evals(row, f'{base}/runs/{n}/null_eval.json', 'null_')
        for k in (150000, 300000, 600000, 900000, 1200000):
            p = f'{base}/runs/{n}/checkpoint_{k:08d}.pth'
            if os.path.exists(p): row[f'share_{k // 1000}k'], row[f'sv_{k // 1000}k'] = al(p)
        last = max((k for k in (300000, 600000, 1200000) if f'sv_{k // 1000}k' in row), default=None)
        if last and 'sv_150k' in row: row['sv_ratio_last_150k'] = row[f'sv_{last // 1000}k'] / row['sv_150k']
        if row.get('strict_recall') is not None:
            final = L[-1][1] if L else float('nan')
            row['useful_healthy'] = bool(row['onset'] is None and math.isfinite(final) and final < 5 and row['strict_recall'] >= 0.30)
        ratios(row, n)
        print(json.dumps(row), flush=True)


def score_kr():
    for n in KR_RUNS:
        d = f'{base}/runs_kr/{n}'
        try:
            txt = open(f'{d}/train.log', errors='replace').read()
        except OSError:
            continue
        it, loss, rec = [], [], []
        for mm in re.finditer(r'ending @ iter (\d+).*?\n  loss: ([^\n]+)\n.*?recall_acc: ([^\n]+)\n', txt, re.S):
            it.append(int(mm.group(1)))
            try: loss.append(float(mm.group(2)))
            except ValueError: loss.append(float('nan'))
            rec.append(float(mm.group(3)))
        L = list(zip(it, loss))
        win = [(L[i][0], sum(v for _, v in L[i:i + 5]) / 5) for i in range(0, len(L) - len(L) % 5, 5)]
        m = re.match(r'kr_plain_absrunC(\d+)_s(\d+)$', n)
        tail = [v for i_, v in L if i_ > (it[-1] - 5000 if it else 0)]
        row = dict(run=n, floor=int(m.group(1)), seed=int(m.group(2)), last_iter=it[-1] if it else None,
                   done='EXIT 0' in txt, blowup_onset=onset(win),
                   max_window=max((v for _, v in win if math.isfinite(v)), default=None),
                   memory_onset=next((i_ for i_, r in zip(it, rec) if r > 0.5), None),
                   mean_loss_10_1000=(lambda e: sum(e) / len(e) if e else None)([v for i_, v in L if 10 <= i_ <= 1000]),
                   mean_last_5k=sum(tail) / len(tail) if tail else None,
                   last_loss=loss[-1] if loss else None, last_recall=rec[-1] if rec else None)
        evals(row, f'{d}/heldout.json', '')
        ratios(row, n)
        print(json.dumps(row), flush=True)


def score_null(dirs):
    for root in dirs or [f'{base}/null']:
        for f in sorted(glob.glob(os.path.join(os.path.expanduser(root), 'null_*.json'))):
            h = json.load(open(f))
            m = re.match(r'null_(.+)_(\d+)k\.json$', os.path.basename(f))
            row = dict(run=m.group(1), iter_k=int(m.group(2)), iteration=h['iteration'], episodes=h['episodes'],
                       dc2=bool(h.get('decouple_fast_feedback')), file=f)
            for p in ('strict', 'no_fast', 'swap', 'swap_raw', 'noise'):
                for k in ('recall_acc', 'loss', 'recall_loss', 'other_loss', 'first_answer_acc'):
                    if f'heldout_{p}/{k}' in h: row[f'{p}_{k}'] = h[f'heldout_{p}/{k}']
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'null':
        score_null(sys.argv[2:])
    else:
        {'lc': score_lc, 'kr': score_kr}[mode]()
