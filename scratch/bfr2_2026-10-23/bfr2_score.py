"""Batch BFR2 scoring on ORC (agent A round 2, 2026-10-23; PREREG_fastlayers.md "Batch BFR2"). Usage (on ORC, conda hebby):
  python bfr2_score.py lc > bfr2_lc.jsonl     python bfr2_score.py kr > bfr2_kr.jsonl
lc: the 5 ln runs extended 600k -> 1.2M in runs2/. Train log = round-1 logs (bfr_14613310_t, spliced with the resumes of tasks 1, 4) followed by every
  logs/bfr2_<job>_t.out in job order (each later log replaces earlier ones from its first printed iteration on). Onset / useful healthy as Batch BFR;
  share in range(B_2) and trunk-2 slow SV at 600k (round-1 checkpoint), 900k and 1.2M (runs2); loss_last10_600k / _1200k = mean of the last ten
  500-it windows ending at 600k / 1.2M (D5); round-1 strict from runs/<name>/final_eval.json (D2).
kr: plain/DC2 ln extended 30k -> 100k in runs2_kr/ (train.log there holds 30k-100k; the 25-30k window comes from round 1: ORC runs_kr/ for DC2,
  manticore's train.log copied to runs_kr/<name>/train_r1.log for plain). mean A-B = mean printed loss over iterations in (A, B]."""
import glob, json, math, os, re, sys
base = os.path.expanduser('~/bfr_2026-10-22')
seeds5 = (620101, 620203, 620307, 620411, 620509)
R1 = {t: [f'bfr_14613310_{t}.out'] for t in range(5)}; R1[1].append('bfr_14648594_1.out'); R1[4].append('bfr_14654958_4.out')


def splice(paths):
    L = []
    for p in paths:
        R = losses(p)
        if R: L = [x for x in L if x[0] < R[0][0]] + R
    return L


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
    """First of five adjacent 500-it windows > 5 (LC prints every 500; KR's prints are averaged into 500-it windows first)."""
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


def mean_in(L, lo, hi):
    v = [x for i, x in L if lo < i <= hi]
    return sum(v) / len(v) if v else None


def score_lc():
    for t, s in enumerate(seeds5):
        n = f'BFR-LC-ln-s{s}'
        r2 = sorted((p for p in glob.glob(f'{base}/logs/bfr2_*_{t}.out') if re.search(r'bfr2_\d+_', p)),
                    key=lambda p: int(re.search(r'bfr2_(\d+)_', p).group(1)))
        L = splice([f'{base}/logs/{x}' for x in R1[t]] + r2)
        L2 = [x for x in L if x[0] > 600000]
        row = dict(run=n, task=t, logs2=[os.path.basename(p) for p in r2], last_iter=L[-1][0] if L else None,
                   onset=onset(L), onset_after_600k=onset(L, 600001),
                   maxloss_600k_1200k=max((v for _, v in L2 if math.isfinite(v)), default=None),
                   nonfinite_after_600k=sum(1 for _, v in L2 if not math.isfinite(v)),
                   loss_last10_600k=mean_in(L, 595000, 600000), loss_last10_1200k=mean_in(L, 1195000, 1200000),
                   loss_at={k: dict(L).get(k) for k in (600000, 750000, 900000, 1050000, 1200000)},
                   train_exit=[e for p in r2 for e in re.findall(r'TRAIN EXIT (\d+)', open(p, errors='replace').read())])
        for tag, fe in (('r1', f'{base}/runs/{n}/final_eval.json'), ('r2', f'{base}/runs2/{n}/final_eval.json')):
            if os.path.exists(fe):
                h = json.load(open(fe)); row[f'{tag}_eval_iter'] = h['iteration']
                for p in ('observed', 'strict', 'no_fast'):
                    row[f'{tag}_{p}_recall'] = h[f'heldout_{p}/recall_acc']; row[f'{tag}_{p}_loss'] = h[f'heldout_{p}/loss']
        for k, p in ((600000, f'{base}/runs/{n}/checkpoint_00600000.pth'), (900000, f'{base}/runs2/{n}/checkpoint_00900000.pth'),
                     (1200000, f'{base}/runs2/{n}/checkpoint_01200000.pth')):
            if os.path.exists(p): row[f'share_{k // 1000}k'], row[f'sv_{k // 1000}k'] = al(p)
        if 'sv_600k' in row and 'sv_1200k' in row: row['sv_ratio_1200_600'] = row['sv_1200k'] / row['sv_600k']
        if row.get('r2_strict_recall') is not None:
            row['useful_healthy'] = (row['onset'] is None and row['nonfinite_after_600k'] == 0 and (row['loss_at'].get(1200000) or 9) < 5
                                     and row['r2_strict_recall'] >= 0.30)
        print(json.dumps(row), flush=True)


def kr_log(p):
    txt = open(p, errors='replace').read()
    out = []
    for a, b in re.findall(r'ending @ iter (\d+).*?\n  loss: ([^\n]+)\n', txt, re.S):
        try: out.append((int(a), float(b)))
        except ValueError: out.append((int(a), float('nan')))
    return out


def score_kr():
    for arm in ('plain', 'DC2'):
        for s in (2718, 3141, 1618, 1414, 5772):
            n = f'kr_{arm}_ln_s{s}'
            r1p = f'{base}/runs_kr/{n}/' + ('train_r1.log' if arm == 'plain' else 'train.log')
            L1 = kr_log(r1p) if os.path.exists(r1p) else []
            p2 = f'{base}/runs2_kr/{n}/train.log'
            L2 = kr_log(p2) if os.path.exists(p2) else []
            L = [x for x in L1 if not L2 or x[0] < L2[0][0]] + L2
            win = [(L[i][0], sum(v for _, v in L[i:i + 5]) / 5) for i in range(0, len(L) - len(L) % 5, 5)]
            row = dict(run=n, arm=arm, seed=s, last_iter=L[-1][0] if L else None, done='EXIT 0' in (open(p2).read() if L2 else ''),
                       mean_25_30k=mean_in(L, 25000, 30000), mean_95_100k=mean_in(L, 95000, 100000),
                       mean_by_10k={k: mean_in(L, k - 5000, k) for k in range(10000, 100001, 10000)},
                       blowup_onset=onset(win), max_window=max((v for _, v in win if math.isfinite(v)), default=None))
            for tag, hj in (('r1', f'{base}/runs_kr/{n}/heldout_r1.json' if arm == 'plain' else f'{base}/runs_kr/{n}/heldout.json'),
                            ('r2', f'{base}/runs2_kr/{n}/heldout.json')):
                if os.path.exists(hj):
                    h = json.load(open(hj)); row[f'{tag}_iter'] = h['iteration']
                    for p in ('observed', 'strict', 'no_fast'):
                        row[f'{tag}_{p}_recall'] = h[f'heldout_{p}/recall_acc']; row[f'{tag}_{p}_loss'] = h[f'heldout_{p}/loss']
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    {'lc': score_lc, 'kr': score_kr}[sys.argv[1]]()
