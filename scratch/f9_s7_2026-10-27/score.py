"""Batch F9 / S7 scoring (queue 2026-10-27). Same measures and outputs as Batch V2's scoring (collect_v2.py for the
loss trajectory and onset, heldout.py final_eval.json for held-out recall, align_orc.py for the readout shares and
trunk-2 slow top SV) and BFR's bfr_lc.py, read from each run directory instead of the slurm logs.

Usage: python score.py [--ref150 CKPT] RUN_DIR [RUN_DIR ...] > scores.jsonl
  RUN_DIR holds train.log, final_eval.json and checkpoint_<iter>.pth (numbered every 50k); for R4ext also
  source_INT2_R4_300k.pth (scored as iteration 300000). --ref150 supplies the 150k checkpoint when the run has none
  (R4ext: B_late-s3141's checkpoint_00150000.pth, deckard ~/overnight_2026-10-01/analysis/ckptB/).
Per run:
  onset = first of five adjacent 500-iteration windows all > 5 or non-finite (V2/battery rule; late if > 30k);
  first_gt1e3; early_windows_gt5 (<= 30k); max and final loss (mean of the last 20 windows); stop = guard/nonfinite;
  heldout_strict_trace (every 50k, 8 batches, from the log); final observed/strict/no_fast recall (128 batches);
  per checkpoint: share of the readout's squared norm in range(B_2) and in range(B_fast) (B_fast redrawn with
  seed 918273 + 2, as ephemeral_model.decoupled_fast_feedback and align_orc.py), trunk-2 slow top SV, the largest
  |slow weight| per trunk layer and the readout (to see what the clamp would have bound); sv_ratio = SV 600k / 150k.
"""
import argparse, glob, json, math, os, re
import torch

torch.set_num_threads(4)


def losses(path):
    L, held, it = [], [], None
    stop = None
    for ln in open(path, errors='replace'):
        m = re.match(r'^(\d+) [\d.]+% \(', ln) or re.search(r'Interval metrics \(ending @ iter (\d+)', ln)
        if m: it = int(m.group(1)); continue
        m = re.match(r'^\s+loss:\s+(\S+)', ln)
        if m and it is not None:
            try: v = float(m.group(1))
            except ValueError: v = float('nan')
            L.append((it, v)); it = None; continue
        m = re.match(r'^\s+heldout_strict/recall_acc:\s+(\S+)', ln)
        if m: held.append(float(m.group(1)))
        if ln.startswith('Early stopping:'): stop = 'guard'
        if ln.startswith('Non-finite loss'): stop = 'nonfinite'
    # a resume re-prints the stretch since its checkpoint: keep the last value printed for each iteration
    d = dict(L)
    return sorted(d.items()), held, stop


def onset(L, after=0, thr=5.0, k=5):
    for i in range(len(L) - k + 1):
        w = L[i:i + k]
        if w[0][0] >= after and all((not math.isfinite(v)) or v > thr for _, v in w): return w[0][0]
    return None


def share(R, B):
    Q, _ = torch.linalg.qr(B.T.double())
    return float((R @ Q).norm() ** 2 / R.norm() ** 2)


def geometry(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = ck['model_state_dict']
    R = sd['i2o.per_sample_weights'].double().mean(0)
    V = R.shape[0]
    B2 = sd['linear_layers.2.feedback_weights']
    g = torch.Generator().manual_seed(918273 + 2)
    Bf = torch.randn(V, B2.shape[1], generator=g) * math.sqrt(2.0 / (V + B2.shape[1]))
    out = dict(iter=int(ck['iter']), B2_share=share(R, B2), Bfast_share=share(R, Bf), random=V / R.shape[1])
    for l in range(3):
        W = sd[f'linear_layers.{l}.per_sample_weights'].double().mean(0)
        m = sd[f'linear_layers.{l}.ephemeral_mask']
        S = torch.where(m, torch.zeros_like(W), W)
        if l == 2: out['W2_top_sv'] = float(torch.linalg.matrix_norm(S, 2))
        out[f'slow_absmax_L{l}'] = float(S.abs().max())
    out['readout_absmax'] = float(R.abs().max())
    return out


ap = argparse.ArgumentParser()
ap.add_argument('--ref150', default=None)
ap.add_argument('runs', nargs='+')
a = ap.parse_args()
for d in a.runs:
    L, held, stop = losses(os.path.join(d, 'train.log'))
    vals = [v for _, v in L]
    row = dict(run=os.path.basename(d.rstrip('/')), prints=len(L), last_iter=L[-1][0] if L else None, stop=stop,
               onset=onset(L), late_onset=onset(L, 30001),
               first_gt1e3=next((i for i, v in L if not math.isfinite(v) or v > 1e3), None),
               early_windows_gt5=sum(1 for i, v in L if i <= 30000 and (not math.isfinite(v) or v > 5)),
               max_loss=max((v for v in vals if math.isfinite(v)), default=None),
               final_loss=sum(vals[-20:]) / len(vals[-20:]) if vals else None,
               windows_gt1p5_after_100k=sum(1 for i, v in L if i > 100000 and (not math.isfinite(v) or v > 1.5)),
               heldout_strict_trace=held)
    fe = os.path.join(d, 'final_eval.json')
    if os.path.exists(fe):
        h = json.load(open(fe)); row['eval_iter'] = h.get('iteration')
        for p in ('observed', 'strict', 'no_fast'):
            row[f'{p}_recall'] = h.get(f'heldout_{p}/recall_acc')
    ckpts = sorted(glob.glob(os.path.join(d, 'checkpoint_*.pth')))
    src = os.path.join(d, 'source_INT2_R4_300k.pth')
    if os.path.exists(src): ckpts.append(src)
    if a.ref150 and not os.path.exists(os.path.join(d, 'checkpoint_00150000.pth')): ckpts.append(a.ref150)
    geo = sorted((geometry(p) for p in ckpts), key=lambda g: g['iter'])
    row['geometry'] = geo
    sv = {g['iter'] // 1000 * 1000: g['W2_top_sv'] for g in geo}
    if 150000 in sv and 600000 in sv: row['sv_ratio'] = sv[600000] / sv[150000]
    print(json.dumps(row), flush=True)
