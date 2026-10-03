"""Where does the slow weight growth go? usage: where_growth.py SPEC.pt OUT.json BASE_CKPT:LATE_CKPT [...]
Per trunk layer, delta = slow(late) - slow(base): its Frobenius norm relative to slow(base), top-6 singular values,
share of ||delta||^2 inside the capped left/right subspace (|U^T delta V|^2 / |delta|^2), and the same for i2o, plus
bias norms change."""
import json, sys, torch
spec = torch.load(sys.argv[1], map_location='cpu', weights_only=False)['layers']
def load(p):
    sd = torch.load(p, map_location='cpu', weights_only=False)['model_state_dict']
    return sd
def slow(sd, n): return sd[n + '.per_sample_weights'].float().mean(0) * (~sd[n + '.ephemeral_mask'])
out = {}
for pair in sys.argv[3:]:
    b, l = pair.split(':'); sb, sl = load(b), load(l); rec = {}
    for n in list(spec) + ['i2o']:
        d = slow(sl, n) - slow(sb, n); s0 = slow(sb, n)
        r = {'rel_fro': float(d.norm() / s0.norm()), 'delta_top6': torch.linalg.svdvals(d)[:6].tolist(),
             'full_top3_base': torch.linalg.svdvals(s0)[:3].tolist(), 'full_top3_late': torch.linalg.svdvals(slow(sl, n))[:3].tolist()}
        if n in spec:
            U, V = spec[n]['U'], spec[n]['V']
            r['frac_in_cap_subspace'] = float((U.T @ d @ V).norm() ** 2 / d.norm() ** 2)
        rec[n] = r
    rec['bias_norm_base_late'] = {k: [float(sb[k].norm()), float(sl[k].norm())] for k in sb if k.endswith('.bias')}
    out[pair] = rec; print(pair, flush=True)
json.dump(out, open(sys.argv[2], 'w'), indent=1)
