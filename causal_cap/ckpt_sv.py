"""Per checkpoint: cap8-block singular values, full slow top-8 singular values and Frobenius norm, per trunk layer.
usage: ckpt_sv.py SPEC.pt OUT.json CKPT [CKPT ...]  (slow matrix = batch-mean of per_sample_weights, fast positions zero)"""
import json, re, sys
import torch
spec = torch.load(sys.argv[1], map_location='cpu', weights_only=False)['layers']
out = {}
for path in sys.argv[3:]:
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = ck['model_state_dict']; it = int(re.findall(r'(\d+)\.pth', path)[0]) if re.search(r'\d+\.pth', path) else ck['iter']
    rec = {}
    for n, d in spec.items():
        S = sd[n + '.per_sample_weights'].float().mean(0) * (~sd[n + '.ephemeral_mask'])
        rec[n] = {'block_sv': torch.linalg.svdvals(d['U'].T @ S @ d['V']).tolist(), 'caps': d['caps'].tolist(),
                  'top8': torch.linalg.svdvals(S)[:8].tolist(), 'fro': float(torch.linalg.matrix_norm(S))}
    out[it] = rec
    print(it, flush=True)
json.dump(out, open(sys.argv[2], 'w'))
