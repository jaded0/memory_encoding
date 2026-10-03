"""Build --norm_cap_file specs from the 150k checkpoint. usage: build_norm_specs.py CKPT OUT_DIR [--margin 1.1]
capall_t: spectrum caps on trunk 0-2; capall_trb: + spectrum cap on i2o + bias norm caps (trunk 0-2, i2o);
frob: Frobenius cap on the trunk slow matrices only."""
import argparse, os, torch
ap = argparse.ArgumentParser(); ap.add_argument('ckpt'); ap.add_argument('out'); ap.add_argument('--margin', type=float, default=1.1)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
sd = torch.load(a.ckpt, map_location='cpu', weights_only=False)['model_state_dict']
slow = lambda n: sd[n + '.per_sample_weights'].float().mean(0) * (~sd[n + '.ephemeral_mask'])
T = ['linear_layers.0', 'linear_layers.1', 'linear_layers.2']
spec = {n: {'mode': 'spectrum', 'caps': a.margin * torch.linalg.svdvals(slow(n))} for n in T}
torch.save({'layers': spec}, f'{a.out}/capall_t.pt')
spec_r = dict(spec); spec_r['i2o'] = {'mode': 'spectrum', 'caps': a.margin * torch.linalg.svdvals(slow('i2o'))}
bias = {n: a.margin * float(sd[n + '.bias'].norm()) for n in T + ['i2o']}
torch.save({'layers': spec_r, 'biases': bias}, f'{a.out}/capall_trb.pt')
torch.save({'layers': {n: {'mode': 'frobenius', 'caps': a.margin * float(slow(n).norm())} for n in T}}, f'{a.out}/frob.pt')
print({n: round(float(slow(n).norm()), 2) for n in T + ['i2o']}, {k: round(v, 3) for k, v in bias.items()})
