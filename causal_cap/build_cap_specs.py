"""Build --sv_cap_file specs from two checkpoints (offline, hindsight: uses the later, collapsed checkpoint).

  python causal_cap/build_cap_specs.py CKPT_150k CKPT_205k OUT_DIR [--margin 1.1] [--seed 0]

Slow matrix of a trunk layer = mean over the batch copies of per_sample_weights, ephemeral positions zeroed.
Delta = slow(late) - slow(base); delta = U S V^T. Specs written (one file each, trunk layers linear_layers.0-2):
  cap8     top 8 singular directions of delta           (CAP)
  cap16    top 16                                        (CAP16)
  sham8    8 random orthonormal left/right directions    (SHAM; random seed per layer)
  sham2_8  singular directions of delta ranked 9-16      (SHAM2; next k)
  sham3_8  top 8 singular directions of the BASE slow matrix (SHAM3: large-sv directions that are not the growth directions)
caps = margin x (singular values of the block U^T S_base V), sorted descending.
"""
import argparse
import os
import sys

import torch

LAYERS = ['linear_layers.0', 'linear_layers.1', 'linear_layers.2']


def slow_matrices(path):
    sd = torch.load(path, map_location='cpu', weights_only=False)['model_state_dict']
    return {n: sd[n + '.per_sample_weights'].float().mean(0) * (~sd[n + '.ephemeral_mask']) for n in LAYERS}


def block_caps(S, U, V, margin):
    return margin * torch.linalg.svdvals(U.T @ S @ V)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('base'); ap.add_argument('late'); ap.add_argument('out_dir')
    ap.add_argument('--margin', type=float, default=1.1)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    base, late = slow_matrices(a.base), slow_matrices(a.late)
    g = torch.Generator().manual_seed(a.seed)
    specs = {k: {} for k in ('cap8', 'cap16', 'sham8', 'sham2_8', 'sham3_8')}
    report = {}
    for n in LAYERS:
        S0 = base[n]
        U, s, Vh = torch.linalg.svd(late[n] - S0)
        V = Vh.T
        U0, s0, V0h = torch.linalg.svd(S0)
        out, inn = S0.shape
        rand = lambda: (torch.linalg.qr(torch.randn(out, 8, generator=g))[0], torch.linalg.qr(torch.randn(inn, 8, generator=g))[0])
        choices = {'cap8': (U[:, :8], V[:, :8]), 'cap16': (U[:, :16], V[:, :16]), 'sham8': rand(),
                   'sham2_8': (U[:, 8:16], V[:, 8:16]), 'sham3_8': (U0[:, :8], V0h.T[:, :8])}
        for k, (Uk, Vk) in choices.items():
            specs[k][n] = {'U': Uk.contiguous(), 'V': Vk.contiguous(), 'caps': block_caps(S0, Uk, Vk, a.margin)}
        report[n] = {k: [round(float(x), 3) for x in specs[k][n]['caps'][:4] / a.margin] for k in specs}
        report[n]['delta_sv'] = [round(float(x), 3) for x in s[:10]]
        report[n]['base_top_sv'] = [round(float(x), 3) for x in s0[:4]]
        report[n]['late_top_sv'] = [round(float(x), 3) for x in torch.linalg.svdvals(late[n])[:4]]
    for k, layers in specs.items():
        torch.save({'layers': layers, 'meta': {'name': k, 'margin': a.margin, 'base': a.base, 'late': a.late}},
                   os.path.join(a.out_dir, f'{k}.pt'))
    for n, r in report.items():
        print(n)
        for k, v in r.items():
            print('  ', k, v)


if __name__ == '__main__':
    main()
