"""Frozen paired kernel-only screening. One load per invocation, no training.

Reuse the exact original strict replay and written-step/store indices. Adds an
observer only: none of the hypothetical kernels changes the replay's fast state.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

sys.path.insert(0, os.getcwd())
import numpy as np
import torch
from torch.nn import functional as F
from ephemeral_model import order2_features, order2_sparse_expansion
from kr_load_tasks import CHARSET, generate_sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint')
    ap.add_argument('--load', type=int, required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.30)
    spec = importlib.util.spec_from_file_location('o2', 'scratch/order2_2026-10-22/o2_keys.py')
    o2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(o2)
    read = o2.read_checkpoint

    def unfused(path):
        ck = read(path)
        ck['config']['fused_update'] = False
        return ck

    def batches(dataset, batch_size, nb, device, split):
        rng = random.Random(20261027)
        for _ in range(nb):
            texts = [generate_sample(rng, 'load', args.load) for _ in range(batch_size)]
            ids = torch.tensor([[CHARSET.index(c) for c in t] for t in texts], device=device)
            yield texts, F.one_hot(ids, len(CHARSET)).float()

    o2.read_checkpoint, o2.load_heldout_batches = unfused, batches
    rows, expansions = {}, {}

    def observe(layer, xs, xq, si, oi, proj, fraction, seed):
        if layer not in (1, 2):
            return
        x = torch.cat((xs, xq[None]))
        maps = {'linear': x}
        for order in ('order1', 'order2'):
            maps[order] = order2_features(x, proj, order, 4., fraction)
        for wide, m, ks in ((False, 273, (8, 17, 34)), (True, 4096, (64, 256))):
            dim = x.shape[1] if wide else 16
            key = (layer, wide)
            if key not in expansions:
                expansions[key] = order2_sparse_expansion(dim, m, seed, layer).to(x)
            # Identity projection bypasses P, but uses exactly the same normalized
            # signed-top-k algorithm; never changes the model or fast writes.
            p = torch.eye(dim, device=x.device, dtype=x.dtype) if wide else proj
            for k in ks:
                name = f'sparse_{"full" if wide else "z16"}_m{m}_k{k}'
                maps[name] = order2_features(x, p, 'sparse', 4., fraction,
                                             sparse_expansion=expansions[key], sparse_k=k)
        for name, features in maps.items():
            kv = (features[:-1] @ features[-1]).cpu().numpy()
            cos = F.cosine_similarity(features[:-1], features[-1:], dim=1).cpu().tolist()
            rows.setdefault(f'L{layer}', {}).setdefault(name, []).append({
                'ratio': o2.ratio(abs(kv[si]), np.abs(kv[oi])),
                'selectivity': o2.selectivity(kv[si], kv[oi]),
                'store_cosine': cos[si], 'wrong_cosines': [cos[j] for j in oi],
                'store_kernel': float(kv[si]), 'wrong_kernels': kv[oi].tolist()})

    start = time.monotonic()
    baseline = o2.run(args.checkpoint, 8, 'cuda', [4.], kernel_observer=observe)
    summary = {}
    for layer, maps in rows.items():
        summary[layer] = {}
        for name, records in maps.items():
            wrong = np.array([v for r in records for v in r['wrong_cosines']])
            summary[layer][name] = {
                'key_ratio': float(np.median([r['ratio'] for r in records])),
                'selectivity': float(np.median([r['selectivity'] for r in records])),
                'wrong_count': len(wrong), 'wrong_mean': float(wrong.mean()),
                'wrong_abs_median': float(np.median(abs(wrong))),
                'wrong_quantiles': dict(zip(('min', 'p05', 'p25', 'p50', 'p75', 'p95', 'max'),
                                            np.quantile(wrong, [0, .05, .25, .5, .75, .95, 1]).tolist()))}
            # Assert the re-emitted metrics are the original ones, not a new statistic.
            tag = {'linear': 'lin', 'order1': 'o1_t4', 'order2': 'o2_t4'}.get(name)
            if tag:
                assert np.isclose(summary[layer][name]['key_ratio'], baseline['layers'][layer][tag+'_ratio'])
                assert np.isclose(summary[layer][name]['selectivity'], baseline['layers'][layer][tag+'_sel'])
    result = dict(load=args.load, episode_seed=20261027, targets=baseline['targets'],
                  checkpoint=args.checkpoint,
                  checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                  code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  elapsed_seconds=time.monotonic()-start,
                  peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
                  summary=summary, records=rows, baseline=baseline)
    Path(args.out).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('records', 'baseline')}), flush=True)


if __name__ == '__main__':
    main()
