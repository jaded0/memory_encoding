"""Render factual sparse-screen hand-back from committed raw JSON (no launching)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAMES = ('linear', 'order1', 'order2', 'sparse_z16_m273_k8', 'sparse_z16_m273_k17',
         'sparse_z16_m273_k34', 'sparse_full_m4096_k64', 'sparse_full_m4096_k256')
SHORT = ('lin', 'o1', 'o2', 'z-k8', 'z-k17', 'z-k34', 'full-k64', 'full-k256')


def report():
    results = [json.loads((ROOT / f'evidence/sparse/load{d}.json').read_text()) for d in (0, 4, 8, 16, 32)]
    lines = ['## Frozen screening table', '',
             'Cells are **key ratio / selectivity**, each the median over 128 paired episodes.',
             'Key ratio = absolute store kernel / median absolute wrong kernel; selectivity =',
             'signed store kernel / RMS wrong kernel. Linear is the unmasked expectation kernel,',
             'not the actual masked vector read. No hypothetical map changes the replay.', '',
             '| D | layer | ' + ' | '.join(SHORT) + ' |',
             '|---|---|' + '---:|'*len(NAMES)]
    for result in results:
        for layer, maps in result['summary'].items():
            cells = [f"{maps[name]['key_ratio']:.3f} / {maps[name]['selectivity']:.3f}" for name in NAMES]
            lines.append(f"| {result['load']} | {layer} | " + ' | '.join(cells) + ' |')
    d8 = results[2]['summary']
    l1, l2 = d8['L1'], d8['L2']
    spar = 'sparse_z16_m273_k17'
    wide = 'sparse_full_m4096_k256'
    held = [l1['order1']['key_ratio'] < l1[spar]['key_ratio'] < l1['order2']['key_ratio'],
            l2[spar]['key_ratio'] < 1.5*l2['order2']['key_ratio'],
            all(l2[wide][key] > l2[spar][key] for key in ('key_ratio', 'selectivity')),
            all(d8[layer]['sparse_z16_m273_k8']['wrong_abs_median'] <
                d8[layer]['sparse_z16_m273_k34']['wrong_abs_median'] for layer in ('L1', 'L2'))]
    lines += ['', '## Predictions (written in the vault draft before screening)', '']
    for n, value in enumerate(held, 1):
        lines.append(f'- SP{n}: **{"held" if value else "falsified"}** (criteria unchanged; D8 table above).')
    lines += ['', '## Wrong-key cosine distributions after mapping', '',
              'Pooled across written wrong steps and all targets (longer episodes contribute more',
              'steps). Signed cosines; quantiles are not quantiles of per-episode means.',
              'All individual wrong cosines/kernels, store cosines/kernels, episode ratios and',
              'selectivities are retained in raw JSON, along with absolute medians and means.', '',
              '| D | layer | map | n | min | p05 | p25 | median | p75 | p95 | max |',
              '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for result in results:
        for layer, maps in result['summary'].items():
            for name, short in zip(NAMES, SHORT):
                r = maps[name]
                quantiles = ' | '.join(f'{v:.3f}' for v in r['wrong_quantiles'].values())
                lines.append(f"| {result['load']} | {layer} | {short} | {r['wrong_count']} | {quantiles} |")
    lines += ['', '## Resource evidence', '',
              f"Frozen checkpoint SHA256: `{results[0]['checkpoint_sha256']}`.",
              f"Screen code: `{results[0]['code_commit']}`; same original baseline metrics asserted in code.",
              f"Timed replay total {sum(r['elapsed_seconds'] for r in results):.1f}s; "
              f"max point {max(r['elapsed_seconds'] for r in results):.1f}s; "
              f"max allocated {max(r['peak_allocated_mb'] for r in results):.1f} MiB.",
              'GPU0 1080 Ti, unfused, allocator capped at .30 (~3.3 GiB); every point timeout 580s.',
              'No training, production run, Slurm submission or main merge.']
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    print(report(), end='')
