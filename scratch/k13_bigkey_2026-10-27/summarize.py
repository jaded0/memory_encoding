"""Summarize pilot diagnostics or confirmatory strict TEST scores without dropping failures."""
import argparse
import json
from pathlib import Path
import statistics

ARMS = ('lin', 'par2', 'par1', 'rep2', 'spar')
SEEDS = (2718, 3141, 1618, 1414, 1732)


def summarize(root, mode):
    root = Path(root)
    lines = []
    if mode == 'pilot':
        lines += ['| K | status | iter | strict | first | post-store | L0 PR | L1 PR | L2 PR | regime checks |',
                  '|---|---|---|---|---|---|---|---|---|---|']
        for k in (4, 8, 16, 32):
            d = root / f'mqar2_{k}_q4-lin-s2718'
            try:
                run = json.loads((d/'pilot_run.json').read_text())
                a = json.loads((d/'analysis.json').read_text())
                r = next(r for r in a['results'] if r['dataset'] == f'mqar2_{k}_q4')
                pr = [r['layers'][f'L{i}']['items']['key_pr_pooled'] for i in range(3)]
                acc = r['acc']
                fields = [k, run['status'], a['iteration'], *(f'{acc[n]:.4f}' for n in ('strict','strict_first','post_store')),
                          *(f'{v:.3f}' for v in pr), r['gate']['regime_verified']]
                lines.append('| ' + ' | '.join(map(str, fields)) + ' |')
            except (OSError, ValueError, KeyError, StopIteration) as exc:
                lines.append(f'| {k} | MISSING/INVALID: {type(exc).__name__} | — | — | — | — | — | — | — | false |')
        lines.append('\nRegime checks alone are NOT launch approval: stability and runtime gates also apply.')
    else:
        datasets = sorted({p.name.rsplit('-', 2)[0] for p in root.glob('mqar2_*-*-s*') if p.is_dir()})
        for dataset in datasets:
            values = {}
            lines += [f'## {dataset}', '| arm | n/5 | mean strict test | per-seed strict |', '|---|---|---|---|']
            for arm in ARMS:
                values[arm], cells = {}, []
                for seed in SEEDS:
                    d = root / f'{dataset}-{arm}-s{seed}'
                    try:
                        a = json.loads((d/'final_eval.json').read_text())
                        if a.get('split') != 'test':
                            raise ValueError('not test')
                        log = (d/'train.log').read_text()
                        if 'TRAIN EXIT 0\n' not in log or 'EVAL EXIT 0\n' not in log:
                            raise ValueError('incomplete')
                        value = float(a['heldout_strict/recall_acc'])
                        if not 0 <= value <= 1:
                            raise ValueError('invalid recall')
                        values[arm][seed] = value
                        cells.append(f'{seed}:{value:.4f}')
                    except (OSError, ValueError, KeyError):
                        cells.append(f'{seed}:MISSING/INVALID')
                mean = f'{statistics.mean(values[arm].values()):.4f}' if len(values[arm]) == 5 else 'INCOMPLETE'
                lines.append(f'| {arm} | {len(values[arm])}/5 | {mean} | {", ".join(cells)} |')
            for left, right in (('spar','par2'), ('par2','par1'), ('rep2','lin'), ('par2','lin'), ('spar','lin')):
                paired = sorted(set(values[left]) & set(values[right]))
                diffs = [values[left][s]-values[right][s] for s in paired]
                text = f'{statistics.mean(diffs):+.4f}; positive {sum(v>0 for v in diffs)}/{len(diffs)}' if len(diffs)==5 else 'INCOMPLETE'
                lines.append(f'- {left} − {right}: {text}; paired differences {dict(zip(paired,diffs))}')
        if not datasets:
            lines.append('No confirmatory run directories found. No result.')
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--mode', choices=('pilot','confirm'), default='pilot')
    args = parser.parse_args()
    print(summarize(args.root, args.mode), end='')
