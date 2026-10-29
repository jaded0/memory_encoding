"""Read-only scientific audit and Markdown tables; NEVER authorizes a launch.

The frozen gate retains its A100 training-walltime guard. This audit deliberately
does not patch that guard, rewrite pilot evidence, or emit an approved manifest.
Usage: python review_deckard.py PILOT_ROOT --out review.json --markdown tables.md
"""
import argparse
import json
from pathlib import Path

from gate import KS, RECIPE, stable_budget


def review(root):
    rows = []
    for k in KS:
        directory = Path(root) / f'mqar2_{k}_q4-lin-s2718'
        analysis, run, curve = [json.loads((directory / name).read_text()) for name in
                                ('analysis.json', 'pilot_run.json', 'learning_curve.json')]
        result = next(r for r in analysis['results'] if r['dataset'] == f'mqar2_{k}_q4')
        assert analysis['split'] == curve['split'] == 'validation'
        assert run['recipe'] == RECIPE and run['train_exit'] == 0
        assert analysis['code_commit'] == run['code_commit']
        assert result['episodes'] == 128 and result['targets'] == 512
        assert (directory / 'train.log').read_text().count('TRAIN EXIT 0\n') == 1
        try:
            iterations = stable_budget(curve, analysis['iteration'])
            stable, reason = True, None
        except ValueError as exc:
            iterations, stable, reason = None, False, str(exc)
        checks = dict(result['gate']['checks'], stable_budget=stable)
        rows.append(dict(k=k, iterations=iterations, checks=checks,
                         scientific_pass=all(checks.values()), stability_reason=reason,
                         curve=curve, result=result, run=run,
                         scoring_seconds=analysis['elapsed_seconds']))
    return dict(launch_approved=False, scientific_candidate=next(
        (r['k'] for r in rows if r['scientific_pass']), None), rows=rows)


def tables(audit):
    lines = ['### Validation recall and suppression', '',
             '| K | Strict | First | Observed | No-fast | Post-store first | Gain | Error recovered | Store-only first | Gap first |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for row in audit['rows']:
        a = row['result']['acc']
        values = [a[n] for n in ('strict', 'strict_first', 'observed', 'no_fast', 'post_store')]
        gain = a['post_store'] - a['strict_first']
        values += [gain, gain / (1 - a['strict_first']), a['store_only'], a['paired_gap']]
        lines.append('| ' + str(row['k']) + ' | ' + ' | '.join(f'{v:.4f}' for v in values) + ' |')
    lines += ['', '### Store-key geometry', '',
              'PR is pooled normalized table-store PR, not median episode PR. Cosines are signed median / absolute median.', '',
              '| K | L0 PR | L1 PR | L2 PR | L0 wrong cosine | L1 wrong cosine | L2 wrong cosine |',
              '|---|---|---|---|---|---|---|']
    for row in audit['rows']:
        items = [row['result']['layers'][f'L{i}']['items'] for i in range(3)]
        values = [f"{v['key_pr_pooled']:.4f}" for v in items]
        values += [f"{v['wrong_cosines']['quantiles']['p50']:.4f} / {v['wrong_cosines']['abs_median']:.4f}" for v in items]
        lines.append('| ' + str(row['k']) + ' | ' + ' | '.join(values) + ' |')
    lines += ['', '### Frozen scientific criteria (not full launch approval)', '',
              '| K | Strict range | First range | Gain ≥.10 | Recovery ≥80% | L0 PR ≥4 | L1 PR ≥2 | Stable | Budget |',
              '|---|---|---|---|---|---|---|---|---|']
    names = ('strict_range', 'strict_first_range', 'post_store_absolute_improvement',
             'post_store_recovers_80pct_error', 'L0_item_key_pr_pooled', 'L1_item_key_pr_pooled', 'stable_budget')
    for row in audit['rows']:
        values = ['PASS' if row['checks'][n] else '**FAIL**' for n in names]
        lines.append('| ' + str(row['k']) + ' | ' + ' | '.join(values) + f" | {row['iterations']} |")
    names = ('linear', 'order1', 'order2', 'sparse_z16_m273_k17',
             'sparse_full_m4096_k64', 'sparse_full_m4096_k256')
    for row in audit['rows']:
        lines += ['', f"### K={row['k']} kernel screen", '',
                  'Hypothetical kernels on lin replay state; all written-step keys, not only table stores. Not trained-arm accuracy.', '',
                  '| Layer | Kernel | Wrong signed median | Wrong absolute median | Absolute p95 | Key ratio | Selectivity |',
                  '|---|---|---|---|---|---|---|']
        for layer, kernels in row['result']['kernels'].items():
            for name in names:
                kernel = kernels[name]
                wrong = kernel['wrong_cosines']
                values = [wrong['quantiles']['p50'], wrong['abs_median'],
                          wrong['abs_quantiles']['p95'], kernel['key_ratio'], kernel['selectivity']]
                lines.append(f'| {layer} | {name} | ' + ' | '.join(f'{v:.4g}' for v in values) + ' |')
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--out', required=True)
    parser.add_argument('--markdown', required=True)
    args = parser.parse_args()
    audit = review(args.root)
    Path(args.out).write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
    Path(args.markdown).write_text(tables(audit))
