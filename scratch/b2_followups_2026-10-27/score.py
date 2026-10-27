"""Paired B2fu CE/tax/horizon summaries from the original B2 evaluator JSON schema.

Fails closed for missing seeds/controls; no smoke selection or partial-seed average.
Usage: python score.py --root /path/to/b2_followups_2026-10-27 --part A --out scores-A.json
"""
import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev

SEEDS = (7201, 7302, 7403, 7504, 7605)
ARMS = {'A': ('DEEP-W16', 'DEEP-F03', 'DEEP-F10'),
        'B': ('REC-NOFAST', 'REC-DEEP', 'RNN-BPTT')}
CLASSES = ('all', 'novel_word_letter', 'repeat_word_letter', 'nonletter')


def summary(values):
    sd = stdev(values)
    half = 2.776445105 * sd / math.sqrt(5)  # t(.975, df=4)
    return {'values': values, 'mean': mean(values), 'sd': sd,
            'ci95_t': [mean(values) - half, mean(values) + half]}


def read(root, sub, arm, seed, stories):
    p = root / sub / f'{arm}-s{seed}.json'
    value = json.loads(p.read_text())
    assert value['stories'] == stories and value['length'] == 512, p
    assert value['iteration'] == 20000, (p, value['iteration'])
    assert value['arm_config']['seed'] == seed, (p, value['arm_config']['seed'])
    return value


def score(root, part):
    arms = ARMS[part]
    final = {a: [read(root, 'eval_out/final', a, s, 4096) for s in SEEDS] for a in arms}
    horizon = {a: [read(root, 'eval_out/horizon', a, s, 1024) for s in SEEDS] for a in arms}
    if part == 'A':
        nofast = [read(root, 'controls/final', 'NOFAST', s, 4096) for s in SEEDS]
        deep = [read(root, 'controls/final', 'DEEP', s, 4096) for s in SEEDS]
        oldwipe = [read(root, 'controls/horizon4096', 'DEEP', s, 4096) for s in SEEDS]
    else:
        nofast = final['REC-NOFAST']
        deep = [read(root, 'controls/final', 'DEEP', s, 4096) for s in SEEDS]
    out = {'part': part, 'seeds': SEEDS, 'units': 'nats/char', 'arms': {}}
    for a in arms:
        fs = final[a]; hs = horizon[a]
        entry = {'ce': {}, 'tax': {}, 'on_minus_wipe16': summary(
            [h['ce']['on']['all'] - h['ce']['wipe16']['all'] for h in hs])}
        for c in CLASSES:
            entry['ce'][c] = {p: summary([f['ce'][c][p] for f in fs]) for p in ('on', 'off')}
            entry['tax'][c] = summary([f['ce'][c]['off'] - n['ce'][c]['off'] for f, n in zip(fs, nofast)])
        if part == 'A':
            entry['old_on_minus_new_on'] = summary([d['ce']['all']['on'] - f['ce']['all']['on']
                                                    for d, f in zip(deep, fs)])
            entry['old_tax_minus_new_tax'] = summary([d['ce']['all']['off'] - f['ce']['all']['off']
                                                      for d, f in zip(deep, fs)])
            entry['old_wipe16_minus_new_on'] = summary([d['ce']['wipe16']['all'] - f['ce']['all']['on']
                                                        for d, f in zip(oldwipe, fs)])
        out['arms'][a] = entry
    if part == 'B':
        for control in ('REC-NOFAST', 'RNN-BPTT'):
            out[f'{control}_minus_REC-DEEP'] = summary([
                c['ce']['all']['on'] - d['ce']['all']['on']
                for c, d in zip(final[control], final['REC-DEEP'])])
        out['old_DEEP_minus_RNN-BPTT'] = summary([
            d['ce']['all']['on'] - r['ce']['all']['on'] for d, r in zip(deep, final['RNN-BPTT'])])
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--part', choices=ARMS, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.write_text(json.dumps(score(args.root, args.part), indent=2) + '\n')
