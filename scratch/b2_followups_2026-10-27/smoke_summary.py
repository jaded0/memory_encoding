"""Extract infrastructure evidence, never select an arm from smoke losses."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
args = p.parse_args()
out = {}
for arm in ('DEEP-W16', 'DEEP-F03', 'DEEP-F10', 'REC-NOFAST', 'REC-DEEP', 'RNN-BPTT'):
    text = (args.root / 'logs' / f'{arm}.log').read_text()
    assert f'SMOKE_OK {arm}' in text, arm
    times = re.findall(r'^2026-\d\d-\d\dT[^\n]+$', text, re.M)
    seconds = (datetime.fromisoformat(times[-1]) - datetime.fromisoformat(times[0])).total_seconds()
    assert seconds <= 600, (arm, seconds)
    peak = int(re.search(r'CUDA_PEAK_RESERVED_BYTES (\d+)', text)[1])
    assert peak < 5_000_000_000
    final = json.loads((args.root / 'eval' / f'{arm}.json').read_text())
    horizon = json.loads((args.root / 'eval' / f'{arm}-horizon.json').read_text())
    assert final['iteration'] == 100
    if arm in ('REC-NOFAST', 'RNN-BPTT'):
        assert final['ce']['all']['on'] == final['ce']['all']['off']
        assert len(set(horizon['ce'][p]['all'] for p in horizon['ce'])) == 1
    out[arm] = {'seconds_train_plus_two_evals': seconds, 'iterations': final['iteration'],
                'reserved_peak_mib': round(peak / 1024**2, 1),
                'allocated_peak_mib': round(int(re.search(r'CUDA_PEAK_ALLOCATED_BYTES (\d+)', text)[1]) / 1024**2, 1),
                'last_interval_iters_per_sec': float(re.findall(r'iters_per_sec: ([\d.]+)', text)[-1]),
                'tiny_stories': final['stories'], 'tiny_length': final['length'],
                'horizon_periods': horizon['wipe_every'], 'status': 'passed'}
args.out.write_text(json.dumps(out, indent=2) + '\n')
