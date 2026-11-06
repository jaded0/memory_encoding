"""Initialize prepared K15 files, never an armed/used queue. Run on deckard only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from k15_queue import atomic, initial


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--krx-supported', action='store_true')
    a = p.parse_args()
    root = a.root.resolve()
    assert not (root/'PREPARED').exists(), 'Already prepared; inspect, never overwrite'
    assert not (root/'ARMED').exists(), 'Already armed'
    assert not (root/'status.txt').read_text().strip(), 'Used status; refusing'
    code = root/'code'
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=code, text=True).strip() == a.commit
    datasets = ['mqar2_16_q4'] + (['long_range_memory_dataset'] if a.krx_supported else [])
    hashes = {}
    for name in datasets:
        files = sorted(p for p in (code/'synth_datasets'/name).rglob('*') if p.is_file())
        assert files, f'missing dataset {name}'
        for file in files:
            hashes[str(file.relative_to(code))] = hashlib.sha256(file.read_bytes()).hexdigest()
    tasks = initial(a.krx_supported)
    state = dict(pending=tasks, active={}, attempts={}, done=[], giveup=[], reuse_added=[])
    atomic(root/'queue_state.json', json.dumps(state, indent=2)+'\n')
    atomic(root/'queue.txt', ''.join(t+'\n' for t in tasks))
    manifest = dict(schema_version=1, launch_approved=False,
        approval_reference='Jaden 2026-11-05: approve K15, launch it after K13 lands',
        commit=a.commit, sink='fixed', krx_supported=a.krx_supported,
        training_timeout_seconds=72*3600, scoring_timeout_seconds=7200,
        checks_passed=False, prereg_reference=None, checks_reference=None, data_sha256=hashes,
        retry_policy='max 3 requeues / 4 total attempts, including timeouts; cumulative 72h training budget',
        krx_note='Dropped: i2h has fast entries; order2 supports trunk layers only' if not a.krx_supported else 'Every-fast-layer replacement verified; see checks_reference')
    atomic(root/'manifest.json', json.dumps(manifest, indent=2)+'\n')
    (root/'PREPARED').write_text(a.commit+'\n')
    (root/'SKELETON').unlink(missing_ok=True)


if __name__ == '__main__':
    main()
