"""Fail-closed K15 launch approval and immutable checkout/data verification."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def validate(path):
    path = Path(path).resolve()
    m = json.loads(path.read_text())
    assert m['launch_approved'] is True, 'K15 not armed/approved'
    assert m['approval_reference'] == 'Jaden 2026-11-05: approve K15, launch it after K13 lands'
    assert m['sink'] == 'fixed', 'This launcher registers the fixed sink only'
    assert m['training_timeout_seconds'] >= 60 * 3600
    assert m['scoring_timeout_seconds'] >= 2160
    assert m['krx_supported'] is False, 'KRX unavailable on this branch: i2h fast entries cannot be replaced'
    assert m['checks_passed'] is True
    assert m['prereg_reference'] and m['checks_reference']
    code = path.parent / 'code'
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=code, text=True).strip() == m['commit']
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=code, text=True).strip()
    for relative, digest in m['data_sha256'].items():
        p = code / relative
        assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == digest, f'data changed: {p}'
    assert m['data_sha256'], 'No verified dataset'
    return m


if __name__ == '__main__':
    validate(sys.argv[1])
