import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location('bigkey_summary', Path(__file__).resolve().parents[1]/'scratch/k13_bigkey_2026-10-27/summarize.py')
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def test_missing_pilot_rows_not_dropped(tmp_path):
    text = summary.summarize(tmp_path, 'pilot')
    assert text.count('MISSING/INVALID') == 4
    assert 'NOT launch approval' in text


def test_confirmation_requires_all_test_seeds_and_completion(tmp_path):
    for arm in summary.ARMS:
        for seed in summary.SEEDS:
            d = tmp_path / f'mqar2_8_q4-{arm}-s{seed}'
            d.mkdir()
            (d/'final_eval.json').write_text(json.dumps({'split':'test','heldout_strict/recall_acc':.5 if arm=='lin' else .6}))
            (d/'train.log').write_text('TRAIN EXIT 0\nEVAL EXIT 0\n')
    text = summary.summarize(tmp_path, 'confirm')
    assert '| lin | 5/5 | 0.5000' in text
    assert 'spar − lin: +0.1000; positive 5/5' in text
    (tmp_path/'mqar2_8_q4-lin-s2718/final_eval.json').write_text(json.dumps({'split':'validation','heldout_strict/recall_acc':1.}))
    text = summary.summarize(tmp_path, 'confirm')
    assert '| lin | 4/5 | INCOMPLETE' in text
    assert 'rep2 − lin: INCOMPLETE' in text
