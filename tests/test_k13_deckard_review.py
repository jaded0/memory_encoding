"""Review helpers do not loosen the frozen gate or authorize confirmation."""
import importlib.util
import json
from pathlib import Path
import runpy

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scratch/k13_bigkey_2026-10-27'
estimate = runpy.run_path(str(SCRIPTS / 'estimate_deckard.py'))['estimate']


def timing_fixture(tmp_path):
    pilot = tmp_path / 'pilot/mqar2_16_q4-lin-s2718'
    pilot.mkdir(parents=True)
    (pilot / 'pilot_run.json').write_text(json.dumps(dict(training_seconds=33000, requested_iterations=200000)))
    timing = tmp_path / 'timing'
    timing.mkdir()
    for arm in ('par2', 'spar'):
        (timing / f'{arm}.status').write_text('TIMING EXIT 0 ELAPSED 200 REQUESTED_ITERS 1000\n')
        (timing / f'{arm}.log').write_text('  iters_per_sec: 5.0\n' * 10)


def test_runtime_includes_contention_scoring_pairing_and_no_approval(tmp_path):
    timing_fixture(tmp_path)
    result = estimate(tmp_path, 195000)
    assert not result['launch_approved'] and not result['within_48_hours']
    assert result['arms']['par2']['seconds_per_iter'] == pytest.approx(.48)
    assert result['arms']['par1']['seconds_per_iter'] == result['arms']['spar']['seconds_per_iter']
    assert result['total_a6000_lane_hours'] == pytest.approx(4 * result['ideal_four_lane_wall_hours'])
    assert result['seed_block_paired_wall_hours'] > result['seven_wave_wall_hours']
    assert result['arms']['lin']['scoring_seconds'] > 0
    assert 'not an empirical' in result['conversion']['basis']


def test_runtime_refuses_failed_smoke(tmp_path):
    timing_fixture(tmp_path)
    (tmp_path / 'timing/spar.status').write_text('TIMING EXIT 124 ELAPSED 575 REQUESTED_ITERS 1000\n')
    with pytest.raises(ValueError, match='incomplete or failed'):
        estimate(tmp_path, 195000)


def test_scientific_audit_is_not_a_manifest(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location('deckard_review', SCRIPTS / 'review_deckard.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for k in module.KS:
        directory = tmp_path / f'mqar2_{k}_q4-lin-s2718'
        directory.mkdir()
        result = dict(dataset=f'mqar2_{k}_q4', episodes=128, targets=512,
                      gate=dict(checks=dict(scientific_test=True)))
        documents = {
            'analysis.json': dict(split='validation', iteration=200000, code_commit='fixture',
                                  results=[result], elapsed_seconds=40),
            'pilot_run.json': dict(recipe=module.RECIPE, train_exit=0, code_commit='fixture',
                                   training_seconds=33000),
            'learning_curve.json': dict(split='validation', points=[
                dict(iteration=195000, strict_recall=.49), dict(iteration=200000, strict_recall=.50)])}
        for name, value in documents.items():
            (directory / name).write_text(json.dumps(value))
        (directory / 'train.log').write_text('TRAIN EXIT 0\n')
    result = module.review(tmp_path)
    assert result['scientific_candidate'] == 4
    assert result['launch_approved'] is False
    assert 'regime_verified' not in result
    assert result['rows'][0]['iterations'] == 195000
