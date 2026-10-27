"""CPU-only tests of the frozen regime gate and manual launch interlock."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bigkey_gate", ROOT / "scratch/k13_bigkey_2026-10-27/gate.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def data(k=4):
    wrong = dict(count=100, quantiles={"p50": .1}, abs_median=.2,
                 abs_quantiles={"p50": .2})
    layers = {f"L{i}": {"items": {"key_pr_pooled": pr, "key_pr": .1,
                                       "wrong_cosines": copy.deepcopy(wrong)}}
              for i, pr in enumerate((4., 2.))}
    analysis = dict(split="validation", iteration=20001, code_commit="abc",
                    results=[dict(dataset=f"mqar2_{k}_q4", episodes=128,
                                  acc=dict(strict=.5, strict_first=.5, post_store=.9,
                                           store_only=.95, paired_gap=.9, observed=.6, no_fast=.1),
                                  layers=layers)])
    curve = dict(split="validation", points=[dict(iteration=i, strict_recall=r)
                                             for i, r in ((5000, .2), (10000, .48),
                                                          (15000, .49), (20000, .5))])
    run = dict(dataset=f"mqar2_{k}_q4", arm="lin", seed=2718, requested_iterations=200000,
               recipe=copy.deepcopy(gate.RECIPE), status="timeout", train_exit=124,
               training_seconds=4500, code_commit="abc")
    return analysis, curve, run


def test_pooled_not_episode_pr_and_stable_budget():
    result = gate.candidate(*data(), 4)
    assert result["qualifies"]
    assert result["n_iters"] == 10000
    analysis, curve, run = data()
    analysis["results"][0]["layers"]["L0"]["items"]["key_pr_pooled"] = 3.99
    assert not gate.candidate(analysis, curve, run, 4)["qualifies"]


@pytest.mark.parametrize("metric,value", [("strict", .19), ("strict", .851),
                                           ("strict_first", .19), ("strict_first", .851),
                                           ("post_store", .899)])
def test_recall_thresholds(metric, value):
    analysis, curve, run = data()
    analysis["results"][0]["acc"][metric] = value
    assert not gate.candidate(analysis, curve, run, 4)["qualifies"]


def test_absolute_improvement_and_fraction_are_both_required():
    analysis, curve, run = data()
    analysis["results"][0]["acc"].update(strict_first=.2, post_store=.31)
    result = gate.candidate(analysis, curve, run, 4)
    assert result["checks"]["post_improvement"]
    assert not result["checks"]["error_recovered"]


@pytest.mark.parametrize("change", ["test", "short", "missing_pr", "missing_abs", "failed", "recipe", "commit"])
def test_missing_or_wrong_evidence_fails_closed(change):
    analysis, curve, run = data()
    if change == "test": analysis["split"] = "test"
    if change == "short": analysis["results"][0]["episodes"] = 127
    if change == "missing_pr": del analysis["results"][0]["layers"]["L0"]["items"]["key_pr_pooled"]
    if change == "missing_abs": del analysis["results"][0]["layers"]["L0"]["items"]["wrong_cosines"]["abs_quantiles"]
    if change == "failed": run["status"] = "failed"
    if change == "recipe": run["recipe"]["layers"] = [1, 2]
    if change == "commit": analysis["code_commit"] = "other"
    with pytest.raises((ValueError, KeyError)):
        gate.candidate(analysis, curve, run, 4)


def test_unstable_budget_blocks_and_noncheckpoint_points_are_refused():
    analysis, curve, run = data()
    curve["points"][-2]["strict_recall"] = .46
    assert not gate.candidate(analysis, curve, run, 4)["qualifies"]
    curve["points"][-2]["strict_recall"] = .49
    curve["points"][0]["iteration"] = 5001
    with pytest.raises(ValueError): gate.stable_budget(curve, 20001)
    curve["points"][0]["iteration"] = 5000
    with pytest.raises(ValueError): gate.stable_budget(curve, 40000)


@pytest.fixture
def pilot(tmp_path):
    for k in gate.KS:
        d = tmp_path / f"mqar2_{k}_q4-lin-s2718"
        d.mkdir()
        for name, content in zip(("analysis.json", "learning_curve.json", "pilot_run.json"), data(k)):
            (d / name).write_text(json.dumps(content))
        (d / "train.log").write_text("TRAIN EXIT 124\n")
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text('{"measured":true}')
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps(dict(arms={arm: dict(seconds_per_iter=.1, scoring_seconds=60,
                                                      basis="conservative_estimate", explanation="lin pilot throughput x3",
                                                      evidence=str(benchmark)) for arm in gate.ARMS})))
    return tmp_path, runtime


def test_build_smallest_k_approval_and_evidence_integrity(pilot):
    root, runtime = pilot
    m = gate.build(root, runtime)
    assert m["dataset"] == "mqar2_4_q4" and not m["launch_approved"]
    with pytest.raises(ValueError): gate.validate_manifest(m)
    with pytest.raises(ValueError): gate.build(root, runtime, True, "some approval")
    m = gate.build(root, runtime, True, gate.APPROVAL)
    assert gate.validate_manifest(m, m["dataset"], m["n_iters"], "spar", 3141)
    for kwargs in (dict(dataset="mqar2_32_q4"), dict(iterations=200000),
                   dict(arm="other"), dict(seed=123)):
        with pytest.raises(ValueError): gate.validate_manifest(m, **kwargs)
    changed = copy.deepcopy(m); changed["estimated_a100_hours"] = 0
    with pytest.raises(ValueError): gate.validate_manifest(changed)
    (root / "mqar2_4_q4-lin-s2718/analysis.json").write_text("{}")
    with pytest.raises(ValueError): gate.validate_manifest(m)


def test_select_smallest_qualifying_k_and_refuse_missing_pilot(pilot):
    root, runtime = pilot
    path = root / "mqar2_4_q4-lin-s2718/analysis.json"
    a = json.loads(path.read_text()); a["results"][0]["acc"]["strict"] = .1
    path.write_text(json.dumps(a))
    assert gate.build(root, runtime)["dataset"] == "mqar2_8_q4"
    path.unlink()
    with pytest.raises(OSError): gate.build(root, runtime)


def test_budget_includes_slowest_arm_scoring_and_all_25_runs(pilot):
    root, runtime = pilot
    costs = json.loads(runtime.read_text())
    costs["arms"]["spar"]["scoring_seconds"] = 6000
    runtime.write_text(json.dumps(costs))
    with pytest.raises(ValueError, match="40 A100-hours"): gate.build(root, runtime)
    costs["arms"].pop("spar"); runtime.write_text(json.dumps(costs))
    with pytest.raises(ValueError, match="all five"): gate.build(root, runtime)


def test_allocation_budget_not_only_runtime_estimate(pilot):
    root, runtime = pilot
    costs = json.loads(runtime.read_text())
    for arm in costs["arms"].values():
        arm["seconds_per_iter"] = .5
    runtime.write_text(json.dumps(costs))
    assert gate.estimate_runtime(costs, 10000)[0] < 40
    with pytest.raises(ValueError, match="allocated walltime"): gate.build(root, runtime)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, True])
def test_runtime_nonfinite_or_invalid_refused(pilot, bad):
    root, runtime = pilot
    costs = json.loads(runtime.read_text())
    costs["arms"]["spar"]["seconds_per_iter"] = bad
    runtime.write_text(json.dumps(costs))
    with pytest.raises(ValueError): gate.build(root, runtime)


def test_unexplained_runtime_estimate_is_not_evidence(pilot):
    root, runtime = pilot
    costs = json.loads(runtime.read_text())
    del costs["arms"]["spar"]["explanation"]
    runtime.write_text(json.dumps(costs))
    with pytest.raises(ValueError): gate.build(root, runtime)


def test_train_exit_marker_and_failed_rebuild_clear_stale_approval(pilot):
    root, runtime = pilot
    (root / "mqar2_4_q4-lin-s2718/train.log").write_text("no exit marker")
    out = root / "manifest.json"
    out.write_text('{"launch_approved":true}')
    assert gate.main(["build", "--pilot-root", str(root), "--runtime-estimate", str(runtime),
                      "--out", str(out)]) == 2
    assert json.loads(out.read_text())["launch_approved"] is False


def test_bash_syntax():
    for filename in ("train.sh", "pilot.sbatch", "confirm.sbatch"):
        subprocess.run(["bash", "-n", str(ROOT / "scratch/k13_bigkey_2026-10-27" / filename)], check=True)


@pytest.mark.parametrize("filename", ["pilot.sbatch", "confirm.sbatch"])
def test_conda_activation_allows_unset_mkl_environment(tmp_path, filename):
    # Reproduce ORC's activate.d hook without sourcing real workstation credentials.
    (tmp_path / '.bashrc').write_text('conda() { echo "$MKL_INTERFACE_LAYER"; }\n')
    script = (ROOT / 'scratch/k13_bigkey_2026-10-27' / filename).read_text()
    activation = script.split(': "${K13B_CODE', 1)[0]
    env = dict(os.environ, HOME=str(tmp_path))
    env.pop('MKL_INTERFACE_LAYER', None)
    subprocess.run(['bash', '-u', '-c', activation], env=env, check=True, capture_output=True)
