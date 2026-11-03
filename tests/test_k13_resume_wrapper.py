"""K13 train.sh resume path (addendum 2026-11-02), with train.py, heldout.py, analyze.py, git and
nvidia-smi replaced by stubs so only the wrapper's bookkeeping runs."""
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SH = ROOT / "scratch/k13_bigkey_2026-10-27/train.sh"

STUB_PYTHON = r'''#!/usr/bin/env bash
# train.py: log one curve point, write a checkpoint, then exit with $FAKE_RC (consumed per call).
if [[ $1 == -u && $2 == train.py ]]; then
  printf '%s\n' "$*" >> "$FAKE_CALLS"
  dir=$(printf '%s\n' "$@" | grep -A1 -x -- --checkpoint_dir | tail -1)
  printf 'Interval metrics (ending @ iter 5000\n  heldout_strict/recall_acc: 0.25\n'
  touch "$dir/latest_checkpoint.pth"
  rc=$(head -n1 "$FAKE_RC"); sed -i 1d "$FAKE_RC"; exit "${rc:-0}"
fi
if [[ $1 == -u && $2 == heldout.py ]]; then exit 0; fi
if [[ $1 == -u && $2 == */analyze.py ]]; then exit 0; fi
if [[ $1 == */gate.py ]]; then exit 0; fi
exec "$REAL_PYTHON" "$@"
'''


def make_stubs(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stubs = {
        "python": STUB_PYTHON,
        "git": '#!/usr/bin/env bash\n[[ $1 == rev-parse ]] && echo abc123\nexit 0\n',
        "nvidia-smi": '#!/usr/bin/env bash\necho "NVIDIA RTX A6000"\n',
    }
    for name, body in stubs.items():
        path = bin_dir / name
        path.write_text(body)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"training_timeout_seconds": 1000, "scoring_timeout_seconds": 100}))
    return bin_dir, manifest


def run_wrapper(tmp_path, bin_dir, manifest, rcs, resume=True):
    (tmp_path / "rcs").write_text("".join(f"{rc}\n" for rc in rcs))
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", HOME=str(tmp_path),
               REAL_PYTHON=sys.executable, FAKE_CALLS=str(tmp_path / "calls"), FAKE_RC=str(tmp_path / "rcs"),
               K13B_CODE=str(ROOT), K13B_COMMIT="abc123", K13B_GPU="A6000", K13B_PILOT="0",
               K13B_GATE_MANIFEST=str(manifest), K13B_RESUME="1" if resume else "0")
    return subprocess.run(["bash", str(TRAIN_SH), "mqar2_16_q4", "195000", "spar", "2718", str(tmp_path / "out")],
                          env=env, capture_output=True, text=True)


def test_failed_run_resumes_from_its_checkpoint_and_completes(tmp_path):
    bin_dir, manifest = make_stubs(tmp_path)
    run_dir = tmp_path / "out/mqar2_16_q4-spar-s2718"
    assert run_wrapper(tmp_path, bin_dir, manifest, [1]).returncode == 1
    second = run_wrapper(tmp_path, bin_dir, manifest, [0])
    assert second.returncode == 0, second.stderr
    calls = (tmp_path / "calls").read_text().splitlines()
    assert "--resume_checkpoint" not in calls[0]
    assert f"--resume_checkpoint {run_dir}/latest_checkpoint.pth" in calls[1]
    assert "--min_free_disk_gb" in calls[1]
    log = (run_dir / "train.log").read_text()
    assert log.count("TRAIN EXIT") == 2 and log.rstrip().endswith("EVAL EXIT 0")
    run = json.loads((run_dir / "pilot_run.json").read_text())
    assert (run["status"], run["resumes"]) == ("completed", 1)
    curve = json.loads((run_dir / "learning_curve.json").read_text())["points"]
    assert curve == [{"iteration": 5000, "strict_recall": 0.25}]  # re-logged point kept once
    # A completed run is never trained again.
    assert run_wrapper(tmp_path, bin_dir, manifest, []).returncode == 0
    assert len((tmp_path / "calls").read_text().splitlines()) == 2


def test_existing_run_is_refused_without_resume_or_with_another_manifest(tmp_path):
    bin_dir, manifest = make_stubs(tmp_path)
    assert run_wrapper(tmp_path, bin_dir, manifest, [1]).returncode == 1
    refused = run_wrapper(tmp_path, bin_dir, manifest, [0], resume=False)
    assert refused.returncode == 2 and "Refusing existing run directory" in refused.stderr
    manifest.write_text(json.dumps({"training_timeout_seconds": 999, "scoring_timeout_seconds": 100}))
    other = run_wrapper(tmp_path, bin_dir, manifest, [0])
    assert other.returncode == 2 and "another gate manifest" in other.stderr


def test_training_budget_is_summed_over_segments(tmp_path):
    bin_dir, manifest = make_stubs(tmp_path)
    run_dir = tmp_path / "out/mqar2_16_q4-spar-s2718"
    assert run_wrapper(tmp_path, bin_dir, manifest, [1]).returncode == 1
    (run_dir / "train_seconds").write_text("1000\n")
    spent = run_wrapper(tmp_path, bin_dir, manifest, [0])
    assert spent.returncode == 124
    assert len((tmp_path / "calls").read_text().splitlines()) == 1
    assert json.loads((run_dir / "pilot_run.json").read_text())["status"] == "timeout"
