import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("k13_screen", ROOT / "scratch/k13_crosstalk_2026-10-27/screen.py")
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)


def write_screen(root, pr):
    for d in (0, 4, 8, 16, 32):
        result = {"acc": {"all": 1. if d == 0 else .6},
                  "layers": {layer: {"key_pr": pr, "cos_oth_abs": .4} for layer in ("L1", "L2")}}
        (root / f"load{d}.json").write_text(json.dumps(result))
        for mode in ("gap", "selective"):
            (root / f"{mode}{d}.json").write_text(json.dumps({"acc": {"all": .9}}))


def test_regime_gate_cannot_authorize_launch(tmp_path):
    write_screen(tmp_path, 2.)
    result = screen.assess(tmp_path)
    assert result["regime_verified"] and result["selected_load"] == 4
    assert not result["launch_approved"] and result["n_iters"] is None


def test_collinear_keys_fail_gate(tmp_path):
    write_screen(tmp_path, 1.3)
    result = screen.assess(tmp_path)
    assert not result["regime_verified"] and result["selected_load"] is None
