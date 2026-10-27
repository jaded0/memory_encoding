"""P3 preserves SG's control metrics and separates raw value from applied gain."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sg = load_script("p3_sg_test", "scratch/p3_2026-10-27/p3_sg.py")
old = load_script("original_sg_test", "scratch/cheap_checks_2026-10-16/surprise_gating.py")


def row(power=0):
    # Store is s=1; no zero norms or key factors. Applied scalar gains preserve directions.
    raw_norms = np.array([1., 2., 1.5, .5])
    gain = raw_norms ** power
    pn = raw_norms * gain
    key = np.array([.8, 3., .4, .2])
    return {"s": 1, "q": 4, "pn": pn, "rn": raw_norms, "xn": np.ones(4),
            "cos": np.array([.1, .9, .2, .05]), "raw": torch.tensor((pn * key)[:, None]),
            "exps": np.arange(4, 0, -1), "c": .3, "text": "0?v0!v"}


def test_control_is_original_salience_decomposition():
    r = row()
    before = old.summarize([r], .99, 0, 1e-8, .01)
    after = sg.summarize([r], .99, 0, 1e-8, .01)
    for key in before:
        assert after[key] == before[key], key
    assert after["ratio_gain"] == 1
    assert after["log_gain_part"] == 0
    assert after["store_largest_no_gain"] == after["store_largest"]


@pytest.mark.parametrize("power", [1, 2, 4])
def test_gained_value_and_log_split(power):
    # Other raw norms are .5, 1, 1.5, whose median is exactly 1.
    base = sg.summarize([row()], .99, 0, 0, .01)
    result = sg.summarize([row(power)], .99, 0, 0, .01)
    assert result["ratio_p_raw"] == 2
    assert result["ratio_p"] == 2 ** (power + 1)
    assert result["ratio_gain"] == 2 ** power
    assert result["ratio_g"] == pytest.approx(base["ratio_g"])
    assert result["log_value_part"] == pytest.approx(base["log_value_part"] + result["log_gain_part"])
    assert result["log_lead_mean"] == pytest.approx(
        result["log_value_part"] + result["log_key_part"] + result["log_decay_part"])
    assert result["store_largest_no_gain"] == base["store_largest"]
