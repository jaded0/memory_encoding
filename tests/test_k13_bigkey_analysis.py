"""Small CPU-only tests; no pilot checkpoints or GPU required."""
import copy
import importlib.util
from pathlib import Path
import random

import numpy as np
import pytest
import torch

import bench_tasks
from ephemeral_model import EphemeralRNN, order2_projection
from heldout import evaluate_held_out, update_mask
from train import positional_encoding

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("k13_bigkey_analysis", ROOT / "scratch/k13_bigkey_2026-10-27/analyze.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def fixture_data():
    name = "mqar2_4_q4"
    rng = random.Random(9)
    texts = [bench_tasks.generate_sample(rng, *bench_tasks.parse_name(name)) for _ in range(2)]
    return name, texts, analysis.encode(texts, bench_tasks.BIGKEY_CHARSET, "cpu")


def test_metadata_and_paired_gap_use_value_index_not_capped_lag():
    name = "mqar2_32_q4"
    text = bench_tasks.generate_sample(random.Random(4), *bench_tasks.parse_name(name))
    answers = analysis.metadata(text, name)
    for answer, source in answers:
        assert source % 3 == 2
        assert text[source - 2:source] == text[answer - 2:answer]
    gap = analysis.paired_gap(text, name)
    first_answer, first_source = answers[0]
    assert len(gap) == len(text)
    assert gap[96:] == text[96:]
    assert gap[:96].count(".") == 93
    assert gap[first_source - 2:first_source + 1] == text[first_source - 2:first_source + 1]
    assert gap[first_answer] == text[first_answer]


def test_masks_include_store_and_differ_before_store():
    name, texts, onehot = fixture_data()
    strict = update_mask("strict", texts, name, onehot)
    post = analysis.suppression_mask(texts, name, onehot, "post_store")
    only = analysis.suppression_mask(texts, name, onehot, "store_only")
    for b, text in enumerate(texts):
        answer, source = analysis.metadata(text, name)[0]
        assert strict[b, answer - 2] and not strict[b, answer - 1]
        assert post[b, source - 1] and not post[b, source:].any()
        assert int(post[b].sum()) == source
        assert only[b, source - 1] and int(only[b].sum()) == 1


def test_pr_uses_uncentered_unit_keys_and_full_wrong_distribution():
    x = torch.eye(4)
    result = analysis.geometry(x, x[0], 0)
    assert result["key_pr"] == 4
    assert result["store_cosine"] == 1
    assert result["wrong_cosines"] == [0, 0, 0]
    assert analysis.geometry(torch.ones(4, 3), torch.ones(3), 0)["key_pr"] == pytest.approx(1)
    assert analysis.distribution([])["count"] == 0


def test_first_answer_gate_and_no_launch_authorization():
    layers = {"L0": {"items": {"key_pr_pooled": 4}}, "L1": {"items": {"key_pr_pooled": 2}}}
    ok = analysis.gate(.5, .5, .9, layers)
    assert ok["regime_verified"] and not ok["launch_approved"]
    assert not analysis.gate(.5, .5, .89, layers)["regime_verified"]
    assert not analysis.gate(.9, .5, 1., layers)["regime_verified"]
    assert not analysis.gate(.5, .9, 1., layers)["regime_verified"]
    assert not analysis.gate(.5, .5, 1., {})["regime_verified"]
    assert not analysis.gate(.5, .5, 1., layers, episodes=16)["regime_verified"]
    low = copy.deepcopy(layers)
    low["L0"]["items"]["key_pr_pooled"] = 3.99
    assert not analysis.gate(.5, .5, 1., low)["regime_verified"]


def test_full_keyset_pooled_pr_is_18_18_not_per_episode_bound():
    # Two disjoint one-hot blocks: the 100 addresses span rank19, PR=200/11.
    keys = torch.stack([torch.cat((torch.eye(10)[i], torch.eye(10)[j]))
                        for i in range(10) for j in range(10)])
    pool = {}
    for episode in keys.split(4):
        analysis.accumulate_pooled(pool, "L0", episode)
    p = pool["L0"]
    assert p["stores"] == 100 and len(p["per_episode_pr"]) == 25
    assert analysis.pooled_pr(p["second_moment_sum"]) == pytest.approx(200 / 11)
    assert int(torch.linalg.matrix_rank(p["second_moment_sum"])) == 19


@pytest.mark.parametrize("repeats", [1, 2])
def test_replay_matches_canonical_heldout_and_does_not_change_slow_weights(repeats):
    torch.set_num_threads(1)
    name, texts, onehot = fixture_data()
    torch.manual_seed(17)
    model = EphemeralRNN(48, 8, 23, 3, bench_tasks.BIGKEY_CHARSET, batch_size=2,
                         updater="dfa", ephemeral_fraction=.5, forget_rate=.1,
                         enable_recurrence=True, fast_backward_per_forward=repeats)
    model.eval()
    slow_before = [(l.per_sample_weights[:, ~l.ephemeral_mask].clone(), l.bias.clone())
                   for l in model.trained_layers()]
    reference = copy.deepcopy(model)
    config = {"input_mode": "last_two", "pe_matrix": positional_encoding(2, "cpu"),
              "learning_rate": .1, "ephemeral_update_clamp": .02, "grad_norm_clip": 0,
              "dataset": name, "seed": 17}
    mask = update_mask("strict", texts, name, onehot)
    expected, _ = evaluate_held_out(reference, onehot, mask, config)
    records = {}
    actual = analysis.strict_replay(model, config, texts, onehot, name, None, records)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for a, b in zip(model.trained_layers(), reference.trained_layers()):
        torch.testing.assert_close(a.per_sample_weights, b.per_sample_weights, rtol=0, atol=0)
    assert set(records) == {"L0", "L1", "L2"}
    for rows in records.values():
        assert len(rows) == 8
        assert all(len(r["items"]["wrong_cosines"]) == 3 for r in rows)
        assert all(r["items"]["key_pr"] <= 4 + 1e-10 for r in rows)
    result = analysis.analyze_dataset(model, config, [(texts, onehot)], name, kernel_screen=False)
    assert result["targets"] == 8 and result["episodes"] == 2
    assert result["acc"]["strict"] == np.mean(analysis.score(expected, onehot, texts, name)[0])
    assert result["kernels"] == {}
    for layer in result["layers"].values():
        assert layer["items"]["pooled_stores"] == 8  # not 8*Q
        assert layer["items"]["pooled_episodes"] == 2
        assert layer["items"]["key_pr"] == layer["items"]["key_pr_pooled"]
    assert all(k in result["acc"] for k in ("observed", "observed_first", "no_fast", "no_fast_first",
                                             "strict_exact", "observed_exact", "no_fast_exact"))
    for layer, (weights, bias) in zip(model.trained_layers(), slow_before):
        torch.testing.assert_close(layer.per_sample_weights[:, ~layer.ephemeral_mask], weights, rtol=0, atol=0)
        torch.testing.assert_close(layer.bias, bias, rtol=0, atol=0)


def test_standalone_gap_item_population_omits_blank_slots():
    name = "mqar2gap_32_q1"
    text = bench_tasks.generate_sample(random.Random(18), *bench_tasks.parse_name(name))
    onehot = analysis.encode([text], bench_tasks.BIGKEY_CHARSET, "cpu")
    model = EphemeralRNN(48, 4, 23, 3, bench_tasks.BIGKEY_CHARSET, batch_size=1,
                         updater="dfa", ephemeral_fraction=.5, forget_rate=.1)
    config = {"input_mode": "last_two", "pe_matrix": positional_encoding(2, "cpu"),
              "learning_rate": .1, "ephemeral_update_clamp": .02, "grad_norm_clip": 0}
    result = analysis.analyze_dataset(model, config, [([text], onehot)], name, kernel_screen=False)
    assert result["targets"] == 1
    for layer in result["layers"].values():
        assert layer["items"]["key_pr"] == pytest.approx(1)
        assert layer["items"]["wrong_cosines"]["count"] == 0


def test_sparse_observer_reuses_original_maps_and_can_observe_L0():
    torch.manual_seed(8)
    xs, xq = torch.randn(4, 6).double(), torch.randn(6).double()
    p = order2_projection(6, 16, 0, 0).double()
    old, old_rows = analysis.sparse.make_observer(analysis.o2)
    old(0, xs, xq, 0, [1, 2, 3], p, .5, 0)
    assert not old_rows
    observer, rows = analysis.sparse.make_observer(analysis.o2, layers=(0, 1, 2))
    before = xs.clone()
    observer(0, xs, xq, 0, [1, 2, 3], p, .5, 0)
    torch.testing.assert_close(xs, before, rtol=0, atol=0)
    assert len(rows["L0"]) == 8  # linear, o1, o2, three z16, two full sparse
    assert rows["L0"]["linear"][0]["wrong_cosines"] == pytest.approx(
        analysis.geometry(xs, xq, 0)["wrong_cosines"])
