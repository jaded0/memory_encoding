import random

import pytest

from kr_load_tasks import CHARSET, PAYLOAD, generate_sample, parse_name, registered_names
from metrics import recall_targets, recall_chance, MAX_LAG
from preprocess import dataset_keys as splits, is_synthetic
from utils import dataset_keys, get_charset


@pytest.mark.parametrize("name", registered_names())
def test_kr_load_contract(name):
    mode, d = parse_name(name)
    text = generate_sample(random.Random(13), mode, d)
    assert len(text) == d + 5
    assert text.count("?") == text.count("!") == 1
    assert text[2] == text[-1] in PAYLOAD
    assert text[3:-2] == "0" * d if mode == "gap" else set(text[3:-2]) <= set(PAYLOAD)
    assert recall_targets(text, name) == ({len(text) - 1: d + 1}, None)
    assert recall_chance(name) == .2
    assert d + 1 < MAX_LAG
    assert dataset_keys[name] == "text" and splits[name] == "train"
    assert is_synthetic(name)
    assert get_charset(name) == CHARSET == get_charset("long_range_memory_dataset")
    assert text == generate_sample(random.Random(13), mode, d)


def test_invalid():
    with pytest.raises(ValueError):
        generate_sample(random.Random(0), "load", -1)
    with pytest.raises(ValueError):
        parse_name("long_range_memory_dataset")
