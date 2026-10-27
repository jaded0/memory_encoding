"""Two-character addressing, heldout sources and unchanged legacy streams."""
import random

import pytest
import torch

import bench_tasks as bt
import preprocess
import utils
from heldout import update_mask
from metrics import recall_chance, recall_targets


def test_names_charset_registration():
    assert bt.parse_name('mqar2_32_q4') == ('mqar2', (32, 4))
    assert bt.parse_name('mqar2gap_32') == ('mqar2gap', (32, 1))
    for bad in ('mqar2_101', 'mqar2_4_q5', 'mqar2gap_0'):
        with pytest.raises(ValueError):
            bt.parse_name(bad)
    for name in ('mqar2_32_q4', 'mqar2gap_32_q1'):
        assert preprocess.is_synthetic(name)
        assert utils.get_charset(name) == bt.BIGKEY_CHARSET
        assert name in bt.registered_names()
        assert recall_chance(name) == .1


@pytest.mark.parametrize('name', ['mqar2_4_q4', 'mqar2_32_q4', 'mqar2gap_32_q1'])
def test_sources_splits_and_seed(name):
    k, q = bt.parse_name(name)[1]
    seen = []
    for split in ('train', 'validation', 'test'):
        texts = bt.generate_split(name, split, 100)
        assert texts == bt.generate_split(name, split, 100)
        assert texts != bt.generate_split(name, split, 100, seed=1)
        seen.append({bt.core(t) for t in texts})
        for text in texts:
            assert len(text) == 3*k + 4*q
            assert set(text) <= set(bt.BIGKEY_CHARSET)
            table = {text[i:i+2]: i+2 for i in range(0, 3*k, 3) if text[i:i+3] != '...'}
            assert len(table) == (q if 'gap' in name else k)
            answers = bt.episode(text, name)['answers']
            assert len(answers) == q
            for a, s in answers.items():
                assert s == table[text[a-2:a]]
                assert text[a] == text[s]
                assert recall_targets(text, name)[0][a] == min(a-s-1, 57)
            assert bt.episode(text+'   ', name) == bt.episode(text, name)
    assert not (seen[0] & seen[1] or seen[0] & seen[2] or seen[1] & seen[2])


def test_layer0_pair_overlap_and_adjacent_write_keys():
    # Last-two order is read from the actual implementation, not assumed.
    charset = bt.BIGKEY_CHARSET
    text = 'ab3ac7?ab3'
    onehot = torch.nn.functional.one_hot(torch.tensor([[charset.index(c) for c in text]]), len(charset)).float()
    key = lambda t: utils.model_input(onehot, t, 'last_two', None).flatten()
    cosine = lambda a,b: torch.nn.functional.cosine_similarity(a,b,dim=0).item()
    assert cosine(key(1),key(4)) == pytest.approx(.5)
    assert cosine(key(1),key(8)) == pytest.approx(1)
    assert cosine(key(1),key(2)) == pytest.approx(0)  # b3, after value
    assert cosine(key(1),key(3)) == pytest.approx(0)  # 3a, next first-letter
    assert cosine(key(1),key(7)) == pytest.approx(0)  # ?a before query pair
    mask = update_mask('strict', [text], 'mqar2_2_q1', onehot)
    assert mask[0,7] and not mask[0,8]


def test_legacy_generator_stream_unchanged():
    # Independent original algorithm and draw order, including random key/query selection.
    rng, ref = random.Random(42), random.Random(42)
    keys = ref.sample(bt.KEYS, 4)
    vals = [ref.choice(bt.VALUES) for _ in range(4)]
    expected = ''.join(k+v for k,v in zip(keys,vals))
    expected += ''.join('?'+keys[i]+vals[i] for i in ref.sample(range(4),4))
    assert bt.generate_sample(rng, 'mqar', (4,4)) == expected


def test_full_pair_set_has_partial_overlap_not_100_independent_dimensions():
    charset = bt.BIGKEY_CHARSET
    keys = []
    for pair in bt.BIGKEY_KEYS:
        hot = torch.nn.functional.one_hot(torch.tensor([[charset.index(c) for c in pair]]), len(charset)).double()
        keys.append(utils.model_input(hot, 1, 'last_two', None)[0])
    keys = torch.nn.functional.normalize(torch.stack(keys), dim=1)
    gram = keys @ keys.T
    assert torch.linalg.matrix_rank(keys).item() == 19
    assert (gram.trace().square()/gram.square().sum()).item() == pytest.approx(200/11)
    wrong = gram[~torch.eye(100, dtype=torch.bool)]
    assert int((wrong > .1).sum()) == 100*18  # nine same-slot alternatives in each slot
    assert torch.allclose(wrong[wrong > .1], torch.full_like(wrong[wrong > .1], .5))


def test_repeated_letter_keys_receive_adjacent_character_write_overlap():
    charset = bt.BIGKEY_CHARSET
    text = 'aa3ab7?aa3'
    hot = torch.nn.functional.one_hot(torch.tensor([[charset.index(c) for c in text]]), len(charset)).float()
    key = lambda t: utils.model_input(hot, t, 'last_two', None)[0]
    cos = lambda t: torch.nn.functional.cosine_similarity(key(1), key(t), dim=0).item()
    assert cos(2) == pytest.approx(.5)  # digit step shares preceding a
    assert cos(3) == pytest.approx(.5)  # next first letter a shares current slot
    assert cos(6) == pytest.approx(0)   # '?' after digit
    assert cos(7) == pytest.approx(.5)  # query's first a after '?'
    assert cos(8) == pytest.approx(1)   # complete query aa
