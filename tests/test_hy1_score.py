import string
import numpy as np
import pytest
from hy1_score import (word_annotations,nonce_transform,swap_transform,sufficient,
                       aggregate,paired_bootstrap,discover_checkpoints)


def test_target_alignment():
    masks,spans = word_annotations('cat cat!')
    assert spans[1]['gap']==4
    assert np.flatnonzero(masks['nonrepeat']).tolist()==[0,1]
    assert np.flatnonzero(masks['lt64']).tolist()==[3,4,5]
    assert not masks['nonrepeat'][2]


@pytest.mark.parametrize('gap,key',[(63,'lt64'),(64,'64_127'),(127,'64_127'),(128,'128_255'),(255,'128_255'),(256,'ge256')])
def test_boundaries(gap,key):
    masks,spans = word_annotations('cat'+' '*(gap-3)+'CAT')
    assert spans[1]['gap']==gap
    assert masks[key].sum()==3
    assert masks['beyond_W'].sum()==(3 if gap>=64 else 0)


def test_nonce_deterministic_and_exact():
    text = 'Alice met Alice, not Malice. Alice smiled.'
    chars = string.ascii_letters+string.punctuation+' '
    a,meta = nonce_transform(text,chars,{'alice','xxxxx'},12)
    b,_ = nonce_transform(text,chars,{'alice','xxxxx'},12)
    assert a==b and len(a)==len(text) and set(a)<=set(chars)
    assert meta['source']=='Alice' and meta['nonce'].lower() not in {'alice','xxxxx'}
    assert a.count(meta['nonce'])==3 and 'Malice' in a
    assert 'no POS model' in meta['heuristic']


def test_nonce_missing_disclosed():
    assert nonce_transform('a cat ran.',string.ascii_letters+' .')[0] is None


def test_swap():
    text = 'a pear and a leaf'
    changed = swap_transform(text,7)
    assert changed[:7]==text[:7]
    assert changed[7:]==text[7:].translate(str.maketrans('ea','ae'))
    assert swap_transform(swap_transform(text))==text


def test_paired_bootstrap():
    a = [sufficient([1,1]),sufficient([4])]
    b = [sufficient([0,0]),sufficient([3])]
    assert aggregate(a)['nats_per_char']==2
    result = paired_bootstrap(a,b,100)
    assert result['delta']==1 and result['ci95']==[1,1]
    assert paired_bootstrap([sufficient([])],[sufficient([])])['delta'] is None


def test_discovery(tmp_path):
    arm = tmp_path/'A1'
    arm.mkdir()
    for name in ('midpoint','final','latest'):
        (arm/(name+'.pt')).touch()
    assert set(discover_checkpoints(tmp_path))=={('A1','midpoint'),('A1','final')}


def test_contrast_missing_support_and_direction():
    from hy1_score import bootstrap_contrast
    first = [sufficient([3,3]),sufficient([10])]
    late = [sufficient([1,1]),sufficient([])]
    result = bootstrap_contrast([first,late],[1,-1],50)
    assert result['estimate']==2 and result['ci95']==[2,2] and result['stories']==1


@pytest.mark.parametrize('arm',['A0','A1','A2','A3'])
def test_real_tiny_checkpoint(tmp_path,arm):
    import torch
    from dataclasses import asdict
    from hy1_model import HY1Config,HY1Model
    from hy1_score import ScoringAdapter,slow_digest,score_checkpoint
    torch.set_num_threads(1)
    chars = string.ascii_letters+' .'
    cfg = HY1Config(vocab_size=len(chars),arm=arm,d_model=8,n_layers=2,
                    n_heads=2,hidden_dim=16,window=4,chunk_size=4)
    model = HY1Model(cfg)
    path = tmp_path/'tiny.pt'
    torch.save(dict(config=asdict(cfg),model=model.state_dict(),chars=chars),path)
    adapter = ScoringAdapter(path,chars,'cpu')
    text = 'Alice saw Alice. Alice ate a pear.'
    before = slow_digest(adapter.model)
    losses = adapter.score(text)
    assert len(losses)==len(text)-1 and np.isfinite(losses).all()
    if arm=='A1':
        def forbidden(*args,**kwargs):
            raise AssertionError('no-fast MUST NOT WRITE')
        commit = adapter.model.commit
        adapter.model.commit = forbidden
        off = adapter.score(text,nofast=True)
        assert np.isfinite(off).all() and not adapter.state.fast.any()
        adapter.model.commit = commit
    scored = score_checkpoint(adapter,[text]*5,chars,{'alice','pear'},arm)
    assert scored['slow_state']['unchanged'] and slow_digest(adapter.model)==before
    assert len(scored['C3'])==1
    if arm!='A1':
        assert scored['C3'][0]['swap_carry']==scored['C3'][0]['swap_reset']


def test_cli_end_to_end_cpu(tmp_path):
    import json
    import torch
    from dataclasses import asdict
    from hy1_model import HY1Config,HY1Model
    from hy1_score import main
    torch.set_num_threads(1)
    chars = string.ascii_letters+' .'
    text = 'Alice '+('ate a pear. '*7)+'Alice '+('ate a pear. '*7)+'Alice ate.'
    data = tmp_path/'data.pt'
    torch.save(dict(chars=chars,train=[text],validation=[text]*5),data)
    for arm in ('A0','A1','A2','A3'):
        cfg = HY1Config(vocab_size=len(chars),arm=arm,d_model=8,n_layers=2,n_heads=2,hidden_dim=16)
        model = HY1Model(cfg)
        run = tmp_path/'runs'/arm
        run.mkdir(parents=True)
        for stage in ('midpoint','final'):
            torch.save(dict(config=asdict(cfg),model=model.state_dict(),chars=chars),run/(stage+'.pt'))
    main(['--runs',str(tmp_path/'runs'),'--data',str(data),'--out',str(tmp_path/'out'),
          '--device','cpu','--stories','5','--bootstrap','10'])
    output = json.loads((tmp_path/'out'/'metrics.json').read_text())
    assert len(output['checkpoints'])==8
    assert output['checkpoints']['A1/final']['slow_state']['unchanged']
    assert output['exploratory_endpoints']['final']['C1_A1_recovery']['stories']==5
    assert (tmp_path/'out'/'final_distance_C1_C2.png').exists()
    assert (tmp_path/'out'/'midpoint_distance_C1_C2.png').exists()
