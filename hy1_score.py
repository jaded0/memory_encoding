"""HY1 heldout probes. Per-story sums/counts support paired document bootstrap."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import string
import numpy as np
import torch

DISTANCE = ('lt64', '64_127', '128_255', 'ge256')
WORD = re.compile(r'[A-Za-z]+')
STOP = set('the a an and or but he she it they his her their was were is are to of in on at for with that this one had said little'.split())


def word_annotations(text):
    masks = {k: np.zeros(max(0,len(text)-1),dtype=bool) for k in (*DISTANCE,'nonrepeat','within_W','beyond_W','ge128')}
    previous, occurrences, spans = {}, Counter(), []
    for m in WORD.finditer(text):
        key = m.group().lower()
        gap = m.start()-previous[key] if key in previous else None
        occurrences[key] += 1
        spans.append(dict(word=m.group(),start=m.start(),end=m.end(),gap=gap,occurrence=occurrences[key]))
        lo,hi = max(1,m.start())-1,m.end()-1
        if gap is None:
            masks['nonrepeat'][lo:hi] = True
        else:
            bucket = DISTANCE[0 if gap<64 else 1 if gap<128 else 2 if gap<256 else 3]
            masks[bucket][lo:hi] = True
            masks['within_W' if gap<64 else 'beyond_W'][lo:hi] = True
            if gap>=128:
                masks['ge128'][lo:hi] = True
        previous[key] = m.start()
    return masks,spans


def nonce_transform(text,chars,vocabulary=(),seed=0):
    matches = list(WORD.finditer(text))
    counts = Counter(m.group() for m in matches)
    candidates = [w for w,n in counts.items() if n>=2 and len(w)>=3 and w.lower() not in STOP]
    names = [w for w in candidates if w[0].isupper()]
    if not candidates:
        return None,dict(skipped='no repeated eligible name/content-word heuristic candidate')
    source = sorted(names or candidates,key=lambda w:(-counts[w],-len(w),w))[0]
    alphabet = sorted(set(chars)&set(string.ascii_lowercase))
    if len(alphabet)<2:
        return None,dict(skipped='charset has fewer than two lowercase letters')
    forbidden = set(vocabulary)|{m.group().lower() for m in matches}|STOP
    rng = random.Random(seed)
    for _ in range(10000):
        nonce = ''.join(rng.choice(alphabet) for _ in source)
        if source[0].isupper() and nonce[0].upper() in chars:
            nonce = nonce[0].upper()+nonce[1:]
        if nonce.lower() not in forbidden and nonce!=source:
            transformed = WORD.sub(lambda m: nonce if m.group()==source else m.group(),text)
            assert len(transformed)==len(text) and set(transformed)<=set(chars)
            return transformed,dict(source=source,nonce=nonce,occurrences=counts[source],heuristic='frequent capitalized token else non-stopword; no POS model')
    return None,dict(skipped='could not generate absent nonce')


def swap_transform(text,start=0):
    return text[:start]+text[start:].translate(str.maketrans('ea','ae'))


def sufficient(loss,mask=None):
    values = np.asarray(loss,dtype=np.float64)
    if mask is not None:
        values = values[mask]
    return dict(sum=float(values.sum()),count=int(values.size),nats_per_char=float(values.mean()) if values.size else None)


def aggregate(records):
    total = sum(r['sum'] for r in records)
    count = sum(r['count'] for r in records)
    return dict(sum=total,count=count,nats_per_char=total/count if count else None)


def paired_bootstrap(left,right,repetitions=2000,seed=20261106):
    if len(left)!=len(right):
        raise ValueError('unpaired records')
    a = np.array([[x['sum'],x['count'],y['sum'],y['count']] for x,y in zip(left,right)],dtype=float).reshape(-1,4)
    a = a[(a[:,1]>0)&(a[:,3]>0)]
    if not len(a):
        return dict(delta=None,ci95=None,stories=0)
    def delta(z):
        return z[...,0]/z[...,1]-z[...,2]/z[...,3]
    rng = np.random.default_rng(seed)
    draws = a[rng.integers(len(a),size=(repetitions,len(a)))].sum(axis=1)
    return dict(delta=float(delta(a.sum(axis=0))),ci95=np.quantile(delta(draws),[.025,.975]).tolist(),stories=len(a))


def slow_digest(model):
    h = hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class ScoringAdapter:
    def __init__(self,checkpoint,chars,device):
        from hy1_model import load_model
        self.model,self.payload = load_model(checkpoint,device=device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.chars = chars
        if self.payload.get('chars',chars)!=chars:
            raise ValueError('checkpoint charset differs from data')
        self.state = None

    def reset(self):
        self.state = None

    @torch.no_grad()
    def score(self,text,carry=False,nofast=False):
        from hy1_model import evaluate_document, HY1State
        # Encode directly: transformed positions must not be re-filtered or shifted.
        stoi = {c:i for i,c in enumerate(self.chars)}
        tokens = torch.tensor([stoi[c] for c in text],dtype=torch.long)
        if nofast:
            # Strict S2: zero reads AND do not call commit/write at all. Keep KV.
            import torch.nn.functional as F
            tokens = tokens.to(self.model.embedding.weight.device)
            state = self.model.new_state()
            pieces = []
            for start in range(0,len(tokens)-1,self.model.config.chunk_size):
                if state.fast is not None:
                    state = HY1State(torch.zeros_like(state.fast),state.cache,state.position)
                x = tokens[start:min(start+self.model.config.chunk_size,len(tokens)-1)]
                y = tokens[start+1:start+1+len(x)]
                logits,state,_ = self.model.forward_chunk(x,state)
                pieces.append(F.cross_entropy(logits,y,reduction='none').cpu())
            losses = torch.cat(pieces) if pieces else torch.empty(0)
            self.state = state
        else:
            losses,self.state = evaluate_document(self.model,tokens,chars=self.chars,
                state=self.state if carry else None)
        values = losses.detach().cpu().numpy()
        assert len(values)==len(text)-1
        return values


def discover_checkpoints(root):
    found = {}
    for p in sorted(Path(root).rglob('*.pt')):
        arms = [a for a in ('A0','A1','A2','A3') if a in str(p)]
        if not arms:
            continue
        label = 'midpoint' if any(t in p.stem.lower() for t in ('mid','half')) else 'final' if 'final' in p.stem.lower() else None
        if label:
            key = arms[0],label
            if key in found:
                raise ValueError(f'ambiguous checkpoint {key}: {found[key]} / {p}')
            found[key] = p
    return found


def score_checkpoint(adapter,stories,chars,vocabulary,arm):
    before = slow_digest(adapter.model)
    result = {}
    for mode in (('normal','no_fast') if arm=='A1' else ('normal',)):
        rows = []
        for sid,text in enumerate(stories):
            clean = adapter.score(text,nofast=mode=='no_fast')
            masks,_ = word_annotations(text)
            row = dict(story=sid,length=len(text),S={k:sufficient(clean,m) for k,m in masks.items()},all=sufficient(clean))
            nonce,metadata = nonce_transform(text,chars,vocabulary,seed=sid)
            row['C1'] = dict(metadata=metadata,kth={},first=sufficient([]),late=sufficient([]),first_eligible_occurrence=None,late_occurrences=0)
            if nonce is not None:
                losses = adapter.score(nonce,nofast=mode=='no_fast')
                _,spans = word_annotations(nonce)
                first = True
                late = np.zeros(len(losses),dtype=bool)
                for span in spans:
                    if span['word']!=metadata['nonce'] or span['gap'] is None or span['gap']<64:
                        continue
                    mask = np.zeros(len(losses),dtype=bool)
                    mask[max(1,span['start'])-1:span['end']-1] = True
                    row['C1']['kth'][str(span['occurrence'])] = sufficient(losses,mask)
                    is_first = first
                    if first:
                        row['C1']['first'] = sufficient(losses,mask)
                        row['C1']['first_eligible_occurrence'] = span['occurrence']
                        first = False
                    if span['occurrence']>=3 and not is_first:
                        late |= mask
                        row['C1']['late_occurrences'] += 1
                row['C1']['late'] = sufficient(losses,late)
            midpoint = len(text)//2
            shifted = adapter.score(swap_transform(text,midpoint),nofast=mode=='no_fast')
            positions = np.arange(1,len(text))
            row['C2'] = {}
            for b in range((len(text)-midpoint+63)//64):
                mask = (positions>=midpoint+b*64)&(positions<midpoint+(b+1)*64)
                changed = mask & np.array([c in 'ea' for c in text[1:]])
                row['C2'][str(b)] = dict(shift=sufficient(shifted,mask),clean=sufficient(clean,mask),
                    excess=sufficient(shifted-clean,mask),changed_char_excess=sufficient(shifted-clean,changed))
            row['C2_endpoints'] = dict(first=sufficient(shifted-clean,(positions>=midpoint)&(positions<midpoint+64)),
                late=sufficient(shifted-clean,(positions>=midpoint+256)&(positions<midpoint+512)))
            rows.append(row)
        c1keys = sorted({k for r in rows for k in r['C1']['kth']},key=int)
        c2keys = sorted({k for r in rows for k in r['C2']},key=int)
        result[mode] = dict(stories=rows,S={k:aggregate([r['S'][k] for r in rows]) for k in rows[0]['S']},
            C1={k:aggregate([r['C1']['kth'][k] for r in rows if k in r['C1']['kth']]) for k in c1keys},
            C1_coverage=dict(selected_stories=sum('skipped' not in r['C1']['metadata'] for r in rows),
                beyond_W_stories=sum(bool(r['C1']['kth']) for r in rows),
                paired_endpoint_stories=sum(r['C1']['first']['count']>0 and r['C1']['late']['count']>0 for r in rows)),
            C2={k:aggregate([r['C2'][k]['excess'] for r in rows if k in r['C2']]) for k in c2keys})
    c3 = []
    for start in range(0,len(stories)-4,5):
        curves = {}
        for transform in ('clean','swap'):
            for boundary in ('carry','reset'):
                adapter.reset()
                curves[transform+'_'+boundary] = []
                for j,text in enumerate(stories[start:start+5]):
                    loss = adapter.score(swap_transform(text) if transform=='swap' else text,carry=boundary=='carry' and j>0)
                    curves[transform+'_'+boundary].append(sufficient(loss))
        c3.append(dict(group=start//5,stories=list(range(start,start+5)),**curves))
    result['C3'] = c3
    result['C3_summary'] = {}
    for j in range(5):
        series = [[group[key][j] for group in c3] for key in ('swap_carry','clean_carry','swap_reset','clean_reset')]
        result['C3_summary'][str(j+1)] = dict(
            swap_carry=aggregate(series[0]), clean_carry=aggregate(series[1]),
            swap_reset=aggregate(series[2]), clean_reset=aggregate(series[3]))
    after = slow_digest(adapter.model)
    if before!=after:
        raise RuntimeError('HELDOUT EVALUATION CHANGED SLOW STATE')
    result['slow_state'] = dict(before=before,after=after,unchanged=True)
    return result


def figures(metrics,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for stage in ('midpoint','final'):
        subset = {k:v for k,v in metrics['checkpoints'].items() if k.endswith('/'+stage)}
        if not subset:
            continue
        fig,axes = plt.subplots(1,3,figsize=(16,4))
        for key,checkpoint in subset.items():
            for mode in ('normal','no_fast'):
                if mode not in checkpoint:
                    continue
                data = checkpoint[mode]
                label = key.split('/')[0]+(' no-fast' if mode!='normal' else '')
                axes[0].plot(range(4),[data['S'][d]['nats_per_char'] if data['S'][d]['count'] else np.nan for d in DISTANCE],marker='o',label=label)
                for ax,probe in zip(axes[1:],('C1','C2')):
                    keys = sorted(data[probe],key=int)
                    ax.plot([int(k) for k in keys],[data[probe][k]['nats_per_char'] for k in keys],marker='.',label=label)
        axes[0].set_xticks(range(4),['<64','64–127','128–255','≥256'])
        axes[0].set(xlabel='Previous word-start gap',ylabel='nats / repeat character',title='S1 distance')
        axes[1].set(xlabel='Nonce occurrence k (gap ≥64)',ylabel='nats / nonce character',title='C1 retrieval / relearning')
        axes[2].axhline(0,color='gray',linewidth=.5)
        axes[2].set(xlabel='64-char bin after e↔a shift',ylabel='shift − clean nats / character',title='C2 adaptation proxy')
        axes[0].legend(fontsize=7)
        fig.suptitle(stage)
        fig.tight_layout()
        fig.savefig(out/f'{stage}_distance_C1_C2.png',dpi=160)
        plt.close(fig)


def bootstrap_contrast(series, coefficients, repetitions=2000, seed=20261106):
    """Token-weighted linear contrast, resampling the same supported docs together."""
    a = np.array([[[r['sum'],r['count']] for r in column] for column in series],dtype=float)
    if not len(series) or not len(series[0]):
        return dict(estimate=None,ci95=None,stories=0)
    keep = (a[:,:,1]>0).all(axis=0)
    a = a[:,keep,:]
    n = int(keep.sum())
    if not n:
        return dict(estimate=None,ci95=None,stories=0)
    def estimate(totals):
        return np.dot(np.asarray(coefficients),totals[:,0]/totals[:,1])
    rng = np.random.default_rng(seed)
    draws = [estimate(a[:,rng.integers(n,size=n),:].sum(axis=1)) for _ in range(repetitions)]
    return dict(estimate=float(estimate(a.sum(axis=1))),ci95=np.quantile(draws,[.025,.975]).tolist(),stories=n)


def endpoint_reports(metrics,repetitions):
    reports = {}
    for result in metrics['checkpoints'].values():
        for j in range(5):
            series = [[g[k][j] for g in result['C3']] for k in ('swap_carry','clean_carry','swap_reset','clean_reset')]
            contrast = bootstrap_contrast(series,[1,-1,-1,1],repetitions)
            contrast['groups'] = contrast.pop('stories')
            result['C3_summary'][str(j+1)]['excess_carry_minus_excess_reset'] = contrast
    for stage in ('midpoint','final'):
        checkpoints = metrics['checkpoints']
        a0 = checkpoints.get('A0/'+stage)
        a1 = checkpoints.get('A1/'+stage)
        if not a0 or not a1:
            continue
        base = a0['normal']['stories']
        on,off = (a1[m]['stories'] for m in ('normal','no_fast'))
        report = {}
        for probe,field in (('C1','C1'),('C2','C2_endpoints')):
            for label,rows in (('A0',base),('A1',on),('A1_no_fast',off)):
                report[probe+'_'+label+'_recovery'] = bootstrap_contrast(
                    [[r[field][endpoint] for r in rows] for endpoint in ('first','late')],[1,-1],repetitions)
            for label,rows in (('A1',on),('A1_no_fast',off)):
                report[probe+'_'+label+'_recovery_minus_A0'] = bootstrap_contrast(
                    [[r[field][e] for r in rs] for rs in (rows,base) for e in ('first','late')],
                    [1,-1,-1,1],repetitions)
        report['S2_on_minus_off'] = paired_bootstrap([r['S']['beyond_W'] for r in on],
                                                   [r['S']['beyond_W'] for r in off],repetitions)
        means = [aggregate([r['S']['beyond_W'] for r in rows])['nats_per_char'] for rows in (base,on,off)]
        gain = means[0]-means[1] if all(x is not None for x in means) else None
        report['S2'] = dict(on_advantage=None if gain is None else gain,
            off_advantage=None if gain is None else means[0]-means[2],
            removal_criterion=None if gain is None else means[0]-means[2]<=.01,
            retained_fraction=(means[0]-means[2])/gain if gain is not None and abs(gain)>1e-12 else None,
            caveat='A1-off retains exclusive slow mask; ratio unstable when on gain near zero')
        a2 = checkpoints.get('A2/'+stage)
        if a2:
            report['A1_minus_A2'] = {k:paired_bootstrap([r['S'][k] for r in on],
                [r['S'][k] for r in a2['normal']['stories']],repetitions) for k in ('64_127','ge128')}
        reports[stage] = report
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runs','data','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--stories','--stories128',dest='stories',type=int,default=128)
    parser.add_argument('--bootstrap',type=int,default=2000)
    parser.add_argument('--allow-incomplete',action='store_true',help='Smoke only: report missing checkpoints')
    args = parser.parse_args(argv)
    if args.stories<1 or args.bootstrap<1:
        parser.error('stories/bootstrap must be positive')
    data = torch.load(args.data,map_location='cpu',weights_only=False)
    chars = data['chars']
    stories = [s for s in data['validation'] if len(s)>=2 and set(s)<=set(chars)][:args.stories]
    if not stories:
        raise ValueError('no eligible complete validation stories')
    if 'e' not in chars or 'a' not in chars:
        raise ValueError('C2 requires e and a in charset')
    vocabulary = {m.group().lower() for s in data['train']+data['validation'] for m in WORD.finditer(s)}
    paths = discover_checkpoints(args.runs)
    missing = [f'{a}/{s}' for a in ('A0','A1','A2','A3') for s in ('midpoint','final') if (a,s) not in paths]
    if missing and not args.allow_incomplete:
        raise FileNotFoundError('missing checkpoints: '+', '.join(missing))
    if not paths:
        raise FileNotFoundError('no midpoint/final checkpoints')
    metrics = dict(protocol=dict(stories=len(stories),requested_stories=args.stories,
        selection='first complete validation strings length>=2 entirely in charset; no crop',
        story_sha256=[hashlib.sha256(s.encode()).hexdigest() for s in stories],
        distance='lowercased alphabetic words; previous word-start gap; bins <64,64–127,128–255,>=256',
        alignment='target position p>=1 maps to loss[p-1]; no BOS',
        attention_caveat='four local attention layers transmit approximately4(W-1); beyondW is not attention-free',
        C1='name/non-stopword heuristic, NO POS model; absent train+validation nonce vocabulary; exact token replacement',
        C2='midpoint lowercase e<->a; same clean-position excess; adaptation proxy, not proof of nonretrieval',
        C3='full-document same swap across five consecutive docs; fast carry vs boundary wipe; attention resets every doc; trailing <5 omitted',
        uncertainty='descriptive paired document bootstrap; C3 resamples five-document groups; one seed, not confirmatory',missing_checkpoints=missing),checkpoints={})
    args.out.mkdir(parents=True,exist_ok=True)
    for (arm,stage),path in sorted(paths.items()):
        print(f'Scoring {arm}/{stage}: {path}',flush=True)
        adapter = ScoringAdapter(path,chars,args.device)
        if adapter.model.config.arm != arm:
            raise ValueError(f'checkpoint payload arm {adapter.model.config.arm} differs from path {arm}')
        if adapter.model.config.seed != 110601:
            raise ValueError('HY1 pilot checkpoint seed must be 110601')
        metadata = dict(checkpoint=str(path),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            tokens=adapter.payload.get('tokens'),steps=adapter.payload.get('steps'),config=adapter.payload.get('config'),
            state_bytes=adapter.model.state_bytes(),
            fixed_buffers_bytes={name:value.numel()*value.element_size() for name,value in adapter.model.named_buffers()},
            workspaces='Dense d_model*hidden_dim FP32 fast scatter and DFA gradient are temporary, not carried state')
        metrics['checkpoints'][f'{arm}/{stage}'] = dict(**metadata,**score_checkpoint(adapter,stories,chars,vocabulary,arm))
        del adapter
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
        (args.out/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')
    final_tokens = {r['tokens'] for k,r in metrics['checkpoints'].items() if k.endswith('/final') and r['tokens'] is not None}
    if len(final_tokens)>1:
        raise ValueError('final checkpoints have unequal token horizons')
    if final_tokens:
        horizon = next(iter(final_tokens))
        for key,result in metrics['checkpoints'].items():
            if key.endswith('/midpoint') and result['tokens'] is not None and not horizon/2 <= result['tokens'] <= horizon/2+64:
                raise ValueError(f'{key}: not a natural half-horizon checkpoint')
    metrics['paired_vs_A0'] = {}
    for key,result in metrics['checkpoints'].items():
        stage = key.split('/')[1]
        baseline = metrics['checkpoints'].get('A0/'+stage)
        if baseline is None:
            continue
        for mode in ('normal','no_fast'):
            if mode not in result:
                continue
            metrics['paired_vs_A0'][key+'/'+mode] = {k:paired_bootstrap([r['S'][k] for r in result[mode]['stories']],
                [r['S'][k] for r in baseline['normal']['stories']],args.bootstrap) for k in result[mode]['S']}
    metrics['exploratory_endpoints'] = endpoint_reports(metrics,args.bootstrap)
    (args.out/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')
    figures(metrics,args.out)
    print(f'Wrote {args.out}/metrics.json and PNG figures',flush=True)


if __name__=='__main__':
    main()
