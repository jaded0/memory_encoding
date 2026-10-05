import json,re,glob,numpy as np,torch
from scipy.stats import spearmanr
exec(open('plots.py').read().split('# Fig1')[0])
feat=json.load(open('feat_local.json')); feat.update(json.load(open('feat_L.json')))
def F(path_end):
    for k,v in feat.items():
        if k.endswith(path_end): return v
# groups: (lineage, stage) -> {m: [classes]}
def group(n,c):
    if c['alpha'] is None or c['start'] is None or not c['done']: return None
    m=round(c['alpha']/1e4,3); st=c['start']
    if n.startswith('E2_') or 'long' in n or n.endswith('_train'): return None
    if n.startswith('X1a'): lin='X1a'
    elif n.startswith('L2718'): lin='L2718'
    elif n.startswith('L4241'): lin='L4241'
    elif n.startswith('B'): lin='B'
    else: return None
    return lin,st,m
G={}
for n,c in R.items():
    g=group(n,c)
    if not g: continue
    k,o=cls(n); G.setdefault(g[:2],{}).setdefault(g[2],[]).append((k,o['peak'],n))
def edge(d):
    ms=sorted(d); bad=[m for m in ms if any(k!='stable' for k,_,_ in d[m])]
    stab=[m for m in ms if all(k=='stable' for k,_,_ in d[m])]
    lo=max([m for m in stab if m<min(bad)] or [None]) if bad else None
    return (lo,min(bad)) if bad else (max(ms),None)
def act0(lin,st):
    if lin=='B': f=f'data/trB/trace_{st:08d}.pt'
    elif lin=='X1a': f=f'data/trX/trace_{st:08d}.pt'
    else:
        fs=sorted(glob.glob(f'runs/{lin}_train/traces/*.pt')); idx=[int(re.search(r'(\d+)\.pt',x).group(1)) for x in fs]
        sel=[x for x,i in zip(fs,idx) if abs(i-st)<=1000]; vals=[float(torch.load(x,weights_only=False)['traces']['act_norm'][0,:,3].median()) for x in sel]
        return float(np.median(vals))
    try: return float(torch.load(f,weights_only=False)['traces']['act_norm'][0,:,3].median())
    except Exception: return float('nan')
def kloss(lin,st):
    src={'B':'data/B_train.log','X1a':'data/X1a_train.log'}.get(lin,f'runs/{lin}_train/train.log')
    p=parse(src); xs=[i for i in sorted(p) if st-2000<i<=st+500 and 'loss' in p[i]]
    if lin=='B': xs=[i for i in sorted(p) if abs(i-st)<=2500]   # 5000-spaced log
    return float(np.mean([p[i]['loss'] for i in xs])) if xs else float('nan')
rows=[]
for (lin,st),d in sorted(G.items()):
    lo,hi=edge(d)
    fk={'B':'ckpt/B_%08d.pth'%st,'X1a':'ckpt/X1a_%08d.pth'%st}.get(lin,f'runs/{lin}_train/checkpoint_{st:08d}.pth')
    f=F(fk) or {}
    nrep={m:len(v) for m,v in d.items()}
    row=dict(lin=lin,stage=st,lo=lo,hi=hi,act0=act0(lin,st),kloss=kloss(lin,st),sv_t2=f.get('sv_t2'),sv_t1=f.get('sv_t1'),i2o_fro=f.get('i2o_fro'),i2o_sv=f.get('i2o_sv'),bias=f.get('bias_norm'),
        outcomes={m:''.join({'stable':'S','transient':'T','non-recovering':'N','runaway':'R'}[k] for k,_,_ in v) for m,v in d.items()})
    rows.append(row)
json.dump(rows,open('health_rows.json','w'),default=float)
for r in rows: print(r['lin'],r['stage'],'edge',r['lo'],r['hi'],'act0 %.3g kloss %.3g sv_t2 %s'%(r['act0'],r['kloss'],r['sv_t2']),r['outcomes'])
