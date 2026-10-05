import re,json,glob,os
def parse(path):
    t=re.sub(r'\x1b\[[0-9;]*m','',open(path,errors='replace').read())
    blocks=re.split(r'--- Interval metrics \(ending @ iter (\d+)[^\n]*\n',t)[1:]
    out={}
    for i in range(0,len(blocks),2):
        it=int(blocks[i]); d={}
        for k,v in re.findall(r'^\s*([\w/]+): (-?[\d.eE+-]+|nan|inf)\s*$',blocks[i+1].split('-----')[0],re.M):
            try: d[k]=float(v)
            except: pass
        out[it]=d
    return out
res={}
for d in sorted(glob.glob('runs/*/')):
    n=os.path.basename(d.rstrip('/'))
    lg=d+'train.log'
    if not os.path.exists(lg): continue
    p=parse(lg)
    txt=open(lg,errors='replace').read()
    m=re.search(r'--plasticity (\S+)',txt)
    ck=re.search(r'resume_checkpoint \S+_(\d+)\.pth',txt)
    res[n]=dict(its=sorted(p),series={k:[p[i].get(k,float('nan')) for i in sorted(p)] for k in ['loss','recall_acc','trace/trunk_act_norm_last','trace/fast_over_slow_drive_last','trace/max_logit_max','trace/loop_gain_median','trace/i2h_pre_norm_last']},
      alpha=float(m.group(1)) if m else None,start=int(ck.group(1)) if ck else None,done='=== exit 0' in txt)
json.dump(res,open('cells.json','w'))
print(len(res),sum(r['done'] for r in res.values()))
