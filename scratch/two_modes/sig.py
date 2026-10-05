import torch,glob,json,os,re
out={}
for d in sorted(glob.glob('runs/*/')):
    n=os.path.basename(d.rstrip('/')); s=[]
    for f in sorted(glob.glob(d+'traces/*.pt')):
        t=torch.load(f,weights_only=False); it=t['iter']; tr=t['traces']
        a=tr['act_norm']; fd=tr['fast_drive']; sd=tr['slow_drive']
        s.append((it,float(a[0,:,3].median()),float(a[-1,:,3].median()),float((fd[-1,:,2]/sd[-1,:,2].clamp_min(1e-9)).median()),float(tr['max_logit'].max(0).values.median())))
    out[n]=s
json.dump(out,open('sig.json','w')); print(len(out))
