import json,numpy as np,matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
exec(open('plots.py').read().split('# Fig1')[0])
mcol={0.7:'#009E73',0.85:'#0072B2',1:'#E69F00',1.2:'#CC79A7',1.5:'#D55E00'}
# Fig4: replicates at B150
fig,axs=plt.subplots(1,3,figsize=(12,3.6),sharey=True)
for a,m in zip(axs,[0.85,1,1.2]):
    mm=str(m).replace('.','p')
    names=[f'B150k_m{mm}']+[f'B150k_m{mm}_rs{r}' for r in (11,12,13)]
    for i,n in enumerate(names):
        if n not in R: continue
        c=R[n]; x=np.array(c['its'])-150000; y=np.maximum(c['series']['loss'],0.5)
        a.plot(x,y,color=mcol[m],lw=1.4,alpha=1 if i==0 else 0.7,ls='-' if i==0 else '--',label='original (seed 3141 data order)' if i==0 else f'replicate {i}')
    a.set_yscale('log'); a.set_title(f'B 150k, m = {m}',fontsize=10,loc='left'); a.set_xlabel('iterations after the kick'); a.axhline(5,color=MUT,lw=0.7,ls=':')
    a.legend(frameon=False,fontsize=7,loc='upper left')
axs[0].set_ylabel('interval loss (dotted: 5)')
fig.tight_layout(); fig.savefig(OUT+'twomodes 2026-10-03 fig4 replicates at 150k.png',dpi=160)
# Fig5: lineages
fig,axs=plt.subplots(1,2,figsize=(11,3.8),sharey=True)
for a,sd in zip(axs,(2718,4241)):
    c=R[f'L{sd}_train']; a.plot(np.array(c['its'])-150000,np.maximum(c['series']['loss'],0.5),color='#888888',lw=1,label='own training (m=1 history)')
    for m in [0.7,0.85,1,1.2,1.5]:
        n=f"L{sd}_150k_m{str(m).replace('.','p')}"
        if n in R:
            c=R[n]; a.plot(np.array(c['its'])-150000,np.maximum(c['series']['loss'],0.5),color=mcol[m],lw=1.4,label=f'm={m}')
    a.set_yscale('log'); a.set_title(f'independent lineage, seed {sd}',fontsize=10,loc='left'); a.set_xlabel('iterations relative to 150k (kick at 0)'); a.axvline(0,color=MUT,lw=0.7)
    a.legend(frameon=False,fontsize=7,ncol=2,loc='upper left')
axs[0].set_ylabel('interval loss')
fig.tight_layout(); fig.savefig(OUT+'twomodes 2026-10-03 fig5 independent lineages.png',dpi=160)
