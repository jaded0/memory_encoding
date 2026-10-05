import json,numpy as np,matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
exec(open('plots.py').read().split('# Fig1')[0])
rows=json.load(open('health_rows.json'))
LC={'B':'#E69F00','L4241':'#0072B2','L2718':'#CC79A7','X1a':'#009E73'}; LM={'B':'o','L4241':'s','L2718':'D','X1a':'^'}
LN={'B':'B (seed 3141)','L4241':'seed 4241','L2718':'seed 2718','X1a':'X1a (seed 3141 fresh)'}
def bar(r):
    lo,hi=r['lo'],r['hi']; hi2=hi if hi is not None else 1.4; lo2=lo if lo is not None else 0.5
    return lo2,hi2
# fig6: edge vs stage
fig,ax=plt.subplots(1,2,figsize=(11,4))
for lin in LC:
    rs=sorted([r for r in rows if r['lin']==lin and not (lin=='X1a' and r['stage'] in (2500,10000))],key=lambda r:r['stage'])
    xs=np.array([r['stage'] for r in rs]); lo=np.array([bar(r)[0] for r in rs]); hi=np.array([bar(r)[1] for r in rs]); mid=np.sqrt(lo*hi)
    off={'B':-0.015,'L4241':0.015,'L2718':0.03,'X1a':0}[lin]
    ax[0].errorbar(xs/1000*(1+off*4),mid,yerr=[mid-lo,hi-mid],fmt=LM[lin]+('-' if len(rs)>1 else ''),color=LC[lin],lw=1.2,capsize=3,ms=6,label=LN[lin])
ax[0].set_xscale('log'); ax[0].set_xticks([5,10,20,50,100,150]); ax[0].set_xticklabels(['5k','10k','20k','50k','100k','150k']); ax[0].axhline(1,color=MUT,ls=':',lw=0.8)
ax[0].set_ylabel('edge: bracket of smallest unstable m'); ax[0].set_xlabel('stage (iteration of the kick)'); ax[0].legend(frameon=False,fontsize=8)
ax[0].set_title('Edge vs stage (bar = last stable m to first unstable m)',fontsize=9,loc='left')
for lin in LC:
    for r in rows:
        if r['lin']!=lin or r.get('sv_t2') is None or (lin=='X1a' and r['stage'] in (2500,10000)): continue
        lo,hi=bar(r); mid=np.sqrt(lo*hi)
        ax[1].errorbar(r['sv_t2'],mid,yerr=[[mid-lo],[hi-mid]],fmt=LM[lin],color=LC[lin],capsize=2,ms=6,label=LN[lin])
h,l=ax[1].get_legend_handles_labels(); d=dict(zip(l,h)); ax[1].legend(d.values(),d.keys(),frameon=False,fontsize=8)
ax[1].axhline(1,color=MUT,ls=':',lw=0.8); ax[1].set_xlabel('top singular value of the trunk-2 slow matrix at the kick'); ax[1].set_ylabel('edge')
ax[1].set_title('Edge vs slow gain (Spearman -0.92 for stages >= 40k, n = 10)',fontsize=9,loc='left')
fig.tight_layout(); fig.savefig(OUT+'twomodes 2026-10-03 fig6 edge by stage and slow gain.png',dpi=160)
# fig7: long horizon
fig,ax=plt.subplots(1,1,figsize=(8,4))
cc={'B150k_m0p85_long_rs31':'#0072B2','B150k_m0p85_long_rs32':'#56B4E9','B150k_m0p85_long_rs33':'#4C6A92','B150k_m0p7_long_rs34':'#009E73'}
for n,col in cc.items():
    c=R[n]; its=np.array(c['its']); L=np.array(c['series']['loss'])
    k=25; sm=np.array([np.median(L[max(0,i-k):i+k+1]) for i in range(len(L))])
    ax.plot(its/1000,sm,color=col,lw=1.5,label=('m=0.7' if '0p7' in n else 'm=0.85 rep %s'%n[-2:]))
c=R['B150k_m1_rs11']; its=np.array(c['its']); L=np.array(c['series']['loss']); ax.plot(its/1000,np.array([np.median(L[max(0,i-25):i+26]) for i in range(len(L))]),color='#D55E00',lw=1.5,label='m=1 (rep 11, 50k only)')
ax.set_yscale('log'); ax.axhline(5,color=MUT,ls=':',lw=0.8); ax.set_xlabel('iteration (k); kick at 150k'); ax.set_ylabel('interval loss (running median of 51 windows)'); ax.legend(frameon=False,fontsize=8)
ax.set_title('Lower alpha delays the collapse: m=0.85 collapses at about 220k, m=0.7 at about 295k',fontsize=9,loc='left')
fig.tight_layout(); fig.savefig(OUT+'twomodes 2026-10-03 fig7 long horizon at m 0p85 and 0p7.png',dpi=160)
print('ok')
