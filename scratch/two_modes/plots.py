import json,numpy as np,re,matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from classify import classify
exec(open('collect.py').read().split('res={}')[0])
R=json.load(open('cells.json')); S=json.load(open('sig.json'))
OUT='/home/jaden/Documents/brain/'
BG='#ffffff'; INK='#222222'; MUT='#666666'
plt.rcParams.update({'font.size':10,'axes.edgecolor':MUT,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'text.color':INK,'figure.facecolor':BG,'axes.facecolor':BG,'savefig.facecolor':BG,'axes.spines.top':False,'axes.spines.right':False})
COL={'stable':'#0072B2','transient':'#E69F00','recovering':'#E69F00','non-recovering':'#CC79A7','delayed':'#CC79A7','runaway':'#D55E00'}
MK={'stable':'o','transient':'s','non-recovering':'D','runaway':'X'}
def cls(n):
    c=R[n]; o=classify(c['its'],c['series']['loss'],c['start'])
    k=o['cls']
    if k=='transient?': k='transient' if np.all(np.array(c['series']['loss'][-3:])<6) else 'non-recovering'
    if k=='delayed collapse': k='non-recovering'
    return k,o
def stage_m(n):
    c=R[n]; return c['start'],round(c['alpha']/1e4,3)
# Fig1
fig,ax=plt.subplots(figsize=(8,4.8))
cells={}
for n in R:
    if n.startswith('E2_'): continue
    s,m=stage_m(n); k,o=cls(n); cells[(s,m)]=(k,o,n)
stages=sorted({s for s,_ in cells}); xs={s:i for i,s in enumerate(stages)}
for (s,m),(k,o,n) in cells.items():
    ax.scatter(xs[s],m,c=COL[k],marker=MK[k],s=70,edgecolor='white',linewidth=0.8,zorder=3)
    pk=o['peak']; lab=('%.0f'%pk if pk>=10 else '%.1f'%pk) if pk<1e4 else '%.0e'%pk
    ax.annotate(lab,(xs[s],m),textcoords='offset points',xytext=(8,-3),fontsize=7,color=MUT)
ax.set_yscale('log'); ax.set_yticks([0.3,0.5,0.7,1,1.5,2,3]); ax.set_yticklabels(['0.3','0.5','0.7','1','1.5','2','3'])
ax.axhline(1,color=MUT,lw=0.8,ls=':')
ax.set_xticks(range(len(stages))); ax.set_xticklabels(['%gk'%(s/1000) for s in stages]); ax.set_xlim(-0.5,len(stages)-0.3)
ax.set_xlabel('stage: iteration of the checkpoint where alpha is multiplied'); ax.set_ylabel('alpha multiplier m (x 1e4)')
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([],[],marker=MK[k],color='w',markerfacecolor=COL[k],markersize=9,label=k) for k in ['stable','transient','non-recovering','runaway']],loc='upper center',bbox_to_anchor=(0.5,-0.14),frameon=False,fontsize=8,ncol=4)
ax.set_title('Outcome of an alpha kick, by stage and multiplier (number = peak window loss)',fontsize=10,loc='left')
fig.tight_layout(); fig.savefig(OUT+'twomodes 2026-10-03 fig1 kick outcome by stage and alpha multiplier.png',dpi=160)
# Fig2 trajectories
fig,axs=plt.subplots(2,3,figsize=(11,5.8),sharex='col')
sets=[('early (X1a, 10k)',['X1a10k_m1','X1a10k_m1p5']),('late (B, 150k)',['B150k_m1','B150k_m1p5'])]
cc={'m1':'#0072B2','m1p5':'#D55E00'}
for r,(t,ns) in enumerate(sets):
    for n in ns:
        c=R[n]; k=c['start']; lab='m=1' if n.endswith('m1') else 'm=1.5'; col=cc['m1' if n.endswith('m1') else 'm1p5']
        its=np.array(c['its'])-k
        axs[r,0].plot(its,np.maximum(c['series']['loss'],0.5),color=col,label=lab,lw=1.5)
        s=np.array(S[n]); axs[r,1].plot(s[:,0]-k,s[:,1],color=col,lw=1.5)
        axs[r,2].plot(s[:,0]-k,s[:,3],color=col,lw=1.5)
    axs[r,0].set_yscale('log'); axs[r,1].set_yscale('log'); axs[r,2].set_yscale('log')
    axs[r,0].set_ylabel(t+'\ninterval loss'); axs[r,0].legend(frameon=False,fontsize=8)
    axs[r,2].axhline(1,color=MUT,lw=0.8,ls=':')
axs[0,1].set_title('slow-only trunk activation (step 0, fast=0)',fontsize=9,loc='left'); axs[0,2].set_title('fast / slow drive, layer 2, last step',fontsize=9,loc='left'); axs[0,0].set_title('loss',fontsize=9,loc='left')
for a in axs[1]: a.set_xlabel('iterations after the kick')
fig.tight_layout(); fig.savefig(OUT+'twomodes 2026-10-03 fig2 early vs late kick trajectories.png',dpi=160)
# Fig3 rescue
fig,axs=plt.subplots(1,5,figsize=(14,3.6),sharey=True)
ctl={}
for nm,f in [('B','data/B_train.log'),('X3','data/X3_train.log')]: ctl[nm]=parse(f)
st=[('B',180000),('B',205000),('X3',230000),('X3',270000),('X3',300000)]
mc={0.5:'#0072B2',0.3:'#009E73',0.1:'#CC79A7'}
for a,(lin,s) in zip(axs,st):
    p=ctl[lin]; xi=[i for i in sorted(p) if s<=i<=s+50000]
    if xi: a.plot(np.array(xi)-s,[p[i]['loss'] for i in xi],color='#555555',ls='--',lw=1.3,label='m=1 (original run)')
    for m in [0.5,0.3,0.1]:
        n=f"E2_{lin}{s//1000}k_m{str(m).replace('.','p')}"
        if n in R:
            c=R[n]; a.plot(np.array(c['its'])-s,np.maximum(c['series']['loss'],0.5),color=mc[m],lw=1.5,label=f'm={m}')
    a.set_yscale('log'); a.set_title(f'{lin} {s//1000}k',fontsize=9,loc='left'); a.set_xlabel('iterations after resume')
axs[0].set_ylabel('interval loss'); axs[0].legend(frameon=False,fontsize=7)
fig.tight_layout(); fig.savefig(OUT+'twomodes 2026-10-03 fig3 rescue after onset.png',dpi=160)
# table dump
rows=[]
for (s,m),(k,o,n) in sorted(cells.items()): rows.append((s,m,k,o['peak'],o['peak_it'],o['recovery'],o['onset']))
json.dump(rows,open('table.json','w'))
print('ok')
