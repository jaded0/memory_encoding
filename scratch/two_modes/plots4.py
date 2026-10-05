import numpy as np,glob,torch,matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
exec(open('plots.py').read().split('# Fig1')[0])
cells=[('B 180k, m=0.5','E2_B180k_m0p5','E2x_B180k_m0p5','#0072B2'),('B 180k, m=0.3','E2_B180k_m0p3','E2x_B180k_m0p3','#009E73'),('B 205k, m=0.3','E2_B205k_m0p3','E2x_B205k_m0p3','#E69F00'),('X3 230k, m=0.3','E2_X3230k_m0p3','E2x_X3230k_m0p3','#CC79A7')]
fig,ax=plt.subplots(1,2,figsize=(11,3.8))
for lab,a,b,col in cells:
    its=[];L=[];act=[]
    for n in (a,b):
        p=parse(f'runs/{n}/train.log')
        for i in sorted(p):
            if i not in its and 'loss' in p[i]: its.append(i);L.append(p[i]['loss'])
        for f in sorted(glob.glob(f'runs/{n}/traces/*.pt'))[::10]:
            t=torch.load(f,weights_only=False); act.append((t['iter'],float(t['traces']['act_norm'][0,:,3].median())))
    its=np.array(its);L=np.array(L); o=np.argsort(its); its=its[o]; L=L[o]
    k=10; sm=np.array([np.median(L[max(0,i-k):i+k+1]) for i in range(len(L))])
    ax[0].plot(its/1000,sm,color=col,lw=1.5,label=lab)
    act=sorted(set(act)); ax[1].plot([x/1000 for x,_ in act],[y for _,y in act],color=col,lw=1.5,label=lab)
ax[0].set_yscale('log'); ax[0].set_xlabel('iteration (k)'); ax[0].set_ylabel('interval loss (running median)'); ax[0].legend(frameon=False,fontsize=8); ax[0].set_title('Rescued runs, 100-150k iterations after the kick',fontsize=9,loc='left')
ax[1].set_yscale('log'); ax[1].set_xlabel('iteration (k)'); ax[1].set_ylabel('slow-only trunk activation'); ax[1].set_title('The slow weights keep drifting under the lower alpha',fontsize=9,loc='left')
fig.tight_layout(); fig.savefig(OUT+'twomodes 2026-10-03 fig8 rescue extension.png',dpi=160)
