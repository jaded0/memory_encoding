import json, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
L = '/home/jaden/causal_cap_local/'; OUT = '/home/jaden/Documents/brain/'
res = json.load(open(L + 'res.json'))
C = {'CTRL': '#6b6b6b', 'SHAM': '#56b4e9', 'SHAM2': '#009e73', 'SHAM3': '#e69f00', 'CAP8': '#d55e00', 'CAP16': '#cc79a7',
     'CAP8_D170': '#0072b2', 'CAP8_D180': '#7a3fb5'}
plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'axes.edgecolor': '#555', 'axes.labelcolor': '#222',
                     'text.color': '#222', 'xtick.color': '#333', 'ytick.color': '#333', 'axes.grid': True, 'grid.color': '#e6e6e6',
                     'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10})
def series(a, key, k=11):
    rows = res[a]['rows']; x = np.array([r['iter'] for r in rows]); y = np.array([r.get(key, np.nan) for r in rows])
    m = np.array([np.nanmedian(y[max(0, i - k // 2): i + k // 2 + 1]) for i in range(len(y))])
    return x, y, m
def panel(ax, arms, key, ylim=None, ylog=False, xlim=None):
    for a in arms:
        x, y, m = series(a, key)
        if xlim: sel = (x >= xlim[0]) & (x <= xlim[1])
        else: sel = x > 0
        ax.plot(x[sel] / 1e3, m[sel], color=C[a], lw=2, label=a)
    if ylog: ax.set_yscale('log')
    if ylim: ax.set_ylim(*ylim)
    ax.set_xlabel('iteration (k)')
# fig1
arms = ['CTRL', 'SHAM', 'SHAM2', 'SHAM3', 'CAP8', 'CAP16']
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
panel(ax[0], arms, 'loss', ylog=True, xlim=(150000, 210000)); ax[0].set_ylabel('loss, running median of 11 windows (500 its each)')
ax[0].axhline(5, color='#999', ls=':', lw=1)
panel(ax[1], arms, 'recall_acc', xlim=(150000, 210000)); ax[1].set_ylabel('recall_acc, running median')
ax[0].legend(frameon=False, ncol=2, fontsize=9)
fig.suptitle('Capping the 8 growth directions prevents the 178k collapse (to 210k); random and tail-rank caps do not, top-of-spectrum cap only partly', fontsize=11, y=1.0)
fig.tight_layout(); fig.savefig(OUT + 'causalcap 2026-10-02 fig1 loss and recall by arm.png', dpi=150, bbox_inches='tight'); plt.close()
# fig2 sv
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for j, n in enumerate(['linear_layers.0', 'linear_layers.1', 'linear_layers.2']):
    for a in ['CTRL', 'CAP8']:
        d = json.load(open(L + f'sv_{a}.json')); its = sorted(d, key=int)
        ax[j].plot([int(i) / 1e3 for i in its], [d[i][n]['block_sv'][0] for i in its], color=C[a], lw=2, label=f'{a}: top block singular value')
    cap = json.load(open(L + 'sv_CAP8.json'))['150000'][n]['caps'][0]
    ax[j].axhline(cap, color='#222', ls='--', lw=1); ax[j].text(152, cap * 1.01, 'cap = 1.1 x value at 150k', fontsize=8)
    ax[j].set_title(f'trunk {n[-1]}'); ax[j].set_xlabel('iteration (k)')
ax[0].set_ylabel('singular value in the capped 8-direction block'); ax[0].legend(frameon=False, fontsize=8, loc='lower right')
fig.suptitle('The cap binds in trunk 1 and 2 from about 170k and holds the block flat; CTRL keeps growing', fontsize=11, y=1.02)
fig.tight_layout(); fig.savefig(OUT + 'causalcap 2026-10-02 fig2 capped singular values.png', dpi=150, bbox_inches='tight'); plt.close()
# fig3 extension & delayed
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
for a in ['CTRL', 'CAP8', 'CAP8_D170', 'CAP8_D180'] + (['CAP16'] if res['CAP16']['last_iter'] > 215000 else []):
    x, y, m = series(a, 'loss'); sel = x >= 150000
    ax[0].plot(x[sel] / 1e3, m[sel], color=C[a], lw=2, label=a)
    x, y, m = series(a, 'recall_acc'); ax[1].plot(x[sel] / 1e3, m[sel], color=C[a], lw=2)
ax[0].set_yscale('log'); ax[0].axhline(5, color='#999', ls=':', lw=1); ax[0].legend(frameon=False, fontsize=9)
ax[0].set_ylabel('loss, running median of 11 windows'); ax[1].set_ylabel('recall_acc, running median')
for a_ in ax: a_.set_xlabel('iteration (k)')
ax[0].axvline(170, color='#bbb', lw=1); ax[0].axvline(180, color='#bbb', lw=1)
fig.suptitle('Caps started at 170k and 180k hold the loss near 3-5 (one excursion to 7); the cap from 150k delays collapse to about 270k', fontsize=11, y=1.0)
fig.tight_layout(); fig.savefig(OUT + 'causalcap 2026-10-02 fig3 extension and delayed cap.png', dpi=150, bbox_inches='tight'); plt.close()
# fig4 where growth goes
g = json.load(open(L + 'growth.json')); keys = list(g)
def key(sub): return [k for k in keys if sub in k][0]
pairs = [('150k to 190k, CTRL', key('CTRL/checkpoint_00190000')), ('210k to 250k, CAP8', key('00210000.pth:runs/CAP8/checkpoint_00250000')), ('250k to 300k, CAP8', key('00250000.pth:runs/CAP8/checkpoint_00300000'))]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
w = 0.25
for i, (lab, k) in enumerate(pairs):
    ax[0].bar(np.arange(3) + (i - 1) * w, [g[k][f'linear_layers.{j}']['frac_in_cap_subspace'] for j in range(3)], w, label=lab, color=['#6b6b6b', '#e69f00', '#d55e00'][i])
    ax[1].bar(np.arange(4) + (i - 1) * w, [g[k][n]['rel_fro'] for n in ['linear_layers.0', 'linear_layers.1', 'linear_layers.2', 'i2o']], w, color=['#6b6b6b', '#e69f00', '#d55e00'][i])
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(['trunk 0', 'trunk 1', 'trunk 2']); ax[0].set_ylabel('share of weight change in capped subspace'); ax[0].legend(frameon=False, fontsize=8)
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(['trunk 0', 'trunk 1', 'trunk 2', 'i2o (readout)']); ax[1].set_ylabel('|change| / |slow weights| at start of window')
fig.suptitle('Under the cap, growth moves out of the capped subspace (share 0.75-0.97 falls to 0.08-0.37) and into the readout', fontsize=11, y=1.02)
fig.tight_layout(); fig.savefig(OUT + 'causalcap 2026-10-02 fig4 where growth goes.png', dpi=150, bbox_inches='tight'); plt.close()
