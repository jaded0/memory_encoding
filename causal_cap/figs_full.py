import json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
L = '/home/jaden/causal_cap_local/'; OUT = '/home/jaden/Documents/brain/'
r1 = json.load(open(L + 'res.json')); r2 = json.load(open(L + 'res2.json')); res = {**r1, **r2}
C = {'CTRL': '#6b6b6b', 'CAP8': '#d55e00', 'CAPALL_T': '#0072b2', 'CAPALL_TRB': '#7a3fb5', 'FROB': '#009e73', 'CAPALL_TRB_late': '#cc79a7'}
plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'axes.edgecolor': '#555', 'axes.labelcolor': '#222', 'text.color': '#222',
                     'xtick.color': '#333', 'ytick.color': '#333', 'axes.grid': True, 'grid.color': '#e6e6e6', 'axes.spines.top': False, 'axes.spines.right': False})
def ser(a, key, k=11):
    rows = res[a]['rows']; x = np.array([r['iter'] for r in rows]); y = np.array([r.get(key, np.nan) for r in rows])
    return x, np.array([np.nanmedian(y[max(0, i - k // 2): i + k // 2 + 1]) for i in range(len(y))])
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
for a in ['CTRL', 'CAP8', 'CAPALL_T', 'CAPALL_TRB', 'FROB', 'CAPALL_TRB_late']:
    for j, key in enumerate(['loss', 'recall_acc']):
        x, m = ser(a, key); s = x >= 150000
        ax[j].plot(x[s] / 1e3, m[s], color=C[a], lw=2, label=a)
ax[0].set_yscale('log'); ax[0].axhline(5, color='#999', ls=':', lw=1); ax[0].set_ylabel('loss, running median of 11 windows'); ax[1].set_ylabel('recall_acc, running median')
for a_ in ax: a_.set_xlabel('iteration (k)')
ax[0].legend(frameon=False, fontsize=8, loc='upper left')
fig.suptitle('Capping every slow singular value, readout and bias norm does not prevent the degradation; a Frobenius-only cap is worse than no cap', fontsize=11, y=1.0)
fig.tight_layout(); fig.savefig(OUT + 'causalcap 2026-10-02 fig5 full-spectrum caps loss and recall.png', dpi=150, bbox_inches='tight'); plt.close()
# fig6: top singular value + fro per arm at checkpoints: use norm_cap logs
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for a in ['CAPALL_TRB', 'FROB']:
    rec = [json.loads(l) for l in open(L + f'runs2/{a}/norm_cap_log.jsonl')]
    its = [r['iter'] / 1e3 for r in rec]
    for j, n in enumerate(['linear_layers.0', 'linear_layers.1', 'linear_layers.2']):
        ax[j].plot(its, [r['layers'][n]['top5'][0] for r in rec], color=C[a], lw=2, label=a)
for j, n in enumerate(['0', '1', '2']):
    d = json.load(open(L + 'sv_CTRL.json')); its = sorted(d, key=int)
    ax[j].plot([int(i) / 1e3 for i in its], [d[i][f'linear_layers.{n}']['top8'][0] for i in its], color=C['CTRL'], lw=2, label='CTRL (to 210k)')
    ax[j].set_title(f'trunk {n}: top singular value of slow matrix'); ax[j].set_xlabel('iteration (k)')
ax[0].legend(frameon=False, fontsize=8)
fig.suptitle('The spectrum cap pins every singular value; the Frobenius cap leaves the top singular values free to keep growing', fontsize=11, y=1.03)
fig.tight_layout(); fig.savefig(OUT + 'causalcap 2026-10-02 fig6 top singular values under full caps.png', dpi=150, bbox_inches='tight'); plt.close()
