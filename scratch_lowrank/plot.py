"""Figures from measure.py's results.json / arrays.npz.  python scratch_lowrank/plot.py DIR"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

d = sys.argv[1]
r = json.load(open(os.path.join(d, "results.json")))
a = np.load(os.path.join(d, "arrays.npz"))
L = ["L0", "L1", "L2", "i2h"]
col = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# 1. spectra of keys and values
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for l, c in zip(L, col):
    e = a[f"geom_eig_{l}"]
    ax[0].semilogy(np.arange(1, 31), np.maximum(e[:30] / e.sum(), 1e-12), "o-", c=c, label=f"{l} (PR {r['geometry'][l]['PR_uncentered_unitnorm']:.1f})")
    p = a[f"p_eig_{l}"]
    ax[1].semilogy(np.arange(1, 31), np.maximum(p[:30] / p.sum(), 1e-12), "o-", c=c, label=f"{l} (PR {r['value_subspace'][l]['PR_p']:.1f})")
ax[0].axhline(1 / 1033, c="k", ls="--", label="isotropic 1/d")
ax[0].set_title("Key second-moment spectrum (unit-norm x_s, all held-out steps)")
ax[1].set_title("Value (projected error p_t) spectrum")
for x in ax:
    x.set_xlabel("eigenvalue rank"); x.set_ylabel("share of trace"); x.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(d, "fig1_spectra.png"), dpi=130); plt.close()

# 2. read decomposition: contribution norm by lag before the query, and signal share
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for l, c in zip(L[:3], col):
    t = a[f"lagterms_{l}"]
    ax[0].plot(np.arange(1, t.shape[1] + 1), np.nanmean(t, 0), "o-", c=c, label=l)
ax[0].set_xlabel("steps before the answer step (lag)"); ax[0].set_ylabel("mean ||term_s|| (masked, closed form)")
ax[0].set_title("Per-step contribution to F q at the answer step"); ax[0].legend()
w = 0.25
for j, (l, c) in enumerate(zip(L[:3], col)):
    ax[1].bar(j, r["read"][l]["sig_share_sq"]["mean"], yerr=r["read"][l]["sig_share_sq"]["std"], color=c)
ax[1].set_xticks(range(3)); ax[1].set_xticklabels(L[:3]); ax[1].set_ylabel("||signal term||^2 / sum_s ||term_s||^2")
ax[1].set_title("Signal share (step holding the key->value)")
plt.tight_layout(); plt.savefig(os.path.join(d, "fig2_read_decomposition.png"), dpi=130); plt.close()

# 3. masked vs phi * unmasked, and mask / F ranks
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for j, (l, c) in enumerate(zip(L, col)):
    x = a[f"ratio_phi_{l}"]
    ax[0].hist(x[np.isfinite(x)], bins=40, alpha=0.5, color=c, label=f"{l} mean {np.nanmean(x):.2f}")
ax[0].axvline(1, c="k", ls="--"); ax[0].set_xlabel("||F q|| / (phi ||unmasked formula read||)"); ax[0].legend(); ax[0].set_title("Masked read vs phi x unmasked read, answer step")
for l, c in zip(L, col):
    rows = r["svd_F"][l][0]
    ax[1].plot([x["t"] for x in rows], [x["rank_closed_f64"] for x in rows], "o-", c=c, label=f"{l} rank(F^(t)) float64")
ax[1].set_xlabel("t (writes so far)"); ax[1].set_ylabel("numerical rank"); ax[1].set_yscale("log"); ax[1].legend(fontsize=8)
ax[1].set_title("Rank of masked F^(t), real sequence")
plt.tight_layout(); plt.savefig(os.path.join(d, "fig3_ratio_and_rank.png"), dpi=130); plt.close()

# 4. fast-weight norms and share of pre-activation over steps
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for l, c in zip(L, col):
    ax[0].errorbar(np.arange(1, 13), np.nanmean(a[f"fn_{l}"], 0), np.nanstd(a[f"fn_{l}"], 0), c=c, marker="o", label=l)
    ax[1].plot(np.arange(1, 13), np.nanmean(a[f"fx_{l}"], 0) / np.nanmean(a[f"pre_{l}"], 0), "o-", c=c, label=l)
ax[0].set_xlabel("step t (forward reads F after t-1 writes)"); ax[0].set_ylabel("||masked F||_F"); ax[0].set_title("Fast-weight norm trajectory"); ax[0].legend()
ax[1].set_xlabel("step t"); ax[1].set_ylabel("||F x_t|| / ||pre-activation||"); ax[1].set_title("Fast share of pre-activation norm"); ax[1].legend()
plt.tight_layout(); plt.savefig(os.path.join(d, "fig4_norms.png"), dpi=130); plt.close()

# 5. ablations
ab = r["ablation"]
names = ["full", "no_fast@all", "signal_only@all", "drop_signal@all"] + [f"{m}@{l}" for m in ("drop_signal", "signal_only", "no_fast") for l in ("L0", "L1", "L2")]
fig, ax = plt.subplots(figsize=(11, 4))
ax.bar(range(len(names)), [ab[n]["recall_acc"] for n in names], color="#4c78a8")
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8); ax.set_ylabel("recall at answer step")
ax.set_title("Causal ablations of fast-weight terms at the answer step")
plt.tight_layout(); plt.savefig(os.path.join(d, "fig5_ablations.png"), dpi=130); plt.close()
print("ok")
