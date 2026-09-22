#!/usr/bin/env python3
"""Plot controller comparison on palindrome task (fixed, lqr, hinf, adaptive)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import os
import re
from collections import defaultdict

def load_csv(path):
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                data.setdefault(k, []).append(float(v))
    return {k: np.array(v) for k, v in data.items()}

def parse_filename(fname):
    """Parse 'mode_aALPHA_sSEED.csv' -> (mode, alpha, seed)"""
    m = re.match(r'(fixed|lqr|hinf|adaptive)_a(\d+)_s(\d+)\.csv', fname)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    return None, None, None

# Use 4-way logs if available, fall back to 2-way
log_dir_4way = os.path.expanduser("~/memory_encoding/palindrome_4way_logs")
log_dir_2way = os.path.expanduser("~/memory_encoding/palindrome_comparison_logs")
log_dir = log_dir_4way if os.path.isdir(log_dir_4way) else log_dir_2way
plot_dir = os.path.expanduser("~/memory_encoding/plots")

# Load all runs
runs = {}  # (mode, alpha, seed) -> data
for f in sorted(os.listdir(log_dir)):
    if not f.endswith(".csv"):
        continue
    mode, alpha, seed = parse_filename(f)
    if mode is None:
        continue
    data = load_csv(os.path.join(log_dir, f))
    if len(data.get("t", [])) < 10:
        continue
    runs[(mode, alpha, seed)] = data

# Group by (mode, alpha)
grouped = defaultdict(list)
for (mode, alpha, seed), data in runs.items():
    grouped[(mode, alpha)].append(data)

alphas = sorted(set(a for (_, a) in grouped.keys()))
print("Available conditions:", list(grouped.keys()))
print("Alphas:", alphas)

# ============================================================
# FIGURE 1: Time-domain traces per alpha (fixed vs adaptive)
# ============================================================
for alpha in alphas:
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(
        "Palindrome Task: Controller Comparison "
        "(α₀={}, γ=0.01, no recurrence)".format(alpha),
        fontsize=13, fontweight='bold')

    colors = {"fixed": "tab:red", "lqr": "tab:green", "hinf": "tab:purple", "adaptive": "tab:blue"}
    all_modes = [m for m in ["fixed", "lqr", "hinf", "adaptive"]
                 if any((m, alpha) == k for k in grouped)]

    for mode in all_modes:
        key = (mode, alpha)
        if key not in grouped:
            continue
        for i, data in enumerate(grouped[key]):
            label = "{} (seed {})".format(mode.title(), i+1) if i == 0 else None
            lw = 1.0
            a = 0.5

            # Loss
            axes[0].plot(data["t"], data["loss"], color=colors[mode],
                         alpha=a, linewidth=lw, label=label)
            # Alpha
            axes[1].plot(data["t"], data["alpha"], color=colors[mode],
                         alpha=a, linewidth=lw, label=label)
            # Weight norm
            axes[2].plot(data["t"], data["x"], color=colors[mode],
                         alpha=a, linewidth=lw, label=label)
            # Accuracy
            # Smooth accuracy with rolling window
            window = 50
            if len(data["acc"]) > window:
                acc_smooth = np.convolve(data["acc"], np.ones(window)/window, mode="valid")
                t_smooth = data["t"][:len(acc_smooth)]
            else:
                acc_smooth = data["acc"]
                t_smooth = data["t"]
            axes[3].plot(t_smooth, acc_smooth, color=colors[mode],
                         alpha=a, linewidth=lw, label=label)

    axes[0].set_ylabel("Loss")
    axes[0].set_yscale("log")
    axes[0].legend(loc="upper right")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("α(t)")
    axes[1].set_yscale("log")
    axes[1].legend(loc="upper right")
    axes[1].set_title("Plasticity Multiplier")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel("‖w_eph‖")
    axes[2].set_yscale("log")
    axes[2].legend(loc="upper right")
    axes[2].set_title("Ephemeral Weight Norm")
    axes[2].grid(True, alpha=0.3)

    axes[3].set_ylabel("Accuracy")
    axes[3].set_ylim(0, 1.05)
    axes[3].legend(loc="lower right")
    axes[3].set_title("Key-Recall Accuracy (smoothed)")
    axes[3].set_xlabel("Training Step")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(plot_dir, "palindrome_traces_a{}.png".format(alpha))
    plt.savefig(out, dpi=150)
    plt.close()
    print("Saved", out)


# ============================================================
# FIGURE 2: Summary bar chart across all alpha values
# ============================================================
modes_present = sorted(set(m for (m, _) in grouped.keys()),
                       key=lambda m: ["fixed", "lqr", "hinf", "adaptive"].index(m)
                       if m in ["fixed", "lqr", "hinf", "adaptive"] else 99)
colors = {"fixed": "tab:red", "lqr": "tab:green", "hinf": "tab:purple", "adaptive": "tab:blue"}
n_modes = len(modes_present)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Palindrome Task: Controller Comparison Summary\n"
             "(enable_recurrence=False, γ=0.01, no normalization, no grad clip)",
             fontsize=12, fontweight='bold')

x_pos = np.arange(len(alphas))
width = 0.8 / max(n_modes, 1)

# Compute per-condition stats
stats = {}
for (mode, alpha), data_list in grouped.items():
    losses = []
    accs = []
    max_losses = []
    for d in data_list:
        last_n = min(100, len(d["loss"]))
        losses.append(np.mean(d["loss"][-last_n:]))
        accs.append(np.mean(d["acc"][-last_n:]))
        max_losses.append(np.max(d["loss"]))
    stats[(mode, alpha)] = {
        "loss_mean": np.mean(losses),
        "loss_std": np.std(losses),
        "acc_mean": np.mean(accs),
        "acc_std": np.std(accs),
        "max_loss_mean": np.mean(max_losses),
        "max_loss_std": np.std(max_losses),
    }

# Panel 1: Average loss (last 100 steps)
for i, mode in enumerate(modes_present):
    vals = [stats.get((mode, a), {}).get("loss_mean", 0) for a in alphas]
    errs = [stats.get((mode, a), {}).get("loss_std", 0) for a in alphas]
    offset = (i - (n_modes - 1) / 2) * width
    axes[0].bar(x_pos + offset, vals, width, yerr=errs, label=mode.title(),
                color=colors.get(mode, "tab:gray"), alpha=0.7, capsize=3)

axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(["α={}".format(a) for a in alphas])
axes[0].set_ylabel("Avg Loss (last 100 steps)")
axes[0].set_yscale("log")
axes[0].legend()
axes[0].set_title("Training Loss")
axes[0].grid(True, alpha=0.3, axis="y")

# Panel 2: Accuracy
for i, mode in enumerate(modes_present):
    vals = [stats.get((mode, a), {}).get("acc_mean", 0) for a in alphas]
    errs = [stats.get((mode, a), {}).get("acc_std", 0) for a in alphas]
    offset = (i - (n_modes - 1) / 2) * width
    axes[1].bar(x_pos + offset, vals, width, yerr=errs, label=mode.title(),
                color=colors.get(mode, "tab:gray"), alpha=0.7, capsize=3)

axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(["α={}".format(a) for a in alphas])
axes[1].set_ylabel("Avg Accuracy (last 100 steps)")
axes[1].set_ylim(0, 1.05)
axes[1].legend()
axes[1].set_title("Accuracy")
axes[1].grid(True, alpha=0.3, axis="y")

# Panel 3: Max loss (stability indicator)
for i, mode in enumerate(modes_present):
    vals = [stats.get((mode, a), {}).get("max_loss_mean", 0) for a in alphas]
    errs = [stats.get((mode, a), {}).get("max_loss_std", 0) for a in alphas]
    offset = (i - (n_modes - 1) / 2) * width
    axes[2].bar(x_pos + offset, vals, width, yerr=errs, label=mode.title(),
                color=colors.get(mode, "tab:gray"), alpha=0.7, capsize=3)

axes[2].set_xticks(x_pos)
axes[2].set_xticklabels(["α={}".format(a) for a in alphas])
axes[2].set_ylabel("Max Loss (peak instability)")
axes[2].set_yscale("log")
axes[2].legend()
axes[2].set_title("Peak Loss (Stability)")
axes[2].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
out = os.path.join(plot_dir, "palindrome_summary.png")
plt.savefig(out, dpi=150)
plt.close()
print("Saved", out)

# ============================================================
# Print numerical summary
# ============================================================
print("\n" + "=" * 70)
print("NUMERICAL SUMMARY")
print("=" * 70)
for alpha in alphas:
    print("\n--- alpha = {} ---".format(alpha))
    for mode in modes_present:
        s = stats.get((mode, alpha))
        if s is None:
            print("  {}: NO DATA".format(mode))
            continue
        n_runs = len(grouped[(mode, alpha)])
        print("  {} ({} runs):".format(mode, n_runs))
        print("    Avg loss (last 100):  {:.2f} ± {:.2f}".format(s["loss_mean"], s["loss_std"]))
        print("    Avg accuracy:         {:.4f} ± {:.4f}".format(s["acc_mean"], s["acc_std"]))
        print("    Max loss:             {:.1f} ± {:.1f}".format(s["max_loss_mean"], s["max_loss_std"]))

    # Compute improvement ratios vs fixed baseline
    fs = stats.get(("fixed", alpha))
    if fs and fs["loss_mean"] > 0:
        for mode in modes_present:
            if mode == "fixed":
                continue
            ms = stats.get((mode, alpha))
            if ms and ms["loss_mean"] > 0:
                ratio = fs["loss_mean"] / ms["loss_mean"]
                print("  => {} {:.1f}x lower loss than fixed".format(mode.title(), ratio))
