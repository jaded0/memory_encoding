"""Figures for the feedback-loop traces (loop_trace.py, trace_replay.py). Static PNGs, light surface,
the dataviz skill's reference palette (categorical in fixed order; diverging blue <-> neutral <-> orange
with warm = amplification).

A source is a run directory (its traces/trace_<iter>.pt, written by --trace_loop_every) or a
trace_replay.py output file; both give a list of (iteration, traces of the first batch).

    python plots/loop_figures.py timeline    SOURCE --out figures/loop_timeline_control.png
    python plots/loop_figures.py heatstrip   SOURCE --out figures/loop_heatstrip_control.png
    python plots/loop_figures.py trajectories control=DIR tanh=DIR ... --at 4000 --out figures/loop_arms.png
    python plots/loop_figures.py arms        control=DIR tanh=DIR ... --out figures/loop_arms_timeline.png
"""
import argparse
import glob
import math
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loop_trace import summarize  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
DIVERGING = LinearSegmentedColormap.from_list("amp", ["#2a78d6", "#f0efec", "#eb6834"])
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
    "axes.titlecolor": INK, "lines.linewidth": 2,
})


def load_source(source):
    """[(iteration, traces)] sorted by iteration."""
    if os.path.isdir(source):
        paths = sorted(glob.glob(os.path.join(source, "traces", "trace_*.pt")))
        saved = [torch.load(p, weights_only=False) for p in paths]
        return [(s["iter"], s["traces"]) for s in saved]
    replay = torch.load(source, weights_only=False)
    return sorted(((r["iter"], r["batches"][0]) for r in replay["checkpoints"]), key=lambda x: x[0])


def parse_arms(items):
    arms = []
    for item in items:
        name, _, path = item.partition("=")
        arms.append((name, load_source(path)))
    return arms


def series_of(records):
    """Per-iteration scalars from traces."""
    its, rows = [], []
    for it, tr in records:
        s = summarize(tr)
        s["loss"] = float(tr["loss"].mean())
        s["fast_norm_max"] = float(tr["fast_norm"].square().sum(2).sqrt().max())
        s["slow_drive_last"] = float(tr["slow_drive"][-1].mean())
        s["fast_drive_last"] = float(tr["fast_drive"][-1].mean())
        its.append(it)
        rows.append(s)
    return np.array(its), rows


def finish(fig, out, title=None, note=None):
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    if note:
        fig.text(0.01, -0.02, note, ha="left", va="top", fontsize=7.5, color=MUTED)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print("wrote", out)


def timeline(records, out, title):
    its, rows = series_of(records)
    panels = [
        ("loss", "loss", True),
        ("trace/max_logit_max", "max logit", True),
        ("trace/trunk_act_norm_max", "max trunk\nactivation norm", True),
        ("trace/fast_norm_last", "fast-weight norm\n(last step)", True),
        ("trace/loop_gain_median", "loop gain\n(median)", False),
        ("trace/frac_gain_gt1", "fraction of steps\nwith gain > 1", False),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(7.5, 1.55 * len(panels)), sharex=True)
    for ax, (key, label, log) in zip(axes, panels):
        values = np.array([r[key] for r in rows], dtype=float)
        ax.plot(its, values, color=SERIES[0], marker="o", markersize=3.5, markeredgecolor=SURFACE)
        if log:
            ax.set_yscale("log")
        if key == "trace/loop_gain_median":
            ax.axhline(1, color=SERIES[1], linewidth=1, linestyle=(0, (4, 3)))
        ax.set_ylabel(label, fontsize=7.5)
    axes[-1].set_xlabel("training iteration")
    finish(fig, out, title, "Traces of one fixed training batch per point. Gain = write-norm ratio of successive steps.")


def heatstrip(records, out, title):
    its = [it for it, _ in records]
    width = max(tr["loop_gain"].shape[0] for _, tr in records)
    grid = np.full((len(records), width), np.nan)
    loss = []
    for r, (_, tr) in enumerate(records):
        gain = tr["loop_gain"].numpy()
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            med = np.nanmedian(np.log2(gain), axis=1) if np.isfinite(gain).any() else np.full(gain.shape[0], np.nan)
        grid[r, :len(med)] = med
        loss.append(float(tr["loss"].mean()))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8, 0.17 * len(records) + 1.8), sharey=True,
                                  gridspec_kw={"width_ratios": [4, 1.2]})
    limit = max(1.0, np.nanpercentile(np.abs(grid), 98)) if np.isfinite(grid).any() else 1.0
    image = ax.imshow(grid, aspect="auto", cmap=DIVERGING, vmin=-limit, vmax=limit, origin="lower",
                      extent=[0.5, width + 0.5, -0.5, len(its) - 0.5], interpolation="nearest")
    ax.grid(False)
    ax.set_xlabel("step within the sequence")
    step = max(1, len(its) // 12)
    ax.set_yticks(range(0, len(its), step))
    ax.set_yticklabels([str(its[i]) for i in range(0, len(its), step)])
    ax.set_ylabel("training iteration")
    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.set_label("log2 loop gain; orange = amplifying", fontsize=7.5)
    ax2.plot(loss, range(len(its)), color=SERIES[0])
    ax2.set_xscale("log")
    ax2.set_xlabel("batch loss")
    finish(fig, out, title, "Loop gain = write norm at step t / at step t-1. Blank = no previous write.")


def trajectories(arms, at, out, title):
    quantities = [("fast_norm", "fast-weight norm (all layers)", True), ("act_norm", "trunk activation norm", True),
                  ("max_logit", "max logit", True), ("loop_gain", "loop gain", True)]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.1))
    for color, (name, records) in zip(SERIES, arms):
        its = np.array([it for it, _ in records])
        it, tr = records[int(np.abs(its - at).argmin())]
        for ax, (key, label, log) in zip(axes, quantities):
            v = tr[key]
            if key == "fast_norm":
                v = v.square().sum(2).sqrt()
            elif key == "act_norm":
                v = v[:, :, -1]
            v = v.numpy()
            with np.errstate(all="ignore"):
                med = np.nanmedian(v, axis=1) if np.isfinite(v).any() else np.full(v.shape[0], np.nan)
            ax.plot(np.arange(len(med)), med, color=color, label=f"{name} (iter {it})")
            ax.set_title(label)
            ax.set_xlabel("step within the sequence")
            if log and np.nanmin(med) > 0:
                ax.set_yscale("log")
    axes[3].axhline(1, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    axes[0].legend(frameon=False, fontsize=7.5)
    finish(fig, out, title, "Median over the batch's sequences. The nearest traced iteration to --at is used per arm.")


def arms_timeline(arms, out, title):
    panels = [("loss", "per-step loss", True), ("trace/max_logit_max", "max logit", True),
              ("trace/fast_norm_last", "fast-weight norm\n(last step)", True),
              ("trace/loop_gain_median", "loop gain (median)", False)]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.1), sharex=True)
    for color, (name, records) in zip(SERIES, arms):
        its, rows = series_of(records)
        for ax, (key, label, log) in zip(axes, panels):
            ax.plot(its, [r[key] for r in rows], color=color, label=name)
            ax.set_title(label)
            ax.set_xlabel("training iteration")
            if log:
                ax.set_yscale("log")
    axes[3].axhline(1, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    axes[0].legend(frameon=False, fontsize=7.5)
    finish(fig, out, title, "One fixed training batch per point.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kind", choices=["timeline", "heatstrip", "trajectories", "arms"])
    parser.add_argument("sources", nargs="+", help="SOURCE, or name=SOURCE for trajectories/arms")
    parser.add_argument("--out", required=True)
    parser.add_argument("--at", type=int, default=0, help="trajectories: the iteration to show")
    parser.add_argument("--title", default=None)
    args = parser.parse_args(argv)
    if args.kind == "timeline":
        timeline(load_source(args.sources[0]), args.out, args.title)
    elif args.kind == "heatstrip":
        heatstrip(load_source(args.sources[0]), args.out, args.title)
    elif args.kind == "trajectories":
        trajectories(parse_arms(args.sources), args.at, args.out, args.title)
    else:
        arms_timeline(parse_arms(args.sources), args.out, args.title)


if __name__ == "__main__":
    main()
