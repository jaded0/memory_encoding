#!/usr/bin/env python3
"""Generate comparison plots for controlled vs uncontrolled training.

Plots:
1. Time-domain: weight norms, α trajectories, loss curves for all modes
2. Stability comparison: % of runs that explode per mode
3. Bode/frequency response of controlled vs uncontrolled plant
4. Stability margins (gain/phase margin)
5. Pole-zero map for 2D model

Usage:
    python plot_results.py --log_dir comparison_logs --sysid sysid_results/sysid_results.json
"""

import argparse
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np

from plant_model import load_sysid_csv

try:
    import control as ctrl
except ImportError:
    ctrl = None
    print("Warning: python-control not installed, skipping frequency-domain plots")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="comparison_logs")
    parser.add_argument("--sysid", type=str, default="sysid_results/sysid_results.json")
    parser.add_argument("--output_dir", type=str, default="plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load system ID results
    with open(args.sysid, "r") as f:
        sysid = json.load(f)

    # Load comparison CSVs
    csv_files = sorted(glob.glob(os.path.join(args.log_dir, "*.csv")))
    if not csv_files:
        print(f"No CSV files in {args.log_dir}")
        return

    # Parse runs into groups
    runs = {}
    for f in csv_files:
        name = os.path.basename(f).replace(".csv", "")
        m = re.match(r"(fixed|lqr|hinf)_alpha(\d+)_seed(\d+)", name)
        if not m:
            continue
        mode, alpha, seed = m.group(1), int(m.group(2)), int(m.group(3))
        key = (mode, alpha)
        if key not in runs:
            runs[key] = []
        data = load_sysid_csv(f)
        data["_seed"] = seed
        data["_name"] = name
        runs[key].append(data)

    # 1. Stability comparison (% explosion)
    plot_stability_comparison(runs, args.output_dir)

    # 2. Time-domain trajectories for a representative α
    for alpha in sorted(set(k[1] for k in runs)):
        plot_trajectories(runs, alpha, args.output_dir)

    # 3. Frequency-domain analysis
    if ctrl is not None:
        plot_frequency_response(sysid, args.output_dir)
        plot_stability_margins(sysid, args.output_dir)

    print(f"Plots saved to {args.output_dir}/")


def is_exploded(data, threshold=1e6):
    """Check if a run exploded (weight norm exceeded threshold or NaN)."""
    x = data["x"]
    return np.any(np.isnan(x)) or np.any(x > threshold)


def plot_stability_comparison(runs, output_dir):
    """Bar chart: % of runs that explode per mode and α."""
    modes = ["fixed", "lqr", "hinf"]
    alphas = sorted(set(k[1] for k in runs))

    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.25
    x_pos = np.arange(len(alphas))

    for i, mode in enumerate(modes):
        explosion_rates = []
        for alpha in alphas:
            key = (mode, alpha)
            if key in runs:
                rate = sum(1 for d in runs[key] if is_exploded(d)) / len(runs[key])
            else:
                rate = 0
            explosion_rates.append(rate * 100)
        ax.bar(x_pos + i * width, explosion_rates, width, label=mode.upper())

    ax.set_xlabel("α₀ (plast_clip)")
    ax.set_ylabel("Explosion Rate (%)")
    ax.set_title("Stability: % of Runs That Explode")
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels([str(a) for a in alphas])
    ax.legend()
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stability_comparison.png"), dpi=150)
    plt.close()


def plot_trajectories(runs, alpha, output_dir):
    """Plot weight norms, α, and loss trajectories for all modes at a given α."""
    modes = ["fixed", "lqr", "hinf"]
    colors = {"fixed": "tab:red", "lqr": "tab:blue", "hinf": "tab:green"}

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    for mode in modes:
        key = (mode, alpha)
        if key not in runs:
            continue
        for data in runs[key]:
            c = colors[mode]
            axes[0].plot(data["x"], color=c, alpha=0.3)
            axes[1].plot(data["alpha"], color=c, alpha=0.3)
            axes[2].plot(data["loss"], color=c, alpha=0.3)
            axes[3].plot(data["acc"], color=c, alpha=0.3)

    # Add legend entries (one per mode)
    for mode in modes:
        axes[0].plot([], [], color=colors[mode], label=mode.upper())

    axes[0].set_ylabel("‖w_eph‖")
    axes[0].set_title(f"Training Dynamics (α₀={alpha})")
    axes[0].legend()
    axes[0].set_yscale("log")
    axes[1].set_ylabel("α(t)")
    axes[1].set_yscale("log")
    axes[2].set_ylabel("Loss")
    axes[3].set_ylabel("Accuracy")
    axes[3].set_xlabel("Step")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"trajectories_alpha{alpha}.png"), dpi=150)
    plt.close()


def plot_frequency_response(sysid, output_dir):
    """Bode plot of open-loop and closed-loop systems."""
    A = sysid["analytical"]["A"]
    B = sysid["analytical"]["B"]

    # Open-loop plant (discrete-time, Ts=1)
    plant = ctrl.tf([B], [1, -A], dt=1)

    # LQR closed-loop
    K_lqr = sysid["lqr"]["K"]
    cl_lqr = ctrl.feedback(plant, K_lqr)

    # H-inf closed-loop
    K_hinf = sysid["hinf"]["K"]
    cl_hinf = ctrl.feedback(plant, K_hinf)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Frequency vector
    omega = np.logspace(-3, np.log10(np.pi), 500)

    for sys, label, ls in [(plant, "Open-loop", "-"),
                           (cl_lqr, "LQR", "--"),
                           (cl_hinf, "H-inf", ":")]:
        mag, phase, w = ctrl.frequency_response(sys, omega)
        mag_db = 20 * np.log10(np.abs(mag).flatten())
        phase_deg = np.angle(mag).flatten() * 180 / np.pi

        axes[0].semilogx(w, mag_db, label=label, linestyle=ls)
        axes[1].semilogx(w, phase_deg, label=label, linestyle=ls)

    axes[0].set_ylabel("Magnitude (dB)")
    axes[0].set_title("Frequency Response")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel("Phase (deg)")
    axes[1].set_xlabel("Frequency (rad/sample)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bode_plot.png"), dpi=150)
    plt.close()


def plot_stability_margins(sysid, output_dir):
    """Print and visualize stability margins for each controller."""
    A = sysid["analytical"]["A"]
    B = sysid["analytical"]["B"]
    plant = ctrl.tf([B], [1, -A], dt=1)

    results = {}
    for name, K in [("LQR", sysid["lqr"]["K"]), ("H-inf", sysid["hinf"]["K"])]:
        loop_tf = plant * K
        try:
            gm, pm, wgm, wpm = ctrl.margin(loop_tf)
            results[name] = {"gain_margin": gm, "phase_margin": pm,
                           "freq_gm": wgm, "freq_pm": wpm}
            print(f"{name}: GM={gm:.2f} dB, PM={pm:.1f}°")
        except Exception as e:
            print(f"{name} margin calculation failed: {e}")
            results[name] = {"error": str(e)}

    # Save margins
    with open(os.path.join(output_dir, "stability_margins.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)


if __name__ == "__main__":
    main()
