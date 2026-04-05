#!/usr/bin/env python3
"""Analyze system ID data: fit plant models, validate, and design controllers.

Usage:
    python analyze_sysid.py --log_dir control_logs --alpha0 1000 --gamma0 0.7

Reads CSV files from system ID runs, fits analytical and unconstrained models,
validates on held-out data, designs LQR and H-inf controllers, and saves results.
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from plant_model import (
    load_sysid_csv,
    estimate_plant_analytical,
    estimate_plant_unconstrained,
    estimate_plant_2d,
    validate_model,
)
from controllers import design_lqr, design_hinf


def main():
    parser = argparse.ArgumentParser(description="System ID analysis")
    parser.add_argument("--log_dir", type=str, default="control_logs")
    parser.add_argument("--alpha0", type=float, default=1000.0,
                        help="Nominal α₀ for analytical model")
    parser.add_argument("--gamma0", type=float, default=0.7,
                        help="Nominal γ₀ for analytical model")
    parser.add_argument("--output_dir", type=str, default="sysid_results")
    parser.add_argument("--Q", type=float, default=1.0, help="State penalty for controllers")
    parser.add_argument("--R", type=float, default=0.01, help="Control penalty for controllers")
    parser.add_argument("--alpha_min", type=float, default=1.0)
    parser.add_argument("--alpha_max", type=float, default=1e5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find all CSV files
    csv_files = sorted(glob.glob(os.path.join(args.log_dir, "*.csv")))
    if not csv_files:
        print(f"No CSV files found in {args.log_dir}")
        return

    print(f"Found {len(csv_files)} CSV files")

    # Split into train/validation
    if len(csv_files) >= 3:
        train_files = csv_files[:-2]
        val_files = csv_files[-2:]
    else:
        train_files = csv_files
        val_files = csv_files[:1]  # validate on training data as fallback

    # Load and concatenate training data
    all_train_data = []
    for f in train_files:
        data = load_sysid_csv(f)
        all_train_data.append(data)
        alpha_val = data["alpha"][0]
        print(f"  {os.path.basename(f)}: {len(data['t'])} steps, α={alpha_val:.1f}, "
              f"mean x={np.mean(data['x']):.4f}, mean g={np.mean(data['g_raw']):.4f}")

    # Concatenate for fitting
    train_data = {}
    for key in all_train_data[0]:
        train_data[key] = np.concatenate([d[key] for d in all_train_data])

    # === Fit Models ===
    print("\n=== Analytical Model (constrained structure) ===")
    model_analytical = estimate_plant_analytical(train_data, args.alpha0, args.gamma0)
    print(f"  A = {model_analytical.A:.6f}")
    print(f"  B = {model_analytical.B:.6f}")
    print(f"  E = {model_analytical.E:.6f}")
    print(f"  H (effective Hessian) = {model_analytical.H:.6f}")
    print(f"  Stable: {model_analytical.is_stable}")

    print("\n=== Unconstrained Model (sanity check) ===")
    model_unconstrained = estimate_plant_unconstrained(train_data)
    print(f"  A = {model_unconstrained.A:.6f}")
    print(f"  B = {model_unconstrained.B:.6f}")
    print(f"  E = {model_unconstrained.E:.6f}")
    print(f"  Stable: {model_unconstrained.is_stable}")

    print("\n=== 2D Model ===")
    model_2d = estimate_plant_2d(train_data, args.alpha0, args.gamma0)
    print(f"  A = \n{model_2d.A}")
    print(f"  B = {model_2d.B.flatten()}")
    print(f"  Eigenvalues: {model_2d.eigenvalues}")
    print(f"  Stable: {model_2d.is_stable}")

    # === Validate ===
    print("\n=== Validation ===")
    for f in val_files:
        val_data = load_sysid_csv(f)
        rmse_a, r2_a = validate_model(model_analytical, val_data)
        rmse_u, r2_u = validate_model(model_unconstrained, val_data)
        print(f"  {os.path.basename(f)}:")
        print(f"    Analytical:     RMSE={rmse_a:.6f}, R²={r2_a:.4f}")
        print(f"    Unconstrained:  RMSE={rmse_u:.6f}, R²={r2_u:.4f}")

    # === Design Controllers ===
    print("\n=== Controller Design ===")
    ctrl_lqr = design_lqr(model_analytical, Q=args.Q, R=args.R,
                           alpha_min=args.alpha_min, alpha_max=args.alpha_max)
    print(f"  LQR: {ctrl_lqr}")

    ctrl_hinf = design_hinf(model_analytical, Q=args.Q, R=args.R,
                             alpha_min=args.alpha_min, alpha_max=args.alpha_max)
    print(f"  H-inf: {ctrl_hinf}")

    # === Save Results ===
    results = {
        "analytical": {
            "A": model_analytical.A, "B": model_analytical.B, "E": model_analytical.E,
            "H": model_analytical.H, "alpha0": args.alpha0, "gamma0": args.gamma0,
            "g_bar": model_analytical.g_bar, "x_bar": model_analytical.x_bar,
            "stable": model_analytical.is_stable,
        },
        "unconstrained": {
            "A": model_unconstrained.A, "B": model_unconstrained.B,
            "E": model_unconstrained.E, "stable": model_unconstrained.is_stable,
        },
        "lqr": {
            "K": ctrl_lqr.K, "A_cl": ctrl_lqr.A_cl,
            "P": ctrl_lqr.P, "x_ref": ctrl_lqr.x_ref,
        },
        "hinf": {
            "K": ctrl_hinf.K, "A_cl": ctrl_hinf.A_cl,
            "gamma_opt": ctrl_hinf.gamma_opt, "P": ctrl_hinf.P,
        },
        "controller_params": {
            "Q": args.Q, "R": args.R,
            "alpha_min": args.alpha_min, "alpha_max": args.alpha_max,
        },
    }

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    results_clean = json.loads(json.dumps(results, default=convert))

    results_path = os.path.join(args.output_dir, "sysid_results.json")
    with open(results_path, "w") as f:
        json.dump(results_clean, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # === Plots ===
    _plot_model_validation(model_analytical, model_unconstrained, val_files, args.output_dir)
    _plot_sysid_overview(all_train_data, train_files, args.output_dir)


def _plot_model_validation(model_a, model_u, val_files, output_dir):
    """Plot model predictions vs actual trajectories on validation data."""
    for f in val_files:
        data = load_sysid_csv(f)
        x = data["x"]
        g_raw = data["g_raw"]
        alpha = data["alpha"]

        T = len(x) - 1
        dx = x - model_a.x_bar
        du = alpha - model_a.alpha0
        dd = g_raw - model_a.g_bar

        # One-step predictions
        pred_a = model_a.A * dx[:T] + model_a.B * du[:T] + model_a.E * dd[:T] + model_a.x_bar
        pred_u = model_u.A * (x[:T] - model_u.x_bar) + model_u.B * (alpha[:T] - model_u.alpha0) + model_u.E * (g_raw[:T] - model_u.g_bar) + model_u.x_bar

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        name = os.path.basename(f).replace(".csv", "")

        axes[0].plot(x[1:], label="Actual", alpha=0.7)
        axes[0].plot(pred_a, label="Analytical", alpha=0.7, linestyle="--")
        axes[0].plot(pred_u, label="Unconstrained", alpha=0.7, linestyle=":")
        axes[0].set_ylabel("‖w_eph‖")
        axes[0].set_title(f"Model Validation: {name}")
        axes[0].legend()

        # Residuals
        axes[1].plot(x[1:] - pred_a, label="Analytical residual", alpha=0.7)
        axes[1].plot(x[1:] - pred_u, label="Unconstrained residual", alpha=0.7)
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Residual")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"validation_{name}.png"), dpi=150)
        plt.close()


def _plot_sysid_overview(all_data, files, output_dir):
    """Plot overview of all system ID runs."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for data, f in zip(all_data, files):
        name = os.path.basename(f).replace(".csv", "")
        alpha_val = data["alpha"][0]
        label = f"α={alpha_val:.0f}"
        axes[0].plot(data["x"], label=label, alpha=0.7)
        axes[1].plot(data["g_raw"], label=label, alpha=0.7)
        axes[2].plot(data["loss"], label=label, alpha=0.7)

    axes[0].set_ylabel("‖w_eph‖")
    axes[0].set_title("System ID Runs: Training Dynamics")
    axes[0].legend()
    axes[1].set_ylabel("‖g_raw‖")
    axes[1].legend()
    axes[2].set_ylabel("Loss")
    axes[2].set_xlabel("Step")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sysid_overview.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
