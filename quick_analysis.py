#!/usr/bin/env python3
"""Quick analysis of system ID data + fit models + test controllers."""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from plant_model import load_sysid_csv, estimate_plant_analytical, estimate_plant_unconstrained, validate_model
from controllers import LQRController, HinfController, FixedController

log_dir = sys.argv[1] if len(sys.argv) > 1 else "control_logs"

# 1. Analyze raw data
print("=" * 60)
print("RAW DATA ANALYSIS")
print("=" * 60)
all_data = {}
for alpha in [100, 1000, 10000]:
    path = os.path.join(log_dir, f"sysid_alpha_{alpha}.csv")
    if not os.path.exists(path):
        continue
    d = load_sysid_csv(path)
    all_data[alpha] = d
    xv = d["x"]
    gv = d["g_raw"]
    gr = d["g_ratio"]
    print(f"\n--- alpha={alpha} ---")
    print(f"  x (weight norm):   mean={np.mean(xv):.6f}  std={np.std(xv):.6f}  max={np.max(xv):.6f}")
    print(f"  g_raw (grad norm): mean={np.mean(gv):.4f}  std={np.std(gv):.4f}")
    print(f"  g_ratio:           mean={np.mean(gr):.4f}")
    print(f"  loss:              mean={np.mean(d['loss']):.4f}  final={d['loss'][-1]:.4f}")
    x_first = np.mean(xv[:50])
    x_last = np.mean(xv[-50:])
    ratio = x_last / x_first if x_first > 1e-12 else 0
    print(f"  x trend:           first50={x_first:.6f}  last50={x_last:.6f}  ratio={ratio:.2f}x")

if not all_data:
    print("No data found!")
    sys.exit(1)

# 2. Fit plant models
print("\n" + "=" * 60)
print("PLANT MODEL FITTING")
print("=" * 60)

# Use alpha=1000 as nominal operating point
alpha0 = 1000.0
gamma0 = 0.7  # Note: actual gamma is tiny due to normalization bug, but this is the param value

# Concatenate all runs for fitting
concat = {}
for key in list(all_data.values())[0]:
    concat[key] = np.concatenate([all_data[a][key] for a in sorted(all_data.keys())])

model_a = estimate_plant_analytical(concat, alpha0, gamma0)
print(f"\nAnalytical model (constrained structure):")
print(f"  A = {model_a.A:.6f}  (stable={model_a.is_stable})")
print(f"  B = {model_a.B:.6f}")
print(f"  E = {model_a.E:.6f}")
print(f"  H (effective Hessian) = {model_a.H:.6f}")
print(f"  g_bar = {model_a.g_bar:.6f}")
print(f"  x_bar = {model_a.x_bar:.6f}")

model_u = estimate_plant_unconstrained(concat)
print(f"\nUnconstrained model (sanity check):")
print(f"  A = {model_u.A:.6f}  (stable={model_u.is_stable})")
print(f"  B = {model_u.B:.6f}")
print(f"  E = {model_u.E:.6f}")

# Validate on each run
print(f"\nValidation (one-step prediction):")
for alpha in sorted(all_data.keys()):
    rmse_a, r2_a = validate_model(model_a, all_data[alpha])
    rmse_u, r2_u = validate_model(model_u, all_data[alpha])
    print(f"  alpha={alpha}: Analytical RMSE={rmse_a:.6f} R2={r2_a:.4f} | Unconstrained RMSE={rmse_u:.6f} R2={r2_u:.4f}")

# 3. Design controllers
print("\n" + "=" * 60)
print("CONTROLLER DESIGN")
print("=" * 60)

try:
    ctrl_lqr = LQRController(
        A=model_a.A, B=model_a.B, Q=1.0, R=0.01,
        alpha0=alpha0, alpha_min=1.0, alpha_max=1e5,
        x_ref=model_a.x_bar
    )
    print(f"\nLQR Controller:")
    print(f"  K = {ctrl_lqr.K:.6f}")
    print(f"  A_cl = {ctrl_lqr.A_cl:.6f}  (stable={abs(ctrl_lqr.A_cl) < 1})")
    print(f"  x_ref = {ctrl_lqr.x_ref:.6f}")
except Exception as e:
    print(f"\nLQR design failed: {e}")
    ctrl_lqr = None

try:
    ctrl_hinf = HinfController(
        A=model_a.A, B=model_a.B, E=model_a.E, Q=1.0, R=0.01,
        alpha0=alpha0, alpha_min=1.0, alpha_max=1e5,
        x_ref=model_a.x_bar
    )
    print(f"\nH-inf Controller:")
    print(f"  K = {ctrl_hinf.K:.6f}")
    print(f"  gamma_opt = {ctrl_hinf.gamma_opt:.6f}")
    print(f"  A_cl = {ctrl_hinf.A_cl:.6f}  (stable={abs(ctrl_hinf.A_cl) < 1})")
except Exception as e:
    print(f"\nH-inf design failed: {e}")
    ctrl_hinf = None

# 4. Simulate controllers on real data
print("\n" + "=" * 60)
print("CONTROLLER SIMULATION (on alpha=10000 data)")
print("=" * 60)

if 10000 in all_data:
    d = all_data[10000]
    xv = d["x"]

    # What alpha would the controllers have chosen?
    if ctrl_lqr:
        lqr_alphas = [ctrl_lqr.compute_alpha(x) for x in xv]
        print(f"\nLQR alpha trajectory:")
        print(f"  mean={np.mean(lqr_alphas):.1f}  min={np.min(lqr_alphas):.1f}  max={np.max(lqr_alphas):.1f}")

    if ctrl_hinf:
        hinf_alphas = [ctrl_hinf.compute_alpha(x) for x in xv]
        print(f"\nH-inf alpha trajectory:")
        print(f"  mean={np.mean(hinf_alphas):.1f}  min={np.min(hinf_alphas):.1f}  max={np.max(hinf_alphas):.1f}")

print("\nDone.")
