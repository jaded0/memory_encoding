#!/usr/bin/env python3
"""Plot fixed vs adaptive controller comparison at alpha=2000."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import os

def load_csv(path):
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                data.setdefault(k, []).append(float(v))
    return {k: np.array(v) for k, v in data.items()}

log_dir = os.path.expanduser("~/memory_encoding/control_logs")
fixed = load_csv(os.path.join(log_dir, "final2_fixed_a2000.csv"))
adaptive = load_csv(os.path.join(log_dir, "final2_adaptive_a2000.csv"))

fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
fig.suptitle("Fixed vs Adaptive Controller (α₀=2000, no normalization, no grad clip)", 
             fontsize=14, fontweight='bold')

# 1. Loss
ax = axes[0]
ax.plot(fixed['t'], fixed['loss'], color='tab:red', alpha=0.7, label='Fixed α=2000')
ax.plot(adaptive['t'], adaptive['loss'], color='tab:blue', alpha=0.7, label='Adaptive α₀=2000')
ax.set_ylabel('Loss')
ax.set_yscale('log')
ax.legend(loc='upper right')
ax.set_title('Training Loss')
ax.grid(True, alpha=0.3)
# Mark explosion threshold
ax.axhline(y=10, color='gray', linestyle='--', alpha=0.5, label='Explosion threshold')

# 2. Alpha trajectory
ax = axes[1]
ax.plot(fixed['t'], fixed['alpha'], color='tab:red', alpha=0.7, label='Fixed')
ax.plot(adaptive['t'], adaptive['alpha'], color='tab:blue', alpha=0.7, label='Adaptive')
ax.set_ylabel('α(t)')
ax.set_yscale('log')
ax.legend(loc='upper right')
ax.set_title('Plasticity Multiplier (α)')
ax.grid(True, alpha=0.3)

# 3. Weight norm
ax = axes[2]
ax.plot(fixed['t'], fixed['x'], color='tab:red', alpha=0.7, label='Fixed')
ax.plot(adaptive['t'], adaptive['x'], color='tab:blue', alpha=0.7, label='Adaptive')
ax.set_ylabel('‖w_eph‖')
ax.set_yscale('log')
ax.legend(loc='upper right')
ax.set_title('Ephemeral Weight Norm')
ax.grid(True, alpha=0.3)

# 4. Gradient norm
ax = axes[3]
# Replace inf with NaN for plotting
g_fixed = np.where(np.isinf(fixed['g_raw']), np.nan, fixed['g_raw'])
g_adaptive = np.where(np.isinf(adaptive['g_raw']), np.nan, adaptive['g_raw'])
ax.plot(fixed['t'], g_fixed, color='tab:red', alpha=0.7, label='Fixed')
ax.plot(adaptive['t'], g_adaptive, color='tab:blue', alpha=0.7, label='Adaptive')
ax.set_ylabel('‖g‖')
ax.set_yscale('log')
ax.legend(loc='upper right')
ax.set_title('Gradient Norm')
ax.set_xlabel('Training Step')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.expanduser("~/memory_encoding/plots/comparison_fixed_vs_adaptive.png"), dpi=150)
print("Saved comparison_fixed_vs_adaptive.png")

# Summary statistics figure
fig2, ax2 = plt.subplots(figsize=(10, 6))
metrics = ['Avg Loss\n(first 500)', 'Avg Loss\n(last 500)', '% Steps\nLoss>10', 'Final α\n(÷100)']

fixed_loss_first = np.mean(fixed['loss'][:500])
fixed_loss_last = np.mean(fixed['loss'][-500:])
fixed_high_pct = np.mean(fixed['loss'] > 10) * 100
fixed_final_alpha = fixed['alpha'][-1] / 100

adapt_loss_first = np.mean(adaptive['loss'][:500])
adapt_loss_last = np.mean(adaptive['loss'][-500:])
adapt_high_pct = np.mean(adaptive['loss'] > 10) * 100
adapt_final_alpha = adaptive['alpha'][-1] / 100

fixed_vals = [fixed_loss_first, fixed_loss_last, fixed_high_pct, fixed_final_alpha]
adapt_vals = [adapt_loss_first, adapt_loss_last, adapt_high_pct, adapt_final_alpha]

x = np.arange(len(metrics))
width = 0.35
bars1 = ax2.bar(x - width/2, fixed_vals, width, label='Fixed α=2000', color='tab:red', alpha=0.7)
bars2 = ax2.bar(x + width/2, adapt_vals, width, label='Adaptive α₀=2000', color='tab:blue', alpha=0.7)
ax2.set_ylabel('Value')
ax2.set_title('Summary: Fixed vs Adaptive Controller at Unstable Operating Point')
ax2.set_xticks(x)
ax2.set_xticklabels(metrics)
ax2.legend()
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars1:
    h = bar.get_height()
    ax2.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
for bar in bars2:
    h = bar.get_height()
    ax2.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.expanduser("~/memory_encoding/plots/summary_comparison.png"), dpi=150)
print("Saved summary_comparison.png")

# Print summary
print(f"\n=== SUMMARY ===")
print(f"Fixed α=2000:")
print(f"  First 500 avg loss: {fixed_loss_first:.2f}")
print(f"  Last 500 avg loss:  {fixed_loss_last:.2f}")
print(f"  % steps loss>10:    {fixed_high_pct:.1f}%")
print(f"  Weight norm range:  {np.min(fixed['x']):.1f} - {np.max(fixed['x']):.1f}")
print(f"\nAdaptive α₀=2000:")
print(f"  First 500 avg loss: {adapt_loss_first:.2f}")
print(f"  Last 500 avg loss:  {adapt_loss_last:.2f}")  
print(f"  % steps loss>10:    {adapt_high_pct:.1f}%")
print(f"  Weight norm range:  {np.min(adaptive['x']):.1f} - {np.max(adaptive['x']):.1f}")
print(f"  Final α:            {adaptive['alpha'][-1]:.1f}")
print(f"  α trajectory: starts={adaptive['alpha'][0]:.0f}, min={np.min(adaptive['alpha']):.1f}, max={np.max(adaptive['alpha']):.1f}")
