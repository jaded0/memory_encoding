"""Lightweight per-step CSV logger for system identification data collection."""

import csv
import os
import torch


class ControlLogger:
    """Logs per-step training dynamics to CSV for system ID and controller analysis."""

    COLUMNS = ["t", "x", "g_raw", "g_ratio", "alpha", "gamma", "loss", "acc"]

    def __init__(self, log_dir="control_logs", run_name="run"):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, f"{run_name}.csv")
        self._file = open(self.path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.COLUMNS)
        self._file.flush()

    def log(self, t, x, g_raw, g_ratio, alpha, gamma, loss, acc):
        self._writer.writerow([t, x, g_raw, g_ratio, alpha, gamma, loss, acc])
        # Flush every 100 steps to balance I/O and data safety
        if t % 100 == 0:
            self._file.flush()

    def close(self):
        self._file.flush()
        self._file.close()


def compute_ephemeral_weight_norm(rnn):
    """Compute L2 norm of all ephemeral (high-plasticity) weights across layers."""
    total_sq = 0.0
    with torch.no_grad():
        for module in rnn.modules():
            if hasattr(module, 'candidate_weights') and hasattr(module, 'mask'):
                weights = module.candidate_weights.data
                mask_expanded = module.mask.unsqueeze(0).expand_as(weights)
                eph_weights = weights[mask_expanded]
                total_sq += eph_weights.pow(2).sum().item()
    return total_sq ** 0.5


def compute_gradient_norms(rnn):
    """Compute raw gradient norm and ratio of high-plast to low-plast gradient norms.

    Uses the stored update norms from HebbianLinear layers (last_high_plast_update_norm
    and last_low_plast_update_norm) since gradients are zeroed after each DFA step.

    Returns (g_raw, g_ratio) where g_raw is the total update norm and
    g_ratio is high_plast_norm / low_plast_norm.
    """
    high_total = 0.0
    low_total = 0.0
    with torch.no_grad():
        for module in rnn.modules():
            if hasattr(module, 'last_high_plast_update_norm'):
                high_total += module.last_high_plast_update_norm.item() ** 2
            if hasattr(module, 'last_low_plast_update_norm'):
                low_total += module.last_low_plast_update_norm.item() ** 2
    g_high = high_total ** 0.5
    g_low = low_total ** 0.5
    g_raw = (high_total + low_total) ** 0.5
    g_ratio = g_high / g_low if g_low > 1e-12 else float('inf')
    return g_raw, g_ratio
