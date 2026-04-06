"""Integration module: bridges controllers with the hebby.py training loop.

This module is imported by the modified hebby.py to add controller support.
It handles controller initialization, per-step alpha computation, and CSV logging.
"""

import json
import os
import numpy as np
import torch

from control_logger import ControlLogger, compute_ephemeral_weight_norm, compute_gradient_norms
from controllers import FixedController, LQRController, HinfController, AdaptiveController


def init_controller(args, config):
    """Initialize controller and logger based on command-line args."""
    alpha0 = config["plast_clip"]
    gamma0 = config["forget_rate"]

    alpha_min = getattr(args, "alpha_min", 1.0)
    alpha_max = getattr(args, "alpha_max", alpha0 * 10)

    logger = None
    if hasattr(args, "control_log_dir") and args.control_log_dir:
        run_name = getattr(args, "control_run_name", "run")
        logger = ControlLogger(log_dir=args.control_log_dir, run_name=run_name)

    control_state = {"current_alpha": alpha0, "step": 0}

    mode = getattr(args, "controller_mode", "fixed")

    if mode == "fixed":
        controller = FixedController(alpha0=alpha0, alpha_min=alpha_min, alpha_max=alpha_max)
        return controller, logger, control_state

    if mode == "adaptive":
        controller = AdaptiveController(
            alpha0=alpha0, alpha_min=alpha_min, alpha_max=alpha_max,
            loss_target=2.0, increase_rate=1.02, decrease_rate=0.5,
        )
        print(f"Initialized Adaptive controller: alpha0={alpha0}, target_loss=2.0")
        return controller, logger, control_state

    # LQR/Hinf require system ID results
    sysid_path = getattr(args, "sysid_results", None)
    if sysid_path is None or not os.path.exists(str(sysid_path)):
        print(f"WARNING: sysid_results not found at {sysid_path}, falling back to fixed controller")
        controller = FixedController(alpha0=alpha0, alpha_min=alpha_min, alpha_max=alpha_max)
        return controller, logger, control_state

    with open(sysid_path, "r") as f:
        sysid = json.load(f)

    plant = sysid["analytical"]
    A, B, E = plant["A"], plant["B"], plant["E"]
    x_ref = plant.get("x_bar", 0.0)
    Q = sysid["controller_params"]["Q"]
    R = sysid["controller_params"]["R"]
    alpha_min = sysid["controller_params"].get("alpha_min", alpha_min)
    alpha_max = sysid["controller_params"].get("alpha_max", alpha_max)

    if mode == "lqr":
        controller = LQRController(
            A=A, B=B, Q=Q, R=R,
            alpha0=alpha0, alpha_min=alpha_min, alpha_max=alpha_max, x_ref=x_ref)
        print(f"Initialized LQR controller: {controller}")
    elif mode == "hinf":
        controller = HinfController(
            A=A, B=B, E=E, Q=Q, R=R,
            alpha0=alpha0, alpha_min=alpha_min, alpha_max=alpha_max, x_ref=x_ref)
        print(f"Initialized H-inf controller: {controller}")
    else:
        raise ValueError(f"Unknown controller_mode: {mode}")

    return controller, logger, control_state


def controller_step(controller, logger, control_state, rnn, loss_val, acc_val):
    """Execute one controller step: measure state, compute alpha, log."""
    x = compute_ephemeral_weight_norm(rnn)
    g_raw, g_ratio = compute_gradient_norms(rnn)

    if isinstance(controller, AdaptiveController):
        new_alpha = controller.compute_alpha(loss_val)
    else:
        new_alpha = controller.compute_alpha(x)

    control_state["current_alpha"] = new_alpha
    t = control_state["step"]

    if logger is not None:
        gamma = 0.0
        for module in rnn.modules():
            if hasattr(module, "forgetting_factor"):
                ff = module.forgetting_factor
                mask = module.mask if hasattr(module, "mask") else None
                if mask is not None:
                    gamma = ff[mask].mean().item()
                else:
                    gamma = ff.mean().item()
                break
        logger.log(t, x, g_raw, g_ratio, new_alpha, gamma, loss_val, acc_val)

    control_state["step"] += 1
    return new_alpha


def update_plasticity(rnn, new_alpha):
    """Update plasticity values for all HebbianLinear layers to new_alpha."""
    with torch.no_grad():
        for module in rnn.modules():
            if hasattr(module, "plasticity") and hasattr(module, "mask"):
                if not getattr(module, "is_last_layer", False):
                    module.plasticity.data[module.mask] = new_alpha
