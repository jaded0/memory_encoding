"""LQR and H-infinity controllers for ephemeral weight plasticity regulation.

Designed for the discrete-time scalar plant:
    δx(t+1) = A·δx(t) + B·δu(t) + E·δd(t)

Control output: u(t) = α(t) - α₀ (deviation from nominal plasticity)
"""

import numpy as np
from scipy.linalg import solve_discrete_are


class BaseController:
    """Base class for all controllers."""

    def __init__(self, alpha0, alpha_min, alpha_max):
        self.alpha0 = alpha0
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    def clamp(self, alpha):
        return max(self.alpha_min, min(self.alpha_max, alpha))

    def compute_alpha(self, x, x_ref=None):
        raise NotImplementedError

    def reset(self):
        """Reset any internal state (for stateful controllers)."""
        pass


class FixedController(BaseController):
    """Baseline: fixed α = α₀."""

    def compute_alpha(self, x, x_ref=None):
        return self.alpha0


class LQRController(BaseController):
    """Discrete-time LQR state feedback: u(t) = -K·(x(t) - x_ref).

    Minimizes J = Σ [Q·(x-x_ref)² + R·u²].
    For the scalar case, K is computed via solve_discrete_are.
    """

    def __init__(self, A, B, Q, R, alpha0, alpha_min, alpha_max, x_ref=0.0):
        super().__init__(alpha0, alpha_min, alpha_max)
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.x_ref = x_ref

        # Solve discrete algebraic Riccati equation
        # For scalar: A'PA - P - A'PB(R + B'PB)^{-1}B'PA + Q = 0
        A_mat = np.array([[A]])
        B_mat = np.array([[B]])
        Q_mat = np.array([[Q]])
        R_mat = np.array([[R]])

        P = solve_discrete_are(A_mat, B_mat, Q_mat, R_mat)
        self.P = P[0, 0]

        # Gain: K = (R + B'PB)^{-1} B'PA
        self.K = (B * self.P * A) / (R + B * self.P * B)

        # Closed-loop eigenvalue
        self.A_cl = A - B * self.K

    def compute_alpha(self, x, x_ref=None):
        if x_ref is None:
            x_ref = self.x_ref
        u = -self.K * (x - x_ref)
        alpha = self.alpha0 + u
        return self.clamp(alpha)

    def __repr__(self):
        return (f"LQRController(K={self.K:.4f}, A_cl={self.A_cl:.4f}, "
                f"α₀={self.alpha0}, range=[{self.alpha_min}, {self.alpha_max}])")


class HinfController(BaseController):
    """Discrete-time H-infinity state feedback for the scalar plant.

    Minimizes the worst-case gain from disturbance d to performance z = [√Q·x, √R·u].

    For the scalar SISO case, the H-inf Riccati equation is:
        P = A²P / (1 + P(B²/R - E²/γ²)) + Q

    where γ is the disturbance attenuation level (not forgetting rate!).
    We solve this iteratively via bisection on γ.
    """

    def __init__(self, A, B, E, Q, R, alpha0, alpha_min, alpha_max, x_ref=0.0,
                 gamma_lb=None, gamma_ub=None, tol=1e-6, max_iter=100):
        super().__init__(alpha0, alpha_min, alpha_max)
        self.A_plant = A
        self.B = B
        self.E = E
        self.Q = Q
        self.R = R
        self.x_ref = x_ref

        # Find optimal gamma via bisection
        self.gamma_opt, self.P, self.K = self._solve_hinf(
            A, B, E, Q, R, gamma_lb, gamma_ub, tol, max_iter
        )
        self.A_cl = A - B * self.K

    def _solve_hinf_riccati(self, A, B, E, Q, R, gamma):
        """Solve the discrete H-inf Riccati for a given gamma.

        Returns P if solution exists with P > 0, else None.
        """
        # For scalar system, iterate the Riccati recursion to convergence
        P = Q  # initial guess
        for _ in range(1000):
            denom = 1 + P * (B**2 / R - E**2 / gamma**2)
            if denom <= 0:
                return None  # No stabilizing solution
            P_new = A**2 * P / denom + Q
            if abs(P_new - P) < 1e-12:
                P = P_new
                break
            P = P_new
        else:
            # Check convergence
            denom = 1 + P * (B**2 / R - E**2 / gamma**2)
            if denom <= 0 or P < 0:
                return None

        if P < 0:
            return None

        # Verify closed-loop stability
        K = B * A * P / (R + B**2 * P - (B * E)**2 * P / (gamma**2 + E**2 * P - 1e-30))
        # Simplified for scalar: K = B*A*P / (R + B²P) when gamma is large enough
        K_simple = B * A * P / (R + B**2 * P)
        A_cl = A - B * K_simple
        if abs(A_cl) >= 1.0:
            return None

        return P

    def _solve_hinf(self, A, B, E, Q, R, gamma_lb, gamma_ub, tol, max_iter):
        """Find optimal gamma via bisection and return (gamma, P, K)."""
        # If no bounds given, find them
        if gamma_ub is None:
            gamma_ub = max(10.0, 100 * abs(E))
        if gamma_lb is None:
            gamma_lb = tol

        # Find a working upper bound
        for _ in range(20):
            P = self._solve_hinf_riccati(A, B, E, Q, R, gamma_ub)
            if P is not None:
                break
            gamma_ub *= 2
        else:
            # Fallback to LQR if H-inf fails
            print("WARNING: H-inf Riccati did not converge, falling back to LQR-like gain")
            A_mat = np.array([[A]])
            B_mat = np.array([[B]])
            Q_mat = np.array([[Q]])
            R_mat = np.array([[R]])
            P_mat = solve_discrete_are(A_mat, B_mat, Q_mat, R_mat)
            P = P_mat[0, 0]
            K = (B * A * P) / (R + B**2 * P)
            return gamma_ub, P, K

        # Bisection: find smallest gamma such that Riccati has a solution
        for _ in range(max_iter):
            gamma_mid = (gamma_lb + gamma_ub) / 2
            P_mid = self._solve_hinf_riccati(A, B, E, Q, R, gamma_mid)
            if P_mid is not None:
                gamma_ub = gamma_mid
                P = P_mid
            else:
                gamma_lb = gamma_mid
            if gamma_ub - gamma_lb < tol:
                break

        # Compute gain at optimal gamma
        K = (B * A * P) / (R + B**2 * P)
        return gamma_ub, P, K

    def compute_alpha(self, x, x_ref=None):
        if x_ref is None:
            x_ref = self.x_ref
        u = -self.K * (x - x_ref)
        alpha = self.alpha0 + u
        return self.clamp(alpha)

    def __repr__(self):
        return (f"HinfController(K={self.K:.4f}, γ_opt={self.gamma_opt:.4f}, "
                f"A_cl={self.A_cl:.4f}, α₀={self.alpha0})")


class LQRController2D(BaseController):
    """LQR for the 2nd-order plant with state [‖w_eph‖, ‖g‖]."""

    def __init__(self, A, B, Q, R, alpha0, alpha_min, alpha_max, x_ref=None):
        super().__init__(alpha0, alpha_min, alpha_max)
        self.A = np.array(A)
        self.B = np.array(B).reshape(2, 1)
        self.Q = np.array(Q) if np.ndim(Q) == 2 else np.diag(Q)
        self.R = np.array([[R]]) if np.isscalar(R) else np.array(R)
        self.x_ref = np.array(x_ref) if x_ref is not None else np.zeros(2)

        P = solve_discrete_are(self.A, self.B, self.Q, self.R)
        self.P = P

        # K = (R + B'PB)^{-1} B'PA
        BtPB = self.B.T @ P @ self.B
        BtPA = self.B.T @ P @ self.A
        self.K = np.linalg.solve(self.R + BtPB, BtPA).flatten()

        self.A_cl = self.A - self.B @ self.K.reshape(1, 2)
        self.eigenvalues_cl = np.linalg.eigvals(self.A_cl)

    def compute_alpha(self, x, x_ref=None):
        if x_ref is None:
            x_ref = self.x_ref
        state = np.array(x) - x_ref
        u = -self.K @ state
        alpha = self.alpha0 + u
        return self.clamp(alpha)


def design_lqr(plant_model, Q=1.0, R=0.01, alpha_min=1.0, alpha_max=1e5, x_ref=None):
    """Convenience function to design an LQR controller from a PlantModel."""
    if x_ref is None:
        x_ref = plant_model.x_bar if plant_model.x_bar is not None else 0.0
    return LQRController(
        A=plant_model.A, B=plant_model.B, Q=Q, R=R,
        alpha0=plant_model.alpha0,
        alpha_min=alpha_min, alpha_max=alpha_max,
        x_ref=x_ref,
    )


def design_hinf(plant_model, Q=1.0, R=0.01, alpha_min=1.0, alpha_max=1e5, x_ref=None):
    """Convenience function to design an H-inf controller from a PlantModel."""
    if x_ref is None:
        x_ref = plant_model.x_bar if plant_model.x_bar is not None else 0.0
    return HinfController(
        A=plant_model.A, B=plant_model.B, E=plant_model.E, Q=Q, R=R,
        alpha0=plant_model.alpha0,
        alpha_min=alpha_min, alpha_max=alpha_max,
        x_ref=x_ref,
    )


class AdaptiveController(BaseController):
    """Adaptive controller that adjusts alpha based on recent loss trend.

    Increases alpha when loss is decreasing (safe to be more aggressive),
    decreases alpha when loss is increasing or high (back off to safety).
    This handles the bifurcation: stays below the instability threshold.
    """

    def __init__(self, alpha0, alpha_min, alpha_max, loss_target=2.0,
                 increase_rate=1.05, decrease_rate=0.5, loss_window=20):
        super().__init__(alpha0, alpha_min, alpha_max)
        self.loss_target = loss_target
        self.increase_rate = increase_rate
        self.decrease_rate = decrease_rate
        self.loss_window = loss_window
        self._losses = []
        self._current_alpha = alpha0

    def compute_alpha(self, x, x_ref=None):
        """x is expected to be the current loss value."""
        self._losses.append(x)

        if len(self._losses) < 2:
            return self._current_alpha

        recent = self._losses[-self.loss_window:]
        mean_loss = sum(recent) / len(recent)

        if mean_loss > self.loss_target * 2:
            # Loss is very high — aggressively reduce alpha
            self._current_alpha *= self.decrease_rate
        elif mean_loss > self.loss_target:
            # Loss is above target — gently reduce alpha
            self._current_alpha *= 0.9
        elif len(self._losses) > self.loss_window:
            # Loss is at or below target — can we push alpha higher?
            old_mean = sum(self._losses[-2*self.loss_window:-self.loss_window]) / self.loss_window if len(self._losses) > 2*self.loss_window else mean_loss
            if mean_loss <= old_mean:
                # Loss is trending down, safe to increase
                self._current_alpha *= self.increase_rate

        self._current_alpha = max(self.alpha_min, min(self.alpha_max, self._current_alpha))
        return self._current_alpha

    def reset(self):
        self._losses = []
        self._current_alpha = self.alpha0
