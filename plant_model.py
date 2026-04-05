"""Analytical plant model and system identification for ephemeral weight dynamics.

The linearized dynamics around a nominal operating point (α₀, γ₀) are:

    δx(t+1) = A·δx(t) + B·δu(t) + E·δd(t)

where:
    x(t) = ‖w_eph(t)‖           (ephemeral weight norm — state)
    u(t) = α(t) - α₀            (control deviation)
    d(t) = ‖g_raw(t)‖ - ḡ       (gradient disturbance)

Analytical structure:
    A = (1-γ)(1 + α₀·H)   where H = ∂‖g‖/∂‖w‖ (effective Hessian)
    B = (1-γ)·ḡ            where ḡ = mean gradient norm
    E = (1-γ)·α₀

Stability requires |A| < 1.
"""

import numpy as np
from scipy.linalg import lstsq
import csv


def load_sysid_csv(path):
    """Load a system ID CSV file into a dict of numpy arrays."""
    data = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for key in rows[0]:
        data[key] = np.array([float(r[key]) for r in rows])
    return data


class PlantModel:
    """Linearized discrete-time plant model for ephemeral weight dynamics."""

    def __init__(self, A, B, E, alpha0, gamma0, g_bar, x_bar=None):
        self.A = A
        self.B = B
        self.E = E
        self.alpha0 = alpha0
        self.gamma0 = gamma0
        self.g_bar = g_bar  # mean gradient norm at operating point
        self.x_bar = x_bar  # mean state at operating point

    @property
    def H(self):
        """Effective Hessian estimated from A."""
        return (self.A / (1 - self.gamma0) - 1) / self.alpha0

    @property
    def is_stable(self):
        return abs(self.A) < 1.0

    def predict(self, x0, u_seq, d_seq):
        """Simulate the linear model forward.

        Args:
            x0: initial state deviation (scalar)
            u_seq: control deviations, shape (T,)
            d_seq: disturbance deviations, shape (T,)

        Returns:
            x_seq: state trajectory, shape (T+1,) starting from x0
        """
        T = len(u_seq)
        x = np.zeros(T + 1)
        x[0] = x0
        for t in range(T):
            x[t + 1] = self.A * x[t] + self.B * u_seq[t] + self.E * d_seq[t]
        return x

    def to_ss_matrices(self):
        """Return (A, B, C, D) as numpy arrays for use with control library.
        Output y = x (full state feedback).
        """
        A = np.array([[self.A]])
        B = np.array([[self.B]])
        C = np.array([[1.0]])
        D = np.array([[0.0]])
        return A, B, C, D


def estimate_plant_analytical(data, alpha0, gamma0):
    """Estimate plant model using analytical structure.

    Given the structure A = (1-γ)(1 + α₀·H), fits H from time-series data.

    Args:
        data: dict with keys 'x', 'g_raw', 'alpha', 'gamma', 'loss'
        alpha0: nominal plasticity multiplier
        gamma0: nominal forgetting rate

    Returns:
        PlantModel instance
    """
    x = data["x"]
    g_raw = data["g_raw"]
    alpha = data["alpha"]

    g_bar = np.mean(g_raw)
    x_bar = np.mean(x)

    # Deviations from operating point
    dx = x - x_bar
    du = alpha - alpha0
    dd = g_raw - g_bar

    # Build regression: dx(t+1) = A·dx(t) + B·du(t) + E·dd(t)
    # We know B = (1-γ)·ḡ and E = (1-γ)·α₀ analytically
    # Only unknown is H in A = (1-γ)(1 + α₀·H)

    T = len(dx) - 1
    B_analytical = (1 - gamma0) * g_bar
    E_analytical = (1 - gamma0) * alpha0

    # Residual after removing known terms: dx(t+1) - B·du(t) - E·dd(t) = A·dx(t)
    lhs = dx[1:] - B_analytical * du[:T] - E_analytical * dd[:T]
    rhs = dx[:T]

    # Least-squares for A
    # lhs = A * rhs  =>  A = (rhs^T rhs)^{-1} rhs^T lhs
    rhs_col = rhs.reshape(-1, 1)
    A_hat, _, _, _ = lstsq(rhs_col, lhs)
    A_hat = A_hat[0]

    return PlantModel(
        A=A_hat,
        B=B_analytical,
        E=E_analytical,
        alpha0=alpha0,
        gamma0=gamma0,
        g_bar=g_bar,
        x_bar=x_bar,
    )


def estimate_plant_unconstrained(data):
    """Unconstrained least-squares fit: dx(t+1) = Â·dx(t) + B̂·du(t) + Ê·dd(t).

    No analytical structure assumed. Used as a sanity check against the
    analytical model.
    """
    x = data["x"]
    g_raw = data["g_raw"]
    alpha = data["alpha"]

    x_bar = np.mean(x)
    g_bar = np.mean(g_raw)
    alpha0 = np.mean(alpha)
    gamma0 = np.mean(data["gamma"])

    dx = x - x_bar
    du = alpha - alpha0
    dd = g_raw - g_bar

    T = len(dx) - 1
    # Regression matrix: [dx(t), du(t), dd(t)]
    Phi = np.column_stack([dx[:T], du[:T], dd[:T]])
    y = dx[1:]

    coeffs, _, _, _ = lstsq(Phi, y)
    A_hat, B_hat, E_hat = coeffs

    return PlantModel(
        A=A_hat,
        B=B_hat,
        E=E_hat,
        alpha0=alpha0,
        gamma0=gamma0,
        g_bar=g_bar,
        x_bar=x_bar,
    )


def validate_model(model, data, warmup=50):
    """Compute prediction error of a fitted model on held-out data.

    Returns:
        rmse: root mean squared error of one-step predictions
        r2: R² score
    """
    x = data["x"]
    g_raw = data["g_raw"]
    alpha = data["alpha"]

    dx = x - model.x_bar
    du = alpha - model.alpha0
    dd = g_raw - model.g_bar

    T = len(dx) - 1
    # One-step predictions
    dx_pred = model.A * dx[:T] + model.B * du[:T] + model.E * dd[:T]
    dx_actual = dx[1:]

    # Skip warmup steps
    dx_pred = dx_pred[warmup:]
    dx_actual = dx_actual[warmup:]

    residuals = dx_actual - dx_pred
    rmse = np.sqrt(np.mean(residuals ** 2))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((dx_actual - np.mean(dx_actual)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    return rmse, r2


class PlantModel2D:
    """Second-order plant model with state [‖w_eph‖, ‖g‖]ᵀ.

    Captures the feedback loop more explicitly:
        x1(t+1) = (1-γ)(x1(t) + α·x2(t))     weight norm dynamics
        x2(t+1) = a21·x1(t) + a22·x2(t) + d(t) gradient norm dynamics
    """

    def __init__(self, A, B, E, alpha0, gamma0, x_bar):
        """
        A: 2x2 state transition matrix
        B: 2x1 control input matrix
        E: 2x1 disturbance input matrix
        """
        self.A = np.array(A)
        self.B = np.array(B).reshape(2, 1)
        self.E = np.array(E).reshape(2, 1)
        self.alpha0 = alpha0
        self.gamma0 = gamma0
        self.x_bar = np.array(x_bar)

    @property
    def is_stable(self):
        eigenvalues = np.linalg.eigvals(self.A)
        return np.all(np.abs(eigenvalues) < 1.0)

    @property
    def eigenvalues(self):
        return np.linalg.eigvals(self.A)

    def predict(self, x0, u_seq, d_seq):
        T = len(u_seq)
        x = np.zeros((T + 1, 2))
        x[0] = x0
        for t in range(T):
            x[t + 1] = self.A @ x[t] + self.B.flatten() * u_seq[t] + self.E.flatten() * d_seq[t]
        return x

    def to_ss_matrices(self):
        C = np.eye(2)
        D = np.zeros((2, 1))
        return self.A, self.B, C, D


def estimate_plant_2d(data, alpha0, gamma0):
    """Estimate 2nd-order plant model from data.

    State: [‖w_eph‖, ‖g_raw‖]
    Control: α - α₀
    """
    x1 = data["x"]      # weight norm
    x2 = data["g_raw"]  # gradient norm
    alpha = data["alpha"]

    x1_bar = np.mean(x1)
    x2_bar = np.mean(x2)
    x_bar = np.array([x1_bar, x2_bar])

    dx1 = x1 - x1_bar
    dx2 = x2 - x2_bar
    du = alpha - alpha0

    T = len(dx1) - 1

    # Fit x1(t+1) = a11*dx1(t) + a12*dx2(t) + b1*du(t)
    Phi1 = np.column_stack([dx1[:T], dx2[:T], du[:T]])
    y1 = dx1[1:]
    c1, _, _, _ = lstsq(Phi1, y1)

    # Fit x2(t+1) = a21*dx1(t) + a22*dx2(t) + b2*du(t)
    Phi2 = np.column_stack([dx1[:T], dx2[:T], du[:T]])
    y2 = dx2[1:]
    c2, _, _, _ = lstsq(Phi2, y2)

    A = np.array([[c1[0], c1[1]], [c2[0], c2[1]]])
    B = np.array([c1[2], c2[2]])
    # Disturbance matrix - identity-like for now (can be refined)
    E = np.array([0.0, 1.0])

    return PlantModel2D(A=A, B=B, E=E, alpha0=alpha0, gamma0=gamma0, x_bar=x_bar)
