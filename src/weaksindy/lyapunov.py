"""Continuous variational Benettin tangent-space Lyapunov integrator and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class LyapunovResult:
    """Result container for variational Benettin Lyapunov spectrum integration.

    Attributes
    ----------
    lyapunov_exponents : np.ndarray of shape (n_exponents,)
        Converged Lyapunov exponents ordered from largest to smallest.
    lambda_max : float
        Largest Lyapunov exponent ``lyapunov_exponents[0]``.
    t_history : np.ndarray of shape (n_renorms,)
        Post-transient renormalization timestamps.
    spectrum_history : np.ndarray of shape (n_renorms, n_exponents)
        Running cumulative Lyapunov spectrum estimates across integration time.
    final_state : np.ndarray of shape (n_dim,)
        Terminal state vector ``y(T_trans + T_run)``.
    """

    lyapunov_exponents: np.ndarray
    lambda_max: float
    t_history: np.ndarray
    spectrum_history: np.ndarray
    final_state: np.ndarray


def _wrap_vector_field(
    rhs: Callable[..., np.ndarray],
    t0: float,
    y0: np.ndarray,
) -> Callable[[float, np.ndarray], np.ndarray]:
    """Normalize ``rhs(t, y)`` or ``rhs(y)`` into a standard ``f(t, y)`` callable."""
    try:
        _ = rhs(t0, y0)
        return lambda t, y: np.asarray(rhs(t, y), dtype=np.float64)
    except TypeError:
        return lambda t, y: np.asarray(rhs(y), dtype=np.float64)


def _wrap_jacobian(
    jacobian: Callable[..., np.ndarray],
    t0: float,
    y0: np.ndarray,
) -> Callable[[float, np.ndarray], np.ndarray]:
    """Normalize ``jacobian(t, y)`` or ``jacobian(y)`` into ``J(t, y)``."""
    try:
        _ = jacobian(t0, y0)
        return lambda t, y: np.asarray(jacobian(t, y), dtype=np.float64)
    except TypeError:
        return lambda t, y: np.asarray(jacobian(y), dtype=np.float64)


def _rk4_step_state(
    f: Callable[[float, np.ndarray], np.ndarray],
    t: float,
    y: np.ndarray,
    dt: float,
) -> np.ndarray:
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def compute_benettin_lyapunov(
    rhs: Callable[..., np.ndarray],
    y0: np.ndarray | list[float],
    *,
    n_exponents: int = 1,
    t_trans: float = 20.0,
    t_run: float = 80.0,
    dt: float = 0.005,
    dt_renorm: float = 0.1,
    jacobian: Callable[..., np.ndarray] | None = None,
    eps: float = 1e-8,
    seed: int = 42,
) -> LyapunovResult:
    """Integrate the tangent-space Lyapunov spectrum via the continuous Benettin QR algorithm.

    Co-integrates the nonlinear flow ``dy/dt = f(t, y)`` and an orthonormal ``k``-frame
    of tangent perturbations ``dV/dt = J(t, y) V`` via classical fourth-order Runge–Kutta
    (RK4), applying periodic Gram–Schmidt / QR orthonormalization every ``dt_renorm``
    time units (Benettin et al., 1980).

    Parameters
    ----------
    rhs : callable
        System vector field ``f(t, y)`` or ``f(y)`` returning shape ``(d,)``.
    y0 : array-like of shape (d,)
        Initial state vector.
    n_exponents : int, default=1
        Number of leading Lyapunov exponents ``1 <= k <= d`` to compute.
    t_trans : float, default=20.0
        Initial transient integration duration before tangent accumulation starts.
    t_run : float, default=80.0
        Post-transient integration horizon over which stretching factors are averaged.
    dt : float, default=0.005
        RK4 integration time step.
    dt_renorm : float, default=0.1
        Interval between successive QR orthonormalizations (``dt_renorm >= dt``).
    jacobian : callable or None, default=None
        Optional analytical Jacobian ``J(t, y)`` or ``J(y)`` of shape ``(d, d)``.
        If ``None``, directional Jacobian-vector products ``(f(t, y + eps * v) - f(t, y)) / eps``
        are evaluated automatically without materializing ``J``.
    eps : float, default=1e-8
        Finite-difference perturbation step used when ``jacobian is None``.
    seed : int, default=42
        Random seed for initializing the orthonormal tangent frame.

    Returns
    -------
    result : LyapunovResult
        Structured result containing ``lyapunov_exponents``, ``lambda_max``,
        ``t_history``, ``spectrum_history``, and ``final_state``.
    """
    y_curr = np.atleast_1d(np.asarray(y0, dtype=np.float64)).copy()
    if y_curr.ndim != 1:
        raise ValueError(f"Initial state y0 must be 1D; got shape {y_curr.shape}.")
    d = y_curr.size

    if not (1 <= n_exponents <= d):
        raise ValueError(f"n_exponents must satisfy 1 <= n_exponents <= {d}; got {n_exponents}.")
    if dt <= 0.0 or t_run <= 0.0 or t_trans < 0.0 or dt_renorm < dt:
        raise ValueError(
            "Require dt > 0, t_run > 0, t_trans >= 0, and dt_renorm >= dt."
        )

    f = _wrap_vector_field(rhs, 0.0, y_curr)
    jac_fn = _wrap_jacobian(jacobian, 0.0, y_curr) if jacobian is not None else None

    # 1. Integrate transient trajectory
    t = 0.0
    n_trans_steps = int(round(t_trans / dt))
    for _ in range(n_trans_steps):
        y_curr = _rk4_step_state(f, t, y_curr, dt)
        t += dt
    t_start = t

    # 2. Initialize orthonormal tangent frame V in R^(d x k)
    rng = np.random.default_rng(seed)
    V_init = rng.standard_normal((d, n_exponents))
    V, _ = np.linalg.qr(V_init, mode="reduced")

    def variational_rhs(t_val: float, y_vec: np.ndarray, V_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fy = f(t_val, y_vec)
        if jac_fn is not None:
            J = jac_fn(t_val, y_vec)
            dV = J @ V_mat
        else:
            dV = np.empty_like(V_mat)
            for j in range(n_exponents):
                v_col = V_mat[:, j]
                v_norm = float(np.linalg.norm(v_col))
                if v_norm == 0.0:
                    dV[:, j] = 0.0
                else:
                    step_eps = eps / v_norm
                    f_pert = f(t_val, y_vec + step_eps * v_col)
                    dV[:, j] = (f_pert - fy) / step_eps
        return fy, dV

    steps_per_renorm = max(1, int(round(dt_renorm / dt)))
    eff_dt_renorm = steps_per_renorm * dt
    n_renorms = max(1, int(round(t_run / eff_dt_renorm)))

    log_stretching_sum = np.zeros(n_exponents, dtype=np.float64)
    t_history = np.empty(n_renorms, dtype=np.float64)
    spectrum_history = np.empty((n_renorms, n_exponents), dtype=np.float64)

    for r in range(n_renorms):
        for _ in range(steps_per_renorm):
            ky1, kV1 = variational_rhs(t, y_curr, V)
            ky2, kV2 = variational_rhs(t + 0.5 * dt, y_curr + 0.5 * dt * ky1, V + 0.5 * dt * kV1)
            ky3, kV3 = variational_rhs(t + 0.5 * dt, y_curr + 0.5 * dt * ky2, V + 0.5 * dt * kV2)
            ky4, kV4 = variational_rhs(t + dt, y_curr + dt * ky3, V + dt * kV3)

            y_curr = y_curr + (dt / 6.0) * (ky1 + 2.0 * ky2 + 2.0 * ky3 + ky4)
            V = V + (dt / 6.0) * (kV1 + 2.0 * kV2 + 2.0 * kV3 + kV4)
            t += dt

        if n_exponents == 1:
            v_norm = float(np.linalg.norm(V[:, 0]))
            v_norm_safe = max(v_norm, 1e-300)
            log_stretching_sum[0] += np.log(v_norm_safe)
            V[:, 0] = V[:, 0] / v_norm_safe
        else:
            Q, R = np.linalg.qr(V, mode="reduced")
            diag_R = np.diag(R)
            signs = np.where(diag_R >= 0.0, 1.0, -1.0)
            Q = Q * signs[np.newaxis, :]
            radii = np.maximum(np.abs(diag_R), 1e-300)
            log_stretching_sum += np.log(radii)
            V = Q

        elapsed = t - t_start
        t_history[r] = elapsed
        spectrum_history[r] = log_stretching_sum / elapsed

    final_exponents = spectrum_history[-1].copy()
    return LyapunovResult(
        lyapunov_exponents=final_exponents,
        lambda_max=float(final_exponents[0]),
        t_history=t_history,
        spectrum_history=spectrum_history,
        final_state=y_curr,
    )


def rosenstein_lyapunov(
    time_series: np.ndarray,
    dt: float,
    m: int = 2,
    tau: int = 1,
    theiler_window: int = 50,
    max_steps: int = 100,
) -> tuple[float, np.ndarray]:
    """Estimate largest Lyapunov exponent from scalar delay coordinates (Rosenstein et al., 1993).

    Provided as a diagnostic baseline: on overdamped relaxing trajectories or near
    numerical solver tolerance floors, delay-coordinate nearest-neighbor divergence can
    exhibit spurious positive slopes even when ``compute_benettin_lyapunov`` proves
    global asymptotic stability (``lambda_max < 0``).

    Returns
    -------
    slope : float
        Linear least-squares slope of the mean log-divergence curve ``<ln d(k dt)>``.
    divergence : np.ndarray of shape (max_steps,)
        Mean logarithmic neighbor divergence curve over separation steps ``0 .. max_steps - 1``.
    """
    ts = np.asarray(time_series, dtype=np.float64).ravel()
    n = ts.size
    n_embed = n - (m - 1) * tau
    if n_embed <= 2 * theiler_window + 2:
        raise ValueError(
            f"Time series length {n} is too short for m={m}, tau={tau}, theiler_window={theiler_window}."
        )

    idx = np.arange(n_embed)[:, np.newaxis] + np.arange(m)[np.newaxis, :] * tau
    X_emb = ts[idx]

    # Pairwise Euclidean distances with Theiler window exclusion
    diff = X_emb[:, np.newaxis, :] - X_emb[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    band = np.abs(np.arange(n_embed)[:, np.newaxis] - np.arange(n_embed)[np.newaxis, :]) <= theiler_window
    dists[band] = np.inf
    nn_indices = np.argmin(dists, axis=1)

    d_avg = np.full(max_steps, np.nan, dtype=np.float64)
    base_i = np.arange(n_embed)
    for k in range(max_steps):
        valid = (base_i + k < n_embed) & (nn_indices + k < n_embed)
        if not np.any(valid):
            break
        sep = np.linalg.norm(X_emb[base_i[valid] + k] - X_emb[nn_indices[valid] + k], axis=-1)
        pos = sep > 0.0
        if np.any(pos):
            d_avg[k] = float(np.mean(np.log(sep[pos])))

    valid_curve = np.isfinite(d_avg)
    t_sep = np.arange(max_steps, dtype=np.float64)[valid_curve] * float(dt)
    slope, _ = np.polyfit(t_sep, d_avg[valid_curve], 1)
    return float(slope), d_avg
