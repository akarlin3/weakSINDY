"""Compactly supported C^p test functions and weak-form fractional projection."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.linalg import toeplitz
from scipy.signal import fftconvolve

from .operators import extract_time_step, gl_weights


def bump_test_function(
    window_size: int,
    degree: int = 4,
    dt: float = 1.0,
    normalize: bool = True,
) -> np.ndarray:
    """Construct a compactly supported ``C^p`` polynomial bump kernel on ``[0, H]``.

    Evaluates ``phi_k = C_p * (1 - xi_k^2)^p`` at ``xi_k = 2k / H - 1 in [-1, 1]``
    for ``k = 0, ..., H`` (``H = window_size``, ``p = degree``). At both window
    boundaries ``xi in {-1, +1}``, ``phi`` and its first ``p - 1`` derivatives vanish.

    Parameters
    ----------
    window_size : int
        Number of discrete intervals ``H >= 2`` (kernel length is ``H + 1``).
    degree : int, default=4
        Polynomial smoothness exponent ``p >= 1``.
    dt : float, default=1.0
        Temporal grid spacing used for unit quadrature normalization.
    normalize : bool, default=True
        If ``True``, scales ``phi`` so ``sum(phi) * dt == 1.0``.

    Returns
    -------
    phi : np.ndarray of shape (window_size + 1,)
        Discrete test-function quadrature weights.
    """
    if window_size < 2:
        raise ValueError(f"window_size H must be >= 2; got {window_size}.")
    if degree < 1:
        raise ValueError(f"Test-function degree p must be >= 1; got {degree}.")
    step = extract_time_step(dt)

    xi = (2.0 * np.arange(window_size + 1, dtype=np.float64) / float(window_size)) - 1.0
    phi = np.maximum(0.0, 1.0 - xi**2) ** int(degree)
    if normalize:
        integral = float(np.sum(phi)) * step
        if integral <= 0.0:
            raise ValueError("Degenerate test function with non-positive quadrature sum.")
        phi = phi / integral
    return phi


def generate_windows(
    n_steps: int,
    window_size: int = 50,
    stride: int = 5,
    k_start: int = 0,
    k_end: int | None = None,
) -> list[tuple[int, int]]:
    """Generate overlapping discrete window index pairs ``(k_a, k_b)`` with ``k_b = k_a + window_size``.

    All returned windows satisfy ``k_start <= k_a < k_b < k_end`` so slice ``[k_a : k_b + 1]``
    lies strictly within ``[k_start, k_end)``.
    """
    stop = n_steps if k_end is None else min(int(k_end), n_steps)
    if k_start < 0 or k_start >= stop:
        raise ValueError(f"Invalid window bounds k_start={k_start}, k_end={stop}.")
    if window_size < 2:
        raise ValueError(f"window_size must be >= 2; got {window_size}.")
    if stride < 1:
        raise ValueError(f"stride must be >= 1; got {stride}.")
    if stop - k_start <= window_size:
        raise ValueError(
            f"Available trajectory span ({stop - k_start} steps) must exceed window_size ({window_size})."
        )

    windows: list[tuple[int, int]] = []
    for k_a in range(int(k_start), stop - window_size, int(stride)):
        k_b = k_a + window_size
        if k_b < stop:
            windows.append((k_a, k_b))
    return windows


def build_weak_weights(
    n_steps: int,
    windows: Sequence[tuple[int, int]],
    alpha: float,
    phi: np.ndarray,
    dt: float,
    toeplitz_cutoff: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble test-function matrix ``Phi`` and fractional adjoint matrix ``Psi(alpha)``.

    By discrete fractional summation by parts (exact transpose of the causal GL Toeplitz
    operator ``L_alpha``), for every window ``m = [k_a, k_b]``:

    .. math::

        \\sum_{k=k_a}^{k_b} (\\mathcal{D}_t^\\alpha u)_k \\phi_{k - k_a} \\Delta t
        = \\Delta t^{1 - \\alpha} \\sum_{i=0}^{k_b} u_i \\psi_i^{(m)}(\\alpha),
        \\qquad \\psi_i^{(m)}(\\alpha) = \\sum_{k=\\max(k_a, i)}^{k_b} w_{k-i}^{(\\alpha)} \\phi_{k - k_a}.

    Returns
    -------
    Phi_scaled : np.ndarray of shape (n_steps, n_windows)
        Feature projection matrix ``Phi * dt``.
    Psi_scaled : np.ndarray of shape (n_steps, n_windows)
        Adjoint fractional target projection matrix ``Psi(alpha) * dt^(1 - alpha)``.
    """
    step = extract_time_step(dt)
    alpha_f = float(alpha)
    if alpha_f < 0.0 or not np.isfinite(alpha_f):
        raise ValueError(f"Fractional order alpha must be finite and >= 0; got {alpha}.")
    if len(windows) == 0:
        raise ValueError("At least one integration window is required.")

    phi_arr = np.asarray(phi, dtype=np.float64)
    n_win = len(windows)
    Phi_raw = np.zeros((n_steps, n_win), dtype=np.float64)
    max_kb = 0
    for j, (k_a, k_b) in enumerate(windows):
        if k_a < 0 or k_b >= n_steps or (k_b - k_a + 1) != phi_arr.size:
            raise ValueError(
                f"Window ({k_a}, {k_b}) incompatible with n_steps={n_steps} and len(phi)={phi_arr.size}."
            )
        Phi_raw[k_a : k_b + 1, j] = phi_arr
        if k_b > max_kb:
            max_kb = k_b

    Phi_scaled = Phi_raw * step
    if alpha_f == 0.0:
        return Phi_scaled, Phi_scaled.copy()

    n_active = max_kb + 1
    w = gl_weights(alpha_f, n_active)
    Psi_raw = np.zeros((n_steps, n_win), dtype=np.float64)

    if n_active <= toeplitz_cutoff:
        first_row = np.zeros(n_active, dtype=np.float64)
        first_row[0] = w[0]
        # W.T is upper-triangular Toeplitz with first row w and first col [w[0], 0, ...]
        W_T = toeplitz(first_row, w)
        Psi_raw[:n_active] = W_T @ Phi_raw[:n_active]
    else:
        # Adjoint correlation: sum_{k >= i} w_{k - i} Phi_{k, j}
        rev_phi = Phi_raw[:n_active][::-1]
        conv_rev = fftconvolve(rev_phi, w[:, np.newaxis], mode="full", axes=0)[:n_active]
        Psi_raw[:n_active] = conv_rev[::-1]

    Psi_scaled = Psi_raw * (step ** (1.0 - alpha_f))
    return Phi_scaled, Psi_scaled


def weak_noise_amplification_factor(
    alpha: float,
    dt: float,
    window_size: int = 50,
    degree: int = 4,
    n_steps: int = 500,
    k_a: int | None = None,
) -> float:
    """Compute the exact white-noise variance factor ``Var(y_weak) / sigma^2`` for a weak window.

    Evaluates ``||Psi_{:, m}(alpha)||_2^2 = dt^(2 - 2 alpha) * sum_{i=0}^{k_b} psi_i^(m)(alpha)^2``
    for a compactly supported ``C^p`` test function on ``[k_a, k_a + window_size]``.
    """
    step = extract_time_step(dt)
    if k_a is None:
        k_a = max(0, n_steps - window_size - 1)
    k_b = k_a + window_size
    if k_b >= n_steps:
        raise ValueError(f"Window [{k_a}, {k_b}] exceeds n_steps={n_steps}.")
    phi = bump_test_function(window_size=window_size, degree=degree, dt=step, normalize=True)
    _, Psi_scaled = build_weak_weights(n_steps, [(k_a, k_b)], alpha=alpha, phi=phi, dt=step)
    return float(np.sum(Psi_scaled[:, 0] ** 2))


def project_weak_form(
    X: np.ndarray,
    Theta: np.ndarray | None = None,
    t: float | np.ndarray = 1.0,
    alpha: float = 1.0,
    *,
    window_size: int = 50,
    stride: int = 5,
    degree: int = 4,
    k_start: int = 0,
    k_end: int | None = None,
    windows: Sequence[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project state trajectories and candidate libraries onto ``C^p`` weak test functions.

    Transfers the causal Grünwald–Letnikov fractional derivative ``D_t^alpha`` off the
    noisy state ``X`` and onto compactly supported ``C^p`` test functions via discrete
    fractional summation by parts.

    Parameters
    ----------
    X : np.ndarray
        State trajectory of shape ``(nt,)``, ``(nt, n_vars)`` for ODEs, or
        ``(nt, nx, n_vars)`` for 1D spatial PDEs.
    Theta : np.ndarray or None, default=None
        Evaluated candidate feature tensor of shape ``(nt, n_features)`` (ODE) or
        ``(nt, nx, n_features)`` (PDE). If ``None``, ``X`` itself is projected as features.
    t : float or 1D np.ndarray, default=1.0
        Uniform time step ``dt`` or 1D uniform time coordinate array.
    alpha : float, default=1.0
        Fractional temporal differentiation order ``alpha >= 0``.
    window_size : int, default=50
        Test-function window width ``H`` in discrete time steps.
    stride : int, default=5
        Stride ``S`` between consecutive overlapping windows.
    degree : int, default=4
        Smoothness exponent ``p`` of the ``C^p`` bump kernel ``(1 - xi^2)^p``.
    k_start : int, default=0
        Earliest starting time index ``k_a`` for test windows (used to skip ``t=0`` transients).
    k_end : int or None, default=None
        Exclusive upper time index bound for test windows (defaults to ``X.shape[0]``).
    windows : sequence of (int, int) or None, default=None
        Explicit list of ``(k_a, k_b)`` windows; overrides ``window_size``, ``stride``,
        ``k_start``, and ``k_end`` when provided.

    Returns
    -------
    Theta_weak : np.ndarray of shape (n_samples, n_features)
        Weak-form projected feature design matrix, where ``n_samples = n_windows`` (ODE)
        or ``n_windows * nx`` (PDE).
    y_weak : np.ndarray of shape (n_samples, n_vars)
        Weak-form projected fractional derivative targets ``<D_t^alpha X, phi_m>``.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    was_1d = X_arr.ndim == 1
    if was_1d:
        X_arr = X_arr[:, np.newaxis]
    if X_arr.ndim not in (2, 3):
        raise ValueError(f"Expected X to have 1, 2, or 3 dimensions; got shape {X_arr.shape}.")

    nt = X_arr.shape[0]
    dt = extract_time_step(t)

    if Theta is None:
        Theta_arr = X_arr
    else:
        Theta_arr = np.asarray(Theta, dtype=np.float64)
        if Theta_arr.ndim == 1:
            Theta_arr = Theta_arr[:, np.newaxis]
        if Theta_arr.ndim != X_arr.ndim or Theta_arr.shape[0] != nt:
            raise ValueError(
                f"Theta shape {Theta_arr.shape} must match X temporal/spatial axes {X_arr.shape}."
            )
        if X_arr.ndim == 3 and Theta_arr.shape[1] != X_arr.shape[1]:
            raise ValueError(
                f"Spatial dimension mismatch between X ({X_arr.shape[1]}) and Theta ({Theta_arr.shape[1]})."
            )

    if windows is None:
        win_list = generate_windows(
            n_steps=nt,
            window_size=window_size,
            stride=stride,
            k_start=k_start,
            k_end=k_end,
        )
        eff_H = window_size
    else:
        win_list = list(windows)
        if len(win_list) == 0:
            raise ValueError("Provided windows sequence must be non-empty.")
        eff_H = win_list[0][1] - win_list[0][0]

    phi = bump_test_function(window_size=eff_H, degree=degree, dt=dt, normalize=True)
    Phi_scaled, Psi_scaled = build_weak_weights(
        n_steps=nt,
        windows=win_list,
        alpha=alpha,
        phi=phi,
        dt=dt,
    )

    if X_arr.ndim == 2:
        # (nt, n_feat).T @ (nt, n_win) -> (n_feat, n_win) -> (n_win, n_feat)
        Theta_weak = (Phi_scaled.T @ Theta_arr)
        y_weak = (Psi_scaled.T @ X_arr)
    else:
        # 3D PDE: (nt, nx, n_feat) tensordot (nt, n_win) -> (nx, n_feat, n_win)
        # Transpose to (n_win, nx, n_feat) and reshape to (n_win * nx, n_feat)
        n_feat = Theta_arr.shape[-1]
        n_vars = X_arr.shape[-1]
        theta_proj = np.tensordot(Theta_arr, Phi_scaled, axes=([0], [0]))
        Theta_weak = np.transpose(theta_proj, (2, 0, 1)).reshape(-1, n_feat)

        y_proj = np.tensordot(X_arr, Psi_scaled, axes=([0], [0]))
        y_weak = np.transpose(y_proj, (2, 0, 1)).reshape(-1, n_vars)

    return Theta_weak, y_weak
