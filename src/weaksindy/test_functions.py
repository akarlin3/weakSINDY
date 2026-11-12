"""Compactly supported C^p tensor-product test functions and N-D weak-form projection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
from numpy.polynomial import Polynomial
from scipy.linalg import toeplitz
from scipy.signal import fftconvolve

from .operators import extract_spatial_steps, extract_time_step, gl_weights, riesz_weights

if TYPE_CHECKING:
    from .library import BaseCandidateLibrary


def bump_test_function(
    window_size: int,
    degree: int = 4,
    dt: float = 1.0,
    normalize: bool = True,
    diff_order: int = 0,
) -> np.ndarray:
    """Construct a compactly supported ``C^p`` polynomial bump kernel (or its derivative).

    Evaluates ``phi(x_k) = C_p * (1 - xi_k^2)^p`` (or its exact analytical ``diff_order``-th
    derivative ``d^m phi / dx^m``) at ``xi_k = 2k / H - 1 in [-1, 1]`` for ``k = 0, ..., H``
    (``H = window_size``, ``p = degree``). At both window boundaries ``xi in {-1, +1}``,
    ``phi`` and its first ``p - 1`` derivatives vanish identically.

    Parameters
    ----------
    window_size : int
        Number of discrete intervals ``H >= 2`` (kernel length is ``H + 1``).
    degree : int, default=4
        Polynomial smoothness exponent ``p >= 1``. Must satisfy ``degree >= diff_order``.
    dt : float, default=1.0
        Grid spacing used for unit quadrature normalization and physical derivative scaling.
    normalize : bool, default=True
        If ``True``, scales ``phi`` so the 0th-order bump satisfies ``sum(phi) * dt == 1.0``.
    diff_order : int, default=0
        Analytical derivative order ``m >= 0`` with respect to the physical coordinate.

    Returns
    -------
    phi_deriv : np.ndarray of shape (window_size + 1,)
        Discrete test-function weights (or ``m``-th physical derivative weights).
    """
    if window_size < 2:
        raise ValueError(f"window_size H must be >= 2; got {window_size}.")
    if degree < 1:
        raise ValueError(f"Test-function degree p must be >= 1; got {degree}.")
    if diff_order < 0:
        raise ValueError(f"diff_order must be >= 0; got {diff_order}.")
    if diff_order > degree:
        raise ValueError(
            f"Test-function degree ({degree}) must be >= diff_order ({diff_order}) "
            "to eliminate integration-by-parts boundary terms."
        )
    step = extract_time_step(dt)

    xi = (2.0 * np.arange(window_size + 1, dtype=np.float64) / float(window_size)) - 1.0
    phi0 = np.maximum(0.0, 1.0 - xi**2) ** int(degree)
    if normalize:
        integral = float(np.sum(phi0)) * step
        if integral <= 0.0:
            raise ValueError("Degenerate test function with non-positive quadrature sum.")
        norm_scale = 1.0 / integral
    else:
        norm_scale = 1.0

    if diff_order == 0:
        return phi0 * norm_scale

    base_poly = Polynomial([1.0, 0.0, -1.0]) ** int(degree)
    deriv_poly = base_poly.deriv(int(diff_order))
    vals = np.asarray(deriv_poly(xi), dtype=np.float64)
    if diff_order < degree:
        vals[0] = 0.0
        vals[-1] = 0.0
    half_width = 0.5 * float(window_size) * step
    return vals * norm_scale * (half_width ** (-int(diff_order)))


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


def resolve_spatial_windows(
    spatial_shape: Sequence[int],
    spatial_window_size: int | Sequence[int] | None = None,
    spatial_stride: int | Sequence[int] | None = None,
) -> list[list[tuple[int, int]]]:
    """Resolve overlapping 1D integration windows along each spatial axis ``d = 1, ..., D``."""
    n_dims = len(spatial_shape)
    if spatial_window_size is None:
        raw_H = [min(20, max(4, (int(n_ax) - 2) // 2)) for n_ax in spatial_shape]
    elif np.ndim(spatial_window_size) == 0:
        raw_H = [int(spatial_window_size)] * n_dims  # type: ignore[arg-type]
    else:
        if len(spatial_window_size) != n_dims:  # type: ignore[arg-type]
            raise ValueError(
                f"spatial_window_size length ({len(spatial_window_size)}) must match spatial dims ({n_dims})."  # type: ignore[arg-type]
            )
        raw_H = [int(h) for h in spatial_window_size]  # type: ignore[union-attr]

    if spatial_stride is None:
        raw_S = [max(1, h // 4) for h in raw_H]
    elif np.ndim(spatial_stride) == 0:
        raw_S = [int(spatial_stride)] * n_dims  # type: ignore[arg-type]
    else:
        if len(spatial_stride) != n_dims:  # type: ignore[arg-type]
            raise ValueError(
                f"spatial_stride length ({len(spatial_stride)}) must match spatial dims ({n_dims})."  # type: ignore[arg-type]
            )
        raw_S = [int(s) for s in spatial_stride]  # type: ignore[union-attr]

    all_wins: list[list[tuple[int, int]]] = []
    for n_ax, h_val, s_val in zip(spatial_shape, raw_H, raw_S):
        eff_h = h_val
        eff_s = s_val
        if n_ax <= eff_h + 1:
            eff_h = max(2, (int(n_ax) - 1) // 2)
            eff_s = max(1, min(eff_s, eff_h // 2))
        wins = generate_windows(n_steps=int(n_ax), window_size=eff_h, stride=eff_s, k_start=0)
        all_wins.append(wins)
    return all_wins


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


def build_spatial_weak_weights(
    n_grid: int,
    windows: Sequence[tuple[int, int]],
    dx: float,
    degree: int = 4,
    diff_order: int = 0,
    fractional_order: float | None = None,
    riesz_order: float | None = None,
    toeplitz_cutoff: int = 4096,
) -> np.ndarray:
    """Assemble 1D spatial weak-form quadrature weights transferring spatial operators onto ``phi``.

    Supports:
    - Integer spatial derivatives ``d^m / dx^m`` via exact analytical integration by parts
      ``(-1)^m * phi^(m)(x) * dx``.
    - Causal Grünwald–Letnikov spatial fractional derivatives ``D_x^beta`` via discrete
      summation by parts.
    - Symmetric Riesz–Grünwald spatial fractional diffusion ``(-d^2/dx^2)^(beta/2)`` via
      self-adjoint centered fractional Toeplitz projection.
    """
    step = extract_time_step(dx)
    if len(windows) == 0:
        raise ValueError("At least one spatial window is required.")
    H = windows[0][1] - windows[0][0]

    if fractional_order is not None:
        phi0 = bump_test_function(window_size=H, degree=degree, dt=step, normalize=True, diff_order=0)
        _, Psi_gl = build_weak_weights(
            n_steps=n_grid,
            windows=windows,
            alpha=float(fractional_order),
            phi=phi0,
            dt=step,
            toeplitz_cutoff=toeplitz_cutoff,
        )
        return Psi_gl

    if riesz_order is not None:
        beta_f = float(riesz_order)
        phi0 = bump_test_function(window_size=H, degree=degree, dt=step, normalize=True, diff_order=0)
        Phi0 = np.zeros((n_grid, len(windows)), dtype=np.float64)
        for j, (k_a, k_b) in enumerate(windows):
            Phi0[k_a : k_b + 1, j] = phi0 * step
        if beta_f == 0.0:
            return Phi0
        g = riesz_weights(beta_f, n_grid)
        scale = step**beta_f
        if n_grid <= toeplitz_cutoff:
            R = toeplitz(g, g)
            return (R @ Phi0) / scale
        kernel = np.concatenate([g[:0:-1], g])[:, np.newaxis]
        conv = fftconvolve(Phi0, kernel, mode="full", axes=0)[n_grid - 1 : 2 * n_grid - 1]
        return conv / scale

    eff_deg = max(int(degree), int(diff_order) + 2) if diff_order > 0 else int(degree)
    phi_m = bump_test_function(
        window_size=H,
        degree=eff_deg,
        dt=step,
        normalize=True,
        diff_order=int(diff_order),
    )
    sign = -1.0 if (int(diff_order) % 2 == 1) else 1.0
    W = np.zeros((n_grid, len(windows)), dtype=np.float64)
    for j, (k_a, k_b) in enumerate(windows):
        if k_a < 0 or k_b >= n_grid or (k_b - k_a + 1) != phi_m.size:
            raise ValueError(
                f"Spatial window ({k_a}, {k_b}) incompatible with n_grid={n_grid} and H={H}."
            )
        W[k_a : k_b + 1, j] = sign * phi_m * step
    return W


def contract_tensor_weak(
    field: np.ndarray,
    axis_weights: Sequence[np.ndarray],
) -> np.ndarray:
    """Contract an ``(N_0, ..., N_D, n_cols)`` tensor with separable 1D test matrices ``(N_a, M_a)``.

    Executes sequential mode-n Kronecker contractions without materializing the dense
    Kronecker product, returning shape ``(M_0 * M_1 * ... * M_D, n_cols)``.
    """
    arr = np.asarray(field, dtype=np.float64)
    n_axes = len(axis_weights)
    if arr.ndim == n_axes:
        arr = arr[..., np.newaxis]
    if arr.ndim != n_axes + 1:
        raise ValueError(
            f"Expected field with {n_axes} or {n_axes + 1} dims for {n_axes} axis weights; got {arr.shape}."
        )
    curr = arr
    for ax, W in enumerate(axis_weights):
        if curr.shape[ax] != W.shape[0]:
            raise ValueError(
                f"Axis {ax} dimension mismatch: field has {curr.shape[ax]}, weight matrix has {W.shape[0]}."
            )
        contracted = np.tensordot(curr, W, axes=([ax], [0]))
        curr = np.moveaxis(contracted, -1, ax)
    return curr.reshape(-1, curr.shape[-1])


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
    dx: float | np.ndarray | Sequence[float | np.ndarray] | None = None,
    library: BaseCandidateLibrary | None = None,
    window_size: int = 50,
    stride: int = 5,
    degree: int = 4,
    spatial_window_size: int | Sequence[int] | None = None,
    spatial_stride: int | Sequence[int] | None = None,
    spatial_degree: int | None = None,
    k_start: int = 0,
    k_end: int | None = None,
    windows: Sequence[tuple[int, int]] | None = None,
    spatial_windows: Sequence[Sequence[tuple[int, int]]] | None = None,
    subdomain_indices: np.ndarray | Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project N-D state fields and candidate libraries onto separable ``C^p`` weak test functions.

    Supports ODE trajectories ``(N_t, n_vars)`` as well as arbitrary 1D, 2D, and 3D
    spatiotemporal PDE fields ``(N_t, *spatial_dims, n_vars)``. When ``dx``, ``spatial_window_size``,
    ``spatial_windows``, or a spatiotemporal ``library`` (such as ``PDELibrary``) is supplied on
    ``N_t x *spatial_dims x n_vars`` inputs, constructs separable tensor-product test functions
    ``Phi(t, x) = phi_t(t) * prod_d phi_{x_d}(x_d)`` and transfers both temporal Grünwald–Letnikov
    fractional derivatives ``D_t^alpha`` and spatial differential operators strictly onto ``Phi``
    via integration by parts.

    Returns
    -------
    Theta_weak : np.ndarray of shape (n_subdomains, n_features)
        Weak-form projected feature design matrix.
    y_weak : np.ndarray of shape (n_subdomains, n_vars)
        Weak-form projected fractional derivative targets ``<D_t^alpha X, Phi_m>``.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr[:, np.newaxis]
    if X_arr.ndim < 2:
        raise ValueError(f"Expected X to have at least 1 dimension; got shape {X_arr.shape}.")

    nt = X_arr.shape[0]
    dt = extract_time_step(t)

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

    phi_t = bump_test_function(window_size=eff_H, degree=degree, dt=dt, normalize=True)
    Phi_t, Psi_t = build_weak_weights(
        n_steps=nt,
        windows=win_list,
        alpha=alpha,
        phi=phi_t,
        dt=dt,
    )

    spatial_ndim = X_arr.ndim - 2
    sp_deg = degree if spatial_degree is None else int(spatial_degree)

    use_spatiotemporal = (
        spatial_ndim >= 1
        and (
            dx is not None
            or spatial_window_size is not None
            or spatial_windows is not None
            or (library is not None and getattr(library, "_supports_spatiotemporal_weak", False))
        )
    )

    if use_spatiotemporal:
        sp_shape = X_arr.shape[1:-1]
        if spatial_windows is not None:
            sp_wins = [list(w) for w in spatial_windows]
            if len(sp_wins) != spatial_ndim:
                raise ValueError(
                    f"Expected {spatial_ndim} spatial window lists; got {len(sp_wins)}."
                )
        else:
            sp_wins = resolve_spatial_windows(
                sp_shape,
                spatial_window_size=spatial_window_size,
                spatial_stride=spatial_stride,
            )

        if dx is not None:
            sp_steps = extract_spatial_steps(dx, spatial_ndim)
        elif library is not None and hasattr(library, "_get_spatial_steps"):
            sp_steps = library._get_spatial_steps(spatial_ndim)
        else:
            sp_steps = (1.0,) * spatial_ndim

        Phi_x0 = [
            build_spatial_weak_weights(
                n_grid=sp_shape[d],
                windows=sp_wins[d],
                dx=sp_steps[d],
                degree=sp_deg,
                diff_order=0,
            )
            for d in range(spatial_ndim)
        ]

        y_weak = contract_tensor_weak(X_arr, [Psi_t, *Phi_x0])

        if library is not None:
            if not hasattr(library, "n_features_in_"):
                library.fit(X_arr)
            if hasattr(library, "_project_spatiotemporal_weak"):
                Theta_weak = library._project_spatiotemporal_weak(
                    X_arr,
                    Phi_t=Phi_t,
                    Phi_x0=Phi_x0,
                    sp_wins=sp_wins,
                    sp_steps=sp_steps,
                    sp_degree=sp_deg,
                )
            else:
                Theta_eval = library.transform(X_arr)
                Theta_weak = contract_tensor_weak(Theta_eval, [Phi_t, *Phi_x0])
        else:
            Theta_src = X_arr if Theta is None else np.asarray(Theta, dtype=np.float64)
            if Theta_src.shape[:-1] != X_arr.shape[:-1]:
                raise ValueError(
                    f"Theta shape {Theta_src.shape} must match X spatiotemporal grid {X_arr.shape[:-1]}."
                )
            Theta_weak = contract_tensor_weak(Theta_src, [Phi_t, *Phi_x0])
    else:
        # Temporal-only weak projection (ODEs or pre-evaluated libraries without spatial test windows)
        if library is not None:
            if not hasattr(library, "n_features_in_"):
                library.fit(X_arr)
            Theta_arr = library.transform(X_arr)
        elif Theta is None:
            Theta_arr = X_arr
        else:
            Theta_arr = np.asarray(Theta, dtype=np.float64)
            if Theta_arr.ndim == 1:
                Theta_arr = Theta_arr[:, np.newaxis]
            if Theta_arr.shape[:-1] != X_arr.shape[:-1]:
                raise ValueError(
                    f"Theta shape {Theta_arr.shape} must match X temporal/spatial axes {X_arr.shape}."
                )

        if X_arr.ndim == 2:
            Theta_weak = Phi_t.T @ Theta_arr
            y_weak = Psi_t.T @ X_arr
        else:
            n_feat = Theta_arr.shape[-1]
            n_vars = X_arr.shape[-1]
            theta_proj = np.moveaxis(np.tensordot(Theta_arr, Phi_t, axes=([0], [0])), -1, 0)
            Theta_weak = theta_proj.reshape(-1, n_feat)

            y_proj = np.moveaxis(np.tensordot(X_arr, Psi_t, axes=([0], [0])), -1, 0)
            y_weak = y_proj.reshape(-1, n_vars)

    if subdomain_indices is not None:
        idx = np.asarray(subdomain_indices, dtype=np.intp)
        return Theta_weak[idx], y_weak[idx]

    return Theta_weak, y_weak
