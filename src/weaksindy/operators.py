"""Discrete Grünwald–Letnikov fractional differential operators and integrators."""

from __future__ import annotations

from typing import Callable, Literal, Sequence

import numpy as np
from scipy.linalg import toeplitz
from scipy.signal import fftconvolve
from scipy.special import gammaln

ArrayLike = np.ndarray | list[float] | tuple[float, ...]


def extract_time_step(t: float | int | np.ndarray) -> float:
    """Extract and validate a uniform step size ``dt`` from a scalar or 1D coordinate array."""
    if np.ndim(t) == 0:
        dt = float(t)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"Step size dt must be finite and strictly positive; got {dt}.")
        return dt

    t_arr = np.asarray(t, dtype=float)
    if t_arr.ndim != 1 or t_arr.size < 2:
        raise ValueError("Coordinate array t must be 1D with at least 2 points.")
    diffs = np.diff(t_arr)
    dt = float(diffs[0])
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"Coordinate array t must be strictly increasing; got initial step {dt}.")
    if not np.allclose(diffs, dt, rtol=1e-4, atol=1e-10):
        raise ValueError("Grünwald–Letnikov discretization requires a uniformly spaced grid.")
    return dt


def extract_spatial_steps(
    dx: float | int | np.ndarray | Sequence[float | int | np.ndarray],
    n_dims: int,
) -> tuple[float, ...]:
    """Extract uniform spatial grid spacings ``(dx_1, ..., dx_D)`` across ``n_dims`` axes."""
    if n_dims < 1:
        raise ValueError(f"n_dims must be >= 1; got {n_dims}.")
    if np.ndim(dx) == 0:
        step = extract_time_step(float(dx))  # type: ignore[arg-type]
        return (step,) * n_dims

    if n_dims == 1:
        if isinstance(dx, (list, tuple)) and len(dx) == 1:
            return (extract_time_step(dx[0]),)
        return (extract_time_step(np.asarray(dx, dtype=float)),)

    if isinstance(dx, np.ndarray) and dx.ndim == 1 and dx.size == n_dims:
        return tuple(extract_time_step(float(v)) for v in dx)

    if isinstance(dx, (list, tuple)) and len(dx) == n_dims:
        steps: list[float] = []
        for item in dx:
            if np.ndim(item) == 0:
                steps.append(extract_time_step(float(item)))  # type: ignore[arg-type]
            else:
                steps.append(extract_time_step(np.asarray(item, dtype=float)))
        return tuple(steps)

    raise ValueError(
        f"Cannot resolve {n_dims} spatial step sizes from dx specification of type {type(dx).__name__}."
    )



def gl_weights(alpha: float, n: int) -> np.ndarray:
    """Compute causal Grünwald–Letnikov binomial weights ``w_j^(alpha) = (-1)^j * binom(alpha, j)``.

    Parameters
    ----------
    alpha : float
        Fractional differentiation order (``alpha >= 0``).
    n : int
        Number of discrete weights to generate (``n >= 1``).

    Returns
    -------
    w : np.ndarray of shape (n,)
        Sequence of causal Grünwald–Letnikov convolution weights.
    """
    if n < 1:
        raise ValueError(f"Number of weights n must be >= 1; got {n}.")
    alpha_f = float(alpha)
    if not np.isfinite(alpha_f) or alpha_f < 0.0:
        raise ValueError(f"Fractional order alpha must be finite and >= 0; got {alpha}.")

    w = np.empty(n, dtype=np.float64)
    w[0] = 1.0
    if n > 1:
        j = np.arange(1, n, dtype=np.float64)
        factors = 1.0 - (alpha_f + 1.0) / j
        w[1:] = np.cumprod(factors)
    return w


def gl_operator_matrix(alpha: float, n: int, dt: float = 1.0) -> np.ndarray:
    """Construct the lower-triangular Toeplitz discrete GL operator ``L_alpha = dt^(-alpha) W^(alpha)``."""
    step = extract_time_step(dt)
    w = gl_weights(alpha, n)
    first_row = np.zeros(n, dtype=np.float64)
    first_row[0] = w[0]
    W = toeplitz(w, first_row)
    return W / (step ** float(alpha))


def fractional_derivative_gl(
    X: np.ndarray,
    t: float | np.ndarray = 1.0,
    alpha: float = 1.0,
    axis: int = 0,
    toeplitz_cutoff: int = 4096,
) -> np.ndarray:
    """Compute the causal Grünwald–Letnikov fractional derivative ``D^alpha X`` along ``axis``.

    Parameters
    ----------
    X : np.ndarray
        Input trajectory or spatiotemporal field of arbitrary dimension (e.g. ``(nt,)``,
        ``(nt, n_vars)`` for ODEs, or ``(nt, nx, n_vars)`` for PDEs).
    t : float or 1D np.ndarray, default=1.0
        Uniform grid spacing ``dt`` or a uniformly spaced 1D coordinate vector.
    alpha : float, default=1.0
        Fractional differentiation order ``alpha >= 0``. ``alpha = 0`` returns a copy of ``X``.
    axis : int, default=0
        Axis along which to evaluate the causal fractional derivative (0 for time, 1 for space).
    toeplitz_cutoff : int, default=4096
        Grid length threshold above which ``O(N log N)`` FFT causal convolution is used
        instead of dense lower-triangular Toeplitz matrix multiplication.

    Returns
    -------
    dX : np.ndarray
        Discrete Grünwald–Letnikov fractional derivative with the same shape as ``X``.
    """
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 0:
        raise ValueError("Input X must be at least 1-dimensional.")

    dt = extract_time_step(t)
    alpha_f = float(alpha)
    if alpha_f < 0.0 or not np.isfinite(alpha_f):
        raise ValueError(f"Fractional order alpha must be finite and >= 0; got {alpha}.")
    if alpha_f == 0.0:
        return arr.copy()

    moved = np.moveaxis(arr, axis, 0)
    n = moved.shape[0]
    rest_shape = moved.shape[1:]
    flat = moved.reshape(n, -1)

    w = gl_weights(alpha_f, n)
    scale = dt**alpha_f

    if n <= toeplitz_cutoff:
        first_row = np.zeros(n, dtype=np.float64)
        first_row[0] = w[0]
        W = toeplitz(w, first_row)
        diff_flat = (W @ flat) / scale
    else:
        conv = fftconvolve(flat, w[:, np.newaxis], mode="full", axes=0)[:n]
        diff_flat = conv / scale

    diff_moved = diff_flat.reshape((n, *rest_shape))
    return np.moveaxis(diff_moved, 0, axis)


def noise_amplification_factor(
    alpha: float,
    dt: float,
    n_steps: int = 500,
    k_start: int | None = None,
) -> float:
    """Compute the Grünwald–Letnikov white-noise variance amplification factor ``A(alpha)``.

    For i.i.d. zero-mean measurement noise ``eta_k ~ N(0, sigma^2)``, the pointwise GL
    fractional derivative has variance ``Var((D_t^alpha eta)_k) = sigma^2 * dt^(-2 alpha) * sum_{j=0}^k w_j^2``.

    Parameters
    ----------
    alpha : float
        Fractional differentiation order.
    dt : float
        Uniform temporal step size.
    n_steps : int, default=500
        Total number of discrete time steps ``N_t``.
    k_start : int or None, default=None
        If ``None``, returns the asymptotic terminal factor ``dt^(-2 alpha) * ||w^(alpha)||_2^2``
        at step ``N_t - 1``. If an integer index ``0 <= k_start < n_steps`` is given, returns
        the mean amplification across evaluated regression steps ``k in [k_start, n_steps - 1]``.
    """
    step = extract_time_step(dt)
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1; got {n_steps}.")
    w = gl_weights(alpha, n_steps)
    cum_w2 = np.cumsum(w**2)
    scale = step ** (-2.0 * float(alpha))
    if k_start is None:
        return float(scale * cum_w2[-1])
    if not (0 <= k_start < n_steps):
        raise ValueError(f"k_start must satisfy 0 <= k_start < n_steps ({n_steps}); got {k_start}.")
    return float(scale * np.mean(cum_w2[k_start:]))


def integrate_fractional_ode(
    rhs: Callable[..., np.ndarray],
    y0: ArrayLike,
    t: np.ndarray | tuple[int, float],
    alpha: float = 1.0,
    n_corrector: int = 2,
) -> np.ndarray:
    """Integrate ``D_t^alpha y(t) = rhs(t, y)`` (or ``rhs(y)``) via causal GL predictor-corrector.

    Parameters
    ----------
    rhs : callable
        Right-hand side vector field accepting either ``(t_k, y_k)`` or ``(y_k,)``.
    y0 : array-like of shape (n_vars,)
        Initial state vector at ``t_0``.
    t : 1D np.ndarray or tuple of (n_steps, dt)
        Uniform time grid vector of length ``N_t``, or ``(N_t, dt)`` pair.
    alpha : float, default=1.0
        Fractional temporal order ``alpha in (0, 1]``.
    n_corrector : int, default=2
        Number of fixed-point corrector passes per step enforcing ``(D_t^alpha y)_k = f(t_k, y_k)``.
        Set ``n_corrector=0`` for the purely explicit one-step lagged predictor.

    Returns
    -------
    y : np.ndarray of shape (N_t, n_vars)
        Simulated state trajectory.
    """
    alpha_f = float(alpha)
    if alpha_f <= 0.0 or not np.isfinite(alpha_f):
        raise ValueError(f"Fractional integration order alpha must be > 0; got {alpha}.")

    if isinstance(t, tuple) and len(t) == 2:
        nt = int(t[0])
        dt = extract_time_step(t[1])
        t_grid = np.arange(nt, dtype=np.float64) * dt
    else:
        t_grid = np.asarray(t, dtype=np.float64)
        dt = extract_time_step(t_grid)
        nt = t_grid.size

    y0_arr = np.atleast_1d(np.asarray(y0, dtype=np.float64))
    if y0_arr.ndim != 1:
        raise ValueError("Initial condition y0 must be 1-dimensional.")

    d = y0_arr.size
    w = gl_weights(alpha_f, nt)
    dt_alpha = dt**alpha_f

    y = np.zeros((nt, d), dtype=np.float64)
    y[0] = y0_arr

    # Detect whether rhs takes (t, y) or (y,)
    two_arg = True
    try:
        _ = rhs(float(t_grid[0]), y0_arr)
    except TypeError:
        two_arg = False

    for k in range(1, nt):
        history = w[1 : k + 1] @ y[k - 1 :: -1]
        f_prev = rhs(float(t_grid[k - 1]), y[k - 1]) if two_arg else rhs(y[k - 1])
        yk = np.asarray(f_prev, dtype=np.float64) * dt_alpha - history
        t_curr = float(t_grid[k])
        for _ in range(int(n_corrector)):
            f_curr = rhs(t_curr, yk) if two_arg else rhs(yk)
            yk = np.asarray(f_curr, dtype=np.float64) * dt_alpha - history
        y[k] = yk

    return y


def riesz_weights(beta: float, n: int) -> np.ndarray:
    """Compute symmetric centered Riesz–Grünwald fractional weights ``g_k^(beta)`` (Ortigueira, 2006).

    Evaluates ``g_k^(beta) = (-1)^k Gamma(beta + 1) / [Gamma(beta/2 - k + 1) Gamma(beta/2 + k + 1)]``
    for ``k = 0, ..., n - 1`` via exact pole-free recurrence. At ``beta = 2``, recovers the
    classical three-point second-difference stencil ``[2, -1, 0, ...]`` for ``-d^2/dx^2``.
    """
    if n < 1:
        raise ValueError(f"Number of weights n must be >= 1; got {n}.")
    beta_f = float(beta)
    if not np.isfinite(beta_f) or beta_f < 0.0:
        raise ValueError(f"Riesz fractional order beta must be finite and >= 0; got {beta}.")

    g = np.empty(n, dtype=np.float64)
    g[0] = float(np.exp(gammaln(beta_f + 1.0) - 2.0 * gammaln(0.5 * beta_f + 1.0)))
    if n > 1:
        k = np.arange(1, n, dtype=np.float64)
        factors = (k - 1.0 - 0.5 * beta_f) / (k + 0.5 * beta_f)
        g[1:] = g[0] * np.cumprod(factors)
    return g


def riesz_operator_matrix(beta: float, n: int, dx: float = 1.0) -> np.ndarray:
    """Construct the symmetric Toeplitz Riesz–Grünwald fractional diffusion matrix ``(-d^2/dx^2)^(beta/2)``."""
    step = extract_time_step(dx)
    g = riesz_weights(beta, n)
    return toeplitz(g, g) / (step ** float(beta))


def fractional_laplacian(
    X: np.ndarray,
    dx: float | np.ndarray | Sequence[float | np.ndarray] = 1.0,
    beta: float = 1.5,
    spatial_axes: Sequence[int] | None = None,
    method: Literal["riesz", "spectral"] = "riesz",
    toeplitz_cutoff: int = 4096,
) -> np.ndarray:
    """Evaluate spatial fractional diffusion ``(-Delta)^(beta/2) X`` across arbitrary spatial axes.

    Parameters
    ----------
    X : np.ndarray
        Field array, e.g. ``(nt, nx, n_vars)``, ``(nt, nx, ny, n_vars)``, or ``(nt, nx, ny, nz, n_vars)``.
    dx : float or sequence of float/1D arrays, default=1.0
        Uniform spatial grid step(s) along ``spatial_axes``.
    beta : float, default=1.5
        Spatial fractional order ``beta >= 0`` (``beta = 2`` corresponds to ``-Delta``).
    spatial_axes : sequence of int or None, default=None
        Axes treated as spatial coordinates. Defaults to ``range(1, X.ndim - 1)`` when
        ``X.ndim >= 3``, else ``(0,)``.
    method : {"riesz", "spectral"}, default="riesz"
        ``"riesz"`` sums symmetric centered Riesz–Grünwald Toeplitz operators along each
        spatial axis; ``"spectral"`` applies the periodic isotropic symbol ``||k||_2^beta``.
    toeplitz_cutoff : int, default=4096
        Grid size threshold above which FFT Toeplitz convolution replaces dense matmul.
    """
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 0:
        raise ValueError("Input X must be at least 1-dimensional.")
    beta_f = float(beta)
    if not np.isfinite(beta_f) or beta_f < 0.0:
        raise ValueError(f"Spatial fractional order beta must be finite and >= 0; got {beta}.")
    if beta_f == 0.0:
        return arr.copy()

    if spatial_axes is None:
        axes = tuple(range(1, arr.ndim - 1)) if arr.ndim >= 3 else (0,)
    else:
        axes = tuple(int(a) % arr.ndim for a in spatial_axes)
    if len(axes) == 0:
        raise ValueError("At least one spatial axis must be specified for fractional_laplacian.")

    steps = extract_spatial_steps(dx, len(axes))

    if method == "spectral":
        k_sq = np.zeros([arr.shape[a] for a in axes], dtype=np.float64)
        for idx, (ax, step) in enumerate(zip(axes, steps)):
            n_ax = arr.shape[ax]
            freqs = 2.0 * np.pi * np.fft.fftfreq(n_ax, d=step)
            shape = [1] * len(axes)
            shape[idx] = n_ax
            k_sq = k_sq + (freqs.reshape(shape) ** 2)
        symbol = k_sq ** (0.5 * beta_f)
        expand_shape = [1] * arr.ndim
        for idx, ax in enumerate(axes):
            expand_shape[ax] = arr.shape[ax]
        spec = np.fft.fftn(arr, axes=axes)
        return np.fft.ifftn(spec * symbol.reshape(expand_shape), axes=axes).real

    if method != "riesz":
        raise ValueError(f"method must be 'riesz' or 'spectral'; got {method!r}.")

    total = np.zeros_like(arr)
    for ax, step in zip(axes, steps):
        moved = np.moveaxis(arr, ax, 0)
        n_ax = moved.shape[0]
        rest = moved.shape[1:]
        flat = moved.reshape(n_ax, -1)
        g = riesz_weights(beta_f, n_ax)
        scale = step**beta_f
        if n_ax <= toeplitz_cutoff:
            R = toeplitz(g, g)
            part_flat = (R @ flat) / scale
        else:
            kernel = np.concatenate([g[:0:-1], g])[:, np.newaxis]
            conv = fftconvolve(flat, kernel, mode="full", axes=0)[n_ax - 1 : 2 * n_ax - 1]
            part_flat = conv / scale
        total += np.moveaxis(part_flat.reshape((n_ax, *rest)), 0, ax)
    return total

