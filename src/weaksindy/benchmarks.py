"""Canonical fractional-order ODE and reaction–diffusion PDE benchmark systems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from .operators import extract_time_step, gl_weights


@dataclass(frozen=True)
class BenchmarkDataset:
    """Container for simulated benchmark data and exact symbolic ground-truth metadata.

    Supports both attribute access (``bench.X``, ``bench.ground_truth``, ``bench.t``)
    and 2-tuple unpacking (``X, ground_truth = simulate_...(...)``).
    """

    X: np.ndarray
    ground_truth: dict[str, Any]
    t: np.ndarray
    x: np.ndarray | None = None
    dx: float | None = None
    X_clean: np.ndarray | None = None

    @property
    def alpha(self) -> float:
        return float(self.ground_truth["alpha"])

    @property
    def input_features(self) -> list[str]:
        return list(self.ground_truth["input_features"])

    def __iter__(self) -> Iterator[Any]:
        yield self.X
        yield self.ground_truth

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> Any:
        if index == 0:
            return self.X
        if index == 1:
            return self.ground_truth
        raise IndexError(index)


def _add_calibrated_snr_noise(X: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = X.copy()
    for ell in range(X.shape[-1]):
        sig_var = float(np.var(X[..., ell]))
        noise_var = max(sig_var, 1e-16) * (10.0 ** (-float(snr_db) / 10.0))
        noisy[..., ell] += rng.normal(0.0, np.sqrt(noise_var), size=X[..., ell].shape)
    return noisy


def _build_1d_laplacian_matrix(nx: int, dx: float, boundary: str = "neumann") -> np.ndarray:
    if nx < 4:
        raise ValueError(f"Spatial grid nx must have at least 4 points; got {nx}.")
    dx2 = dx * dx
    L = np.zeros((nx, nx), dtype=np.float64)
    idx = np.arange(1, nx - 1)
    L[idx, idx - 1] = 1.0 / dx2
    L[idx, idx] = -2.0 / dx2
    L[idx, idx + 1] = 1.0 / dx2

    if boundary == "neumann":
        L[0, 0] = -2.0 / dx2
        L[0, 1] = 2.0 / dx2
        L[-1, -2] = 2.0 / dx2
        L[-1, -1] = -2.0 / dx2
    elif boundary == "periodic":
        L[0, -1] = 1.0 / dx2
        L[0, 0] = -2.0 / dx2
        L[0, 1] = 1.0 / dx2
        L[-1, -2] = 1.0 / dx2
        L[-1, -1] = -2.0 / dx2
        L[-1, 0] = 1.0 / dx2
    elif boundary == "dirichlet":
        L[0, 0] = -2.0 / dx2
        L[0, 1] = 1.0 / dx2
        L[-1, -2] = 1.0 / dx2
        L[-1, -1] = -2.0 / dx2
    else:
        raise ValueError(f"Unsupported boundary {boundary!r}; expected 'neumann', 'periodic', or 'dirichlet'.")
    return L


def simulate_fractional_lorenz(
    t: np.ndarray | None = None,
    alpha: float = 0.85,
    *,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    y0: tuple[float, float, float] | list[float] | np.ndarray = (1.0, 1.0, 1.0),
    snr_db: float | None = None,
    seed: int = 42,
) -> BenchmarkDataset:
    """Simulate the 3D fractional Grünwald–Letnikov Lorenz system ``D_t^alpha [x, y, z]``.

    Integrates ``D_t^alpha x = sigma (y - x)``, ``D_t^alpha y = x (rho - z) - y``,
    ``D_t^alpha z = x y - beta z`` via an A-stable implicit Grünwald–Letnikov Newton
    solver so discrete fractional trajectories are free of explicit time-step blow-up.
    """
    t_grid = np.linspace(0.0, 10.0, 1001) if t is None else np.asarray(t, dtype=np.float64)
    dt = extract_time_step(t_grid)
    nt = t_grid.size
    alpha_f = float(alpha)
    if not (0.0 < alpha_f <= 1.5):
        raise ValueError(f"Fractional order alpha must be in (0, 1.5]; got {alpha}.")

    y_init = np.asarray(y0, dtype=np.float64)
    if y_init.shape != (3,):
        raise ValueError(f"y0 for Lorenz must have shape (3,); got {y_init.shape}.")

    w = gl_weights(alpha_f, nt)
    dt_a = dt**alpha_f
    eye3 = np.eye(3, dtype=np.float64)

    X_clean = np.zeros((nt, 3), dtype=np.float64)
    X_clean[0] = y_init

    for k in range(1, nt):
        hist = w[1 : k + 1] @ X_clean[k - 1 :: -1]
        yk = X_clean[k - 1].copy()
        for _ in range(6):
            xk, yv, zk = yk
            f_val = np.array(
                [sigma * (yv - xk), xk * (rho - zk) - yv, xk * yv - beta * zk],
                dtype=np.float64,
            )
            res = yk - dt_a * f_val + hist
            if float(np.max(np.abs(res))) < 1e-13:
                break
            J_f = np.array(
                [
                    [-sigma, sigma, 0.0],
                    [rho - zk, -1.0, -xk],
                    [yv, xk, -beta],
                ],
                dtype=np.float64,
            )
            delta = np.linalg.solve(eye3 - dt_a * J_f, res)
            yk = yk - delta
        X_clean[k] = yk

    X_obs = _add_calibrated_snr_noise(X_clean, snr_db=snr_db, seed=seed) if snr_db is not None else X_clean.copy()

    ground_truth: dict[str, Any] = {
        "system": "fractional_lorenz",
        "alpha": alpha_f,
        "dt": dt,
        "input_features": ["x", "y", "z"],
        "parameters": {"sigma": float(sigma), "rho": float(rho), "beta": float(beta)},
        "terms": {
            "x": {"x": -float(sigma), "y": float(sigma)},
            "y": {"x": float(rho), "y": -1.0, "x z": -1.0},
            "z": {"z": -float(beta), "x y": 1.0},
        },
        "equations": [
            f"D_t^{alpha_f:g} x = -{sigma:g} x + {sigma:g} y",
            f"D_t^{alpha_f:g} y = {rho:g} x - 1 y - 1 x z",
            f"D_t^{alpha_f:g} z = -{beta:.4f} z + 1 x y",
        ],
    }
    return BenchmarkDataset(X=X_obs, ground_truth=ground_truth, t=t_grid, X_clean=X_clean)


def simulate_subdiffusive_fisher_kpp_1d(
    t: np.ndarray | None = None,
    x: np.ndarray | None = None,
    alpha: float = 0.7,
    D: float = 0.1,
    r: float = 1.0,
    *,
    boundary: str = "neumann",
    u0: np.ndarray | None = None,
    snr_db: float | None = None,
    seed: int = 42,
) -> BenchmarkDataset:
    """Simulate the 1D subdiffusive Fisher–KPP reaction–diffusion equation.

    Integrates ``D_t^alpha u = D u_xx + r u (1 - u)`` via an unconditionally stable
    semi-implicit / Newton Grünwald–Letnikov scheme, returning field ``X`` of shape
    ``(nt, nx, 1)`` and reference ground-truth dictionary.
    """
    t_grid = np.linspace(0.0, 4.0, 251) if t is None else np.asarray(t, dtype=np.float64)
    x_grid = np.linspace(0.0, 10.0, 64) if x is None else np.asarray(x, dtype=np.float64)
    dt = extract_time_step(t_grid)
    dx = extract_time_step(x_grid)
    nt = t_grid.size
    nx = x_grid.size
    alpha_f = float(alpha)
    if not (0.0 < alpha_f <= 1.5):
        raise ValueError(f"Fractional order alpha must be in (0, 1.5]; got {alpha}.")

    L_dom = float(x_grid[-1] - x_grid[0])
    x0 = float(x_grid[0])
    if u0 is None:
        u_init = 0.65 * np.exp(-((x_grid - (x0 + 0.35 * L_dom)) ** 2) / (0.025 * L_dom**2)) + 0.25 * np.exp(
            -((x_grid - (x0 + 0.72 * L_dom)) ** 2) / (0.04 * L_dom**2)
        )
    else:
        u_init = np.asarray(u0, dtype=np.float64).ravel()
        if u_init.size != nx:
            raise ValueError(f"u0 length ({u_init.size}) must match x length ({nx}).")

    L_xx = _build_1d_laplacian_matrix(nx, dx, boundary=boundary)
    w = gl_weights(alpha_f, nt)
    dt_a = dt**alpha_f
    lhs_diff = np.eye(nx, dtype=np.float64) - dt_a * float(D) * L_xx
    lu_diff = lu_factor(lhs_diff)

    u_sol = np.zeros((nt, nx), dtype=np.float64)
    u_sol[0] = u_init

    for k in range(1, nt):
        hist = w[1 : k + 1] @ u_sol[k - 1 :: -1]
        uk = u_sol[k - 1].copy()
        for _ in range(5):
            rhs_vec = dt_a * float(r) * uk * (1.0 - uk) - hist
            uk_next = lu_solve(lu_diff, rhs_vec)
            if float(np.max(np.abs(uk_next - uk))) < 1e-12:
                uk = uk_next
                break
            uk = uk_next
        u_sol[k] = uk

    X_clean = u_sol[:, :, np.newaxis]
    X_obs = _add_calibrated_snr_noise(X_clean, snr_db=snr_db, seed=seed) if snr_db is not None else X_clean.copy()

    ground_truth: dict[str, Any] = {
        "system": "subdiffusive_fisher_kpp_1d",
        "alpha": alpha_f,
        "dt": dt,
        "dx": dx,
        "input_features": ["u"],
        "parameters": {"D": float(D), "r": float(r), "boundary": boundary},
        "terms": {
            "u": {"u": float(r), "u^2": -float(r), "u_xx": float(D)},
        },
        "equations": [
            f"D_t^{alpha_f:g} u = {r:g} u - {r:g} u^2 + {D:g} u_xx",
        ],
    }
    return BenchmarkDataset(
        X=X_obs,
        ground_truth=ground_truth,
        t=t_grid,
        x=x_grid,
        dx=dx,
        X_clean=X_clean,
    )


def simulate_vascular_stromal_rd(
    t: np.ndarray | None = None,
    x: np.ndarray | None = None,
    alpha: float = 0.75,
    *,
    Dp: float = 0.05,
    Dc: float = 0.10,
    Dv: float = 0.01,
    gam_p: float = 0.05,
    Sp: float = 0.10,
    gam_c: float = 0.20,
    Sc: float = 0.30,
    rho: float = 0.20,
    gam_n: float = 0.10,
    boundary: str = "neumann",
    snr_db: float | None = None,
    seed: int = 42,
) -> BenchmarkDataset:
    """Simulate the 3-field coupled vascular–stromal reaction–diffusion PDE (*Chaos*, 2026).

    Models coupled interstitial fluid pressure ``p``, oxygen concentration ``c``, and
    microvascular density ``n`` governed by fractional temporal memory ``D_t^alpha``,
    Fickian diffusion, and Michaelis–Menten saturation kinetics:

    - ``D_t^alpha p = Dp p_xx - gam_p p + Sp n (1 - p)``
    - ``D_t^alpha c = Dc c_xx - gam_c c p/(1 + p) + Sc n``
    - ``D_t^alpha n = Dv n_xx + rho n (1 - n) c/(1 + c) - gam_n p n``

    Returns ``BenchmarkDataset`` with ``X`` of shape ``(nt, nx, 3)``.
    """
    t_grid = np.linspace(0.0, 5.0, 251) if t is None else np.asarray(t, dtype=np.float64)
    x_grid = np.linspace(0.0, 10.0, 60) if x is None else np.asarray(x, dtype=np.float64)
    dt = extract_time_step(t_grid)
    dx = extract_time_step(x_grid)
    nt = t_grid.size
    nx = x_grid.size
    alpha_f = float(alpha)
    if not (0.0 < alpha_f <= 1.5):
        raise ValueError(f"Fractional order alpha must be in (0, 1.5]; got {alpha}.")

    L_dom = float(x_grid[-1] - x_grid[0])
    x_mid = 0.5 * float(x_grid[0] + x_grid[-1])
    p0 = np.exp(-((x_grid - x_mid) ** 2) / 2.0)
    c0 = 1.0 - 0.5 * np.exp(-((x_grid - x_mid) ** 2) / 2.0)
    n0 = 0.1 + 0.2 * np.sin(np.pi * (x_grid - x_grid[0]) / L_dom)

    L_xx = _build_1d_laplacian_matrix(nx, dx, boundary=boundary)
    w = gl_weights(alpha_f, nt)
    dt_a = dt**alpha_f
    eye_x = np.eye(nx, dtype=np.float64)

    lu_p = lu_factor(eye_x - dt_a * float(Dp) * L_xx)
    lu_c = lu_factor(eye_x - dt_a * float(Dc) * L_xx)
    lu_n = lu_factor(eye_x - dt_a * float(Dv) * L_xx)

    X_clean = np.zeros((nt, nx, 3), dtype=np.float64)
    X_clean[0, :, 0] = p0
    X_clean[0, :, 1] = c0
    X_clean[0, :, 2] = n0

    for k in range(1, nt):
        hist_p = w[1 : k + 1] @ X_clean[k - 1 :: -1, :, 0]
        hist_c = w[1 : k + 1] @ X_clean[k - 1 :: -1, :, 1]
        hist_n = w[1 : k + 1] @ X_clean[k - 1 :: -1, :, 2]

        pk = X_clean[k - 1, :, 0].copy()
        ck = X_clean[k - 1, :, 1].copy()
        nk = X_clean[k - 1, :, 2].copy()

        for _ in range(5):
            p_pos = np.clip(pk, -0.95, None)
            c_pos = np.clip(ck, -0.95, None)
            react_p = -gam_p * pk + Sp * nk * (1.0 - pk)
            react_c = -gam_c * ck * (p_pos / (1.0 + p_pos)) + Sc * nk
            react_n = rho * nk * (1.0 - nk) * (c_pos / (1.0 + c_pos)) - gam_n * nk * pk

            pk_next = lu_solve(lu_p, dt_a * react_p - hist_p)
            ck_next = lu_solve(lu_c, dt_a * react_c - hist_c)
            nk_next = lu_solve(lu_n, dt_a * react_n - hist_n)

            diff_max = max(
                float(np.max(np.abs(pk_next - pk))),
                float(np.max(np.abs(ck_next - ck))),
                float(np.max(np.abs(nk_next - nk))),
            )
            pk, ck, nk = pk_next, ck_next, nk_next
            if diff_max < 1e-12:
                break

        X_clean[k, :, 0] = pk
        X_clean[k, :, 1] = ck
        X_clean[k, :, 2] = nk

    X_obs = _add_calibrated_snr_noise(X_clean, snr_db=snr_db, seed=seed) if snr_db is not None else X_clean.copy()

    ground_truth: dict[str, Any] = {
        "system": "vascular_stromal_rd",
        "alpha": alpha_f,
        "dt": dt,
        "dx": dx,
        "input_features": ["p", "c", "n"],
        "parameters": {
            "Dp": float(Dp),
            "Dc": float(Dc),
            "Dv": float(Dv),
            "gam_p": float(gam_p),
            "Sp": float(Sp),
            "gam_c": float(gam_c),
            "Sc": float(Sc),
            "rho": float(rho),
            "gam_n": float(gam_n),
            "boundary": boundary,
        },
        "terms": {
            "p": {"p": -float(gam_p), "n": float(Sp), "p n": -float(Sp), "p_xx": float(Dp)},
            "c": {"n": float(Sc), "c_xx": float(Dc), "c*p/(1+p)": -float(gam_c)},
            "n": {
                "p n": -float(gam_n),
                "n_xx": float(Dv),
                "n*c/(1+c)": float(rho),
                "n^2*c/(1+c)": -float(rho),
            },
        },
        "equations": [
            f"D_t^{alpha_f:g} p = -{gam_p:g} p + {Sp:g} n - {Sp:g} p n + {Dp:g} p_xx",
            f"D_t^{alpha_f:g} c = {Sc:g} n + {Dc:g} c_xx - {gam_c:g} c*p/(1+p)",
            f"D_t^{alpha_f:g} n = -{gam_n:g} p n + {Dv:g} n_xx + {rho:g} n*c/(1+c) - {rho:g} n^2*c/(1+c)",
        ],
    }
    return BenchmarkDataset(
        X=X_obs,
        ground_truth=ground_truth,
        t=t_grid,
        x=x_grid,
        dx=dx,
        X_clean=X_clean,
    )
