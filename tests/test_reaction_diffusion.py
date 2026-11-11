"""Three-field vascular–stromal reaction–diffusion PDE and Benettin Lyapunov benchmark tests."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from weaksindy import (
    PDELibrary,
    WeakSINDy,
    compute_benettin_lyapunov,
    gl_weights,
    rosenstein_lyapunov,
)


def _neumann_laplacian(u: np.ndarray, dx: float) -> np.ndarray:
    lap = np.empty_like(u)
    dx2 = dx * dx
    lap[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / dx2
    lap[0] = (2.0 * u[1] - 2.0 * u[0]) / dx2
    lap[-1] = (2.0 * u[-2] - 2.0 * u[-1]) / dx2
    return lap


def _vascular_stromal_rhs(
    _t: float,
    y: np.ndarray,
    nx: int,
    dx: float,
    Dp: float = 0.05,
    Dc: float = 0.10,
    Dv: float = 0.01,
    gam_p: float = 0.05,
    Sp: float = 0.10,
    gam_c: float = 0.20,
    Sc: float = 0.30,
    rho: float = 0.20,
    gam_n: float = 0.10,
) -> np.ndarray:
    """Coupled 3-field interstitial pressure (p), oxygen (c), and vessel density (n) PDE RHS."""
    p = np.clip(y[:nx], 0.0, None)
    c = np.clip(y[nx : 2 * nx], 0.0, None)
    n = np.clip(y[2 * nx :], 0.0, 1.5)

    dp = Dp * _neumann_laplacian(p, dx) - gam_p * p + Sp * n * (1.0 - p)
    dc = Dc * _neumann_laplacian(c, dx) - gam_c * c * (p / (1.0 + p)) + Sc * n
    dn = Dv * _neumann_laplacian(n, dx) + rho * n * (1.0 - n) * (c / (1.0 + c)) - gam_n * n * p
    return np.concatenate([dp, dc, dn])


def _simulate_fractional_three_field(alpha: float, nt: int = 400, nx: int = 30) -> tuple[np.ndarray, float, np.ndarray]:
    L, T = 10.0, 4.0
    t = np.linspace(0.0, T, nt)
    dt = float(t[1] - t[0])
    x = np.linspace(0.0, L, nx)
    dx = float(x[1] - x[0])

    p0 = np.exp(-((x - 0.5 * L) ** 2) / 2.0)
    c0 = 1.0 - 0.5 * np.exp(-((x - 0.5 * L) ** 2) / 2.0)
    n0 = 0.1 + 0.2 * np.sin(np.pi * x / L)
    y0 = np.concatenate([p0, c0, n0])

    w = gl_weights(alpha, nt)
    y_sol = np.zeros((nt, 3 * nx), dtype=np.float64)
    y_sol[0] = y0
    dt_a = dt**alpha
    for k in range(1, nt):
        p = y_sol[k - 1, :nx]
        c = y_sol[k - 1, nx : 2 * nx]
        n = y_sol[k - 1, 2 * nx :]
        rhs = np.concatenate([-0.5 * p + 0.2 * c, -0.3 * c + 0.1 * n, -0.2 * n])
        hist = w[1 : k + 1] @ y_sol[k - 1 :: -1]
        y_sol[k] = rhs * dt_a - hist

    u = np.stack([y_sol[:, :nx], y_sol[:, nx : 2 * nx], y_sol[:, 2 * nx :]], axis=-1)
    return t, dx, u


def _add_pde_noise(u: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = u.copy()
    for ell in range(u.shape[-1]):
        sig_var = float(np.var(u[..., ell]))
        noise_var = sig_var * (10.0 ** (-snr_db / 10.0))
        noisy[..., ell] += rng.normal(0.0, np.sqrt(noise_var), size=u[..., ell].shape)
    return noisy


def test_vascular_stromal_pde_integer_order_specificity() -> None:
    """On integer-order 3-field reaction–diffusion data, WeakSINDy refutes fractional orders."""
    nx, nt = 60, 200
    L, T = 10.0, 5.0
    x = np.linspace(0.0, L, nx)
    dx = float(x[1] - x[0])
    t = np.linspace(0.0, T, nt)

    p0 = np.exp(-((x - 0.5 * L) ** 2) / 2.0)
    c0 = 1.0 - 0.5 * np.exp(-((x - 0.5 * L) ** 2) / 2.0)
    n0 = 0.1 + 0.2 * np.sin(np.pi * x / L)
    y0 = np.concatenate([p0, c0, n0])

    sol = solve_ivp(
        _vascular_stromal_rhs,
        [0.0, T],
        y0,
        t_eval=t,
        method="Radau",
        args=(nx, dx),
        rtol=1e-8,
        atol=1e-10,
    )
    u = np.stack([sol.y[:nx].T, sol.y[nx : 2 * nx].T, sol.y[2 * nx :].T], axis=-1)

    lib = PDELibrary(
        dx=dx,
        poly_degree=2,
        derivative_orders=(1, 2),
        fractional_orders=(0.5, 1.5),
        include_michaelis_menten=True,
        boundary="neumann",
    )
    model = WeakSINDy(
        library=lib,
        threshold=0.005,
        use_weak=True,
        window_size=30,
        stride=4,
        k_start=10,
    )
    model.fit(t, u, orders=[0.6, 0.8, 1.0], input_features=["p", "c", "n"])

    assert model.order_ == pytest.approx(1.0)
    assert model.order_scores_[1.0] > 0.99
    assert model.order_scores_[1.0] - max(model.order_scores_[0.6], model.order_scores_[0.8]) > 0.25


@pytest.mark.parametrize("true_alpha", [0.5, 0.7, 0.9])
def test_three_field_fractional_recovery_clean_and_low_snr(true_alpha: float) -> None:
    """Verify exact recovery on clean data and at 20 dB SNR where pointwise GL rails downward."""
    t, dx, u_clean = _simulate_fractional_three_field(true_alpha, nt=400, nx=25)
    candidates = [0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
    lib = PDELibrary(dx=dx, poly_degree=2, derivative_orders=(1, 2), fractional_orders=(0.5,))

    # 1. Noise-free exact recovery
    clean_model = WeakSINDy(library=lib, threshold=0.02, window_size=45, stride=5, k_start=20)
    clean_model.fit(t, u_clean, orders=candidates, input_features=["p", "c", "n"])
    assert clean_model.order_ == pytest.approx(true_alpha)

    # 2. Low-SNR (20 dB) recovery: Weak-form succeeds while Pointwise GL biases toward floor
    u_noisy = _add_pde_noise(u_clean, snr_db=20.0, seed=42 + int(true_alpha * 10))

    weak_noisy = WeakSINDy(
        library=lib,
        threshold=0.02,
        use_weak=True,
        window_size=45,
        stride=5,
        k_start=20,
        refit=False,
    )
    weak_noisy.fit(
        t, u_noisy, orders=candidates, X_clean=u_clean, true_order=true_alpha
    )
    assert weak_noisy.order_ == pytest.approx(true_alpha)

    pw_noisy = WeakSINDy(
        library=lib,
        threshold=0.02,
        use_weak=False,
        k_start=20,
        refit=False,
    )
    pw_noisy.fit(
        t, u_noisy, orders=candidates, X_clean=u_clean, true_order=true_alpha
    )
    assert pw_noisy.order_ < true_alpha


def test_benettin_lyapunov_vascular_stromal_and_lorenz_spectrum() -> None:
    """Verify negative Benettin LLE on vascular–stromal PDE vs positive Lorenz-63 chaotic spectrum."""
    nx = 40
    L = 10.0
    x = np.linspace(0.0, L, nx)
    dx = float(x[1] - x[0])
    y0 = np.concatenate(
        [
            np.exp(-((x - 0.5 * L) ** 2) / 2.0),
            1.0 - 0.5 * np.exp(-((x - 0.5 * L) ** 2) / 2.0),
            0.1 + 0.2 * np.sin(np.pi * x / L),
        ]
    )

    res_pde = compute_benettin_lyapunov(
        lambda t_val, y_val: _vascular_stromal_rhs(t_val, y_val, nx=nx, dx=dx),
        y0,
        n_exponents=1,
        t_trans=20.0,
        t_run=80.0,
        dt=0.02,
        dt_renorm=0.2,
        seed=42,
    )
    assert res_pde.lambda_max == pytest.approx(-0.0734, abs=0.008)
    assert res_pde.lambda_max < 0.0

    # Contrast with spurious positive Rosenstein slope on transient spatial-mean pressure
    t_sim = np.linspace(0.0, 150.0, 1500)
    sol = solve_ivp(
        _vascular_stromal_rhs,
        [0.0, 150.0],
        y0,
        t_eval=t_sim,
        method="Radau",
        args=(nx, dx),
    )
    p_mean = np.mean(sol.y[:nx], axis=0)
    ros_slope, _ = rosenstein_lyapunov(p_mean, dt=float(t_sim[1] - t_sim[0]), m=2, tau=20, theiler_window=25, max_steps=60)
    assert ros_slope > 0.0  # Known transient artifact refuted by Benettin

    # Full 3-exponent spectrum on Lorenz-63 with analytical Jacobian
    sigma, rho_l, beta_l = 10.0, 28.0, 8.0 / 3.0

    def lorenz_rhs(s: np.ndarray) -> np.ndarray:
        return np.array([sigma * (s[1] - s[0]), s[0] * (rho_l - s[2]) - s[1], s[0] * s[1] - beta_l * s[2]])

    def lorenz_jac(s: np.ndarray) -> np.ndarray:
        return np.array(
            [
                [-sigma, sigma, 0.0],
                [rho_l - s[2], -1.0, -s[0]],
                [s[1], s[0], -beta_l],
            ]
        )

    res_lor = compute_benettin_lyapunov(
        lorenz_rhs,
        [1.0, 1.0, 1.0],
        n_exponents=3,
        t_trans=10.0,
        t_run=60.0,
        dt=0.005,
        dt_renorm=0.05,
        jacobian=lorenz_jac,
        seed=7,
    )
    l1, l2, l3 = res_lor.lyapunov_exponents
    assert l1 > 0.75
    assert abs(l2) < 0.12
    assert l3 < -13.5
    # Dissipation identity: sum of Lyapunov exponents = trace(J) = -(sigma + 1 + beta) = -13.6667
    assert float(np.sum(res_lor.lyapunov_exponents)) == pytest.approx(-(sigma + 1.0 + beta_l), rel=2e-2)
