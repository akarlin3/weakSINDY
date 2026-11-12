"""Unit tests for 1D/2D/3D weak-form tensor PDE operators, Riesz diffusion, and custom/rational libraries."""

from __future__ import annotations

import numpy as np
import pytest

from weaksindy import (
    CombinedLibrary,
    CustomLibrary,
    PDELibrary,
    PolynomialLibrary,
    RationalLibrary,
    WeakSINDy,
    fractional_laplacian,
    gl_weights,
    integrate_fractional_ode,
    project_weak_form,
    riesz_weights,
)


def _add_snr_noise(u: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = u.copy()
    for ell in range(u.shape[-1]):
        var_s = float(np.var(u[..., ell]))
        var_n = max(var_s, 1e-16) * (10.0 ** (-snr_db / 10.0))
        noisy[..., ell] += rng.normal(0.0, np.sqrt(var_n), size=u[..., ell].shape)
    return noisy


def test_riesz_weights_exact_integer_limits_and_symmetry() -> None:
    """Centered Riesz–Grünwald weights recover -d^2/dx^2 at beta=2 and d^4/dx^4 at beta=4."""
    g2 = riesz_weights(2.0, 8)
    np.testing.assert_allclose(g2, [2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], atol=1e-14)

    g4 = riesz_weights(4.0, 8)
    np.testing.assert_allclose(g4, [6.0, -4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], atol=1e-13)

    # Fractional order beta=1 (Cauchy / half-Laplacian) is finite and strictly alternating/decaying
    g1 = riesz_weights(1.0, 10)
    assert g1[0] == pytest.approx(4.0 / np.pi, rel=1e-12)
    assert np.all(g1[1:] < 0.0)


def test_2d_spatiotemporal_advection_diffusion_weak_recovery() -> None:
    """Recover 2D advection-diffusion u_t = 0.15 laplacian(u) - 0.8 v.grad(u) - 0.3 u under 22 dB noise."""
    nx, ny, nt = 36, 36, 121
    x = np.linspace(0.0, 2.0 * np.pi, nx, endpoint=False)
    y = np.linspace(0.0, 2.0 * np.pi, ny, endpoint=False)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    t = np.linspace(0.0, 2.0, nt)

    X_g, Y_g = np.meshgrid(x, y, indexing="ij")
    D_diff = 0.15
    vx, vy = 0.8, -0.4
    lam = 0.30

    # Multi-mode exact traveling/decaying periodic solution so u, laplacian(u), and v.grad(u) are independent
    # Mode 1: k=(1, 1), |k|^2 = 2; Mode 2: k=(2, 1), |k|^2 = 5
    T_g = t[:, None, None]
    X_b = X_g[None, :, :]
    Y_b = Y_g[None, :, :]

    u1 = np.exp(-(lam + 2.0 * D_diff) * T_g) * np.sin(X_b - vx * T_g + Y_b - vy * T_g)
    u2 = 0.7 * np.exp(-(lam + 5.0 * D_diff) * T_g) * np.cos(2.0 * (X_b - vx * T_g) + (Y_b - vy * T_g))
    U_clean = (u1 + u2)[..., np.newaxis]  # (nt, nx, ny, 1)

    U_noisy = _add_snr_noise(U_clean, snr_db=22.0, seed=17)

    lib = PDELibrary(
        dx=(dx, dy),
        poly_degree=1,
        derivative_orders=(),
        include_bias=False,
        include_laplacian=True,
        advection_velocity=(vx, vy),
        boundary="periodic",
    )
    model = WeakSINDy(
        library=lib,
        threshold=0.04,
        window_size=28,
        stride=4,
        spatial_window_size=(14, 14),
        spatial_stride=(4, 4),
        test_degree=5,
        k_start=5,
    )
    model.fit(t, U_noisy, orders=1.0, input_features=["u"])

    names = model.feature_names_
    assert model.coef_[0, names.index("u")] == pytest.approx(-lam, abs=0.03)
    assert model.coef_[0, names.index("laplacian(u)")] == pytest.approx(D_diff, abs=0.03)
    assert model.coef_[0, names.index("v.grad(u)")] == pytest.approx(-1.0, abs=0.04)


def test_2d_biharmonic_fourth_order_pde_weak_ibp_no_fd_blowup() -> None:
    """Verify 4th-order biharmonic nabla^4 u transfers onto C^p test functions under 25 dB noise."""
    nx, ny, nt = 36, 36, 101
    x = np.linspace(0.0, 2.0 * np.pi, nx, endpoint=False)
    y = np.linspace(0.0, 2.0 * np.pi, ny, endpoint=False)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    t = np.linspace(0.0, 1.2, nt)

    X_g, Y_g = np.meshgrid(x, y, indexing="ij")
    T_g = t[:, None, None]
    kappa = 0.04
    alpha_lin = -0.25

    # u_t = alpha_lin * u - kappa * nabla^4 u
    # Mode (1,1): |k|^4 = (1+1)^2 = 4; Mode (2,1): |k|^4 = (4+1)^2 = 25
    m1 = np.exp((alpha_lin - 4.0 * kappa) * T_g) * np.sin(X_g[None] + 0.3) * np.sin(Y_g[None] + 0.1)
    m2 = 0.6 * np.exp((alpha_lin - 25.0 * kappa) * T_g) * np.cos(2.0 * X_g[None]) * np.sin(Y_g[None])
    U_clean = (m1 + m2)[..., np.newaxis]
    U_noisy = _add_snr_noise(U_clean, snr_db=25.0, seed=99)

    lib = PDELibrary(
        dx=(dx, dy),
        poly_degree=1,
        derivative_orders=(),
        include_bias=False,
        include_biharmonic=True,
        boundary="periodic",
    )
    model = WeakSINDy(
        library=lib,
        threshold=0.01,
        window_size=24,
        stride=4,
        spatial_window_size=(16, 16),
        spatial_stride=(4, 4),
        test_degree=6,
        k_start=4,
    )
    model.fit(t, U_noisy, orders=1.0, input_features=["u"])

    names = model.feature_names_
    assert model.coef_[0, names.index("u")] == pytest.approx(alpha_lin, abs=0.025)
    assert model.coef_[0, names.index("biharmonic(u)")] == pytest.approx(-kappa, abs=0.01)


def test_1d_burgers_nonlinear_advection_weak_ibp() -> None:
    """Recover viscous Burgers equation u_t = -u*u_x + nu*u_xx via exact weak flux IBP."""
    nx, nt = 96, 241
    L = 2.0 * np.pi
    x = np.linspace(0.0, L, nx, endpoint=False)
    dx = float(x[1] - x[0])
    t = np.linspace(0.0, 2.0, nt)
    nu = 0.25

    # Exact 2-mode Cole-Hopf solution: phi_t = nu * phi_xx -> u = -2 nu * phi_x / phi solves Burgers
    T_g = t[:, None]
    X_g = x[None, :]
    phi = (
        1.0
        + 0.50 * np.exp(-nu * T_g) * np.sin(X_g)
        + 0.35 * np.exp(-4.0 * nu * T_g) * np.cos(2.0 * X_g)
    )
    phi_x = (
        0.50 * np.exp(-nu * T_g) * np.cos(X_g)
        - 0.70 * np.exp(-4.0 * nu * T_g) * np.sin(2.0 * X_g)
    )
    U_clean = (-2.0 * nu * phi_x / phi)[:, :, np.newaxis]
    U_noisy = _add_snr_noise(U_clean, snr_db=25.0, seed=31)

    lib = PDELibrary(
        dx=dx,
        poly_degree=1,
        derivative_orders=(2,),
        include_bias=False,
        include_advection=True,
        boundary="periodic",
    )
    model = WeakSINDy(
        library=lib,
        threshold=0.05,
        window_size=50,
        stride=5,
        spatial_window_size=24,
        spatial_stride=4,
        test_degree=4,
        k_start=5,
    )
    model.fit(t, U_noisy, orders=1.0, input_features=["u"])

    names = model.feature_names_
    assert model.coef_[0, names.index("u*u_x")] == pytest.approx(-1.0, abs=0.03)
    assert model.coef_[0, names.index("u_xx")] == pytest.approx(nu, abs=0.02)
    assert model.coef_[0, names.index("u")] == pytest.approx(0.0, abs=1e-12)


def test_3d_spatiotemporal_fractional_pde_tensor_contraction() -> None:
    """Recover 3D fractional reaction-diffusion D_t^0.8 u = 0.12 laplacian(u) - 0.4 u on (nt, nx, ny, nz, 1)."""
    nx, ny, nz, nt = 16, 16, 16, 81
    L = 2.0 * np.pi
    x = np.linspace(0.0, L, nx, endpoint=False)
    dx = float(x[1] - x[0])
    t = np.linspace(0.0, 2.0, nt)
    dt = float(t[1] - t[0])

    true_alpha = 0.8
    D_diff = 0.12
    gam = 0.40

    X_g, Y_g, Z_g = np.meshgrid(x, x, x, indexing="ij")
    mode1 = np.sin(X_g) * np.sin(Y_g) * np.sin(Z_g)  # laplacian = -3 * mode1
    mode2 = np.cos(2.0 * X_g) * np.sin(Y_g) * np.cos(Z_g)  # laplacian = -6 * mode2

    # Integrate exact discrete GL modal amplitudes so D_t^alpha a_i = -(gam + k_i^2 D_diff) a_i
    w = gl_weights(true_alpha, nt)
    dt_a = dt**true_alpha
    a1 = np.zeros(nt)
    a2 = np.zeros(nt)
    a1[0], a2[0] = 1.0, 0.7
    lam1 = -(gam + 3.0 * D_diff)
    lam2 = -(gam + 6.0 * D_diff)
    for k in range(1, nt):
        a1[k] = -float(w[1 : k + 1] @ a1[k - 1 :: -1]) / (1.0 - dt_a * lam1)
        a2[k] = -float(w[1 : k + 1] @ a2[k - 1 :: -1]) / (1.0 - dt_a * lam2)

    U_3d = (a1[:, None, None, None] * mode1[None] + a2[:, None, None, None] * mode2[None])[
        ..., np.newaxis
    ]
    assert U_3d.shape == (nt, nx, ny, nz, 1)

    lib = PDELibrary(
        dx=(dx, dx, dx),
        poly_degree=1,
        derivative_orders=(),
        include_bias=False,
        include_laplacian=True,
        boundary="periodic",
    )
    # Direct project_weak_form check on 5D array
    Theta_wk, y_wk = project_weak_form(
        U_3d,
        t=t,
        alpha=true_alpha,
        library=lib,
        window_size=20,
        stride=5,
        spatial_window_size=(8, 8, 8),
        spatial_stride=(4, 4, 4),
        k_start=10,
    )
    assert Theta_wk.ndim == 2 and y_wk.ndim == 2
    assert Theta_wk.shape[0] == y_wk.shape[0]

    model = WeakSINDy(
        library=lib,
        threshold=0.02,
        window_size=20,
        stride=4,
        spatial_window_size=(8, 8, 8),
        spatial_stride=(4, 4, 4),
        test_degree=5,
        k_start=10,
    )
    model.fit(t, U_3d, orders=[0.6, 0.8, 1.0], input_features=["u"])

    assert model.order_ == pytest.approx(true_alpha)
    names = model.feature_names_
    assert model.coef_[0, names.index("u")] == pytest.approx(-gam, rel=5e-2)
    assert model.coef_[0, names.index("laplacian(u)")] == pytest.approx(D_diff, rel=8e-2)


@pytest.mark.parametrize("riesz_method", ["riesz", "spectral"])
def test_spatial_fractional_laplacian_weak_discovery(riesz_method: str) -> None:
    """Recover space-fractional superdiffusion u_t = -0.25 (-Delta)^(0.75) u - 0.15 u."""
    nx, nt = 48, 101
    x = np.linspace(0.0, 2.0 * np.pi, nx, endpoint=False)
    dx = float(x[1] - x[0])
    t = np.linspace(0.0, 1.5, nt)
    dt = float(t[1] - t[0])

    beta = 1.5  # (-Delta)^(beta/2) = (-Delta)^0.75
    kappa = 0.25
    damp = 0.15

    # Generate clean trajectory using exact operator matrix / spectral step
    U = np.zeros((nt, nx, 1), dtype=np.float64)
    U[0, :, 0] = np.sin(x) + 0.6 * np.cos(2.0 * x)
    # Implicit midpoint / RK4 step with exact fractional_laplacian operator
    def rhs(state_1d: np.ndarray) -> np.ndarray:
        flap = fractional_laplacian(
            state_1d[:, None], dx=dx, beta=beta, spatial_axes=(0,), method=riesz_method  # type: ignore[arg-type]
        )[:, 0]
        return -kappa * flap - damp * state_1d

    for k in range(1, nt):
        s = U[k - 1, :, 0]
        k1 = rhs(s)
        k2 = rhs(s + 0.5 * dt * k1)
        k3 = rhs(s + 0.5 * dt * k2)
        k4 = rhs(s + dt * k3)
        U[k, :, 0] = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    U_noisy = _add_snr_noise(U, snr_db=26.0, seed=55)

    lib = PDELibrary(
        dx=dx,
        poly_degree=1,
        derivative_orders=(),
        riesz_orders=(beta,),
        riesz_method=riesz_method,  # type: ignore[arg-type]
        include_bias=False,
    )
    model = WeakSINDy(
        library=lib,
        threshold=0.03,
        window_size=24,
        stride=4,
        spatial_window_size=16,
        spatial_stride=4,
        k_start=4,
    )
    model.fit(t, U_noisy, orders=1.0, input_features=["u"])
    names = model.feature_names_
    assert model.coef_[0, names.index("(-Delta)^(0.75) u")] == pytest.approx(-kappa, abs=0.025)
    assert model.coef_[0, names.index("u")] == pytest.approx(-damp, abs=0.025)


def test_rational_and_custom_libraries_hill_kinetics() -> None:
    """Recover fractional Hill gene-regulatory circuit using RationalLibrary and CustomLibrary."""
    t = np.linspace(0.0, 12.0, 601)
    # D_t^0.8 x = -0.5 x + 1.4 y^2 / (1 + y^2)
    # D_t^0.8 y = 0.8 x / (1 + x) - 0.4 y
    def hill_rhs(s: np.ndarray) -> np.ndarray:
        x, y = s
        dx_dt = -0.5 * x + 1.4 * (y**2) / (1.0 + y**2)
        dy_dt = 0.8 * x / (1.0 + x) - 0.4 * y
        return np.array([dx_dt, dy_dt], dtype=np.float64)

    X = integrate_fractional_ode(hill_rhs, y0=[0.1, 2.5], t=t, alpha=0.8)

    rat_lib = CombinedLibrary(
        [
            PolynomialLibrary(degree=1, include_bias=False),
            RationalLibrary(numerator_degree=1, denominator_degree=2, half_saturation=(1.0,)),
        ]
    )
    model = WeakSINDy(library=rat_lib, threshold=0.08, window_size=40, stride=5, k_start=15)
    model.fit(t, X, orders=0.8, input_features=["x", "y"])
    names = model.feature_names_

    assert model.coef_[0, names.index("x")] == pytest.approx(-0.5, abs=0.03)
    assert model.coef_[0, names.index("y^2/(1+y^2)")] == pytest.approx(1.4, abs=0.05)
    assert model.coef_[1, names.index("x/(1+x)")] == pytest.approx(0.8, abs=0.04)
    assert model.coef_[1, names.index("y")] == pytest.approx(-0.4, abs=0.03)

    # Equivalent recovery via multi-argument CustomLibrary callables
    cust_lib = CustomLibrary(
        functions=[
            lambda x, y: x,
            lambda x, y: y,
            lambda x, y: x / (1.0 + x),
            lambda x, y: (y**2) / (1.0 + y**2),
        ],
        function_names=["x", "y", "mm_x", "hill2_y"],
    )
    c_model = WeakSINDy(library=cust_lib, threshold=0.08, window_size=40, stride=5, k_start=15)
    c_model.fit(t, X, orders=0.8, input_features=["x", "y"])
    assert c_model.coef_[0, 3] == pytest.approx(1.4, abs=0.04)
    assert c_model.coef_[1, 2] == pytest.approx(0.8, abs=0.04)
