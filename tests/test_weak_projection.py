"""Unit tests for C^p weak-form test functions, adjoint summation by parts, libraries, and CLI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from weaksindy import (
    SR3,
    STLSQ,
    CombinedLibrary,
    FourierLibrary,
    PDELibrary,
    PolynomialLibrary,
    WeakSINDy,
    build_weak_weights,
    bump_test_function,
    fractional_derivative_gl,
    generate_windows,
    noise_amplification_factor,
    project_weak_form,
    weak_noise_amplification_factor,
)
from weaksindy.cli import main as cli_main


def test_bump_test_function_properties() -> None:
    dt = 0.01
    phi = bump_test_function(window_size=50, degree=4, dt=dt, normalize=True)
    assert phi.shape == (51,)
    # Compact support vanishes at window endpoints
    assert phi[0] == pytest.approx(0.0, abs=1e-15)
    assert phi[-1] == pytest.approx(0.0, abs=1e-15)
    # Strictly positive in interior and symmetric
    assert np.all(phi[1:-1] > 0.0)
    np.testing.assert_allclose(phi, phi[::-1], atol=1e-15)
    # Unit quadrature normalization
    assert float(np.sum(phi) * dt) == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_discrete_fractional_summation_by_parts_exactness(alpha: float) -> None:
    """Adjoint weak projection <X, L_alpha^* phi> must match <L_alpha X, phi> to machine precision."""
    rng = np.random.default_rng(7)
    nt, nx, n_vars = 120, 8, 2
    dt = 0.015
    X_3d = rng.standard_normal((nt, nx, n_vars))
    Theta_3d = rng.standard_normal((nt, nx, 5))

    H, S, k_start = 30, 7, 10
    wins = generate_windows(nt, window_size=H, stride=S, k_start=k_start)
    phi = bump_test_function(window_size=H, degree=4, dt=dt, normalize=True)

    Theta_weak, y_weak = project_weak_form(
        X_3d, Theta=Theta_3d, t=dt, alpha=alpha, degree=4, windows=wins
    )

    # Brute-force inner product of pointwise GL derivative against phi on each window
    dX_pw = fractional_derivative_gl(X_3d, t=dt, alpha=alpha, axis=0)
    expected_y = []
    expected_theta = []
    for k_a, k_b in wins:
        expected_y.append(np.sum(dX_pw[k_a : k_b + 1] * phi[:, None, None], axis=0) * dt)
        expected_theta.append(np.sum(Theta_3d[k_a : k_b + 1] * phi[:, None, None], axis=0) * dt)

    expected_y_flat = np.concatenate(expected_y, axis=0)
    expected_theta_flat = np.concatenate(expected_theta, axis=0)

    np.testing.assert_allclose(y_weak, expected_y_flat, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(Theta_weak, expected_theta_flat, rtol=1e-12, atol=1e-12)


def test_build_weak_weights_toeplitz_matches_fft_adjoint() -> None:
    nt = 100
    dt = 0.01
    wins = generate_windows(nt, window_size=24, stride=6, k_start=8)
    phi = bump_test_function(window_size=24, degree=4, dt=dt)

    Phi_t, Psi_t = build_weak_weights(nt, wins, alpha=0.8, phi=phi, dt=dt, toeplitz_cutoff=4096)
    Phi_f, Psi_f = build_weak_weights(nt, wins, alpha=0.8, phi=phi, dt=dt, toeplitz_cutoff=16)

    np.testing.assert_allclose(Phi_t, Phi_f, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(Psi_t, Psi_f, rtol=1e-11, atol=1e-12)


def test_weak_noise_variance_collapse_and_monte_carlo() -> None:
    """Verify >25 dB variance reduction of weak projection vs pointwise GL across alpha."""
    nt = 500
    dt = 5.0 / (nt - 1)
    for alpha in [0.5, 0.7, 0.9, 1.0]:
        a_pw = noise_amplification_factor(alpha, dt=dt, n_steps=nt)
        a_wk = weak_noise_amplification_factor(alpha, dt=dt, window_size=50, degree=4, n_steps=nt)
        db_reduction = 10.0 * np.log10(a_pw / a_wk)
        assert db_reduction > 25.0

    # Empirical Monte Carlo verification of weak_noise_amplification_factor
    rng = np.random.default_rng(99)
    sigma = 0.1
    noise = rng.normal(0.0, sigma, size=(nt, 600))
    _, y_wk = project_weak_form(
        noise, t=dt, alpha=0.8, window_size=50, stride=50, degree=4, windows=[(400, 450)]
    )
    emp_var_ratio = float(np.var(y_wk[0]) / (sigma**2))
    ana_var_ratio = weak_noise_amplification_factor(
        0.8, dt=dt, window_size=50, degree=4, n_steps=nt, k_a=400
    )
    assert emp_var_ratio == pytest.approx(ana_var_ratio, rel=0.08)


def test_libraries_and_sr3_trigonometric_recovery() -> None:
    """Recover damped pendulum dx0/dt = x1, dx1/dt = -0.25 x1 - 2.0 sin(x0) with CombinedLibrary + SR3."""
    from scipy.integrate import solve_ivp

    t = np.linspace(0.0, 12.0, 2401)
    sol = solve_ivp(
        lambda _t, s: [s[1], -0.25 * s[1] - 2.0 * np.sin(s[0])],
        [0.0, 12.0],
        [1.2, 0.0],
        t_eval=t,
        rtol=1e-9,
        atol=1e-11,
    )
    X = sol.y.T

    lib = CombinedLibrary(
        [PolynomialLibrary(degree=2, include_bias=True), FourierLibrary(frequencies=(1.0,))]
    )
    model = WeakSINDy(
        library=lib,
        optimizer=SR3(threshold=0.05, nu=0.5, reg_type="l0"),
        window_size=80,
        stride=10,
        k_start=10,
    )
    model.fit(t, X, orders=1.0, input_features=["theta", "omega"])

    names = model.feature_names_
    assert names.count("1") == 1  # Deduplicated bias
    idx_omega = names.index("omega")
    idx_sin = names.index("sin(theta)")

    assert model.coef_[0, idx_omega] == pytest.approx(1.0, abs=1e-2)
    assert model.coef_[1, idx_omega] == pytest.approx(-0.25, abs=1e-2)
    assert model.coef_[1, idx_sin] == pytest.approx(-2.0, abs=2e-2)
    assert int(np.sum(model.coef_ != 0.0)) == 3


def test_pde_library_and_stlsq_cli_roundtrip(tmp_path: Path) -> None:
    """Test CLI discovery on CSV and NPZ time-series files."""
    t = np.linspace(0.0, 6.0, 301)
    # Linear oscillator dx/dt = y, dy/dt = -x - 0.2 y
    from scipy.integrate import solve_ivp

    sol = solve_ivp(
        lambda _t, s: [s[1], -s[0] - 0.2 * s[1]],
        [0.0, 6.0],
        [1.0, 0.0],
        t_eval=t,
        rtol=1e-9,
        atol=1e-11,
    )
    X = sol.y.T

    csv_path = tmp_path / "oscillator.csv"
    header = "t,x,y\n"
    rows = "\n".join(f"{ti:.8f},{xi:.8f},{yi:.8f}" for ti, (xi, yi) in zip(t, X))
    csv_path.write_text(header + rows + "\n", encoding="utf-8")

    report_path = tmp_path / "report.json"
    runner = CliRunner()
    res = runner.invoke(
        cli_main,
        [
            str(csv_path),
            "--orders",
            "0.6,0.8,1.0",
            "--degree",
            "2",
            "--threshold",
            "0.05",
            "--window-size",
            "30",
            "-o",
            str(report_path),
        ],
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["selected_order"] == pytest.approx(1.0)
    assert payload["n_active_terms"] == 3

    # Also test 3D PDE NPZ input via CLI
    x_grid = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    dx = float(x_grid[1] - x_grid[0])
    # Heat decay u_t = 0.1 u_xx with u(x, t) = exp(-0.1 t) sin(x) -> u_xx = -u
    u_pde = (np.exp(-0.1 * t[:, None]) * np.sin(x_grid[None, :]))[:, :, None]
    npz_path = tmp_path / "heat.npz"
    np.savez(npz_path, t=t, x=x_grid, u=u_pde)

    res_npz = runner.invoke(
        cli_main,
        [str(npz_path), "--orders", "1.0", "--degree", "1", "--threshold", "0.02", "-q"],
    )
    assert res_npz.exit_code == 0, res_npz.output
    assert PDELibrary(dx=dx).fit(u_pde).n_output_features_ > 0
    assert STLSQ(threshold=0.05).fit(np.eye(4), np.ones((4, 1))).coef_.shape == (1, 4)
