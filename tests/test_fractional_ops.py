"""Unit tests for Grünwald–Letnikov fractional operators, amplification factors, and integrators."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import binom, gamma

from weaksindy import (
    fractional_derivative_gl,
    gl_operator_matrix,
    gl_weights,
    integrate_fractional_ode,
    noise_amplification_factor,
)


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0])
def test_gl_weights_match_alternating_binomial(alpha: float) -> None:
    n = 40
    w = gl_weights(alpha, n)
    j = np.arange(n, dtype=np.float64)
    expected = ((-1.0) ** j) * binom(alpha, j)
    assert w.shape == (n,)
    np.testing.assert_allclose(w, expected, rtol=1e-12, atol=1e-14)


def test_gl_weights_integer_orders_are_exact_finite_differences() -> None:
    w1 = gl_weights(1.0, 8)
    np.testing.assert_allclose(w1, [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], atol=1e-15)

    w2 = gl_weights(2.0, 8)
    np.testing.assert_allclose(w2, [1.0, -2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], atol=1e-15)


@pytest.mark.parametrize("alpha", [0.3, 0.5, 0.7, 0.9, 1.0])
def test_fractional_derivative_gl_power_law_accuracy(alpha: float) -> None:
    """For f(t) = t^3, D_t^alpha (t^3) = Gamma(4) / Gamma(4 - alpha) * t^(3 - alpha)."""
    t = np.linspace(0.0, 2.0, 1001)
    f = t**3
    exact = (gamma(4.0) / gamma(4.0 - alpha)) * (t ** (3.0 - alpha))
    num = fractional_derivative_gl(f, t=t, alpha=alpha)

    # First-order O(dt) GL accuracy away from t=0
    np.testing.assert_allclose(num[100:], exact[100:], rtol=1.5e-2, atol=1e-2)


def test_fractional_derivative_toeplitz_matches_fft_and_axes() -> None:
    rng = np.random.default_rng(123)
    X_3d = rng.standard_normal((80, 30, 3))
    dt = 0.02

    d_toep = fractional_derivative_gl(X_3d, t=dt, alpha=0.65, axis=0, toeplitz_cutoff=4096)
    d_fft = fractional_derivative_gl(X_3d, t=dt, alpha=0.65, axis=0, toeplitz_cutoff=16)
    np.testing.assert_allclose(d_toep, d_fft, rtol=1e-11, atol=1e-12)

    # Spatial axis=1 consistency with explicit operator matrix
    L_space = gl_operator_matrix(0.5, 30, dt=0.1)
    d_space = fractional_derivative_gl(X_3d, t=0.1, alpha=0.5, axis=1)
    expected_slice = L_space @ X_3d[5, :, :]
    np.testing.assert_allclose(d_space[5], expected_slice, rtol=1e-12, atol=1e-13)


def test_noise_amplification_factor_monotonicity_and_monte_carlo() -> None:
    nt = 500
    dt = 5.0 / (nt - 1)
    orders = [0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
    amps = [noise_amplification_factor(a, dt=dt, n_steps=nt) for a in orders]

    # Strictly increasing across three orders of magnitude for dt ~ 0.01
    assert all(amps[i] < amps[i + 1] for i in range(len(amps) - 1))
    assert amps[0] == pytest.approx(6.62, rel=0.02)
    assert amps[-1] == pytest.approx(2.0 * (dt**-2.0), rel=1e-6)

    # Empirical Monte Carlo check at alpha = 0.7
    rng = np.random.default_rng(42)
    sigma = 0.05
    noise = rng.normal(0.0, sigma, size=(nt, 400))
    d_noise = fractional_derivative_gl(noise, t=dt, alpha=0.7, axis=0)
    emp_amp = float(np.mean(np.var(d_noise[20:], axis=1)) / (sigma**2))
    ana_amp = noise_amplification_factor(0.7, dt=dt, n_steps=nt, k_start=20)
    assert emp_amp == pytest.approx(ana_amp, rel=0.05)


def test_integrate_fractional_ode_analytical_solution() -> None:
    """Solve D_t^alpha y = -lam * y + g(t), y(0) = 0 with exact solution y(t) = t^3."""
    alpha = 0.7
    lam = 0.8
    t = np.linspace(0.0, 2.0, 801)
    coef_t3 = gamma(4.0) / gamma(4.0 - alpha)

    def forced_rhs(t_val: float, y_val: np.ndarray) -> np.ndarray:
        forcing = coef_t3 * (t_val ** (3.0 - alpha)) + lam * (t_val**3)
        return -lam * y_val + forcing

    y_num = integrate_fractional_ode(forced_rhs, y0=[0.0], t=t, alpha=alpha)[:, 0]
    y_exact = t**3
    np.testing.assert_allclose(y_num, y_exact, rtol=1.5e-2, atol=5e-3)


def test_operator_input_validation() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        fractional_derivative_gl(np.ones(10), t=-0.1, alpha=0.5)
    with pytest.raises(ValueError, match="uniformly spaced"):
        fractional_derivative_gl(np.ones(4), t=np.array([0.0, 0.1, 0.25, 0.4]), alpha=0.5)
    with pytest.raises(ValueError, match=">= 0"):
        gl_weights(-0.2, 10)
