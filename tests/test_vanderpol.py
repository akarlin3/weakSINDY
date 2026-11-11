"""Fractional Van der Pol oscillator benchmark tests (clean & low-SNR recovery)."""

from __future__ import annotations

import numpy as np
import pytest

from weaksindy import (
    SR3,
    PolynomialLibrary,
    WeakSINDy,
    integrate_fractional_ode,
)

MU = 2.0
CANDIDATES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def _vdp_rhs(state: np.ndarray) -> np.ndarray:
    x, y = state
    return np.array([y, MU * (1.0 - x * x) * y - x], dtype=np.float64)


def _add_snr_noise(X: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = X.copy()
    for i in range(X.shape[-1]):
        var_sig = float(np.var(X[:, i]))
        var_noise = var_sig * (10.0 ** (-snr_db / 10.0))
        noisy[:, i] += rng.normal(0.0, np.sqrt(var_noise), size=X.shape[0])
    return noisy


@pytest.mark.parametrize("true_alpha", [0.5, 0.7, 0.9, 1.0])
def test_vanderpol_clean_two_sided_identifiability(true_alpha: float) -> None:
    """Verify exact fractional sensitivity (0.5, 0.7, 0.9) and integer specificity (1.0)."""
    t = np.linspace(0.0, 20.0, 800)
    X_clean = integrate_fractional_ode(_vdp_rhs, y0=[0.5, 0.5], t=t, alpha=true_alpha)

    model = WeakSINDy(
        library=PolynomialLibrary(degree=3, include_bias=True),
        threshold=0.1,
        use_weak=True,
        window_size=50,
        stride=5,
        k_start=20,
    )
    model.fit(t, X_clean, orders=CANDIDATES, input_features=["x", "y"])

    assert model.order_ == pytest.approx(true_alpha, abs=1e-8)
    assert model.r2_score_ > 0.98

    # Verify sparse structure: D^a x = 1*y, D^a y = -1*x + 2*y - 2*x^2 y
    names = model.feature_names_
    ix = names.index("x")
    iy = names.index("y")
    ix2y = names.index("x^2 y")

    assert model.coef_[0, iy] == pytest.approx(1.0, rel=5e-2)
    assert model.coef_[1, ix] == pytest.approx(-1.0, rel=8e-2)
    assert model.coef_[1, iy] == pytest.approx(MU, rel=8e-2)
    assert model.coef_[1, ix2y] == pytest.approx(-MU, rel=8e-2)
    assert int(np.sum(model.coef_ != 0.0)) == 4


def test_vanderpol_sr3_sparse_coefficient_recovery() -> None:
    t = np.linspace(0.0, 20.0, 800)
    X_clean = integrate_fractional_ode(_vdp_rhs, y0=[0.5, 0.5], t=t, alpha=0.8)

    model = WeakSINDy(
        library=PolynomialLibrary(degree=3, include_bias=True),
        optimizer=SR3(threshold=0.15, nu=0.5, reg_type="l0"),
        window_size=50,
        stride=5,
        k_start=20,
    )
    model.fit(t, X_clean, orders=[0.6, 0.8, 1.0], input_features=["x", "y"])
    assert model.order_ == pytest.approx(0.8)

    pred = model.predict(t, X_clean)
    assert pred.shape == X_clean.shape


def test_vanderpol_low_snr_weak_vs_pointwise_rescue() -> None:
    """At SNR = 15 dB for alpha = 0.9, weak-form recovers exact order while pointwise fails."""
    true_alpha = 0.9
    t = np.linspace(0.0, 20.0, 800)
    X_clean = integrate_fractional_ode(_vdp_rhs, y0=[0.5, 0.5], t=t, alpha=true_alpha)

    weak_successes = 0
    pw_successes = 0
    n_trials = 8

    for trial in range(n_trials):
        X_noisy = _add_snr_noise(X_clean, snr_db=15.0, seed=100 + trial)

        weak_model = WeakSINDy(
            library=PolynomialLibrary(degree=3, include_bias=True),
            threshold=0.05,
            use_weak=True,
            window_size=50,
            stride=5,
            k_start=20,
            refit=False,
        )
        weak_model.fit(
            t, X_noisy, orders=CANDIDATES, X_clean=X_clean, true_order=true_alpha
        )
        if abs(weak_model.order_ - true_alpha) < 1e-6:
            weak_successes += 1

        pw_model = WeakSINDy(
            library=PolynomialLibrary(degree=3, include_bias=True),
            threshold=0.05,
            use_weak=False,
            k_start=20,
            refit=False,
        )
        pw_model.fit(
            t, X_noisy, orders=CANDIDATES, X_clean=X_clean, true_order=true_alpha
        )
        if abs(pw_model.order_ - true_alpha) < 1e-6:
            pw_successes += 1

    assert weak_successes >= 7
    assert pw_successes <= 2
