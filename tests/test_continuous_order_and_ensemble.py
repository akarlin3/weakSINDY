"""Unit tests for continuous fractional order optimization, EnsembleWeakSINDy UQ, benchmarks, and CLI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from weaksindy import (
    EnsembleWeakSINDy,
    PDELibrary,
    PolynomialLibrary,
    WeakSINDy,
    integrate_fractional_ode,
    simulate_fractional_lorenz,
    simulate_subdiffusive_fisher_kpp_1d,
    simulate_vascular_stromal_rd,
)
from weaksindy.cli import main as cli_main


def test_continuous_fractional_order_optimization_precision() -> None:
    """Verify bounded Brent + Pareto BIC pinpoints off-grid fractional alpha to ~1e-3 precision."""
    true_alpha = 0.734
    t = np.linspace(0.0, 16.0, 801)
    X = integrate_fractional_ode(
        lambda s: np.array([s[1], 1.5 * (1.0 - s[0] ** 2) * s[1] - s[0]]),
        y0=[0.5, 0.5],
        t=t,
        alpha=true_alpha,
    )

    model = WeakSINDy(
        library=PolynomialLibrary(degree=3, include_bias=False),
        threshold=0.1,
        window_size=45,
        stride=5,
        k_start=20,
    )
    model.fit(
        t,
        X,
        optimize_order=True,
        order_bounds=(0.2, 1.0),
        criterion="bic",
        input_features=["x", "y"],
    )

    assert abs(model.order_ - true_alpha) <= 3e-3
    assert np.isfinite(model.aic_) and np.isfinite(model.bic_)
    assert "orders" in model.order_profile_
    assert "bic" in model.order_profile_
    assert "aic" in model.order_profile_
    assert len(model.order_profile_["orders"]) >= 15


def test_fractional_lorenz_benchmark_and_continuous_order() -> None:
    """Test simulate_fractional_lorenz and continuous order + sparse recovery."""
    t = np.linspace(0.0, 6.0, 601)
    X, truth = simulate_fractional_lorenz(t, alpha=0.85)
    assert X.shape == (601, 3)
    assert truth["alpha"] == pytest.approx(0.85)
    assert truth["terms"]["x"]["y"] == pytest.approx(10.0)

    model = WeakSINDy(
        library=PolynomialLibrary(degree=2, include_bias=False),
        threshold=0.25,
        window_size=40,
        stride=5,
        k_start=15,
    )
    model.fit(t, X, optimize_order=True, order_bounds=(0.4, 1.0), input_features=["x", "y", "z"])

    assert abs(model.order_ - 0.85) <= 2e-3
    names = model.feature_names_
    assert model.coef_[0, names.index("x")] == pytest.approx(-10.0, rel=2e-2)
    assert model.coef_[0, names.index("y")] == pytest.approx(10.0, rel=2e-2)
    assert model.coef_[1, names.index("x")] == pytest.approx(28.0, rel=2e-2)
    assert model.coef_[1, names.index("x z")] == pytest.approx(-1.0, rel=2e-2)
    assert model.coef_[2, names.index("x y")] == pytest.approx(1.0, rel=2e-2)
    assert model.coef_[2, names.index("z")] == pytest.approx(-8.0 / 3.0, rel=2e-2)


@pytest.mark.parametrize("agg_mode", ["median", "mean", "stability_selection"])
def test_ensemble_weaksindy_noisy_fisher_kpp_uq_and_conformal_ci(agg_mode: str) -> None:
    """Test EnsembleWeakSINDy inclusion probabilities and conformal CIs on 22 dB noisy Fisher-KPP."""
    t = np.linspace(0.0, 3.5, 211)
    x = np.linspace(0.0, 10.0, 56)
    bench = simulate_subdiffusive_fisher_kpp_1d(
        t, x, alpha=0.7, D=0.12, r=1.0, snr_db=22.0, seed=42
    )
    assert bench.X.shape == (211, 56, 1)

    pde_lib = PDELibrary(
        dx=bench.dx,  # type: ignore[arg-type]
        poly_degree=2,
        derivative_orders=(1, 2),
        include_bias=False,
        boundary="neumann",
    )
    ens = EnsembleWeakSINDy(
        library=pde_lib,
        threshold=0.04,
        n_models=40,
        subsample_fraction=0.7,
        replace=True,
        aggregate=agg_mode,  # type: ignore[arg-type]
        inclusion_threshold=0.5,
        window_size=36,
        stride=5,
        spatial_window_size=16,
        spatial_stride=4,
        k_start=12,
        random_state=123,
    )
    ens.fit(t, bench.X, orders=0.7, input_features=["u"])

    names = ens.feature_names_
    idx_u = names.index("u")
    idx_u2 = names.index("u^2")
    idx_uxx = names.index("u_xx")
    idx_ux = names.index("u_x")

    # True PDE terms have high inclusion probabilities (> 0.85), spurious u_x has low inclusion
    assert ens.inclusion_probabilities_[0, idx_u] >= 0.85
    assert ens.inclusion_probabilities_[0, idx_u2] >= 0.85
    assert ens.inclusion_probabilities_[0, idx_uxx] >= 0.85
    assert ens.inclusion_probabilities_[0, idx_ux] <= 0.35

    # Recovered aggregated coefficients match ground truth D=0.12, r=1.0, -r=-1.0
    assert ens.coef_[0, idx_u] == pytest.approx(1.0, abs=0.12)
    assert ens.coef_[0, idx_u2] == pytest.approx(-1.0, abs=0.15)
    assert ens.coef_[0, idx_uxx] == pytest.approx(0.12, abs=0.03)
    assert ens.coef_[0, idx_ux] == pytest.approx(0.0, abs=1e-12)

    # Conformal & percentile confidence intervals
    ci_lo_95, ci_hi_95 = ens.coef_ci(alpha=0.05, method="conformal")
    ci_lo_80, ci_hi_80 = ens.coef_ci(alpha=0.20, method="conformal")
    p_lo_95, p_hi_95 = ens.coef_ci(alpha=0.05, method="percentile")

    assert ci_lo_95.shape == ens.coef_.shape
    assert np.all(ci_lo_95 <= ci_hi_95)
    # 95% conformal intervals are at least as wide as 80% conformal intervals and 95% raw percentiles
    assert np.all((ci_hi_95 - ci_lo_95) >= (ci_hi_80 - ci_lo_80) - 1e-12)
    assert np.all((ci_hi_95 - ci_lo_95) >= (p_hi_95 - p_lo_95) - 1e-12)

    # True parameters lie inside 95% conformal intervals
    assert ci_lo_95[0, idx_u] <= 1.0 <= ci_hi_95[0, idx_u]
    assert ci_lo_95[0, idx_uxx] <= 0.12 <= ci_hi_95[0, idx_uxx]

    # Ensemble uncertainty propagation in predict
    pred_mean, pred_std = ens.predict(t, bench.X, return_std=True)
    assert pred_mean.shape == bench.X.shape
    assert pred_std.shape == bench.X.shape
    assert np.all(pred_std >= 0.0)


def test_ensemble_library_subsampling_and_vascular_stromal_benchmark() -> None:
    """Verify library subsampling in EnsembleWeakSINDy and simulate_vascular_stromal_rd benchmark."""
    t = np.linspace(0.0, 4.0, 161)
    x = np.linspace(0.0, 10.0, 48)
    bench = simulate_vascular_stromal_rd(t, x, alpha=0.75)
    assert bench.X.shape == (161, 48, 3)
    assert bench.ground_truth["alpha"] == pytest.approx(0.75)
    assert set(bench.input_features) == {"p", "c", "n"}

    lib = PDELibrary(
        dx=bench.dx,  # type: ignore[arg-type]
        poly_degree=2,
        derivative_orders=(2,),
        include_bias=False,
        include_michaelis_menten=True,
        boundary="neumann",
    )
    ens = EnsembleWeakSINDy(
        library=lib,
        threshold=0.008,
        n_models=30,
        subsample_fraction=0.75,
        library_subsample_fraction=0.85,
        aggregate="stability_selection",
        window_size=28,
        stride=4,
        spatial_window_size=16,
        spatial_stride=4,
        k_start=10,
        random_state=7,
    )
    ens.fit(t, bench.X, orders=[0.5, 0.75, 1.0], input_features=["p", "c", "n"])
    assert ens.order_ == pytest.approx(0.75)
    assert ens.r2_score_ > 0.98
    eqs_uq = ens.equations(include_uncertainty=True)
    assert len(eqs_uq) == 3
    assert "CI=" in eqs_uq[0]


def test_cli_ensemble_optimize_order_json_and_markdown(tmp_path: Path) -> None:
    """Test CLI `weaksindy --ensemble --n-models 25 --optimize-order` JSON and Markdown outputs."""
    t = np.linspace(0.0, 20.0, 801)
    X_clean = integrate_fractional_ode(
        lambda s: np.array([s[1], 2.0 * (1.0 - s[0] ** 2) * s[1] - s[0]]),
        y0=[0.5, 0.5],
        t=t,
        alpha=0.82,
    )
    rng = np.random.default_rng(11)
    X = X_clean + rng.normal(
        0.0, np.sqrt(np.var(X_clean, axis=0) * (10.0 ** (-28.0 / 10.0))), size=X_clean.shape
    )

    csv_path = tmp_path / "frac_vdp.csv"
    header = "t,x,y\n"
    body = "\n".join(f"{ti:.6f},{row[0]:.6f},{row[1]:.6f}" for ti, row in zip(t, X))
    csv_path.write_text(header + body + "\n", encoding="utf-8")

    json_out = tmp_path / "uq_report.json"
    md_out = tmp_path / "uq_report.md"

    runner = CliRunner()
    res_json = runner.invoke(
        cli_main,
        [
            str(csv_path),
            "--ensemble",
            "--n-models",
            "25",
            "--optimize-order",
            "--order-bounds",
            "0.4,1.0",
            "--degree",
            "3",
            "--threshold",
            "0.10",
            "-o",
            str(json_out),
        ],
    )
    assert res_json.exit_code == 0, res_json.output
    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert data["ensemble"] is True
    assert data["optimize_order"] is True
    assert abs(data["selected_order"] - 0.82) < 0.015
    assert "inclusion_probabilities" in data
    assert "confidence_intervals" in data
    assert len(data["confidence_intervals"]["ci_low"]) == 2

    res_md = runner.invoke(
        cli_main,
        [
            str(csv_path),
            "--ensemble",
            "--n-models",
            "20",
            "--orders",
            "0.82",
            "--degree",
            "3",
            "--threshold",
            "0.10",
            "-o",
            str(md_out),
            "-q",
        ],
    )
    assert res_md.exit_code == 0, res_md.output
    md_text = md_out.read_text(encoding="utf-8")
    assert "# weakSINDY Discovery Report" in md_text
    assert "Inclusion Prob" in md_text
    assert "Conformal CI" in md_text
