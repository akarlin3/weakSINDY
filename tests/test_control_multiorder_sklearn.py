"""Deterministic unit tests for WeakSINDyControl (weakSINDYc), MultiOrderWeakSINDy, and SklearnWeakSINDyRegressor."""
from __future__ import annotations

import numpy as np
import pytest

import weaksindy
from weaksindy import (
    ControlLibrary,
    MultiOrderWeakSINDy,
    SklearnWeakSINDyRegressor,
    WeakSINDyControl,
    fractional_derivative_gl,
    generate_controlled_fractional_system,
    integrate_multi_order_ode,
    multi_order_fractional_derivative,
    weakSINDYc,
)
from weaksindy._sklearn_compat import PureBaseEstimator, PureRegressorMixin


def test_version_and_control_aliases():
    assert weaksindy.__version__ == "1.2.0"
    assert weakSINDYc is WeakSINDyControl


def test_weaksindy_control_radiotherapy_pk_benchmark_and_simulation():
    """Verify WeakSINDyControl recovers bilinear x1*u_rad, linear u_drug, and fractional order alpha."""
    bench = generate_controlled_fractional_system(
        alpha=0.80,
        snr_db=36.0,
        seed=20260613,
    )
    # Determinism check
    bench_repeat = generate_controlled_fractional_system(
        alpha=0.80,
        snr_db=36.0,
        seed=20260613,
    )
    np.testing.assert_array_equal(bench.X, bench_repeat.X)
    np.testing.assert_array_equal(bench.U, bench_repeat.U)

    model = WeakSINDyControl(
        state_degree=2,
        control_degree=1,
        include_bias=False,
        include_bilinear=True,
        threshold=0.05,
        optimize_order=True,
        order_bounds=(0.45, 1.0),
        window_size=40,
        stride=4,
    )
    model.fit(
        bench.t,
        bench.X,
        bench.U,
        state_names=["x1", "x2"],
        control_names=["u_rad", "u_drug"],
    )

    assert abs(model.order_ - 0.80) < 0.04
    assert model.r2_score_ > 0.98

    fmap = {name: idx for idx, name in enumerate(model.feature_names_)}
    assert "x1 u_rad" in fmap
    assert "u_drug" in fmap

    # Equation 0 (x1): r*x1 - kappa*x1^2 - gamma*x1*x2 - beta_rad*x1*u_rad
    c0 = model.coef_[0]
    assert np.isclose(c0[fmap["x1"]], 0.45, atol=0.06)
    assert np.isclose(c0[fmap["x1^2"]], -0.30, atol=0.08)
    assert np.isclose(c0[fmap["x1 x2"]], -0.50, atol=0.08)
    assert np.isclose(c0[fmap["x1 u_rad"]], -0.85, atol=0.08)
    assert c0[fmap["u_drug"]] == 0.0

    # Equation 1 (x2): -k_e*x2 + b_in*u_drug
    c1 = model.coef_[1]
    assert np.isclose(c1[fmap["x2"]], -0.60, atol=0.05)
    assert np.isclose(c1[fmap["u_drug"]], 1.00, atol=0.05)

    # Forward simulation under novel control schedule
    X_sim = model.simulate(bench.U, x0=bench.X_clean[0], t=bench.t)
    assert X_sim.shape == bench.X.shape
    assert float(np.sqrt(np.mean((X_sim - bench.X_clean) ** 2))) < 0.04
    assert len(model.equations()) == 2


def test_weaksindy_control_spatiotemporal_1d_pde():
    """Verify WeakSINDyControl discovers forced 1D diffusion-decay PDE with spatial IBP."""
    nt, nx = 141, 32
    t = np.linspace(0.0, 2.0, nt)
    x = np.linspace(0.0, 2.0 * np.pi, nx, endpoint=False)
    dx = float(x[1] - x[0])
    T_g, X_g = np.meshgrid(t, x, indexing="ij")

    # Manufactured exact solution: u(t, x) = exp(-0.4 t) sin(x) + 0.3 sin(2 t) cos(x)
    # Under d/dt u = 0.2 u_xx - 0.2 u + f_ext(t, x), choose exogenous forcing f_ext analytically:
    state = np.exp(-0.4 * T_g) * np.sin(X_g) + 0.3 * np.sin(2.0 * T_g) * np.cos(X_g)
    du_dt = -0.4 * np.exp(-0.4 * T_g) * np.sin(X_g) + 0.6 * np.cos(2.0 * T_g) * np.cos(X_g)
    u_xx = -state
    u_ctrl = (du_dt - (0.2 * u_xx - 0.2 * state)) / 0.8  # true control weight = +0.8

    lib = ControlLibrary(
        state_degree=1,
        control_degree=1,
        include_bias=False,
        include_bilinear=False,
        dx=dx,
        derivative_orders=(2,),
        boundary="periodic",
    )
    pde_ctrl = WeakSINDyControl(
        library=lib,
        threshold=0.05,
        orders=1.0,
        window_size=28,
        spatial_window_size=12,
        spatial_stride=4,
    )
    pde_ctrl.fit(t, state[..., np.newaxis], u_ctrl[..., np.newaxis], dx=dx, state_names=["u"], control_names=["f"])
    fmap = {n: i for i, n in enumerate(pde_ctrl.feature_names_)}
    assert np.isclose(pde_ctrl.coef_[0, fmap["u"]], -0.20, atol=0.03)
    assert np.isclose(pde_ctrl.coef_[0, fmap["u_xx"]], 0.20, atol=0.03)
    assert np.isclose(pde_ctrl.coef_[0, fmap["f"]], 0.80, atol=0.03)


def test_multi_order_weaksindy_incommensurate_and_two_scale():
    """Verify MultiOrderWeakSINDy recovers distinct vector orders alpha=(0.62, 0.91) and two-scale operators."""
    t = np.linspace(0.0, 18.0, 721)
    true_alphas = (0.62, 0.91)

    def rhs(y: np.ndarray) -> np.ndarray:
        return np.array([-0.45 * y[0] + 1.15 * y[1], -1.15 * y[0] - 0.20 * y[1]], dtype=np.float64)

    X = integrate_multi_order_ode(rhs, y0=[1.0, 0.25], t=t, alphas=true_alphas, n_corrector=3)
    # Cross-check multi_order_fractional_derivative consistency on trajectory tail
    lhs_gl = multi_order_fractional_derivative(X, t=t, alphas=true_alphas)
    rhs_true = np.column_stack([-0.45 * X[:, 0] + 1.15 * X[:, 1], -1.15 * X[:, 0] - 0.20 * X[:, 1]])
    np.testing.assert_allclose(lhs_gl[40:], rhs_true[40:], atol=1e-2)

    mo = MultiOrderWeakSINDy(
        threshold=0.06,
        order_bounds=(0.25, 1.0),
        window_size=44,
        stride=4,
    )
    mo.fit(t, X, input_features=["x", "y"])
    np.testing.assert_allclose(mo.orders_, np.array(true_alphas), atol=0.025)
    assert mo.r2_score_ > 0.995

    fmap = {n: i for i, n in enumerate(mo.feature_names_)}
    assert np.isclose(mo.coef_[0, fmap["x"]], -0.45, atol=0.03)
    assert np.isclose(mo.coef_[0, fmap["y"]], 1.15, atol=0.03)
    assert np.isclose(mo.coef_[1, fmap["x"]], -1.15, atol=0.03)
    assert np.isclose(mo.coef_[1, fmap["y"]], -0.20, atol=0.03)

    # Two-scale nonlinear Van der Pol / Duffing oscillator (breaks linear sinusoidal subspace degeneracy)
    X_ts = integrate_multi_order_ode(
        lambda y: np.array(
            [1.2 * y[1] - 0.6 * (y[0] ** 3), -1.2 * y[0] + 0.8 * (1.0 - y[0] ** 2) * y[1]],
            dtype=np.float64,
        ),
        y0=[1.1, 0.2],
        t=t,
        alphas=(0.92, 0.88),
        second_alphas=(0.42, 0.38),
        second_weights=(0.65, 0.50),
        n_corrector=3,
    )
    mo_ts = MultiOrderWeakSINDy(
        threshold=0.06,
        two_scale=True,
        min_order_gap=0.25,
        order_bounds=(0.20, 1.0),
        window_size=44,
        stride=4,
    )
    mo_ts.fit(t, X_ts, input_features=["x", "y"])
    assert mo_ts.second_orders_ is not None and mo_ts.second_weights_ is not None
    np.testing.assert_allclose(mo_ts.orders_, np.array([0.92, 0.88]), atol=0.05)
    np.testing.assert_allclose(mo_ts.second_orders_, np.array([0.42, 0.38]), atol=0.06)
    np.testing.assert_allclose(mo_ts.second_weights_, np.array([0.65, 0.50]), atol=0.08)
    assert len(mo_ts.equations()) == 2


def test_sklearn_adapter_contract_pure_fallback_and_gridsearchcv():
    """Verify SklearnWeakSINDyRegressor adheres to get_params/set_params/fit/predict/score and GridSearchCV."""
    t = np.linspace(0.0, 10.0, 401)
    dt = float(t[1] - t[0])
    X = integrate_multi_order_ode(
        lambda s: np.array([s[1], -s[0] - 0.25 * s[1]]),
        y0=[1.0, 0.0],
        t=t,
        alphas=(0.85, 0.85),
    )
    y_deriv = fractional_derivative_gl(X, t=dt, alpha=0.85, axis=0)

    # 1. Pure fallback get_params / set_params / score verification (zero-sklearn path)
    class _DummyPureEst(PureBaseEstimator, PureRegressorMixin):
        def __init__(self, alpha: float = 1.0) -> None:
            self.alpha = alpha

        def fit(self, X: np.ndarray, y: np.ndarray) -> _DummyPureEst:
            self.w_ = float(self.alpha)
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.asarray(X, dtype=np.float64) * self.w_

    pure_est = _DummyPureEst(alpha=0.5).set_params(alpha=2.0).fit(X, 2.0 * X)
    assert pure_est.get_params()["alpha"] == 2.0
    assert np.isclose(pure_est.score(X, 2.0 * X), 1.0)

    # 2. SklearnWeakSINDyRegressor supervised (X, y) & trajectory (X, y=None) modes
    reg = SklearnWeakSINDyRegressor(dt=dt, degree=2, include_bias=False, threshold=0.05, order=0.85)
    params = reg.get_params(deep=True)
    assert params["threshold"] == 0.05
    reg.set_params(threshold=0.04)
    assert reg.threshold == 0.04

    reg.fit(X[20:], y_deriv[20:])
    assert reg.score(X[20:], y_deriv[20:]) > 0.995
    pred = reg.predict(X[20:])
    assert pred.shape == y_deriv[20:].shape

    # 1D target y shape preservation
    reg_1d = SklearnWeakSINDyRegressor(dt=dt, degree=2, include_bias=False, threshold=0.04)
    reg_1d.fit(X[20:], y_deriv[20:, 0])
    assert reg_1d.predict(X[20:]).ndim == 1

    # Self-supervised trajectory discovery fit(X)
    reg_traj = SklearnWeakSINDyRegressor(
        dt=dt, degree=2, include_bias=False, threshold=0.04, order=0.85
    ).fit(X)
    assert reg_traj.score(X) > 0.995

    # 3. Direct drop-in into sklearn.model_selection.GridSearchCV & Pipeline (when sklearn installed)
    sklearn_ms = pytest.importorskip("sklearn.model_selection")
    sklearn_pipe = pytest.importorskip("sklearn.pipeline")
    sklearn_base = pytest.importorskip("sklearn.base")

    cloned = sklearn_base.clone(reg)
    assert cloned.get_params() == reg.get_params()

    gs = sklearn_ms.GridSearchCV(
        SklearnWeakSINDyRegressor(dt=dt, degree=2, include_bias=False, order=0.85),
        param_grid={"threshold": [0.02, 0.05, 0.20], "optimizer": ["stlsq", "sr3"]},
        cv=3,
    )
    gs.fit(X[20:], y_deriv[20:])
    assert gs.best_score_ > 0.99
    assert hasattr(gs.best_estimator_, "coef_")

    pipe = sklearn_pipe.Pipeline(
        [("sindy", SklearnWeakSINDyRegressor(dt=dt, degree=2, include_bias=False, threshold=0.04))]
    )
    pipe.fit(X[20:], y_deriv[20:])
    assert pipe.score(X[20:], y_deriv[20:]) > 0.995
