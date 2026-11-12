"""Strict ``scikit-learn`` Estimator, Pipeline, and ``GridSearchCV`` adapter for ``weakSINDY``.

Provides :class:`SklearnWeakSINDyRegressor` adhering strictly to the ``scikit-learn``
estimator API:
  - ``fit(X, y=None)``
  - ``predict(X)``
  - ``score(X, y=None)``
  - ``get_params(deep=True)``
  - ``set_params(**params)``

Works seamlessly inside ``sklearn.pipeline.Pipeline`` and ``sklearn.model_selection.GridSearchCV``
when ``scikit-learn`` is installed, while maintaining zero required core dependency on
``scikit-learn`` when running in a pure NumPy/SciPy environment.
"""
from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

from ._sklearn_compat import BaseEstimator, RegressorMixin
from .library import PolynomialLibrary
from .operators import fractional_derivative_gl
from .solvers import SR3, STLSQ, WeakSINDy, _mean_r2_score
from .test_functions import build_weak_weights, bump_test_function, generate_windows


class SklearnWeakSINDyRegressor(BaseEstimator, RegressorMixin):
    """``scikit-learn``-compatible WeakSINDy sparse dynamical regressor.

    Supports two standard workflows:
      1. **Supervised derivative / target regression** ``fit(X, y)``:
         ``X`` has shape ``(n_samples, n_features)`` (state snapshots) and ``y`` has shape
         ``(n_samples,)`` or ``(n_samples, n_targets)`` (target fractional/time derivatives).
         When ``use_weak=True`` and ``contiguous=True``, projects both ``Theta(X)`` and ``y``
         against compact ``C^p`` bump test functions before sparse STLSQ/SR3 regression.
      2. **Self-supervised trajectory discovery** ``fit(X, y=None)``:
         Treats ``X`` as a uniformly sampled trajectory with step ``dt`` and runs full
         :class:`WeakSINDy` causal Grünwald–Letnikov weak-form discovery (including optional
         continuous ``optimize_order``), enabling ``GridSearchCV(est, param_grid).fit(X)``
         directly on raw state trajectories.
    """

    def __init__(
        self,
        dt: float = 0.01,
        degree: int = 3,
        include_bias: bool = True,
        include_interaction: bool = True,
        optimizer: Literal["stlsq", "sr3"] = "stlsq",
        threshold: float = 0.05,
        alpha_reg: float = 1e-12,
        nu: float = 1.0,
        order: float = 1.0,
        orders: Sequence[float] | None = None,
        optimize_order: bool = False,
        order_bounds: tuple[float, float] = (0.15, 1.0),
        criterion: Literal["bic", "aic"] = "bic",
        use_weak: bool = True,
        contiguous: bool = False,
        window_size: int = 32,
        stride: int = 4,
        test_degree: int = 4,
        k_start: int = 10,
    ) -> None:
        self.dt = dt
        self.degree = degree
        self.include_bias = include_bias
        self.include_interaction = include_interaction
        self.optimizer = optimizer
        self.threshold = threshold
        self.alpha_reg = alpha_reg
        self.nu = nu
        self.order = order
        self.orders = orders
        self.optimize_order = optimize_order
        self.order_bounds = order_bounds
        self.criterion = criterion
        self.use_weak = use_weak
        self.contiguous = contiguous
        self.window_size = window_size
        self.stride = stride
        self.test_degree = test_degree
        self.k_start = k_start

    def _build_sparse_solver(self) -> STLSQ | SR3:
        if self.optimizer == "stlsq":
            return STLSQ(threshold=float(self.threshold), alpha=float(self.alpha_reg))
        if self.optimizer == "sr3":
            return SR3(threshold=float(self.threshold), nu=float(self.nu))
        raise ValueError(f"Unsupported optimizer {self.optimizer!r}; expected 'stlsq' or 'sr3'.")

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> SklearnWeakSINDyRegressor:
        """Fit sparse SINDy model from ``(X, y)`` or directly from trajectory ``X`` when ``y is None``."""
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, np.newaxis]
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D array X of shape (n_samples, n_features); got {X_arr.shape}.")

        n_samples, n_features = X_arr.shape
        lib = PolynomialLibrary(
            degree=int(self.degree),
            include_bias=bool(self.include_bias),
            include_interaction=bool(self.include_interaction),
        )
        lib.fit(X_arr)

        if y is None:
            cand_orders: float | Sequence[float] = (
                self.orders if self.orders is not None else (float(self.order),)
            )
            ws = WeakSINDy(
                library=lib,
                optimizer=self._build_sparse_solver(),
                threshold=float(self.threshold),
                alpha_reg=float(self.alpha_reg),
                orders=cand_orders,
                optimize_order=bool(self.optimize_order),
                order_bounds=(float(self.order_bounds[0]), float(self.order_bounds[1])),
                criterion=self.criterion,
                use_weak=bool(self.use_weak),
                window_size=int(self.window_size),
                stride=int(self.stride),
                test_degree=int(self.test_degree),
                k_start=int(self.k_start),
            )
            ws.fit(float(self.dt), X_arr)
            self.model_ = ws
            self.library_ = lib
            self.coef_ = ws.coef_.copy()
            self.ind_ = ws.coef_ != 0.0
            self.order_ = float(ws.order_)
            self.feature_names_ = lib.get_feature_names()
            self.n_features_in_ = n_features
            self._y_was_1d = False
            self._fitted_trajectory_mode = True
            return self

        Y_raw = np.asarray(y, dtype=np.float64)
        y_was_1d = Y_raw.ndim == 1
        Y_arr = Y_raw[:, np.newaxis] if y_was_1d else Y_raw
        if Y_arr.ndim != 2 or Y_arr.shape[0] != n_samples:
            raise ValueError(
                f"Incompatible shapes: X has {n_samples} samples, y has shape {Y_raw.shape}."
            )

        Theta = lib.transform(X_arr)
        if self.use_weak and self.contiguous and n_samples >= max(10, int(self.window_size) + 4):
            H = min(int(self.window_size), max(4, (n_samples - 2) // 2))
            S = max(1, min(int(self.stride), H // 4))
            wins = generate_windows(n_steps=n_samples, window_size=H, stride=S, k_start=0, k_end=n_samples)
            phi_t = bump_test_function(
                window_size=H, degree=int(self.test_degree), dt=float(self.dt), normalize=True
            )
            Phi_t, _ = build_weak_weights(
                n_steps=n_samples, windows=wins, alpha=0.0, phi=phi_t, dt=float(self.dt)
            )
            X_fit = Phi_t.T @ Theta
            Y_fit = Phi_t.T @ Y_arr
        else:
            X_fit = Theta
            Y_fit = Y_arr

        solver = self._build_sparse_solver().fit(X_fit, Y_fit)
        self.library_ = lib
        self.solver_ = solver
        self.coef_ = solver.coef_.copy()
        self.ind_ = solver.ind_.copy()
        self.order_ = float(self.order)
        self.feature_names_ = lib.get_feature_names()
        self.n_features_in_ = n_features
        self._y_was_1d = y_was_1d
        self._fitted_trajectory_mode = False
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target derivatives ``Theta(X) @ coef_.T``."""
        if not hasattr(self, "coef_"):
            raise RuntimeError("SklearnWeakSINDyRegressor must be fitted before calling predict.")
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, np.newaxis]
        if X_arr.ndim != 2 or X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected X with {self.n_features_in_} features; got shape {X_arr.shape}."
            )
        Theta = self.library_.transform(X_arr)
        pred = Theta @ self.coef_.T
        if self._y_was_1d and pred.shape[1] == 1:
            return pred[:, 0]
        return pred

    def score(self, X: np.ndarray, y: np.ndarray | None = None) -> float:
        """Return multi-output ``R^2`` score against ``y`` (or weak-form trajectory ``R^2`` if ``y is None``)."""
        if not hasattr(self, "coef_"):
            raise RuntimeError("SklearnWeakSINDyRegressor must be fitted before calling score.")
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, np.newaxis]

        if y is not None:
            y_pred = self.predict(X_arr)
            return _mean_r2_score(np.asarray(y, dtype=np.float64), y_pred)

        # Self-supervised trajectory R^2 on contiguous trajectory segment X
        n_samples = X_arr.shape[0]
        Theta = self.library_.transform(X_arr)
        eff_k = min(int(self.k_start), max(0, n_samples // 5))
        if self.use_weak and (n_samples - eff_k) >= 8:
            H = min(int(self.window_size), max(4, (n_samples - eff_k - 2) // 2))
            S = max(1, min(int(self.stride), H // 4))
            wins = generate_windows(
                n_steps=n_samples, window_size=H, stride=S, k_start=eff_k, k_end=n_samples
            )
            phi_t = bump_test_function(
                window_size=H, degree=int(self.test_degree), dt=float(self.dt), normalize=True
            )
            Phi_t, Psi_t = build_weak_weights(
                n_steps=n_samples,
                windows=wins,
                alpha=float(self.order_),
                phi=phi_t,
                dt=float(self.dt),
            )
            Y_true = Psi_t.T @ X_arr
            Y_pred = (Phi_t.T @ Theta) @ self.coef_.T
            return _mean_r2_score(Y_true, Y_pred)

        X_dot = fractional_derivative_gl(X_arr, t=float(self.dt), alpha=float(self.order_), axis=0)
        return _mean_r2_score(X_dot[eff_k:], (Theta @ self.coef_.T)[eff_k:])
