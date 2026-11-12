"""Sparse regression solvers (STLSQ, SR3) and WeakSINDy with continuous order optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from scipy.optimize import minimize_scalar

from ._sklearn_compat import BaseEstimator, RegressorMixin
from .library import BaseCandidateLibrary, PDELibrary, PolynomialLibrary
from .operators import extract_spatial_steps, extract_time_step, fractional_derivative_gl
from .test_functions import (
    build_spatial_weak_weights,
    build_weak_weights,
    bump_test_function,
    contract_tensor_weak,
    generate_windows,
    project_weak_form,
    resolve_spatial_windows,
)


def _mean_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute variable-averaged R^2 score with numerical floor protection."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.ndim == 1:
        yt = yt[:, np.newaxis]
        yp = yp[:, np.newaxis]
    ss_res = np.sum((yt - yp) ** 2, axis=0)
    ss_tot = np.sum((yt - np.mean(yt, axis=0, keepdims=True)) ** 2, axis=0)
    r2_per_var = 1.0 - ss_res / (ss_tot + 1e-10)
    return float(np.mean(r2_per_var))


def _compute_aic_bic(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_active: int,
    *,
    noise_gain: float | None = None,
    state_vars: np.ndarray | None = None,
    n_indep: float | None = None,
    eps_floor: float = 1e-16,
) -> tuple[float, float]:
    """Compute adjoint-whitened Pareto AIC and BIC across multi-output targets."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.ndim == 1:
        yt = yt[:, np.newaxis]
        yp = yp[:, np.newaxis]
    n_samples, n_vars = yt.shape
    n_eff = float(max(4.0, n_indep)) if n_indep is not None else float(max(1, n_samples * n_vars))

    if noise_gain is not None and state_vars is not None:
        mse_per_var = np.mean((yt - yp) ** 2, axis=0)
        rel_var = float(np.mean(mse_per_var / (max(noise_gain, 1e-30) * state_vars)))
        loss_metric = max(rel_var, eps_floor)
    else:
        ss_res = np.sum((yt - yp) ** 2, axis=0)
        ss_tot = np.sum((yt - np.mean(yt, axis=0, keepdims=True)) ** 2, axis=0)
        loss_metric = max(float(np.mean(ss_res / (ss_tot + 1e-12))), eps_floor)

    k = int(n_active)
    aic = float(n_eff * np.log(loss_metric) + 2.0 * k)
    bic = float(n_eff * np.log(loss_metric) + np.log(n_eff) * k)
    return aic, bic


@dataclass
class _ProjectedSystemCache:
    """Pre-contracted spatial/feature weak projection cache for O(1) alpha sweeps."""

    use_weak: bool
    X_design: np.ndarray
    X_source: np.ndarray
    dt: float
    k_start: int
    k_end: int
    windows: list[tuple[int, int]] | None = None
    phi_t: np.ndarray | None = None
    X_spatial: np.ndarray | None = None
    spatial_noise_gain: float = 1.0
    overlap_factor: float = 1.0

    def target_and_gain(self, alpha: float) -> tuple[np.ndarray, float]:
        n_vars = self.X_source.shape[-1]
        if not self.use_weak:
            X_dot = fractional_derivative_gl(self.X_source, t=self.dt, alpha=alpha, axis=0)
            y_out = X_dot[self.k_start : self.k_end].reshape(-1, n_vars)
            return y_out, float(self.dt ** (-2.0 * float(alpha)))

        assert self.windows is not None and self.phi_t is not None
        _, Psi_t = build_weak_weights(
            n_steps=self.X_source.shape[0],
            windows=self.windows,
            alpha=alpha,
            phi=self.phi_t,
            dt=self.dt,
        )
        t_gain = float(np.mean(np.sum(Psi_t * Psi_t, axis=0)))
        total_gain = max(t_gain * float(self.spatial_noise_gain), 1e-30)
        if self.X_spatial is not None:
            proj = np.tensordot(Psi_t, self.X_spatial, axes=([0], [0]))
            return proj.reshape(-1, n_vars), total_gain
        if self.X_source.ndim == 2:
            return Psi_t.T @ self.X_source, total_gain
        proj = np.moveaxis(np.tensordot(self.X_source, Psi_t, axes=([0], [0])), -1, 0)
        return proj.reshape(-1, n_vars), total_gain

    def target(self, alpha: float) -> np.ndarray:
        y_out, _ = self.target_and_gain(alpha)
        return y_out


def _ridge_solve(X_sub: np.ndarray, Y_sub: np.ndarray, alpha: float) -> np.ndarray:
    """Solve ``argmin_B ||Y_sub - X_sub @ B||_F^2 + alpha ||B||_F^2`` via SVD without squaring ``cond(X)``."""
    reg = float(alpha)
    if reg <= 0.0:
        B, *_ = np.linalg.lstsq(X_sub, Y_sub, rcond=None)
        return B
    U, s, Vt = np.linalg.svd(X_sub, full_matrices=False)
    filt = s / (s * s + reg)
    return (Vt.T * filt) @ (U.T @ Y_sub)


class STLSQ(BaseEstimator, RegressorMixin):
    """Sequentially Thresholded Least Squares (STLSQ) sparse regression solver.

    Solves ``argmin_Xi ||Y - X @ Xi.T||_F^2 + alpha ||Xi||_F^2`` with iterative
    hard thresholding on coefficient entries below ``threshold``.
    """

    def __init__(
        self,
        threshold: float = 0.01,
        alpha: float = 1e-12,
        max_iter: int = 20,
        normalize_columns: bool = True,
        threshold_scaled: bool = False,
        unbias: bool = True,
    ) -> None:
        self.threshold = threshold
        self.alpha = alpha
        self.max_iter = max_iter
        self.normalize_columns = normalize_columns
        self.threshold_scaled = threshold_scaled
        self.unbias = unbias

    def fit(self, X: np.ndarray, y: np.ndarray) -> STLSQ:
        X_arr = np.asarray(X, dtype=np.float64)
        Y_arr = np.asarray(y, dtype=np.float64)
        was_1d = Y_arr.ndim == 1
        if was_1d:
            Y_arr = Y_arr[:, np.newaxis]
        if X_arr.ndim != 2 or Y_arr.ndim != 2 or X_arr.shape[0] != Y_arr.shape[0]:
            raise ValueError(
                f"Incompatible shapes for STLSQ: X={X_arr.shape}, y={Y_arr.shape}."
            )

        n_features = X_arr.shape[1]
        n_targets = Y_arr.shape[1]

        if self.normalize_columns:
            norms = np.linalg.norm(X_arr, axis=0)
            norms[norms < 1e-12] = 1.0
            X_work = X_arr / norms
        else:
            norms = np.ones(n_features, dtype=np.float64)
            X_work = X_arr

        base_beta = _ridge_solve(X_work, Y_arr, float(self.alpha))

        coefs = np.zeros((n_targets, n_features), dtype=np.float64)
        for d in range(n_targets):
            y_d = Y_arr[:, d]
            beta = base_beta[:, d].copy()
            active = np.ones(n_features, dtype=bool)

            for _ in range(int(self.max_iter)):
                phys = beta / norms
                magnitudes = np.abs(beta) if self.threshold_scaled else np.abs(phys)
                new_active = magnitudes >= float(self.threshold)
                if not np.any(new_active):
                    beta[:] = 0.0
                    active = new_active
                    break

                beta_sub = _ridge_solve(X_work[:, new_active], y_d, float(self.alpha))
                beta_new = np.zeros(n_features, dtype=np.float64)
                beta_new[new_active] = beta_sub

                if np.array_equal(new_active, active) and np.allclose(beta, beta_new, rtol=1e-8, atol=1e-10):
                    beta = beta_new
                    active = new_active
                    break
                beta = beta_new
                active = new_active

            if self.unbias and np.any(active):
                X_act = X_arr[:, active]
                coef_act, *_ = np.linalg.lstsq(X_act, Y_arr[:, d], rcond=None)
                if np.all(np.isfinite(coef_act)):
                    if not self.threshold_scaled:
                        small = np.abs(coef_act) < float(self.threshold)
                        if np.any(small):
                            coef_act[small] = 0.0
                            coefs[d, active] = coef_act
                            still_active = coefs[d] != 0.0
                            if np.any(still_active):
                                coef_act2, *_ = np.linalg.lstsq(
                                    X_arr[:, still_active], Y_arr[:, d], rcond=None
                                )
                                if np.all(np.isfinite(coef_act2)):
                                    coefs[d, still_active] = coef_act2
                        else:
                            coefs[d, active] = coef_act
                    else:
                        coefs[d, active] = coef_act
                else:
                    coefs[d] = beta / norms
            else:
                coefs[d] = beta / norms

        self.coef_ = coefs
        self.ind_ = coefs != 0.0
        self.n_features_in_ = n_features
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "coef_"):
            raise RuntimeError("STLSQ must be fitted before calling predict.")
        X_arr = np.asarray(X, dtype=np.float64)
        return X_arr @ self.coef_.T


class SR3(BaseEstimator, RegressorMixin):
    """Sparse Relaxed Regularized Regression (SR3) solver."""

    def __init__(
        self,
        threshold: float = 0.05,
        nu: float = 1.0,
        reg_type: Literal["l0", "l1"] = "l0",
        max_iter: int = 100,
        tol: float = 1e-6,
        normalize_columns: bool = True,
        unbias: bool = True,
    ) -> None:
        self.threshold = threshold
        self.nu = nu
        self.reg_type = reg_type
        self.max_iter = max_iter
        self.tol = tol
        self.normalize_columns = normalize_columns
        self.unbias = unbias

    def _prox(self, Z: np.ndarray, tau: float | np.ndarray) -> np.ndarray:
        if self.reg_type == "l0":
            out = Z.copy()
            out[np.abs(Z) < tau] = 0.0
            return out
        if self.reg_type == "l1":
            return np.sign(Z) * np.maximum(np.abs(Z) - tau, 0.0)
        raise ValueError(f"reg_type must be 'l0' or 'l1'; got {self.reg_type!r}.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> SR3:
        if self.nu <= 0.0:
            raise ValueError(f"SR3 relaxation parameter nu must be > 0; got {self.nu}.")
        X_arr = np.asarray(X, dtype=np.float64)
        Y_arr = np.asarray(y, dtype=np.float64)
        if Y_arr.ndim == 1:
            Y_arr = Y_arr[:, np.newaxis]
        if X_arr.ndim != 2 or Y_arr.ndim != 2 or X_arr.shape[0] != Y_arr.shape[0]:
            raise ValueError(
                f"Incompatible shapes for SR3: X={X_arr.shape}, y={Y_arr.shape}."
            )

        n_features = X_arr.shape[1]
        n_targets = Y_arr.shape[1]

        if self.normalize_columns:
            norms = np.linalg.norm(X_arr, axis=0)
            norms[norms < 1e-12] = 1.0
            X_work = X_arr / norms
            tau_vec = (float(self.threshold) * norms)[:, np.newaxis]
            inv_nu_vec = 1.0 / (float(self.nu) * (norms**2))
        else:
            norms = np.ones(n_features, dtype=np.float64)
            X_work = X_arr
            tau_vec = float(self.threshold)
            inv_nu_vec = np.full(n_features, 1.0 / float(self.nu), dtype=np.float64)

        XtX = X_work.T @ X_work
        Xty = X_work.T @ Y_arr
        lhs = XtX + np.diag(inv_nu_vec)

        Xi_init = np.linalg.solve(XtX + 1e-5 * np.eye(n_features, dtype=np.float64), Xty)
        W_T = self._prox(Xi_init, tau_vec)

        for _ in range(int(self.max_iter)):
            Xi_T = np.linalg.solve(lhs, Xty + inv_nu_vec[:, np.newaxis] * W_T)
            W_T_new = self._prox(Xi_T, tau_vec)
            denom = max(float(np.linalg.norm(W_T)), 1e-12)
            if float(np.linalg.norm(W_T_new - W_T)) / denom < float(self.tol):
                W_T = W_T_new
                break
            W_T = W_T_new

        coefs = (W_T / norms[:, np.newaxis]).T
        if self.unbias:
            for d in range(n_targets):
                active = coefs[d] != 0.0
                if np.any(active):
                    coef_act, *_ = np.linalg.lstsq(X_arr[:, active], Y_arr[:, d], rcond=None)
                    if np.all(np.isfinite(coef_act)):
                        small = np.abs(coef_act) < float(self.threshold)
                        coef_act[small] = 0.0
                        coefs[d] = 0.0
                        coefs[d, active] = coef_act
                        still_active = coefs[d] != 0.0
                        if np.any(small) and np.any(still_active):
                            coef_act2, *_ = np.linalg.lstsq(
                                X_arr[:, still_active], Y_arr[:, d], rcond=None
                            )
                            coefs[d, still_active] = coef_act2

        self.coef_ = coefs
        self.ind_ = coefs != 0.0
        self.n_features_in_ = n_features
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "coef_"):
            raise RuntimeError("SR3 must be fitted before calling predict.")
        X_arr = np.asarray(X, dtype=np.float64)
        return X_arr @ self.coef_.T


class WeakSINDy(BaseEstimator):
    """Weak-form sparse identification of integer- and fractional-order dynamics and N-D PDEs.

    Transfers causal Grünwald–Letnikov fractional temporal derivatives ``D_t^alpha`` and
    spatial PDE operators onto separable compactly supported ``C^p`` test functions.
    Supports both discrete fractional-order candidate sweeps and continuous bounded
    Brent refinement (``optimize_order=True``) via Pareto AIC/BIC model selection.
    """

    def __init__(
        self,
        library: BaseCandidateLibrary | None = None,
        optimizer: Literal["stlsq", "sr3"] | STLSQ | SR3 = "stlsq",
        threshold: float = 0.01,
        alpha_reg: float = 1e-12,
        orders: float | Sequence[float] = (1.0,),
        optimize_order: bool = False,
        order_bounds: tuple[float, float] = (0.1, 1.0),
        criterion: Literal["bic", "aic"] = "bic",
        use_weak: bool = True,
        window_size: int = 50,
        stride: int = 5,
        spatial_window_size: int | Sequence[int] | None = None,
        spatial_stride: int | Sequence[int] | None = None,
        test_degree: int = 4,
        k_start: int = 20,
        val_split: float = 0.2,
        dx: float | np.ndarray | Sequence[float | np.ndarray] | None = None,
        refit: bool = True,
    ) -> None:
        self.library = library
        self.optimizer = optimizer
        self.threshold = threshold
        self.alpha_reg = alpha_reg
        self.orders = orders
        self.optimize_order = optimize_order
        self.order_bounds = order_bounds
        self.criterion = criterion
        self.use_weak = use_weak
        self.window_size = window_size
        self.stride = stride
        self.spatial_window_size = spatial_window_size
        self.spatial_stride = spatial_stride
        self.test_degree = test_degree
        self.k_start = k_start
        self.val_split = val_split
        self.dx = dx
        self.refit = refit

    @property
    def alpha_(self) -> float:
        """Alias for the recovered fractional temporal order ``order_``."""
        return self.order_

    def _make_optimizer(self) -> STLSQ | SR3:
        if isinstance(self.optimizer, (STLSQ, SR3)):
            params = self.optimizer.get_params()
            return self.optimizer.__class__(**params)
        if self.optimizer == "stlsq":
            return STLSQ(threshold=self.threshold, alpha=self.alpha_reg)
        if self.optimizer == "sr3":
            return SR3(threshold=self.threshold)
        raise ValueError(f"Unsupported optimizer {self.optimizer!r}; expected 'stlsq' or 'sr3'.")

    def _make_library(
        self,
        X: np.ndarray,
        dx: float | np.ndarray | Sequence[float | np.ndarray] | None,
    ) -> BaseCandidateLibrary:
        if self.library is not None:
            return self.library
        if X.ndim >= 3:
            eff_dx = dx if dx is not None else self.dx
            if eff_dx is None:
                raise ValueError(
                    "Spatiotemporal PDE input (ndim >= 3) requires dx (pass dx=... or supply a PDELibrary)."
                )
            return PDELibrary(dx=eff_dx, poly_degree=2, include_interaction=False)
        return PolynomialLibrary(degree=3, include_bias=True, include_interaction=True)

    def _resolve_window_params(self, span: int) -> tuple[int, int]:
        H = int(self.window_size)
        S = int(self.stride)
        if span <= H + 2:
            H = max(4, (span - 2) // 2)
            S = max(1, min(S, H // 4))
        return H, S

    def _build_cache(
        self,
        X_src: np.ndarray,
        lib: BaseCandidateLibrary,
        dt: float,
        k_start: int,
        k_end: int,
        dx: float | np.ndarray | Sequence[float | np.ndarray] | None = None,
    ) -> _ProjectedSystemCache:
        eff_dx = dx if dx is not None else self.dx
        if not self.use_weak:
            Theta_eval = lib.transform(X_src)
            X_des = Theta_eval[k_start:k_end].reshape(-1, Theta_eval.shape[-1])
            return _ProjectedSystemCache(
                use_weak=False,
                X_design=X_des,
                X_source=X_src,
                dt=dt,
                k_start=k_start,
                k_end=k_end,
            )

        H, S = self._resolve_window_params(k_end - k_start)
        wins = generate_windows(
            n_steps=X_src.shape[0],
            window_size=H,
            stride=S,
            k_start=k_start,
            k_end=k_end,
        )
        phi_t = bump_test_function(window_size=H, degree=self.test_degree, dt=dt, normalize=True)
        Phi_t, _ = build_weak_weights(
            n_steps=X_src.shape[0],
            windows=wins,
            alpha=0.0,
            phi=phi_t,
            dt=dt,
        )

        spatial_ndim = X_src.ndim - 2
        use_spatiotemporal = spatial_ndim >= 1 and (
            getattr(lib, "_supports_spatiotemporal_weak", False)
            or eff_dx is not None
            or self.spatial_window_size is not None
        )

        if use_spatiotemporal:
            sp_shape = X_src.shape[1:-1]
            sp_wins = resolve_spatial_windows(
                sp_shape,
                spatial_window_size=self.spatial_window_size,
                spatial_stride=self.spatial_stride,
            )
            if eff_dx is not None:
                sp_steps = extract_spatial_steps(eff_dx, spatial_ndim)
            elif hasattr(lib, "_get_spatial_steps"):
                sp_steps = lib._get_spatial_steps(spatial_ndim)
            else:
                sp_steps = (1.0,) * spatial_ndim

            Phi_x0 = [
                build_spatial_weak_weights(
                    n_grid=sp_shape[d],
                    windows=sp_wins[d],
                    dx=sp_steps[d],
                    degree=self.test_degree,
                    diff_order=0,
                )
                for d in range(spatial_ndim)
            ]

            if hasattr(lib, "_project_spatiotemporal_weak"):
                X_des = lib._project_spatiotemporal_weak(
                    X_src,
                    Phi_t=Phi_t,
                    Phi_x0=Phi_x0,
                    sp_wins=sp_wins,
                    sp_steps=sp_steps,
                    sp_degree=self.test_degree,
                )
            else:
                Theta_eval = lib.transform(X_src)
                X_des = contract_tensor_weak(Theta_eval, [Phi_t, *Phi_x0])

            # Pre-contract X_src along spatial axes 1..D so temporal Psi_t(alpha) applies in O(nt * M_sp)
            curr_sp = X_src
            for d in range(spatial_ndim):
                c = np.tensordot(curr_sp, Phi_x0[d], axes=([1 + d], [0]))
                curr_sp = np.moveaxis(c, -1, 1 + d)
            X_sp = curr_sp.reshape(X_src.shape[0], -1, X_src.shape[-1])

            sp_gain = float(
                np.prod([float(np.mean(np.sum(w * w, axis=0))) for w in Phi_x0])
            )
            sp_overlap = 1.0
            for d_idx, wins_d in enumerate(sp_wins):
                if len(wins_d) >= 2:
                    h_d = max(1, wins_d[0][1] - wins_d[0][0])
                    s_d = max(1, wins_d[1][0] - wins_d[0][0])
                    sp_overlap *= max(1.0, h_d / s_d)
                else:
                    sp_overlap *= 1.0
            t_overlap = max(1.0, H / max(1, S)) if len(wins) >= 2 else 1.0

            return _ProjectedSystemCache(
                use_weak=True,
                X_design=X_des,
                X_source=X_src,
                dt=dt,
                k_start=k_start,
                k_end=k_end,
                windows=wins,
                phi_t=phi_t,
                X_spatial=X_sp,
                spatial_noise_gain=sp_gain,
                overlap_factor=t_overlap * sp_overlap,
            )

        Theta_eval = lib.transform(X_src)
        if X_src.ndim == 2:
            X_des = Phi_t.T @ Theta_eval
        else:
            t_proj = np.moveaxis(np.tensordot(Theta_eval, Phi_t, axes=([0], [0])), -1, 0)
            X_des = t_proj.reshape(-1, Theta_eval.shape[-1])

        t_overlap = max(1.0, H / max(1, S)) if len(wins) >= 2 else 1.0
        return _ProjectedSystemCache(
            use_weak=True,
            X_design=X_des,
            X_source=X_src,
            dt=dt,
            k_start=k_start,
            k_end=k_end,
            windows=wins,
            phi_t=phi_t,
            X_spatial=None,
            spatial_noise_gain=1.0,
            overlap_factor=t_overlap,
        )

    def _assemble_system(
        self,
        X: np.ndarray,
        Theta: np.ndarray | None,
        dt: float,
        alpha: float,
        k_start: int,
        k_end: int,
        library: BaseCandidateLibrary | None = None,
        dx: float | np.ndarray | Sequence[float | np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        eff_dx = dx if dx is not None else self.dx
        if self.use_weak:
            H, S = self._resolve_window_params(k_end - k_start)
            wins = generate_windows(
                n_steps=X.shape[0],
                window_size=H,
                stride=S,
                k_start=k_start,
                k_end=k_end,
            )
            return project_weak_form(
                X,
                Theta=Theta,
                t=dt,
                alpha=alpha,
                dx=eff_dx,
                library=library,
                degree=self.test_degree,
                spatial_window_size=self.spatial_window_size,
                spatial_stride=self.spatial_stride,
                windows=wins,
            )

        Theta_arr = library.transform(X) if (Theta is None and library is not None) else Theta
        assert Theta_arr is not None
        X_dot = fractional_derivative_gl(X, t=dt, alpha=alpha, axis=0)
        Theta_slice = Theta_arr[k_start:k_end].reshape(-1, Theta_arr.shape[-1])
        Y_slice = X_dot[k_start:k_end].reshape(-1, X.shape[-1])
        return Theta_slice, Y_slice

    def fit(
        self,
        t: float | np.ndarray,
        X: np.ndarray,
        orders: float | Sequence[float] | None = None,
        *,
        optimize_order: bool | None = None,
        order_bounds: tuple[float, float] | None = None,
        criterion: Literal["bic", "aic"] | None = None,
        dx: float | np.ndarray | Sequence[float | np.ndarray] | None = None,
        X_clean: np.ndarray | None = None,
        true_order: float | None = None,
        input_features: Sequence[str] | None = None,
    ) -> WeakSINDy:
        """Fit sparse governing equations and identify discrete or continuous fractional order.

        Parameters
        ----------
        t : float or 1D np.ndarray
            Uniform time step ``dt`` or 1D time coordinate array of length ``nt``.
        X : np.ndarray
            Observed state trajectory of shape ``(nt, n_vars)`` (ODE) or
            ``(nt, *spatial_dims, n_vars)`` (1D, 2D, or 3D spatial PDE).
        orders : float or sequence of float or None, default=None
            Discrete candidate fractional temporal orders ``alpha``.
        optimize_order : bool or None, default=None
            If ``True``, performs bounded continuous Brent refinement via
            ``scipy.optimize.minimize_scalar`` using Pareto AIC/BIC model selection to
            pinpoint ``order_`` to ``~1e-3`` precision. Defaults to ``self.optimize_order``.
        order_bounds : tuple of (float, float) or None, default=None
            Search interval ``(alpha_min, alpha_max)`` when ``optimize_order=True``.
            Defaults to ``self.order_bounds`` ``(0.1, 1.0)``.
        criterion : {"bic", "aic"} or None, default=None
            Pareto information criterion minimized during continuous order optimization.
            Defaults to ``self.criterion`` (``"bic"``).
        dx : float or sequence of float/1D arrays or None, default=None
            Optional spatial grid spacing(s) for N-D PDE data.
        X_clean : np.ndarray or None, default=None
            Optional noise-free reference trajectory for oracle benchmark validation.
        true_order : float or None, default=None
            Optional true fractional order used with ``X_clean`` for oracle scoring.
        input_features : sequence of str or None, default=None
            Optional variable names (e.g. ``["x", "y", "z"]`` or ``["p", "c", "n"]``).
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, np.newaxis]
        if X_arr.ndim < 2:
            raise ValueError(f"Expected X to have at least 2 dimensions; got shape {X_arr.shape}.")

        nt = X_arr.shape[0]
        dt = extract_time_step(t)
        if np.ndim(t) == 1 and np.asarray(t).size != nt:
            raise ValueError(
                f"Length of time vector t ({np.asarray(t).size}) must match X.shape[0] ({nt})."
            )

        do_optimize = self.optimize_order if optimize_order is None else bool(optimize_order)
        bounds = self.order_bounds if order_bounds is None else (float(order_bounds[0]), float(order_bounds[1]))
        sel_criterion = self.criterion if criterion is None else criterion
        if sel_criterion not in ("bic", "aic"):
            raise ValueError(f"criterion must be 'bic' or 'aic'; got {sel_criterion!r}.")

        raw_orders = self.orders if orders is None else orders
        if np.ndim(raw_orders) == 0:
            cand_orders = [float(raw_orders)]  # type: ignore[arg-type]
        else:
            cand_orders = [float(a) for a in raw_orders]  # type: ignore[union-attr]
        if not do_optimize and len(cand_orders) == 0:
            raise ValueError("Candidate orders list must be non-empty.")

        eff_k_start = min(int(self.k_start), max(0, nt // 4))

        lib = self._make_library(X_arr, dx=dx)
        lib.fit(X_arr)

        X_val_src = X_arr
        if X_clean is not None:
            X_val_src = np.asarray(X_clean, dtype=np.float64)
            if X_val_src.ndim == 1:
                X_val_src = X_val_src[:, np.newaxis]

        order_scores: dict[float, float] = {}
        profile_records: dict[float, dict[str, float | int]] = {}

        if do_optimize:
            a_min, a_max = float(bounds[0]), float(bounds[1])
            if not (np.isfinite(a_min) and np.isfinite(a_max) and 0.0 < a_min < a_max):
                raise ValueError(
                    f"order_bounds must satisfy 0 < alpha_min < alpha_max; got {bounds}."
                )

            full_cache = self._build_cache(
                X_arr, lib=lib, dt=dt, k_start=eff_k_start, k_end=nt, dx=dx
            )

            n_sub = full_cache.X_design.shape[0]
            n_vars_tot = X_arr.shape[-1]
            state_vars = np.array(
                [max(float(np.var(X_arr[..., d])), 1e-12) for d in range(n_vars_tot)],
                dtype=np.float64,
            )
            n_indep = max(4.0, (n_sub / max(1.0, full_cache.overlap_factor)) * n_vars_tot)

            oracle_val_cache: _ProjectedSystemCache | None = None
            oracle_tr_cache: _ProjectedSystemCache | None = None
            if X_clean is not None and true_order is not None:
                nt_train = int((1.0 - float(self.val_split)) * nt)
                oracle_tr_cache = self._build_cache(
                    X_arr, lib=lib, dt=dt, k_start=eff_k_start, k_end=nt_train, dx=dx
                )
                oracle_val_cache = self._build_cache(
                    X_val_src, lib=lib, dt=dt, k_start=nt_train, k_end=nt, dx=dx
                )
                Y_oracle_val, oracle_gain = oracle_val_cache.target_and_gain(float(true_order))

            def evaluate_alpha(alpha_val: float) -> float:
                key = round(float(alpha_val), 6)
                if key in profile_records:
                    return float(profile_records[key][sel_criterion])

                if oracle_tr_cache is not None and oracle_val_cache is not None:
                    Y_tr_o = oracle_tr_cache.target(key)
                    opt_o = self._make_optimizer().fit(oracle_tr_cache.X_design, Y_tr_o)
                    Y_pred_o = opt_o.predict(oracle_val_cache.X_design)
                    r2_v = _mean_r2_score(Y_oracle_val, Y_pred_o)
                    n_act = int(np.sum(opt_o.coef_ != 0.0))
                    aic_v, bic_v = _compute_aic_bic(
                        Y_oracle_val,
                        Y_pred_o,
                        n_act,
                        noise_gain=oracle_gain,
                        state_vars=state_vars,
                        n_indep=n_indep,
                    )
                else:
                    Y_all, g_alpha = full_cache.target_and_gain(key)
                    X_all = full_cache.X_design
                    opt_full = self._make_optimizer().fit(X_all, Y_all)
                    n_act = int(np.sum(opt_full.coef_ != 0.0))
                    Y_hat = opt_full.predict(X_all)
                    r2_v = _mean_r2_score(Y_all, Y_hat)
                    if n_act == 0:
                        aic_v, bic_v = 1e12, 1e12
                    else:
                        aic_v, bic_v = _compute_aic_bic(
                            Y_all,
                            Y_hat,
                            n_act,
                            noise_gain=g_alpha,
                            state_vars=state_vars,
                            n_indep=n_indep,
                        )

                order_scores[key] = r2_v
                profile_records[key] = {
                    "r2": r2_v,
                    "aic": aic_v,
                    "bic": bic_v,
                    "n_active": n_act,
                }
                return float(bic_v if sel_criterion == "bic" else aic_v)

            # 1. Global coarse bracket scan across [a_min, a_max] + user candidate orders in bounds
            coarse_grid = list(np.linspace(a_min, a_max, 25))
            if orders is not None:
                coarse_grid.extend(a for a in cand_orders if a_min <= a <= a_max)
            coarse_unique = sorted({round(float(a), 6) for a in coarse_grid})
            coarse_vals = [evaluate_alpha(a) for a in coarse_unique]

            seed_indices = {int(np.argmin(coarse_vals))}
            positive_acts = [
                int(profile_records[a]["n_active"])
                for a in coarse_unique
                if int(profile_records[a]["n_active"]) > 0
            ]
            if positive_acts:
                min_act = min(positive_acts)
                sparse_idxs = [
                    idx
                    for idx, a in enumerate(coarse_unique)
                    if int(profile_records[a]["n_active"]) == min_act
                ]
                seed_indices.add(min(sparse_idxs, key=lambda idx: coarse_vals[idx]))

            # 2. Continuous golden-section / Brent refinement to < 1e-3 precision
            h_step = (a_max - a_min) / 24.0
            for s_idx in sorted(seed_indices):
                a_seed = coarse_unique[s_idx]
                b_lo = max(a_min, a_seed - 1.5 * h_step)
                b_hi = min(a_max, a_seed + 1.5 * h_step)
                res = minimize_scalar(
                    evaluate_alpha,
                    bounds=(b_lo, b_hi),
                    method="bounded",
                    options={"xatol": 1e-4, "maxiter": 50},
                )
                _ = evaluate_alpha(float(res.x))

            # Select overall optimum across all bracket + Brent points (including exact endpoints)
            best_order = min(
                profile_records.keys(),
                key=lambda k: (float(profile_records[k][sel_criterion]), -float(profile_records[k]["r2"])),
            )

            Y_final, g_final = full_cache.target_and_gain(best_order)
            opt = self._make_optimizer().fit(full_cache.X_design, Y_final)
            Y_final_pred = opt.predict(full_cache.X_design)
            best_r2 = float(profile_records[best_order]["r2"])
            final_aic, final_bic = _compute_aic_bic(
                Y_final,
                Y_final_pred,
                int(np.sum(opt.coef_ != 0.0)),
                noise_gain=g_final,
                state_vars=state_vars,
                n_indep=n_indep,
            )

        elif len(cand_orders) == 1:
            best_order = cand_orders[0]
            full_cache = self._build_cache(
                X_arr, lib=lib, dt=dt, k_start=eff_k_start, k_end=nt, dx=dx
            )
            Y_target = full_cache.target(best_order)
            opt = self._make_optimizer().fit(full_cache.X_design, Y_target)
            y_pred = opt.predict(full_cache.X_design)
            best_r2 = _mean_r2_score(Y_target, y_pred)
            n_act = int(np.sum(opt.coef_ != 0.0))
            final_aic, final_bic = _compute_aic_bic(Y_target, y_pred, n_act)
            order_scores[best_order] = best_r2
            profile_records[best_order] = {
                "r2": best_r2,
                "aic": final_aic,
                "bic": final_bic,
                "n_active": n_act,
            }
        else:
            nt_train = int((1.0 - float(self.val_split)) * nt)
            if nt_train - eff_k_start < 8 or nt - nt_train < 6:
                raise ValueError(
                    f"Trajectory length nt={nt} is too short for order selection split "
                    f"(k_start={eff_k_start}, val_split={self.val_split})."
                )

            tr_cache = self._build_cache(
                X_arr, lib=lib, dt=dt, k_start=eff_k_start, k_end=nt_train, dx=dx
            )
            val_cache = self._build_cache(
                X_val_src, lib=lib, dt=dt, k_start=nt_train, k_end=nt, dx=dx
            )

            best_order = cand_orders[0]
            best_r2 = -np.inf
            best_opt: STLSQ | SR3 | None = None

            for alpha in cand_orders:
                Y_tr = tr_cache.target(alpha)
                val_alpha = (
                    float(true_order) if (X_clean is not None and true_order is not None) else alpha
                )
                Y_val = val_cache.target(val_alpha)
                opt_cand = self._make_optimizer().fit(tr_cache.X_design, Y_tr)
                Y_pred = opt_cand.predict(val_cache.X_design)
                r2 = _mean_r2_score(Y_val, Y_pred)
                n_act = int(np.sum(opt_cand.coef_ != 0.0))
                aic_val, bic_val = _compute_aic_bic(Y_val, Y_pred, n_act)
                order_scores[alpha] = r2
                profile_records[alpha] = {
                    "r2": r2,
                    "aic": aic_val,
                    "bic": bic_val,
                    "n_active": n_act,
                }
                if r2 > best_r2:
                    best_r2 = r2
                    best_order = alpha
                    best_opt = opt_cand

            if self.refit or best_opt is None:
                full_cache = self._build_cache(
                    X_arr, lib=lib, dt=dt, k_start=eff_k_start, k_end=nt, dx=dx
                )
                Y_full = full_cache.target(best_order)
                opt = self._make_optimizer().fit(full_cache.X_design, Y_full)
                Y_full_pred = opt.predict(full_cache.X_design)
                final_aic, final_bic = _compute_aic_bic(
                    Y_full, Y_full_pred, int(np.sum(opt.coef_ != 0.0))
                )
            else:
                opt = best_opt
                final_aic = float(profile_records[best_order]["aic"])
                final_bic = float(profile_records[best_order]["bic"])

        sorted_alphas = sorted(profile_records.keys())
        self.order_profile_ = {
            "orders": [float(a) for a in sorted_alphas],
            "r2": [float(profile_records[a]["r2"]) for a in sorted_alphas],
            "aic": [float(profile_records[a]["aic"]) for a in sorted_alphas],
            "bic": [float(profile_records[a]["bic"]) for a in sorted_alphas],
            "n_active": [int(profile_records[a]["n_active"]) for a in sorted_alphas],
        }

        self.dt_ = dt
        self.dx_ = dx if dx is not None else self.dx
        self.library_ = lib
        self.optimizer_ = opt
        self.coef_ = opt.coef_
        self.order_ = float(best_order)
        self.order_scores_ = order_scores
        self.r2_score_ = float(best_r2)
        self.aic_ = float(final_aic)
        self.bic_ = float(final_bic)
        self.n_features_in_ = X_arr.shape[-1]
        self.input_features_ = lib._default_feature_names(self.n_features_in_, input_features)
        self.feature_names_ = lib.get_feature_names(self.input_features_)
        return self

    def predict(self, t: float | np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
        """Predict fractional time derivatives ``D_t^(order_) X = Theta(X) @ coef_.T``."""
        if not hasattr(self, "coef_"):
            raise RuntimeError("WeakSINDy must be fitted before calling predict.")

        if X is None:
            X_arr = np.asarray(t, dtype=np.float64)
        else:
            _ = extract_time_step(t)
            X_arr = np.asarray(X, dtype=np.float64)

        was_1d = X_arr.ndim == 1
        if was_1d:
            X_arr = X_arr[:, np.newaxis]

        Theta = self.library_.transform(X_arr)
        lead_shape = Theta.shape[:-1]
        Theta_flat = Theta.reshape(-1, Theta.shape[-1])
        pred_flat = Theta_flat @ self.coef_.T
        pred = pred_flat.reshape((*lead_shape, self.coef_.shape[0]))
        return pred[:, 0] if was_1d else pred

    def score(
        self,
        t: float | np.ndarray,
        X: np.ndarray,
        y_true: np.ndarray | None = None,
    ) -> float:
        """Compute mean ``R^2`` score of the fitted model on ``(t, X)``."""
        if not hasattr(self, "coef_"):
            raise RuntimeError("WeakSINDy must be fitted before calling score.")
        dt = extract_time_step(t)
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, np.newaxis]

        if y_true is not None:
            yt = np.asarray(y_true, dtype=np.float64).reshape(-1, self.coef_.shape[0])
            yp = self.predict(dt, X_arr).reshape(-1, self.coef_.shape[0])
            return _mean_r2_score(yt, yp)

        eff_k = min(int(self.k_start), max(0, X_arr.shape[0] // 4))
        cache = self._build_cache(
            X_arr, lib=self.library_, dt=dt, k_start=eff_k, k_end=X_arr.shape[0], dx=self.dx_
        )
        Y_tar = cache.target(self.order_)
        return _mean_r2_score(Y_tar, cache.X_design @ self.coef_.T)

    def equations(
        self,
        precision: int = 4,
        input_features: Sequence[str] | None = None,
    ) -> list[str]:
        """Format discovered sparse fractional differential equations as strings."""
        if not hasattr(self, "coef_"):
            raise RuntimeError("WeakSINDy must be fitted before calling equations.")
        state_names = (
            self.library_._default_feature_names(self.n_features_in_, input_features)
            if input_features is not None
            else self.input_features_
        )
        feat_names = self.library_.get_feature_names(state_names)

        if np.isclose(self.order_, 1.0, atol=5e-4):
            lhs_op = "d/dt"
        elif abs(round(self.order_, 2) - round(self.order_, 3)) < 1e-6:
            lhs_op = f"D_t^{self.order_:.2f}"
        else:
            lhs_op = f"D_t^{self.order_:.3f}"

        eqs: list[str] = []
        for d, s_name in enumerate(state_names):
            terms: list[str] = []
            for c_val, f_name in zip(self.coef_[d], feat_names):
                if c_val == 0.0:
                    continue
                sign_str = "-" if c_val < 0.0 else "+"
                mag_str = f"{abs(c_val):.{precision}f}"
                term_body = mag_str if f_name == "1" else f"{mag_str} {f_name}"
                if not terms:
                    terms.append(f"-{term_body}" if c_val < 0.0 else term_body)
                else:
                    terms.append(f"{sign_str} {term_body}")
            rhs_str = " ".join(terms) if terms else "0"
            eqs.append(f"{lhs_op} {s_name} = {rhs_str}")
        return eqs

    def print(
        self,
        precision: int = 4,
        input_features: Sequence[str] | None = None,
    ) -> None:
        """Print discovered governing equations to stdout."""
        for line in self.equations(precision=precision, input_features=input_features):
            print(line)
