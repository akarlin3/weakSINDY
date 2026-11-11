"""Sparse regression solvers (STLSQ, SR3) and high-level WeakSINDy estimator."""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from .library import BaseCandidateLibrary, PDELibrary, PolynomialLibrary
from .operators import extract_time_step, fractional_derivative_gl
from .test_functions import generate_windows, project_weak_form


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


class STLSQ(BaseEstimator, RegressorMixin):
    """Sequentially Thresholded Least Squares (STLSQ) sparse regression solver.

    Solves ``argmin_Xi ||Y - X @ Xi.T||_F^2 + alpha ||Xi||_F^2`` with iterative
    hard thresholding on coefficient entries below ``threshold``.

    Parameters
    ----------
    threshold : float, default=0.01
        Sparsity hard-thresholding cutoff.
    alpha : float, default=1e-4
        Tikhonov / ridge regularization penalty ``eta_ridge >= 0`` for ill-conditioned
        candidate Gram matrices.
    max_iter : int, default=20
        Maximum number of sequential thresholding + debiasing iterations.
    normalize_columns : bool, default=True
        Normalize columns of ``X`` to unit l2 norm prior to normal-equation solves.
    threshold_scaled : bool, default=False
        If ``False`` (default), applies ``|Xi_{d,j}| < threshold`` in physical units
        after rescaling by column norms. If ``True``, applies thresholding directly
        on column-norm-scaled regression weights.
    unbias : bool, default=True
        Perform a final unpenalized (or ridge-stabilized) least-squares polish on the
        converged active support.
    """

    def __init__(
        self,
        threshold: float = 0.01,
        alpha: float = 1e-4,
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

        XtX = X_work.T @ X_work
        Xty_all = X_work.T @ Y_arr
        reg_eye = float(self.alpha) * np.eye(n_features, dtype=np.float64)
        base_beta = np.linalg.solve(XtX + reg_eye, Xty_all)

        coefs = np.zeros((n_targets, n_features), dtype=np.float64)
        for d in range(n_targets):
            Xty = Xty_all[:, d]
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

                X_sub = XtX[np.ix_(new_active, new_active)]
                Xty_sub = Xty[new_active]
                k_act = int(np.sum(new_active))
                beta_sub = np.linalg.solve(
                    X_sub + float(self.alpha) * np.eye(k_act, dtype=np.float64),
                    Xty_sub,
                )
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
                # Fall back to ridge-regularized estimate if active columns are nearly singular
                if np.all(np.isfinite(coef_act)):
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
    """Sparse Relaxed Regularized Regression (SR3) solver.

    Minimizes the relaxed splitting objective (Zheng et al., 2019; Champion et al., 2020):

    .. math::

        \\min_{\\Xi, W} \\frac{1}{2} \\|Y - X \\Xi^\\top\\|_F^2
        + \\frac{1}{2\\nu} \\|\\Xi - W\\|_F^2 + \\lambda R(W)

    via alternating least-squares and proximal hard (``l0``) or soft (``l1``) thresholding,
    followed by an active-support debiasing polish.

    Parameters
    ----------
    threshold : float, default=0.05
        Proximal thresholding cutoff ``tau`` applied to auxiliary weights ``W``.
    nu : float, default=1.0
        Relaxation coupling parameter ``nu > 0`` controlling ``||Xi - W||_F^2 / (2 nu)``.
    reg_type : {"l0", "l1"}, default="l0"
        Regularizer family: ``"l0"`` for hard thresholding or ``"l1"`` for soft thresholding.
    max_iter : int, default=100
        Maximum alternating minimization iterations.
    tol : float, default=1e-6
        Relative convergence tolerance on auxiliary sparse weights ``W``.
    normalize_columns : bool, default=True
        Normalize columns of ``X`` during the alternating solve for conditioning.
    unbias : bool, default=True
        Refit least squares restricted to the converged nonzero support of ``W``.
    """

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

        # Warm-start auxiliary sparse weights W from weakly regularized ridge estimate
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

        coefs = (W_T / norms[:, np.newaxis]).T  # (n_targets, n_features)
        if self.unbias:
            for d in range(n_targets):
                active = coefs[d] != 0.0
                if np.any(active):
                    coef_act, *_ = np.linalg.lstsq(X_arr[:, active], Y_arr[:, d], rcond=None)
                    if np.all(np.isfinite(coef_act)):
                        # Re-prune any terms that dropped below threshold after debiasing
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
    """Weak-form sparse identification of integer- and fractional-order dynamics and PDEs.

    Transfers the causal Grünwald–Letnikov fractional operator ``D_t^alpha`` onto
    compactly supported ``C^p`` polynomial test functions to suppress the high-frequency
    pointwise variance amplification ``A(alpha) = dt^(-2 alpha) ||w^(alpha)||_2^2``.
    Supports joint recovery of unknown fractional temporal orders ``alpha in (0, 1]``
    and sparse nonlinear ODE/PDE governing equations.

    Parameters
    ----------
    library : BaseCandidateLibrary or None, default=None
        Candidate feature library. Defaults to ``PolynomialLibrary(degree=3)`` for 2D ODE
        trajectories or ``PDELibrary(dx=dx, poly_degree=2)`` for 3D spatiotemporal fields.
    optimizer : {"stlsq", "sr3"} or STLSQ or SR3, default="stlsq"
        Sparse regression solver or solver name.
    threshold : float, default=0.01
        Sparsity threshold passed when instantiating the default ``optimizer``.
    alpha_reg : float, default=1e-4
        Ridge penalty (for ``STLSQ``) or relaxation parameter ``nu`` when custom solver
        is not explicitly supplied.
    orders : float or sequence of float, default=(1.0,)
        Default candidate fractional differentiation orders ``alpha`` if not passed to ``fit``.
    use_weak : bool, default=True
        If ``True``, uses weak-form ``C^p`` test-function projection. If ``False``, uses
        pointwise Grünwald–Letnikov differentiation (useful for baseline benchmarking).
    window_size : int, default=50
        Weak-form window width ``H`` in time steps.
    stride : int, default=5
        Stride ``S`` between adjacent weak-form integration windows.
    test_degree : int, default=4
        Smoothness exponent ``p`` of the ``C^p`` bump kernel ``(1 - xi^2)^p``.
    k_start : int, default=20
        Initial time steps discarded from regression to avoid ``t = 0`` GL startup transients.
    val_split : float, default=0.2
        Fraction of terminal time steps reserved for held-out ``R^2`` order selection when
        sweeping multiple candidate ``orders``.
    dx : float or None, default=None
        Default spatial grid spacing ``dx`` for 3D PDE inputs when ``library=None``.
    refit : bool, default=True
        After selecting ``order_`` on the validation split, refit sparse coefficients across
        the full post-transient trajectory ``[k_start, nt)``.
    """

    def __init__(
        self,
        library: BaseCandidateLibrary | None = None,
        optimizer: Literal["stlsq", "sr3"] | STLSQ | SR3 = "stlsq",
        threshold: float = 0.01,
        alpha_reg: float = 1e-4,
        orders: float | Sequence[float] = (1.0,),
        use_weak: bool = True,
        window_size: int = 50,
        stride: int = 5,
        test_degree: int = 4,
        k_start: int = 20,
        val_split: float = 0.2,
        dx: float | None = None,
        refit: bool = True,
    ) -> None:
        self.library = library
        self.optimizer = optimizer
        self.threshold = threshold
        self.alpha_reg = alpha_reg
        self.orders = orders
        self.use_weak = use_weak
        self.window_size = window_size
        self.stride = stride
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

    def _make_library(self, X: np.ndarray, dx: float | None) -> BaseCandidateLibrary:
        if self.library is not None:
            return self.library
        if X.ndim == 3:
            eff_dx = dx if dx is not None else self.dx
            if eff_dx is None:
                raise ValueError("3D PDE input requires dx (pass dx=... or supply a PDELibrary).")
            return PDELibrary(dx=eff_dx, poly_degree=2, include_interaction=False)
        return PolynomialLibrary(degree=3, include_bias=True, include_interaction=True)

    def _resolve_window_params(self, span: int) -> tuple[int, int]:
        H = int(self.window_size)
        S = int(self.stride)
        if span <= H + 2:
            H = max(4, (span - 2) // 2)
            S = max(1, min(S, H // 4))
        return H, S

    def _assemble_system(
        self,
        X: np.ndarray,
        Theta: np.ndarray,
        dt: float,
        alpha: float,
        k_start: int,
        k_end: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_feat = Theta.shape[-1]
        n_vars = X.shape[-1]
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
                degree=self.test_degree,
                windows=wins,
            )

        X_dot = fractional_derivative_gl(X, t=dt, alpha=alpha, axis=0)
        Theta_slice = Theta[k_start:k_end].reshape(-1, n_feat)
        Y_slice = X_dot[k_start:k_end].reshape(-1, n_vars)
        return Theta_slice, Y_slice

    def fit(
        self,
        t: float | np.ndarray,
        X: np.ndarray,
        orders: float | Sequence[float] | None = None,
        *,
        dx: float | None = None,
        X_clean: np.ndarray | None = None,
        true_order: float | None = None,
        input_features: Sequence[str] | None = None,
    ) -> WeakSINDy:
        """Fit sparse governing equations and select fractional order ``order_``.

        Parameters
        ----------
        t : float or 1D np.ndarray
            Uniform time step ``dt`` or 1D time coordinate array of length ``nt``.
        X : np.ndarray
            Observed state trajectory of shape ``(nt, n_vars)`` (ODE) or
            ``(nt, nx, n_vars)`` (1D spatial PDE).
        orders : float or sequence of float or None, default=None
            Candidate fractional temporal orders ``alpha`` to evaluate. Defaults to
            ``self.orders``.
        dx : float or None, default=None
            Optional spatial step size for 3D PDE data when ``self.library is None``.
        X_clean : np.ndarray or None, default=None
            Optional noise-free reference trajectory used solely for oracle benchmark
            scoring on the validation split (defaults to ``None`` for deployment-realistic
            noisy validation).
        true_order : float or None, default=None
            Optional true fractional order used with ``X_clean`` for oracle scoring.
        input_features : sequence of str or None, default=None
            Optional variable names (e.g. ``["x", "y"]`` or ``["p", "c", "n"]``).

        Returns
        -------
        self : WeakSINDy
            Fitted estimator.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, np.newaxis]
        if X_arr.ndim not in (2, 3):
            raise ValueError(f"Expected X to be 2D or 3D; got shape {X_arr.shape}.")

        nt = X_arr.shape[0]
        dt = extract_time_step(t)
        if np.ndim(t) == 1 and np.asarray(t).size != nt:
            raise ValueError(
                f"Length of time vector t ({np.asarray(t).size}) must match X.shape[0] ({nt})."
            )

        raw_orders = self.orders if orders is None else orders
        if np.ndim(raw_orders) == 0:
            cand_orders = [float(raw_orders)]  # type: ignore[arg-type]
        else:
            cand_orders = [float(a) for a in raw_orders]  # type: ignore[union-attr]
        if len(cand_orders) == 0:
            raise ValueError("Candidate orders list must be non-empty.")

        eff_k_start = min(int(self.k_start), max(0, nt // 4))

        lib = self._make_library(X_arr, dx=dx)
        lib.fit(X_arr)
        Theta_full = lib.transform(X_arr)

        if X_clean is not None:
            X_val_src = np.asarray(X_clean, dtype=np.float64)
            if X_val_src.ndim == 1:
                X_val_src = X_val_src[:, np.newaxis]
            Theta_val_src = lib.transform(X_val_src)
        else:
            X_val_src = X_arr
            Theta_val_src = Theta_full

        order_scores: dict[float, float] = {}

        if len(cand_orders) == 1:
            best_order = cand_orders[0]
            X_design, Y_target = self._assemble_system(
                X_arr, Theta_full, dt=dt, alpha=best_order, k_start=eff_k_start, k_end=nt
            )
            opt = self._make_optimizer().fit(X_design, Y_target)
            y_pred = opt.predict(X_design)
            best_r2 = _mean_r2_score(Y_target, y_pred)
            order_scores[best_order] = best_r2
        else:
            nt_train = int((1.0 - float(self.val_split)) * nt)
            if nt_train - eff_k_start < 8 or nt - nt_train < 6:
                raise ValueError(
                    f"Trajectory length nt={nt} is too short for order selection split "
                    f"(k_start={eff_k_start}, val_split={self.val_split})."
                )

            best_order = cand_orders[0]
            best_r2 = -np.inf
            best_opt: STLSQ | SR3 | None = None

            for alpha in cand_orders:
                X_tr, Y_tr = self._assemble_system(
                    X_arr, Theta_full, dt=dt, alpha=alpha, k_start=eff_k_start, k_end=nt_train
                )
                val_alpha = float(true_order) if (X_clean is not None and true_order is not None) else alpha
                X_val, Y_val = self._assemble_system(
                    X_val_src,
                    Theta_val_src,
                    dt=dt,
                    alpha=val_alpha,
                    k_start=nt_train,
                    k_end=nt,
                )
                opt_cand = self._make_optimizer().fit(X_tr, Y_tr)
                Y_pred = opt_cand.predict(X_val)
                r2 = _mean_r2_score(Y_val, Y_pred)
                order_scores[alpha] = r2
                if r2 > best_r2:
                    best_r2 = r2
                    best_order = alpha
                    best_opt = opt_cand

            if self.refit or best_opt is None:
                X_full_design, Y_full_target = self._assemble_system(
                    X_arr, Theta_full, dt=dt, alpha=best_order, k_start=eff_k_start, k_end=nt
                )
                opt = self._make_optimizer().fit(X_full_design, Y_full_target)
            else:
                opt = best_opt

        self.dt_ = dt
        self.library_ = lib
        self.optimizer_ = opt
        self.coef_ = opt.coef_
        self.order_ = float(best_order)
        self.order_scores_ = order_scores
        self.r2_score_ = float(best_r2)
        self.n_features_in_ = X_arr.shape[-1]
        self.input_features_ = lib._default_feature_names(self.n_features_in_, input_features)
        self.feature_names_ = lib.get_feature_names(self.input_features_)
        return self

    def predict(self, t: float | np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
        """Predict fractional time derivatives ``D_t^(order_) X = Theta(X) @ coef_.T``.

        Parameters
        ----------
        t : float or np.ndarray
            Time coordinate vector ``t`` or step ``dt`` (or state array ``X`` if called
            with a single argument).
        X : np.ndarray or None, default=None
            State array of shape ``(nt, n_vars)`` or ``(nt, nx, n_vars)``.

        Returns
        -------
        dX_pred : np.ndarray
            Predicted fractional derivatives matching the shape of ``X``.
        """
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

        Theta = self.library_.transform(X_arr)
        eff_k = min(int(self.k_start), max(0, X_arr.shape[0] // 4))
        X_des, Y_tar = self._assemble_system(
            X_arr, Theta, dt=dt, alpha=self.order_, k_start=eff_k, k_end=X_arr.shape[0]
        )
        return _mean_r2_score(Y_tar, X_des @ self.coef_.T)

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

        lhs_op = (
            "d/dt"
            if np.isclose(self.order_, 1.0)
            else f"D_t^{self.order_:.2f}"
        )
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
