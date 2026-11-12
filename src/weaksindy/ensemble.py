"""Uncertainty quantification and conformal ensemble weak-form SINDy."""

from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np

from ._sklearn_compat import BaseEstimator
from .library import BaseCandidateLibrary
from .operators import extract_time_step
from .solvers import SR3, STLSQ, WeakSINDy, _compute_aic_bic, _mean_r2_score


class EnsembleWeakSINDy(BaseEstimator):
    """Weak-form Subdomain Bagging, Library Ensemble, and Split-Conformal UQ for SINDy.

    Constructs an ensemble of ``n_models`` sparse estimators over bootstrapped weak-form
    spatiotemporal subdomains and optional candidate library sub-samples. Quantifies
    finite-sample term inclusion probabilities, posterior medians/standard deviations,
    and distribution-free conformalized quantile confidence intervals ``coef_ci(alpha)``.

    Parameters
    ----------
    library : BaseCandidateLibrary or None, default=None
        Candidate basis library (defaults to ``PolynomialLibrary`` for ODEs or
        ``PDELibrary`` for N-D PDEs).
    optimizer : {"stlsq", "sr3"} or STLSQ or SR3, default="stlsq"
        Base sparse regression solver.
    threshold : float, default=0.01
        Sparsity hard-thresholding cutoff.
    alpha_reg : float, default=1e-4
        Tikhonov / ridge regularization penalty for ``STLSQ``.
    n_models : int, default=50
        Number of bootstrap ensemble models ``B >= 2``.
    subsample_fraction : float, default=0.7
        Fraction of weak-form subdomains sampled per ensemble model ``(0, 1]``.
    library_subsample_fraction : float, default=1.0
        Fraction of candidate library columns sampled per ensemble model ``(0, 1]``.
    replace : bool, default=True
        Whether weak-form subdomains are sampled with replacement (bootstrap bagging)
        or without replacement (subsampling).
    aggregate : {"median", "mean", "stability_selection"}, default="median"
        Ensemble aggregation rule producing ``coef_``.
    inclusion_threshold : float, default=0.5
        Minimum empirical inclusion probability ``p_{d,j} in [0, 1]`` required to retain
        a candidate term in the aggregated governing equation.
    orders : float or sequence of float, default=(1.0,)
        Candidate fractional temporal orders ``alpha``.
    optimize_order : bool, default=False
        If ``True``, continuously optimizes ``alpha`` via bounded Brent refinement and
        Pareto AIC/BIC before running the subdomain ensemble.
    order_bounds : tuple of (float, float), default=(0.1, 1.0)
        Bounds ``(alpha_min, alpha_max)`` for continuous fractional order refinement.
    criterion : {"bic", "aic"}, default="bic"
        Information criterion used when ``optimize_order=True``.
    use_weak : bool, default=True
        Use separable ``C^p`` weak-form projection (``True``) or pointwise GL (``False``).
    window_size : int, default=50
        Temporal test-function window width ``H_t``.
    stride : int, default=5
        Temporal test-function window stride ``S_t``.
    spatial_window_size : int or sequence of int or None, default=None
        Spatial test-function window width(s) ``H_{x_d}`` for N-D PDEs.
    spatial_stride : int or sequence of int or None, default=None
        Spatial test-function stride(s) ``S_{x_d}`` for N-D PDEs.
    test_degree : int, default=4
        Smoothness exponent ``p`` of the ``C^p`` bump kernel.
    k_start : int, default=20
        Initial time steps skipped to avoid ``t=0`` GL startup transients.
    val_split : float, default=0.2
        Validation fraction for discrete order sweeps.
    dx : float or sequence of float/1D arrays or None, default=None
        Spatial grid spacing(s) for N-D PDEs.
    ci_method : {"conformal", "percentile"}, default="conformal"
        Default interval construction method in ``coef_ci()``.
    random_state : int or np.random.Generator or None, default=42
        Random seed or generator for reproducible bootstrap resampling.
    """

    def __init__(
        self,
        library: BaseCandidateLibrary | None = None,
        optimizer: Literal["stlsq", "sr3"] | STLSQ | SR3 = "stlsq",
        threshold: float = 0.01,
        alpha_reg: float = 1e-12,
        n_models: int = 50,
        subsample_fraction: float = 0.7,
        library_subsample_fraction: float = 1.0,
        replace: bool = True,
        aggregate: Literal["median", "mean", "stability_selection"] = "median",
        inclusion_threshold: float = 0.5,
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
        ci_method: Literal["conformal", "percentile"] = "conformal",
        random_state: int | np.random.Generator | None = 42,
    ) -> None:
        self.library = library
        self.optimizer = optimizer
        self.threshold = threshold
        self.alpha_reg = alpha_reg
        self.n_models = n_models
        self.subsample_fraction = subsample_fraction
        self.library_subsample_fraction = library_subsample_fraction
        self.replace = replace
        self.aggregate = aggregate
        self.inclusion_threshold = inclusion_threshold
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
        self.ci_method = ci_method
        self.random_state = random_state

    @property
    def alpha_(self) -> float:
        """Alias for the recovered fractional temporal order ``order_``."""
        return self.order_

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
    ) -> EnsembleWeakSINDy:
        """Fit the weak-form subdomain ensemble, inclusion probabilities, and conformal UQ."""
        if int(self.n_models) < 2:
            raise ValueError(f"n_models must be >= 2; got {self.n_models}.")
        if not (0.0 < float(self.subsample_fraction) <= 1.0):
            raise ValueError(
                f"subsample_fraction must be in (0, 1]; got {self.subsample_fraction}."
            )
        if not (0.0 < float(self.library_subsample_fraction) <= 1.0):
            raise ValueError(
                f"library_subsample_fraction must be in (0, 1]; got {self.library_subsample_fraction}."
            )
        if self.aggregate not in ("median", "mean", "stability_selection"):
            raise ValueError(
                f"aggregate must be 'median', 'mean', or 'stability_selection'; got {self.aggregate!r}."
            )
        if not (0.0 <= float(self.inclusion_threshold) <= 1.0):
            raise ValueError(
                f"inclusion_threshold must be in [0, 1]; got {self.inclusion_threshold}."
            )

        base = WeakSINDy(
            library=self.library,
            optimizer=self.optimizer,
            threshold=self.threshold,
            alpha_reg=self.alpha_reg,
            orders=self.orders,
            optimize_order=self.optimize_order,
            order_bounds=self.order_bounds,
            criterion=self.criterion,
            use_weak=self.use_weak,
            window_size=self.window_size,
            stride=self.stride,
            spatial_window_size=self.spatial_window_size,
            spatial_stride=self.spatial_stride,
            test_degree=self.test_degree,
            k_start=self.k_start,
            val_split=self.val_split,
            dx=self.dx,
            refit=True,
        )
        base.fit(
            t,
            X,
            orders=orders,
            optimize_order=optimize_order,
            order_bounds=order_bounds,
            criterion=criterion,
            dx=dx,
            X_clean=X_clean,
            true_order=true_order,
            input_features=input_features,
        )

        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, np.newaxis]
        nt = X_arr.shape[0]
        eff_k_start = min(int(self.k_start), max(0, nt // 4))

        full_cache = base._build_cache(
            X_arr,
            lib=base.library_,
            dt=base.dt_,
            k_start=eff_k_start,
            k_end=nt,
            dx=base.dx_,
        )
        X_design = full_cache.X_design
        Y_target = full_cache.target(base.order_)

        n_sub, n_features = X_design.shape
        n_vars = Y_target.shape[1]
        B = int(self.n_models)

        m_sub = max(2, int(round(float(self.subsample_fraction) * n_sub)))
        if not self.replace:
            m_sub = min(m_sub, max(2, n_sub - 1)) if n_sub > 2 else n_sub

        p_sub = (
            n_features
            if float(self.library_subsample_fraction) >= 1.0
            else max(1, int(round(float(self.library_subsample_fraction) * n_features)))
        )

        rng = (
            self.random_state
            if isinstance(self.random_state, np.random.Generator)
            else np.random.default_rng(self.random_state)
        )

        coef_ensemble = np.zeros((B, n_vars, n_features), dtype=np.float64)
        eligible_mask = np.zeros((B, n_features), dtype=bool)
        cal_replicates: list[np.ndarray] = []

        for b in range(B):
            row_idx = rng.choice(n_sub, size=m_sub, replace=bool(self.replace))
            if p_sub < n_features:
                col_idx = np.sort(rng.choice(n_features, size=p_sub, replace=False))
                # Guarantee every feature is sampled at least once by the final model
                if b == B - 1:
                    missing = np.flatnonzero(~np.any(eligible_mask, axis=0))
                    if missing.size > 0:
                        col_idx = np.unique(np.concatenate([col_idx, missing]))
            else:
                col_idx = np.arange(n_features)

            eligible_mask[b, col_idx] = True
            opt_b = base._make_optimizer().fit(X_design[np.ix_(row_idx, col_idx)], Y_target[row_idx])
            coef_ensemble[b][:, col_idx] = opt_b.coef_

            # Disjoint out-of-bag subdomain calibration replicate for split-conformal CQR
            in_bag = np.zeros(n_sub, dtype=bool)
            in_bag[row_idx] = True
            oob_idx = np.flatnonzero(~in_bag)
            if oob_idx.size >= max(4, min(8, n_sub // 4)):
                # Match sample size scale where possible via bootstrap from disjoint OOB pool
                oob_draw = rng.choice(oob_idx, size=m_sub, replace=True)
                opt_cal = base._make_optimizer().fit(X_design[oob_draw], Y_target[oob_draw])
                cal_replicates.append(opt_cal.coef_)

        # Fallback split-conformal block if subsample_fraction=1.0 with replace=False left no OOB
        if len(cal_replicates) < 4:
            n_cal = max(2, n_sub // 3)
            for _ in range(max(10, B // 2)):
                cal_draw = rng.choice(n_sub, size=n_cal, replace=True)
                opt_cal = base._make_optimizer().fit(X_design[cal_draw], Y_target[cal_draw])
                cal_replicates.append(opt_cal.coef_)

        if p_sub == n_features:
            inc_prob = np.mean(coef_ensemble != 0.0, axis=0)
            c_med = np.median(coef_ensemble, axis=0)
            c_mean = np.mean(coef_ensemble, axis=0)
            c_std = np.std(coef_ensemble, axis=0, ddof=1 if B > 1 else 0)
        else:
            inc_prob = np.zeros((n_vars, n_features), dtype=np.float64)
            c_med = np.zeros((n_vars, n_features), dtype=np.float64)
            c_mean = np.zeros((n_vars, n_features), dtype=np.float64)
            c_std = np.zeros((n_vars, n_features), dtype=np.float64)
            for j in range(n_features):
                elig = eligible_mask[:, j]
                sub_vals = coef_ensemble[elig, :, j]  # (n_elig, n_vars)
                inc_prob[:, j] = np.mean(sub_vals != 0.0, axis=0)
                c_med[:, j] = np.median(sub_vals, axis=0)
                c_mean[:, j] = np.mean(sub_vals, axis=0)
                c_std[:, j] = np.std(sub_vals, axis=0, ddof=1 if sub_vals.shape[0] > 1 else 0)

        # Aggregate final sparse coefficients
        agg_coef = np.zeros((n_vars, n_features), dtype=np.float64)
        prob_mask = inc_prob >= float(self.inclusion_threshold)

        if self.aggregate == "median":
            keep = prob_mask & (np.abs(c_med) >= float(self.threshold))
            agg_coef[keep] = c_med[keep]
        elif self.aggregate == "mean":
            keep = prob_mask & (np.abs(c_mean) >= float(self.threshold))
            agg_coef[keep] = c_mean[keep]
        elif self.aggregate == "stability_selection":
            for d in range(n_vars):
                act = prob_mask[d].copy()
                if np.any(act):
                    c_refit, *_ = np.linalg.lstsq(X_design[:, act], Y_target[:, d], rcond=None)
                    small = np.abs(c_refit) < float(self.threshold)
                    c_refit[small] = 0.0
                    agg_coef[d, act] = c_refit
                    still_act = agg_coef[d] != 0.0
                    if np.any(small) and np.any(still_act):
                        c_refit2, *_ = np.linalg.lstsq(
                            X_design[:, still_act], Y_target[:, d], rcond=None
                        )
                        agg_coef[d, still_act] = c_refit2

        Y_pred_full = X_design @ agg_coef.T
        n_active_final = int(np.sum(agg_coef != 0.0))
        final_aic, final_bic = _compute_aic_bic(Y_target, Y_pred_full, n_active_final)

        self._base_model_ = base
        self._eligible_mask_ = eligible_mask
        self._cal_replicates_ = np.stack(cal_replicates, axis=0)
        self.coef_ensemble_ = coef_ensemble
        self.inclusion_probabilities_ = inc_prob
        self.coef_median_ = c_med
        self.coef_mean_ = c_mean
        self.coef_std_ = c_std
        self.coef_ = agg_coef

        self.dt_ = base.dt_
        self.dx_ = base.dx_
        self.library_ = base.library_
        self.order_ = base.order_
        self.order_scores_ = base.order_scores_
        self.order_profile_ = base.order_profile_
        self.r2_score_ = _mean_r2_score(Y_target, Y_pred_full)
        self.aic_ = float(final_aic)
        self.bic_ = float(final_bic)
        self.n_features_in_ = base.n_features_in_
        self.input_features_ = base.input_features_
        self.feature_names_ = base.feature_names_
        return self

    def coef_ci(
        self,
        alpha: float = 0.05,
        method: Literal["conformal", "percentile"] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute ``(1 - alpha)`` confidence intervals ``(ci_low, ci_high)`` for all coefficients.

        Parameters
        ----------
        alpha : float, default=0.05
            Miscoverage significance level ``alpha in (0, 1)`` (e.g. ``0.05`` for 95% CIs).
        method : {"conformal", "percentile"} or None, default=None
            - ``"percentile"``: empirical bootstrap quantile interval ``[q_{alpha/2}, q_{1-alpha/2}]``.
            - ``"conformal"``: distribution-free out-of-bag split-conformalized quantile
              interval (Romano et al., 2019) using finite-sample rank ``ceil((1 - alpha)(C + 1)) / C``.

        Returns
        -------
        ci_low : np.ndarray of shape (n_vars, n_features)
            Lower confidence bounds.
        ci_high : np.ndarray of shape (n_vars, n_features)
            Upper confidence bounds.
        """
        if not hasattr(self, "coef_ensemble_"):
            raise RuntimeError("EnsembleWeakSINDy must be fitted before calling coef_ci.")
        alpha_f = float(alpha)
        if not (0.0 < alpha_f < 1.0):
            raise ValueError(f"Confidence level alpha must satisfy 0 < alpha < 1; got {alpha}.")

        eff_method = self.ci_method if method is None else method
        if eff_method not in ("conformal", "percentile"):
            raise ValueError(f"method must be 'conformal' or 'percentile'; got {eff_method!r}.")

        n_vars, n_features = self.coef_.shape
        q_lo = np.zeros((n_vars, n_features), dtype=np.float64)
        q_hi = np.zeros((n_vars, n_features), dtype=np.float64)

        for j in range(n_features):
            elig = self._eligible_mask_[:, j]
            vals = self.coef_ensemble_[elig, :, j]  # (n_elig, n_vars)
            q_lo[:, j] = np.quantile(vals, 0.5 * alpha_f, axis=0)
            q_hi[:, j] = np.quantile(vals, 1.0 - 0.5 * alpha_f, axis=0)

        if eff_method == "percentile":
            return q_lo, q_hi

        # Split-conformal CQR nonconformity calibration on out-of-bag subdomain replicates
        cal = self._cal_replicates_  # (C, n_vars, n_features)
        C = cal.shape[0]
        scores = np.maximum(
            np.maximum(q_lo[np.newaxis, :, :] - cal, cal - q_hi[np.newaxis, :, :]),
            0.0,
        )
        conf_level = min(1.0, float(np.ceil((1.0 - alpha_f) * (C + 1))) / float(C))
        margin = np.quantile(scores, conf_level, axis=0)
        return q_lo - margin, q_hi + margin

    def predict(
        self,
        t: float | np.ndarray,
        X: np.ndarray | None = None,
        *,
        return_std: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Predict fractional time derivatives using aggregated ``coef_`` (with optional ensemble std)."""
        if not hasattr(self, "coef_"):
            raise RuntimeError("EnsembleWeakSINDy must be fitted before calling predict.")

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
        pred = (Theta_flat @ self.coef_.T).reshape((*lead_shape, self.coef_.shape[0]))
        if was_1d:
            pred = pred[:, 0]

        if not return_std:
            return pred

        # Propagate ensemble variance across B models: shape (B, N_flat, n_vars)
        ens_preds = np.einsum("nf,bvf->bnv", Theta_flat, self.coef_ensemble_)
        std_flat = np.std(ens_preds, axis=0, ddof=1 if self.coef_ensemble_.shape[0] > 1 else 0)
        pred_std = std_flat.reshape((*lead_shape, self.coef_.shape[0]))
        if was_1d:
            pred_std = pred_std[:, 0]
        return pred, pred_std

    def score(
        self,
        t: float | np.ndarray,
        X: np.ndarray,
        y_true: np.ndarray | None = None,
    ) -> float:
        """Compute mean ``R^2`` score of the aggregated ensemble model on ``(t, X)``."""
        if not hasattr(self, "coef_"):
            raise RuntimeError("EnsembleWeakSINDy must be fitted before calling score.")
        dt = extract_time_step(t)
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, np.newaxis]

        if y_true is not None:
            yt = np.asarray(y_true, dtype=np.float64).reshape(-1, self.coef_.shape[0])
            yp = np.asarray(self.predict(dt, X_arr), dtype=np.float64).reshape(-1, self.coef_.shape[0])
            return _mean_r2_score(yt, yp)

        eff_k = min(int(self.k_start), max(0, X_arr.shape[0] // 4))
        cache = self._base_model_._build_cache(
            X_arr, lib=self.library_, dt=dt, k_start=eff_k, k_end=X_arr.shape[0], dx=self.dx_
        )
        Y_tar = cache.target(self.order_)
        return _mean_r2_score(Y_tar, cache.X_design @ self.coef_.T)

    def summary_table(
        self,
        alpha: float = 0.05,
        active_only: bool = True,
        min_inclusion: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Return structured UQ summary records for discovered/candidate basis terms."""
        if not hasattr(self, "coef_"):
            raise RuntimeError("EnsembleWeakSINDy must be fitted before calling summary_table.")
        ci_lo, ci_hi = self.coef_ci(alpha=alpha)
        rows: list[dict[str, Any]] = []
        for d, s_name in enumerate(self.input_features_):
            for j, f_name in enumerate(self.feature_names_):
                prob = float(self.inclusion_probabilities_[d, j])
                c_val = float(self.coef_[d, j])
                if active_only and c_val == 0.0 and prob <= min_inclusion:
                    continue
                if prob < min_inclusion and c_val == 0.0:
                    continue
                rows.append(
                    {
                        "state": s_name,
                        "term": f_name,
                        "coef": c_val,
                        "median": float(self.coef_median_[d, j]),
                        "std": float(self.coef_std_[d, j]),
                        "inclusion_probability": prob,
                        "ci_low": float(ci_lo[d, j]),
                        "ci_high": float(ci_hi[d, j]),
                    }
                )
        return rows

    def equations(
        self,
        precision: int = 4,
        input_features: Sequence[str] | None = None,
        include_uncertainty: bool = False,
        alpha: float = 0.05,
    ) -> list[str]:
        """Format discovered ensemble fractional differential equations as strings."""
        if not hasattr(self, "coef_"):
            raise RuntimeError("EnsembleWeakSINDy must be fitted before calling equations.")
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

        ci_lo, ci_hi = self.coef_ci(alpha=alpha) if include_uncertainty else (None, None)

        eqs: list[str] = []
        for d, s_name in enumerate(state_names):
            terms: list[str] = []
            for j, (c_val, f_name) in enumerate(zip(self.coef_[d], feat_names)):
                if c_val == 0.0:
                    continue
                sign_str = "-" if c_val < 0.0 else "+"
                mag_str = f"{abs(c_val):.{precision}f}"
                if include_uncertainty and ci_lo is not None and ci_hi is not None:
                    prob = self.inclusion_probabilities_[d, j]
                    uq_tag = f" [p={prob:.2f}, CI=({ci_lo[d, j]:.{precision}f}, {ci_hi[d, j]:.{precision}f})]"
                else:
                    uq_tag = ""
                term_body = f"{mag_str}{uq_tag}" if f_name == "1" else f"{mag_str} {f_name}{uq_tag}"
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
        include_uncertainty: bool = False,
        alpha: float = 0.05,
    ) -> None:
        """Print discovered ensemble governing equations to stdout."""
        for line in self.equations(
            precision=precision,
            input_features=input_features,
            include_uncertainty=include_uncertainty,
            alpha=alpha,
        ):
            print(line)
