"""Vector-valued multi-order and two-scale fractional differential operators & discovery.

Supports:
  1. Distinct per-state fractional orders ``alpha = (alpha_1, ..., alpha_d) in (0, 1]^d``:

     .. math::

         D_t^{\\alpha_j} x_j = \\Theta(\\mathbf{x}, \\nabla \\mathbf{x}) \\boldsymbol{\\xi}_j, \\qquad j = 1, \\dots, d

  2. Two-scale distributed/composite fractional operators per state variable:

     .. math::

         \\left(D_t^{\\alpha_{1,j}} + c_j D_t^{\\alpha_{2,j}}\\right) x_j = \\Theta(\\mathbf{x}, \\nabla \\mathbf{x}) \\boldsymbol{\\xi}_j

  with coordinate + bounded ``L-BFGS-B`` Variable-Projection (VarPro) order refinement.
"""
from __future__ import annotations

from typing import Callable, Literal, Sequence

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from ._sklearn_compat import BaseEstimator
from .library import BaseCandidateLibrary, PDELibrary, PolynomialLibrary
from .operators import (
    extract_spatial_steps,
    extract_time_step,
    fractional_derivative_gl,
    gl_weights,
)
from .solvers import SR3, STLSQ, _compute_aic_bic, _mean_r2_score
from .test_functions import (
    build_spatial_weak_weights,
    build_weak_weights,
    bump_test_function,
    contract_tensor_weak,
    generate_windows,
    resolve_spatial_windows,
)


def multi_order_fractional_derivative(
    X: np.ndarray,
    t: float | np.ndarray = 1.0,
    alphas: Sequence[float] | np.ndarray = (1.0,),
    *,
    second_alphas: Sequence[float] | np.ndarray | None = None,
    second_weights: Sequence[float] | np.ndarray | None = None,
    axis: int = 0,
) -> np.ndarray:
    """Evaluate distinct causal GL fractional derivatives per state channel ``j = 1, ..., d``.

    When ``second_alphas`` and ``second_weights`` are supplied, evaluates the two-scale
    composite operator ``(D_t^{alpha_{1,j}} + c_j D_t^{alpha_{2,j}}) x_j``.
    """
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    n_vars = arr.shape[-1]
    a1 = np.asarray(alphas, dtype=np.float64).ravel()
    if a1.size == 1 and n_vars > 1:
        a1 = np.full(n_vars, float(a1[0]), dtype=np.float64)
    if a1.size != n_vars:
        raise ValueError(f"Length of alphas ({a1.size}) must match X.shape[-1] ({n_vars}).")

    out = np.empty_like(arr)
    for j in range(n_vars):
        out[..., j] = fractional_derivative_gl(arr[..., j], t=t, alpha=float(a1[j]), axis=axis)

    if second_alphas is not None:
        a2 = np.asarray(second_alphas, dtype=np.float64).ravel()
        c_vec = (
            np.ones(n_vars, dtype=np.float64)
            if second_weights is None
            else np.asarray(second_weights, dtype=np.float64).ravel()
        )
        if a2.size != n_vars or c_vec.size != n_vars:
            raise ValueError("second_alphas and second_weights must match X.shape[-1].")
        for j in range(n_vars):
            if abs(float(c_vec[j])) > 0.0:
                out[..., j] += float(c_vec[j]) * fractional_derivative_gl(
                    arr[..., j], t=t, alpha=float(a2[j]), axis=axis
                )

    return out[0] if np.asarray(X).ndim == 1 else out


def integrate_multi_order_ode(
    rhs: Callable[..., np.ndarray],
    y0: Sequence[float] | np.ndarray,
    t: np.ndarray | tuple[int, float],
    alphas: Sequence[float] | np.ndarray,
    *,
    second_alphas: Sequence[float] | np.ndarray | None = None,
    second_weights: Sequence[float] | np.ndarray | None = None,
    n_corrector: int = 2,
) -> np.ndarray:
    """Integrate an incommensurate multi-order (or two-scale) fractional ODE system.

    Solves ``D_t^{alpha_j} y_j(t) = rhs_j(t, y(t))`` (or the two-scale composite operator
    ``(D_t^{alpha_{1,j}} + c_j D_t^{alpha_{2,j}}) y_j(t) = rhs_j(t, y(t))``) via causal
    Grünwald–Letnikov predictor-corrector iteration with component-specific memory kernels.
    """
    if isinstance(t, tuple) and len(t) == 2:
        nt = int(t[0])
        dt = extract_time_step(t[1])
        t_grid = np.arange(nt, dtype=np.float64) * dt
    else:
        t_grid = np.asarray(t, dtype=np.float64)
        dt = extract_time_step(t_grid)
        nt = t_grid.size

    y0_arr = np.atleast_1d(np.asarray(y0, dtype=np.float64))
    d = y0_arr.size
    a1 = np.asarray(alphas, dtype=np.float64).ravel()
    if a1.size != d:
        raise ValueError(f"Length of alphas ({a1.size}) must match len(y0) ({d}).")
    if np.any(a1 <= 0.0) or not np.all(np.isfinite(a1)):
        raise ValueError(f"All fractional orders must be finite and > 0; got {a1}.")

    # Build effective causal GL convolution kernel K_j[m] for each component j:
    # (L_j y_j)_k = sum_{m=0}^k K_j[m] y_j[k - m] = f_j(t_k, y_k)
    kernels = np.zeros((d, nt), dtype=np.float64)
    for j in range(d):
        kernels[j] = gl_weights(float(a1[j]), nt) / (dt ** float(a1[j]))

    if second_alphas is not None:
        a2 = np.asarray(second_alphas, dtype=np.float64).ravel()
        c_vec = (
            np.ones(d, dtype=np.float64)
            if second_weights is None
            else np.asarray(second_weights, dtype=np.float64).ravel()
        )
        if a2.size != d or c_vec.size != d:
            raise ValueError("second_alphas and second_weights must have length len(y0).")
        for j in range(d):
            if float(c_vec[j]) != 0.0:
                kernels[j] += float(c_vec[j]) * gl_weights(float(a2[j]), nt) / (dt ** float(a2[j]))

    diag_k0 = kernels[:, 0]  # shape (d,)
    if np.any(diag_k0 <= 0.0):
        raise ValueError("Composite GL operator diagonal weight must remain strictly positive.")

    y = np.zeros((nt, d), dtype=np.float64)
    y[0] = y0_arr

    two_arg = True
    try:
        _ = rhs(float(t_grid[0]), y0_arr)
    except TypeError:
        two_arg = False

    for k in range(1, nt):
        y_rev = y[k - 1 :: -1]  # shape (k, d)
        hist = np.einsum("jk,kj->j", kernels[:, 1 : k + 1], y_rev)
        f_prev = np.asarray(
            rhs(float(t_grid[k - 1]), y[k - 1]) if two_arg else rhs(y[k - 1]),
            dtype=np.float64,
        )
        yk = (f_prev - hist) / diag_k0
        t_curr = float(t_grid[k])
        for _ in range(int(n_corrector)):
            f_curr = np.asarray(
                rhs(t_curr, yk) if two_arg else rhs(yk),
                dtype=np.float64,
            )
            yk = (f_curr - hist) / diag_k0
        y[k] = yk

    return y


class MultiOrderWeakSINDy(BaseEstimator):
    """Weak-form discovery of vector-valued multi-order and two-scale fractional systems.

    Discovers distinct fractional differentiation orders ``alpha = (alpha_1, ..., alpha_d)``
    per state variable (and optional two-scale ``D_t^{alpha_{1,j}} + c_j D_t^{alpha_{2,j}}``
    composite operators when ``two_scale=True``) via coordinate warm-starting and joint
    bounded ``L-BFGS-B`` Variable-Projection (VarPro) optimization.
    """

    def __init__(
        self,
        library: BaseCandidateLibrary | None = None,
        optimizer: Literal["stlsq", "sr3"] | STLSQ | SR3 = "stlsq",
        threshold: float = 0.02,
        alpha_reg: float = 1e-12,
        order_bounds: tuple[float, float] = (0.15, 1.0),
        two_scale: bool = False,
        min_order_gap: float = 0.20,
        weight_bounds: tuple[float, float] = (0.0, 5.0),
        criterion: Literal["bic", "aic"] = "bic",
        use_weak: bool = True,
        window_size: int = 44,
        stride: int = 4,
        spatial_window_size: int | Sequence[int] | None = None,
        spatial_stride: int | Sequence[int] | None = None,
        test_degree: int = 4,
        k_start: int = 16,
        dx: float | Sequence[float] | None = None,
        n_alternating: int = 2,
    ) -> None:
        self.library = library
        self.optimizer = optimizer
        self.threshold = threshold
        self.alpha_reg = alpha_reg
        self.order_bounds = order_bounds
        self.two_scale = two_scale
        self.min_order_gap = min_order_gap
        self.weight_bounds = weight_bounds
        self.criterion = criterion
        self.use_weak = use_weak
        self.window_size = window_size
        self.stride = stride
        self.spatial_window_size = spatial_window_size
        self.spatial_stride = spatial_stride
        self.test_degree = test_degree
        self.k_start = k_start
        self.dx = dx
        self.n_alternating = n_alternating

    @property
    def alphas_(self) -> np.ndarray:
        return self.orders_

    def _make_optimizer(self) -> STLSQ | SR3:
        if isinstance(self.optimizer, (STLSQ, SR3)):
            return self.optimizer.__class__(**self.optimizer.get_params())
        if self.optimizer == "stlsq":
            return STLSQ(threshold=self.threshold, alpha=self.alpha_reg)
        if self.optimizer == "sr3":
            return SR3(threshold=self.threshold)
        raise ValueError(f"Unsupported optimizer {self.optimizer!r}; expected 'stlsq' or 'sr3'.")

    def _make_library(self, X: np.ndarray, eff_dx: float | Sequence[float] | None) -> BaseCandidateLibrary:
        if self.library is not None:
            return self.library
        if X.ndim >= 3:
            if eff_dx is None:
                raise ValueError("Spatiotemporal PDE input requires dx=...")
            return PDELibrary(dx=eff_dx, poly_degree=2, include_interaction=False)
        return PolynomialLibrary(degree=3, include_bias=False, include_interaction=True)

    def fit(
        self,
        t: float | np.ndarray,
        X: np.ndarray,
        *,
        order_bounds: tuple[float, float] | None = None,
        dx: float | Sequence[float] | None = None,
        input_features: Sequence[str] | None = None,
    ) -> MultiOrderWeakSINDy:
        """Fit per-state fractional orders ``orders_`` (and two-scale parameters) plus sparse ``coef_``."""
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, np.newaxis]
        nt = X_arr.shape[0]
        n_vars = X_arr.shape[-1]
        dt = extract_time_step(t)
        if np.ndim(t) == 1 and np.asarray(t).size != nt:
            raise ValueError(f"Length of t ({np.asarray(t).size}) must match X.shape[0] ({nt}).")

        eff_dx = dx if dx is not None else self.dx
        bounds = (
            self.order_bounds
            if order_bounds is None
            else (float(order_bounds[0]), float(order_bounds[1]))
        )
        a_min, a_max = float(bounds[0]), float(bounds[1])
        if not (0.0 < a_min < a_max):
            raise ValueError(f"order_bounds must satisfy 0 < alpha_min < alpha_max; got {bounds}.")

        lib = self._make_library(X_arr, eff_dx)
        lib.fit(X_arr)
        eff_k_start = min(int(self.k_start), max(0, nt // 5))
        sp_ndim = X_arr.ndim - 2

        if self.use_weak:
            span = nt - eff_k_start
            H = int(self.window_size)
            S = int(self.stride)
            if span <= H + 2:
                H = max(4, (span - 2) // 2)
                S = max(1, min(S, H // 4))
            wins = generate_windows(
                n_steps=nt, window_size=H, stride=S, k_start=eff_k_start, k_end=nt
            )
            phi_t = bump_test_function(window_size=H, degree=self.test_degree, dt=dt, normalize=True)
            Phi_t, _ = build_weak_weights(n_steps=nt, windows=wins, alpha=0.0, phi=phi_t, dt=dt)

            if sp_ndim >= 1:
                sp_shape = X_arr.shape[1:-1]
                sp_wins = resolve_spatial_windows(
                    sp_shape,
                    spatial_window_size=self.spatial_window_size,
                    spatial_stride=self.spatial_stride,
                )
                sp_steps = (
                    extract_spatial_steps(eff_dx, sp_ndim)
                    if eff_dx is not None
                    else (1.0,) * sp_ndim
                )
                Phi_x0 = [
                    build_spatial_weak_weights(
                        n_grid=sp_shape[d],
                        windows=sp_wins[d],
                        dx=sp_steps[d],
                        degree=self.test_degree,
                        diff_order=0,
                    )
                    for d in range(sp_ndim)
                ]
                if hasattr(lib, "_project_spatiotemporal_weak"):
                    G = lib._project_spatiotemporal_weak(
                        X_arr,
                        Phi_t=Phi_t,
                        Phi_x0=Phi_x0,
                        sp_wins=sp_wins,
                        sp_steps=sp_steps,
                        sp_degree=self.test_degree,
                    )
                else:
                    Theta_pt = lib.transform(X_arr)
                    G = contract_tensor_weak(Theta_pt, [Phi_t, *Phi_x0])

                curr_sp = X_arr
                for d in range(sp_ndim):
                    c_sp = np.tensordot(curr_sp, Phi_x0[d], axes=([1 + d], [0]))
                    curr_sp = np.moveaxis(c_sp, -1, 1 + d)
                X_sp = curr_sp.reshape(nt, -1, n_vars)
                sp_gain = float(np.prod([float(np.mean(np.sum(w * w, axis=0))) for w in Phi_x0]))
                overlap = max(1.0, H / max(1, S))
            else:
                Theta_pt = lib.transform(X_arr)
                G = Phi_t.T @ Theta_pt
                X_sp = None
                sp_gain = 1.0
                overlap = max(1.0, H / max(1, S)) if len(wins) >= 2 else 1.0

            def component_target_and_gain(j_idx: int, alpha_v: float) -> tuple[np.ndarray, float]:
                _, Psi_t = build_weak_weights(
                    n_steps=nt, windows=wins, alpha=float(alpha_v), phi=phi_t, dt=dt
                )
                t_gain = float(np.mean(np.sum(Psi_t * Psi_t, axis=0)))
                tot_gain = max(t_gain * sp_gain, 1e-30)
                if X_sp is not None:
                    proj = np.tensordot(Psi_t, X_sp[:, :, j_idx], axes=([0], [0]))
                    return proj.ravel(), tot_gain
                return Psi_t.T @ X_arr[:, j_idx], tot_gain

        else:
            Theta_pt = lib.transform(X_arr)
            G = Theta_pt[eff_k_start:nt].reshape(-1, Theta_pt.shape[-1])
            overlap = 1.0

            def component_target_and_gain(j_idx: int, alpha_v: float) -> tuple[np.ndarray, float]:
                xdot = fractional_derivative_gl(X_arr[..., j_idx], t=dt, alpha=float(alpha_v), axis=0)
                return xdot[eff_k_start:nt].ravel(), float(dt ** (-2.0 * float(alpha_v)))

        state_vars = np.array(
            [max(float(np.var(X_arr[..., j])), 1e-12) for j in range(n_vars)],
            dtype=np.float64,
        )
        n_indep_per_eq = max(4.0, G.shape[0] / max(1.0, overlap))
        n_features = G.shape[1]

        if not self.two_scale:
            # Stage 1: Coordinate coarse + Brent initialization per state j
            alpha_init = np.zeros(n_vars, dtype=np.float64)
            active_supports: list[np.ndarray] = []
            coarse_grid = np.linspace(a_min, a_max, 21, dtype=np.float64)

            for j in range(n_vars):
                sv_arr = np.array([state_vars[j]], dtype=np.float64)

                def eq_crit(a_val: float, j_loc: int = j, sv_loc: np.ndarray = sv_arr) -> float:
                    yj, gj = component_target_and_gain(j_loc, float(a_val))
                    opt_j = self._make_optimizer().fit(G, yj)
                    yhat_j = opt_j.predict(G).ravel()
                    n_act = int(np.sum(opt_j.coef_ != 0.0))
                    if n_act == 0:
                        return 1e12
                    aic_j, bic_j = _compute_aic_bic(
                        yj, yhat_j, n_act, noise_gain=gj, state_vars=sv_loc, n_indep=n_indep_per_eq
                    )
                    return bic_j if self.criterion == "bic" else aic_j

                c_scores = [eq_crit(float(a)) for a in coarse_grid]
                idx0 = int(np.argmin(c_scores))
                h = (a_max - a_min) / 20.0
                blo = max(a_min, float(coarse_grid[idx0]) - 1.5 * h)
                bhi = min(a_max, float(coarse_grid[idx0]) + 1.5 * h)
                res_brent = minimize_scalar(eq_crit, bounds=(blo, bhi), method="bounded", options={"xatol": 2e-4})
                a_best = float(res_brent.x) if res_brent.fun <= c_scores[idx0] else float(coarse_grid[idx0])
                alpha_init[j] = a_best
                yj_init, _ = component_target_and_gain(j, a_best)
                opt_init = self._make_optimizer().fit(G, yj_init)
                active_supports.append(opt_init.coef_[0] != 0.0)

            # Stage 2: Joint vector L-BFGS-B VarPro refinement alternating with STLSQ support updates
            alphas_curr = alpha_init.copy()
            box_bounds = [(a_min, a_max)] * n_vars

            for _ in range(max(1, int(self.n_alternating))):
                proj_bases: list[np.ndarray | None] = []
                for j in range(n_vars):
                    supp = active_supports[j]
                    if np.any(supp):
                        Q_j, _ = np.linalg.qr(G[:, supp], mode="reduced")
                        proj_bases.append(Q_j)
                    else:
                        Q_full, _ = np.linalg.qr(G, mode="reduced")
                        proj_bases.append(Q_full)

                def joint_varpro_loss(alpha_vec: np.ndarray) -> float:
                    total_loss = 0.0
                    for j_idx in range(n_vars):
                        aj = float(np.clip(alpha_vec[j_idx], a_min, a_max))
                        yj, gj = component_target_and_gain(j_idx, aj)
                        Qj = proj_bases[j_idx]
                        assert Qj is not None
                        res_j = yj - Qj @ (Qj.T @ yj)
                        mse_j = float(np.mean(res_j * res_j))
                        rel_j = max(mse_j / (max(gj, 1e-30) * state_vars[j_idx]), 1e-16)
                        total_loss += n_indep_per_eq * float(np.log(rel_j))
                    return total_loss

                res_lbfgs = minimize(
                    joint_varpro_loss,
                    x0=alphas_curr,
                    method="L-BFGS-B",
                    bounds=box_bounds,
                    options={"ftol": 1e-11, "gtol": 1e-8, "eps": 1e-5, "maxiter": 80},
                )
                if np.all(np.isfinite(res_lbfgs.x)):
                    alphas_curr = np.clip(res_lbfgs.x.astype(np.float64), a_min, a_max)

                for j in range(n_vars):
                    yj_ref, _ = component_target_and_gain(j, float(alphas_curr[j]))
                    opt_ref = self._make_optimizer().fit(G, yj_ref)
                    new_supp = opt_ref.coef_[0] != 0.0
                    if np.any(new_supp):
                        active_supports[j] = new_supp

            coefs = np.zeros((n_vars, n_features), dtype=np.float64)
            Y_final = np.zeros((G.shape[0], n_vars), dtype=np.float64)
            Y_pred = np.zeros((G.shape[0], n_vars), dtype=np.float64)
            gains_final = np.zeros(n_vars, dtype=np.float64)

            for j in range(n_vars):
                yj, gj = component_target_and_gain(j, float(alphas_curr[j]))
                opt_f = self._make_optimizer().fit(G, yj)
                coefs[j] = opt_f.coef_[0]
                Y_final[:, j] = yj
                Y_pred[:, j] = opt_f.predict(G).ravel()
                gains_final[j] = gj

            self.orders_ = alphas_curr
            self.second_orders_ = None
            self.second_weights_ = None
        else:
            # Two-scale operator discovery: (D_t^{alpha_{1,j}} + c_j D_t^{alpha_{2,j}}) x_j = Theta(x) xi_j
            gap = float(self.min_order_gap)
            c_lo, c_hi = float(self.weight_bounds[0]), float(self.weight_bounds[1])
            alpha1_out = np.zeros(n_vars, dtype=np.float64)
            alpha2_out = np.zeros(n_vars, dtype=np.float64)
            c_out = np.zeros(n_vars, dtype=np.float64)
            coefs = np.zeros((n_vars, n_features), dtype=np.float64)
            Y_final = np.zeros((G.shape[0], n_vars), dtype=np.float64)
            Y_pred = np.zeros((G.shape[0], n_vars), dtype=np.float64)
            gains_final = np.zeros(n_vars, dtype=np.float64)

            grid_a1 = np.linspace(a_min + gap, a_max, 10, dtype=np.float64)
            grid_a2 = np.linspace(a_min, a_max - gap, 10, dtype=np.float64)

            for j in range(n_vars):
                # Precompute QR of G for fast VarPro joint projection of [G, -y(alpha2)]
                best_obj = float("inf")
                best_triplet = (min(a_max, 0.9), max(a_min, 0.4), 0.5)

                for a1_c in grid_a1:
                    y1_c, g1_c = component_target_and_gain(j, float(a1_c))
                    for a2_c in grid_a2:
                        if a1_c - a2_c < gap - 1e-9:
                            continue
                        y2_c, g2_c = component_target_and_gain(j, float(a2_c))
                        # Solve min_{xi, c} || y1 + c y2 - G xi ||_2^2 via augmented least squares
                        A_aug = np.column_stack([G, -y2_c])
                        sol_aug, *_ = np.linalg.lstsq(A_aug, y1_c, rcond=None)
                        c_hat = float(np.clip(sol_aug[-1], c_lo, c_hi))
                        lhs_c = y1_c + c_hat * y2_c
                        opt_c = self._make_optimizer().fit(G, lhs_c)
                        n_act = int(np.sum(opt_c.coef_ != 0.0))
                        if n_act == 0:
                            continue
                        pred_c = opt_c.predict(G).ravel()
                        mse_c = float(np.mean((lhs_c - pred_c) ** 2))
                        g_tot = max(g1_c + (c_hat * c_hat) * g2_c, 1e-30)
                        rel_c = max(mse_c / (g_tot * state_vars[j]), 1e-16)
                        pen = np.log(n_indep_per_eq) * (n_act + (1 if c_hat > 1e-3 else 0))
                        bic_c = float(n_indep_per_eq * np.log(rel_c) + pen)
                        if bic_c < best_obj:
                            best_obj = bic_c
                            best_triplet = (float(a1_c), float(a2_c), c_hat)

                # Refine (alpha1, delta=alpha1-alpha2, c) jointly with bounded L-BFGS-B on sparse support
                a1_0, a2_0, c_0 = best_triplet
                lhs_0 = component_target_and_gain(j, a1_0)[0] + c_0 * component_target_and_gain(j, a2_0)[0]
                supp_j = self._make_optimizer().fit(G, lhs_0).coef_[0] != 0.0
                G_sub = G[:, supp_j] if np.any(supp_j) else G
                Q_sub, _ = np.linalg.qr(G_sub, mode="reduced")

                # Parameterize as (alpha2 in [a_min, a_max - gap], gap_val in [gap, a_max - a_min]) with exact VarPro c_hat
                def _opt_c_for_pair(
                    y1_v: np.ndarray, y2_v: np.ndarray, Q_loc: np.ndarray
                ) -> tuple[float, np.ndarray]:
                    r1 = y1_v - Q_loc @ (Q_loc.T @ y1_v)
                    r2 = y2_v - Q_loc @ (Q_loc.T @ y2_v)
                    denom = float(np.dot(r2, r2))
                    if denom <= 1e-18:
                        c_opt = 0.0
                    else:
                        c_opt = float(np.clip(-float(np.dot(r1, r2)) / denom, c_lo, c_hi))
                    return c_opt, r1 + c_opt * r2

                def two_scale_loss(p: np.ndarray, j_loc: int = j, Q_loc: np.ndarray = Q_sub) -> float:
                    a2_v = float(p[0])
                    a1_v = float(min(a_max, a2_v + p[1]))
                    if a1_v - a2_v < gap * 0.95:
                        return 1e12
                    y1_v, g1_v = component_target_and_gain(j_loc, a1_v)
                    y2_v, g2_v = component_target_and_gain(j_loc, a2_v)
                    c_v, res_v = _opt_c_for_pair(y1_v, y2_v, Q_loc)
                    mse_v = float(np.mean(res_v * res_v))
                    g_tot = max(g1_v + (c_v * c_v) * g2_v, 1e-30)
                    rel_v = max(mse_v / (g_tot * state_vars[j_loc]), 1e-16)
                    return float(n_indep_per_eq * np.log(rel_v))

                x0_ts = np.array([a2_0, max(gap, a1_0 - a2_0)], dtype=np.float64)
                ts_bounds = [(a_min, a_max - gap), (gap, a_max - a_min)]
                res_ts = minimize(
                    two_scale_loss,
                    x0=x0_ts,
                    method="L-BFGS-B",
                    bounds=ts_bounds,
                    options={"ftol": 1e-11, "gtol": 1e-8, "eps": 1e-5, "maxiter": 80},
                )
                if np.all(np.isfinite(res_ts.x)):
                    a2_f = float(res_ts.x[0])
                    a1_f = float(min(a_max, a2_f + res_ts.x[1]))
                    y1_tmp, _ = component_target_and_gain(j, a1_f)
                    y2_tmp, _ = component_target_and_gain(j, a2_f)
                    c_f, _ = _opt_c_for_pair(y1_tmp, y2_tmp, Q_sub)
                else:
                    a1_f, a2_f, c_f = a1_0, a2_0, c_0

                y1_f, g1_f = component_target_and_gain(j, a1_f)
                y2_f, g2_f = component_target_and_gain(j, a2_f)
                lhs_f = y1_f + c_f * y2_f
                opt_f = self._make_optimizer().fit(G, lhs_f)

                alpha1_out[j] = a1_f
                alpha2_out[j] = a2_f
                c_out[j] = c_f
                coefs[j] = opt_f.coef_[0]
                Y_final[:, j] = lhs_f
                Y_pred[:, j] = opt_f.predict(G).ravel()
                gains_final[j] = g1_f + (c_f * c_f) * g2_f

            self.orders_ = alpha1_out
            self.second_orders_ = alpha2_out
            self.second_weights_ = c_out

        n_active_tot = int(np.sum(coefs != 0.0))
        aic_tot, bic_tot = _compute_aic_bic(
            Y_final,
            Y_pred,
            n_active_tot,
            noise_gain=float(np.mean(gains_final)),
            state_vars=state_vars,
            n_indep=n_indep_per_eq * n_vars,
        )
        self.library_ = lib
        self.coef_ = coefs
        self.ind_ = coefs != 0.0
        self.r2_score_ = _mean_r2_score(Y_final, Y_pred)
        self.aic_ = float(aic_tot)
        self.bic_ = float(bic_tot)
        self.feature_names_ = lib.get_feature_names(input_features=input_features)
        self.input_features_ = (
            [f"x{j}" for j in range(n_vars)]
            if input_features is None
            else [str(s) for s in input_features]
        )
        self.dt_ = dt
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict multi-order (or two-scale) left-hand-side fractional derivatives ``Theta(X) Xi.T``."""
        if not hasattr(self, "coef_"):
            raise RuntimeError("MultiOrderWeakSINDy must be fitted before calling predict.")
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, np.newaxis]
        Theta = self.library_.transform(X_arr)
        return Theta @ self.coef_.T

    def equations(self, precision: int = 4) -> list[str]:
        if not hasattr(self, "coef_"):
            raise RuntimeError("MultiOrderWeakSINDy must be fitted before calling equations.")
        eqs: list[str] = []
        for j, s_name in enumerate(self.input_features_):
            a1 = float(self.orders_[j])
            if self.two_scale and self.second_orders_ is not None and self.second_weights_ is not None:
                a2 = float(self.second_orders_[j])
                cj = float(self.second_weights_[j])
                lhs = f"(D_t^{a1:.3f} + {cj:.3f} D_t^{a2:.3f}) {s_name}"
            else:
                lhs = f"D_t^{a1:.3f} {s_name}"
            terms = [
                f"{c_val:+.{precision}f} {self.feature_names_[idx]}"
                for idx, c_val in enumerate(self.coef_[j])
                if abs(c_val) > 0.0
            ]
            rhs = " ".join(terms).lstrip("+ ") if terms else "0"
            eqs.append(f"{lhs} = {rhs}")
        return eqs

    def print(self, precision: int = 4) -> None:
        for line in self.equations(precision=precision):
            print(line)
