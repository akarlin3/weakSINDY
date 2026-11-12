"""Exogenous control and forcing discovery (``WeakSINDyControl`` / ``weakSINDYc``).

Discovers integer- and fractional-order ODEs and spatiotemporal PDEs driven by
known external control, dosing, or forcing schedules ``u(t)`` / ``u(t, x)``:

.. math::

    D_t^\\alpha \\mathbf{x} = \\Theta(\\mathbf{x}, \\nabla \\mathbf{x}, \\mathbf{u}) \\Xi

Includes polynomial state terms, spatial PDE derivative operators ``nabla x, nabla^2 x``,
exogenous control terms ``u_j``, and bilinear state-control interaction terms ``x_i u_j``
(essential for radiotherapy LQ/cell-kill kinetics and receptor/pharmacokinetic modulation),
plus a deterministic ``PCG64`` pulsed fractionated radiotherapy-PK benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Any, Iterator, Literal, Sequence

import numpy as np
from scipy.optimize import minimize_scalar

from ._sklearn_compat import BaseEstimator
from .library import _coord_labels, _weak_mollified_derivative, spatial_finite_difference
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


@dataclass(frozen=True)
class ControlledBenchmarkDataset:
    """Container for simulated controlled fractional dynamical benchmarks."""

    X: np.ndarray
    U: np.ndarray
    t: np.ndarray
    ground_truth: dict[str, Any]
    X_clean: np.ndarray | None = None

    @property
    def alpha(self) -> float:
        return float(self.ground_truth["alpha"])

    def __iter__(self) -> Iterator[Any]:
        yield self.t
        yield self.X
        yield self.U
        yield self.ground_truth

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> Any:
        return (self.t, self.X, self.U, self.ground_truth)[idx]


def _broadcast_control(X: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Align exogenous control ``U`` with state array ``X`` of shape ``(nt, *sp_dims, n_vars)``."""
    u_arr = np.asarray(U, dtype=np.float64)
    if u_arr.ndim == 1:
        u_arr = u_arr[:, np.newaxis]
    if u_arr.shape[0] != X.shape[0]:
        raise ValueError(
            f"Control U time length ({u_arr.shape[0]}) must match X.shape[0] ({X.shape[0]})."
        )
    if X.ndim == 2:
        if u_arr.ndim != 2:
            raise ValueError(f"For ODE state X {X.shape}, U must be 1D or 2D; got {u_arr.shape}.")
        return u_arr

    # Spatiotemporal X: (nt, *sp_dims, n_vars)
    sp_shape = X.shape[1:-1]
    if u_arr.ndim == 2:
        expand = u_arr.reshape((X.shape[0], *([1] * len(sp_shape)), u_arr.shape[-1]))
        return np.broadcast_to(expand, (X.shape[0], *sp_shape, u_arr.shape[-1])).copy()
    if u_arr.shape[:-1] == X.shape[:-1]:
        return u_arr
    if u_arr.shape == X.shape[:-1]:
        return u_arr[..., np.newaxis]
    raise ValueError(
        f"Incompatible control shape U={u_arr.shape} for spatiotemporal state X={X.shape}."
    )


class ControlLibrary(BaseEstimator):
    """Weak-form candidate dictionary ``Theta(x, nabla x, u)`` with bilinear ``x_i u_j`` terms.

    Parameters
    ----------
    state_degree : int, default=2
        Maximum polynomial degree in state variables ``x``.
    control_degree : int, default=1
        Maximum polynomial degree in exogenous inputs ``u``.
    include_bias : bool, default=False
        Include constant ``1`` column.
    include_bilinear : bool, default=True
        Include all bilinear state-control interaction terms ``x_i u_j``.
    include_quadratic_cross : bool, default=False
        Include ``(x_i x_k) u_j`` cross-interaction terms when ``state_degree >= 2``.
    dx : float or sequence of float or None, default=None
        Spatial grid spacing(s) when ``X`` is a spatiotemporal PDE field ``(nt, *sp, d)``.
    derivative_orders : sequence of int, default=(1, 2)
        Spatial derivative orders included for PDE state fields when ``dx`` is provided.
    boundary : {"one_sided", "neumann", "periodic"}, default="neumann"
        Boundary treatment for spatial derivatives.
    """

    def __init__(
        self,
        state_degree: int = 2,
        control_degree: int = 1,
        include_bias: bool = False,
        include_bilinear: bool = True,
        include_quadratic_cross: bool = False,
        dx: float | Sequence[float] | None = None,
        derivative_orders: Sequence[int] = (1, 2),
        boundary: str = "neumann",
    ) -> None:
        self.state_degree = state_degree
        self.control_degree = control_degree
        self.include_bias = include_bias
        self.include_bilinear = include_bilinear
        self.include_quadratic_cross = include_quadratic_cross
        self.dx = dx
        self.derivative_orders = derivative_orders
        self.boundary = boundary

    def fit(
        self,
        X: np.ndarray,
        U: np.ndarray,
        *,
        state_names: Sequence[str] | None = None,
        control_names: Sequence[str] | None = None,
    ) -> ControlLibrary:
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, np.newaxis]
        U_arr = _broadcast_control(X_arr, U)

        n_x = X_arr.shape[-1]
        n_u = U_arr.shape[-1]
        s_names = (
            [f"x{i}" for i in range(n_x)]
            if state_names is None
            else [str(s) for s in state_names]
        )
        c_names = (
            [f"u{j}" for j in range(n_u)]
            if control_names is None
            else [str(c) for c in control_names]
        )
        if len(s_names) != n_x or len(c_names) != n_u:
            raise ValueError("Length of state_names/control_names must match X/U dimensions.")

        names: list[str] = []
        specs: list[tuple[str, Any]] = []

        if self.include_bias:
            names.append("1")
            specs.append(("bias", None))

        # State polynomials
        for deg in range(1, int(self.state_degree) + 1):
            for comb in combinations_with_replacement(range(n_x), deg):
                counts: dict[int, int] = {}
                for idx in comb:
                    counts[idx] = counts.get(idx, 0) + 1
                parts = [
                    s_names[idx] if p == 1 else f"{s_names[idx]}^{p}"
                    for idx, p in counts.items()
                ]
                names.append(" ".join(parts))
                specs.append(("state_poly", comb))

        # Spatial PDE derivatives of state x when spatiotemporal + dx given
        sp_ndim = X_arr.ndim - 2
        if sp_ndim >= 1 and self.dx is not None:
            coord_lbls = _coord_labels(sp_ndim)
            for order in self.derivative_orders:
                for ax_idx in range(sp_ndim):
                    deriv_tag = coord_lbls[ax_idx] * int(order)
                    for v_idx in range(n_x):
                        names.append(f"{s_names[v_idx]}_{deriv_tag}")
                        specs.append(("spatial_deriv", (v_idx, ax_idx, int(order))))

        # Exogenous control polynomials
        for deg in range(1, int(self.control_degree) + 1):
            for comb in combinations_with_replacement(range(n_u), deg):
                counts_u: dict[int, int] = {}
                for idx in comb:
                    counts_u[idx] = counts_u.get(idx, 0) + 1
                parts_u = [
                    c_names[idx] if p == 1 else f"{c_names[idx]}^{p}"
                    for idx, p in counts_u.items()
                ]
                names.append(" ".join(parts_u))
                specs.append(("control_poly", comb))

        # Bilinear state-control interactions x_i u_j
        if self.include_bilinear:
            for i in range(n_x):
                for j in range(n_u):
                    names.append(f"{s_names[i]} {c_names[j]}")
                    specs.append(("bilinear", (i, j)))

        if self.include_quadratic_cross and int(self.state_degree) >= 2:
            for comb in combinations_with_replacement(range(n_x), 2):
                counts_q: dict[int, int] = {}
                for idx in comb:
                    counts_q[idx] = counts_q.get(idx, 0) + 1
                s_part = " ".join(
                    s_names[idx] if p == 1 else f"{s_names[idx]}^{p}"
                    for idx, p in counts_q.items()
                )
                for j in range(n_u):
                    names.append(f"{s_part} {c_names[j]}")
                    specs.append(("quad_cross", (comb, j)))

        self.n_state_features_ = n_x
        self.n_control_features_ = n_u
        self.state_names_ = s_names
        self.control_names_ = c_names
        self.feature_names_ = names
        self._specs = specs
        self.n_output_features_ = len(names)
        return self

    def get_feature_names_out(self) -> list[str]:
        return list(self.feature_names_)

    def transform(
        self,
        X: np.ndarray,
        U: np.ndarray,
        *,
        weak_spatial: bool = False,
    ) -> np.ndarray:
        if not hasattr(self, "_specs"):
            raise RuntimeError("ControlLibrary must be fitted before calling transform.")
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, np.newaxis]
        U_arr = _broadcast_control(X_arr, U)

        sp_ndim = X_arr.ndim - 2
        sp_steps = (
            extract_spatial_steps(self.dx, sp_ndim)
            if (sp_ndim >= 1 and self.dx is not None)
            else ()
        )

        lead_shape = X_arr.shape[:-1]
        cols: list[np.ndarray] = []
        for kind, payload in self._specs:
            if kind == "bias":
                cols.append(np.ones(lead_shape, dtype=np.float64))
            elif kind == "state_poly":
                term = np.ones(lead_shape, dtype=np.float64)
                for idx in payload:
                    term = term * X_arr[..., idx]
                cols.append(term)
            elif kind == "spatial_deriv":
                v_idx, ax_idx, order = payload
                step = sp_steps[ax_idx]
                field_v = X_arr[..., v_idx]
                if weak_spatial:
                    d_field = _weak_mollified_derivative(
                        field_v, dx=step, order=order, axis=1 + ax_idx, boundary=self.boundary
                    )
                else:
                    d_field = spatial_finite_difference(
                        field_v, dx=step, order=order, axis=1 + ax_idx, boundary=self.boundary
                    )
                cols.append(d_field)
            elif kind == "control_poly":
                term = np.ones(lead_shape, dtype=np.float64)
                for idx in payload:
                    term = term * U_arr[..., idx]
                cols.append(term)
            elif kind == "bilinear":
                i, j = payload
                cols.append(X_arr[..., i] * U_arr[..., j])
            elif kind == "quad_cross":
                comb, j = payload
                term = U_arr[..., j].copy()
                for idx in comb:
                    term = term * X_arr[..., idx]
                cols.append(term)

        return np.stack(cols, axis=-1)

    def project_spatiotemporal_weak(
        self,
        X: np.ndarray,
        U: np.ndarray,
        *,
        Phi_t: np.ndarray,
        Phi_x0: list[np.ndarray],
        sp_wins: list[list[tuple[int, int]]],
        sp_steps: tuple[float, ...],
        sp_degree: int,
    ) -> np.ndarray:
        """Project ``Theta(x, nabla x, u)`` onto separable bump test functions with exact spatial IBP."""
        X_arr = np.asarray(X, dtype=np.float64)
        U_arr = _broadcast_control(X_arr, U)
        sp_ndim = X_arr.ndim - 2
        cols_2d: list[np.ndarray] = []

        # Cache differentiated spatial weight matrices for integration by parts
        deriv_weights_cache: dict[tuple[int, int], np.ndarray] = {}
        for kind, payload in self._specs:
            if kind == "spatial_deriv":
                v_idx, ax_idx, order = payload
                key = (ax_idx, order)
                if key not in deriv_weights_cache:
                    deriv_weights_cache[key] = build_spatial_weak_weights(
                        n_grid=X_arr.shape[1 + ax_idx],
                        windows=sp_wins[ax_idx],
                        dx=sp_steps[ax_idx],
                        degree=max(sp_degree, order + 2),
                        diff_order=order,
                    )
                weights = [Phi_t] + [
                    deriv_weights_cache[key] if d == ax_idx else Phi_x0[d]
                    for d in range(sp_ndim)
                ]
                proj = contract_tensor_weak(X_arr[..., v_idx : v_idx + 1], weights)
                cols_2d.append(proj[:, 0])
            else:
                if kind == "bias":
                    arr = np.ones(X_arr.shape[:-1], dtype=np.float64)
                elif kind == "state_poly":
                    arr = np.ones(X_arr.shape[:-1], dtype=np.float64)
                    for idx in payload:
                        arr = arr * X_arr[..., idx]
                elif kind == "control_poly":
                    arr = np.ones(X_arr.shape[:-1], dtype=np.float64)
                    for idx in payload:
                        arr = arr * U_arr[..., idx]
                elif kind == "bilinear":
                    i, j = payload
                    arr = X_arr[..., i] * U_arr[..., j]
                else:
                    comb, j = payload
                    arr = U_arr[..., j].copy()
                    for idx in comb:
                        arr = arr * X_arr[..., idx]
                proj = contract_tensor_weak(arr[..., np.newaxis], [Phi_t, *Phi_x0])
                cols_2d.append(proj[:, 0])

        return np.column_stack(cols_2d)


class WeakSINDyControl(BaseEstimator):
    """Weak-form sparse identification of controlled fractional ODEs and PDEs (``weakSINDYc``).

    Discovers governing equations driven by exogenous forcing or dosing inputs ``u(t, x)``:

    .. math::

        D_t^\\alpha \\mathbf{x} = \\Theta(\\mathbf{x}, \\nabla \\mathbf{x}, \\mathbf{u}) \\Xi
    """

    def __init__(
        self,
        library: ControlLibrary | None = None,
        state_degree: int = 2,
        control_degree: int = 1,
        include_bias: bool = False,
        include_bilinear: bool = True,
        optimizer: Literal["stlsq", "sr3"] | STLSQ | SR3 = "stlsq",
        threshold: float = 0.02,
        alpha_reg: float = 1e-12,
        orders: float | Sequence[float] = (1.0,),
        optimize_order: bool = False,
        order_bounds: tuple[float, float] = (0.15, 1.0),
        criterion: Literal["bic", "aic"] = "bic",
        use_weak: bool = True,
        window_size: int = 40,
        stride: int = 4,
        spatial_window_size: int | Sequence[int] | None = None,
        spatial_stride: int | Sequence[int] | None = None,
        test_degree: int = 4,
        k_start: int = 15,
        dx: float | Sequence[float] | None = None,
    ) -> None:
        self.library = library
        self.state_degree = state_degree
        self.control_degree = control_degree
        self.include_bias = include_bias
        self.include_bilinear = include_bilinear
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
        self.dx = dx

    @property
    def alpha_(self) -> float:
        return self.order_

    def _make_optimizer(self) -> STLSQ | SR3:
        if isinstance(self.optimizer, (STLSQ, SR3)):
            return self.optimizer.__class__(**self.optimizer.get_params())
        if self.optimizer == "stlsq":
            return STLSQ(threshold=self.threshold, alpha=self.alpha_reg)
        if self.optimizer == "sr3":
            return SR3(threshold=self.threshold)
        raise ValueError(f"Unsupported optimizer {self.optimizer!r}; expected 'stlsq' or 'sr3'.")

    def _make_library(self, eff_dx: float | Sequence[float] | None) -> ControlLibrary:
        if self.library is not None:
            return self.library
        return ControlLibrary(
            state_degree=self.state_degree,
            control_degree=self.control_degree,
            include_bias=self.include_bias,
            include_bilinear=self.include_bilinear,
            dx=eff_dx,
        )

    def fit(
        self,
        t: float | np.ndarray,
        X: np.ndarray,
        U: np.ndarray,
        *,
        orders: float | Sequence[float] | None = None,
        optimize_order: bool | None = None,
        order_bounds: tuple[float, float] | None = None,
        dx: float | Sequence[float] | None = None,
        state_names: Sequence[str] | None = None,
        control_names: Sequence[str] | None = None,
    ) -> WeakSINDyControl:
        """Identify sparse controlled fractional dynamics ``D_t^alpha x = Theta(x, nabla x, u) Xi``."""
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, np.newaxis]
        U_arr = _broadcast_control(X_arr, U)

        nt = X_arr.shape[0]
        dt = extract_time_step(t)
        if np.ndim(t) == 1 and np.asarray(t).size != nt:
            raise ValueError(f"Length of t ({np.asarray(t).size}) must match X.shape[0] ({nt}).")

        eff_dx = dx if dx is not None else self.dx
        if X_arr.ndim >= 3 and eff_dx is None and self.library is None:
            raise ValueError("Spatiotemporal PDE input (X.ndim >= 3) requires dx=...")

        lib = self._make_library(eff_dx)
        lib.fit(X_arr, U_arr, state_names=state_names, control_names=control_names)

        eff_k_start = min(int(self.k_start), max(0, nt // 5))
        n_vars = X_arr.shape[-1]
        sp_ndim = X_arr.ndim - 2

        # Precompute design matrix G and temporal/spatial projection structures
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
                eff_sp_dx = eff_dx if eff_dx is not None else lib.dx
                sp_steps = (
                    extract_spatial_steps(eff_sp_dx, sp_ndim)
                    if eff_sp_dx is not None
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
                X_design = lib.project_spatiotemporal_weak(
                    X_arr,
                    U_arr,
                    Phi_t=Phi_t,
                    Phi_x0=Phi_x0,
                    sp_wins=sp_wins,
                    sp_steps=sp_steps,
                    sp_degree=self.test_degree,
                )
                curr_sp = X_arr
                for d in range(sp_ndim):
                    c = np.tensordot(curr_sp, Phi_x0[d], axes=([1 + d], [0]))
                    curr_sp = np.moveaxis(c, -1, 1 + d)
                X_sp = curr_sp.reshape(nt, -1, n_vars)
                sp_gain = float(np.prod([float(np.mean(np.sum(w * w, axis=0))) for w in Phi_x0]))
                overlap = max(1.0, H / max(1, S))
            else:
                Theta_pt = lib.transform(X_arr, U_arr)
                X_design = Phi_t.T @ Theta_pt
                X_sp = None
                sp_gain = 1.0
                overlap = max(1.0, H / max(1, S)) if len(wins) >= 2 else 1.0

            def eval_target_and_gain(alpha_v: float) -> tuple[np.ndarray, float]:
                _, Psi_t = build_weak_weights(
                    n_steps=nt, windows=wins, alpha=alpha_v, phi=phi_t, dt=dt
                )
                t_gain = float(np.mean(np.sum(Psi_t * Psi_t, axis=0)))
                tot_gain = max(t_gain * sp_gain, 1e-30)
                if X_sp is not None:
                    proj = np.tensordot(Psi_t, X_sp, axes=([0], [0]))
                    return proj.reshape(-1, n_vars), tot_gain
                return Psi_t.T @ X_arr, tot_gain

        else:
            Theta_pt = lib.transform(X_arr, U_arr)
            X_design = Theta_pt[eff_k_start:nt].reshape(-1, Theta_pt.shape[-1])
            overlap = 1.0

            def eval_target_and_gain(alpha_v: float) -> tuple[np.ndarray, float]:
                X_dot = fractional_derivative_gl(X_arr, t=dt, alpha=alpha_v, axis=0)
                return X_dot[eff_k_start:nt].reshape(-1, n_vars), float(dt ** (-2.0 * float(alpha_v)))

        state_vars = np.array(
            [max(float(np.var(X_arr[..., d])), 1e-12) for d in range(n_vars)],
            dtype=np.float64,
        )
        n_indep = max(4.0, (X_design.shape[0] / max(1.0, overlap)) * n_vars)

        do_opt = self.optimize_order if optimize_order is None else bool(optimize_order)
        bounds = self.order_bounds if order_bounds is None else (float(order_bounds[0]), float(order_bounds[1]))
        raw_orders = self.orders if orders is None else orders
        cand_orders = (
            [float(raw_orders)] if np.ndim(raw_orders) == 0 else [float(a) for a in raw_orders]  # type: ignore[arg-type,union-attr]
        )

        cache_scores: dict[float, tuple[float, float, float, np.ndarray]] = {}

        def score_alpha(alpha_val: float) -> float:
            key = round(float(alpha_val), 6)
            if key not in cache_scores:
                Y_a, g_a = eval_target_and_gain(key)
                opt_a = self._make_optimizer().fit(X_design, Y_a)
                Y_hat = opt_a.predict(X_design)
                n_act = int(np.sum(opt_a.coef_ != 0.0))
                r2_a = _mean_r2_score(Y_a, Y_hat)
                if n_act == 0:
                    aic_a, bic_a = 1e12, 1e12
                else:
                    aic_a, bic_a = _compute_aic_bic(
                        Y_a, Y_hat, n_act, noise_gain=g_a, state_vars=state_vars, n_indep=n_indep
                    )
                cache_scores[key] = (aic_a, bic_a, r2_a, opt_a.coef_)
            aic_v, bic_v, _, _ = cache_scores[key]
            return bic_v if self.criterion == "bic" else aic_v

        if do_opt:
            a_min, a_max = float(bounds[0]), float(bounds[1])
            coarse = sorted({round(float(a), 6) for a in np.linspace(a_min, a_max, 21)})
            c_vals = [score_alpha(a) for a in coarse]
            best_idx = int(np.argmin(c_vals))
            h = (a_max - a_min) / 20.0
            b_lo = max(a_min, coarse[best_idx] - 1.5 * h)
            b_hi = min(a_max, coarse[best_idx] + 1.5 * h)
            res = minimize_scalar(score_alpha, bounds=(b_lo, b_hi), method="bounded", options={"xatol": 2e-4})
            score_alpha(float(res.x))
            best_alpha = min(
                cache_scores.keys(),
                key=lambda k: cache_scores[k][1 if self.criterion == "bic" else 0],
            )
        elif len(cand_orders) == 1:
            best_alpha = float(cand_orders[0])
            score_alpha(best_alpha)
        else:
            for a in cand_orders:
                score_alpha(a)
            best_alpha = min(
                cache_scores.keys(),
                key=lambda k: cache_scores[k][1 if self.criterion == "bic" else 0],
            )

        aic_best, bic_best, r2_best, coef_best = cache_scores[round(float(best_alpha), 6)]
        self.library_ = lib
        self.coef_ = coef_best
        self.ind_ = coef_best != 0.0
        self.order_ = float(best_alpha)
        self.aic_ = float(aic_best)
        self.bic_ = float(bic_best)
        self.r2_score_ = float(r2_best)
        self.feature_names_ = lib.get_feature_names_out()
        self.state_names_ = list(lib.state_names_)
        self.control_names_ = list(lib.control_names_)
        self.dt_ = dt
        return self

    def predict(self, t: float | np.ndarray, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Predict fractional time derivatives ``D_t^{alpha_} x`` given state ``X`` and control ``U``."""
        if not hasattr(self, "coef_"):
            raise RuntimeError("WeakSINDyControl must be fitted before calling predict.")
        _ = extract_time_step(t)
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, np.newaxis]
        Theta = self.library_.transform(X_arr, U)
        return Theta @ self.coef_.T

    def simulate(
        self,
        U: np.ndarray,
        x0: Sequence[float] | np.ndarray,
        t: np.ndarray,
        *,
        n_corrector: int = 2,
    ) -> np.ndarray:
        """Simulate the discovered controlled fractional ODE forward under exogenous input ``U(t)``."""
        if not hasattr(self, "coef_"):
            raise RuntimeError("WeakSINDyControl must be fitted before calling simulate.")
        t_arr = np.asarray(t, dtype=np.float64)
        dt = extract_time_step(t_arr)
        nt = t_arr.size
        u_arr = np.asarray(U, dtype=np.float64)
        if u_arr.ndim == 1:
            u_arr = u_arr[:, np.newaxis]
        if u_arr.shape[0] != nt:
            raise ValueError(f"U length ({u_arr.shape[0]}) must match len(t) ({nt}).")

        x0_arr = np.asarray(x0, dtype=np.float64).ravel()
        d = x0_arr.size
        w = gl_weights(self.order_, nt)
        dt_alpha = dt ** self.order_

        X_sim = np.zeros((nt, d), dtype=np.float64)
        X_sim[0] = x0_arr

        def eval_rhs(x_vec: np.ndarray, u_vec: np.ndarray) -> np.ndarray:
            th = self.library_.transform(x_vec[np.newaxis, :], u_vec[np.newaxis, :])
            return (th @ self.coef_.T)[0]

        for k in range(1, nt):
            hist = w[1 : k + 1] @ X_sim[k - 1 :: -1]
            f_prev = eval_rhs(X_sim[k - 1], u_arr[k - 1])
            xk = f_prev * dt_alpha - hist
            for _ in range(int(n_corrector)):
                f_curr = eval_rhs(xk, u_arr[k])
                xk = f_curr * dt_alpha - hist
            X_sim[k] = xk
        return X_sim

    def equations(self, precision: int = 4) -> list[str]:
        if not hasattr(self, "coef_"):
            raise RuntimeError("WeakSINDyControl must be fitted before calling equations.")
        eqs: list[str] = []
        lhs_op = "d/dt" if abs(self.order_ - 1.0) < 1e-6 else f"D_t^{self.order_:.3f}"
        for d_idx, s_name in enumerate(self.state_names_):
            terms: list[str] = []
            for j_idx, c_val in enumerate(self.coef_[d_idx]):
                if abs(c_val) > 0.0:
                    fname = self.feature_names_[j_idx]
                    terms.append(f"{c_val:+.{precision}f} {fname}")
            rhs = " ".join(terms).lstrip("+ ") if terms else "0"
            eqs.append(f"{lhs_op} {s_name} = {rhs}")
        return eqs

    def print(self, precision: int = 4) -> None:
        for line in self.equations(precision=precision):
            print(line)


# Canonical lowercase alias matching weakSINDYc naming convention
weakSINDYc = WeakSINDyControl


def generate_controlled_fractional_system(
    t: np.ndarray | None = None,
    *,
    alpha: float = 0.80,
    r_growth: float = 0.45,
    carrying_inv: float = 0.30,
    beta_rad: float = 0.85,
    gamma_pk: float = 0.50,
    k_elim: float = 0.60,
    b_infusion: float = 1.00,
    x0: Sequence[float] = (0.75, 0.05),
    snr_db: float | None = None,
    seed: int = 20260613,
) -> ControlledBenchmarkDataset:
    """Simulate a pulsed fractionated radiotherapy + pharmacokinetic fractional control benchmark.

    Governing coupled system (anomalous trapping order ``alpha in (0, 1]``):

    .. math::

        D_t^\\alpha x_1 &= r x_1 - \\kappa x_1^2 - \\beta_{\\text{rad}} x_1 u_1 - \\gamma_{\\text{pk}} x_1 x_2 \\\\
        D_t^\\alpha x_2 &= -k_e x_2 + b_{\\text{in}} u_2

    where:
      - ``x_1(t)``: normalized viable tumor burden,
      - ``x_2(t)``: intratumoral radiosensitizer / drug concentration,
      - ``u_1(t)``: pulsed fractionated radiation beam schedule (Gaussian-smoothed daily pulses),
      - ``u_2(t)``: staggered bolus/infusion pharmacokinetic dosing schedule.

    All stochastic measurement noise uses a strictly deterministic ``PCG64`` generator.
    """
    t_grid = np.linspace(0.0, 20.0, 801, dtype=np.float64) if t is None else np.asarray(t, dtype=np.float64)
    dt = extract_time_step(t_grid)
    nt = t_grid.size

    # Deterministic multi-fraction pulsed radiation u_1(t) and staggered PK infusion u_2(t)
    t_span = float(t_grid[-1] - t_grid[0])
    pulse_centers_rad = t_grid[0] + t_span * np.array([0.12, 0.28, 0.46, 0.64, 0.82])
    u1 = np.zeros(nt, dtype=np.float64)
    width_rad = max(0.022 * t_span, 2.5 * dt)
    for idx_p, tc in enumerate(pulse_centers_rad):
        amp = 1.0 + 0.25 * ((idx_p % 2) * 2 - 1)
        u1 += amp * np.exp(-0.5 * ((t_grid - tc) / width_rad) ** 2)

    pulse_centers_pk = t_grid[0] + t_span * np.array([0.08, 0.35, 0.60, 0.78])
    u2 = 0.15 * (1.0 + np.sin(2.0 * np.pi * (t_grid - t_grid[0]) / (0.45 * t_span)))
    width_pk = max(0.035 * t_span, 3.0 * dt)
    for tc in pulse_centers_pk:
        u2 += 0.9 * np.exp(-0.5 * ((t_grid - tc) / width_pk) ** 2)

    U = np.column_stack([u1, u2])

    w = gl_weights(alpha, nt)
    dt_alpha = dt ** float(alpha)
    X_clean = np.zeros((nt, 2), dtype=np.float64)
    X_clean[0] = np.asarray(x0, dtype=np.float64)

    def rhs_ctrl(x_vec: np.ndarray, u_vec: np.ndarray) -> np.ndarray:
        x1, x2 = float(x_vec[0]), float(x_vec[1])
        ur, ud = float(u_vec[0]), float(u_vec[1])
        dx1 = r_growth * x1 - carrying_inv * (x1 * x1) - beta_rad * x1 * ur - gamma_pk * x1 * x2
        dx2 = -k_elim * x2 + b_infusion * ud
        return np.array([dx1, dx2], dtype=np.float64)

    for k in range(1, nt):
        hist = w[1 : k + 1] @ X_clean[k - 1 :: -1]
        xk = rhs_ctrl(X_clean[k - 1], U[k - 1]) * dt_alpha - hist
        for _ in range(3):
            xk = rhs_ctrl(xk, U[k]) * dt_alpha - hist
        X_clean[k] = xk

    if snr_db is not None:
        rng = np.random.Generator(np.random.PCG64(int(seed)))
        X_obs = X_clean.copy()
        for ell in range(2):
            sig_var = max(float(np.var(X_clean[:, ell])), 1e-16)
            n_std = np.sqrt(sig_var * (10.0 ** (-float(snr_db) / 10.0)))
            X_obs[:, ell] += rng.normal(0.0, n_std, size=nt)
    else:
        X_obs = X_clean.copy()

    gt = {
        "alpha": float(alpha),
        "state_names": ["x1", "x2"],
        "control_names": ["u_rad", "u_drug"],
        "terms": {
            "x1": {
                "x1": float(r_growth),
                "x1^2": -float(carrying_inv),
                "x1 x2": -float(gamma_pk),
                "x1 u_rad": -float(beta_rad),
            },
            "x2": {
                "x2": -float(k_elim),
                "u_drug": float(b_infusion),
            },
        },
    }
    return ControlledBenchmarkDataset(
        X=X_obs,
        U=U,
        t=t_grid,
        ground_truth=gt,
        X_clean=X_clean,
    )
