"""Polynomial, trigonometric, rational, custom, and N-D weak PDE candidate basis libraries."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from itertools import combinations_with_replacement
from typing import Callable, Literal, Sequence

import numpy as np
from scipy.ndimage import convolve1d

from ._sklearn_compat import BaseEstimator, TransformerMixin
from .operators import (
    extract_spatial_steps,
    extract_time_step,
    fractional_derivative_gl,
    fractional_laplacian,
)
from .test_functions import (
    build_spatial_weak_weights,
    bump_test_function,
    contract_tensor_weak,
)


def _coord_labels(n_dims: int) -> tuple[str, ...]:
    if n_dims <= 3:
        return ("x", "y", "z")[:n_dims]
    return tuple(f"x{d}" for d in range(n_dims))


def spatial_finite_difference(
    X: np.ndarray,
    dx: float,
    order: int = 1,
    axis: int = 1,
    boundary: str = "one_sided",
) -> np.ndarray:
    """Compute second-order accurate spatial finite differences along ``axis`` (orders 1..4).

    Parameters
    ----------
    X : np.ndarray
        Input array with spatial dimension along ``axis``.
    dx : float
        Uniform spatial step size.
    order : int, default=1
        Derivative order in ``{1, 2, 3, 4}``.
    axis : int, default=1
        Spatial axis index.
    boundary : {"one_sided", "neumann", "periodic"}, default="one_sided"
        Boundary stencil treatment.
    """
    step = extract_time_step(dx)
    if order not in (1, 2, 3, 4):
        raise ValueError(f"Supported integer spatial derivative orders are 1, 2, 3, 4; got {order}.")
    if boundary not in ("one_sided", "neumann", "periodic"):
        raise ValueError(
            f"boundary must be 'one_sided', 'neumann', or 'periodic'; got {boundary!r}."
        )

    if order == 3:
        d2 = spatial_finite_difference(X, dx=step, order=2, axis=axis, boundary=boundary)
        return spatial_finite_difference(d2, dx=step, order=1, axis=axis, boundary=boundary)
    if order == 4:
        d2 = spatial_finite_difference(X, dx=step, order=2, axis=axis, boundary=boundary)
        return spatial_finite_difference(d2, dx=step, order=2, axis=axis, boundary=boundary)

    arr = np.asarray(X, dtype=np.float64)
    moved = np.moveaxis(arr, axis, 0)
    nx = moved.shape[0]
    if nx < 4:
        raise ValueError(f"Spatial axis must have at least 4 grid points; got {nx}.")

    out = np.zeros_like(moved)
    if order == 1:
        out[1:-1] = (moved[2:] - moved[:-2]) / (2.0 * step)
        if boundary == "one_sided":
            out[0] = (-1.5 * moved[0] + 2.0 * moved[1] - 0.5 * moved[2]) / step
            out[-1] = (1.5 * moved[-1] - 2.0 * moved[-2] + 0.5 * moved[-3]) / step
        elif boundary == "neumann":
            out[0] = 0.0
            out[-1] = 0.0
        elif boundary == "periodic":
            out[0] = (moved[1] - moved[-1]) / (2.0 * step)
            out[-1] = (moved[0] - moved[-2]) / (2.0 * step)
    else:
        dx2 = step * step
        out[1:-1] = (moved[2:] - 2.0 * moved[1:-1] + moved[:-2]) / dx2
        if boundary == "one_sided":
            out[0] = (2.0 * moved[0] - 5.0 * moved[1] + 4.0 * moved[2] - moved[3]) / dx2
            out[-1] = (2.0 * moved[-1] - 5.0 * moved[-2] + 4.0 * moved[-3] - moved[-4]) / dx2
        elif boundary == "neumann":
            out[0] = (2.0 * moved[1] - 2.0 * moved[0]) / dx2
            out[-1] = (2.0 * moved[-2] - 2.0 * moved[-1]) / dx2
        elif boundary == "periodic":
            out[0] = (moved[1] - 2.0 * moved[0] + moved[-1]) / dx2
            out[-1] = (moved[0] - 2.0 * moved[-1] + moved[-2]) / dx2

    return np.moveaxis(out, 0, axis)


def _weak_mollified_derivative(
    X: np.ndarray,
    dx: float,
    order: int,
    axis: int,
    boundary: str = "one_sided",
) -> np.ndarray:
    """Evaluate smooth compact-bump mollified derivative ``X * (-1)^m phi^(m)`` without FD blow-up."""
    n_ax = X.shape[axis]
    H = min(10, max(4, (n_ax - 2) // 3))
    if H % 2 == 1:
        H += 1
    if H >= n_ax:
        H = max(2, (n_ax - 1) // 2 * 2)
    deg = max(4, order + 2)
    phi_m = bump_test_function(window_size=H, degree=deg, dt=dx, normalize=True, diff_order=order)
    # Discrete convolution sum_j X[i - j] kernel[j] = sum_r X[i + r] (-1)^m phi^(m)(r) dx
    sign = -1.0 if (order % 2 == 1) else 1.0
    kernel = (sign * phi_m * dx)[::-1]
    mode = "wrap" if boundary == "periodic" else ("reflect" if boundary == "neumann" else "nearest")
    return convolve1d(X, kernel, axis=axis, mode=mode)


class BaseCandidateLibrary(BaseEstimator, TransformerMixin, ABC):
    """Abstract base class for weakSINDY candidate basis libraries."""

    n_features_in_: int
    n_output_features_: int
    _supports_spatiotemporal_weak: bool = False

    @staticmethod
    def _validate_input(X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        if arr.ndim < 2:
            raise ValueError(f"Expected >=2D state array (ODE or N-D PDE); got shape {arr.shape}.")
        return arr

    @staticmethod
    def _default_feature_names(n_features: int, input_features: Sequence[str] | None) -> list[str]:
        if input_features is None:
            return [f"x{i}" for i in range(n_features)]
        if len(input_features) != n_features:
            raise ValueError(
                f"Expected {n_features} input_features names; got {len(input_features)}."
            )
        return [str(name) for name in input_features]

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> BaseCandidateLibrary:
        """Fit library metadata from state array ``X``."""

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Evaluate candidate library terms on ``X``."""

    @abstractmethod
    def get_feature_names(self, input_features: Sequence[str] | None = None) -> list[str]:
        """Return human-readable symbolic names for all candidate columns."""


class PolynomialLibrary(BaseCandidateLibrary):
    """Multivariate polynomial candidate library up to ``degree``.

    Parameters
    ----------
    degree : int, default=3
        Maximum polynomial total degree ``>= 0``.
    include_bias : bool, default=True
        If ``True``, includes the constant column ``1``.
    include_interaction : bool, default=True
        If ``False``, retains only pure powers ``x_i^k`` without cross-terms.
    """

    def __init__(
        self,
        degree: int = 3,
        include_bias: bool = True,
        include_interaction: bool = True,
    ) -> None:
        self.degree = degree
        self.include_bias = include_bias
        self.include_interaction = include_interaction

    def _build_combinations(self, n_features: int) -> list[tuple[int, ...]]:
        if self.degree < 0:
            raise ValueError(f"Polynomial degree must be >= 0; got {self.degree}.")
        combos: list[tuple[int, ...]] = []
        if self.include_bias:
            combos.append(())
        for deg in range(1, self.degree + 1):
            if self.include_interaction:
                combos.extend(combinations_with_replacement(range(n_features), deg))
            else:
                for idx in range(n_features):
                    combos.append((idx,) * deg)
        return combos

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> PolynomialLibrary:
        arr = self._validate_input(X)
        self.n_features_in_ = arr.shape[-1]
        self._combinations_ = self._build_combinations(self.n_features_in_)
        self.n_output_features_ = len(self._combinations_)
        return self

    def get_feature_names(self, input_features: Sequence[str] | None = None) -> list[str]:
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("PolynomialLibrary must be fitted before calling get_feature_names.")
        base = self._default_feature_names(self.n_features_in_, input_features)
        names: list[str] = []
        for comb in self._combinations_:
            if len(comb) == 0:
                names.append("1")
                continue
            counts: dict[int, int] = {}
            for idx in comb:
                counts[idx] = counts.get(idx, 0) + 1
            parts = [
                base[idx] if exp == 1 else f"{base[idx]}^{exp}" for idx, exp in counts.items()
            ]
            names.append(" ".join(parts))
        return names

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("PolynomialLibrary must be fitted before calling transform.")
        arr = self._validate_input(X)
        if arr.shape[-1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} state variables; got {arr.shape[-1]}."
            )
        lead_shape = arr.shape[:-1]
        out = np.empty((*lead_shape, self.n_output_features_), dtype=np.float64)
        for col_idx, comb in enumerate(self._combinations_):
            if len(comb) == 0:
                out[..., col_idx] = 1.0
            else:
                prod = arr[..., comb[0]].copy()
                for v_idx in comb[1:]:
                    prod *= arr[..., v_idx]
                out[..., col_idx] = prod
        return out


class FourierLibrary(BaseCandidateLibrary):
    """Trigonometric Fourier candidate library ``sin(freq * x_i)``, ``cos(freq * x_i)``.

    Parameters
    ----------
    frequencies : sequence of float, default=(1.0,)
        Harmonic frequencies to evaluate on each state variable.
    include_sin : bool, default=True
        Include ``sin(w * x_i)`` basis functions.
    include_cos : bool, default=True
        Include ``cos(w * x_i)`` basis functions.
    """

    def __init__(
        self,
        frequencies: Sequence[float] = (1.0,),
        include_sin: bool = True,
        include_cos: bool = True,
    ) -> None:
        self.frequencies = frequencies
        self.include_sin = include_sin
        self.include_cos = include_cos

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> FourierLibrary:
        arr = self._validate_input(X)
        if not self.include_sin and not self.include_cos:
            raise ValueError("At least one of include_sin or include_cos must be True.")
        freqs = [float(f) for f in self.frequencies]
        if len(freqs) == 0:
            raise ValueError("frequencies must be non-empty.")
        self._freqs_ = freqs
        self.n_features_in_ = arr.shape[-1]
        n_per_freq = int(bool(self.include_sin)) + int(bool(self.include_cos))
        self.n_output_features_ = len(freqs) * self.n_features_in_ * n_per_freq
        return self

    def get_feature_names(self, input_features: Sequence[str] | None = None) -> list[str]:
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("FourierLibrary must be fitted before calling get_feature_names.")
        base = self._default_feature_names(self.n_features_in_, input_features)
        names: list[str] = []
        for freq in self._freqs_:
            prefix = "" if np.isclose(freq, 1.0) else f"{freq:g}*"
            for f_name in base:
                if self.include_sin:
                    names.append(f"sin({prefix}{f_name})")
                if self.include_cos:
                    names.append(f"cos({prefix}{f_name})")
        return names

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("FourierLibrary must be fitted before calling transform.")
        arr = self._validate_input(X)
        if arr.shape[-1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} state variables; got {arr.shape[-1]}."
            )
        cols: list[np.ndarray] = []
        for freq in self._freqs_:
            for v_idx in range(self.n_features_in_):
                scaled = freq * arr[..., v_idx : v_idx + 1]
                if self.include_sin:
                    cols.append(np.sin(scaled))
                if self.include_cos:
                    cols.append(np.cos(scaled))
        return np.concatenate(cols, axis=-1)


class RationalLibrary(BaseCandidateLibrary):
    """Rational, Michaelis–Menten, and Hill saturation candidate basis library.

    Generates well-conditioned rational saturation basis functions ``x_i^h / (k + x_i^h)``
    across Hill exponents ``h`` and half-saturation constants ``k``, with optional pure
    reciprocal terms ``1 / (k + x_i^h)`` and cross-state modulation terms
    ``x_j^p * x_i^h / (k + x_i^h)``.

    Parameters
    ----------
    numerator_degree : int, default=1
        Controls cross-state polynomial modulation degree ``p <= numerator_degree - 1``
        when ``include_cross_terms=True`` or ``numerator_degree >= 2``.
    denominator_degree : int, default=1
        Maximum Hill exponent ``h`` when ``hill_exponents`` is ``None``.
    half_saturation : float or sequence of float, default=(1.0,)
        Positive half-saturation constants ``k`` in ``(k + x_i^h)``.
    hill_exponents : sequence of int or None, default=None
        Explicit Hill exponents ``h >= 1``. Defaults to ``range(1, denominator_degree + 1)``.
    include_cross_terms : bool, default=False
        Include cross-state products ``x_j * x_i^h / (k + x_i^h)`` for ``j != i``.
    include_reciprocal : bool, default=False
        Include pure inhibition/reciprocal terms ``1 / (k + x_i^h)``.
    denom_floor : float, default=1e-6
        Minimum absolute denominator magnitude to prevent division singularities on noisy data.
    """

    def __init__(
        self,
        numerator_degree: int = 1,
        denominator_degree: int = 1,
        half_saturation: float | Sequence[float] = (1.0,),
        hill_exponents: Sequence[int] | None = None,
        include_cross_terms: bool = False,
        include_reciprocal: bool = False,
        denom_floor: float = 1e-6,
    ) -> None:
        self.numerator_degree = numerator_degree
        self.denominator_degree = denominator_degree
        self.half_saturation = half_saturation
        self.hill_exponents = hill_exponents
        self.include_cross_terms = include_cross_terms
        self.include_reciprocal = include_reciprocal
        self.denom_floor = denom_floor

    def _safe_denom(self, raw_denom: np.ndarray) -> np.ndarray:
        floor = float(self.denom_floor)
        signs = np.where(raw_denom >= 0.0, 1.0, -1.0)
        return np.where(np.abs(raw_denom) < floor, signs * floor, raw_denom)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> RationalLibrary:
        arr = self._validate_input(X)
        if self.denominator_degree < 1:
            raise ValueError(f"denominator_degree must be >= 1; got {self.denominator_degree}.")
        if self.numerator_degree < 0:
            raise ValueError(f"numerator_degree must be >= 0; got {self.numerator_degree}.")

        if np.ndim(self.half_saturation) == 0:
            ks = [float(self.half_saturation)]  # type: ignore[arg-type]
        else:
            ks = [float(k) for k in self.half_saturation]  # type: ignore[union-attr]
        if len(ks) == 0 or any(k <= 0.0 for k in ks):
            raise ValueError("half_saturation constants must be non-empty and strictly positive.")

        if self.hill_exponents is None:
            hills = list(range(1, int(self.denominator_degree) + 1))
        else:
            hills = [int(h) for h in self.hill_exponents]
        if len(hills) == 0 or any(h < 1 for h in hills):
            raise ValueError("hill_exponents must be non-empty positive integers.")

        self._ks_ = ks
        self._hills_ = hills
        self.n_features_in_ = arr.shape[-1]
        self.n_output_features_ = len(self.get_feature_names())
        return self

    def get_feature_names(self, input_features: Sequence[str] | None = None) -> list[str]:
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("RationalLibrary must be fitted before calling get_feature_names.")
        base = self._default_feature_names(self.n_features_in_, input_features)
        names: list[str] = []
        cross_max = max(int(self.numerator_degree) - 1, 1 if self.include_cross_terms else 0)
        for k_val in self._ks_:
            k_str = f"{k_val:g}"
            for h in self._hills_:
                for i, f_i in enumerate(base):
                    num_str = f_i if h == 1 else f"{f_i}^{h}"
                    denom_str = f"({k_str}+{num_str})"
                    if self.include_reciprocal:
                        names.append(f"1/{denom_str}")
                    if self.numerator_degree >= 1:
                        hill_term = f"{num_str}/{denom_str}"
                        names.append(hill_term)
                        if cross_max >= 1:
                            for j, f_j in enumerate(base):
                                if j != i:
                                    for p in range(1, cross_max + 1):
                                        mod = f_j if p == 1 else f"{f_j}^{p}"
                                        names.append(f"{mod}*{hill_term}")
        return names

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("RationalLibrary must be fitted before calling transform.")
        arr = self._validate_input(X)
        if arr.shape[-1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} state variables; got {arr.shape[-1]}."
            )
        cols: list[np.ndarray] = []
        cross_max = max(int(self.numerator_degree) - 1, 1 if self.include_cross_terms else 0)
        for k_val in self._ks_:
            for h in self._hills_:
                for i in range(self.n_features_in_):
                    x_i = arr[..., i : i + 1]
                    x_pow = x_i ** h if h > 1 else x_i
                    # For odd h on biochemical states, avoid singularities near -k_val^(1/h)
                    denom = self._safe_denom(k_val + x_pow)
                    if self.include_reciprocal:
                        cols.append(1.0 / denom)
                    if self.numerator_degree >= 1:
                        hill_col = x_pow / denom
                        cols.append(hill_col)
                        if cross_max >= 1:
                            for j in range(self.n_features_in_):
                                if j != i:
                                    x_j = arr[..., j : j + 1]
                                    for p in range(1, cross_max + 1):
                                        cols.append((x_j**p) * hill_col)
        return np.concatenate(cols, axis=-1)


class PDELibrary(BaseCandidateLibrary):
    """N-Dimensional PDE candidate library with weak-form tensor integration by parts.

    Supports 1D, 2D, and 3D (and general ``D``-dimensional) spatiotemporal fields
    ``(N_t, *spatial_dims, n_vars)``. Generates polynomial reaction terms, directional
    spatial derivatives (``nabla``, ``u_xx``, ..., ``u_xxxx``), isotropic Laplacian
    ``nabla^2``, biharmonic ``nabla^4``, advection ``u * nabla`` / ``v . nabla``,
    causal Grünwald–Letnikov spatial fractional derivatives ``D_{x_d}^beta``, symmetric
    Riesz / spectral fractional diffusion ``(-Delta)^(beta/2)``, and Michaelis–Menten
    saturation terms.

    When projected via ``project_weak_form`` or ``WeakSINDy(use_weak=True)``, all spatial
    differential operators are transferred strictly onto separable ``C^p`` test functions
    ``Phi(t, x) = phi_t(t) prod_d phi_{x_d}(x_d)`` via integration by parts without
    applying finite differences to noisy state measurements.
    """

    _supports_spatiotemporal_weak: bool = True

    def __init__(
        self,
        dx: float | np.ndarray | Sequence[float | np.ndarray],
        poly_degree: int = 2,
        derivative_orders: Sequence[int] = (1, 2),
        fractional_orders: Sequence[float] = (),
        riesz_orders: Sequence[float] = (),
        include_bias: bool = True,
        include_interaction: bool = False,
        include_michaelis_menten: bool = False,
        include_laplacian: bool = False,
        include_biharmonic: bool = False,
        include_advection: bool = False,
        advection_velocity: Sequence[float | np.ndarray] | np.ndarray | None = None,
        riesz_method: Literal["riesz", "spectral"] = "riesz",
        boundary: str = "one_sided",
        spatial_ndim: int | None = None,
    ) -> None:
        self.dx = dx
        self.poly_degree = poly_degree
        self.derivative_orders = derivative_orders
        self.fractional_orders = fractional_orders
        self.riesz_orders = riesz_orders
        self.include_bias = include_bias
        self.include_interaction = include_interaction
        self.include_michaelis_menten = include_michaelis_menten
        self.include_laplacian = include_laplacian
        self.include_biharmonic = include_biharmonic
        self.include_advection = include_advection
        self.advection_velocity = advection_velocity
        self.riesz_method = riesz_method
        self.boundary = boundary
        self.spatial_ndim = spatial_ndim

    def _infer_spatial_ndim(self, arr: np.ndarray) -> int:
        if self.spatial_ndim is not None:
            return int(self.spatial_ndim)
        if isinstance(self.dx, (list, tuple)) and len(self.dx) > 1:
            return len(self.dx)
        if arr.ndim == 2:
            return 1
        return arr.ndim - 2

    def _get_spatial_steps(self, n_dims: int | None = None) -> tuple[float, ...]:
        eff_dims = self.spatial_ndim_ if n_dims is None else int(n_dims)
        return extract_spatial_steps(self.dx, eff_dims)

    def _resolve_advection_velocity(
        self, sp_shape: tuple[int, ...], steps: tuple[float, ...]
    ) -> tuple[list[np.ndarray | float], np.ndarray | float]:
        assert self.advection_velocity is not None
        D = len(sp_shape)
        vel_raw = self.advection_velocity
        if np.ndim(vel_raw) == 1 and len(vel_raw) == D and all(np.ndim(v) == 0 for v in vel_raw):
            comps: list[np.ndarray | float] = [float(v) for v in vel_raw]
            return comps, 0.0
        if len(vel_raw) != D:
            raise ValueError(
                f"advection_velocity must provide {D} spatial velocity components; got {len(vel_raw)}."
            )
        comps = []
        div_total: np.ndarray | float = 0.0
        for d, v_comp in enumerate(vel_raw):
            if np.ndim(v_comp) == 0:
                comps.append(float(v_comp))  # type: ignore[arg-type]
            else:
                v_arr = np.asarray(v_comp, dtype=np.float64)
                if v_arr.shape != sp_shape:
                    raise ValueError(
                        f"Velocity component {d} shape {v_arr.shape} must match spatial shape {sp_shape}."
                    )
                comps.append(v_arr[np.newaxis, ..., np.newaxis])
                dv_d = spatial_finite_difference(
                    v_arr[..., np.newaxis],
                    dx=steps[d],
                    order=1,
                    axis=d,
                    boundary=self.boundary,
                )[np.newaxis, ...]
                div_total = div_total + dv_d
        return comps, div_total

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> PDELibrary:
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim < 2:
            raise ValueError(
                f"PDELibrary expects (*spatial_dims, n_vars) or (nt, *spatial_dims, n_vars); got {arr.shape}."
            )
        sp_ndim = self._infer_spatial_ndim(arr)
        if sp_ndim < 1 or arr.ndim not in (sp_ndim + 1, sp_ndim + 2):
            raise ValueError(
                f"Incompatible input shape {arr.shape} for spatial_ndim={sp_ndim}."
            )
        self.spatial_ndim_ = sp_ndim
        self.spatial_steps_ = extract_spatial_steps(self.dx, sp_ndim)
        self.n_features_in_ = arr.shape[-1]
        for d_ord in self.derivative_orders:
            if int(d_ord) not in (1, 2, 3, 4):
                raise ValueError(
                    f"PDELibrary derivative_orders must be in {{1, 2, 3, 4}}; got {d_ord}."
                )
        sample_arr = arr if arr.ndim >= 3 else arr[np.newaxis, ...]
        self._poly_lib_ = PolynomialLibrary(
            degree=self.poly_degree,
            include_bias=self.include_bias,
            include_interaction=True,
        ).fit(sample_arr)
        self.n_output_features_ = len(self.get_feature_names())
        return self

    def get_feature_names(self, input_features: Sequence[str] | None = None) -> list[str]:
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("PDELibrary must be fitted before calling get_feature_names.")
        base = self._default_feature_names(self.n_features_in_, input_features)
        coords = _coord_labels(self.spatial_ndim_)
        names = list(self._poly_lib_.get_feature_names(base))

        deriv_names: list[str] = []
        for f_name in base:
            for d_ord in self.derivative_orders:
                for coord in coords:
                    suffix = coord * int(d_ord)
                    deriv_names.append(f"{f_name}_{suffix}")

        if self.include_laplacian:
            for f_name in base:
                deriv_names.append(f"laplacian({f_name})")

        if self.include_biharmonic:
            for f_name in base:
                deriv_names.append(f"biharmonic({f_name})")

        if self.advection_velocity is not None:
            for f_name in base:
                deriv_names.append(f"v.grad({f_name})")

        if self.include_advection:
            for f_name in base:
                for coord in coords:
                    deriv_names.append(f"{f_name}*{f_name}_{coord}")

        for beta in self.fractional_orders:
            beta_str = f"{float(beta):g}".replace(".", "_")
            for coord in coords:
                for f_name in base:
                    deriv_names.append(f"D_{coord}^{beta_str} {f_name}")

        for beta in self.riesz_orders:
            half_str = f"{0.5 * float(beta):g}"
            for f_name in base:
                deriv_names.append(f"(-Delta)^({half_str}) {f_name}")

        names.extend(deriv_names)

        if self.include_interaction:
            for f_name in base:
                for d_name in deriv_names:
                    names.append(f"{f_name}*{d_name}")

        if self.include_michaelis_menten:
            for i, f_i in enumerate(base):
                mm_term = f"{f_i}/(1+{f_i})"
                names.append(mm_term)
                for j, f_j in enumerate(base):
                    if j != i:
                        names.append(f"{f_j}*{mm_term}")
                        names.append(f"{f_j}^2*{mm_term}")

        return names

    def _evaluate_michaelis_menten(self, arr_nd: np.ndarray) -> list[np.ndarray]:
        mm_cols: list[np.ndarray] = []
        for i in range(self.n_features_in_):
            u_i = np.clip(arr_nd[..., i : i + 1], -0.99, None)
            mm_col = u_i / (1.0 + u_i)
            mm_cols.append(mm_col)
            for j in range(self.n_features_in_):
                if j != i:
                    u_j = arr_nd[..., j : j + 1]
                    mm_cols.append(u_j * mm_col)
                    mm_cols.append((u_j * u_j) * mm_col)
        return mm_cols

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("PDELibrary must be fitted before calling transform.")
        arr = np.asarray(X, dtype=np.float64)
        D = self.spatial_ndim_
        was_snapshot = arr.ndim == D + 1
        if was_snapshot:
            arr_nd = arr[np.newaxis, ...]
        elif arr.ndim == D + 2:
            arr_nd = arr
        else:
            raise ValueError(
                f"Expected {D + 1}D or {D + 2}D array for {D}D PDELibrary; got shape {arr.shape}."
            )

        if arr_nd.shape[-1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} state variables; got {arr_nd.shape[-1]}."
            )

        steps = self.spatial_steps_
        sp_axes = tuple(range(1, 1 + D))
        cols: list[np.ndarray] = [self._poly_lib_.transform(arr_nd)]

        # Cache directional integer derivatives (order, axis_idx) -> array
        needed_orders = set(int(o) for o in self.derivative_orders)
        if self.include_laplacian or self.include_biharmonic:
            needed_orders.add(2)
        if self.include_biharmonic:
            needed_orders.add(4)
        if self.include_advection or self.advection_velocity is not None:
            needed_orders.add(1)

        deriv_cache: dict[tuple[int, int], np.ndarray] = {}
        for d_ord in sorted(needed_orders):
            for d in range(D):
                deriv_cache[(d_ord, d)] = spatial_finite_difference(
                    arr_nd, dx=steps[d], order=d_ord, axis=1 + d, boundary=self.boundary
                )

        deriv_cols: list[np.ndarray] = []
        for v_idx in range(self.n_features_in_):
            for d_ord in self.derivative_orders:
                for d in range(D):
                    deriv_cols.append(deriv_cache[(int(d_ord), d)][..., v_idx : v_idx + 1])

        if self.include_laplacian:
            lap_all = sum(deriv_cache[(2, d)] for d in range(D))
            for v_idx in range(self.n_features_in_):
                deriv_cols.append(lap_all[..., v_idx : v_idx + 1])  # type: ignore[index]

        if self.include_biharmonic:
            bih_all = sum(deriv_cache[(4, d)] for d in range(D))
            for d1 in range(D):
                for d2 in range(d1 + 1, D):
                    mixed = spatial_finite_difference(
                        deriv_cache[(2, d1)],
                        dx=steps[d2],
                        order=2,
                        axis=1 + d2,
                        boundary=self.boundary,
                    )
                    bih_all = bih_all + 2.0 * mixed
            for v_idx in range(self.n_features_in_):
                deriv_cols.append(bih_all[..., v_idx : v_idx + 1])  # type: ignore[index]

        if self.advection_velocity is not None:
            vel_comps, _ = self._resolve_advection_velocity(arr_nd.shape[1:-1], steps)
            adv_v = sum(vel_comps[d] * deriv_cache[(1, d)] for d in range(D))
            for v_idx in range(self.n_features_in_):
                deriv_cols.append(adv_v[..., v_idx : v_idx + 1])  # type: ignore[index]

        if self.include_advection:
            for v_idx in range(self.n_features_in_):
                u_col = arr_nd[..., v_idx : v_idx + 1]
                for d in range(D):
                    deriv_cols.append(u_col * deriv_cache[(1, d)][..., v_idx : v_idx + 1])

        for beta in self.fractional_orders:
            for d in range(D):
                fd_arr = fractional_derivative_gl(arr_nd, t=steps[d], alpha=float(beta), axis=1 + d)
                for v_idx in range(self.n_features_in_):
                    deriv_cols.append(fd_arr[..., v_idx : v_idx + 1])

        for beta in self.riesz_orders:
            riesz_arr = fractional_laplacian(
                arr_nd,
                dx=steps,
                beta=float(beta),
                spatial_axes=sp_axes,
                method=self.riesz_method,
            )
            for v_idx in range(self.n_features_in_):
                deriv_cols.append(riesz_arr[..., v_idx : v_idx + 1])

        if deriv_cols:
            cols.extend(deriv_cols)

        if self.include_interaction and deriv_cols:
            for v_idx in range(self.n_features_in_):
                state_col = arr_nd[..., v_idx : v_idx + 1]
                for d_col in deriv_cols:
                    cols.append(state_col * d_col)

        if self.include_michaelis_menten:
            cols.extend(self._evaluate_michaelis_menten(arr_nd))

        out = np.concatenate(cols, axis=-1)
        return out[0] if was_snapshot else out

    def _project_spatiotemporal_weak(
        self,
        X_arr: np.ndarray,
        *,
        Phi_t: np.ndarray,
        Phi_x0: list[np.ndarray],
        sp_wins: list[list[tuple[int, int]]],
        sp_steps: tuple[float, ...],
        sp_degree: int,
    ) -> np.ndarray:
        """Project all PDE basis columns via exact weak integration by parts onto ``Phi(t, x)``."""
        D = self.spatial_ndim_
        sp_shape = X_arr.shape[1:-1]
        W_base = [Phi_t, *Phi_x0]

        weak_blocks: list[np.ndarray] = []

        # 1. Polynomial local algebraic features contracted with smooth 0th-order tensor bump
        poly_eval = self._poly_lib_.transform(X_arr)
        weak_blocks.append(contract_tensor_weak(poly_eval, W_base))

        # Precompute 1D spatial derivative test-function matrices needed
        needed_orders = set(int(o) for o in self.derivative_orders)
        if self.include_laplacian or self.include_biharmonic:
            needed_orders.add(2)
        if self.include_biharmonic:
            needed_orders.add(4)
        if self.include_advection or self.advection_velocity is not None or self.include_interaction:
            needed_orders.add(1)

        Phi_xd: dict[tuple[int, int], np.ndarray] = {}
        for d_ord in sorted(needed_orders):
            for d in range(D):
                Phi_xd[(d_ord, d)] = build_spatial_weak_weights(
                    n_grid=sp_shape[d],
                    windows=sp_wins[d],
                    dx=sp_steps[d],
                    degree=sp_degree,
                    diff_order=d_ord,
                )

        # Contract state tensor X_arr once per (order, axis) pair for reuse across variables
        state_deriv_proj: dict[tuple[int, int], np.ndarray] = {}
        for d_ord in sorted(needed_orders):
            if d_ord in self.derivative_orders or (d_ord == 2 and (self.include_laplacian or self.include_biharmonic)) or (d_ord == 4 and self.include_biharmonic):
                for d in range(D):
                    W_op = list(W_base)
                    W_op[1 + d] = Phi_xd[(d_ord, d)]
                    state_deriv_proj[(d_ord, d)] = contract_tensor_weak(X_arr, W_op)

        deriv_weak_cols: list[np.ndarray] = []
        deriv_specs: list[tuple[str, int, int, float]] = []

        # 2. Directional integer partial derivatives
        for v_idx in range(self.n_features_in_):
            for d_ord in self.derivative_orders:
                for d in range(D):
                    col = state_deriv_proj[(int(d_ord), d)][:, v_idx : v_idx + 1]
                    deriv_weak_cols.append(col)
                    deriv_specs.append(("int", v_idx, d, float(d_ord)))

        # 3. Isotropic Laplacian nabla^2 u
        if self.include_laplacian:
            lap_proj = sum(state_deriv_proj[(2, d)] for d in range(D))
            for v_idx in range(self.n_features_in_):
                deriv_weak_cols.append(lap_proj[:, v_idx : v_idx + 1])  # type: ignore[index]
                deriv_specs.append(("lap", v_idx, 0, 2.0))

        # 4. Isotropic Biharmonic nabla^4 u = sum_d d_x_d^4 u + 2 sum_{d1 < d2} d_{x_d1}^2 d_{x_d2}^2 u
        if self.include_biharmonic:
            bih_proj = sum(state_deriv_proj[(4, d)] for d in range(D))
            for d1 in range(D):
                for d2 in range(d1 + 1, D):
                    W_mix = list(W_base)
                    W_mix[1 + d1] = Phi_xd[(2, d1)]
                    W_mix[1 + d2] = Phi_xd[(2, d2)]
                    bih_proj = bih_proj + 2.0 * contract_tensor_weak(X_arr, W_mix)
            for v_idx in range(self.n_features_in_):
                deriv_weak_cols.append(bih_proj[:, v_idx : v_idx + 1])  # type: ignore[index]
                deriv_specs.append(("bih", v_idx, 0, 4.0))

        # 5a. Prescribed velocity advection (v . nabla) u -> -sum_d <v_d u, d_{x_d} Phi> - <(div v) u, Phi>
        if self.advection_velocity is not None:
            vel_comps, div_v = self._resolve_advection_velocity(sp_shape, sp_steps)
            adv_proj = np.zeros((Phi_t.shape[1] * int(np.prod([w.shape[1] for w in Phi_x0])), self.n_features_in_), dtype=np.float64)
            for d in range(D):
                W_adv = list(W_base)
                W_adv[1 + d] = Phi_xd[(1, d)]
                adv_proj += contract_tensor_weak(vel_comps[d] * X_arr, W_adv)
            if not (np.ndim(div_v) == 0 and float(div_v) == 0.0):  # type: ignore[arg-type]
                adv_proj -= contract_tensor_weak(div_v * X_arr, W_base)
            for v_idx in range(self.n_features_in_):
                deriv_weak_cols.append(adv_proj[:, v_idx : v_idx + 1])
                deriv_specs.append(("v_adv", v_idx, 0, 1.0))

        # 5b. Nonlinear Burgers self-advection u_i * d_{x_d} u_i = 0.5 * d_{x_d}(u_i^2)
        if self.include_advection:
            half_X2 = 0.5 * (X_arr**2)
            burgers_by_d = {}
            for d in range(D):
                W_adv = list(W_base)
                W_adv[1 + d] = Phi_xd[(1, d)]
                burgers_by_d[d] = contract_tensor_weak(half_X2, W_adv)
            for v_idx in range(self.n_features_in_):
                for d in range(D):
                    deriv_weak_cols.append(burgers_by_d[d][:, v_idx : v_idx + 1])
                    deriv_specs.append(("burgers", v_idx, d, 1.0))

        # 6. Causal spatial Grünwald–Letnikov fractional derivatives D_{x_d}^beta u
        for beta in self.fractional_orders:
            beta_f = float(beta)
            for d in range(D):
                Psi_gl_d = build_spatial_weak_weights(
                    n_grid=sp_shape[d],
                    windows=sp_wins[d],
                    dx=sp_steps[d],
                    degree=sp_degree,
                    fractional_order=beta_f,
                )
                W_gl = list(W_base)
                W_gl[1 + d] = Psi_gl_d
                gl_proj = contract_tensor_weak(X_arr, W_gl)
                for v_idx in range(self.n_features_in_):
                    deriv_weak_cols.append(gl_proj[:, v_idx : v_idx + 1])
                    deriv_specs.append(("gl", v_idx, d, beta_f))

        # 7. Symmetric Riesz / spectral spatial fractional diffusion (-Delta)^(beta/2) u
        if len(self.riesz_orders) > 0:
            if self.riesz_method == "riesz":
                for beta in self.riesz_orders:
                    beta_f = float(beta)
                    riesz_proj = np.zeros(
                        (Phi_t.shape[1] * int(np.prod([w.shape[1] for w in Phi_x0])), self.n_features_in_),
                        dtype=np.float64,
                    )
                    for d in range(D):
                        Psi_riesz_d = build_spatial_weak_weights(
                            n_grid=sp_shape[d],
                            windows=sp_wins[d],
                            dx=sp_steps[d],
                            degree=sp_degree,
                            riesz_order=beta_f,
                        )
                        W_riesz = list(W_base)
                        W_riesz[1 + d] = Psi_riesz_d
                        riesz_proj += contract_tensor_weak(X_arr, W_riesz)
                    for v_idx in range(self.n_features_in_):
                        deriv_weak_cols.append(riesz_proj[:, v_idx : v_idx + 1])
                        deriv_specs.append(("riesz", v_idx, 0, beta_f))
            else:
                # Exact Parseval-separable spectral adjoint on Phi_x0 in k-space
                U_t = np.moveaxis(np.tensordot(X_arr, Phi_t, axes=([0], [0])), -1, 0)
                sp_axes_ut = tuple(range(1, 1 + D))
                U_hat_conj = np.conj(np.fft.fftn(U_t, axes=sp_axes_ut))
                Phi_x_hat = [np.fft.fft(Phi_x0[d], axis=0) for d in range(D)]
                k_sq = np.zeros(sp_shape, dtype=np.float64)
                for d in range(D):
                    freqs = 2.0 * np.pi * np.fft.fftfreq(sp_shape[d], d=sp_steps[d])
                    shp = [1] * D
                    shp[d] = sp_shape[d]
                    k_sq = k_sq + (freqs.reshape(shp) ** 2)
                inv_n_prod = 1.0 / float(np.prod(sp_shape))
                for beta in self.riesz_orders:
                    beta_f = float(beta)
                    sym = (k_sq ** (0.5 * beta_f))[np.newaxis, ..., np.newaxis]
                    weighted_spec = U_hat_conj * sym * inv_n_prod
                    curr_c = weighted_spec
                    for d in range(D):
                        contracted = np.tensordot(curr_c, Phi_x_hat[d], axes=([1 + d], [0]))
                        curr_c = np.moveaxis(contracted, -1, 1 + d)
                    spec_proj = curr_c.real.reshape(-1, self.n_features_in_)
                    for v_idx in range(self.n_features_in_):
                        deriv_weak_cols.append(spec_proj[:, v_idx : v_idx + 1])
                        deriv_specs.append(("riesz_spec", v_idx, 0, beta_f))

        if deriv_weak_cols:
            weak_blocks.extend(deriv_weak_cols)

        # 8. Nonlinear state-derivative interactions (exact IBP where conservative, mollified otherwise)
        if self.include_interaction and deriv_specs:
            # Precompute mollified derivatives only if needed for non-conservative cross-terms
            mollified_cache: dict[tuple[str, int, int, float], np.ndarray] = {}
            for state_idx in range(self.n_features_in_):
                u_state = X_arr[..., state_idx : state_idx + 1]
                for spec in deriv_specs:
                    kind, v_idx, d_ax, ord_val = spec
                    if kind == "int" and int(ord_val) == 1 and state_idx == v_idx:
                        W_op = list(W_base)
                        W_op[1 + d_ax] = Phi_xd[(1, d_ax)]
                        weak_blocks.append(contract_tensor_weak(0.5 * (u_state**2), W_op))
                        continue
                    if spec not in mollified_cache:
                        if kind == "int":
                            mollified_cache[spec] = _weak_mollified_derivative(
                                X_arr[..., v_idx : v_idx + 1],
                                dx=sp_steps[d_ax],
                                order=int(ord_val),
                                axis=1 + d_ax,
                                boundary=self.boundary,
                            )
                        elif kind == "lap":
                            mollified_cache[spec] = sum(  # type: ignore[assignment]
                                _weak_mollified_derivative(
                                    X_arr[..., v_idx : v_idx + 1],
                                    dx=sp_steps[d],
                                    order=2,
                                    axis=1 + d,
                                    boundary=self.boundary,
                                )
                                for d in range(D)
                            )
                        elif kind == "gl":
                            mollified_cache[spec] = fractional_derivative_gl(
                                X_arr[..., v_idx : v_idx + 1],
                                t=sp_steps[d_ax],
                                alpha=ord_val,
                                axis=1 + d_ax,
                            )
                        else:
                            mollified_cache[spec] = fractional_laplacian(
                                X_arr[..., v_idx : v_idx + 1],
                                dx=sp_steps,
                                beta=ord_val,
                                spatial_axes=tuple(range(1, 1 + D)),
                                method=self.riesz_method,
                            )
                    weak_blocks.append(contract_tensor_weak(u_state * mollified_cache[spec], W_base))

        # 9. Michaelis–Menten rational saturation terms
        if self.include_michaelis_menten:
            mm_eval = np.concatenate(self._evaluate_michaelis_menten(X_arr), axis=-1)
            weak_blocks.append(contract_tensor_weak(mm_eval, W_base))

        return np.concatenate(weak_blocks, axis=-1)


SpatialDiffusionLibrary = PDELibrary


class CustomLibrary(BaseCandidateLibrary):
    """User-defined candidate library from arbitrary nonlinear callables.

    Accepts callables that take either unpacked state components ``f(x0, x1, ...)``
    or the full state tensor ``f(X)`` (where the state dimension is the last axis).

    Parameters
    ----------
    functions : list or sequence of callables
        Basis functions evaluating nonlinear, Hill, rational, or trigonometric terms.
    function_names : list or sequence of str or callables
        Symbolic names (strings such as ``"u/(1+u^2)"`` or format callables ``lambda x, y: ...``).
    """

    def __init__(
        self,
        functions: Sequence[Callable[..., np.ndarray]],
        function_names: Sequence[Callable[..., str] | str],
    ) -> None:
        self.functions = functions
        self.function_names = function_names

    @staticmethod
    def _eval_callable(
        fn: Callable[..., np.ndarray],
        arr: np.ndarray,
    ) -> np.ndarray:
        n_vars = arr.shape[-1]
        lead_shape = arr.shape[:-1]

        # Inspect positional parameter count to support both fn(x0, x1, ...) and fn(X)
        use_unpacked = False
        try:
            sig = inspect.signature(fn)
            params = [
                p
                for p in sig.parameters.values()
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
            if has_varargs or (n_vars >= 2 and len(params) == n_vars):
                use_unpacked = True
        except (TypeError, ValueError):
            use_unpacked = False

        if use_unpacked:
            raw = fn(*(arr[..., i] for i in range(n_vars)))
        else:
            try:
                raw = fn(arr)
                raw_arr = np.asarray(raw, dtype=np.float64)
                if raw_arr.shape not in (lead_shape, (*lead_shape, 1)) and n_vars == 1:
                    raw = fn(arr[..., 0])
            except (TypeError, IndexError, ValueError):
                raw = fn(*(arr[..., i] for i in range(n_vars)))

        val = np.asarray(raw, dtype=np.float64)
        if val.ndim == 0:
            return np.full((*lead_shape, 1), float(val), dtype=np.float64)
        if val.shape == lead_shape:
            return val[..., np.newaxis]
        if val.shape == (*lead_shape, 1):
            return val
        raise ValueError(
            f"CustomLibrary function returned shape {val.shape}; expected {lead_shape} or {(*lead_shape, 1)}."
        )

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> CustomLibrary:
        arr = self._validate_input(X)
        if len(self.functions) == 0 or len(self.functions) != len(self.function_names):
            raise ValueError("functions and function_names must be non-empty and have equal length.")
        self.n_features_in_ = arr.shape[-1]
        self.n_output_features_ = len(self.functions)
        return self

    def get_feature_names(self, input_features: Sequence[str] | None = None) -> list[str]:
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("CustomLibrary must be fitted before calling get_feature_names.")
        base = self._default_feature_names(self.n_features_in_, input_features)
        names: list[str] = []
        for fn_name in self.function_names:
            if callable(fn_name):
                names.append(str(fn_name(*base)))
            else:
                names.append(str(fn_name))
        return names

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("CustomLibrary must be fitted before calling transform.")
        arr = self._validate_input(X)
        if arr.shape[-1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} state variables; got {arr.shape[-1]}."
            )
        cols = [self._eval_callable(fn, arr) for fn in self.functions]
        return np.concatenate(cols, axis=-1)


class CombinedLibrary(BaseCandidateLibrary):
    """Concatenate multiple candidate libraries along the feature dimension."""

    def __init__(
        self,
        libraries: Sequence[BaseCandidateLibrary],
        deduplicate_bias: bool = True,
    ) -> None:
        self.libraries = libraries
        self.deduplicate_bias = deduplicate_bias

    @property
    def _supports_spatiotemporal_weak(self) -> bool:  # type: ignore[override]
        return any(getattr(lib, "_supports_spatiotemporal_weak", False) for lib in self.libraries)

    def _get_spatial_steps(self, n_dims: int) -> tuple[float, ...]:
        for lib in self.libraries:
            if hasattr(lib, "_get_spatial_steps"):
                return lib._get_spatial_steps(n_dims)
        return (1.0,) * n_dims

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> CombinedLibrary:
        arr = self._validate_input(X)
        if len(self.libraries) == 0:
            raise ValueError("CombinedLibrary requires at least one sub-library.")
        self.n_features_in_ = arr.shape[-1]
        for lib in self.libraries:
            lib.fit(arr)

        keep_masks: list[np.ndarray] = []
        seen_bias = False
        for lib in self.libraries:
            sub_names = lib.get_feature_names()
            mask = np.ones(len(sub_names), dtype=bool)
            if self.deduplicate_bias:
                for idx, name in enumerate(sub_names):
                    if name == "1":
                        if seen_bias:
                            mask[idx] = False
                        else:
                            seen_bias = True
            keep_masks.append(mask)
        self._keep_masks_ = keep_masks
        self.n_output_features_ = int(sum(int(np.sum(m)) for m in keep_masks))
        return self

    def get_feature_names(self, input_features: Sequence[str] | None = None) -> list[str]:
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("CombinedLibrary must be fitted before calling get_feature_names.")
        names: list[str] = []
        for lib, mask in zip(self.libraries, self._keep_masks_):
            sub_names = lib.get_feature_names(input_features)
            names.extend(n for n, keep in zip(sub_names, mask) if keep)
        return names

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("CombinedLibrary must be fitted before calling transform.")
        arr = self._validate_input(X)
        blocks = [
            lib.transform(arr)[..., mask] for lib, mask in zip(self.libraries, self._keep_masks_)
        ]
        return np.concatenate(blocks, axis=-1)

    def _project_spatiotemporal_weak(
        self,
        X_arr: np.ndarray,
        *,
        Phi_t: np.ndarray,
        Phi_x0: list[np.ndarray],
        sp_wins: list[list[tuple[int, int]]],
        sp_steps: tuple[float, ...],
        sp_degree: int,
    ) -> np.ndarray:
        W_base = [Phi_t, *Phi_x0]
        blocks: list[np.ndarray] = []
        for lib, mask in zip(self.libraries, self._keep_masks_):
            if hasattr(lib, "_project_spatiotemporal_weak"):
                proj = lib._project_spatiotemporal_weak(
                    X_arr,
                    Phi_t=Phi_t,
                    Phi_x0=Phi_x0,
                    sp_wins=sp_wins,
                    sp_steps=sp_steps,
                    sp_degree=sp_degree,
                )
            else:
                eval_block = lib.transform(X_arr)
                proj = contract_tensor_weak(eval_block, W_base)
            blocks.append(proj[:, mask])
        return np.concatenate(blocks, axis=-1)
