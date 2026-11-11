"""Polynomial, trigonometric, and spatial diffusion candidate feature libraries."""

from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import combinations_with_replacement
from typing import Callable, Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .operators import extract_time_step, fractional_derivative_gl


def spatial_finite_difference(
    X: np.ndarray,
    dx: float,
    order: int = 1,
    axis: int = 1,
    boundary: str = "one_sided",
) -> np.ndarray:
    """Compute second-order accurate spatial finite differences along ``axis``.

    Parameters
    ----------
    X : np.ndarray
        Input array with spatial dimension of size ``nx >= 4`` along ``axis``.
    dx : float
        Uniform spatial step size.
    order : int, default=1
        Derivative order (``1`` for ``d/dx``, ``2`` for ``d^2/dx^2``).
    axis : int, default=1
        Spatial axis index.
    boundary : {"one_sided", "neumann", "periodic"}, default="one_sided"
        Boundary stencil treatment:
        - ``"one_sided"``: second-order one-sided stencils at both endpoints.
        - ``"neumann"``: homogeneous Neumann zero-flux ghost-node reflection
          (``u_{-1} = u_1``, ``u_{nx} = u_{nx-2}``).
        - ``"periodic"``: periodic wrap-around stencils.
    """
    step = extract_time_step(dx)
    if order not in (1, 2):
        raise ValueError(f"Supported integer spatial derivative orders are 1 and 2; got {order}.")
    if boundary not in ("one_sided", "neumann", "periodic"):
        raise ValueError(
            f"boundary must be 'one_sided', 'neumann', or 'periodic'; got {boundary!r}."
        )

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


class BaseCandidateLibrary(BaseEstimator, TransformerMixin, ABC):
    """Abstract base class for weakSINDY candidate basis libraries."""

    n_features_in_: int
    n_output_features_: int

    @staticmethod
    def _validate_input(X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        if arr.ndim not in (2, 3):
            raise ValueError(f"Expected 2D (ODE) or 3D (PDE) state array; got shape {arr.shape}.")
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


class PDELibrary(BaseCandidateLibrary):
    """Reaction–diffusion PDE library with polynomial, integer spatial, and fractional spatial terms.

    Generates polynomial reaction terms up to ``poly_degree``, integer-order spatial
    derivatives (``u_x``, ``u_xx``), optional spatial Grünwald–Letnikov fractional
    derivatives ``D_x^beta u``, optional Michaelis–Menten rational saturation terms
    ``u_i / (1 + u_i)``, and optional state-derivative / state-rational interactions.

    Parameters
    ----------
    dx : float
        Uniform spatial grid spacing along axis 1 (or axis 0 for a single snapshot).
    poly_degree : int, default=2
        Maximum degree of local polynomial reaction terms.
    derivative_orders : sequence of int, default=(1, 2)
        Integer spatial derivative orders to include (subset of ``{1, 2}``).
    fractional_orders : sequence of float, default=()
        Spatial Grünwald–Letnikov fractional derivative orders ``beta`` along space.
    include_bias : bool, default=True
        Whether to include the constant term ``1``.
    include_interaction : bool, default=False
        Whether to include ``u_i * derivative`` cross-terms.
    include_michaelis_menten : bool, default=False
        Whether to include rational saturation basis terms ``u_i / (1 + u_i)`` and
        products with degree-1/2 polynomials ``u_j * u_i / (1 + u_i)`` for biochemical
        reaction–diffusion discovery.
    boundary : {"one_sided", "neumann", "periodic"}, default="one_sided"
        Boundary finite-difference stencil for integer spatial derivatives.
    """

    def __init__(
        self,
        dx: float,
        poly_degree: int = 2,
        derivative_orders: Sequence[int] = (1, 2),
        fractional_orders: Sequence[float] = (),
        include_bias: bool = True,
        include_interaction: bool = False,
        include_michaelis_menten: bool = False,
        boundary: str = "one_sided",
    ) -> None:
        self.dx = dx
        self.poly_degree = poly_degree
        self.derivative_orders = derivative_orders
        self.fractional_orders = fractional_orders
        self.include_bias = include_bias
        self.include_interaction = include_interaction
        self.include_michaelis_menten = include_michaelis_menten
        self.boundary = boundary

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> PDELibrary:
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim not in (2, 3):
            raise ValueError(f"PDELibrary expects 2D (nx, n_vars) or 3D (nt, nx, n_vars); got {arr.shape}.")
        _ = extract_time_step(self.dx)
        self.n_features_in_ = arr.shape[-1]
        self._poly_lib_ = PolynomialLibrary(
            degree=self.poly_degree,
            include_bias=self.include_bias,
            include_interaction=True,
        ).fit(arr if arr.ndim == 3 else arr[np.newaxis, ...])
        self.n_output_features_ = len(self.get_feature_names())
        return self

    def get_feature_names(self, input_features: Sequence[str] | None = None) -> list[str]:
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("PDELibrary must be fitted before calling get_feature_names.")
        base = self._default_feature_names(self.n_features_in_, input_features)
        names = list(self._poly_lib_.get_feature_names(base))

        deriv_names: list[str] = []
        for f_name in base:
            for d_ord in self.derivative_orders:
                suffix = "x" * int(d_ord)
                deriv_names.append(f"{f_name}_{suffix}")

        for beta in self.fractional_orders:
            beta_str = f"{float(beta):g}".replace(".", "_")
            for f_name in base:
                deriv_names.append(f"D_x^{beta_str} {f_name}")

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

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("PDELibrary must be fitted before calling transform.")
        arr = np.asarray(X, dtype=np.float64)
        was_2d = arr.ndim == 2
        if was_2d:
            arr_3d = arr[np.newaxis, :, :]
        elif arr.ndim == 3:
            arr_3d = arr
        else:
            raise ValueError(f"Expected 2D or 3D array for PDELibrary; got shape {arr.shape}.")

        if arr_3d.shape[-1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} state variables; got {arr_3d.shape[-1]}."
            )

        cols: list[np.ndarray] = [self._poly_lib_.transform(arr_3d)]

        deriv_by_order: dict[int, np.ndarray] = {}
        for d_ord in self.derivative_orders:
            deriv_by_order[int(d_ord)] = spatial_finite_difference(
                arr_3d, dx=self.dx, order=int(d_ord), axis=1, boundary=self.boundary
            )

        deriv_cols: list[np.ndarray] = []
        for v_idx in range(self.n_features_in_):
            for d_ord in self.derivative_orders:
                deriv_cols.append(deriv_by_order[int(d_ord)][..., v_idx : v_idx + 1])

        for beta in self.fractional_orders:
            fd_arr = fractional_derivative_gl(arr_3d, t=self.dx, alpha=float(beta), axis=1)
            for v_idx in range(self.n_features_in_):
                deriv_cols.append(fd_arr[..., v_idx : v_idx + 1])

        if deriv_cols:
            cols.extend(deriv_cols)

        if self.include_interaction and deriv_cols:
            for v_idx in range(self.n_features_in_):
                state_col = arr_3d[..., v_idx : v_idx + 1]
                for d_col in deriv_cols:
                    cols.append(state_col * d_col)

        if self.include_michaelis_menten:
            for i in range(self.n_features_in_):
                u_i = np.clip(arr_3d[..., i : i + 1], -0.99, None)
                mm_col = u_i / (1.0 + u_i)
                cols.append(mm_col)
                for j in range(self.n_features_in_):
                    if j != i:
                        u_j = arr_3d[..., j : j + 1]
                        cols.append(u_j * mm_col)
                        cols.append((u_j * u_j) * mm_col)

        out = np.concatenate(cols, axis=-1)
        return out[0] if was_2d else out


SpatialDiffusionLibrary = PDELibrary


class CustomLibrary(BaseCandidateLibrary):
    """User-defined candidate library from callables ``f(X)``."""

    def __init__(
        self,
        functions: Sequence[Callable[[np.ndarray], np.ndarray]],
        function_names: Sequence[Callable[..., str] | str],
    ) -> None:
        self.functions = functions
        self.function_names = function_names

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
        cols: list[np.ndarray] = []
        for fn in self.functions:
            val = np.asarray(fn(arr), dtype=np.float64)
            if val.ndim == arr.ndim - 1:
                val = val[..., np.newaxis]
            cols.append(val)
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
