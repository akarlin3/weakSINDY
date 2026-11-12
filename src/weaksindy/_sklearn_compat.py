"""Lightweight optional ``scikit-learn`` compatibility layer.

When ``scikit-learn`` is installed, re-exports the canonical ``BaseEstimator``,
``RegressorMixin``, and ``TransformerMixin`` classes from ``sklearn.base`` so all
``weaksindy`` estimators pass ``sklearn.base.clone`` and ``GridSearchCV`` checks.
When ``scikit-learn`` is not installed, provides pure-Python/NumPy drop-in mixins
implementing ``get_params(deep=True)``, ``set_params(**params)``, ``score(X, y)``,
and ``fit_transform(X, y=None)`` with zero external dependencies beyond NumPy/SciPy.
"""
from __future__ import annotations

import inspect
from typing import Any

import numpy as np


class PureBaseEstimator:
    """Minimal ``sklearn.base.BaseEstimator`` fallback using ``inspect.signature``."""

    @classmethod
    def _get_param_names(cls) -> list[str]:
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            return []
        sig = inspect.signature(init)
        parameters = [
            p
            for p in sig.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    f"Estimators should not specify varargs *args in __init__: {cls.__name__}."
                )
        return sorted([p.name for p in parameters])

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                deep_items = value.get_params().items()
                out.update((f"{key}__{k}", val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params: Any) -> Any:
        if not params:
            return self
        valid_params = self.get_params(deep=True)
        nested_params: dict[str, dict[str, Any]] = {}
        for key, value in params.items():
            key_root, delim, sub_key = key.partition("__")
            if key not in valid_params and key_root not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self.__class__.__name__}. "
                    f"Valid parameters are: {sorted(valid_params.keys())}."
                )
            if delim:
                nested_params.setdefault(key_root, {})[sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value
        for key_root, sub_params in nested_params.items():
            valid_params[key_root].set_params(**sub_params)
        return self

    def __repr__(self) -> str:
        params = self.get_params(deep=False)
        items = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"{self.__class__.__name__}({items})"


class PureRegressorMixin:
    """Minimal ``sklearn.base.RegressorMixin`` fallback computing multi-output R^2."""

    _estimator_type = "regressor"

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
        y_pred = np.asarray(self.predict(X), dtype=np.float64)  # type: ignore[attr-defined]
        y_true = np.asarray(y, dtype=np.float64)
        if y_true.ndim == 1:
            y_true = y_true[:, np.newaxis]
        if y_pred.ndim == 1:
            y_pred = y_pred[:, np.newaxis]
        if sample_weight is not None:
            w = np.asarray(sample_weight, dtype=np.float64).reshape(-1, 1)
            w_sum = max(float(np.sum(w)), 1e-12)
            y_mean = np.sum(w * y_true, axis=0, keepdims=True) / w_sum
            ss_res = np.sum(w * (y_true - y_pred) ** 2, axis=0)
            ss_tot = np.sum(w * (y_true - y_mean) ** 2, axis=0)
        else:
            ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
            ss_tot = np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2, axis=0)
        return float(np.mean(1.0 - ss_res / (ss_tot + 1e-12)))


class PureTransformerMixin:
    """Minimal ``sklearn.base.TransformerMixin`` fallback providing ``fit_transform``."""

    def fit_transform(self, X: np.ndarray, y: Any = None, **fit_params: Any) -> np.ndarray:
        if y is None:
            return self.fit(X, **fit_params).transform(X)  # type: ignore[attr-defined]
        return self.fit(X, y, **fit_params).transform(X)  # type: ignore[attr-defined]


try:
    from sklearn.base import (  # type: ignore[import-not-found]
        BaseEstimator,
        RegressorMixin,
        TransformerMixin,
    )

    HAS_SKLEARN = True
except ImportError:  # pragma: no cover - used when scikit-learn is not installed
    BaseEstimator = PureBaseEstimator  # type: ignore[misc,assignment]
    RegressorMixin = PureRegressorMixin  # type: ignore[misc,assignment]
    TransformerMixin = PureTransformerMixin  # type: ignore[misc,assignment]
    HAS_SKLEARN = False

__all__ = [
    "BaseEstimator",
    "HAS_SKLEARN",
    "PureBaseEstimator",
    "PureRegressorMixin",
    "PureTransformerMixin",
    "RegressorMixin",
    "TransformerMixin",
]
