"""Shared AveryCorp AuditReport contract for ``weak-sindy`` (Suite II: PDE & Dynamical Identifiability Gate)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .ensemble import EnsembleWeakSINDy
from .library import PolynomialLibrary
from .operators import integrate_fractional_ode

SuiteName = Literal[
    "Suite I: Clinical Quantitative-MRI Trust Spine",
    "Suite II: Scientific Identifiability & Null-Audit Suite",
]
AuditStatus = Literal["PASS", "WARN", "HALT"]

PACKAGE_NAME = "weak-sindy"
PACKAGE_VERSION = "1.2.0"
SUITE_NAME: SuiteName = "Suite II: Scientific Identifiability & Null-Audit Suite"
STAGE_ROLE = "Suite II: Weak-Form Fractional PDE/ODE Identifiability & Ensemble Inclusion Gate"

__all__ = ["AuditReport", "run_audit"]


@dataclass(frozen=True)
class AuditReport:
    """Standardized machine-readable AveryCorp scientific diagnostic report."""

    package: str
    version: str
    suite: SuiteName
    stage_role: str
    status: AuditStatus
    verdict: str
    metrics: dict[str, float | int | str | bool]
    thresholds: dict[str, float | int | str]
    seed: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize the diagnostic report into a JSON-ready dictionary."""
        return {
            "package": self.package,
            "version": self.version,
            "suite": self.suite,
            "stage_role": self.stage_role,
            "status": self.status,
            "verdict": self.verdict,
            "metrics": dict(self.metrics),
            "thresholds": dict(self.thresholds),
            "seed": int(self.seed),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize the diagnostic report as formatted JSON."""
        return json.dumps(self.to_dict(), indent=indent)

    def summary_line(self) -> str:
        """Return a concise single-line human-readable audit summary."""
        return (
            f"[{self.status}] {self.package} v{self.version} "
            f"({self.suite} | {self.stage_role}) — {self.verdict}"
        )


def _vdp_rhs(_t: float, state: np.ndarray) -> np.ndarray:
    x0, x1 = float(state[0]), float(state[1])
    return np.array([x1, -1.0 * x0 + 1.0 * x1 - 1.0 * (x0**2) * x1], dtype=np.float64)


def run_audit(
    seed: int = 42,
    *,
    noise_sigma_rel: float = 0.015,
    n_models: int = 24,
) -> AuditReport:
    """Run a seeded noisy fractional ODE support-recovery and ensemble inclusion identifiability check."""
    seed_int = int(seed)
    alpha_true = 0.80
    dt = 0.02
    n_steps = 900
    t_grid = np.arange(n_steps, dtype=np.float64) * dt

    X_clean = integrate_fractional_ode(
        _vdp_rhs,
        y0=np.array([1.5, 0.0], dtype=np.float64),
        t=t_grid,
        alpha=alpha_true,
    )
    rng = np.random.Generator(np.random.PCG64(seed_int))
    state_std = np.std(X_clean, axis=0, keepdims=True)
    X_noisy = X_clean + float(noise_sigma_rel) * state_std * rng.standard_normal(X_clean.shape)

    ens_rng = np.random.Generator(np.random.PCG64(seed_int + 1))
    library = PolynomialLibrary(degree=3, include_bias=False, include_interaction=True)
    ens = EnsembleWeakSINDy(
        library=library,
        optimizer="stlsq",
        threshold=0.20,
        n_models=int(n_models),
        subsample_fraction=0.75,
        aggregate="stability_selection",
        inclusion_threshold=0.65,
        orders=(0.6, 0.7, 0.8, 0.9, 1.0),
        optimize_order=True,
        order_bounds=(0.45, 1.0),
        window_size=50,
        stride=4,
        k_start=25,
        random_state=ens_rng,
    )
    ens.fit(dt, X_noisy, orders=(0.6, 0.7, 0.8, 0.9, 1.0), optimize_order=True)

    names = list(ens.feature_names_)
    true_support = np.zeros_like(ens.coef_, dtype=bool)
    true_support[0, names.index("x1")] = True
    true_support[1, names.index("x0")] = True
    true_support[1, names.index("x1")] = True
    true_support[1, names.index("x0^2 x1")] = True

    est_support = ens.coef_ != 0.0
    tp = int(np.sum(true_support & est_support))
    fp = int(np.sum((~true_support) & est_support))
    fn = int(np.sum(true_support & (~est_support)))
    denom = 2 * tp + fp + fn
    support_f1 = float((2 * tp) / denom) if denom > 0 else 0.0

    alpha_hat = float(ens.order_)
    alpha_err = abs(alpha_hat - alpha_true)
    min_active_inc = float(np.min(ens.inclusion_probabilities_[true_support]))

    thresholds: dict[str, float | int | str] = {
        "min_support_f1": 0.90,
        "max_order_abs_error": 0.08,
        "min_active_inclusion_prob": 0.75,
    }

    if support_f1 >= 0.90 and alpha_err <= 0.08 and min_active_inc >= 0.75:
        status: AuditStatus = "PASS"
        verdict_code = "IDENTIFIABLE_SUPPORT_RECOVERED"
        verdict = (
            f"IDENTIFIABLE_SUPPORT_RECOVERED: Weak-form ensemble recovered sparse support "
            f"(F1={support_f1:.3f}), fractional order |alpha_hat - alpha*|={alpha_err:.4f}, "
            f"and min active inclusion probability={min_active_inc:.3f}."
        )
    elif support_f1 >= 0.75 and alpha_err <= 0.15:
        status = "WARN"
        verdict_code = "PARTIAL_SUPPORT_DRIFT"
        verdict = (
            f"PARTIAL_SUPPORT_DRIFT: Partial support recovery (F1={support_f1:.3f}, "
            f"|alpha_hat - alpha*|={alpha_err:.4f}, min p_inc={min_active_inc:.3f})."
        )
    else:
        status = "HALT"
        verdict_code = "NOISE_FLOOR_COLLAPSE"
        verdict = (
            f"NOISE_FLOOR_COLLAPSE: Support recovery collapsed below noise floor "
            f"(F1={support_f1:.3f}, |alpha_hat - alpha*|={alpha_err:.4f})."
        )

    metrics: dict[str, float | int | str | bool] = {
        "support_f1": round(support_f1, 6),
        "true_order": alpha_true,
        "recovered_order": round(alpha_hat, 6),
        "order_abs_error": round(alpha_err, 6),
        "min_active_inclusion_prob": round(min_active_inc, 6),
        "r2_score": round(float(ens.r2_score_), 6),
        "n_active_terms": int(np.sum(est_support)),
        "noise_sigma_rel": float(noise_sigma_rel),
        "verdict_code": verdict_code,
    }

    return AuditReport(
        package=PACKAGE_NAME,
        version=PACKAGE_VERSION,
        suite=SUITE_NAME,
        stage_role=STAGE_ROLE,
        status=status,
        verdict=verdict,
        metrics=metrics,
        thresholds=thresholds,
        seed=seed_int,
    )
