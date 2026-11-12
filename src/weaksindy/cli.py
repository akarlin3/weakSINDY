"""Command-line interface for weak-form fractional SINDy and Ensemble UQ discovery."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .audit import run_audit
from .ensemble import EnsembleWeakSINDy
from .library import PDELibrary, PolynomialLibrary
from .solvers import WeakSINDy

try:
    import click
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'weaksindy' CLI requires optional dependency 'click>=8.0.0'. "
        "Install with: pip install 'weak-sindy[cli]'"
    ) from exc


def _load_timeseries(
    path: Path,
    dt_override: float | None,
    dx_override: float | None,
) -> tuple[float | np.ndarray, np.ndarray, float | tuple[float, ...] | None, list[str] | None]:
    """Load CSV, NPY, or NPZ trajectory data into ``(t, X, dx, feature_names)``."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        X = np.asarray(np.load(path), dtype=np.float64)
        t: float | np.ndarray = 1.0 if dt_override is None else float(dt_override)
        return t, X, dx_override, None

    if suffix == ".npz":
        archive = np.load(path, allow_pickle=False)
        state_key = None
        for candidate in ("X", "u", "data", "y", "states"):
            if candidate in archive.files:
                state_key = candidate
                break
        if state_key is None:
            coord_keys = {"t", "time", "x", "y_coord", "z", "dt", "dx", "dy", "dz"}
            non_coord = [k for k in archive.files if k not in coord_keys]
            if not non_coord:
                raise click.ClickException(f"Could not locate state array in {path}.")
            state_key = non_coord[0]

        X = np.asarray(archive[state_key], dtype=np.float64)
        if dt_override is not None:
            t = float(dt_override)
        elif "t" in archive.files:
            t = np.asarray(archive["t"], dtype=np.float64)
        elif "time" in archive.files:
            t = np.asarray(archive["time"], dtype=np.float64)
        elif "dt" in archive.files:
            t = float(archive["dt"])
        else:
            t = 1.0

        dx_val: float | tuple[float, ...] | None = dx_override
        if dx_val is None:
            if "dx" in archive.files:
                dx_arr = np.asarray(archive["dx"], dtype=np.float64)
                dx_val = float(dx_arr) if dx_arr.ndim == 0 else tuple(float(v) for v in dx_arr.ravel())
            elif X.ndim >= 3:
                sp_ndim = X.ndim - 2
                coord_names = ("x", "y", "z")[:sp_ndim]
                extracted: list[float] = []
                for cname in coord_names:
                    if cname in archive.files and cname != state_key:
                        c_arr = np.asarray(archive[cname], dtype=np.float64)
                        if c_arr.ndim == 1 and c_arr.size >= 2:
                            extracted.append(float(c_arr[1] - c_arr[0]))
                if len(extracted) == sp_ndim:
                    dx_val = extracted[0] if sp_ndim == 1 else tuple(extracted)
                elif len(extracted) == 1:
                    dx_val = extracted[0]
        return t, X, dx_val, None

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            first_row = next(reader, None)
            if first_row is None:
                raise click.ClickException(f"CSV file {path} is empty.")

            has_header = False
            try:
                _ = [float(cell) for cell in first_row]
            except ValueError:
                has_header = True

            rows: list[list[float]] = []
            if not has_header:
                rows.append([float(cell) for cell in first_row])
            for row in reader:
                if any(cell.strip() for cell in row):
                    rows.append([float(cell) for cell in row])

        arr = np.asarray(rows, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 4:
            raise click.ClickException(f"CSV {path} must contain at least 4 time rows.")

        headers = (
            [h.strip() for h in first_row]
            if has_header
            else [f"col{i}" for i in range(arr.shape[1])]
        )
        lower_headers = [h.lower() for h in headers]

        if has_header and lower_headers[0] in ("t", "time"):
            t_col = arr[:, 0]
            X = arr[:, 1:]
            names: list[str] | None = headers[1:]
            t = float(dt_override) if dt_override is not None else t_col
        else:
            X = arr
            names = headers if has_header else None
            t = float(dt_override) if dt_override is not None else 1.0

        return t, X, dx_override, names

    raise click.ClickException(
        f"Unsupported file format '{suffix}'. Expected .csv, .npy, or .npz."
    )


def _parse_orders(orders_str: str) -> list[float]:
    try:
        vals = [float(tok.strip()) for tok in orders_str.split(",") if tok.strip()]
    except ValueError as exc:
        raise click.ClickException(f"Invalid --orders specification '{orders_str}'.") from exc
    if not vals:
        raise click.ClickException("At least one fractional order must be specified in --orders.")
    return vals


def _parse_bounds(bounds_str: str) -> tuple[float, float]:
    try:
        parts = [float(tok.strip()) for tok in bounds_str.split(",") if tok.strip()]
    except ValueError as exc:
        raise click.ClickException(f"Invalid --order-bounds '{bounds_str}'.") from exc
    if len(parts) != 2 or not (0.0 < parts[0] < parts[1]):
        raise click.ClickException(
            f"--order-bounds must be 'min,max' with 0 < min < max; got '{bounds_str}'."
        )
    return (parts[0], parts[1])


def _render_markdown_report(payload: dict[str, Any]) -> str:
    lines: list[str] = [
        "# weakSINDY Discovery Report",
        "",
        f"- **Input**: `{payload['input']}`",
        f"- **Formulation**: `{payload['mode']}` (`{payload['solver']}`)",
        f"- **Recovered Fractional Order**: `{payload['selected_order']:.4f}`",
        f"- **R^2 Score**: `{payload['r2_score']:.5f}`",
        f"- **Pareto AIC / BIC**: `{payload['aic']:.2f}` / `{payload['bic']:.2f}`",
        f"- **Active Terms**: `{payload['n_active_terms']}`",
        "",
        "## Discovered Governing Equations",
        "",
        "```text",
        *payload["equations"],
        "```",
    ]

    if payload.get("ensemble"):
        ci_pct = int(round((1.0 - float(payload["ci_alpha"])) * 100))
        lines.extend(
            [
                "",
                f"## Ensemble Uncertainty Quantification (`n_models={payload['n_models']}`)",
                "",
                f"| State | Candidate Term | Coef | Median | Std | Inclusion Prob | {ci_pct}% Conformal CI |",
                "| :--- | :--- | ---: | ---: | ---: | ---: | :---: |",
            ]
        )
        for row in payload.get("uq_summary", []):
            lines.append(
                f"| `{row['state']}` | `{row['term']}` | {row['coef']:.4f} | "
                f"{row['median']:.4f} | {row['std']:.4f} | {row['inclusion_probability']:.3f} | "
                f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}] |"
            )

    return "\n".join(lines) + "\n"


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument(
    "input_path",
    required=False,
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--audit",
    is_flag=True,
    help="Run standardized AveryCorp Suite II weak-form identifiability audit and exit.",
)
@click.option(
    "--json",
    "json_flag",
    is_flag=True,
    help="Emit machine-readable JSON output (for --audit or standard discovery).",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Deterministic PCG64 seed used with --audit.",
)
@click.option("--dt", type=float, default=None, help="Uniform time step dt (overrides time column).")
@click.option("--dx", type=float, default=None, help="Spatial step size dx for N-D PDE datasets.")
@click.option(
    "--orders",
    type=str,
    default="0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    show_default=True,
    help="Comma-separated candidate fractional orders alpha.",
)
@click.option(
    "--optimize-order/--no-optimize-order",
    default=False,
    show_default=True,
    help="Continuously refine fractional order alpha via Brent + Pareto BIC/AIC.",
)
@click.option(
    "--order-bounds",
    type=str,
    default="0.1,1.0",
    show_default=True,
    help="Continuous fractional order search bounds 'min,max'.",
)
@click.option("--degree", type=int, default=3, show_default=True, help="Polynomial library degree.")
@click.option("--threshold", type=float, default=0.05, show_default=True, help="Sparsity cutoff.")
@click.option(
    "--solver",
    type=click.Choice(["stlsq", "sr3"], case_sensitive=False),
    default="stlsq",
    show_default=True,
    help="Sparse regression optimizer.",
)
@click.option(
    "--ensemble/--no-ensemble",
    default=False,
    show_default=True,
    help="Run EnsembleWeakSINDy subdomain bagging and conformal UQ.",
)
@click.option("--n-models", type=int, default=50, show_default=True, help="Number of ensemble models.")
@click.option(
    "--subsample-fraction",
    type=float,
    default=0.7,
    show_default=True,
    help="Subdomain bootstrap fraction per ensemble model.",
)
@click.option(
    "--aggregate",
    type=click.Choice(["median", "mean", "stability_selection"], case_sensitive=False),
    default="median",
    show_default=True,
    help="Ensemble coefficient aggregation mode.",
)
@click.option(
    "--ci-alpha",
    type=float,
    default=0.05,
    show_default=True,
    help="Miscoverage level alpha for (1 - alpha) conformal intervals.",
)
@click.option("--window-size", type=int, default=50, show_default=True, help="Weak window size H.")
@click.option("--stride", type=int, default=5, show_default=True, help="Weak window stride S.")
@click.option("--k-start", type=int, default=20, show_default=True, help="Initial steps skipped.")
@click.option(
    "--weak/--pointwise",
    "use_weak",
    default=True,
    show_default=True,
    help="Use C^p weak-form projection or naive pointwise GL differentiation.",
)
@click.option(
    "--format",
    "report_format",
    type=click.Choice(["auto", "json", "markdown"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Report serialization format (auto-detected from --output extension).",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write JSON or Markdown discovery report to file.",
)
@click.option("-q", "--quiet", is_flag=True, help="Suppress console summary banner.")
def main(
    input_path: Path | None,
    audit: bool,
    json_flag: bool,
    seed: int,
    dt: float | None,
    dx: float | None,
    orders: str,
    optimize_order: bool,
    order_bounds: str,
    degree: int,
    threshold: float,
    solver: str,
    ensemble: bool,
    n_models: int,
    subsample_fraction: float,
    aggregate: str,
    ci_alpha: float,
    window_size: int,
    stride: int,
    k_start: int,
    use_weak: bool,
    report_format: str,
    output: Path | None,
    quiet: bool,
) -> None:
    """Discover integer- or fractional-order ODEs and N-D PDEs from CSV/NumPy time-series."""
    if audit:
        rep = run_audit(seed=seed)
        if json_flag or report_format.lower() == "json":
            click.echo(rep.to_json(indent=2))
        elif not quiet:
            click.echo(rep.summary_line())
            for k, v in rep.metrics.items():
                click.echo(f"  - {k}: {v}")
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rep.to_json(indent=2) + "\n", encoding="utf-8")
        return

    if input_path is None:
        raise click.UsageError("Missing argument 'INPUT_PATH'.")

    t_val, X, eff_dx, var_names = _load_timeseries(input_path, dt_override=dt, dx_override=dx)
    cand_orders: Sequence[float] = _parse_orders(orders)
    bounds = _parse_bounds(order_bounds)

    if X.ndim >= 3:
        if eff_dx is None:
            raise click.ClickException(
                "Spatiotemporal PDE array detected; provide --dx or include 'x'/'dx' in .npz."
            )
        library = PDELibrary(dx=eff_dx, poly_degree=degree, include_interaction=False)
    else:
        library = PolynomialLibrary(degree=degree, include_bias=True, include_interaction=True)

    if ensemble:
        ens_model = EnsembleWeakSINDy(
            library=library,
            optimizer=solver.lower(),  # type: ignore[arg-type]
            threshold=threshold,
            n_models=n_models,
            subsample_fraction=subsample_fraction,
            aggregate=aggregate.lower(),  # type: ignore[arg-type]
            optimize_order=optimize_order,
            order_bounds=bounds,
            use_weak=use_weak,
            window_size=window_size,
            stride=stride,
            k_start=k_start,
        )
        ens_model.fit(
            t_val,
            X,
            orders=cand_orders,
            optimize_order=optimize_order,
            order_bounds=bounds,
            input_features=var_names,
        )
        eqs = ens_model.equations(precision=4)
        ci_lo, ci_hi = ens_model.coef_ci(alpha=ci_alpha)
        payload: dict[str, Any] = {
            "input": str(input_path),
            "mode": "weak" if use_weak else "pointwise",
            "solver": solver.lower(),
            "ensemble": True,
            "n_models": n_models,
            "subsample_fraction": subsample_fraction,
            "aggregate": aggregate.lower(),
            "ci_alpha": ci_alpha,
            "optimize_order": optimize_order,
            "selected_order": ens_model.order_,
            "r2_score": ens_model.r2_score_,
            "aic": ens_model.aic_,
            "bic": ens_model.bic_,
            "n_active_terms": int(np.sum(ens_model.coef_ != 0.0)),
            "equations": eqs,
            "feature_names": ens_model.feature_names_,
            "coefficients": ens_model.coef_.tolist(),
            "coef_median": ens_model.coef_median_.tolist(),
            "coef_std": ens_model.coef_std_.tolist(),
            "inclusion_probabilities": ens_model.inclusion_probabilities_.tolist(),
            "confidence_intervals": {
                "alpha": ci_alpha,
                "method": "conformal",
                "ci_low": ci_lo.tolist(),
                "ci_high": ci_hi.tolist(),
            },
            "uq_summary": ens_model.summary_table(
                alpha=ci_alpha, active_only=False, min_inclusion=0.05
            ),
            "order_scores": {f"{k:g}": v for k, v in ens_model.order_scores_.items()},
            "order_profile": ens_model.order_profile_,
        }
    else:
        model = WeakSINDy(
            library=library,
            optimizer=solver.lower(),  # type: ignore[arg-type]
            threshold=threshold,
            optimize_order=optimize_order,
            order_bounds=bounds,
            use_weak=use_weak,
            window_size=window_size,
            stride=stride,
            k_start=k_start,
        )
        model.fit(
            t_val,
            X,
            orders=cand_orders,
            optimize_order=optimize_order,
            order_bounds=bounds,
            input_features=var_names,
        )
        eqs = model.equations(precision=4)
        payload = {
            "input": str(input_path),
            "mode": "weak" if use_weak else "pointwise",
            "solver": solver.lower(),
            "ensemble": False,
            "optimize_order": optimize_order,
            "selected_order": model.order_,
            "r2_score": model.r2_score_,
            "aic": model.aic_,
            "bic": model.bic_,
            "n_active_terms": int(np.sum(model.coef_ != 0.0)),
            "equations": eqs,
            "feature_names": model.feature_names_,
            "coefficients": model.coef_.tolist(),
            "order_scores": {f"{k:g}": v for k, v in model.order_scores_.items()},
            "order_profile": model.order_profile_,
        }

    fmt = report_format.lower()
    if not quiet:
        if fmt == "markdown" and output is None:
            click.echo(_render_markdown_report(payload), nl=False)
        else:
            mode_label = "Weak-Form GL" if use_weak else "Pointwise GL"
            ens_tag = f" + Ensemble(B={n_models})" if ensemble else ""
            click.echo(
                f"weakSINDY ({mode_label}{ens_tag}) | Recovered order alpha_hat = {payload['selected_order']:.3f}"
            )
            click.echo(
                f"R^2 = {payload['r2_score']:.5f} | BIC = {payload['bic']:.2f} | "
                f"Active terms = {payload['n_active_terms']}"
            )
            click.echo("-" * 68)
            for line in eqs:
                click.echo(f"  {line}")
            if ensemble and payload.get("uq_summary"):
                click.echo("-" * 68)
                ci_pct = int(round((1.0 - ci_alpha) * 100))
                for row in payload["uq_summary"]:
                    if row["coef"] != 0.0:
                        click.echo(
                            f"  [{row['state']}] {row['term']:>14s}: "
                            f"p_inc={row['inclusion_probability']:.2f}, "
                            f"{ci_pct}% CI=[{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
                        )

    if output is not None:
        is_md = fmt == "markdown" or (
            fmt == "auto" and output.suffix.lower() in (".md", ".markdown")
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        if is_md:
            output.write_text(_render_markdown_report(payload), encoding="utf-8")
            if not quiet:
                click.echo(f"\nSaved Markdown report to {output}")
        else:
            output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            if not quiet:
                click.echo(f"\nSaved JSON report to {output}")


if __name__ == "__main__":  # pragma: no cover
    main()
