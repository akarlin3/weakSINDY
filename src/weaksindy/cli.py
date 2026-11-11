"""Command-line interface for running weak-form fractional SINDy discovery on CSV/NumPy data."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np

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
) -> tuple[float | np.ndarray, np.ndarray, float | None, list[str] | None]:
    """Load CSV, NPY, or NPZ trajectory data into ``(t, X, dx, feature_names)``."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        X = np.load(path)
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
            non_coord = [k for k in archive.files if k not in ("t", "time", "x", "dt", "dx")]
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

        dx = dx_override
        if dx is None:
            if "dx" in archive.files:
                dx = float(archive["dx"])
            elif "x" in archive.files:
                x_arr = np.asarray(archive["x"], dtype=np.float64)
                if x_arr.size >= 2:
                    dx = float(x_arr[1] - x_arr[0])
        return t, X, dx, None

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

        headers = [h.strip() for h in first_row] if has_header else [f"col{i}" for i in range(arr.shape[1])]
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


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--dt", type=float, default=None, help="Uniform time step dt (overrides time column).")
@click.option("--dx", type=float, default=None, help="Spatial step size dx for 3D PDE datasets.")
@click.option(
    "--orders",
    type=str,
    default="0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    show_default=True,
    help="Comma-separated candidate fractional orders alpha.",
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
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write JSON discovery report to file.",
)
@click.option("-q", "--quiet", is_flag=True, help="Suppress console summary banner.")
def main(
    input_path: Path,
    dt: float | None,
    dx: float | None,
    orders: str,
    degree: int,
    threshold: float,
    solver: str,
    window_size: int,
    stride: int,
    k_start: int,
    use_weak: bool,
    output: Path | None,
    quiet: bool,
) -> None:
    """Discover integer- or fractional-order ODEs and PDEs from CSV/NumPy time-series."""
    t_val, X, eff_dx, var_names = _load_timeseries(input_path, dt_override=dt, dx_override=dx)
    cand_orders: Sequence[float] = _parse_orders(orders)

    if X.ndim == 3:
        if eff_dx is None:
            raise click.ClickException("3D PDE array detected; provide --dx or include 'x'/'dx' in .npz.")
        library = PDELibrary(dx=eff_dx, poly_degree=degree, include_interaction=False)
    else:
        library = PolynomialLibrary(degree=degree, include_bias=True, include_interaction=True)

    model = WeakSINDy(
        library=library,
        optimizer=solver.lower(),  # type: ignore[arg-type]
        threshold=threshold,
        use_weak=use_weak,
        window_size=window_size,
        stride=stride,
        k_start=k_start,
    )
    model.fit(t_val, X, orders=cand_orders, input_features=var_names)
    eqs = model.equations(precision=4)

    if not quiet:
        mode_label = "Weak-Form GL" if use_weak else "Pointwise GL"
        click.echo(f"weakSINDY ({mode_label}) | Recovered order alpha_hat = {model.order_:.3f}")
        click.echo(f"Validation R^2 = {model.r2_score_:.5f} | Active terms = {int(np.sum(model.coef_ != 0.0))}")
        click.echo("-" * 64)
        for line in eqs:
            click.echo(f"  {line}")

    if output is not None:
        payload = {
            "input": str(input_path),
            "mode": "weak" if use_weak else "pointwise",
            "solver": solver.lower(),
            "selected_order": model.order_,
            "r2_score": model.r2_score_,
            "n_active_terms": int(np.sum(model.coef_ != 0.0)),
            "equations": eqs,
            "feature_names": model.feature_names_,
            "coefficients": model.coef_.tolist(),
            "order_scores": {f"{k:g}": v for k, v in model.order_scores_.items()},
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        if not quiet:
            click.echo(f"\nSaved JSON report to {output}")


if __name__ == "__main__":  # pragma: no cover
    main()
