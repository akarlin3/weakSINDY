"""weakSINDY: Noise-robust N-D weak-form fractional PDE discovery, exogenous control, multi-order operators, and conformal ensemble UQ."""

from .benchmarks import (
    BenchmarkDataset,
    simulate_fractional_lorenz,
    simulate_subdiffusive_fisher_kpp_1d,
    simulate_vascular_stromal_rd,
)
from .control import (
    ControlledBenchmarkDataset,
    ControlLibrary,
    WeakSINDyControl,
    generate_controlled_fractional_system,
    weakSINDYc,
)
from .ensemble import EnsembleWeakSINDy
from .library import (
    BaseCandidateLibrary,
    CombinedLibrary,
    CustomLibrary,
    FourierLibrary,
    PDELibrary,
    PolynomialLibrary,
    RationalLibrary,
    SpatialDiffusionLibrary,
    spatial_finite_difference,
)
from .lyapunov import (
    LyapunovResult,
    compute_benettin_lyapunov,
    rosenstein_lyapunov,
)
from .multi_order import (
    MultiOrderWeakSINDy,
    integrate_multi_order_ode,
    multi_order_fractional_derivative,
)
from .operators import (
    extract_spatial_steps,
    extract_time_step,
    fractional_derivative_gl,
    fractional_laplacian,
    gl_operator_matrix,
    gl_weights,
    integrate_fractional_ode,
    noise_amplification_factor,
    riesz_operator_matrix,
    riesz_weights,
)
from .sklearn_adapter import SklearnWeakSINDyRegressor
from .solvers import SR3, STLSQ, WeakSINDy
from .test_functions import (
    build_spatial_weak_weights,
    build_weak_weights,
    bump_test_function,
    contract_tensor_weak,
    generate_windows,
    project_weak_form,
    resolve_spatial_windows,
    weak_noise_amplification_factor,
)

__version__ = "1.2.0"

__all__ = [
    "__version__",
    "WeakSINDy",
    "WeakSINDyControl",
    "weakSINDYc",
    "ControlLibrary",
    "ControlledBenchmarkDataset",
    "generate_controlled_fractional_system",
    "MultiOrderWeakSINDy",
    "multi_order_fractional_derivative",
    "integrate_multi_order_ode",
    "SklearnWeakSINDyRegressor",
    "EnsembleWeakSINDy",
    "fractional_derivative_gl",
    "fractional_laplacian",
    "project_weak_form",
    "compute_benettin_lyapunov",
    "STLSQ",
    "SR3",
    "BaseCandidateLibrary",
    "PolynomialLibrary",
    "FourierLibrary",
    "RationalLibrary",
    "PDELibrary",
    "SpatialDiffusionLibrary",
    "CustomLibrary",
    "CombinedLibrary",
    "spatial_finite_difference",
    "extract_time_step",
    "extract_spatial_steps",
    "gl_weights",
    "gl_operator_matrix",
    "riesz_weights",
    "riesz_operator_matrix",
    "noise_amplification_factor",
    "integrate_fractional_ode",
    "bump_test_function",
    "generate_windows",
    "resolve_spatial_windows",
    "build_weak_weights",
    "build_spatial_weak_weights",
    "contract_tensor_weak",
    "weak_noise_amplification_factor",
    "LyapunovResult",
    "rosenstein_lyapunov",
    "BenchmarkDataset",
    "simulate_fractional_lorenz",
    "simulate_subdiffusive_fisher_kpp_1d",
    "simulate_vascular_stromal_rd",
]
