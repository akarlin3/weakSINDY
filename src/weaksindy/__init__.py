"""weakSINDY: Noise-robust weak-form fractional and integer-order sparse identification."""

from .library import (
    BaseCandidateLibrary,
    CombinedLibrary,
    CustomLibrary,
    FourierLibrary,
    PDELibrary,
    PolynomialLibrary,
    SpatialDiffusionLibrary,
    spatial_finite_difference,
)
from .lyapunov import (
    LyapunovResult,
    compute_benettin_lyapunov,
    rosenstein_lyapunov,
)
from .operators import (
    fractional_derivative_gl,
    gl_operator_matrix,
    gl_weights,
    integrate_fractional_ode,
    noise_amplification_factor,
)
from .solvers import SR3, STLSQ, WeakSINDy
from .test_functions import (
    build_weak_weights,
    bump_test_function,
    generate_windows,
    project_weak_form,
    weak_noise_amplification_factor,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "WeakSINDy",
    "fractional_derivative_gl",
    "project_weak_form",
    "compute_benettin_lyapunov",
    "STLSQ",
    "SR3",
    "BaseCandidateLibrary",
    "PolynomialLibrary",
    "FourierLibrary",
    "PDELibrary",
    "SpatialDiffusionLibrary",
    "CustomLibrary",
    "CombinedLibrary",
    "spatial_finite_difference",
    "gl_weights",
    "gl_operator_matrix",
    "noise_amplification_factor",
    "integrate_fractional_ode",
    "bump_test_function",
    "generate_windows",
    "build_weak_weights",
    "weak_noise_amplification_factor",
    "LyapunovResult",
    "rosenstein_lyapunov",
]
