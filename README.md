# weakSINDY

[![CI](https://github.com/akarlin3/weakSINDY/actions/workflows/ci.yml/badge.svg)](https://github.com/akarlin3/weakSINDY/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20637003.svg)](https://doi.org/10.5281/zenodo.20637003)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![Typed: PEP 561](https://img.shields.io/badge/typing-PEP%20561-informational.svg)](src/weaksindy/py.typed)

**General-purpose N-dimensional weak-form sparse identification of integer- and fractional-order ODEs and PDEs with continuous order refinement and conformal ensemble uncertainty quantification.**

`weakSINDY` (`weak-sindy`, imported as `weaksindy`) is a standalone Python library for discovering sparse governing ordinary and partial differential equations—jointly with unknown Grünwald–Letnikov fractional differentiation orders $\alpha \in (0, 1]$—from noisy time-series and 1D/2D/3D spatiotemporal fields. By transferring both causal temporal fractional operators $\mathcal{D}_t^\alpha$ and spatial integer/fractional differential operators ($\nabla$, $\nabla^2$, $\nabla^4$, $\mathbf{u}\cdot\nabla$, and Riesz fractional diffusion $(-\Delta)^{\beta/2}$) onto separable compactly supported $C^p$ test functions $\Phi(t, \mathbf{x}) = \phi_t(t)\prod_{d=1}^D \phi_{x_d}(x_d)$, `weakSINDY` eliminates pointwise finite-difference and $\Delta t^{-2\alpha}$ noise blow-up, collapsing breakdown thresholds by **15–25 dB** (oracle ceiling) and **15–30 dB** (deployment-realistic noisy validation). Version `1.1.0` adds bounded continuous Brent/Pareto AIC/BIC order optimization, **`EnsembleWeakSINDy`** subdomain bagging with split-conformal confidence intervals, canonical fractional benchmarks, and a continuous variational **Benettin QR tangent-space integrator**.

---

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/akarlin3/weakSINDY.git
```

Or install from PyPI:

```bash
pip install weak-sindy
```

To install with the command-line interface (`weaksindy`):

```bash
pip install "weak-sindy[cli] @ git+https://github.com/akarlin3/weakSINDY.git"
# or via PyPI:
pip install "weak-sindy[cli]"
```

For local development and running the benchmark test suite:

```bash
git clone https://github.com/akarlin3/weakSINDY.git
cd weakSINDY
pip install -e ".[dev]"
pytest
```

---

## Quickstart

### 1. Continuous Fractional Order Discovery (`WeakSINDy`)

```python
import numpy as np
from weaksindy import (
    PolynomialLibrary,
    WeakSINDy,
    integrate_fractional_ode,
)

# 1. Simulate a fractional Van der Pol oscillator (true order alpha = 0.73)
mu = 2.0
t = np.linspace(0.0, 20.0, 800)
X = integrate_fractional_ode(
    lambda s: np.array([s[1], mu * (1.0 - s[0] ** 2) * s[1] - s[0]]),
    y0=[0.5, 0.5],
    t=t,
    alpha=0.73,
)

# 2. Pinpoint continuous fractional order alpha to ~1e-3 via bounded Brent + Pareto BIC
model = WeakSINDy(
    library=PolynomialLibrary(degree=3),
    threshold=0.1,
    window_size=50,
    stride=5,
)
model.fit(
    t,
    X,
    optimize_order=True,
    order_bounds=(0.2, 1.0),
    input_features=["x", "y"],
)

print(f"Recovered fractional order: {model.order_:.4f} (BIC = {model.bic_:.2f}, R^2 = {model.r2_score_:.4f})")
model.print()
# D_t^0.730 x = 1.0000 y
# D_t^0.730 y = -1.0000 x + 2.0000 y - 2.0000 x^2 y

# 3. Predict fractional derivatives on new state trajectories
dX_pred = model.predict(t, X)
```

### 2. Uncertainty Quantification & Conformal Ensemble (`EnsembleWeakSINDy`)

```python
import numpy as np
from weaksindy import (
    EnsembleWeakSINDy,
    PDELibrary,
    simulate_subdiffusive_fisher_kpp_1d,
)

# Simulate noisy 1D subdiffusive Fisher–KPP reaction–diffusion PDE (22 dB SNR, alpha = 0.7)
t = np.linspace(0.0, 3.5, 211)
x = np.linspace(0.0, 10.0, 56)
bench = simulate_subdiffusive_fisher_kpp_1d(t, x, alpha=0.7, D=0.12, r=1.0, snr_db=22.0)

ensemble = EnsembleWeakSINDy(
    library=PDELibrary(dx=bench.dx, poly_degree=2, derivative_orders=(1, 2), include_bias=False, boundary="neumann"),
    threshold=0.04,
    n_models=50,
    subsample_fraction=0.7,
    replace=True,
    aggregate="median",
    window_size=36,
    spatial_window_size=16,
)
ensemble.fit(t, bench.X, optimize_order=True, order_bounds=(0.3, 1.0), input_features=["u"])

ci_low, ci_high = ensemble.coef_ci(alpha=0.05)  # 95% out-of-bag split-conformal intervals
print("Inclusion probabilities:", dict(zip(ensemble.feature_names_, np.round(ensemble.inclusion_probabilities_[0], 3))))
ensemble.print()
```

### Command-Line Interface

Run discovery and conformal ensemble UQ directly on `.csv`, `.npy`, or `.npz` trajectories:

```bash
weaksindy trajectory.csv --orders 0.3,0.5,0.7,0.9,1.0 --degree 3 --threshold 0.05 -o report.json
weaksindy trajectory.csv --ensemble --n-models 50 --optimize-order --order-bounds 0.2,1.0 -o uq_report.md
weaksindy fields_3d.npz --dx 0.2 --ensemble --n-models 50 --solver sr3 -o pde_report.json
```

---

## Mathematical Theory

### 1. Derivation of the Grünwald–Letnikov Noise Amplification Factor $A(\alpha)$

On a uniform temporal grid $t_k = k \Delta t$ ($k = 0, \dots, N_t - 1$), the causal Grünwald–Letnikov (GL) fractional derivative of order $\alpha \in (0, 1]$ is given by the backward convolution

$$
\left(\mathcal{D}_t^\alpha u\right)_k = \frac{1}{\Delta t^\alpha} \sum_{j=0}^{k} w_j^{(\alpha)} u_{k - j}, \qquad w_0^{(\alpha)} = 1, \quad w_j^{(\alpha)} = \left(1 - \frac{\alpha + 1}{j}\right) w_{j-1}^{(\alpha)} = (-1)^j \binom{\alpha}{j}.
$$

In matrix form over $N_t$ samples, $\mathcal{D}_t^\alpha \mathbf{u} = \mathcal{L}_\alpha \mathbf{u}$, where $\mathcal{L}_\alpha = \Delta t^{-\alpha} \mathbf{W}^{(\alpha)} \in \mathbb{R}^{N_t \times N_t}$ is lower-triangular Toeplitz with $W_{k,m}^{(\alpha)} = w_{k-m}^{(\alpha)}$ for $k \ge m$.

Suppose observations are corrupted by additive zero-mean white measurement noise, $\tilde{\mathbf{u}} = \mathbf{u}_{\text{clean}} + \boldsymbol{\eta}$ with $\boldsymbol{\eta} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$. Because $\mathcal{L}_\alpha$ is linear, the induced perturbation in the regression target $\boldsymbol{\varepsilon}(\alpha) = \mathcal{L}_\alpha \boldsymbol{\eta}$ has covariance $\mathrm{Cov}(\mathcal{L}_\alpha \boldsymbol{\eta}) = \sigma^2 \mathcal{L}_\alpha \mathcal{L}_\alpha^\top$. At time step $k = N_t - 1$, the diagonal noise variance is

$$
\mathrm{Var}\!\left((\mathcal{D}_t^\alpha \eta)_{N_t - 1}\right) = \sigma^2 \|\mathcal{L}_\alpha(N_t - 1, :)\|_2^2 = \sigma^2 \Delta t^{-2\alpha} \sum_{j=0}^{N_t - 1} \left(w_j^{(\alpha)}\right)^2,
$$

defining the **pointwise GL noise amplification factor**:

$$
A(\alpha) \equiv \frac{\mathrm{Var}\!\left((\mathcal{D}_t^\alpha \eta)_{N_t - 1}\right)}{\sigma^2} = \Delta t^{-2\alpha} \|\mathbf{w}^{(\alpha)}\|_2^2.
$$

By Chu–Vandermonde summation of squared binomial coefficients, $\|\mathbf{w}^{(\alpha)}\|_2^2 \to \sum_{j=0}^\infty \binom{\alpha}{j}^2 = \frac{\Gamma(1 + 2\alpha)}{[\Gamma(1 + \alpha)]^2} \in [1, 2]$ for $\alpha \in [0, 1]$. For any resolved step size $\Delta t \ll 1$ (e.g., $\Delta t \approx 0.01$ at $N_t = 500$), the geometric factor $\Delta t^{-2\alpha}$ completely dominates $\|\mathbf{w}^{(\alpha)}\|_2^2$:

| Fractional Order $\alpha$ | $\|\mathbf{w}^{(\alpha)}\|_2^2$ | $A(\alpha) = \Delta t^{-2\alpha}\|\mathbf{w}^{(\alpha)}\|_2^2$ ($\Delta t \approx 0.01$) |
| :---: | :---: | :---: |
| $0.2$ | $1.055$ | $6.62 \times 10^{0}$ |
| $0.3$ | $1.106$ | $1.76 \times 10^{1}$ |
| $0.5$ | $1.272$ | $1.27 \times 10^{2}$ |
| $0.7$ | $1.547$ | $9.96 \times 10^{2}$ |
| $0.9$ | $1.856$ | $7.19 \times 10^{3}$ |
| $1.0$ | $2.000$ | $1.99 \times 10^{4}$ |

**Low-SNR Floor Railing ($\hat{\alpha}_t \to \alpha_{\min}$):** In pointwise sparse regression, target noise $\boldsymbol{\varepsilon}(\alpha)$ inflates least-squares coefficients by $\mathcal{O}(\sigma \Delta t^{-\alpha})$, generating held-out prediction residual variance proportional to $\sigma^2 A(\alpha) \propto \Delta t^{-2\alpha}$. Below $35\text{–}45\text{ dB}$, the validation $R^2(\alpha)$ objective is dominated by $-\sigma^2 \Delta t^{-2\alpha}$, forcing candidate sweeps to step monotonically downward and rail deterministically against the smallest candidate order in the grid ($\hat{\alpha}_t \to 0.20$).

---

### 2. Proof: Why Weak-Form Projection Collapses the Breakdown Bracket by 15–25 dB

**Proposition (Adjoint Variance Cancellation & Order-Invariant Breakdown).** *Let $\phi \in C_c^p([t_a, t_b])$ be a compactly supported polynomial bump kernel on a window $[t_{k_a}, t_{k_b}]$ of width $T_w = H \Delta t$ ($k_b = k_a + H$, $p \ge 2$), normalized such that $\sum_{k=0}^H \phi_k \Delta t = 1$ with $\phi_k = T_w^{-1} \hat{\phi}(k / H)$ and $\hat{\phi}(\xi) = c_p (1 - (2\xi - 1)^2)^p$. Then weak-form projection eliminates the $\Delta t^{-2\alpha}$ noise singularity, reducing target noise variance by a factor of $\mathcal{O}(H^{2\alpha + 1})$ relative to pointwise GL differentiation.*

**Proof.** Taking the discrete $L^2([t_{k_a}, t_{k_b}])$ inner product of the pointwise GL derivative $\mathcal{L}_\alpha \mathbf{u}$ against $\boldsymbol{\phi}^{(m)}$ (supported on indices $k \in [k_a, k_b]$ with $\phi^{(m)}_k = \phi_{k - k_a}$) and exchanging summation indices yields exact **discrete fractional summation by parts**:

$$
\begin{aligned}
y_{\text{weak}}^{(m)}(\alpha)
&= \sum_{k=k_a}^{k_b} \left(\mathcal{L}_\alpha \mathbf{u}\right)_k \phi_{k - k_a} \Delta t
= \Delta t^{1 - \alpha} \sum_{k=k_a}^{k_b} \left(\sum_{i=0}^{k} w_{k - i}^{(\alpha)} u_i\right) \phi_{k - k_a} \\
&= \Delta t^{1 - \alpha} \sum_{i=0}^{k_b} u_i \underbrace{\left(\sum_{k=\max(k_a, i)}^{k_b} w_{k - i}^{(\alpha)} \phi_{k - k_a}\right)}_{\equiv \, \psi_i^{(m)}(\alpha)}
= \Delta t \left\langle \mathbf{u},\, \mathcal{L}_\alpha^\top \boldsymbol{\phi}^{(m)} \right\rangle.
\end{aligned}
$$

Here $\left(\mathcal{L}_\alpha^\top \boldsymbol{\phi}^{(m)}\right)_i = \Delta t^{-\alpha} \psi_i^{(m)}(\alpha)$ is the discrete right-sided (anti-causal) GL fractional derivative of order $\alpha$ acting on the deterministic $C_c^p$ test function $\phi$ rather than on the rough noisy data $\mathbf{u}$. Because $\phi \in C_c^p([t_a, t_b])$ vanishes to order $p > \alpha$ at both endpoints $t_a$ and $t_b$, its continuous right-sided Riemann–Liouville derivative $\left({}_{t}\mathcal{D}_{t_b}^\alpha \phi\right)(t)$ exists everywhere on $[0, t_b]$, is bounded and square-integrable, and satisfies the exact scale invariance

$$
\left({}_{t}\mathcal{D}_{t_b}^\alpha \phi\right)(t) = T_w^{-(\alpha + 1)} \left({}_{\tau}\mathcal{D}_{1}^\alpha \hat{\phi}\right)\!\left(\frac{t - t_a}{T_w}\right), \qquad T_w = H \Delta t,
$$

with first-order quadrature consistency $\Delta t^{-\alpha} \psi_i^{(m)}(\alpha) = \left({}_{t}\mathcal{D}_{t_b}^\alpha \phi\right)(t_i) + \mathcal{O}(\Delta t)$. Substituting $\tilde{\mathbf{u}} = \mathbf{u}_{\text{clean}} + \boldsymbol{\eta}$ with $\boldsymbol{\eta} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$, the weak-form target noise variance is

$$
\begin{aligned}
\frac{\mathrm{Var}\!\left(y_{\text{weak}}^{(m)}(\alpha)\right)}{\sigma^2}
&= \Delta t^{2 - 2\alpha} \sum_{i=0}^{k_b} \left(\psi_i^{(m)}(\alpha)\right)^2
= \Delta t \sum_{i=0}^{k_b} \left|\left({}_{t}\mathcal{D}_{t_b}^\alpha \phi\right)(t_i)\right|^2 \Delta t + \mathcal{O}(\Delta t^2) \\
&= \Delta t \, T_w^{-(2\alpha + 1)} C_{p, \alpha} + \mathcal{O}(\Delta t^2)
= H^{-(2\alpha + 1)} \Delta t^{-2\alpha} C_{p, \alpha} + \mathcal{O}(\Delta t^2),
\end{aligned}
$$

where $C_{p, \alpha} = \int_{-\infty}^{1} \left|({}_{\tau}\mathcal{D}_1^\alpha \hat{\phi})(\tau)\right|^2 d\tau < \infty$ is an $\mathcal{O}(1)$ dimensionless constant depending only on $(p, \alpha)$. Taking the ratio of pointwise to weak-form noise variance per target sample gives

$$
\frac{\mathrm{Var}\!\left((\mathcal{D}_t^\alpha \eta)_{N_t - 1}\right)}{\mathrm{Var}\!\left(y_{\text{weak}}^{(m)}(\alpha)\right)}
= \frac{\Delta t^{-2\alpha} \|\mathbf{w}^{(\alpha)}\|_2^2}{\Delta t \, T_w^{-(2\alpha + 1)} C_{p,\alpha}}
= H^{2\alpha + 1} \frac{\|\mathbf{w}^{(\alpha)}\|_2^2}{C_{p,\alpha}} = \mathcal{O}\!\left(H^{2\alpha + 1}\right). \qquad \blacksquare
$$

**Consequence for SNR Breakdown Brackets:**
1. Because $H = T_w / \Delta t$, the factor $H^{-(2\alpha + 1)}$ cancels the grid-step blow-up $\Delta t^{-2\alpha}$ identically, leaving weak-form variance governed by the macroscopic window scale $T_w^{-(2\alpha + 1)} \Delta t$.
2. Because the suppression gain $10 \log_{10}(H^{2\alpha + 1}) = (2\alpha + 1) \, 10 \log_{10} H$ grows linearly with $\alpha$, higher fractional orders—which suffer the worst pointwise amplification—receive the largest variance reduction ($+34.0\text{ dB}$, $+40.8\text{ dB}$, and $+47.6\text{ dB}$ per window at $H = 50$ for $\alpha = 0.5, 0.7, 0.9$).
3. Across $N_{\text{trials}} = 500$ Monte Carlo realizations on the three-field reaction–diffusion system, this order-dependent rescue compensates the rising pointwise fragility ($[30, 35]\text{ dB}$, $[35, 40]\text{ dB}$, and $[40, 45]\text{ dB}$ for $\alpha_t = 0.5, 0.7, 0.9$), collapsing all three breakdown thresholds to a **common $[15, 20]\text{ dB}$ bracket**—a net **15 dB, 20 dB, and 25 dB** improvement.

---

## Worked Examples

### Example 1: Fractional ODE Discovery under Measurement Noise

```python
import numpy as np
from weaksindy import (
    FourierLibrary,
    CombinedLibrary,
    PolynomialLibrary,
    SR3,
    WeakSINDy,
    integrate_fractional_ode,
)

# Fractional damped Duffing-like oscillator: D_t^0.8 x = y, D_t^0.8 y = -x - 0.5 y - x^3
t = np.linspace(0.0, 15.0, 750)
X_clean = integrate_fractional_ode(
    lambda s: np.array([s[1], -s[0] - 0.5 * s[1] - s[0] ** 3]),
    y0=[1.0, 0.0],
    t=t,
    alpha=0.8,
)

# Corrupt with 20 dB Gaussian measurement noise
rng = np.random.default_rng(42)
X_noisy = X_clean + rng.normal(0.0, 0.1 * np.std(X_clean, axis=0), size=X_clean.shape)

# Combine polynomial and trigonometric candidate bases with SR3 relaxation
library = CombinedLibrary([PolynomialLibrary(degree=3), FourierLibrary(frequencies=(1.0,))])
model = WeakSINDy(
    library=library,
    optimizer=SR3(threshold=0.08, nu=0.5, reg_type="l0"),
    window_size=50,
    stride=5,
)
model.fit(t, X_noisy, orders=[0.4, 0.6, 0.8, 1.0], input_features=["x", "y"])
model.print()
```

### Example 2: Coupled Reaction–Diffusion PDE & Benettin Tangent-Space Lyapunov Spectrum

```python
import numpy as np
from scipy.integrate import solve_ivp
from weaksindy import PDELibrary, WeakSINDy, compute_benettin_lyapunov

# 1D spatial grid with homogeneous Neumann zero-flux boundaries
nx = 60
L = 10.0
x = np.linspace(0.0, L, nx)
dx = float(x[1] - x[0])
t = np.linspace(0.0, 5.0, 200)

def neumann_lap(u: np.ndarray) -> np.ndarray:
    lap = np.empty_like(u)
    lap[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (dx ** 2)
    lap[0] = (2.0 * u[1] - 2.0 * u[0]) / (dx ** 2)
    lap[-1] = (2.0 * u[-2] - 2.0 * u[-1]) / (dx ** 2)
    return lap

# Three-field vascular-stromal reaction-diffusion PDE (pressure p, oxygen c, vessels n)
def pde_rhs(_t: float, y: np.ndarray) -> np.ndarray:
    p, c, n = y[:nx], y[nx : 2 * nx], y[2 * nx :]
    dp = 0.05 * neumann_lap(p) - 0.05 * p + 0.10 * n * (1.0 - p)
    dc = 0.10 * neumann_lap(c) - 0.20 * c * (p / (1.0 + p)) + 0.30 * n
    dn = 0.01 * neumann_lap(n) + 0.20 * n * (1.0 - n) * (c / (1.0 + c)) - 0.10 * n * p
    return np.concatenate([dp, dc, dn])

y0 = np.concatenate([
    np.exp(-((x - 0.5 * L) ** 2) / 2.0),
    1.0 - 0.5 * np.exp(-((x - 0.5 * L) ** 2) / 2.0),
    0.1 + 0.2 * np.sin(np.pi * x / L),
])

sol = solve_ivp(pde_rhs, [0.0, 5.0], y0, t_eval=t, method="Radau")
U = np.stack([sol.y[:nx].T, sol.y[nx : 2 * nx].T, sol.y[2 * nx :].T], axis=-1)

# Weak-form PDE order selection and sparse identification
pde_lib = PDELibrary(
    dx=dx,
    poly_degree=2,
    derivative_orders=(1, 2),
    fractional_orders=(0.5, 1.5),
    include_michaelis_menten=True,
    boundary="neumann",
)
pde_model = WeakSINDy(library=pde_lib, threshold=0.005, window_size=30, stride=4, k_start=10)
pde_model.fit(t, U, orders=[0.6, 0.8, 1.0], input_features=["p", "c", "n"])
print("Selected PDE temporal order:", pde_model.order_)  # 1.0

# Continuous variational Benettin integration confirming global asymptotic stability
lyap = compute_benettin_lyapunov(pde_rhs, y0, n_exponents=1, t_trans=20.0, t_run=60.0, dt=0.01)
print(f"Benettin largest Lyapunov exponent: {lyap.lambda_max:.6f}")  # ~ -0.0734 (< 0, stable)
```

### Example 3: N-Dimensional Spatiotemporal Tensor PDEs & Custom/Rational Bases

```python
import numpy as np
from weaksindy import CombinedLibrary, CustomLibrary, PDELibrary, RationalLibrary, WeakSINDy

# 2D spatiotemporal field U of shape (N_t, N_x, N_y, 1) with separable C^p tensor weak IBP
nt, nx, ny = 121, 32, 32
t = np.linspace(0.0, 2.0, nt)
x = np.linspace(0.0, 2.0 * np.pi, nx, endpoint=False)
dx = float(x[1] - x[0])
T_g, X_g, Y_g = np.meshgrid(t, x, x, indexing="ij")
U = (np.exp(-0.30 * T_g) * np.sin(X_g) * np.cos(Y_g))[..., np.newaxis]

pde_2d = PDELibrary(
    dx=(dx, dx),
    poly_degree=2,
    derivative_orders=(),
    include_laplacian=True,
    include_biharmonic=True,
    riesz_orders=(1.5,),
    include_bias=False,
)
model_2d = WeakSINDy(
    library=pde_2d,
    threshold=0.02,
    window_size=30,
    spatial_window_size=(12, 12),
    spatial_stride=(4, 4),
)
model_2d.fit(t, U, orders=1.0, input_features=["u"])
model_2d.print()  # D_t u = 0.1500 laplacian(u)

# Combine arbitrary user-defined nonlinearities and Hill/Michaelis-Menten rational bases
bio_lib = CombinedLibrary([
    RationalLibrary(numerator_degree=1, denominator_degree=2, half_saturation=1.0),
    CustomLibrary(functions=[lambda u: np.sin(u)], function_names=[lambda s: f"sin({s})"]),
])
```

---

## Public API Reference

| Symbol | Module | Description |
| :--- | :--- | :--- |
| `WeakSINDy` | `weaksindy.solvers` | Weak-form / fractional SINDy estimator supporting discrete `orders=[...]` and continuous bounded Brent refinement (`optimize_order=True`, `order_profile_`, `aic_`, `bic_`). |
| `EnsembleWeakSINDy` | `weaksindy.ensemble` | Subdomain-bagging & library-subsampling ensemble exposing `inclusion_probabilities_`, `coef_median_`, `coef_std_`, and split-conformal intervals `coef_ci(alpha=0.05)`. |
| `fractional_derivative_gl`, `fractional_laplacian` | `weaksindy.operators` | Vectorized causal Grünwald–Letnikov derivative $\mathcal{D}^\alpha$ and symmetric Riesz / spectral fractional Laplacian $(-\Delta)^{\beta/2}$ in 1D–3D. |
| `project_weak_form`, `contract_tensor_weak` | `weaksindy.test_functions` | Separable N-D tensor-product $C^p$ weak-form integration by parts for temporal fractional and spatial integer/fractional PDE operators. |
| `PDELibrary`, `PolynomialLibrary`, `FourierLibrary`, `RationalLibrary`, `CustomLibrary`, `CombinedLibrary` | `weaksindy.library` | Composable candidate libraries for ODEs, biochemical Hill/Michaelis–Menten kinetics, custom nonlinearities, and 1D/2D/3D fractional PDEs. |
| `simulate_fractional_lorenz`, `simulate_subdiffusive_fisher_kpp_1d`, `simulate_vascular_stromal_rd` | `weaksindy.benchmarks` | Canonical fractional ODE and reaction–diffusion PDE synthetic data generators with exact symbolic ground-truth metadata (`BenchmarkDataset`). |
| `compute_benettin_lyapunov` | `weaksindy.lyapunov` | Continuous RK4 variational QR integrator for leading $k$ tangent-space Lyapunov exponents (Jacobian-free or analytical Jacobian). |
| `STLSQ`, `SR3` | `weaksindy.solvers` | Sequentially Thresholded Least Squares and Sparse Relaxed Regularized Regression ($\ell_0$ / $\ell_1$) solvers. |
| `noise_amplification_factor`, `weak_noise_amplification_factor` | `weaksindy.operators`, `weaksindy.test_functions` | Analytical white-noise variance amplification metrics $A(\alpha)$ and $A_{\text{weak}}(\alpha)$. |

---

## Citation

If you use `weakSINDY` in academic work, please cite the accompanying *Chaos* (AIP Publishing) manuscript and Zenodo software archive (see [`CITATION.cff`](CITATION.cff)):

> Karlin, A. (2026). Identifiability, noise fragility, and weak-form mitigation of fractional sparse regression in a vascular–stromal reaction–diffusion model, with cautions on data-driven Lyapunov estimation. *Chaos: An Interdisciplinary Journal of Nonlinear Science* (AIP Publishing). Zenodo DOI: [10.5281/zenodo.20637003](https://doi.org/10.5281/zenodo.20637003).

```bibtex
@article{karlin2026weaksindy,
  author  = {Karlin, Avery},
  title   = {Identifiability, noise fragility, and weak-form mitigation of fractional sparse regression in a vascular--stromal reaction--diffusion model, with cautions on data-driven {L}yapunov estimation},
  journal = {Chaos: An Interdisciplinary Journal of Nonlinear Science},
  year    = {2026},
  doi     = {10.5281/zenodo.20637003}
}
```

## License

Released under the [MIT License](LICENSE).
