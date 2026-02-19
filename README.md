<p align="center">
  <img src="docs/src/assets/logo.png" width="200" alt="NoLimits.jl logo"/>
</p>


<p align="center">
  <a href="https://github.com/manuhuth/NoLimits.jl/actions/workflows/CI.yml">
    <img src="https://github.com/manuhuth/NoLimits.jl/actions/workflows/CI.yml/badge.svg?branch=main" alt="CI"/>
  </a>
  <a href="https://manuhuth.github.io/NoLimits.jl">
    <img src="https://img.shields.io/badge/docs-stable-blue.svg" alt="Documentation"/>
  </a>
  <a href="https://codecov.io/gh/manuhuth/NoLimits.jl">
    <img src="https://codecov.io/gh/manuhuth/NoLimits.jl/branch/main/graph/badge.svg" alt="Coverage"/>
  </a>
</p>

NoLimits.jl provides a unified, open-source framework for specifying, estimating, and diagnosing hierarchical models of longitudinal data. It is designed for life-science applications — from pharmacokinetics and systems biology to ecology and neuroscience — where population variability, mechanistic dynamics, and complex outcome structures must be modeled jointly.



## Why NoLimits.jl?

Nonlinear mixed-effects (NLME) models are the standard tool for longitudinal analysis in the biomedical sciences, but existing software enforces trade-offs between expressiveness, estimation flexibility, and modern machine-learning integration. NoLimits.jl addresses these gaps through a single, composable modeling language in which mechanistic structure, learned components, flexible random-effect distributions, and diverse outcome types coexist within one coherent specification — and can be estimated with multiple inference strategies without rewriting the model.



## Key Features

### Composable Model Specification

| Component | Capabilities |
|---|---|
| **Structural model** | Algebraic functions, ODE systems (via OrdinaryDiffEq.jl), derived signals |
| **Machine-learning blocks** | Neural networks (Lux.jl), soft decision trees, B-splines — embeddable in formulas, ODE right-hand sides, initial conditions, or RE distributions |
| **Random effects** | Univariate and multivariate; multiple grouping structures simultaneously (e.g., subject + site) |
| **RE distributions** | Gaussian, non-Gaussian (heavy-tailed, skewed, positive-valued), normalizing planar flows |
| **Outcome model** | Normal, LogNormal, Poisson, Bernoulli, NegativeBinomial, and arbitrary `Distributions.jl` families; hidden Markov models with random effects |
| **Covariates** | Time-varying, group-constant, and interpolated dynamic covariates (8 interpolation types) |
| **Censoring** | Left-censored and interval-censored observations |

All components are freely composable: a single model can simultaneously use ODE dynamics, neural-network subterms, multiple RE grouping levels with flow-based distributions, and mixed outcome types.

### Unified Estimation API

All methods share a single `fit_model` interface, enabling direct comparison across inference paradigms on the same model and dataset.

| Inference paradigm | Methods |
|---|---|
| **Frequentist (fixed effects)** | MLE, MAP, Multistart |
| **Frequentist (mixed effects)** | Laplace, LaplaceMAP, FOCEI, FOCEIMAP, SAEM, MCEM |
| **Bayesian** | MCMC (Turing.jl) |

### Uncertainty Quantification

- Wald intervals (Hessian or sandwich covariance)
- Profile-likelihood intervals (LikelihoodProfiler.jl)
- MCMC-based posterior intervals
- All accessible through a unified `compute_uq` interface

### Diagnostics and Visualization

Visual predictive checks (VPCs), residual diagnostics (QQ, PIT, ACF), random-effects distribution diagnostics, observation-level predictive distribution plots, multistart waterfall plots, and UQ parameter distribution plots.



## Quickstart

The example below fits a one-compartment pharmacokinetic model with subject-level random effects on clearance and volume using a Laplace approximation.

```julia
using NoLimits, DataFrames, Distributions, OrdinaryDiffEq

# --- 1. Define the model ---
model = @Model begin
    @fixedEffects begin
        log_cl   = RealNumber(log(5.0))          # log-clearance (population)
        log_v    = RealNumber(log(30.0))          # log-volume (population)
        omega_cl = RealNumber(0.3, scale=:log)    # RE SD for clearance
        omega_v  = RealNumber(0.3, scale=:log)    # RE SD for volume
        sigma    = RealNumber(0.2, scale=:log)    # residual SD
    end

    @covariates begin
        t = Covariate()                           # observation time column
    end

    @randomEffects begin
        eta_cl = RandomEffect(Normal(0.0, omega_cl); column=:ID)
        eta_v  = RandomEffect(Normal(0.0, omega_v);  column=:ID)
    end

    @preDifferentialEquation begin
        cl = exp(log_cl + eta_cl)
        v  = exp(log_v  + eta_v)
    end

    @DifferentialEquation begin
        D(A) ~ -(cl / v) * A                     # one-compartment elimination
    end

    @initialDE begin
        A = 100.0                                 # bolus dose at t = 0
    end

    @formulas begin
        conc = A(t) / v
        y ~ Normal(conc, sigma)
    end
end

# --- 2. Bind to data ---
dm = DataModel(model, df; primary_id=:ID, time_col=:t)

# --- 3. Fit ---
res = fit_model(dm, Laplace())

# --- 4. Inspect results ---
get_params(res; scale=:untransformed)
re  = get_random_effects(res)
uq  = compute_uq(res; method=:wald)

# --- 5. Diagnostics ---
plot_fits(res)
plot_vpc(res; n_simulations=200)
plot_residuals(res)
```

More examples — including neural-ODE models, HMM outcomes, normalizing-flow random effects, and multi-method comparison — are available in the [Tutorials](https://manuhuth.github.io/NoLimits.jl/tutorials/).




## Installation

NoLimits.jl requires Julia 1.12 or later. Install directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/manuhuth/NoLimits.jl")
```

Registry-based installation (`Pkg.add("NoLimits")`) will be available once the package is registered in the Julia General Registry.



## Documentation

Full documentation, including tutorials, method descriptions, and API reference:

**[https://manuhuth.github.io/NoLimits.jl](https://manuhuth.github.io/NoLimits.jl)**



## Citation

A manuscript describing NoLimits.jl is in preparation. In the meantime, if you use this package in published work, please cite this repository:

```bibtex
@software{NoLimits_jl_2026,
  title  = {{NoLimits.jl}},
  author = {Huth, Manuel and Arruda, Jonas and Hasenauer, Jan},
  year   = {2026},
  url    = {https://github.com/manuhuth/NoLimits.jl}
}
```



## License

NoLimits.jl is released under the MIT License. See [LICENSE](LICENSE) for details.
