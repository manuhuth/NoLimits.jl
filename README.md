<p align="center">
  <img src="docs/src/assets/logo.png" width="200" alt="NoLimits.jl logo"/>
</p>

<h1 align="center">NoLimits.jl</h1>

<p align="center">
  <strong>NoLimits</strong> stands for <strong>NO</strong>n <strong>LI</strong>near <strong>MI</strong>xed effec<strong>TS</strong>.
</p>

<p align="center">
  <em>Nonlinear mixed-effects modeling for longitudinal data: mechanistic ODEs, Markov models,<br/>
  differentiable machine learning components, and frequentist and Bayesian estimation,<br/>
  composed in one framework and fit through one interface.</em>
</p>

<p align="center">
  <a href="https://github.com/manuhuth/NoLimits.jl/actions/workflows/CI.yml">
    <img src="https://github.com/manuhuth/NoLimits.jl/actions/workflows/CI.yml/badge.svg?branch=main" alt="CI"/>
  </a>
  <a href="https://manuhuth.github.io/NoLimits.jl/dev/">
    <img src="https://img.shields.io/badge/docs-dev-blue.svg" alt="Documentation"/>
  </a>
  <a href="https://codecov.io/gh/manuhuth/NoLimits.jl">
    <img src="https://codecov.io/gh/manuhuth/NoLimits.jl/branch/main/graph/badge.svg" alt="Coverage"/>
  </a>
  <a href="https://github.com/JuliaTesting/Aqua.jl">
    <img src="https://juliatesting.github.io/Aqua.jl/dev/assets/badge.svg" alt="Aqua QA"/>
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"/>
  </a>
  <a href="https://julialang.org">
    <img src="https://img.shields.io/badge/Julia-1.12%2B-9558B2.svg?logo=julia&logoColor=white" alt="Julia 1.12+"/>
  </a>
  <a href="https://www.repostatus.org/#active">
    <img src="https://www.repostatus.org/badges/latest/active.svg" alt="Project Status: Active"/>
  </a>
</p>

NoLimits.jl is an open-source framework for building, estimating, and diagnosing hierarchical
models of longitudinal data. It is aimed at life-science applications such as pharmacometrics
and systems biology, where population variability and mechanistic dynamics must be modeled
together.

The package is developed and maintained by the
[Hasenauer Lab](https://www.mathematics-and-life-sciences.uni-bonn.de/en/research/hasenauer-group)
at the University of Bonn, with
[Manuel Huth](https://www.mathematics-and-life-sciences.uni-bonn.de/en/group-members/people/hasenauer-group-members/manuel-huth)
as the lead developer.

## Why NoLimits.jl?

Nonlinear mixed-effects (NLME) models are the standard tool for longitudinal analysis in the
biomedical sciences, but existing software enforces trade-offs between *expressiveness*,
*estimation flexibility*, and *modern machine-learning integration*. Mechanistic ODE tools
rarely support latent-state outcomes or learned components; general mixed-effects packages do
not handle ODE systems; and probabilistic programming languages leave the NLME machinery to
the user.

NoLimits.jl provides a single, composable modeling language in which
mechanistic structure, learned components, flexible random-effect distributions, and diverse
outcome types coexist in one coherent specification, and can be estimated with multiple
inference paradigms without rewriting the model.

NoLimits.jl is designed for mixed-effects models, but it can equally be used for
fixed-effects-only analysis when random effects are not required.

## Key Features

### Composable model specification

A model is assembled from freely composable blocks: a structural model (algebraic functions,
ODE systems via OrdinaryDiffEq.jl, and derived signals); differentiable machine-learning
components (neural networks, soft decision trees, and B-splines) embeddable in formulas, ODE
right-hand sides, initial conditions, or random-effect distributions; univariate and
multivariate random effects over multiple simultaneous grouping structures (e.g., subject and
site), with Gaussian, non-Gaussian, or normalizing-flow distributions optionally parameterized
by covariates and learned functions; outcomes from any `Distributions.jl` family (such as
Normal, LogNormal, Poisson, Bernoulli, and NegativeBinomial) as well as observed-state and
hidden Markov models; time-varying, group-constant, and interpolated dynamic covariates (eight
interpolation types); and likelihood-based handling of missing data and left- or
interval-censored observations. A single model can use all of these at once.

### One model, many estimators

Every method shares a single `fit_model` interface, so inference paradigms can be compared
directly on the same model and dataset without rewriting it.

| Inference paradigm | Methods |
|---|---|
| **Fixed effects** | MLE, MAP, MCMC, VI |
| **Mixed effects** | Laplace, FOCEI, GHQuadrature, MCEM, SAEM, MCMC |
| **Pooled (mixed effects)** | Pooled, PooledMap |
| **Cross-method** | Multistart |

### Uncertainty quantification

A unified `compute_uq` interface exposes:

- Wald intervals (Hessian or sandwich covariance)
- Profile-likelihood intervals (LikelihoodProfiler.jl)
- Posterior intervals from MCMC chains or VI variational posteriors

### Diagnostics and visualization

Visual predictive checks (VPCs), residual diagnostics (QQ, PIT, ACF), random-effects
distribution diagnostics, observation-level predictive distribution plots, multistart waterfall
plots, UQ parameter distributions, and cross-validation workflows with principled handling of
random-effects predictions for both seen and unseen individuals.

## Installation

NoLimits.jl requires Julia 1.12 or later. Install directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/manuhuth/NoLimits.jl")
```

Registry-based installation (`Pkg.add("NoLimits")`) will be available once the package is
registered in the Julia General Registry.

## Quickstart

The example below fits a one-compartment pharmacokinetic model with subject-level random effects
on clearance and volume using a Laplace approximation.

```julia
using NoLimits, DataFrames, Distributions, OrdinaryDiffEq, Random

# --- 1. Define the model ---
model = @Model begin
    @fixedEffects begin
        log_cl   = RealNumber(log(5.0))           # log-clearance (population)
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
        D(A) ~ -(cl / v) * A                      # one-compartment elimination
    end

    @initialDE begin
        A = 100.0                                 # bolus dose at t = 0
    end

    @formulas begin
        conc = A(t) / v
        y ~ Normal(conc, sigma)
    end
end

# --- 2. Simulate a small dataset (or bring your own DataFrame with columns :ID, :t, :y) ---
rng   = MersenneTwister(1)
times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
df = DataFrame(ID=Int[], t=Float64[], y=Float64[])
for id in 1:12
    cl = 5.0  * exp(0.3 * randn(rng))      # subject-specific clearance
    v  = 30.0 * exp(0.3 * randn(rng))      # subject-specific volume
    for t in times
        conc = (100.0 / v) * exp(-(cl / v) * t)
        push!(df, (id, t, conc + 0.2 * randn(rng)))
    end
end

# --- 3. Bind to data ---
dm = DataModel(model, df; primary_id=:ID, time_col=:t)

# --- 4. Fit ---
res = fit_model(dm, Laplace())

# --- 5. Inspect results ---
get_params(res; scale=:untransformed)
re  = get_random_effects(res)
uq  = compute_uq(res; method=:wald)

# --- 6. Diagnostics ---
plot_fits(res)
plot_vpc(res; n_simulations=200)
plot_residuals(res)
```

Swapping the inference paradigm is a one-line change: `fit_model(dm, SAEM())`, `fit_model(dm, MCEM())`,
or `fit_model(dm, MCMC())` all fit the *same* model. More examples (neural-ODE models, Markov-model
outcomes, normalizing-flow random effects, count outcomes, censored data, and multi-method
comparison) are in the [Tutorials](https://manuhuth.github.io/NoLimits.jl/dev/tutorials/mixed-effects-multiple-methods).

## Built on the Julia ecosystem

NoLimits.jl integrates directly with established Julia packages, so users familiar with any of
them will find the interfaces immediately recognizable. It interfaces with
[OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) for ODE solving,
[Distributions.jl](https://github.com/JuliaStats/Distributions.jl) for observation and
random-effect distributions,
[Optimization.jl](https://github.com/SciML/Optimization.jl) for numerical optimization,
[Turing.jl](https://github.com/TuringLang/Turing.jl) and
[MCMCChains.jl](https://github.com/TuringLang/MCMCChains.jl) for MCMC sampling and diagnostics,
[Lux.jl](https://github.com/LuxDL/Lux.jl) and
[SimpleChains.jl](https://github.com/PumasAI/SimpleChains.jl) for neural-network components,
[ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) for automatic differentiation,
[ComponentArrays.jl](https://github.com/jonniedie/ComponentArrays.jl) for named parameter arrays,
[DataFrames.jl](https://github.com/JuliaData/DataFrames.jl) for tabular data,
[DataInterpolations.jl](https://github.com/SciML/DataInterpolations.jl) for dynamic covariates,
[LikelihoodProfiler.jl](https://github.com/insysbio/LikelihoodProfiler.jl) for profile-likelihood
uncertainty quantification, and [Plots.jl](https://github.com/JuliaPlots/Plots.jl) for visualization.

## Documentation

Full documentation is hosted at **[manuhuth.github.io/NoLimits.jl](https://manuhuth.github.io/NoLimits.jl/dev/)**.

| Start here | |
|---|---|
| [Installation](https://manuhuth.github.io/NoLimits.jl/dev/installation) · [Quickstart](https://manuhuth.github.io/NoLimits.jl/dev/quickstart) | Get up and running |
| [Capabilities](https://manuhuth.github.io/NoLimits.jl/dev/capabilities) | A concise overview of what the package can do |
| [NLME Methodology](https://manuhuth.github.io/NoLimits.jl/dev/nlme-methodology) | The mathematical foundations |
| [Model Building](https://manuhuth.github.io/NoLimits.jl/dev/model-building/) · [Estimation](https://manuhuth.github.io/NoLimits.jl/dev/estimation/) · [Uncertainty Quantification](https://manuhuth.github.io/NoLimits.jl/dev/uncertainty-quantification/) | Reference guides |
| [Tutorials](https://manuhuth.github.io/NoLimits.jl/dev/tutorials/mixed-effects-multiple-methods) | Hands-on, end-to-end examples |
| [API](https://manuhuth.github.io/NoLimits.jl/dev/api) | Full function reference |

## Getting help & contributing

- **Questions and ideas**: open a [GitHub Discussion](https://github.com/manuhuth/NoLimits.jl/discussions).
- **Bugs and feature requests**: open an [issue](https://github.com/manuhuth/NoLimits.jl/issues);
  a minimal reproducible example helps enormously.
- **Contributions** are welcome. See the
  [How to Contribute](https://manuhuth.github.io/NoLimits.jl/dev/how-to-contribute) and
  [Developers Guide](https://manuhuth.github.io/NoLimits.jl/dev/developers-guide) pages before
  opening a pull request.

## Citation

If you use NoLimits.jl in published work, please cite the software. GitHub's
**"Cite this repository"** button (generated from [`CITATION.cff`](CITATION.cff)) provides
ready-made APA and BibTeX exports; the BibTeX entry is:

```bibtex
@software{NoLimits_jl_2026,
  title  = {{NoLimits.jl}: Flexible and composable nonlinear mixed-effects modeling in Julia},
  author = {Huth, Manuel and Arruda, Jonas and Peiter, Clemens and Gusinow, Roy and Schmid, Nina and Hasenauer, Jan},
  year   = {2026},
  url    = {https://github.com/manuhuth/NoLimits.jl}
}
```

## Development and AI assistance

NoLimits.jl was developed with substantial assistance from large language models,
primarily Anthropic's Claude (via Claude Code), used for code generation, refactoring,
test authoring, and documentation. All contributions were reviewed, tested, and are
understood by the maintainers, who take full responsibility for the correctness and
behavior of the package. This disclosure follows the Julia General Registry's guidance on
AI-assisted packages.

## License

NoLimits.jl is released under the MIT License. See [LICENSE](LICENSE) for details.
