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
models of longitudinal data, aimed at life-science applications such as pharmacometrics and
systems biology where population variability and mechanistic dynamics must be modeled together.
It is developed and maintained by the
[Hasenauer Lab](https://www.mathematics-and-life-sciences.uni-bonn.de/en/research/hasenauer-group)
at the University of Bonn, with
[Manuel Huth](https://www.mathematics-and-life-sciences.uni-bonn.de/en/group-members/people/hasenauer-group-members/manuel-huth)
as lead developer.

## Why NoLimits.jl?

Nonlinear mixed-effects (NLME) models are the standard tool for longitudinal analysis in the
biomedical sciences, but existing software forces trade-offs between *expressiveness*,
*estimation flexibility*, and *machine-learning integration*: mechanistic ODE tools rarely
support latent-state outcomes or learned components, general mixed-effects packages do not
handle ODE systems, and probabilistic programming languages leave the NLME machinery to the user.

NoLimits.jl provides a single, composable modeling language in which mechanistic structure,
learned components, flexible random-effect distributions, and diverse outcome types coexist in
one specification, and can be estimated with several inference paradigms without rewriting the
model. It is built for mixed-effects models but works equally for fixed-effects-only analysis.

## Key Features

### Composable model specification

A model is assembled from freely composable blocks: a structural model (algebraic functions,
ODE systems via OrdinaryDiffEq.jl, and derived signals); differentiable machine-learning
components (neural networks, soft decision trees, B-splines) embeddable in formulas, ODE
right-hand sides, initial conditions, or random-effect distributions; univariate and
multivariate random effects over multiple grouping structures (e.g. subject and site) with
Gaussian, non-Gaussian, or normalizing-flow distributions; outcomes from any `Distributions.jl`
family as well as observed-state and hidden Markov models; time-varying, constant, and
interpolated covariates; and likelihood-based handling of missing and censored data. A single
model can use all of these at once.

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

Visual predictive checks (VPCs); residual diagnostics (QQ, PIT, ACF); random-effects
distribution diagnostics; observation-level predictive plots; UQ parameter distributions; and
cross-validation with principled random-effects predictions for seen and unseen individuals.

See [Capabilities](https://manuhuth.github.io/NoLimits.jl/dev/capabilities) for the complete list.

## Installation

NoLimits.jl requires Julia 1.12 or later. It is registered in the Julia General Registry, so
you can install it with:

```julia
using Pkg
Pkg.add("NoLimits")
```

or from the REPL package mode (press `]`):

```
pkg> add NoLimits
```

To install the latest development version directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/manuhuth/NoLimits.jl")
```

## Quickstart

This example fits an exponential-decay model with a subject-level random intercept by Laplace
approximation. The snippet is self-contained — copy it into a Julia session and run.

```julia
using NoLimits, DataFrames, Distributions, CairoMakie

model = @Model begin
    @fixedEffects begin
        A0    = RealNumber(10.0, scale=:log)   # population baseline
        k     = RealNumber(0.5,  scale=:log)   # population decay rate
        omega = RealNumber(0.3,  scale=:log)   # between-subject SD
        sigma = RealNumber(0.5,  scale=:log)   # residual SD
    end

    @covariates begin
        time = Covariate()
    end

    @randomEffects begin
        eta = RandomEffect(Normal(0.0, omega); column=:ID)
    end

    @formulas begin
        pred = A0 * exp(eta) * exp(-k * time)
        y ~ Normal(pred, sigma)
    end
end

df = DataFrame(
    ID   = repeat([:s1, :s2, :s3, :s4], inner=4),
    time = repeat([0.0, 1.0, 2.0, 4.0], outer=4),
    y    = [10.2, 6.1, 3.6, 1.4, 12.5, 7.8, 4.9, 1.9,
             8.1, 4.9, 3.0, 1.1, 11.0, 6.5, 4.1, 1.6],
)

dm  = DataModel(model, df; primary_id=:ID, time_col=:time)
res = fit_model(dm, Laplace())

get_params(res; scale=:untransformed)   # population estimates
get_random_effects(res)                 # per-subject effects
compute_uq(res; method=:wald)           # uncertainty
plot_fits(res)                          # fit vs. data
```

Swapping the inference paradigm is a one-line change: `fit_model(dm, SAEM())`, `fit_model(dm, MCEM())`,
or `fit_model(dm, MCMC())` all fit the *same* model. More examples — neural-ODE models, Markov-model
outcomes, normalizing-flow random effects, count outcomes, censored data, and multi-method
comparison — are in the [Tutorials](https://manuhuth.github.io/NoLimits.jl/dev/tutorials/mixed-effects-multiple-methods).

## Built on the Julia ecosystem

NoLimits.jl builds directly on established Julia packages, so their interfaces stay familiar:
[OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) for ODE solving,
[Distributions.jl](https://github.com/JuliaStats/Distributions.jl) for outcome and random-effect
distributions, [Turing.jl](https://github.com/TuringLang/Turing.jl) for MCMC and variational
inference, [Lux.jl](https://github.com/LuxDL/Lux.jl) and
[SimpleChains.jl](https://github.com/PumasAI/SimpleChains.jl) for neural-network components,
[ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) for automatic differentiation, and
[Optimization.jl](https://github.com/SciML/Optimization.jl) for optimization — among others.

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

If you use NoLimits.jl in published work, please cite it — GitHub's **"Cite this repository"**
button (from [`CITATION.cff`](CITATION.cff)) provides APA and BibTeX exports:

```bibtex
@software{NoLimits_jl_2026,
  title  = {{NoLimits.jl}: Flexible and composable nonlinear mixed-effects modeling in Julia},
  author = {Huth, Manuel and Arruda, Jonas and Peiter, Clemens and Gusinow, Roy and Schmid, Nina and Hasenauer, Jan},
  year   = {2026},
  url    = {https://github.com/manuhuth/NoLimits.jl}
}
```

## Development and AI assistance

NoLimits.jl was developed with substantial assistance from large language models, primarily
Anthropic's Claude (via Claude Code), for code generation, refactoring, test authoring, and
documentation. All contributions were reviewed, tested, and are understood by the maintainers,
who take full responsibility for the package — disclosed per the Julia General Registry's
guidance on AI-assisted packages.

## License

NoLimits.jl is released under the MIT License. See [LICENSE](LICENSE) for details.
