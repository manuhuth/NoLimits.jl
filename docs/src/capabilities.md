# Capabilities

NoLimits.jl provides a broad set of modeling, estimation, and diagnostic capabilities for nonlinear mixed-effects analysis. All features listed below are implemented and tested in the current package.

## Modeling

- **Nonlinear mixed-effects models** for longitudinal data, with one or multiple random-effect grouping structures (e.g., subject-level and site-level variability in a single model).
- **Fixed-effects-only models** for settings where random effects are not needed.
- **ODE-based and non-ODE models** within the same modeling framework: algebraic structural models and mechanistic ODE systems share the same specification language.
- **Multiple outcomes** in one model, including mixed outcome types (e.g., continuous and count outcomes jointly).
- **Hidden Markov outcome models** via `DiscreteTimeDiscreteStatesHMM` and `ContinuousTimeDiscreteStatesHMM`, enabling latent-state-dependent observation processes.
- **Left-censored and interval-censored observations** through `censored(...)` in the observation model.

## Random-Effects Flexibility

- Univariate and multivariate random effects.
- Random-effect distributions beyond the Gaussian family: heavy-tailed (`TDist`), skewed (`SkewNormal`), positive-valued (`LogNormal`, `Gamma`), and other distributions from `Distributions.jl`.
- Flow-based random effects via `NormalizingPlanarFlow` for highly flexible latent distributions.
- Random-effect distributions parameterized by covariates, neural networks, soft decision trees, or spline functions -- enabling covariate-dependent heterogeneity that goes beyond standard variance models.

## Machine-Learning Integration

- **Neural-network parameter blocks** (`NNParameters`) can be embedded in formulas, ODE dynamics, initial conditions, and random-effect distribution parameterizations. Neural-ODE-style models arise naturally when learned components appear inside `@DifferentialEquation`.
- **Soft decision tree parameter blocks** (`SoftTreeParameters`) provide an alternative differentiable function approximator with the same integration points as neural networks.
- **Spline parameter blocks** (`SplineParameters`) for smooth, learnable basis-function expansions.
- All learned components can be used at the population level (fixed effects only) or individualized through full-parameter random effects.

## Covariate Handling

- **Time-varying covariates** that change across observations within an individual.
- **Constant covariates** that are fixed within a grouping level (e.g., baseline age, treatment arm).
- **Dynamic (interpolated) covariates** that provide continuous-time access within ODE integration, with support for eight interpolation methods from `DataInterpolations.jl`.

## Estimation Methods

| Model type | Available methods |
| --- | --- |
| Mixed-effects | Laplace approximation, LaplaceMAP, MCEM, SAEM, MCMC |
| Fixed-effects only | MLE, MAP, MCMC |
| Cross-method | Multistart optimization wrapper |

All methods share a unified `fit_model` interface, allowing direct comparison of estimation approaches on the same model and data.

## Uncertainty Quantification

- **Wald-based intervals** from Hessian or sandwich covariance estimates.
- **Profile-likelihood intervals** via `LikelihoodProfiler.jl`.
- **MCMC-based intervals** from posterior chains (direct or refit).
- A unified `compute_uq` interface across all backends.

## Diagnostics and Visualization

- Fitted-trajectory plots overlaid on observed data.
- Visual predictive checks (VPCs).
- Residual diagnostics: QQ plots, PIT histograms, autocorrelation plots.
- Random-effects diagnostics: marginal density plots, pairwise scatter, standardized EBE distributions.
- Observation-distribution plots showing full predictive distributions at selected data points.
- Uncertainty distribution plots for estimated parameters.
- Multistart objective waterfalls and parameter stability summaries.

## Composability

A defining feature of NoLimits.jl is that the capabilities above are freely composable. A single model can simultaneously use ODE dynamics, multiple learned function approximators, several random-effect grouping levels with non-Gaussian distributions, covariates at different temporal resolutions, and multiple outcome types. This composability is central to the package design and is exercised throughout the test suite.
