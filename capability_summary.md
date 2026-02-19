# Capability Summary

This file provides a consolidated overview of the current capability stack in `NoLimits.jl` across model building, estimation, uncertainty quantification, and plotting.

## Status and Scope
- Package status: active and broad in functionality, with API evolution still possible.
- Primary domain: nonlinear longitudinal data models with optional random effects, ODE dynamics, and HMM outcomes.
- Central interfaces:
  - Model definition: `@Model ...`
  - Data binding and validation: `DataModel(...)`
  - Estimation: `fit_model(dm, method; kwargs...)`
  - Uncertainty quantification: `compute_uq(res; kwargs...)`
  - Diagnostics and visualization: plotting API in `src/plotting`.

## 1) Model-Building Capabilities

### 1.1 Model DSL and Required Structure
- `@Model` composes these blocks:
  - `@helpers`
  - `@fixedEffects`
  - `@covariates`
  - `@randomEffects`
  - `@preDifferentialEquation`
  - `@DifferentialEquation`
  - `@initialDE`
  - `@formulas`
- Validation rules:
  - `@formulas` is required.
  - `@DifferentialEquation` requires `@initialDE`.
  - At least one fixed effect or random effect is required.

### 1.2 Fixed Effects
- Parameter blocks supported:
  - `RealNumber`, `RealVector`
  - `RealPSDMatrix`, `RealDiagonalMatrix`
  - `NNParameters`, `SoftTreeParameters`, `SplineParameters`, `NPFParameter`
- Transform-aware optimization support:
  - Typical scales include identity, `:log`, matrix transforms such as Cholesky/expm paths.
- Priors supported for Bayesian/MAP workflows.

### 1.3 Helper Functions and Learned Function Blocks
- `@helpers` functions are available in formulas/DE/preDE.
- Functions induced by `NNParameters`, `SoftTreeParameters`, and `SplineParameters` are registered in `model_funs` and callable in formulas/DE/preDE.

### 1.4 Random Effects: Flexible Distribution Support
- Random effects are declared as `RandomEffect(dist; column=:GROUP)`.
- Supports scalar and multivariate random effects.
- Multiple RE groups are supported simultaneously.
- Flexible RE distribution families are supported, including:
  - Gaussian and multivariate Gaussian (`Normal`, `MvNormal`)
  - Skewed/heavy-tail/positive families where valid in model expressions (for example `LogNormal`, `Gamma`, `Weibull`, `InverseGaussian`, `Laplace`, `TDist`, `SkewNormal`, `Gumbel`)
  - `NormalizingPlanarFlow` (flow-based RE distributions)
- RE distributions can be parameterized by model expressions involving:
  - fixed effects
  - helper functions
  - neural-network outputs
  - spline outputs
  - soft-tree outputs
  - constant covariates
- RE distributions parameterized by constant covariates are explicitly supported in model construction, simulation, and estimation paths.

### 1.5 Outcome Formulas: Flexible Distribution Support
- Observation nodes are declared via `lhs ~ Distribution(...)`.
- Supports broad outcome-distribution usage in formulas through distribution expressions.
- Specialized kernels exist for key likelihood paths (`Normal`, `Bernoulli`, `Poisson`, `LogNormal`) for speed.
- Non-Gaussian outcomes are supported in estimation workflows (method-specific details below).
- HMM outcome distributions are supported:
  - `DiscreteTimeDiscreteStatesHMM`
  - `ContinuousTimeDiscreteStatesHMM`

### 1.6 Covariates
- Covariate types:
  - `Covariate`, `CovariateVector` (varying)
  - `ConstantCovariate`, `ConstantCovariateVector` (group-constant)
  - `DynamicCovariate`, `DynamicCovariateVector` (time-interpolated)
- Dynamic covariates support multiple interpolation schemes from `DataInterpolations.jl`.
- Validation includes:
  - missingness checks for used covariates
  - per-individual time sorting for dynamic covariates
  - minimum data-point requirements by interpolation type
  - constant-on-group constraints for constant covariates

### 1.7 ODE, preDE, Initial Conditions, and Events
- `@preDifferentialEquation` supports time-constant derived quantities used by DE/formulas.
- `@DifferentialEquation` supports state dynamics and derived signals.
- `@initialDE` defines initial state values.
- DE covariate usage is validated (for example varying covariates are rejected directly in DE equations).
- Event-aware ODE workflows supported via DataModel event columns (`EVID`, `AMT`, `RATE`, `CMT`).

### 1.8 Solver Configuration
- `set_solver_config(model; saveat_mode=:dense|:saveat|:auto, alg, kwargs, args)`.
- Save-at behavior integrates with formula time offsets and ODE solve strategy.

### 1.9 DataModel Validation and Batching
- `DataModel` validates schema, required columns, missingness, RE grouping coherence, and formula/data consistency.
- Random-effect pairing/batching via transitive overlap across RE grouping columns is built in.
- Identifiability warning emitted when RE grouping is effectively one level per observation.

## 2) Estimation Capabilities

### 2.1 Unified Fit Interface
- `fit_model(dm::DataModel, method::FittingMethod; kwargs...)`
- Shared controls include:
  - `constants` for fixed-effect constants
  - `constants_re` for fixed random-effect levels
  - `penalty` (optimization methods)
  - ODE and serialization controls
  - RNG controls
  - custom start values (`theta_0_untransformed`)

### 2.2 Fixed-Effects-Only Methods
- `MLE`: maximum likelihood optimization.
- `MAP`: posterior mode optimization with priors.
- `MCMC` (Turing): Bayesian sampling.
- ODE and non-ODE models are supported.

### 2.3 Random-Effects Methods
- `Laplace` and `LaplaceMAP`:
  - Batchwise EBE mode optimization + Hessian/logdet corrections.
  - Supports multiple/multivariate REs, ODE models, threading, `constants_re`.
  - Hessian controls: `jitter`, `max_tries`, `growth`, `adaptive`, `scale_factor`, `use_trace_logdet_grad`, `use_hutchinson`, `hutchinson_n`.
- `FOCEI` and `FOCEIMAP`:
  - Information-based approximation with `info_mode=:fisher_common|:custom`.
  - `:fisher_common` supports key outcome families (`Normal`, `LogNormal`, `Bernoulli`, `Poisson`, `Exponential`, `Geometric`, `Binomial`) and core RE priors (`Normal`, `LogNormal`, `Exponential`, `MvNormal`).
  - `:custom` supports arbitrary outcome/RE distributions (for example flow RE with OPG info callback).
- `MCEM`:
  - MCMC E-step + optimization M-step.
  - Supports ODE models, multigroup/multivariate RE, threading.
- `SAEM`:
  - Stochastic-approximation EM with configurable schedules and optional built-in sufficient-statistics updates.
  - Supports ODE models, multigroup/multivariate RE, threading.
- `MCMC` (random effects):
  - Full Bayesian path with fixed effects + RE levels.
  - Supports multivariate RE, non-Gaussian RE, flow RE, ODE models, NN/SoftTree/Spline model components, and HMM outcomes.

### 2.4 Multistart Estimation
- `Multistart(...)` wrapper supports multiple start generation and refits.
- Sampling modes include random and LHS.
- Priors or explicit sampling dists can drive starts.
- Best run and failed runs are accessible via dedicated accessors.

### 2.5 Diagnostics and Accessors
- Standard accessors include:
  - `get_params`, `get_objective`, `get_converged`, `get_loglikelihood`
  - method-specific outputs such as `get_chain`, `get_random_effects`, `get_laplace_random_effects`

### 2.6 Identifiability Diagnostics
- `identifiability_report(...)` supports MLE/MAP/Laplace/FOCEI families (method-compatible by model type).
- Reports Hessian/SVD rank, null directions, condition metrics, and RE information diagnostics.

## 3) Uncertainty Quantification (UQ)

### 3.1 Unified UQ API
- `compute_uq(res::FitResult; kwargs...)`
- Active parameters are fixed-effect coordinates that are both free and marked `calculate_se=true`.

### 3.2 UQ Backends
- `:wald`
  - For optimization/approximation methods including RE methods.
  - Covariance options: Hessian or sandwich.
  - Supports transformed and natural scale summaries.
- `:chain`
  - For `MCMC` fits.
  - Uses posterior draws directly.
- `:profile`
  - Profile-likelihood intervals for supported optimization methods.
- `:mcmc_refit`
  - Refit with MCMC from non-MCMC source fit to obtain chain-based UQ.

### 3.3 UQ Outputs and Accessors
- `UQResult` contains:
  - estimates
  - intervals
  - covariance matrices
  - draws
  - diagnostics
- Accessors include:
  - `get_uq_backend`, `get_uq_source_method`, `get_uq_parameter_names`
  - `get_uq_estimates`, `get_uq_intervals`, `get_uq_vcov`, `get_uq_draws`, `get_uq_diagnostics`

## 4) Plotting and Diagnostics

### 4.1 Plotting Infrastructure
- Core types:
  - `PlotStyle`
  - `PlotCache`
- `build_plot_cache(...)` caches expensive ingredients (ODE solutions, predictive distributions, chain summaries, RE summaries).

### 4.2 Data and Fit Plots
- `plot_data(...)`
- `plot_fits(...)`
  - Supports observed overlays, predictive summaries, and MCMC quantile bands.

### 4.3 Observation Distribution Plots
- `plot_observation_distributions(...)`
- Handles continuous (density) and discrete (PMF) outcomes.
- MCMC posterior summaries are supported.

### 4.4 Residual Diagnostics
- `get_residuals(...)` metrics include:
  - `:pit`, `:quantile`, `:raw`, `:pearson`, `:logscore`
- Plot functions:
  - `plot_residuals`
  - `plot_residual_distribution`
  - `plot_residual_qq`
  - `plot_residual_pit`
  - `plot_residual_acf`

### 4.5 Random-Effects Diagnostics
- Supported for Laplace/FOCEI/MCEM/SAEM/MCMC-family fits.
- Functions include:
  - `plot_random_effect_distributions`
  - `plot_random_effect_pit`
  - `plot_random_effect_standardized`
  - `plot_random_effect_standardized_scatter`
  - `plot_random_effects_pdf`
  - `plot_random_effects_scatter`
  - `plot_random_effect_pairplot`
- Flow-based RE diagnostics use sample-based approximations as needed.

### 4.6 VPC and UQ Plots
- `plot_vpc(...)` for predictive checks (continuous and discrete outcomes).
- `plot_uq_distributions(...)` for UQ summaries on natural or transformed scales.

## 5) Simulation and Utility Capabilities
- Simulation:
  - `simulate_data`
  - `simulate_data_model`
- Supports simulation with RE draws, ODE solves, event handling, and flexible distributions in formulas.

## 6) Cross-Cutting Notes
- `ComponentArray` is used end-to-end in estimation paths (no unnecessary flattening step required).
- ForwardDiff buffers/configs are cached in key gradient paths for allocation/performance improvements.
- Threaded execution is supported in major random-effects estimation workflows via serialization controls.

## 7) Method-Specific Distribution Notes
- Distribution flexibility is broad at the modeling layer for both outcomes and random effects.
- Some approximation methods have narrower analytic assumptions:
  - FOCEI `info_mode=:fisher_common` supports a defined subset of outcomes/RE priors.
  - FOCEI `info_mode=:custom` expands to arbitrary distributions through user-supplied information approximation.
- Bayesian and simulation paths are generally the most distribution-flexible, subject to valid `logpdf`/`rand` behavior and stable autodiff where required.

## 8) Representative Validation Coverage
- Model and data model behavior: `test/model_tests.jl`, `test/data_model_tests.jl`, `test/data_model_ode_tests.jl`
- RE distribution flexibility and AD: `test/random_effects_tests.jl`, `test/ad_random_effects.jl`, `test/ad_random_effects_values.jl`
- Estimation methods:
  - Laplace: `test/estimation_laplace_tests.jl`, `test/estimation_laplace_fit_tests.jl`
  - FOCEI: `test/estimation_focei_tests.jl`, `test/estimation_focei_map_tests.jl`
  - MCEM/SAEM: `test/estimation_mcem_tests.jl`, `test/estimation_saem_tests.jl`, `test/estimation_saem_suffstats_tests.jl`
  - MCMC RE including HMM: `test/estimation_mcmc_re_tests.jl`
- UQ: `test/uq_tests.jl`, `test/uq_edge_cases_tests.jl`, `test/uq_plotting_tests.jl`
- Plotting: `test/plotting_functions_tests.jl`, `test/plot_random_effects_tests.jl`, `test/vpc_tests.jl`
