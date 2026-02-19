# Fixed-Effects-Only Capabilities

This document summarizes the current fixed-effects-only estimation capabilities in NoLimits.jl.
It is intended as an LLM-ready reference for documentation and future extensions.

## Scope
- Applies to models **without random effects**.
- Supports nonlinear models with and without ODEs.
- Methods: **MLE**, **MAP**, **MCMC** (Turing).

## Unified API
```
fit_model(dm::DataModel, method::FittingMethod; kwargs...)
```

### Shared kwargs (fixed-effects only)
- `constants::NamedTuple = NamedTuple()`
  - Fixed values for **transformed** parameters.
  - Constants are removed from the optimizer state and injected during evaluation.
- `penalty::NamedTuple = NamedTuple()`
  - Per-parameter penalties on the **natural** scale.
  - Not supported by MCMC.
- `ode_args::Tuple = ()`, `ode_kwargs::NamedTuple = NamedTuple()`
  - Passed through to ODE solving inside likelihood evaluation.
- `serialization::EnsembleSerial = EnsembleSerial()`
  - Controls loglikelihood parallelization (Serial or Threads).
- `rng::AbstractRNG = Random.default_rng()`
- `theta_0_untransformed::Union{Nothing, ComponentArray} = nothing`
  - Optional custom starting values on **natural** scale.

## Methods

### MLE (Maximum Likelihood)
```
fit_model(dm, MLE(; optimizer, optim_kwargs, adtype, lb, ub))
```
- Optimizes **negative loglikelihood**.
- Requires at least one free fixed effect.
- Bounds are defined on the **transformed scale** (from fixed-effects specs).
- If all bounds are infinite, no bounds are passed to Optimization.jl.
- BlackBoxOptim methods require finite bounds:
  - Use `default_bounds_from_start(dm; margin=...)` to build bounds.

### MAP (Maximum A Posteriori)
```
fit_model(dm, MAP(; optimizer, optim_kwargs, adtype, lb, ub))
```
- Optimizes **negative logposterior** (loglikelihood + logprior).
- **Hard error** if no fixed-effect prior is provided (at least one non-`Priorless` prior is required).
- Uses the same optimization infrastructure as MLE.

### MCMC (Turing)
```
fit_model(dm, MCMC(; sampler, turing_kwargs, adtype))
```
- Builds a Turing model and samples fixed effects.
- **Hard error** if any **free** fixed effect is `Priorless`.
- **Ignores parameter scale** (e.g., `:log`, `:cholesky`); sampling is on natural scale.
  - Emits info message if a scale is set.
- Penalties are not supported (use MAP instead).

## Parameter Handling
- Fixed-effects parameters are stored as `ComponentArray`.
- Optimization happens on **transformed** parameters:
  - `:log` for positive scalars/vectors.
  - `:cholesky` or `:expm` for PSD matrices.
- Priors and penalties are evaluated on the **natural** scale.
- Starting values:
  - Default: model-provided initial parameters.
  - Custom: `theta_0_untransformed` (natural scale).

## ODE Models (Fixed Effects)
- Likelihood handles ODE solving per individual:
  - Uses cached `ODEProblem` and compiled DE parameters.
  - On solver failure (non-success retcode), returns `Inf`.
- ODE parameters and initial conditions can depend on fixed effects.
- Saveat mode:
  - `:dense` (default), `:saveat`, or `:auto` via `set_solver_config`.

## Specialized Likelihood Kernels
For speed and reduced allocations, specialized logpdf kernels exist for:
- `Normal`
- `Bernoulli`
- `Poisson`
- `LogNormal`

All other distributions use generic `logpdf`.

## Output Structure
All methods return a `FitResult`:
- `method` :: FittingMethod
- `result` :: MethodResult
- `summary` :: FitSummary
- `diagnostics` :: FitDiagnostics

### Accessors
- `get_params(res; scale=:transformed|:untransformed|:both)` returns parameters on the requested scale.
- `get_objective(res)` and `get_converged(res)` summarize the fit.
- `get_loglikelihood(dm, res; ...)` evaluates the fitted loglikelihood (MLE/MAP only).

### Method results
- `MLEResult`: Optimization solution + objective + iterations.
- `MAPResult`: Same as MLE, plus logprior term.
- `MCMCResult`: `Chains` + sampler + number of samples + observations.

## Validation / Benchmarks
- Simulation validation lives in:
  - `simulation_validation/fixed_effects/_setup.jl`
  - `simulation_validation/fixed_effects/mle_validation.jl`
  - `simulation_validation/fixed_effects/map_validation.jl`
- Benchmarks in `benchmarks/estimation_fit_benchmarks.jl`.
- Many fixed-effects-only examples are in the test suite (e.g. `test/estimation_mle_tests.jl`, `test/estimation_map_tests.jl`, `test/estimation_mcmc_tests.jl`, and various ODE/model tests).

## Notes / Limitations
- Random effects estimation is **not** covered here.
- Random-effects MCMC is supported via a separate path; see `RANDOM_EFFECTS_CAPABILITIES.mde`.
- No standard errors/covariance reporting yet.
- Penalties are per-parameter (no custom functions to avoid recompilation).
