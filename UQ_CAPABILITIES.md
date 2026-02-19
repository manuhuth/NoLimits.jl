# Uncertainty Quantification (UQ) Capabilities

This document summarizes the current UQ implementation in `NoLimits.jl`.
It is an implementation-level reference for users and developers.

## Scope

- Covers post-fit UQ for:
  - `MLE`, `MAP`
  - `Laplace`, `LaplaceMAP`
  - `FOCEI`, `FOCEIMAP`
  - `MCEM`, `SAEM`
  - `MCMC`
- UQ is computed from a fitted model via `compute_uq(res; ...)`.
- UQ acts on fixed-effect coordinates that are both:
  - free (not fixed by `constants`)
  - marked with `calculate_se=true`.

## Unified API

```julia
uq = compute_uq(res::FitResult; kwargs...)
```

Primary backend selector:

- `method=:auto` (default):
  - if fit method is `MCMC` -> `:chain`
  - otherwise -> `:wald`
  - if `interval=:profile` is passed -> `:profile`
- explicit backends:
  - `method=:wald`
  - `method=:chain`
  - `method=:profile`
  - `method=:mcmc_refit`

Common key kwargs:

- `level=0.95`
- `constants::NamedTuple=nothing`
- `constants_re::NamedTuple=nothing`
- `penalty::NamedTuple=nothing`
- `ode_args`, `ode_kwargs`, `serialization`
- `rng`

Backend-specific key kwargs:

- Wald:
  - `vcov=:hessian` or `:sandwich`
  - `hessian_backend=:auto|:forwarddiff|:fd_gradient`
  - `pseudo_inverse=false`
  - `fd_abs_step`, `fd_rel_step`, `fd_max_tries`
  - `n_draws=2000`
  - for `MCEM`/`SAEM`: `re_approx=:auto|:laplace|:focei`, `re_approx_method`
- Chain:
  - `mcmc_warmup`, `mcmc_draws`
- Profile:
  - `profile_method`, `profile_scan_width`, `profile_scan_tol`, `profile_loss_tol`
  - `profile_local_alg`, `profile_max_iter`, `profile_ftol_abs`, `profile_kwargs`
- MCMC refit:
  - `mcmc_method`, or (`mcmc_sampler`, `mcmc_turing_kwargs`, `mcmc_adtype`)
  - `mcmc_fit_kwargs`, `mcmc_warmup`, `mcmc_draws`

## UQ Result Object

`compute_uq` returns a `UQResult` with:

- `backend::Symbol`
- `source_method::Symbol`
- `parameter_names::Vector{Symbol}` (flat coordinate names)
- estimates on transformed and natural scales
- intervals on transformed and natural scales (or `nothing`)
- covariance matrices on transformed and natural scales (or `nothing`)
- draws on transformed and natural scales (or `nothing`)
- diagnostics `NamedTuple`

Accessors:

- `get_uq_backend(uq)`
- `get_uq_source_method(uq)`
- `get_uq_parameter_names(uq)`
- `get_uq_estimates(uq; scale=:natural|:transformed, as_component=true)`
- `get_uq_intervals(uq; scale=:natural|:transformed, as_component=true)`
- `get_uq_vcov(uq; scale=:natural|:transformed)`
- `get_uq_draws(uq; scale=:natural|:transformed)`
- `get_uq_diagnostics(uq)`

## Parameter Selection Semantics

- UQ is restricted to fixed-effect coordinates with `calculate_se=true`.
- `constants` removes fixed effects from the UQ-active set.
- Flat coordinate names come from fixed-effect transformed layout (e.g. matrix entries like `:Ω_1_2`).
- If no active coordinates remain, `compute_uq` throws an informative error.

## Backend Capabilities

### 1) Wald (`method=:wald`)

Supported source methods:

- no random effects path:
  - `MLE`, `MAP`
- random effects path:
  - `Laplace`, `LaplaceMAP`, `FOCEI`, `FOCEIMAP`
  - `MCEM`, `SAEM` (via approximation method selection)

Core behavior:

- Builds objective around fitted point and computes transformed-scale Hessian.
- Covariance options:
  - `:hessian`: inverse Hessian (or pseudo-inverse)
  - `:sandwich`: sandwich estimator (`bread * B * bread'`)
- Draws:
  - samples Gaussian on transformed scale with chosen covariance
  - maps draws through inverse parameter transform for natural-scale draws
- Intervals:
  - quantile-based from draws
- Natural covariance:
  - empirical covariance of transformed-to-natural draws

Important details:

- For RE methods, Hessian backend defaults to finite-difference-gradient path (`:auto -> :fd_gradient`) to avoid deeper AD nesting.
- `MCEM`/`SAEM` Wald requires an RE approximation:
  - `re_approx=:laplace` or `:focei` (default `:auto -> :laplace`)
  - or pass explicit `re_approx_method` instance.
- Diagnostics include:
  - `hessian_backend`, `vcov`, `pseudo_inverse`, `n_draws`
  - `n_active_parameters`
  - `coordinate_transforms`
  - approximation/fallback diagnostics for RE paths (including FOCEI fallback counters where relevant).

### 2) Chain (`method=:chain`)

Supported source method:

- `MCMC` only

Core behavior:

- Pulls post-warmup chain samples for active coordinates.
- Optional subsampling via `mcmc_draws`.
- Computes estimates, intervals, and covariance directly from chain draws.

Notes:

- Chain UQ is natural-scale based (as sampled by Turing).
- Intervals are equal-tail quantile intervals.

### 3) Profile (`method=:profile`)

Supported source methods:

- `MLE`, `MAP`, `Laplace`, `LaplaceMAP`, `FOCEI`, `FOCEIMAP`

Core behavior:

- Uses `LikelihoodProfiler.jl` per active coordinate.
- Profiles objective with all other active coordinates re-optimized.
- Returns profile-based interval endpoints on transformed and natural scales.

Notes:

- `vcov` and draws are not returned for profile backend (`nothing`).
- Diagnostics include profile statuses, endpoint flags, counters, and errors.

### 4) MCMC Refit (`method=:mcmc_refit`)

Supported source methods:

- non-MCMC fits (intended for optimization-based and EM-like methods)

Core behavior:

- Constructs/uses MCMC method and refits from fitted point.
- Respects constants/constants_re.
- Automatically fixes non-UQ fixed effects (`calculate_se=false`) unless user already fixed them.
- Then applies chain-based UQ extraction on refit result.

Notes:

- Throws if sampled fixed effects do not have priors.
- Throws if nothing is sampled (no sampled fixed effects and no random effects).
- Returns backend `:mcmc_refit` and diagnostics including sampled fixed names.

## Scale and Transform Semantics

UQ stores both transformed and natural representations:

- transformed scale:
  - optimizer parameterization (`log`, `cholesky`, `expm`, etc.)
- natural scale:
  - model parameterization used in distributions and ODE formulas

For Wald:

- Gaussian approximation is defined on transformed scale.
- Natural-scale distributions are induced through nonlinear transforms and represented through transformed draws.

## Interval and VCV Semantics

- Wald:
  - intervals are quantiles of draws (not hard-coded symmetric Wald intervals)
  - transformed VCV from Hessian/sandwich
  - natural VCV from transformed draws mapped to natural scale
- Chain/MCMC refit:
  - intervals/covariance from sampled draws
- Profile:
  - intervals from profile likelihood inversion
  - no covariance matrix output

## Plotting UQ (`plot_uq_distributions`)

`plot_uq_distributions(uq; ...)` supports:

- `scale=:natural|:transformed`
- `parameters` filtering
- `plot_type=:density` (default) or `:histogram`
- interval shading + estimate line options
- KDE bandwidth control (`kde_bandwidth`)

Axis labels:

- parameter labels are exact (`string(symbol)`), no title-casing.

Density mode behavior:

- posterior backends (`:chain`, `:mcmc_refit`):
  - KDE from draws
- Wald backend:
  - closed-form where available:
    - transformed scale: Normal marginals
    - natural scale:
      - `:identity` coordinates -> Normal
      - `:log` coordinates -> LogNormal
  - otherwise fallback to sampling + KDE

Fallback transparency:

- If any selected parameter uses sampling+KDE in density mode, an `@info` message is emitted listing those parameters.

Histogram mode behavior:

- Uses draws directly.
- Requires draws at the requested scale.

## Current Constraints / Notes

- UQ requires fit results with stored `DataModel` (`store_data_model=true` at fit time).
- Wald backend currently targets:
  - `MLE`, `MAP`, `Laplace`, `LaplaceMAP`, `FOCEI`, `FOCEIMAP`, `MCEM`, `SAEM`.
- Profile backend currently targets:
  - `MLE`, `MAP`, `Laplace`, `LaplaceMAP`, `FOCEI`, `FOCEIMAP`.
- Chain backend is `MCMC` only.
- MCMC refit is for non-MCMC source fits.
- Profile intervals may fail for difficult objectives; detailed status/error diagnostics are returned.

## Minimal Usage Examples

```julia
# Wald UQ (auto for non-MCMC)
uq = compute_uq(res; method=:wald, vcov=:hessian, n_draws=2000)

# Chain UQ from an MCMC fit
uq_chain = compute_uq(res_mcmc; method=:chain, mcmc_draws=1000)

# Profile-likelihood intervals
uq_prof = compute_uq(res; method=:profile, profile_scan_width=2.0)

# MCMC refit UQ from an optimization fit
uq_refit = compute_uq(res; method=:mcmc_refit,
                      mcmc_turing_kwargs=(n_samples=1000, n_adapt=500, progress=false))

# Plot densities (closed-form where available, otherwise KDE fallback)
plot_uq_distributions(uq; scale=:natural, plot_type=:density)

# Plot histograms
plot_uq_distributions(uq; scale=:natural, plot_type=:histogram)
```
