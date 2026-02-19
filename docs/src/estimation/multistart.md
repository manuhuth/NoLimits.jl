# Multistart

Nonlinear models frequently have multiple local optima, making the estimated parameters sensitive to initialization. The `Multistart` wrapper addresses this by running multiple fits from different initial parameter values and selecting the best result. This strategy is especially important for complex models where a single optimization run provides limited confidence that the global optimum has been found.

`Multistart` operates as a method-agnostic wrapper around `fit_model`: it generates a set of candidate starting points, dispatches independent fits, and returns the run with the best objective value. Because each fit is independent, the individual runs can be executed in parallel.

The call pattern is:

```julia
res_ms = fit_model(ms, dm, method; kwargs...)
```

where `ms` is a `NoLimits.Multistart(...)` object and `method` is any fitting method.

## Supported Methods

`Multistart` wraps any `FittingMethod` and has been tested with:

- `MLE`
- `MAP`
- `Laplace`
- `LaplaceMAP`
- `MCEM`
- `SAEM`
- `MCMC` (supported, but usually not recommended as a primary restart strategy)

## Recommendation

`Multistart` is most beneficial for optimization- and EM-based methods (`MLE`, `MAP`, `Laplace`, `LaplaceMAP`, `MCEM`, `SAEM`), where the choice of starting values strongly influences which local optimum is found.

For `MCMC`, multistart is technically supported but generally not recommended as the primary strategy. In most Bayesian workflows, tuning sampler settings and chain diagnostics is more effective than varying initial values across restarts.

## Constructor

```julia
using NoLimits
using Random
using SciMLBase

ms = NoLimits.Multistart(;
    dists=NamedTuple(),
    n_draws_requested=100,
    n_draws_used=50,
    sampling=:random,               # :random or :lhs
    serialization=EnsembleSerial(), # controls parallelism across starts
    rng=Random.default_rng(),
)
```

## How Starts Are Built

Starting points are constructed as follows:

- **Start 1** is always the model's default initial fixed-effect values (obtained via `get_θ0_untransformed`).
- **Subsequent starts** are sampled independently for each parameter, drawing from:
  - the distribution specified in `ms.dists` for that parameter name, if provided;
  - the fixed-effect prior, if one is defined;
  - otherwise, the parameter retains its default value across all starts.

All sampled values are validated against the natural-scale bounds of each parameter. If any sampled value violates its bounds, an error is raised before fitting begins.

## Distribution Inputs

The `dists` argument is a `NamedTuple` keyed by fixed-effect names. The expected distribution type depends on the parameter structure:

- **Scalar parameters**: a univariate distribution (e.g., `Normal(...)`).
- **Vector or matrix parameters**: either a single multivariate/matrix-variate distribution, or an array of element-wise univariate distributions.

For square-matrix parameters, `Multistart` symmetrizes the sampled matrix and applies a small diagonal perturbation if needed to ensure numerical stability.

## Sampling Modes

Two sampling strategies are available:

- **`sampling=:random`** -- Draws are taken directly from the specified distributions.
- **`sampling=:lhs`** -- Latin Hypercube Sampling is used when the distribution supports quantile-based inversion, producing more uniform coverage of the parameter space. For distributions where a direct LHS quantile path is unavailable, the method falls back to random draws.

## Requested vs. Used Draws

The `n_draws_requested` and `n_draws_used` parameters decouple the sampling and fitting stages:

- `n_draws_requested` controls how many candidate starting points are sampled per parameter.
- `n_draws_used` determines how many fits are actually executed (including the default start as the first run).

If `n_draws_used` exceeds `n_draws_requested`, the number of requested draws is automatically increased and a warning is emitted.

## Scoring and Best-Run Selection

After all fits complete, successful runs are ranked by the following scoring rule:

1. The objective value from `get_objective`, if finite.
2. Otherwise, the negative log-likelihood from `-get_loglikelihood`, if finite.
3. Otherwise, `Inf` (effectively deprioritizing the run).

The run with the lowest score is selected as the best result:

```julia
if @isdefined(res_ms) && res_ms !== nothing
    best = get_multistart_best(res_ms)
    best_idx = get_multistart_best_index(res_ms)
end
```

If all runs fail, `Multistart` raises an error reporting the first recorded failure.

## Parallelism and RNG Behavior

The `ms.serialization` field controls execution of the individual fits:

- **`EnsembleSerial()`**: Fits run sequentially in a single thread.
- **`EnsembleThreads()`**: Fits run in parallel across available threads.

Random number generator behavior depends on how `rng` is supplied:

- If `rng` is not passed to `fit_model(ms, ...)`, each start receives an internally spawned child RNG to ensure independence.
- If `rng` is explicitly provided in the fit keywords, that generator is forwarded to every underlying fit call.

## Fit Keyword Forwarding

All fit keywords are forwarded to the wrapped method, with one exception:

- **`theta_0_untransformed`** is ignored (with a warning), because `Multistart` manages starting points internally.

All other keywords -- such as `constants`, `constants_re`, `serialization`, and `store_data_model` -- are passed through unchanged.

## Multistart Result Accessors

The `MultistartFitResult` provides detailed access to both successful and failed runs:

```julia
if @isdefined(res_ms) && res_ms !== nothing
    ok_runs = get_multistart_results(res_ms)
    ok_starts = get_multistart_starts(res_ms)

    failed_runs = get_multistart_failed_results(res_ms)
    failed_starts = get_multistart_failed_starts(res_ms)
    failed_errors = get_multistart_errors(res_ms)

    best_run = get_multistart_best(res_ms)
    best_idx = get_multistart_best_index(res_ms)
end
```

Standard fit accessors also work directly on a `MultistartFitResult`, dispatching to the best run:

```julia
if @isdefined(res_ms) && res_ms !== nothing
    theta_best = get_params(res_ms; scale=:untransformed)
    obj_best = get_objective(res_ms)
end
```

## Example: Fixed-Effects MLE

The following example demonstrates multistart optimization for a simple fixed-effects model using Latin Hypercube Sampling to generate initial values:

```julia
using NoLimits
using DataFrames
using Distributions

model = @Model begin
    @covariates begin
        t = Covariate()
    end

    @fixedEffects begin
        a = RealNumber(0.2)
        sigma = RealNumber(0.5, scale=:log)
    end

    @formulas begin
        y ~ Laplace(a, sigma)
    end
end

df = DataFrame(
    ID = [:A, :A, :B, :B],
    t = [0.0, 1.0, 0.0, 1.0],
    y = [0.1, 0.2, 0.0, -0.1],
)

dm = DataModel(model, df; primary_id=:ID, time_col=:t)

ms = NoLimits.Multistart(;
    dists=(; a=Normal(0.0, 1.0)),
    n_draws_requested=6,
    n_draws_used=4,
    sampling=:lhs,
)

res_ms = fit_model(ms, dm, NoLimits.MLE(; optim_kwargs=(maxiters=80,)))

best = get_multistart_best(res_ms)
theta_best = get_params(res_ms; scale=:untransformed)
```

## Optional: MCMC with Multistart (Supported, Usually Not Recommended)

While multistart is primarily designed for optimization-based methods, it can be used with MCMC when a restart-style sampling workflow is explicitly desired:

```julia
using NoLimits
using DataFrames
using Distributions
using Turing

model = @Model begin
    @covariates begin
        t = Covariate()
    end

    @fixedEffects begin
        a = RealNumber(0.2, prior=Normal(0.0, 1.0))
        sigma = RealNumber(0.5, scale=:log, prior=LogNormal(0.0, 0.2))
    end

    @formulas begin
        y ~ LogNormal(a, sigma)
    end
end

df = DataFrame(
    ID = [:A, :A, :B, :B],
    t = [0.0, 1.0, 0.0, 1.0],
    y = [1.0, 1.1, 0.9, 1.0],
)

dm = DataModel(model, df; primary_id=:ID, time_col=:t)

ms = NoLimits.Multistart(; n_draws_requested=4, n_draws_used=3)

res_ms = fit_model(
    ms,
    dm,
    NoLimits.MCMC(; sampler=MH(), turing_kwargs=(n_samples=200, n_adapt=0, progress=false)),
)

chain_best = get_chain(res_ms)
```

Use this pattern only when a restart-style MCMC workflow is explicitly needed.
