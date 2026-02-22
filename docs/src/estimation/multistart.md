# Multistart

Nonlinear models frequently have multiple local optima, making the estimated parameters sensitive to initialization. The `Multistart` wrapper addresses this by running multiple fits from different initial parameter values and selecting the best result. This strategy is especially important for complex models where a single optimization run provides limited confidence that the global optimum has been found.

`Multistart` operates as a method-agnostic wrapper around `fit_model`: it samples many candidate starting points, **screens** them with a cheap log-likelihood evaluation to identify the most promising ones, then fully optimizes only the top candidates. Because each optimization is independent, the runs can be executed in parallel.

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
- `FOCEI`
- `FOCEIMAP`
- `MCEM`
- `SAEM`
- `VI` (see note below)
- `MCMC` (supported, but usually not recommended as a primary restart strategy)

## Recommendation

`Multistart` is most beneficial for optimization- and EM-based methods (`MLE`, `MAP`, `Laplace`, `LaplaceMAP`, `FOCEI`, `FOCEIMAP`, `MCEM`, `SAEM`, `VI`), where the choice of starting values strongly influences which local optimum is found.

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

## Two-Phase Workflow

`Multistart` uses a two-phase approach to avoid running full optimizations from all sampled candidates:

**Phase 1 — Screening.** All `n_draws_requested` candidates are evaluated cheaply by computing the marginal log-likelihood at η = 0 (no random-effects perturbation) using a pre-compiled ODE and covariate cache. Candidates are ranked by this screening log-likelihood and the top `n_draws_used` are selected via a partial sort — no full optimization is performed during this phase.

**Phase 2 — Optimization.** The selected `n_draws_used` candidates are passed to the wrapped fitting method as independent starting points. Each run produces a full `FitResult`. The run with the best final objective is returned as the best result.

If `n_draws_requested == n_draws_used`, Phase 1 is skipped entirely — no screening cache is built and all candidates proceed directly to optimization.

### Progress Logging

At the start of each `fit_model` call, a summary is logged:

```
┌ Info: Multistart
│   candidates = 20
│   selected = 5
│   varying = "a, σ"
│   best_screening_ll = -12.4
└   worst_screening_ll = -18.7
```

- `candidates` — total starting points generated (= `n_draws_requested`, after any automatic adjustment).
- `selected` — candidates forwarded to full optimization (= `n_draws_used`).
- `varying` — names of parameters whose values differ across starts (those with a prior or an entry in `dists`).
- `best_screening_ll` / `worst_screening_ll` — log-likelihood range of the selected candidates at η = 0. Omitted when screening is skipped (`candidates == selected`) or when all candidates produce non-finite likelihoods.

### Screening and Method Direction

Screening always ranks candidates by the marginal log-likelihood (higher is better). This is the correct direction for every supported method:

- **MLE / MAP / Laplace / FOCEI / MCEM / SAEM / VI** — all internally minimize the negative log-likelihood (or a penalized variant). Selecting candidates with the highest screening LL puts the optimizer in the most promising region.
- **MCMC** — though MCMC does not optimize, starting from a high-likelihood region improves early mixing and reduces warm-up cost.

No sign adjustment is made per-method; the screening criterion is uniform.

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

The `n_draws_requested` and `n_draws_used` parameters control the two phases:

| Parameter | Phase | Effect |
|---|---|---|
| `n_draws_requested` | Screening | Total candidates sampled. More candidates give better coverage but increase screening cost. |
| `n_draws_used` | Optimization | Candidates forwarded to full optimization. Larger values improve coverage but increase runtime proportionally. |

If `n_draws_used` exceeds `n_draws_requested`, the number of requested draws is automatically increased to match and a warning is emitted.

Setting `n_draws_requested == n_draws_used` disables screening: all candidates proceed directly to optimization without any pre-evaluation.

## Scoring and Best-Run Selection

After all fits complete, successful runs are ranked by the following scoring rule:

1. The objective value from `get_objective`, if finite.
2. Otherwise, the negative log-likelihood from `-get_loglikelihood`, if finite.
3. Otherwise, `Inf` (effectively deprioritizing the run).

The run with the lowest score is selected as the best result:

```julia
best = get_multistart_best(res_ms)
best_idx = get_multistart_best_index(res_ms)
```

If all runs fail, `Multistart` raises an error reporting the first recorded failure.

## Parallelism and RNG Behavior

The `ms.serialization` field controls execution of the individual fits:

- **`EnsembleSerial()`**: Fits run sequentially in a single thread.
- **`EnsembleThreads()`**: Fits run in parallel across available threads.

Note that screening (Phase 1) always runs serially regardless of `serialization`; only Phase 2 (the full optimizations) is parallelized.

Random number generator behavior depends on how `rng` is supplied:

- If `rng` is not passed to `fit_model(ms, ...)`, each start receives an internally spawned child RNG to ensure independence.
- If `rng` is explicitly provided in the fit keywords, that generator is forwarded to every underlying fit call.

## Fit Keyword Forwarding

All fit keywords are forwarded to the wrapped method, with one exception:

- **`theta_0_untransformed`** is ignored (with a warning), because `Multistart` manages starting points internally.

All other keywords -- such as `constants`, `constants_re`, `ode_args`, `ode_kwargs`, `serialization`, and `store_data_model` -- are passed through unchanged.

## Multistart Result Accessors

The `MultistartFitResult` provides detailed access to both successful and failed runs:

```julia
ok_runs    = get_multistart_results(res_ms)
ok_starts  = get_multistart_starts(res_ms)

failed_runs   = get_multistart_failed_results(res_ms)
failed_starts = get_multistart_failed_starts(res_ms)
failed_errors = get_multistart_errors(res_ms)

best_run = get_multistart_best(res_ms)
best_idx = get_multistart_best_index(res_ms)
```

Standard fit accessors also work directly on a `MultistartFitResult`, dispatching to the best run:

```julia
theta_best = get_params(res_ms; scale=:untransformed)
obj_best   = get_objective(res_ms)
```

## Example: Fixed-Effects MLE with Screening

The following example generates 20 candidates via LHS, screens them to the top 5 by log-likelihood, and runs 5 full optimizations:

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
    ID    = [:A, :A, :B, :B],
    t     = [0.0, 1.0, 0.0, 1.0],
    y     = [0.1, 0.2, 0.0, -0.1],
)

dm = DataModel(model, df; primary_id=:ID, time_col=:t)

ms = NoLimits.Multistart(;
    dists             = (; a=Normal(0.0, 1.0)),
    n_draws_requested = 20,
    n_draws_used      = 5,
    sampling          = :lhs,
)

res_ms = fit_model(ms, dm, NoLimits.MLE(; optim_kwargs=(maxiters=80,)))

best      = get_multistart_best(res_ms)
theta_best = get_params(res_ms; scale=:untransformed)
```

The logged output will look similar to:

```
┌ Info: Multistart
│   candidates = 20
│   selected = 5
│   varying = "a"
│   best_screening_ll = -2.1
└   worst_screening_ll = -8.4
```

## Example: Variational Inference with Multistart

`VI` benefits from multistart when the ELBO landscape is multimodal. The usage is identical to any other method:

```julia
using NoLimits
using DataFrames
using Distributions
using Random

model = @Model begin
    @covariates begin
        t = Covariate()
    end

    @fixedEffects begin
        a     = RealNumber(0.0, prior=Normal(0.0, 1.0))
        sigma = RealNumber(0.5, scale=:log, prior=LogNormal(0.0, 0.5))
    end

    @formulas begin
        y ~ Normal(a, sigma)
    end
end

df = DataFrame(
    ID = [:A, :A, :B, :B],
    t  = [0.0, 1.0, 0.0, 1.0],
    y  = [0.1, 0.2, 0.0, -0.1],
)

dm = DataModel(model, df; primary_id=:ID, time_col=:t)

ms = NoLimits.Multistart(;
    n_draws_requested = 10,
    n_draws_used      = 3,
    sampling          = :lhs,
    rng               = Random.Xoshiro(42),
)

res_ms = fit_model(
    ms, dm,
    NoLimits.VI(; turing_kwargs=(max_iter=300, family=:meanfield, progress=false)),
    rng=Random.Xoshiro(1),
)

posterior = get_variational_posterior(res_ms)
objective = get_objective(res_ms)   # final ELBO of the best run
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
        a     = RealNumber(0.2, prior=Normal(0.0, 1.0))
        sigma = RealNumber(0.5, scale=:log, prior=LogNormal(0.0, 0.2))
    end

    @formulas begin
        y ~ LogNormal(a, sigma)
    end
end

df = DataFrame(
    ID = [:A, :A, :B, :B],
    t  = [0.0, 1.0, 0.0, 1.0],
    y  = [1.0, 1.1, 0.9, 1.0],
)

dm = DataModel(model, df; primary_id=:ID, time_col=:t)

ms = NoLimits.Multistart(; n_draws_requested=6, n_draws_used=3)

res_ms = fit_model(
    ms, dm,
    NoLimits.MCMC(; sampler=MH(), turing_kwargs=(n_samples=200, n_adapt=0, progress=false)),
)

chain_best = get_chain(res_ms)
```

Use this pattern only when a restart-style MCMC workflow is explicitly needed.
