# Laplace

The Laplace approximation is a widely used technique for integrating out random effects in nonlinear mixed-effects models. Rather than evaluating the marginal likelihood integral exactly -- which is intractable for nonlinear models -- the Laplace method approximates it via a second-order Taylor expansion around the empirical Bayes (EB) mode of each individual's random-effects vector. The resulting closed-form approximation can be optimized over the fixed effects using standard gradient-based methods, combining computational efficiency with support for complex nonlinear model structures.

## Applicability

- Requires a model with random effects.
- Requires at least one free fixed effect.
- Supports nonlinear models, including ODE-based models.

If the model has no random effects, `Laplace` will raise an error. Use a fixed-effects method such as [`MLE`](mle.md) or [`MAP`](mle-map.md) instead.

## Basic Usage

The following example fits a simple nonlinear mixed-effects model with a single subject-level random effect.

```julia
using NoLimits
using DataFrames
using Distributions

model = @Model begin
    @fixedEffects begin
        a = RealNumber(0.2)
        b = RealNumber(0.1)
        sigma = RealNumber(0.3, scale=:log)
    end

    @covariates begin
        t = Covariate()
    end

    @randomEffects begin
        eta_id = RandomEffect(TDist(6.0); column=:ID)
    end

    @formulas begin
        mu = a + b * t + exp(eta_id)   # nonlinear in random effects
        y ~ Normal(mu, sigma)
    end
end

df = DataFrame(
    ID = [:A, :A, :B, :B, :C, :C],
    t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    y = [1.0, 1.3, 0.9, 1.2, 1.1, 1.5],
)

dm = DataModel(model, df; primary_id=:ID, time_col=:t)

res = fit_model(dm, NoLimits.Laplace(; optim_kwargs=(maxiters=100,)))
```

## Constructor Options

The `Laplace` constructor exposes options that control the outer fixed-effects optimization, the inner EB optimization, Hessian stabilization, and the computational strategy for log-determinant gradients. Most users will only need to adjust a few of these; the defaults are chosen to work well across a range of model types.

```julia
using Optimization
using OptimizationOptimJL
using LineSearches

laplace_method = NoLimits.Laplace(;
    optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
    optim_kwargs=NamedTuple(),
    adtype=Optimization.AutoForwardDiff(),
    inner_optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
    inner_kwargs=NamedTuple(),
    inner_adtype=Optimization.AutoForwardDiff(),
    inner_grad_tol=:auto,
    multistart_n=50,
    multistart_k=10,
    multistart_grad_tol=:auto,
    multistart_max_rounds=1,
    multistart_sampling=:lhs,
    jitter=1e-6,
    max_tries=6,
    jitter_growth=10.0,
    adaptive_jitter=true,
    jitter_scale=1e-6,
    use_trace_logdet_grad=true,
    use_hutchinson=false,
    hutchinson_n=8,
    theta_tol=0.0,
    lb=nothing,
    ub=nothing,
)
```

Notes:

- `inner_grad_tol=:auto` uses method-specific defaults (`1e-8` for non-ODE, `1e-2` for ODE paths).
- `use_trace_logdet_grad=true` is the default gradient path for the log-determinant term.
- `use_hutchinson=true` activates the stochastic Hutchinson approximation for logdet-related terms.
- `lb`/`ub` are bounds on transformed fixed-effect parameters.

### Option Groups

The constructor keywords fall into several logical groups, summarized in the table below.

| Group | Keywords | What they control |
| --- | --- | --- |
| Outer optimization | `optimizer`, `optim_kwargs`, `adtype` | Optimization over fixed effects. |
| Inner EB optimization | `inner_optimizer`, `inner_kwargs`, `inner_adtype`, `inner_grad_tol` | Optimization of batch-level EB modes used by Laplace objective/gradient evaluation. |
| EB multistart | `multistart_n`, `multistart_k`, `multistart_grad_tol`, `multistart_max_rounds`, `multistart_sampling` | Number/selection of initial points and restart policy for EB optimization. |
| Hessian stabilization | `jitter`, `max_tries`, `jitter_growth`, `adaptive_jitter`, `jitter_scale` | Cholesky stabilization for `-H` in log-determinant calculations. |
| Logdet gradient strategy | `use_trace_logdet_grad`, `use_hutchinson`, `hutchinson_n` | Computational path for logdet-related derivatives. |
| Caching | `theta_tol` | Reuse tolerance for objective/gradient cache across nearby fixed-effect values. |
| Bounds | `lb`, `ub` | Optional transformed-scale bounds for free fixed effects. |

### Inner vs Outer Optimizer Choices (Optimization.jl Interface)

`Laplace` uses a two-level (nested) optimization scheme. Both levels dispatch through the [Optimization.jl](https://docs.sciml.ai/Optimization/stable/) interface:

- **Outer layer**: optimizes the fixed effects (`theta`) to maximize the Laplace-approximated marginal log-likelihood.
- **Inner layer**: for each outer iteration, finds the EB mode (`b*`) for each batch of random effects.

Practical implications:

- Outer optimizer (`optimizer`, `optim_kwargs`, `adtype`)
  - Runs once at the top level.
  - Can use local gradient methods (default `LBFGS`) or global/derivative-free methods.
  - If using BlackBoxOptim (`OptimizationBBO.*`), finite bounds are required.
- Inner optimizer (`inner_optimizer`, `inner_kwargs`, `inner_adtype`, `inner_grad_tol`)
  - Runs repeatedly across batches and across outer iterations.
  - Should typically be a fast local optimizer; defaults are chosen for that path.
  - Tighter inner tolerances can improve objective/gradient quality but increase runtime.

Repository-verified behavior:

- Default outer/inner optimizers are `OptimizationOptimJL.LBFGS(...)`.
- Outer BlackBoxOptim is supported with finite bounds (`lb`, `ub`); without bounds, an error is raised.

Examples:

```julia
using NoLimits
using OptimizationOptimJL
using OptimizationBBO
using LineSearches

# 1) Local-gradient outer + local-gradient inner (default style)
laplace_local = NoLimits.Laplace(;
    optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
    inner_optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
)

# 2) Global outer search + local inner EB solves
#    (requires finite transformed-scale bounds for free fixed effects)
lb, ub = default_bounds_from_start(dm; margin=1.0)
laplace_global_outer = NoLimits.Laplace(;
    optimizer=OptimizationBBO.BBO_adaptive_de_rand_1_bin_radiuslimited(),
    optim_kwargs=(maxiters=80,),
    lb=lb,
    ub=ub,
    inner_optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
    inner_kwargs=(maxiters=40,),
)
```

### Detailed Behavior

This section provides additional detail on each option group.

- `inner_grad_tol`
  - `:auto` resolves to `1e-8` for non-ODE models and `1e-2` for ODE models.
- `multistart_sampling`
  - Supported values are `:lhs` (Latin hypercube sampling) and `:random`.
- `multistart_n`, `multistart_k`
  - Control the size of the candidate start pool and the number of starts selected for EB multistart optimization.
- `jitter`, `max_tries`, `jitter_growth`
  - If the Cholesky factorization of `-H` fails, it is retried with increasing diagonal jitter (`jitter * jitter_growth^attempt`).
- `adaptive_jitter`, `jitter_scale`
  - When `adaptive_jitter=true`, the initial jitter is scaled by the Hessian diagonal magnitude (`jitter_scale * mean(abs(diag(-H)))`), then bounded below by `jitter`.
- `use_trace_logdet_grad`
  - Uses trace-based derivatives of logdet terms (default path).
- `use_hutchinson`, `hutchinson_n`
  - Uses stochastic Hutchinson estimation for logdet derivative terms with `hutchinson_n` probe vectors.
  - When `use_hutchinson=true`, this path is used instead of the exact logdet gradient.
- `theta_tol`
  - Cache tolerance for reusing objective/gradient values at nearby parameter vectors.
  - `0.0` means reuse only for effectively identical vectors.
- `lb`, `ub`
  - Bounds are interpreted on transformed parameters and are applied only to free fixed effects.
  - If constants are set via `constants`, bounds for those constant parameters are ignored.

### Advanced Option Containers

For more granular control, `Laplace` also accepts structured option containers that replace the corresponding keyword groups:

- `inner_options`
- `hessian_options`
- `cache_options`
- `multistart_options`

When one of these is provided, it replaces the corresponding scalar keyword bundle in that option group.

### Tuning Examples

The examples below illustrate common tuning strategies for different use cases.

```julia
# Fast exploratory run
laplace_fast = NoLimits.Laplace(;
    optim_kwargs=(maxiters=60,),
    multistart_n=0,
    multistart_k=0,
)

# More robust EB search
laplace_robust = NoLimits.Laplace(;
    inner_grad_tol=1e-4,
    multistart_n=120,
    multistart_k=24,
    multistart_max_rounds=3,
    multistart_sampling=:lhs,
)

# Stochastic logdet-gradient path
laplace_hutch = NoLimits.Laplace(;
    use_hutchinson=true,
    hutchinson_n=16,
)
```

## Fixing Known Random-Effect Levels (`constants_re`)

In some settings, certain random-effect levels are known a priori -- for example, a reference group set to zero or a level estimated in a previous analysis. `Laplace` supports fixing selected random-effect levels to specified values while estimating the remaining levels via EB.

```julia
constants_re = (; eta_id=(; A=0.0))

res_fixed = fit_model(
    dm,
    NoLimits.Laplace(; optim_kwargs=(maxiters=100,));
    constants_re=constants_re,
)
```

Validation rules for `constants_re`:

- Keys must match random-effect names declared in `@randomEffects`.
- Level names must exist in the corresponding grouping column.
- Values must have the correct dimension (scalar for univariate RE; vector for multivariate RE).

## Accessing Results

After fitting, results are accessed through the standard accessor interface.

```julia
theta_u = get_params(res; scale=:untransformed)
theta_t = get_params(res; scale=:transformed)
obj = get_objective(res)
ok = get_converged(res)

re_df = get_random_effects(res)
re_df_laplace = get_laplace_random_effects(res; flatten=true, include_constants=true)

ll = get_loglikelihood(res)
```

`get_random_effects` and `get_laplace_random_effects` each return one `DataFrame` per random effect, with rows corresponding to the levels of the grouping column (e.g., one row per individual).

## Serialization

Laplace supports serial and threaded evaluation of individual- or batch-level computations through the `serialization` keyword:

```julia
using SciMLBase

res_serial = fit_model(dm, NoLimits.Laplace(); serialization=EnsembleSerial())
res_threads = fit_model(dm, NoLimits.Laplace(); serialization=EnsembleThreads())
```

## Bounds and BlackBoxOptim

When using a derivative-free global optimizer from BlackBoxOptim, finite bounds on all free fixed effects are required. A convenience function generates default bounds from the initial parameter values:

```julia
lb, ub = default_bounds_from_start(dm; margin=1.0)
```

These bounds are then passed directly to the `Laplace` constructor via `lb` and `ub`.
