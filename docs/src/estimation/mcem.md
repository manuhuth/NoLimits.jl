# MCEM

The Monte Carlo Expectation-Maximization (MCEM) algorithm is a likelihood-based approach for fitting nonlinear mixed-effects models when the marginal likelihood cannot be computed in closed form. It alternates between two steps:

- an **MCMC E-step** that draws samples from the conditional distribution of random effects given the current fixed-effect estimates, and
- an **optimization-based M-step** that updates the fixed effects by maximizing the Monte Carlo approximation of the expected complete-data log-likelihood.

This formulation accommodates arbitrary nonlinear observation models, including those defined through ordinary differential equations, without requiring analytically tractable integrals over the random effects.

## Applicability

MCEM is designed for models that include both fixed and random effects:

- The model must declare at least one random effect and at least one free fixed effect.
- Multiple random-effect grouping columns and multivariate random effects are fully supported.

If fixed-effect priors are defined in the model, MCEM ignores them in its objective. To incorporate priors, use `LaplaceMAP` or `MCMC` instead.

## Basic Usage

The following example illustrates a minimal MCEM workflow with a nonlinear observation model.

```julia
using NoLimits
using DataFrames
using Distributions
using Turing

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
        eta = RandomEffect(TDist(6.0); column=:ID)
    end

    @formulas begin
        mu = a + b * t + exp(eta)   # nonlinear in random effects
        y ~ Normal(mu, sigma)
    end
end

df = DataFrame(
    ID = [:A, :A, :B, :B, :C, :C],
    t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    y = [1.0, 1.3, 0.9, 1.2, 1.1, 1.5],
)

dm = DataModel(model, df; primary_id=:ID, time_col=:t)

method = NoLimits.MCEM(;
    sampler=MH(),
    turing_kwargs=(n_samples=20, n_adapt=0, progress=false),
    maxiters=10,
)

res = fit_model(dm, method)
```

## Constructor Options

The full set of constructor arguments is shown below. All arguments have defaults and are keyword-only.

```julia
using Optimization
using OptimizationOptimJL
using LineSearches
using Turing

method = NoLimits.MCEM(;
    optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
    optim_kwargs=NamedTuple(),
    adtype=Optimization.AutoForwardDiff(),
    sampler=Turing.NUTS(0.75),
    turing_kwargs=NamedTuple(),
    sample_schedule=250,
    warm_start=true,
    verbose=false,
    progress=true,
    maxiters=100,
    rtol_theta=1e-4,
    atol_theta=1e-6,
    rtol_Q=1e-4,
    atol_Q=1e-6,
    consecutive_params=3,
    ebe_optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
    ebe_optim_kwargs=NamedTuple(),
    ebe_adtype=Optimization.AutoForwardDiff(),
    ebe_grad_tol=:auto,
    ebe_multistart_n=50,
    ebe_multistart_k=10,
    ebe_multistart_max_rounds=5,
    ebe_multistart_sampling=:lhs,
    ebe_rescue_on_high_grad=true,
    ebe_rescue_multistart_n=128,
    ebe_rescue_multistart_k=32,
    ebe_rescue_max_rounds=8,
    ebe_rescue_grad_tol=:auto,
    ebe_rescue_multistart_sampling=:lhs,
    lb=nothing,
    ub=nothing,
)
```

## Option Groups

The constructor arguments are organized into the following functional groups.

| Group | Keywords | What they control |
| --- | --- | --- |
| M-step optimizer | `optimizer`, `optim_kwargs`, `adtype` | Optimization of fixed effects using [Optimization.jl](https://docs.sciml.ai/Optimization/stable/). |
| E-step sampler | `sampler`, `turing_kwargs`, `sample_schedule`, `warm_start` | Sampling random effects per iteration. |
| EM stopping | `maxiters`, `rtol_theta`, `atol_theta`, `rtol_Q`, `atol_Q`, `consecutive_params` | Iteration limit and convergence checks. |
| Final EB estimation | `ebe_*` and `ebe_rescue_*` options | Post-fit empirical Bayes mode computation used by random-effects accessors and diagnostics. |
| Bounds | `lb`, `ub` | Optional transformed-scale bounds for free fixed effects in M-step optimization. |

The E-step sampling interface is built on [Turing.jl](https://turinglang.org/): the `sampler` and `turing_kwargs` arguments are forwarded to the underlying Turing sampling calls.

## Constructor Input Reference

### M-step Optimization Inputs

These arguments configure the fixed-effect optimization performed at each MCEM iteration.

- `optimizer`
  - Optimizer for fixed effects in each MCEM iteration.
  - Default: `OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking())`.
- `optim_kwargs`
  - Keyword arguments forwarded to `Optimization.solve` for the M-step optimizer.
- `adtype`
  - Automatic differentiation backend for the M-step objective in `OptimizationFunction`.

### E-step Sampling Inputs

These arguments control the MCMC sampling of random effects at each iteration.

- `sampler`
  - Turing sampler used for random-effect sampling in each E-step.
  - Default: `Turing.NUTS(0.75)`.
- `turing_kwargs`
  - Additional keyword arguments passed through to Turing sampling calls.
  - The keys `n_samples` and `n_adapt` are interpreted explicitly by MCEM, then passed as sampling and adaptation sizes.
- `sample_schedule`
  - Controls the number of MCMC samples drawn per iteration.
  - Accepted forms: an integer (constant across iterations), a vector (iteration-indexed), or a function `iter -> n_samples`.
  - If a schedule value is `<= 0`, MCEM falls back to `turing_kwargs[:n_samples]` (or `100` if absent).
- `warm_start`
  - When `true`, reuses previous batch latent-state parameter values as initialization for the next E-step.
- `verbose`
  - Enables iteration-level logging of diagnostic quantities (`Q`, `dtheta`, `dQ`, sample count).
- `progress`
  - Enables or disables the progress bar.

### EM Convergence Inputs

MCEM monitors both fixed-effect parameter stability and Q-function stability to determine convergence.

- `maxiters`
  - Maximum number of MCEM iterations.
- `rtol_theta`, `atol_theta`
  - Relative and absolute tolerances for fixed-effect parameter stabilization.
- `rtol_Q`, `atol_Q`
  - Relative and absolute tolerances for Q-function stabilization.
- `consecutive_params`
  - Number of consecutive iterations that must simultaneously satisfy both parameter and Q stabilization criteria before convergence is declared.

### Final EB Mode Inputs

After the EM iterations complete, MCEM computes empirical Bayes (EB) modal estimates of the random effects. These estimates are used by downstream accessors and diagnostic functions.

- `ebe_optimizer`, `ebe_optim_kwargs`, `ebe_adtype`
  - Optimizer, solve arguments, and AD backend for EB mode optimization.
- `ebe_grad_tol`
  - Gradient tolerance for the EB optimization (`:auto` selects a data-adaptive value).
- `ebe_multistart_n`, `ebe_multistart_k`, `ebe_multistart_max_rounds`, `ebe_multistart_sampling`
  - Multistart configuration for EB optimization, controlling the number of initial points, top candidates retained, maximum restart rounds, and sampling strategy.
- `ebe_rescue_on_high_grad`
  - Enables a rescue multistart procedure if the final EB gradient norm remains above threshold.
- `ebe_rescue_multistart_n`, `ebe_rescue_multistart_k`, `ebe_rescue_max_rounds`, `ebe_rescue_grad_tol`, `ebe_rescue_multistart_sampling`
  - Configuration for the rescue strategy.

### Bound Inputs

- `lb`, `ub`
  - Optional transformed-scale bounds for free fixed effects in the M-step optimization.
  - Parameters held constant (via the `constants` fit keyword) are removed from the optimized vector; any corresponding bound entries are ignored.

## Detailed Behavior

This section provides additional details on selected options.

- `sample_schedule`
  - Can be:
    - an integer (constant samples per MCEM iteration),
    - a vector (iteration-indexed schedule),
    - a function `(iter -> n_samples)`.
- `sampler`
  - Common tested choices are `MH()` and `NUTS(...)`.
- Convergence
  - MCEM checks both parameter stabilization and Q stabilization.
  - Both must satisfy tolerances for `consecutive_params` iterations.
- `warm_start=true`
  - Reuses previous latent-state parameterization in the E-step where possible.
- `store_eb_modes` (fit keyword)
  - `true`: stores final EB modes in fit result.
  - `false`: EB modes can be recomputed later when calling random-effects accessors.

## Fit Keywords

In addition to the method constructor arguments, `fit_model` accepts several keyword arguments that control data-level settings for the MCEM run.

```julia
res = fit_model(
    dm,
    method;
    constants=(a=0.2,),
    constants_re=(; eta=(; A=0.0,)),
    penalty=NamedTuple(),
    ode_args=(),
    ode_kwargs=NamedTuple(),
    serialization=EnsembleSerial(),
    rng=Random.default_rng(),
    theta_0_untransformed=nothing,
    store_eb_modes=true,
)
```

The `constants_re` argument allows specific random-effect levels to be fixed at known values while the remaining levels are estimated.

## Optimization.jl Interface (M-step and EB)

MCEM uses [Optimization.jl](https://docs.sciml.ai/Optimization/stable/) in two places:

1. The M-step optimization over fixed effects at each iteration.
2. The EB mode optimization (final or recomputed EB modes) through the `ebe_*` configuration.

Tested M-step optimizers include:

- `OptimizationOptimJL.LBFGS(...)` (default)
- `OptimizationOptimisers.Adam(...)`
- `OptimizationBBO.BBO_adaptive_de_rand_1_bin_radiuslimited()`

When using derivative-free optimizers such as those from `OptimizationBBO`, finite bounds must be supplied:

```julia
using OptimizationBBO

lb, ub = default_bounds_from_start(dm; margin=1.0)

method_bbo = NoLimits.MCEM(;
    optimizer=OptimizationBBO.BBO_adaptive_de_rand_1_bin_radiuslimited(),
    optim_kwargs=(iterations=20,),
    sampler=MH(),
    turing_kwargs=(n_samples=10, n_adapt=0, progress=false),
    maxiters=5,
    lb=lb,
    ub=ub,
)
```

The M-step and EB optimizers can be configured independently. The following example uses L-BFGS for both but with distinct iteration limits and multistart settings:

```julia
using OptimizationOptimJL
using LineSearches

method_two_stage = NoLimits.MCEM(;
    optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
    optim_kwargs=(maxiters=120,),
    ebe_optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
    ebe_optim_kwargs=(maxiters=60,),
    ebe_grad_tol=:auto,
    ebe_multistart_n=80,
    ebe_multistart_k=16,
)
```

## Accessing Results

After fitting, results are accessed through the standard accessor interface. MCEM returns point estimates (the EM solution) rather than a posterior chain.

```julia
theta_u = NoLimits.get_params(res; scale=:untransformed)
obj = get_objective(res)
ok = get_converged(res)

re_df = get_random_effects(res)
```
