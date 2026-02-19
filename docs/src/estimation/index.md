# Estimation

Parameter estimation is the process of inferring model parameters from observed data. NoLimits.jl provides a unified, method-driven interface: you pass a `DataModel` and a method object to `fit_model`, which returns a structured `FitResult`. Because the same interface is used across all estimation methods, switching between approaches -- or comparing several on the same model and data -- requires changing only the method argument.

## Unified Entry Point

```julia
using NoLimits

res = fit_model(dm, NoLimits.Laplace())
```

Every `FitResult` provides a common set of accessors:

```julia
theta_u = get_params(res; scale=:untransformed)
objective = get_objective(res)
converged = get_converged(res)
summary = get_summary(res)
diagnostics = get_diagnostics(res)
```

Use `NoLimits.summarize(res)` for a compact summary table that includes the objective value, parameter estimates, and -- for mixed-effects fits -- random-effects summaries:

```julia
fit_summary = NoLimits.summarize(res)
fit_summary
```

## Available Methods

The choice of method depends on whether the model includes random effects and on the inferential framework you require:

| Model type | Methods | Notes |
| --- | --- | --- |
| Mixed-effects | `Laplace`, `LaplaceMAP`, `MCEM`, `SAEM`, `MCMC` | Require random effects in the model |
| Fixed-effects only | `MLE`, `MAP`, `MCMC` | `MLE` is likelihood-only; `MAP` adds priors; `MCMC` is Bayesian |
| Cross-method | `Multistart` | Wrapper that runs repeated fits from different starting values |

## Common Fit Keywords

Several keyword arguments are shared across methods (though not all apply to every method):

- `constants`: fix selected fixed effects to known values (specified on the natural, untransformed scale).
- `constants_re`: fix selected random-effect levels to known values (available for mixed-effects methods only).
- `penalty`: L2-style parameter penalties (not supported by `MCMC`; use `MAP` for penalized estimation instead).
- `ode_args`, `ode_kwargs`: forwarded to the ODE solver during likelihood evaluation.
- `serialization`: `EnsembleSerial()` or `EnsembleThreads()` for parallel evaluation across individuals.
- `rng`: random-number generator for reproducibility.
- `theta_0_untransformed`: custom starting values on the natural (untransformed) scale.
- `store_data_model`: stores the `DataModel` inside the `FitResult` for use by downstream accessors (default `true`).
- `store_eb_modes`: for `MCEM`/`SAEM`, controls whether empirical Bayes modes are stored during fitting.

**Prior requirements by method:**

- `MCMC` requires priors on all free fixed effects.
- `LaplaceMAP` requires priors on all fixed effects.
- `MAP` requires at least one fixed-effect prior.
- `MCEM` and `SAEM` do not incorporate fixed-effect priors in their objective.

## Multistart Wrapper

For optimization-based methods, the solution found can depend on the initial parameter values. `Multistart` addresses this sensitivity by running multiple fits from different starting points and selecting the best result:

```julia
using NoLimits
using Distributions

ms = NoLimits.Multistart(;
    dists=(; a=Normal(0.0, 1.0)),
    n_draws_requested=12,
    n_draws_used=6,
    sampling=:lhs,
)

res_ms = fit_model(ms, dm, NoLimits.Laplace(; optim_kwargs=(maxiters=80,)))
res_best = get_multistart_best(res_ms)
```

Run-level inspection is available via `get_multistart_results`, `get_multistart_errors`, `get_multistart_starts`, and `get_multistart_best_index`.

## Example: Mixed-Effects Model Fitted with Multiple Methods

The following example defines a nonlinear mixed-effects model and fits it with Laplace, MCEM, SAEM, and MCMC, illustrating how the same `DataModel` can be passed to all estimators without modification:

```julia
using NoLimits
using DataFrames
using Distributions

model = @Model begin
    @fixedEffects begin
        a = RealNumber(0.2, prior=Normal(0.0, 1.0))
        b = RealNumber(0.1, prior=Normal(0.0, 1.0))
        sigma = RealNumber(0.3, scale=:log, prior=LogNormal(0.0, 0.5))
    end

    @covariates begin
        t = Covariate()
    end

    @randomEffects begin
        eta = RandomEffect(TDist(6.0); column=:ID)
    end

    @formulas begin
        mu = a + b * t + exp(eta)   # nonlinear in the random effect
        y ~ Laplace(mu, sigma)
    end
end

df = DataFrame(
    ID = [:A, :A, :B, :B, :C, :C],
    t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    y = [1.0, 1.4, 0.8, 1.1, 1.3, 1.7],
)

dm = DataModel(model, df; primary_id=:ID, time_col=:t)
```

```julia
laplace_method = NoLimits.Laplace(; optim_kwargs=(maxiters=80,))
mcem_method = NoLimits.MCEM(;
    maxiters=8,
    sample_schedule=20,
    turing_kwargs=(n_samples=20, n_adapt=5, progress=false),
)
saem_method = NoLimits.SAEM(;
    maxiters=20,
    mcmc_steps=8,
    turing_kwargs=(n_adapt=8, progress=false),
)
mcmc_method = NoLimits.MCMC(; turing_kwargs=(n_samples=300, n_adapt=100, progress=false))

res_laplace = fit_model(dm, laplace_method)
res_mcem = fit_model(dm, mcem_method)
res_saem = fit_model(dm, saem_method)
res_mcmc = fit_model(dm, mcmc_method)
```

```julia
theta_laplace = get_params(res_laplace; scale=:untransformed)
re_laplace = get_random_effects(res_laplace)
ll_laplace = get_loglikelihood(res_laplace)

chain_mcmc = get_chain(res_mcmc)
```

## Fixing Known Random-Effect Levels

In some workflows, random-effect values are known for a subset of individuals -- for instance, from a previous analysis or an external constraint. The `constants_re` keyword fixes these levels at their known values while allowing all remaining levels to be estimated normally:

```julia
constants_re = (; eta=(; A=0.0))

res_laplace_fixed = fit_model(
    dm,
    NoLimits.Laplace(; optim_kwargs=(maxiters=80,));
    constants_re=constants_re,
)
```

## Fixed-Effects-Only Path

For models without random effects, use `MLE`, `MAP`, or `MCMC`:

```julia
model_fixed = @Model begin
    @fixedEffects begin
        a = RealNumber(0.2, prior=Normal(0.0, 1.0))
        b = RealNumber(0.1, prior=Normal(0.0, 1.0))
        sigma = RealNumber(0.3, scale=:log, prior=LogNormal(0.0, 0.5))
    end

    @covariates begin
        t = Covariate()
    end

    @formulas begin
        y ~ LogNormal(a + b * t, sigma)
    end
end

dm_fixed = DataModel(model_fixed, df; primary_id=:ID, time_col=:t)

res_mle = fit_model(dm_fixed, NoLimits.MLE(; optim_kwargs=(maxiters=120,)))
res_mcmc_fixed = fit_model(
    dm_fixed,
    NoLimits.MCMC(; turing_kwargs=(n_samples=300, n_adapt=100, progress=false)),
)
```
