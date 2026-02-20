# VI

Variational inference (VI) provides approximate Bayesian inference by optimizing a parameterized posterior family instead of drawing Markov chains. In NoLimits.jl, `VI` is integrated through Turing + AdvancedVI and supports both fixed-effects-only and mixed-effects models.

Compared with `MCMC`, VI is often faster and easier to scale, but it returns an approximation whose quality depends on the selected variational family.

## Applicability

The following conditions must hold to use `VI`:

- All free fixed effects must have priors.
- At least one parameter must be sampled.
  - Fixed-only models: at least one fixed effect must remain free.
  - Mixed-effects models: random effects can be sampled even if all fixed effects are held constant.
- `penalty` is not supported.

`VI` samples on the natural (untransformed) parameter scale.

## Basic Usage (Fixed Effects)

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
        a = RealNumber(0.0, prior=Normal(0.0, 1.0))
        b = RealNumber(0.3, prior=Normal(0.0, 1.0))
        sigma = RealNumber(0.2, scale=:log, prior=LogNormal(-1.5, 0.3))
    end

    @formulas begin
        y ~ Normal(a + b * t, sigma)
    end
end

df = DataFrame(
    ID = [:A, :A, :B, :B, :C, :C],
    t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    y = [0.1, 0.45, -0.05, 0.22, 0.02, 0.32],
)

dm = DataModel(model, df; primary_id=:ID, time_col=:t)

res = fit_model(
    dm,
    NoLimits.VI(; turing_kwargs=(max_iter=300, family=:meanfield, progress=false)),
    rng=Random.Xoshiro(1),
)
```

## Constructor Options

```julia
using NoLimits

method = NoLimits.VI(; turing_kwargs=NamedTuple())
```

`turing_kwargs` are forwarded to `Turing.vi` after NoLimits consumes VI-specific control keys:

- `max_iter::Int` (default `1000`)
- `family::Symbol` (`:meanfield` or `:fullrank`, default `:meanfield`)
- `q_init` (optional custom variational initialization)
- `adtype` (default `Turing.AutoForwardDiff()`)
- `progress` / `show_progress` (default `false`)
- `algorithm` (optional AdvancedVI algorithm)
- `convergence_window`, `convergence_rtol`, `convergence_atol` (NoLimits convergence rule)

## Fit Keywords

`fit_model(dm, NoLimits.VI(...); ...)` supports:

- `constants` for fixed effects
- `constants_re` for selected random-effect levels
- `ode_args`, `ode_kwargs`
- `serialization`
- `rng`
- `theta_0_untransformed` (currently only relevant when custom `q_init` uses it)
- `store_data_model`

Not supported:

- `penalty`

## Mixed-Effects Pattern

```julia
using NoLimits
using DataFrames
using Distributions
using Random

model_re = @Model begin
    @covariates begin
        t = Covariate()
    end

    @fixedEffects begin
        a = RealNumber(0.0, prior=Normal(0.0, 1.0))
        b = RealNumber(0.2, prior=Normal(0.0, 1.0))
        omega = RealNumber(0.4, scale=:log, prior=LogNormal(-1.0, 0.4))
        sigma = RealNumber(0.2, scale=:log, prior=LogNormal(-1.5, 0.3))
    end

    @randomEffects begin
        eta = RandomEffect(Normal(0.0, omega); column=:ID)
    end

    @formulas begin
        y ~ Normal(a + b * t + eta, sigma)
    end
end

df_re = DataFrame(
    ID = [:A, :A, :B, :B, :C, :C, :D, :D],
    t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    y = [0.05, 0.25, -0.1, 0.15, 0.15, 0.35, -0.05, 0.10],
)

dm_re = DataModel(model_re, df_re; primary_id=:ID, time_col=:t)

res_re = fit_model(
    dm_re,
    NoLimits.VI(; turing_kwargs=(max_iter=400, family=:fullrank, progress=false)),
    rng=Random.Xoshiro(2),
)
```

## Accessing VI Outputs

`VI` does not return an MCMC chain. Use VI-specific accessors instead:

```julia
objective = get_objective(res)                    # final ELBO
converged = get_converged(res)                    # NoLimits convergence flag
trace = get_vi_trace(res)                         # per-iteration trace entries
state = get_vi_state(res)                         # final optimizer state
posterior = get_variational_posterior(res)        # variational posterior object

draws = sample_posterior(
    res;
    n_draws=200,
    rng=Random.Xoshiro(3),
    return_names=true,
)
```

You can inspect compact summaries with:

```julia
fit_summary = NoLimits.summarize(res)
fit_summary
```

## Uncertainty Quantification with VI

For VI fits, use `compute_uq(...; method=:chain)` to build intervals from posterior samples drawn from the variational posterior:

```julia
uq = compute_uq(
    res;
    method=:chain,
    level=0.95,
    mcmc_draws=150,
    rng=Random.Xoshiro(4),
)
```

`mcmc_warmup` is ignored for VI because there is no chain adaptation phase.

## Practical Notes

- Start with `family=:meanfield` for speed, then compare with `:fullrank` when posterior correlations are expected.
- Check `get_vi_trace(res)` and downstream predictive diagnostics (`plot_fits`, residual plots, VPC) to assess approximation quality.
- For highly multimodal or strongly non-Gaussian posteriors, `MCMC` remains the more faithful Bayesian baseline.
