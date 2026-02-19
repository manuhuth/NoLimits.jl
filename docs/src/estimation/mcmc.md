# MCMC

Markov chain Monte Carlo (MCMC) methods provide a principled approach to Bayesian inference by drawing samples from the posterior distribution of model parameters. Rather than returning a single point estimate, the `MCMC` method in NoLimits.jl produces a full set of posterior samples, enabling rich uncertainty characterization. This is particularly valuable when the posterior landscape is complex, multimodal, or when downstream analyses require propagating parameter uncertainty into predictions. NoLimits interfaces with [Turing.jl](https://turinglang.org/) for all MCMC sampling.

`MCMC` supports both:

- **Mixed-effects models** with random effects jointly sampled alongside fixed effects
- **Fixed-effects-only models** where only population-level parameters are inferred

## Applicability

The following conditions must hold to use `MCMC`:

- All free fixed effects must have prior distributions assigned.
- At least one parameter must be sampled: either a free fixed effect, or random effects in a mixed-effects model (even if all fixed effects are held constant via `constants`).
- Penalty terms are not supported; use `MAP` for penalized estimation instead.

Note that `MCMC` samples on the natural (untransformed) parameter scale. The parameter transforms used by optimization-based methods (such as `MLE` or `Laplace`) do not apply during sampling.

## Basic Usage (Mixed Effects, Nonlinear in Random Effects)

```julia
using NoLimits
using DataFrames
using Distributions
using Turing

model = @Model begin
    @fixedEffects begin
        a = RealNumber(0.2, prior=Normal(0.0, 1.0))
        b = RealNumber(0.1, prior=Normal(0.0, 1.0))
        tau = RealNumber(0.4, scale=:log, prior=LogNormal(log(0.4), 0.5))
        sigma = RealNumber(0.3, scale=:log, prior=LogNormal(log(0.3), 0.5))
    end

    @covariates begin
        t = Covariate()
    end

    @randomEffects begin
        eta = RandomEffect(Normal(0.0, tau); column=:ID)
    end

    @formulas begin
        mu = exp(a + b * t + eta)   # nonlinear in random effects
        y ~ LogNormal(log(mu), sigma)
    end
end

df = DataFrame(
    ID = [:A, :A, :B, :B, :C, :C],
    t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    y = [1.00, 1.26, 0.93, 1.12, 1.08, 1.37],
)

dm = DataModel(model, df; primary_id=:ID, time_col=:t)

method = NoLimits.MCMC(;
    sampler=NUTS(0.75),
    turing_kwargs=(n_samples=500, n_adapt=250, progress=false),
)

res = fit_model(dm, method)
```

## Constructor Options

The `MCMC` constructor accepts the following keyword arguments, which control the sampler, the number of draws, and the automatic differentiation backend:

```julia
using NoLimits
using Turing

method = NoLimits.MCMC(;
    sampler=Turing.NUTS(0.75),
    turing_kwargs=NamedTuple(),
    adtype=Turing.AutoForwardDiff(),
    progress=false,
)
```

## Sampler-Dependent Defaults

When `n_samples` and `n_adapt` are not explicitly provided in `turing_kwargs`, NoLimits applies sampler-specific defaults:

| Sampler kind | Default `n_samples` | Default `n_adapt` |
| --- | --- | --- |
| `NUTS` | `1000` | `500` |
| `HMC` | `1500` | `750` |
| `MH` | `2500` | `0` |
| Other | `1000` | `500` |

## Turing.jl Interface

`MCMC` interfaces directly with [Turing.jl](https://turinglang.org/). The interaction between NoLimits and Turing is as follows:

- `sampler` is passed directly to `Turing.sample`.
- `turing_kwargs` are forwarded to `Turing.sample` after removing `n_samples` and `n_adapt`, which NoLimits handles explicitly.
- If `progress` is not set in `turing_kwargs`, NoLimits uses the value from `MCMC(progress=...)`.
- If `verbose` is not set in `turing_kwargs`, the default is `false`.
- `adtype` configures Turing's automatic differentiation backend before sampling begins.

## Fit Keywords

The call `fit_model(dm, NoLimits.MCMC(...); ...)` accepts the following keyword arguments:

- **`constants`** -- Fixes selected fixed effects to known values on the natural scale, removing them from the sampled parameter set.
- **`constants_re`** -- For mixed-effects models, fixes specific random-effect levels to known values while the remaining levels continue to be sampled. Level existence and dimensionality are validated.
- **`ode_args`, `ode_kwargs`** -- Forwarded to ODE solvers used during likelihood evaluation.
- **`serialization`** -- Controls whether likelihood evaluation proceeds via `EnsembleSerial()` or `EnsembleThreads()`.
- **`rng`** -- Random number generator used for sampling.
- **`theta_0_untransformed`** -- When provided (and `turing_kwargs` does not already include `initial_params`), this value initializes the free fixed effects.
- **`store_data_model`** -- If `true`, stores the `DataModel` inside the `FitResult` for use by downstream accessor functions.

Not supported:

- **`penalty`** -- Raises an error. Use `MAP` for penalized inference.

## Pattern: Fix Known Random-Effect Levels (`constants_re`)

In some workflows, certain random-effect levels are known a priori -- for example, from a previous analysis or by experimental design. These can be held fixed while the remaining levels are sampled:

```julia
constants_re = (; eta=(; A=0.0))

res_fixed_levels = fit_model(
    dm,
    NoLimits.MCMC(; sampler=NUTS(0.75), turing_kwargs=(n_samples=300, n_adapt=150, progress=false));
    constants_re=constants_re,
)
```

## Pattern: Sample Random Effects with All Fixed Effects Held Constant

For mixed-effects models, it can be useful to fix all population-level parameters and sample only the individual-level random effects. This is valid because random effects are treated as sampled parameters in the Bayesian framework:

```julia
res_re_only = fit_model(
    dm,
    NoLimits.MCMC(; sampler=MH(), turing_kwargs=(n_samples=400, n_adapt=0, progress=false));
    constants=(a=0.2, b=0.1, tau=0.4, sigma=0.3),
)
```

## Fixed-Effects-Only Example

When the model contains no random effects, `MCMC` samples only the fixed effects. The following example illustrates Bayesian inference for a Poisson regression model with count outcomes:

```julia
using NoLimits
using DataFrames
using Distributions
using Turing

model_fixed = @Model begin
    @covariates begin
        t = Covariate()
        z = Covariate()
    end

    @fixedEffects begin
        a = RealNumber(0.0, prior=Normal(0.0, 1.0))
        b = RealNumber(0.2, prior=Normal(0.0, 1.0))
    end

    @formulas begin
        lambda = exp(a + b * z)
        y ~ Poisson(lambda)
    end
end

df_fixed = DataFrame(
    ID = [1, 1, 2, 2, 3, 3],
    t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    z = [0.0, 0.2, 0.5, 0.7, 1.0, 1.2],
    y = [1, 1, 2, 2, 2, 3],
)

dm_fixed = DataModel(model_fixed, df_fixed; primary_id=:ID, time_col=:t)

res_fixed = fit_model(
    dm_fixed,
    NoLimits.MCMC(; sampler=MH(), turing_kwargs=(n_samples=300, n_adapt=0, progress=false)),
)
```

## HMM Example (Mixed Effects)

Hidden Markov models (HMMs) with subject-level random effects can also be estimated via MCMC. In the following example, random effects enter the emission probabilities nonlinearly:

```julia
using NoLimits
using DataFrames
using Distributions
using Turing

model_hmm = @Model begin
    @covariates begin
        t = Covariate()
    end

    @fixedEffects begin
        p1_r = RealNumber(0.0, prior=Normal(0.0, 1.0))
        p2_r = RealNumber(0.0, prior=Normal(0.0, 1.0))
    end

    @randomEffects begin
        eta = RandomEffect(Normal(0.0, 1.0); column=:ID)
    end

    @formulas begin
        p1 = 0.8 / (1 + exp(-(p1_r + eta))) + 0.1  # nonlinear in random effects
        p2 = 0.8 / (1 + exp(-p2_r)) + 0.1
        P = [0.9 0.1; 0.2 0.8]
        y ~ DiscreteTimeDiscreteStatesHMM(
            P,
            (Bernoulli(p1), Bernoulli(p2)),
            Categorical([0.6, 0.4]),
        )
    end
end

df_hmm = DataFrame(
    ID = [:A, :A, :A, :B, :B, :B],
    t = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
    y = [0, 1, 1, 1, 0, 1],
)

dm_hmm = DataModel(model_hmm, df_hmm; primary_id=:ID, time_col=:t)

res_hmm = fit_model(
    dm_hmm,
    NoLimits.MCMC(; sampler=MH(), turing_kwargs=(n_samples=200, n_adapt=0, progress=false)),
)
```

## Accessing Results

Because `MCMC` produces posterior samples rather than a single point estimate, some result accessors differ from those of optimization-based methods:

```julia
chain = get_chain(res)
sampler_used = get_sampler(res)
n_samples = get_n_samples(res)
observed = get_observed(res)
diagnostics = get_diagnostics(res)
```

Since there is no single optimized objective value, `get_objective(res)` returns `NaN` and `get_converged(res)` returns `missing`.
