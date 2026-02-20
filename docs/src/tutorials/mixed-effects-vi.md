# Mixed-Effects Tutorial 7: Variational Inference (VI)

This tutorial demonstrates VI on a mixed-effects model, including random-effects-aware posterior plotting and VI-based uncertainty quantification.

## What You Will Learn

- How to run `VI` on a model with random effects.
- How to inspect fixed and random-effect posterior coordinates.
- How to use posterior-draw plotting functions with VI.
- How to compute chain-style UQ intervals from VI posterior draws.

## Step 1: Build a Small Mixed-Effects Dataset

```julia
using NoLimits
using DataFrames
using Distributions
using Random

Random.seed!(321)

df = DataFrame(
    ID = [:A, :A, :B, :B, :C, :C, :D, :D, :E, :E],
    t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    y = [0.05, 0.30, -0.10, 0.14, 0.12, 0.38, -0.06, 0.19, 0.02, 0.33],
)
```

## Step 2: Define a Mixed-Effects Model

```julia
model = @Model begin
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

dm = DataModel(model, df; primary_id=:ID, time_col=:t)
```

## Step 3: Fit with VI

```julia
res_vi = fit_model(
    dm,
    VI(; turing_kwargs=(max_iter=450, family=:fullrank, progress=false)),
    rng=Random.Xoshiro(20),
)
```

## Step 4: Inspect Fit and Posterior Coordinates

```julia
fit_summary = NoLimits.summarize(res_vi)
fit_summary

draws_named = sample_posterior(
    res_vi;
    n_draws=200,
    rng=Random.Xoshiro(21),
    return_names=true,
)

first(draws_named.names, 8)
```

You should see fixed-effect coordinates and random-effect coordinates such as `eta_vals[...]`.

## Step 5: VI Chain UQ

```julia
uq_vi = compute_uq(
    res_vi;
    method=:chain,
    level=0.95,
    mcmc_draws=150,
    rng=Random.Xoshiro(22),
)

uq_summary = NoLimits.summarize(res_vi, uq_vi)
uq_summary
```

## Step 6: Posterior-Based Plots for Mixed Effects

```julia
p_fit_vi = plot_fits(
    res_vi;
    observable=:y,
    individuals_idx=[1, 2, 3],
    ncols=3,
    plot_mcmc_quantiles=true,
    mcmc_draws=120,
)

p_re_vi = plot_random_effect_distributions(
    res_vi;
    mcmc_draws=120,
)

p_vpc_vi = plot_vpc(
    res_vi;
    n_simulations=40,
    n_bins=3,
    mcmc_draws=120,
    rng=Random.Xoshiro(23),
)

p_qq_vi = plot_residual_qq(
    res_vi;
    residual=:quantile,
    mcmc_draws=120,
)
```

Fit plot:

```julia
p_fit_vi
```

Random-effects distribution plot:

```julia
p_re_vi
```

VPC plot:

```julia
p_vpc_vi
```

Residual QQ plot:

```julia
p_qq_vi
```

## Step 7: Optional Pattern - Sample Only Random Effects

When random effects are present, you can hold all fixed effects constant and still run VI:

```julia
res_re_only = fit_model(
    dm,
    VI(; turing_kwargs=(max_iter=200, progress=false)),
    constants=(a=0.0, b=0.2, omega=0.4, sigma=0.2),
    rng=Random.Xoshiro(24),
)
```

This is useful for conditional random-effects analyses around known fixed-effect values.

## Summary

This mixed-effects VI workflow gives:

- Fast approximate Bayesian inference with random effects.
- Direct posterior sampling via `sample_posterior`.
- Posterior-draw plotting in the same APIs used by MCMC.
- Chain-style UQ intervals from VI posterior samples.
