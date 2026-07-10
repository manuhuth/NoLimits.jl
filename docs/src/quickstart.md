# Quickstart

This page builds, fits, and inspects a complete nonlinear mixed-effects model in a few
lines. It assumes the package is already installed - see [Installation](installation.md)
if not. The example is fully self-contained: copy it into a Julia session and run.

## 1. Define a model

We model an exponential decay in which each subject has its own baseline. The population
baseline `A0`, decay rate `k`, between-subject standard deviation `omega`, and residual
standard deviation `sigma` are fixed effects; the subject-specific deviation `eta` is a
random effect grouped by the `:ID` column.

```julia
using NoLimits
using DataFrames
using Distributions

model = @Model begin
    @fixedEffects begin
        A0    = RealNumber(10.0, scale=:log)   # population baseline
        k     = RealNumber(0.5,  scale=:log)   # population decay rate
        omega = RealNumber(0.3,  scale=:log)   # between-subject SD (log scale)
        sigma = RealNumber(0.5,  scale=:log)   # residual SD
    end

    @covariates begin
        time = Covariate()
    end

    @randomEffects begin
        eta = RandomEffect(Normal(0.0, omega); column=:ID)
    end

    @formulas begin
        pred = A0 * exp(eta) * exp(-k * time)
        y ~ Normal(pred, sigma)
    end
end
```

## 2. Bind the model to data

A [`DataModel`](data-model-construction.md) pairs the model with a `DataFrame`, validates
the schema, and groups the rows by individual. The `primary_id` and `time_col` arguments
name the subject-identifier and time columns.

```julia
df = DataFrame(
    ID   = repeat([:s1, :s2, :s3, :s4], inner=4),
    time = repeat([0.0, 1.0, 2.0, 4.0], outer=4),
    y    = [10.2, 6.1, 3.6, 1.4,
            12.5, 7.8, 4.9, 1.9,
             8.1, 4.9, 3.0, 1.1,
            11.0, 6.5, 4.1, 1.6],
)

dm = DataModel(model, df; primary_id=:ID, time_col=:time)
```

## 3. Fit

The same [`fit_model`](@ref) entry point is used for every estimation method. Here we use
the [Laplace](estimation/laplace.md) approximation, a fast and general choice for
mixed-effects models.

```julia
res = fit_model(dm, NoLimits.Laplace())
```

To try a different estimator, change only the method argument - the model and data are
untouched. For example, `fit_model(dm, NoLimits.SAEM())` or, with priors on the fixed
effects, `fit_model(dm, NoLimits.MCMC())`.

## 4. Inspect results

Results are read through accessor functions rather than field access.

```julia
get_params(res; scale=:untransformed)   # population parameter estimates
get_objective(res)                      # objective value at the optimum
get_converged(res)                      # convergence flag
get_random_effects(res)                 # empirical Bayes estimates per subject
```

## 5. Visualize the fit

```julia
using CairoMakie

plot_fits(res)
```

`plot_fits` overlays the model predictions on the observed data for each individual. See
[Plotting](plotting/index.md) for visual predictive checks, residual diagnostics, and
random-effects plots.

## Where to go next

- [Model Building](model-building/index.md) - the full `@Model` specification language.
- [Estimation](estimation/index.md) - every estimation method and the unified interface.
- [Tutorials](tutorials/mixed-effects-multiple-methods.md) - end-to-end worked analyses,
  including ODE-based models, neural-network components, count outcomes, and censoring.
- [NLME Methodology](nlme-methodology.md) - the mathematical framework behind the methods.
