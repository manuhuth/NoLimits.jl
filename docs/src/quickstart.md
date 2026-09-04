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
get_converged(res)                      # optimizer stopping flag
get_random_effects(res)                 # empirical Bayes estimates per subject
```

```
ComponentVector{Float64}(A0 = 10.287307571175845, k = 0.4906846606420441, omega = 0.16169743770538822, sigma = 0.11738571003728736)
```

`NoLimits.summarize(res)` collects all of it into one table:

```julia
NoLimits.summarize(res)
```

```
FitResultSummary
════════════════════════════════════════════════════════════════════════════════════════════════
Overview
  method                              : laplace
  inference                           : frequentist
  scale                               : natural
  objective                           : -0.1041
  iterations                          : 24
  parameters shown (reported / total) : 4 / 4

Parameter estimates
  parameter      Estimate
  -----------------------
  A0              10.2873
  k                0.4907
  omega            0.1617
  sigma            0.1174

Outcome data coverage
  outcome       n_obs   n_missing          unit
  ---------------------------------------------
  y                16           0           row
  TOTAL            16           0

Empirical Bayes random effects summary (across RE levels)
  random effect       n          mean            sd           q25        median           q75
  ---------------------------------------------------------------------------
  eta                 4        0.0001        0.1614       -0.0754        0.0192        0.0947
```

`A0` and `k` recover the shape of the data: a population baseline just above 10 decaying at
roughly 0.49 per time unit, with a between-subject SD of 0.16 on the log scale. Note that
`get_converged` reports the optimizer's stopping criterion, not the quality of the fit - see
[Troubleshooting](troubleshooting.md) if it is `false`.

## 5. Visualize the fit

Plotting lives in a package extension, so a Makie backend has to be installed and loaded
alongside NoLimits: `using Pkg; Pkg.add("CairoMakie")` once, then

```julia
using CairoMakie

plot_fits(res; ncols=2)
```

![Fitted exponential-decay trajectories against the observations, one panel per subject.](figures/qs/p_fit.png)

`plot_fits` overlays the model predictions on the observed data for each individual. See
[Plotting](plotting/index.md) for visual predictive checks, residual diagnostics, and
random-effects plots.

## Where to go next

- [Model Building](model-building/index.md) - the full `@Model` specification language.
- [Estimation](estimation/index.md) - every estimation method and the unified interface.
- [Tutorials](tutorials/index.md) - thirteen end-to-end worked analyses,
  including ODE-based models, neural-network components, count outcomes, and censoring.
- [NLME Methodology](nlme-methodology.md) - the mathematical framework behind the methods.
- [Troubleshooting](troubleshooting.md) - what to do when a fit fails or looks wrong.
