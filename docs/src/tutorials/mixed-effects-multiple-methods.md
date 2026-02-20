# Mixed-Effects Tutorial 1: Nonlinear Random-Effects Model Across Multiple Estimation Methods

Nonlinear mixed-effects (NLME) models are a cornerstone of longitudinal data analysis in the biological sciences. They describe how individual trajectories vary around a shared population-level trend -- capturing, for example, how different organisms grow at different rates toward different asymptotes, even when the underlying biological mechanism is the same. A natural question arises in practice: **how sensitive are my conclusions to the estimation algorithm I choose?** This tutorial addresses that question directly. You will fit a single nonlinear growth model to a classic biological dataset using five distinct estimation strategies, then compare the results in terms of fitted trajectories, observation-level predictions, and parameter uncertainty. By the end, you will have a practical template for multi-method comparison and a clear intuition for when each approach is most appropriate.

## What You Will Learn

By the end of this tutorial, you will be able to:

- **Build** a nonlinear mixed-effects model with lognormal random effects on a growth asymptote.
- **Configure** five fundamentally different estimation strategies -- Laplace approximation, MCEM, SAEM, full Bayesian MCMC, and variational inference (VI) -- with sensible defaults.
- **Compare** methods in predictive space using NoLimits' diagnostic and visualization tools, rather than relying solely on objective function values.
- **Interpret** where the estimators converge, where they diverge, and what each method uniquely provides.

The goal is not just to run five fits, but to build understanding of the trade-offs involved in choosing an estimation strategy for your own longitudinal analyses.

## Step 1: Data Setup

In this first step, you will load the Orange tree growth dataset, a classic longitudinal dataset originally from Draper and Smith (1981) and available in R's `datasets` package. The dataset records the trunk circumference of five orange trees measured at seven time points over approximately four years. Although small, it is representative of a broad class of problems in biology: repeated measurements of a continuous outcome on a set of individuals, with growth that follows a saturating nonlinear trajectory. The between-tree variation in final size makes it a natural candidate for random-effects modeling.

The code below loads the required packages and retrieves the data directly from the Rdatasets GitHub repository.

```julia
using NoLimits
using CSV
using DataFrames
using Distributions
using Downloads
using Random
using SciMLBase
using Turing

include(joinpath(@__DIR__, "_data_loaders.jl"))

Random.seed!(42)

df = load_orange()

first(df, 8)
```

## Step 2: Define the Nonlinear Mixed-Effects Model

Next, you will specify the statistical model. The biological reasoning is straightforward: each tree's circumference follows a saturating growth curve, increasing from an initial size toward a tree-specific maximum. You will model this maximum (the asymptote) as a random effect, allowing each tree to have its own upper bound while sharing a common growth shape across the population.

Concretely, the model uses a logistic-style saturating function with three population-level parameters: an initial size `phi1`, a log-scale population mean for the asymptote `log_vmax`, and a midpoint parameter `phi3` that controls the timing of the growth inflection. Each tree's individual asymptote `vmax_i` is drawn from a lognormal distribution, which ensures positivity and places between-tree variability on a multiplicative scale -- a natural choice when larger individuals tend to show proportionally larger variation. The observation model is also lognormal, so residual variability scales with the predicted circumference rather than being additive. To maintain numerical stability during optimization, the predicted mean is passed through a `softplus` function that enforces positivity without introducing hard discontinuities.

All fixed effects are given weakly informative priors. These priors are not strictly necessary for the optimization-based methods (Laplace, MCEM, SAEM), but they are required for MCMC and VI and serve to regularize the likelihood surface for all methods.

```julia
model = @Model begin
    @helpers begin
        softplus(u) = u > 20 ? u : log1p(exp(u))
    end

    @covariates begin
        age = Covariate()
    end

    @fixedEffects begin
        phi1     = RealNumber(30.0,  prior=LogNormal(log(30.0), 0.30), calculate_se=true)
        log_vmax = RealNumber(10.0,  prior=Normal(5.00, 0.35),          calculate_se=true)
        phi3     = RealNumber(700.0, prior=LogNormal(log(700.0), 0.30), calculate_se=true)
        omega    = RealNumber(0.3, scale=:log, prior=LogNormal(log(0.155), 0.35), calculate_se=true)
        sigma    = RealNumber(0.3, scale=:log, prior=LogNormal(log(0.113), 0.30), calculate_se=true)
    end

    @randomEffects begin
        vmax_i = RandomEffect(LogNormal(log_vmax, omega); column=:Tree)
    end

    @formulas begin
        mu_raw = phi1 + (vmax_i - phi1) / (1 + exp(-(age - phi3) / 100))
        mu = softplus(mu_raw) + 1e-6
        circumference ~ LogNormal(log(mu), sigma)
    end
end
```

### Model Summary

You can inspect the model structure to verify that all blocks were parsed correctly and that the parameter dimensions, scales, and priors match what you intended.

```julia
model_summary = NoLimits.summarize(model)
model_summary
```

## Step 3: Build the DataModel and Configure Estimation Methods

In this step, you will wrap the model and data together into a `DataModel` -- a structure that validates the data schema, groups individuals into batches for the random effects structure, and prepares internal representations for each estimation method.

You will then configure five estimation methods. Each represents a fundamentally different strategy for handling the random effects integral that appears in the marginal likelihood:

- **Laplace** approximates the integral analytically using a second-order Taylor expansion around each individual's best estimate of the random effects (the empirical Bayes estimate). It is fast and deterministic, making it a good default for moderate-sized problems. However, the approximation can lose accuracy when the true distribution of random effects is far from Gaussian.

- **MCEM** (Monte Carlo Expectation-Maximization) uses MCMC sampling within each iteration to approximate the expected complete-data log-likelihood, then maximizes that approximation. In plain terms, it alternates between "filling in" the missing random effects via sampling and updating the population parameters given those samples. It is more robust than Laplace to non-Gaussian random effects but requires more computation.

- **SAEM** (Stochastic Approximation EM) follows a similar alternating logic but replaces the full sampling step with a stochastic approximation that updates a running average of sufficient statistics. This means it converges with fewer samples per iteration than MCEM, making it attractive for larger problems, though its stochastic nature can make convergence harder to diagnose.

- **MCMC** (Markov chain Monte Carlo) samples the full joint posterior over both fixed and random effects. Rather than returning a single "best" parameter estimate, it produces a collection of plausible parameter sets that together characterize uncertainty. This provides the richest picture of parameter uncertainty -- including asymmetric or multimodal posteriors -- but is the most computationally expensive and requires careful convergence assessment.

- **VI** (variational inference) optimizes an approximate posterior family instead of drawing full chains. It is often much faster than MCMC and still provides posterior draws for predictive and uncertainty analyses, but it is approximate and can miss complex posterior features if the chosen variational family is too restrictive.

The configuration values below are chosen to balance runtime and stability for this tutorial; in a research setting, you would typically increase iteration counts, sample sizes, and warmup periods.

```julia
dm = DataModel(model, df; primary_id=:Tree, time_col=:age)

laplace_method = NoLimits.Laplace(; multistart_n=0, multistart_k=0, optim_kwargs=(maxiters=120,))

mcem_method = NoLimits.MCEM(;
    maxiters=6,
    sample_schedule=i -> min(40 + 20 * (i - 1), 140),
    turing_kwargs=(n_samples=40, n_adapt=15, progress=false),
    optim_kwargs=(maxiters=120,),
    progress=false,
)

saem_method = NoLimits.SAEM(;
    maxiters=80,
    mcmc_steps=16,
    t0=15,
    kappa=0.65,
    turing_kwargs=(n_adapt=20, progress=false),
    optim_kwargs=(maxiters=160,),
    verbose=false,
    progress=false,
)

mcmc_method = NoLimits.MCMC(;
    sampler=NUTS(0.75),
    progress=false,
    turing_kwargs=(n_samples=1000, n_adapt=500, progress=false),
)

vi_method = NoLimits.VI(;
    turing_kwargs=(max_iter=30000, progress=false),
)

serialization = SciMLBase.EnsembleThreads()
```

### DataModel Summary

Before proceeding to estimation, inspect the DataModel summary to confirm that individuals, covariates, and random effect groupings were detected correctly.

```julia
dm_summary = NoLimits.summarize(dm)
dm_summary
```

## Step 4: Fit All Methods

With the model, data, and methods configured, you will now fit the same DataModel with all five estimators. Each call to `fit_model` returns a `FitResult` object that stores the estimated parameters, convergence diagnostics, and a reference to the DataModel for downstream analysis.

Each method receives a different random seed to ensure reproducibility while allowing independent stochastic behavior across methods.

```julia
res_laplace = fit_model(dm, laplace_method; serialization=serialization, rng=Random.Xoshiro(11))
res_mcem = fit_model(dm, mcem_method; serialization=serialization, rng=Random.Xoshiro(12))
res_saem = fit_model(dm, saem_method; serialization=serialization, rng=Random.Xoshiro(13))
res_mcmc = fit_model(dm, mcmc_method; serialization=serialization, rng=Random.Xoshiro(14))
res_vi = fit_model(dm, vi_method; serialization=serialization, rng=Random.Xoshiro(12))

```

### FitResult Summaries

Each fit result can be summarized to display estimated parameter values, convergence status, and method-specific diagnostics. Reviewing these summaries side by side is a quick first check for whether the methods have arrived at broadly similar parameter estimates.

```julia
fit_summary_laplace = NoLimits.summarize(res_laplace)
fit_summary_mcem = NoLimits.summarize(res_mcem)
fit_summary_saem = NoLimits.summarize(res_saem)
fit_summary_mcmc = NoLimits.summarize(res_mcmc)
fit_summary_vi = NoLimits.summarize(res_vi)

fit_summary_laplace
```

## Step 5: Compare Objective Values (Laplace, MCEM, SAEM, VI)

It is tempting to compare objective function values across methods, but this requires care: each method optimizes a different quantity, so raw values are not directly comparable.

```julia
objectives = (
    laplace=NoLimits.get_objective(res_laplace),
    mcem=NoLimits.get_objective(res_mcem),
    saem=NoLimits.get_objective(res_saem),
    vi=NoLimits.get_objective(res_vi),
)

objectives
```

The signs and magnitudes differ because each method defines its objective differently:

- **Laplace** reports the minimized value of the Laplace-approximated marginal likelihood (a loss function, so lower is better).
- **MCEM** and **SAEM** report the EM auxiliary quantity `Q` at the final iterate, which is a lower bound on the log-likelihood (higher is better).
- **VI** reports the final evidence lower bound (ELBO), which is optimized over the variational family (higher is better within the same VI configuration).

Because these quantities differ by construction, raw values are not directly comparable across methods. Within a single method family, however, objective values can be compared meaningfully -- for example, when evaluating different starting points or model specifications with the same estimator.

## Step 6: Fitted Trajectories for the First Two Individuals

The most informative way to compare methods is in predictive space: do the fitted trajectories agree when overlaid on the observed data? In this step, you will generate fit plots for the first two trees under each method. For MCMC and VI, you will additionally overlay posterior predictive quantile bands (5th and 95th percentiles), which provide a visual summary of prediction uncertainty that accounts for both parameter and random-effect uncertainty.

```julia
inds = collect(1:min(2, length(dm.individuals)))

p_fit_laplace = plot_fits(
    res_laplace;
    observable=:circumference,
    individuals_idx=inds,
    ncols=2,
    shared_x_axis=true,
    shared_y_axis=true,
)

p_fit_mcem = plot_fits(
    res_mcem;
    observable=:circumference,
    individuals_idx=inds,
    ncols=2,
    shared_x_axis=true,
    shared_y_axis=true,
)

p_fit_saem = plot_fits(
    res_saem;
    observable=:circumference,
    individuals_idx=inds,
    ncols=2,
    shared_x_axis=true,
    shared_y_axis=true,
)

p_fit_mcmc = plot_fits(
    res_mcmc;
    observable=:circumference,
    individuals_idx=inds,
    ncols=2,
    shared_x_axis=true,
    shared_y_axis=true,
    plot_mcmc_quantiles=true,
    mcmc_quantiles=[5, 95],
    mcmc_warmup=500,
    mcmc_draws=300,
)

p_fit_vi = plot_fits(
    res_vi;
    observable=:circumference,
    #individuals_idx=inds,
    ncols=2,
    shared_x_axis=true,
    shared_y_axis=true,
    plot_mcmc_quantiles=true,
    mcmc_quantiles=[5, 95],
    mcmc_draws=300,
)

```

When all five methods are well-calibrated, you should see broadly similar trajectories. Differences, when they appear, tend to be most visible in the tails of the growth curve where data are sparse.

Laplace fit plot:

```julia
p_fit_laplace
```

MCEM fit plot:

```julia
p_fit_mcem
```

SAEM fit plot:

```julia
p_fit_saem
```

MCMC fit plot (with posterior predictive bands):

```julia
p_fit_mcmc
```

VI fit plot (with posterior predictive bands):

```julia
p_fit_vi
```

## Step 7: Observation Distribution Diagnostics (First Individual)

Beyond trajectory-level agreement, you can examine how well each method captures the observation-level distribution. The plots below compare the observed circumference value for the first observation of the first tree against the model-implied observation distribution at that data point. If the model is well-specified, the observed value should fall in a region of reasonable density under the predicted distribution.

This diagnostic is particularly useful for detecting model misspecification. If the observed value consistently falls in the tail of the predicted distribution across individuals and methods, the observation model (here, lognormal) may need revision.

```julia
p_obs_laplace = plot_observation_distributions(
    res_laplace;
    observables=:circumference,
    individuals_idx=1,
    obs_rows=1,
)

p_obs_mcem = plot_observation_distributions(
    res_mcem;
    observables=:circumference,
    individuals_idx=1,
    obs_rows=1,
)

p_obs_saem = plot_observation_distributions(
    res_saem;
    observables=:circumference,
    individuals_idx=1,
    obs_rows=1,
)

p_obs_mcmc = plot_observation_distributions(
    res_mcmc;
    observables=:circumference,
    individuals_idx=1,
    obs_rows=1,
    mcmc_warmup=500,
    mcmc_draws=300,
)

p_obs_vi = plot_observation_distributions(
    res_vi;
    observables=:circumference,
    individuals_idx=1,
    obs_rows=1,
    mcmc_draws=300,
)

```

Laplace observation distribution:

```julia
p_obs_laplace
```

MCEM observation distribution:

```julia
p_obs_mcem
```

SAEM observation distribution:

```julia
p_obs_saem
```

MCMC observation distribution:

```julia
p_obs_mcmc
```

VI observation distribution:

```julia
p_obs_vi
```

## Step 8: Uncertainty Quantification Across Methods

A key reason to compare methods is to understand how they characterize parameter uncertainty. The optimization-based methods (Laplace, MCEM, SAEM) return point estimates; you can obtain approximate uncertainty via Wald-type confidence intervals, which are derived from the curvature of the objective function at the optimum. Intuitively, a sharply peaked objective implies tight uncertainty, while a flat objective implies wide intervals. MCMC and VI, by contrast, produce posterior draws (exact-chain and variational-approximation respectively), enabling distribution-based uncertainty summaries.

Below, you will compute uncertainty quantification (UQ) summaries for all five methods and generate density plots of the resulting parameter distributions on the natural (untransformed) scale.

```julia
uq_laplace = compute_uq(
    res_laplace;
    method=:wald,
    vcov=:hessian,
    pseudo_inverse=true,
    serialization=serialization,
    n_draws=400,
    rng=Random.Xoshiro(101),
)
uq_mcem = compute_uq(
    res_mcem;
    method=:wald,
    vcov=:hessian,
    re_approx=:laplace,
    pseudo_inverse=true,
    serialization=serialization,
    n_draws=400,
    rng=Random.Xoshiro(102),
)
uq_saem = compute_uq(
    res_saem;
    method=:wald,
    vcov=:hessian,
    re_approx=:laplace,
    pseudo_inverse=true,
    serialization=serialization,
    n_draws=400,
    rng=Random.Xoshiro(103),
)
uq_mcmc = compute_uq(
    res_mcmc;
    method=:chain,
    serialization=serialization,
    mcmc_warmup=500,
    mcmc_draws=300,
    rng=Random.Xoshiro(104),
)
uq_vi = compute_uq(
    res_vi;
    method=:chain,
    serialization=serialization,
    mcmc_draws=300,
    rng=Random.Xoshiro(105),
)

p_uq_laplace = plot_uq_distributions(uq_laplace; scale=:natural, plot_type=:density, show_legend=false)
p_uq_mcem = plot_uq_distributions(uq_mcem; scale=:natural, plot_type=:density, show_legend=false)
p_uq_saem = plot_uq_distributions(uq_saem; scale=:natural, plot_type=:density, show_legend=false)
p_uq_mcmc = plot_uq_distributions(uq_mcmc; scale=:natural, plot_type=:density, show_legend=false)
p_uq_vi = plot_uq_distributions(uq_vi; scale=:natural, plot_type=:density, show_legend=false)

```

### UQ Summaries

The summaries below report point estimates alongside uncertainty intervals for each parameter. Where the methods agree, you can be confident that inference is robust. Where they diverge, the discrepancy may signal sensitivity to the estimation strategy, an under-identified parameter, or simply a need for more data.

```julia
uq_summary_laplace = NoLimits.summarize(uq_laplace)
uq_summary_mcem = NoLimits.summarize(uq_mcem)
uq_summary_saem = NoLimits.summarize(uq_saem)
uq_summary_mcmc = NoLimits.summarize(uq_mcmc)
uq_summary_vi = NoLimits.summarize(uq_vi)

fit_uq_summary_laplace = NoLimits.summarize(res_laplace, uq_laplace)
fit_uq_summary_mcem = NoLimits.summarize(res_mcem, uq_mcem)
fit_uq_summary_saem = NoLimits.summarize(res_saem, uq_saem)
fit_uq_summary_mcmc = NoLimits.summarize(res_mcmc, uq_mcmc)
fit_uq_summary_vi = NoLimits.summarize(res_vi, uq_vi)

fit_uq_summary_laplace
```

Laplace UQ distribution:

```julia
p_uq_laplace
```

MCEM UQ distribution:

```julia
p_uq_mcem
```

SAEM UQ distribution:

```julia
p_uq_saem
```

MCMC UQ distribution:

```julia
p_uq_mcmc
```

VI UQ distribution:

```julia
p_uq_vi
```

## Interpretation and Practical Guidance

Several principles emerge from this multi-method comparison:

**Evaluate agreement in predictive space, not objective space.** Because each method optimizes a different quantity, comparing raw objective values across methods is misleading. Instead, compare fitted trajectories, observation-level distributions, and uncertainty intervals. When methods agree in predictive space, you can be more confident that your conclusions are robust to the choice of estimator.

**Method agreement on central structure is the norm for well-specified models.** For a dataset like Orange, where the model is a reasonable description of the data-generating process, Laplace, MCEM, and SAEM will typically recover similar point estimates and trajectory shapes. Disagreement is a useful diagnostic signal -- it may indicate model misspecification, insufficient iterations, or a challenging likelihood surface.

**MCMC provides the richest uncertainty characterization, while VI provides a faster approximate posterior alternative.** Posterior predictive bands and marginal posterior distributions from MCMC capture asymmetry, multimodality, and correlation structure most faithfully. VI can be substantially faster and often gives useful posterior summaries, but remains an approximation whose quality should be checked against predictive diagnostics and, when feasible, MCMC.

**Choose your method based on your inferential goal.** If you need fast point estimates with approximate standard errors for model selection or screening, Laplace is often sufficient. If you need robust estimates under flexible random-effect distributions, SAEM or MCEM may be preferable. If full posterior inference is the goal -- for example, for decision-making under uncertainty or for propagating parameter uncertainty into downstream predictions -- MCMC is the strongest choice. If you need Bayesian posterior-style outputs with lower runtime, VI is a practical middle ground.
