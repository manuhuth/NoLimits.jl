# Plotting

Effective model assessment depends on graphical diagnostics that expose patterns invisible in summary statistics alone. NoLimits provides a unified plotting interface for data inspection, fitted-model evaluation, residual analysis, random-effects diagnostics, and uncertainty visualization. Each function targets a specific aspect of model adequacy -- from individual-level trajectory fits to distributional calibration of predictions -- and all follow a consistent API.

This page contains executable examples that render directly during the documentation build.

All plotting functions accept a file-path keyword for saving output directly:
- `save_path="path/to/plot.png"` (existing)
- `plot_path="path/to/plot.png"` (alias)

## Plotting APIs

The following functions are organized by the diagnostic question they address.

**Core trajectory and data plotting:**

- `build_plot_cache`: Precompute reusable plot inputs (parameters, random effects, ODE solves, optional observation distributions) from a `FitResult` or `DataModel`. Avoids redundant computation when generating multiple plots from the same fit.
- `plot_data`: Plot observed trajectories by individual.
- `plot_fits`: Plot model-implied trajectories, with optional predictive-density overlays.
- `plot_fits_comparison`: Overlay fitted trajectories from multiple fit results on the same individual panels, enabling direct visual comparison of competing models or estimation methods.
- `plot_multistart_waterfall`: Plot successful multistart runs as objective-valued dots, sorted from best to worst by the package multistart ranking criterion.
- `plot_multistart_fixed_effect_variability`: Plot fixed-effect variability across top multistart runs as z-scores (single panel).

**Observation and predictive diagnostics:**

- `plot_observation_distributions`: Plot predictive outcome distributions at selected observations, with observed values overlaid. Reveals local miscalibration that aggregate summaries may obscure.
- `plot_vpc`: Visual predictive check comparing simulated observations with observed data. Provides a calibration assessment in the observation space.

**Residual diagnostics:**

- `get_residuals`: Return residual metrics in tabular form (`:quantile`, `:pit`, `:raw`, `:pearson`, `:logscore`).
- `plot_residuals`: Residuals versus a chosen x-axis variable (time or any varying covariate).
- `plot_residual_distribution`: Residual distribution histograms by observable.
- `plot_residual_qq`: QQ diagnostics for selected residual metrics.
- `plot_residual_pit`: Probability integral transform (PIT) diagnostics via histogram, KDE, or QQ view.
- `plot_residual_acf`: Residual autocorrelation by lag, useful for detecting unmodeled temporal structure.

**Random-effects diagnostics:**

- `plot_random_effects_pdf`: Marginal random-effect density, including posterior summaries for MCMC fits.
- `plot_random_effect_distributions`: Distribution-level diagnostics with empirical Bayes estimate (EBE) or posterior-mean overlays by level.
- `plot_random_effect_pit`: PIT diagnostics for random effects.
- `plot_random_effect_standardized`: Distribution of standardized EBEs, assessing whether the assumed random-effect variance is appropriate.
- `plot_random_effect_standardized_scatter`: Standardized EBEs versus level index or a constant covariate.
- `plot_random_effect_pairplot`: Pairwise EBE dependency visualization for detecting unexplained correlations.
- `plot_random_effects_scatter`: EBE or posterior-mean scatter versus level index or a constant covariate.

**Uncertainty quantification diagnostics:**

- `plot_uq_distributions`: Parameter uncertainty visualization from a `UQResult` (density or histogram).

## Executable setup

The following model and data are used throughout this page. The model specifies a linear trend with normally distributed random intercepts, fitted via the Laplace approximation. The data comprise four individuals, each observed at three time points.

```@example plotting_overview
using NoLimits
using DataFrames
using Distributions
using Random
using Turing

Random.seed!(12)

model = @Model begin
    @fixedEffects begin
        a = RealNumber(0.3, calculate_se=true)
        b = RealNumber(0.1, calculate_se=true)
        ω = RealNumber(0.4, scale=:log, calculate_se=true)
        σ = RealNumber(0.2, scale=:log, calculate_se=false)
    end

    @covariates begin
        t = Covariate()
    end

    @randomEffects begin
        η = RandomEffect(Normal(0.0, ω); column=:ID)
    end

    @formulas begin
        μ = a + b * t + exp(η)
        y ~ Normal(μ, σ)
    end
end

df = DataFrame(
    ID=repeat([:A, :B, :C, :D], inner=3),
    t=repeat([0.0, 1.0, 2.0], 4),
    y=[1.2, 1.6, 2.1, 1.1, 1.4, 1.8, 0.9, 1.1, 1.4, 1.3, 1.8, 2.2],
)

dm = DataModel(model, df; primary_id=:ID, time_col=:t)
res = fit_model(dm, NoLimits.Laplace(; optim_kwargs=(maxiters=8,)))
cache = build_plot_cache(res; cache_obs_dists=true)
nothing
```

## Reusing computations with `build_plot_cache`

Many diagnostic plots share the same underlying computations: parameter extraction, random-effect estimation, and (for ODE models) numerical integration. Calling `build_plot_cache` once stores these intermediate results so that subsequent plotting calls avoid redundant work.

The cache contains:

- Fixed effects used for plotting (including `params` overrides, when provided)
- Random-effect values for each individual (respecting `constants_re`, when provided)
- ODE solutions for each individual when the model includes a `@DifferentialEquation` block
- Optional per-observation outcome distributions when `cache_obs_dists=true`

For MCMC fits, cache construction also accepts `mcmc_draws` and `mcmc_warmup` to control how posterior draws are summarized before plotting.

```@example plotting_overview
cache_fast = build_plot_cache(res; cache_obs_dists=false)
cache_full = build_plot_cache(
    res;
    cache_obs_dists=true,
    params=(a=0.35,),
    constants_re=(η=Dict(:A => 0.0),),
)
nothing
```

## Data plot

Before evaluating any fitted model, it is good practice to examine the raw observations. `plot_data` displays observed trajectories for each individual, providing an immediate sense of data coverage over time, individual-level variability, and potential outliers.

```@example plotting_overview
p_data = plot_data(res)
p_data
```

## Fitted trajectories

`plot_fits` overlays observed data with model-implied trajectories for each individual. This is the primary tool for assessing whether the structural model captures the dominant trends in the data and whether individual-level fits are adequate. Discrepancies visible here may indicate model misspecification or insufficient flexibility in the random-effects structure.

```@example plotting_overview
p_fits = plot_fits(res; cache=cache)
p_fits
```

## Multistart objective-value plot

When the optimization landscape is multimodal, a single fit may converge to a local rather than global optimum. `plot_multistart_waterfall` visualizes the objective values from all successful multistart runs, sorted by rank. A flat plateau among the top runs suggests convergence to a consistent solution, while large gaps or a smooth decline may signal identifiability issues or the presence of multiple basins of attraction.

```@example plotting_overview
ms = NoLimits.Multistart(;
    dists=(; a=Normal(0.0, 0.5), b=Normal(0.0, 0.5)),
    n_draws_requested=3,
    n_draws_used=2,
    sampling=:lhs,
)

laplace_quick = NoLimits.Laplace(;
    optim_kwargs=(maxiters=4,),
    inner_kwargs=(maxiters=20,),
    multistart_n=0,
    multistart_k=0,
)

res_ms = fit_model(ms, dm, laplace_quick)
p_ms = plot_multistart_waterfall(res_ms)
p_ms
```

## Fitted trajectory comparison across models

Comparing fitted trajectories from different estimation methods or model specifications is a natural step in model selection. `plot_fits_comparison` overlays trajectories from multiple fit results on the same individual panels, making differences in structural predictions immediately visible.

For vectors, legend labels are assigned as `Model 1`, `Model 2`, and so on in input order. For `NamedTuple` and `Dict` inputs, the provided keys serve as legend labels. Per-model line styles can be customized through `PlotStyle(comparison_line_styles=Dict(...))`.

```@example plotting_overview
saem_quick = NoLimits.SAEM(;
    sampler=MH(),
    maxiters=20,
    mcmc_steps=8,
    t0=8,
    kappa=0.7,
    turing_kwargs=(n_samples=20, n_adapt=0, progress=false),
    optim_kwargs=(maxiters=80,),
    progress=false,
    verbose=false,
)
res_saem_quick = fit_model(dm, saem_quick; rng=Random.Xoshiro(24))
comparison_style = PlotStyle(comparison_line_styles=Dict("SAEM" => :dash))
p_compare = plot_fits_comparison(
    (Laplace=res, SAEM=res_saem_quick);
    individuals_idx=1:2,
    style=comparison_style,
)
p_compare
```

## Multistart fixed-effect variability

Beyond comparing objective values, it is informative to examine how fixed-effect estimates vary across top-ranked multistart runs. `plot_multistart_fixed_effect_variability` displays this variation as z-scores in a single panel, providing a concise summary of parameter stability across optimization restarts.

Configuration options:

- `scale=:untransformed` (default): display parameters on the natural scale
- Only top-level fixed-effect blocks with `calculate_se=true` are included by default
- `mode=:points` shows all values from the top-`k_best` runs; `mode=:quantiles` shows quantile summaries instead
- Use `include_parameters` or `exclude_parameters` to select specific parameter blocks by name

```@example plotting_overview
p_ms_var_points = plot_multistart_fixed_effect_variability(
    res_ms;
    k_best=3,
    mode=:points,
)
p_ms_var_points
```

```@example plotting_overview
p_ms_var_quant = plot_multistart_fixed_effect_variability(
    res_ms;
    k_best=3,
    mode=:quantiles,
    quantiles=[0.1, 0.5, 0.9],
    include_parameters=[:σ],
)
p_ms_var_quant
```

## Observation distribution diagnostic

Mean-level trajectory plots can mask important miscalibration in the predictive distribution. `plot_observation_distributions` addresses this by displaying the full predicted outcome distribution at selected observations alongside the observed value. When the observed value falls consistently in the tails of the predicted distribution, this signals that the assumed error model or its parameterization may need revision.

```@example plotting_overview
p_obs_dist = plot_observation_distributions(
    res;
    cache=cache,
    individuals_idx=1,
    obs_rows=2,
    observables=:y,
)
p_obs_dist
```

## Residual QQ diagnostic

Quantile-quantile plots provide a sensitive assessment of whether residuals conform to their expected reference distribution. `plot_residual_qq` compares observed residual quantiles against theoretical quantiles; systematic departures from the diagonal indicate structural misspecification, heavy tails, or incorrect assumptions about the observation noise.

```@example plotting_overview
p_qq = plot_residual_qq(res; cache=cache, residual=:quantile)
p_qq
```

## Visual predictive check

The visual predictive check (VPC) is a widely used diagnostic in longitudinal modelling. It evaluates a model's ability to reproduce the distribution of observed data through simulation. `plot_vpc` generates predictive envelopes from repeated simulations under the fitted model and overlays them on the observed data summaries. Agreement between the simulated envelopes and observed trends indicates that the model captures both the central tendency and the variability structure of the data.

```@example plotting_overview
p_vpc = plot_vpc(res; n_simulations=20, percentiles=[5, 50, 95])
p_vpc
```

## Random-effects distribution diagnostic

A well-specified random-effects model should produce empirical Bayes estimates consistent with the assumed distributional form. `plot_random_effects_pdf` overlays estimated random-effect values on the model-implied density, providing a direct visual check for distributional adequacy. Departures such as multimodality, skewness, or outlying values may indicate that a more flexible random-effects distribution is needed.

```@example plotting_overview
p_re_pdf = plot_random_effects_pdf(res)
p_re_pdf
```

## Uncertainty quantification plot

Reliable inference requires understanding the precision of parameter estimates. `plot_uq_distributions` visualizes parameter-level uncertainty from a computed `UQResult` object, revealing asymmetry, spread, and potential boundary effects that point summaries alone cannot convey. This is particularly informative for parameters estimated on transformed scales, where uncertainty may be highly asymmetric on the natural scale.

```@example plotting_overview
uq = compute_uq(res; method=:wald, n_draws=80, rng=Random.Xoshiro(7))
p_uq = plot_uq_distributions(uq; scale=:natural, plot_type=:density)
p_uq
```
