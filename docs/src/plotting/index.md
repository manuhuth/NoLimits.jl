# Plotting

Effective model assessment depends on graphical diagnostics that expose patterns invisible in summary statistics alone. NoLimits provides a unified plotting interface for data inspection, fitted-model evaluation, residual analysis, random-effects diagnostics, and uncertainty visualization. Each function targets a specific aspect of model adequacy - from individual-level trajectory fits to distributional calibration of predictions - and all follow a consistent API.

This page contains executable examples that render directly during the documentation build.

All plotting functions accept a file-path keyword for saving output directly:
- `save_path="path/to/plot.png"` (existing)
- `plot_path="path/to/plot.png"` (alias)

## Plotting APIs

The plotting functions fall into a handful of diagnostic groups:

- **Trajectory and data plotting:** `plot_data` (raw observed trajectories), `plot_fits` and `plot_fits_comparison` (model-implied trajectories, optionally overlaying multiple fits), and `build_plot_cache` (precompute parameters, random effects, ODE solves, and optional observation distributions for reuse across plots).
- **Multistart diagnostics:** `plot_multistart_waterfall` (sorted objective values) and `plot_multistart_fixed_effect_variability` (parameter stability across restarts).
- **Observation and predictive diagnostics:** `plot_observation_distributions` (per-observation predicted densities) and `plot_vpc` (visual predictive check).
- **Residual diagnostics:** `get_residuals` (tabular metrics: `:quantile`, `:pit`, `:raw`, `:pearson`, `:logscore`) plus `plot_residuals`, `plot_residual_distribution`, `plot_residual_qq`, `plot_residual_pit`, and `plot_residual_acf`.
- **Random-effects diagnostics:** `plot_random_effects_pdf`, `plot_random_effect_distributions`, `plot_random_effect_pit`, `plot_random_effect_standardized`, `plot_random_effect_standardized_scatter`, `plot_random_effect_pairplot`, and `plot_random_effects_scatter`.
- **Uncertainty quantification:** `plot_uq_distributions` (parameter uncertainty from a `UQResult`).

See the [Plotting and Diagnostics](../api.md#Plotting-and-Diagnostics) section of the API reference for the full list of functions and their arguments.

## Executable setup

The following model and data are used throughout this page. The model specifies a linear trend with normally distributed random intercepts, fitted via the Laplace approximation. The data comprise four individuals, each observed at three time points.

```@example plotting_overview
using NoLimits
using CairoMakie
CairoMakie.activate!(type = "png")
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

For vectors, legend labels are assigned as `Model 1`, `Model 2`, and so on in input order. For `NamedTuple` and `Dict` inputs, the provided keys serve as legend labels. Per-model line styles can be customized through `PlotStyle(comparison_line_styles=Dict(...))`; `PlotStyle` is the shared styling object accepted by all plotting functions (colors, line styles, layout) - see its docstring in the API reference for the full set of fields. The `individuals_idx` keyword (here `1:2`) restricts the panels to the selected individuals.

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

By default it plots blocks with `calculate_se=true` on the natural scale (`scale=:untransformed`) using `mode=:points` (all values from the top-`k_best` runs); `mode=:quantiles` shows quantile summaries instead, and `include_parameters`/`exclude_parameters` select specific blocks by name. See the [Plotting and Diagnostics](../api.md#Plotting-and-Diagnostics) API reference for the full argument list.

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

Here `obs_rows` selects which observation rows (per individual) to display and `observables` selects which outcome(s) to plot; see the API reference for the full signature.

## Residual QQ diagnostic

Quantile-quantile plots provide a sensitive assessment of whether residuals conform to their expected reference distribution. `plot_residual_qq` compares observed residual quantiles against theoretical quantiles; systematic departures from the diagonal indicate structural misspecification, heavy tails, or incorrect assumptions about the observation noise.

```@example plotting_overview
p_qq = plot_residual_qq(res; cache=cache, residual=:quantile)
p_qq
```

The `residual` keyword chooses which residual metric to plot (one of the columns returned by `get_residuals`: `:quantile`, `:pit`, `:raw`, `:pearson`, `:logscore`).

## Visual predictive check

The visual predictive check (VPC) is a widely used diagnostic in longitudinal modeling. It evaluates a model's ability to reproduce the distribution of observed data through simulation. `plot_vpc` generates predictive envelopes from repeated simulations under the fitted model and overlays them on the observed data summaries. Agreement between the simulated envelopes and observed trends indicates that the model captures both the central tendency and the variability structure of the data.

```@example plotting_overview
p_vpc = plot_vpc(res; n_simulations=20, percentiles=[5, 50, 95])
p_vpc
```

`n_simulations` sets the number of simulated datasets used to build the predictive envelopes, and `percentiles` selects the envelope quantiles (here the 5th, 50th, and 95th); see the API reference for the remaining keywords.

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
