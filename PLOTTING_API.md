# Plotting API

This document summarizes the current exported plotting API in `src/plotting`.

## Core Types and Cache

### PlotStyle
- Exported type: `PlotStyle`
- Purpose: publication-oriented styling (colors, fonts, line/marker widths, subplot sizing).
- Used by all plotting functions via `style=PlotStyle()`.

### PlotCache
- Exported type: `PlotCache`
- Fields: `signature`, `sols`, `obs_dists`, `chain`, `params`, `random_effects`, `meta`
- Purpose: reuse expensive plot inputs (ODE solutions, observation distributions, MCMC chain snapshots, random effects).

### build_plot_cache
Signatures (key args):
- `build_plot_cache(res::FitResult; dm=nothing, params=NamedTuple(), constants_re=NamedTuple(), cache_obs_dists=false, ode_args=(), ode_kwargs=NamedTuple(), mcmc_draws=1000, mcmc_warmup=nothing, rng=Random.default_rng())`
- `build_plot_cache(res::MultistartFitResult; kwargs...)`
- `build_plot_cache(dm::DataModel; params=NamedTuple(), constants_re=NamedTuple(), cache_obs_dists=false, ode_args=(), ode_kwargs=NamedTuple(), rng=Random.default_rng())`

Behavior:
- For `FitResult`, `constants_re` defaults to values stored in `res.fit_kwargs` when present.
- For MCMC fits, cache stores posterior-fixed means and sampled random effects summaries.
- For DE models, cache can include solved trajectories (`sols`).
- With `cache_obs_dists=true`, cache includes per-observation distributions (`obs_dists`).

## Data and Fit Plots

### plot_data
Signatures:
- `plot_data(res::FitResult; dm=nothing, x_axis_feature=nothing, individuals_idx=nothing, shared_x_axis=true, shared_y_axis=true, ncols=DEFAULT_PLOT_COLS, style=PlotStyle(), kwargs_subplot=NamedTuple(), kwargs_layout=NamedTuple(), save_path=nothing)`
- `plot_data(dm::DataModel; x_axis_feature=nothing, individuals_idx=nothing, shared_x_axis=true, shared_y_axis=true, ncols=DEFAULT_PLOT_COLS, style=PlotStyle(), kwargs_subplot=NamedTuple(), kwargs_layout=NamedTuple(), save_path=nothing)`

Behavior:
- Scatter plot of observed data, one panel per individual.
- X-axis defaults to time (`t`), or uses `x_axis_feature`.

### plot_fits
Signatures:
- `plot_fits(res::FitResult; dm=nothing, plot_density=false, plot_func=mean, plot_data_points=true, observable=nothing, individuals_idx=nothing, x_axis_feature=nothing, shared_x_axis=true, shared_y_axis=true, ncols=DEFAULT_PLOT_COLS, style=PlotStyle(), kwargs_subplot=NamedTuple(), kwargs_layout=NamedTuple(), save_path=nothing, cache=nothing, params=NamedTuple(), constants_re=NamedTuple(), cache_obs_dists=false, plot_mcmc_quantiles=false, mcmc_quantiles=[5,95], mcmc_quantiles_alpha=0.8, mcmc_draws=1000, mcmc_warmup=nothing, rng=Random.default_rng())`
- `plot_fits(dm::DataModel; plot_density=false, plot_func=mean, plot_data_points=true, observable=nothing, individuals_idx=nothing, x_axis_feature=nothing, shared_x_axis=true, shared_y_axis=true, ncols=DEFAULT_PLOT_COLS, style=PlotStyle(), kwargs_subplot=NamedTuple(), kwargs_layout=NamedTuple(), save_path=nothing, params=NamedTuple(), constants_re=NamedTuple(), cache_obs_dists=false, mcmc_draws=1000, mcmc_warmup=nothing, rng=Random.default_rng())`

Behavior:
- Overlay observed points and fitted summary curve.
- `plot_func` is applied to each predicted observation distribution (default `mean`).
- `plot_density=true` overlays density/PMF information.
- MCMC mode supports posterior bands via `plot_mcmc_quantiles=true`.
- If multiple observables exist and `observable=nothing`, first observable is used (with warning).

## Observation Distribution Plots

### plot_observation_distributions
Signatures:
- `plot_observation_distributions(res::FitResult; dm=nothing, individuals_idx=nothing, obs_rows=nothing, observables=nothing, x_axis_feature=nothing, shared_x_axis=true, shared_y_axis=true, ncols=DEFAULT_PLOT_COLS, style=PlotStyle(), cache=nothing, cache_obs_dists=false, constants_re=NamedTuple(), mcmc_quantiles=[5,95], mcmc_quantiles_alpha=0.8, mcmc_draws=1000, mcmc_warmup=nothing, rng=Random.default_rng(), save_path=nothing, kwargs_subplot=NamedTuple(), kwargs_layout=NamedTuple())`
- `plot_observation_distributions(dm::DataModel; kwargs...)`

Behavior:
- One panel per selected observation row and observable.
- Shows the predicted observation distribution at that row, plus observed value marker.
- Continuous outcomes: density curves.
- Discrete outcomes: PMF bars/points.
- MCMC: posterior mean curve/PMF with quantile envelopes.
- Default selection is first individual when `individuals_idx=nothing`.
- Default selection is first observable when `observables=nothing` (warns if multiple observables exist).

## Residual API

### get_residuals
Signatures:
- `get_residuals(res::FitResult; dm=nothing, cache=nothing, observables=nothing, individuals_idx=nothing, obs_rows=nothing, x_axis_feature=nothing, params=NamedTuple(), constants_re=NamedTuple(), cache_obs_dists=true, residuals=[:quantile,:pit,:raw,:pearson,:logscore], fitted_stat=mean, randomize_discrete=true, cdf_fallback_mc=0, ode_args=(), ode_kwargs=NamedTuple(), mcmc_draws=1000, mcmc_warmup=nothing, mcmc_quantiles=[5,95], rng=Random.default_rng(), return_draw_level=false)`
- `get_residuals(dm::DataModel; same keywords...)`

Residual metrics:
- `:pit`
- `:quantile`
- `:raw`
- `:pearson`
- `:logscore`

Output:
- A `DataFrame` with observation identity columns plus residual summaries.
- Includes `pit_qlo/pit_qhi`, `res_quantile_qlo/res_quantile_qhi`, etc.
- Includes `draw` and `n_draws` for MCMC summaries/draw-level output.

Behavior:
- MCMC defaults to summary across draws; set `return_draw_level=true` for per-draw rows.
- For non-MCMC, computation can reuse/build `PlotCache` with `cache_obs_dists=true`.

### plot_residuals
Signatures:
- `plot_residuals(res::FitResult; residual=:quantile, ...)`
- `plot_residuals(dm::DataModel; residual=:quantile, ...)`

Behavior:
- Scatter of selected residual metric vs x-axis (time or selected varying covariate), grouped by individual and observable.

### plot_residual_distribution
Signatures:
- `plot_residual_distribution(res::FitResult; residual=:quantile, bins=20, ...)`
- `plot_residual_distribution(dm::DataModel; residual=:quantile, bins=20, ...)`

Behavior:
- Histogram per observable for selected residual metric.

### plot_residual_qq
Signatures:
- `plot_residual_qq(res::FitResult; residual=:quantile, ...)`
- `plot_residual_qq(dm::DataModel; residual=:quantile, ...)`

Behavior:
- QQ plots per observable.
- For `:pit`, compares to Uniform quantiles.
- For non-`:pit` metrics, compares to Normal quantiles.

### plot_residual_pit
Signatures:
- `plot_residual_pit(res::FitResult; show_hist=true, show_kde=false, show_qq=false, ...)`
- `plot_residual_pit(dm::DataModel; show_hist=true, show_kde=false, show_qq=false, ...)`

Behavior:
- PIT diagnostics per observable.
- If multiple of `show_hist/show_kde/show_qq` are true, function warns and falls back to histogram.

### plot_residual_acf
Signatures:
- `plot_residual_acf(res::FitResult; residual=:quantile, max_lag=5, ...)`
- `plot_residual_acf(dm::DataModel; residual=:quantile, max_lag=5, ...)`

Behavior:
- Residual autocorrelation by observable (average across individuals), up to `max_lag`.

## Random-Effects Diagnostics

Supported fit types:
- `Laplace`
- `LaplaceMAP`
- `FOCEI`
- `FOCEIMAP`
- `MCEM`
- `SAEM`
- `MCMC`

Not supported:
- `MLE`
- `MAP`

### plot_random_effect_distributions
Signature (key args):
- `plot_random_effect_distributions(res; dm=nothing, re_names=nothing, levels=nothing, individuals_idx=nothing, shared_x_axis=true, shared_y_axis=true, ncols=DEFAULT_PLOT_COLS, style=PlotStyle(), mcmc_draws=1000, mcmc_warmup=nothing, mcmc_quantiles=[5,95], mcmc_quantiles_alpha=0.8, flow_samples=500, flow_plot=:kde, flow_bins=20, flow_bandwidth=nothing, rng=Random.default_rng(), save_path=nothing, kwargs_subplot=NamedTuple(), kwargs_layout=NamedTuple())`

Behavior:
- Per-level marginal RE distribution with marker at EBE for Laplace/LaplaceMAP/FOCEI/FOCEIMAP/MCEM/SAEM.
- Per-level marginal RE distribution with marker at posterior mean for MCMC.
- For `NormalizingPlanarFlow`, uses sampling (`flow_samples`) and KDE/hist approximations.
- MCMC flow plots are posterior-averaged with quantile bands.

### plot_random_effect_pit
Signature (key args):
- `plot_random_effect_pit(res; dm=nothing, re_names=nothing, levels=nothing, individuals_idx=nothing, x_covariate=nothing, show_hist=true, show_kde=false, show_qq=true, shared_x_axis=true, shared_y_axis=true, ncols=DEFAULT_PLOT_COLS, style=PlotStyle(), kde_bandwidth=nothing, mcmc_draws=1000, mcmc_warmup=nothing, mcmc_quantiles=[5,95], mcmc_quantiles_alpha=0.8, flow_samples=500, rng=Random.default_rng(), save_path=nothing, kwargs_subplot=NamedTuple(), kwargs_layout=NamedTuple())`

Behavior:
- `x_covariate` is currently not supported and throws `ArgumentError` (kept for API consistency).
- PIT diagnostics of RE values under RE distributions.
- Flow RE PIT uses empirical CDF from samples.
- If multiple plot-type toggles are true, warns and falls back to histogram.

### plot_random_effect_standardized
Signature (key args):
- `plot_random_effect_standardized(res; dm=nothing, re_names=nothing, levels=nothing, individuals_idx=nothing, show_hist=true, show_kde=false, kde_bandwidth=nothing, mcmc_draws=1000, flow_samples=500, ncols=DEFAULT_PLOT_COLS, style=PlotStyle(), save_path=nothing, kwargs_subplot=NamedTuple(), kwargs_layout=NamedTuple())`

Behavior:
- Standardized EBE diagnostics (z-scale), histogram and/or KDE.
- For flow RE distributions, moments are sampling-based.

### plot_random_effect_standardized_scatter
Signature (key args):
- `plot_random_effect_standardized_scatter(res; dm=nothing, re_names=nothing, levels=nothing, individuals_idx=nothing, x_covariate=nothing, mcmc_draws=1000, flow_samples=500, ncols=DEFAULT_PLOT_COLS, style=PlotStyle(), save_path=nothing, kwargs_subplot=NamedTuple(), kwargs_layout=NamedTuple())`

Behavior:
- Scatter of standardized RE values vs level/index or constant covariate.

### plot_random_effects_pdf
Signature (key args):
- `plot_random_effects_pdf(res; dm=nothing, re_names=nothing, levels=nothing, individuals_idx=nothing, shared_x_axis=true, shared_y_axis=true, ncols=DEFAULT_PLOT_COLS, style=PlotStyle(), mcmc_draws=1000, mcmc_warmup=nothing, mcmc_quantiles=[5,95], mcmc_quantiles_alpha=0.8, flow_samples=500, flow_plot=:kde, flow_bins=20, flow_bandwidth=nothing, rng=Random.default_rng(), save_path=nothing, kwargs_subplot=NamedTuple(), kwargs_layout=NamedTuple())`

Behavior:
- Only RE distributions that do not depend on covariates are plotted.
- Shows marginal RE PDFs/PMFs (not per-level EBE markers).
- Flow and MCMC behavior mirrors `plot_random_effect_distributions`.

### plot_random_effects_scatter
Signature (key args):
- `plot_random_effects_scatter(res; dm=nothing, re_names=nothing, levels=nothing, individuals_idx=nothing, x_covariate=nothing, mcmc_draws=1000, ncols=DEFAULT_PLOT_COLS, style=PlotStyle(), save_path=nothing, kwargs_subplot=NamedTuple(), kwargs_layout=NamedTuple())`

Behavior:
- Only RE distributions without covariate dependence.
- Scatter of EBE/posterior mean vs level/index or constant covariate.

### plot_random_effect_pairplot
Signature (key args):
- `plot_random_effect_pairplot(res; dm=nothing, re_names=nothing, levels=nothing, individuals_idx=nothing, ncols=DEFAULT_PLOT_COLS, style=PlotStyle(), kde_bandwidth=nothing, mcmc_draws=1000, rng=Random.default_rng(), save_path=nothing, kwargs_subplot=NamedTuple(), kwargs_layout=NamedTuple())`

Behavior:
- Only RE distributions without covariate dependence.
- Pair plots grouped by random-effect grouping column.
- Diagonal: histogram (+ optional KDE).
- Off-diagonal: EBE/posterior mean scatter.

## Visual Predictive Check

### plot_vpc
Signature (key args):
- `plot_vpc(res::FitResult; dm=nothing, n_simulations=100, n_sim=nothing, percentiles=[5,50,95], show_obs_points=true, show_obs_percentiles=true, n_bins=nothing, seed=12345, observables=nothing, x_axis_feature=nothing, ncols=DEFAULT_PLOT_COLS, serialization=nothing, kwargs_plot=NamedTuple(), save_path=nothing, obs_percentiles_mode=:pooled, bandwidth=nothing, obs_percentiles_method=:quantile, constants_re=NamedTuple(), mcmc_draws=1000, mcmc_warmup=nothing, style=PlotStyle())`

Behavior:
- `n_sim` is supported as alias of `n_simulations`.
- Passing `serialization` is currently unsupported and throws `ArgumentError`.
- Supports continuous and discrete outcomes.
- For MCMC, simulations are driven by posterior draws.
- Observed percentile options include `obs_percentiles_method=:quantile` or `:kernel`.
- Observed percentile mode includes `obs_percentiles_mode=:pooled` or `:per_individual` (per-individual only for quantile mode).

## Notes

- Many plotting functions accept `constants_re`; when called with a `FitResult`, defaults are inherited from fit kwargs if available.
- For non-ODE paths in VPC/residual/observation-distribution diagnostics, `x_axis_feature` must be time or a varying covariate.
- Quantile vectors (for MCMC bands and summaries) are validated to be in `[0, 100]` with length at least 2.
