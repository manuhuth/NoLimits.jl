# NoLimits.jl Plotting Reference

Complete reference for all plotting functions, the publication-quality styling system, and customization options. Built on `Plots.jl`.

---

## Table of Contents

1. [Styling System (PlotStyle & PlotDefaults)](#1-styling-system)
2. [Model Fits & Data](#2-model-fits--data)
3. [Random Effects Visualization](#3-random-effects-visualization)
4. [Residual Diagnostics](#4-residual-diagnostics)
5. [Visual Predictive Check (VPC)](#5-visual-predictive-check-vpc)
6. [Observation Distribution Analysis](#6-observation-distribution-analysis)
7. [Multistart Optimization Visualization](#7-multistart-optimization-visualization)
8. [Bootstrap Distribution Visualization](#8-bootstrap-distribution-visualization)
9. [SAEM Convergence Trajectories](#9-saem-convergence-trajectories)
10. [Publication-Ready Plotting Guide](#10-publication-ready-plotting-guide)

---

## 1. Styling System

### 1.1 PlotStyle Struct

All plotting functions accept a `style::PlotStyle` keyword argument. `PlotStyle` is a `@kwdef` struct with these fields and defaults:

```julia
Base.@kwdef struct PlotStyle
    # Colors
    color_primary::String    = "#0173B2"    # Blue  - main data/lines
    color_secondary::String  = "#029E73"    # Green - fitted values/predictions
    color_accent::String     = "#DE8F05"    # Orange - highlights/density
    color_dark::String       = "#2C3E50"    # Dark blue-gray - text/references
    color_density::String    = "#DE8F05"    # Orange (alias for accent)
    color_reference::String  = "#2C3E50"    # Dark gray (alias for dark)

    # Fonts
    font_family::String          = "Helvetica"
    font_size_title::Int         = 11
    font_size_label::Int         = 10
    font_size_tick::Int          = 9
    font_size_legend::Int        = 8
    font_size_annotation::Int    = 7

    # Lines and Markers
    line_width_primary::Float64    = 2.0
    line_width_secondary::Float64  = 1.5
    marker_size::Int               = 5
    marker_size_small::Int         = 3
    marker_alpha::Float64          = 0.7
    marker_stroke_width::Float64   = 0.5

    # Layout
    base_subplot_width::Int  = 350    # Pixels per subplot
    base_subplot_height::Int = 280    # Pixels per subplot
end
```

Create custom styles by overriding any field:

```julia
custom_style = PlotStyle(
    color_primary = "#E63946",
    font_size_title = 14,
    line_width_primary = 2.5,
    marker_size = 6
)
plot_fits(fitted; style=custom_style)
```

### 1.2 Global Color Constants

| Constant | Hex | Use |
|----------|-----|-----|
| `COLOR_PRIMARY` | `#0173B2` | Main data/lines |
| `COLOR_SECONDARY` | `#029E73` | Fitted values/predictions |
| `COLOR_ACCENT` | `#DE8F05` | Highlights/density |
| `COLOR_DARK` | `#2C3E50` | Reference lines/text |
| `COLOR_LIGHT_GRAY` | `#7F8C8D` | Background elements |
| `COLOR_ERROR` | `#D55E00` | Error/warning |
| `COLOR_DATA` | `#0173B2` | Observed data points |
| `COLOR_PREDICTION` | `#2C3E50` | Model predictions |
| `COLOR_DENSITY` | `#DE8F05` | Density/heatmap |
| `COLOR_CI` | `#56B4E9` | Confidence intervals |
| `COLOR_REFERENCE` | `#2C3E50` | Reference lines |

### 1.3 Layout Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `DEFAULT_PLOT_COLS` | `3` | Default grid columns |
| `BASE_SUBPLOT_WIDTH` | `350` px | Base subplot width |
| `BASE_SUBPLOT_HEIGHT` | `280` px | Base subplot height |
| `MIN_FIGURE_WIDTH` | `400` px | Minimum total figure width |
| `MAX_FIGURE_WIDTH` | `1800` px | Maximum total figure width |
| `MIN_FIGURE_HEIGHT` | `300` px | Minimum total figure height |
| `MAX_FIGURE_HEIGHT` | `2400` px | Maximum total figure height |
| `PLOT_MARGIN` | `3` mm | Subplot margin |
| `DEFAULT_DPI` | `300` | Raster output resolution |

### 1.4 Auto-Sizing Logic

Figure size is computed by `calculate_plot_size(nplots, ncols)`:

1. `ncols = min(ncols, nplots)`, `nrows = ceil(nplots / ncols)`
2. A scale factor is applied based on total subplot count:

| Subplots | Scale Factor |
|----------|-------------|
| 1-4 | 1.00 |
| 5-9 | 0.95 |
| 10-16 | 0.85 |
| 17-25 | 0.75 |
| 26+ | 0.65 |

3. `width = round(ncols * base_width * scale_factor)`, same for height
4. Clamped to `[MIN, MAX]` bounds

### 1.5 Default Subplot Styling

Every subplot gets these defaults via `default_plot_kwargs()`:

```julia
(
    framestyle       = :box,          # Box frame around plot
    grid             = :y,            # Horizontal grid lines only
    gridalpha        = 0.3,           # Semi-transparent grid
    gridlinewidth    = 0.5,
    foreground_color_grid = :gray,
    legend           = :best,         # Automatic positioning
    titlefontsize    = 11,
    guidefontsize    = 10,            # Axis labels
    tickfontsize     = 9,
    legendfontsize   = 8,
    margin           = 3mm,
    fontfamily       = "Helvetica",
)
```

### 1.6 Shared Axes

`_apply_shared_axes!(plots_buffer, shared_x, shared_y; padding=0.05)` enforces uniform axis limits across all subplots:

- Collects global min/max from all series data across all subplots
- Adds 5% padding (configurable) to prevent data touching axes
- Edge case: if all values are identical, uses `+/-1` range
- Applied in-place to the plot vector before combining into layout

### 1.7 Saving Plots

All plotting functions accept `save_path::Union{Nothing, String} = nothing`. Format is determined by file extension:

| Extension | Format |
|-----------|--------|
| `.png` | Raster (300 DPI default) |
| `.pdf` | Vector |
| `.svg` | Scalable vector |
| `.eps` | Encapsulated PostScript |

```julia
plot_fits(fitted; save_path="figures/fits.pdf")
```

### 1.8 Common Kwargs Pattern

Most plotting functions accept these keyword arguments:

| Kwarg | Type | Default | Purpose |
|-------|------|---------|---------|
| `ncols` | `Int` | `3` | Grid columns |
| `shared_x_axis` | `Bool` | `true` | Uniform x-axis limits across subplots |
| `shared_y_axis` | `Bool` | `true` | Uniform y-axis limits across subplots |
| `style` | `PlotStyle` | `PlotStyle()` | Full style customization |
| `kwargs_subplot` | `NamedTuple` | `NamedTuple()` | Override individual subplot options |
| `kwargs_layout` | `NamedTuple` | `NamedTuple()` | Override layout (size, dpi, etc.) |
| `save_path` | `Union{Nothing,String}` | `nothing` | File path to save |

User kwargs always take precedence over defaults (via `merge`).

### 1.9 Helper Functions

| Function | Purpose |
|----------|---------|
| `create_styled_plot(; title, xlabel, ylabel, kwargs...)` | New plot with default styling |
| `create_styled_scatter!(p, x, y; label, color, kwargs...)` | Add styled scatter series |
| `create_styled_line!(p, x, y; label, color, kwargs...)` | Add styled line series |
| `add_reference_line!(p, value; orientation=:horizontal, kwargs...)` | Dashed reference line |
| `add_annotation!(p, x, y, text; fontsize=7, halign=:right)` | Text annotation |
| `combine_plots(plots::Vector; ncols=3, kwargs...)` | Combine subplots into grid |
| `apply_default_style!(p)` | Apply defaults to existing plot |

### 1.10 Style Presets

| Preset | Key Values |
|--------|-----------|
| `individual_fit_style()` | `linewidth=2.0, markersize=5, markeralpha=0.7` |
| `density_plot_style()` | `linewidth=2.0, fillalpha=0.3` |
| `residual_plot_style()` | `markersize=3, markeralpha=0.7` |
| `qq_plot_style()` | `markersize=3, markeralpha=0.6` |
| `vpc_plot_style()` | `linewidth=2.0, fillalpha=0.3, ribbon_alpha=0.3` |

---

## 2. Model Fits & Data

### 2.1 `plot_fits` -- Model Predictions vs Observations

```julia
plot_fits(fitted_model;
    plot_density = false,                              # Show prediction uncertainty as heatmap
    plot_func = mean,                                  # Statistic: mean, median, mode -> applied to the outcome dist to obtain point fit
    plot_data_points = true,                           # Show observed data points
    observable = nothing,                              # Which observable to plot (Symbol); if nothing, uses first
    individuals_idx = nothing,                         # Which individuals to plot (indices); if nothing, all
    x_axis_feature::Union{Nothing, Symbol} = nothing,  # feature as x-axis
    shared_x_axis::Bool = true,
    shared_y_axis::Bool = true,
    ncols::Int = 3,
    style::PlotStyle = PlotStyle(),
    kwargs_subplot = NamedTuple(),
    kwargs_layout = NamedTuple(),
    save_path::Union{Nothing, String} = nothing
) → Plot
```

**What it produces:** Grid of individual-level subplots. Each subplot shows:
- Observed data points (scatter, primary color) if `plot_data_points=true`
- Model prediction line (secondary color)
- Optional density heatmap showing prediction uncertainty if `plot_density=true`
- Title: individual ID

**Behavior:**
- If `observable=nothing` and multiple observables exist, warns and uses the first
- For discrete models, dispatches to `plot_individual` with `x_axis_feature` support
- For ODE models, plots over time axis

**Works on:** fitted model

### 2.2 `plot_data` -- Raw Data Visualization

```julia
plot_data(data_model;
    x_axis_feature::Union{Symbol, Nothing} = nothing,  # feature for x-axis
    individuals_idx = nothing,                          # Which individuals; if nothing, all
    shared_x_axis::Bool = true,
    shared_y_axis::Bool = true,
    ncols::Int = 3,
    style::PlotStyle = PlotStyle(),
    kwargs_subplot = NamedTuple(),
    kwargs_layout = NamedTuple(),
    save_path::Union{Nothing, String} = nothing
) → Plot
```

**What it produces:** Grid of individual-level scatter plots showing raw observed data.
- ODE models: x-axis = x vairable or index, y-axis = observable values
- Title: individual ID

**Works on:** data models

---

## 3. Random Effects Visualization

### 3.1 `plot_re_distribution_sampling` -- Random Effects Density Plots

```julia
plot_re_distribution_sampling(fitted_model;
    nsamples::Int = 10_000,          # Monte Carlo samples for KDE
    npoints::Int = 200,              # Points for KDE evaluation
    fitted::Bool = true,             # true=fitted params, false=initial values
    ncols::Int = 3,
    style::PlotStyle = PlotStyle(),
    kwargs_subplot = NamedTuple(),
    kwargs_layout = NamedTuple(),
    save_path::Union{Nothing, String} = nothing
) → Plot
```

**What it produces:**
- **Univariate distributions:** One KDE density curve per random effect, with vertical dashed line at mean, annotation showing distributional parameters
- **Multivariate distributions:** One marginal KDE per dimension (e.g., `eta[1]`, `eta[2]`, `eta[3]`), each with mean line and distribtuional aprameetr annotation

**Annotation placement:** Bottom-right corner (95% x-range, 15% y-range)

### 3.2 `plot_re_bivariate_kdes` -- Random Effects Correlations

```julia
plot_re_bivariate_kdes(fitted_model;
    nsamples::Int = 100_000,         # Monte Carlo samples for 2D KDE
    fitted::Bool = true,
    ncols::Int = 3,
    style::PlotStyle = PlotStyle(),
    kwargs_subplot = NamedTuple(),
    kwargs_layout = NamedTuple(),
    save_path::Union{Nothing, String} = nothing
) → Plot
```

**What it produces:**
- Bivariate contour plots for all pairs `(i, j)` where `i < j`
- For `n` random effects dimensions: `n*(n-1)/2` subplots
- Color scale: `:viridis`
- Univariate distributions: shows 1D KDE as fallback
- Useful for verifying correlation structure and diagnosing identifiability

### 3.3 `plot_qq_random_effects` -- Random Effects QQ Plots

```julia
plot_qq_random_effects(fitted_model;
    ncols::Int = 3,
    kwargs_plot = NamedTuple(),
    save_path::Union{Nothing, String} = nothing
) → Plot
```

**What it produces:**
- QQ plot per random effect component comparing empirical Bayes estimates against theoretical quantiles
- Reference line (y=x) in dark gray
- Annotation: empirical and theoretical `mu`, `sigma` (or medians for distributions without moments)
- Multivariate: extracts marginal distribution per component (e.g., `Normal(mu_i, sqrt(Sigma_ii))` for `MvNormal`)

**Note:** Uses conditional modes (empirical Bayes estimates), not posterior samples.

---

## 4. Residual Diagnostics

### 4.1 `plot_residuals` -- Residual Plots

```julia
plot_residuals(fitted_model;
    individual_plots::Bool = false,      # true=one plot per individual, false=combined
    display_type::Symbol = :scatter,     # :scatter, :histogram, :kde, :hist_kde
    statistic = mean,                    # mean, median, mode
    observables = nothing,               # Which observables; nothing=all
    ncols::Int = 3,
    kwargs_plot = NamedTuple(),
    save_path::Union{Nothing, String} = nothing
) → Plot
```

**Display types:**
- `:scatter` -- Residuals vs observations with horizontal reference lines at 0 and mean
- `:histogram` -- Histogram with `mu` and `sigma` annotation
- `:kde` -- Kernel density estimate curve with `mu` and `sigma` annotation
- `:hist_kde` -- Histogram with overlaid KDE curve

**Layout:**
- `individual_plots=false`: One subplot per observable (all individuals combined)
- `individual_plots=true`: One subplot per individual, all observables combined

### 4.2 `plot_qq_pearson_residuals` -- QQ Plots for Pearson Residuals

```julia
plot_qq_pearson_residuals(fitted_model;
    individual_plots::Bool = false,
    reference_dist::UnivariateDistribution = Normal(0, 1),
    statistic = mean,
    observables = nothing,
    ncols::Int = 3,
    kwargs_plot = NamedTuple(),
    save_path::Union{Nothing, String} = nothing
) → Plot
```

**What it produces:**
- QQ plot comparing Pearson residuals to theoretical quantiles from `reference_dist`
- Reference line (y=x) in dark gray
- Annotation: `mu`, `sigma` of observed Pearson residuals

**Interpretation:**
- Points on reference line: residuals follow reference distribution
- Heavy tails: points above line at both ends
- Light tails: points below line at both ends
- Skewness: asymmetric deviations

**Works on:** `FittedODEModel`, `FittedDiscreteModel`, `FittedODEModelSAEM`, `FittedDiscreteModelSAEM`

---

## 5. Visual Predictive Check (VPC)

### 5.1 `plot_vpc` -- ODE Models

```julia
plot_vpc(fitted_model;
    n_simulations::Int = 100,
    percentiles::Vector{<:Real} = [5, 50, 95],
    show_obs_points::Bool = true,
    show_obs_percentiles::Bool = true,
    n_bins::Union{Nothing, Int} = nothing,
    seed::Int = 12345,
    observables = nothing,
    x_axis_feature::Union{Nothing, Symbol} = nothing,    #default is the t variable used for fiting
    ncols::Int = 3,
    serialization = EnsembleThreads(),
    kwargs_plot = NamedTuple(),
    save_path::Union{Nothing, String} = nothing
) → Plot
```


**What it produces:**
- **Shaded prediction intervals**: Ribbon from lowest to highest percentile (e.g., 5th-95th), fill alpha 0.3
- **Median prediction**: Dashed line at 50th percentile of simulated data
- **Observed percentiles**: Smooth lines using Gaussian kernel smoothing (orange; solid = median, dotted = others)
- **Individual observations**: Gray scatter points (alpha 0.3) if `show_obs_points=true`
- One panel per observable

**Percentile validation:**
- Must be between 0 and 100
- At least 2 percentiles required
- Sorted ascending

**Binning:** `n_bins=nothing` uses actual time/feature values. Setting `n_bins` to an integer bins the x-axis for computational efficiency.

**Simulation process:**
1. For each of `n_simulations` replicates, sample random effects from fitted distribution
2. Solve ODE / evaluate discrete model
3. Sample observations from predicted distributions
4. Compute percentiles across simulations at each time/feature point

**Kernel smoothing:** Gaussian kernel with bandwidth = `(x_max - x_min) / 10` for observed percentile curves.

---

## 6. Observation Distribution Analysis

### 6.1 `plot_observation_distribution` -- Single Observation

```julia
plot_observation_distribution(
    fitted_model,
    individual_idx::Int,
    obs_idx::Int;
    observable::Union{Nothing, Symbol} = nothing,
    n_points::Int = 500,
    coverage::Float64 = 0.995,
    style::PlotStyle = PlotStyle(),
    kwargs_subplot::NamedTuple = NamedTuple(),
    save_path::Union{Nothing, String} = nothing
) → Plot
```

**What it produces (by distribution type):**

- **Continuous** (Normal, LogNormal, Gamma, etc.): Smooth PDF curve with shaded area, vertical dashed line at observed value, parameter annotation (e.g., `mu`, `sigma`)
- **Discrete** (Poisson, NegativeBinomial, etc.): Stem plot (lollipops) at integer support values, vertical dashed line at observed value, parameter annotation (e.g., `lambda`)
- **Bernoulli**: Bar plot for P(Y=0) and P(Y=1) with probability annotations above/inside bars, diamond marker at observed value, y-limits `[0, max(p0, p1) * 1.2]`

**X-axis range:** Determined by `coverage` parameter (default 99.5% probability mass) via quantiles.

**Parameter annotation:** Dispatches on distribution type via `_get_distribution_params_string(dist)`, supporting 20+ distribution types: Normal (`mu`, `sigma`), Poisson (`lambda`), Bernoulli (`p`), Binomial (`n`, `p`), Gamma (`alpha`, `theta`), LogNormal (`mu`, `sigma`), etc.

**Works on:** All model types (ODE Laplace/FOCEI, discrete Laplace/FOCEI, ODE SAEM, discrete SAEM).

### 6.2 `plot_observation_distributions` -- Multiple Observations Grid

```julia
plot_observation_distributions(
    fitted_model,
    individual_idx::Int;
    obs_idx::Union{Nothing, AbstractVector{Int}} = nothing,
    observable::Union{Nothing, Symbol} = nothing,
    ncols::Int = 3,
    n_points::Int = 500,
    coverage::Float64 = 0.995,
    style::PlotStyle = PlotStyle(),
    kwargs_subplot::NamedTuple = NamedTuple(),
    kwargs_layout::NamedTuple = NamedTuple(),
    save_path::Union{Nothing, String} = nothing
) → Plot
```

**What it produces:** Grid of `plot_observation_distribution` subplots.
- `obs_idx=nothing`: plots all observations for the individual
- `obs_idx=[1, 3, 5]`: plots only those observation indices
- Automatically skips missing observations

---

## 7. Multistart Optimization Visualization

### 7.1 `plot_waterfall_multistart` -- Objective Value Waterfall

```julia
plot_waterfall_multistart(fitted_multistart_model;
    plt = nothing,                                 # Overlay on existing plot
    title::AbstractString = "Objective values",
    n_show::Union{Nothing, Int} = nothing,         # Show only top N fits
    label::Union{Nothing, AbstractString} = nothing,
    kwargs_plot = NamedTuple(),
    save_path::Union{Nothing, String} = nothing
) → Plot
```

**What it produces:** Line plot with circle markers, x = fit index (sorted best-to-worst), y = objective value. Useful for assessing multistart coverage.

**Overlay usage:**
```julia
p = plot_waterfall_multistart(fitted1; label="Model 1")
plot_waterfall_multistart(fitted2; plt=p, label="Model 2")
```

### 7.2 `plot_parameter_summaries` -- Parameter Variation Across Fits

```julia
plot_parameter_summaries(fitted_multistart_model;
    title::AbstractString = "Parameters across fits",
    n_show::Union{Nothing, Int} = nothing,
    mode::Symbol = :all,                  # :all, :mean, :minmax
    exclude_params = (),                  # Parameters to exclude
    label::Union{Nothing, AbstractString} = nothing,
    kwargs_plot = NamedTuple(),
    save_path::Union{Nothing, String} = nothing
) → Plot
```

**Modes:**
- `:all` -- Scatter plot showing all individual parameter values across fits
- `:mean` -- Points with symmetric `+/-1 sigma` error bars
- `:minmax` -- Points with asymmetric min/max error bars

**Axes:** y-axis = parameter names (leaf parameters flattened), x-axis = parameter values. Fits sorted by objective value.

### 7.3 `plot_violin_marginal_random_effects` -- Violin Plots

```julia
plot_violin_marginal_random_effects(fitted_models;
    nsamples::Int = 1_000,
    fitted::Bool = true,
    n_fits::Union{Nothing, Int} = nothing,
    ncols::Int = 3,
    model_labels::Union{Nothing, AbstractVector{<:AbstractString}} = nothing,
    align_parameters_by::Symbol = :name,     # :name or :position
    param_labels::Union{Nothing, AbstractVector{<:AbstractString}} = nothing,
    save_path::Union{Nothing, String} = nothing
) → (plot=Plot, index_map=...)
```

**What it produces:** One panel per random effect parameter, each showing violin plots for each fit/model. Fits grouped by model with separator lines.

**Arguments:**
- `fitted_models`: single model or vector of models (for multi-model comparison)
- `align_parameters_by`: `:name` matches parameters by name across models; `:position` matches by position
- `model_labels`: custom labels (default: `"model 1"`, `"model 2"`, ...)


---

The defaults (Helvetica font, box frame, horizontal grid, 300 DPI, consistent colors) are designed for direct journal submission.

### 10.2 Custom Style for Journal Requirements

```julia
# Example: journal requires Times New Roman, larger fonts
journal_style = PlotStyle(
    font_family = "Times New Roman",
    font_size_title = 14,
    font_size_label = 12,
    font_size_tick = 10,
    font_size_legend = 9,
    font_size_annotation = 8,
    line_width_primary = 2.5,
    marker_size = 6
)

plot_fits(fitted; style=journal_style, save_path="fig1.pdf")
plot_residuals(fitted; display_type=:hist_kde, save_path="fig2.pdf")
plot_vpc(fitted; n_simulations=500, save_path="fig3.pdf")
```

### 10.3 High-Resolution Export

```julia
# Override DPI for high-resolution raster output
plot_fits(fitted;
    kwargs_layout = (dpi=600,),
    save_path="figures/fits_hires.png"
)

# Vector formats (PDF/SVG) are resolution-independent
plot_fits(fitted; save_path="figures/fits.pdf")
```

### 10.4 Custom Figure Dimensions

```julia
# Override auto-sizing
plot_fits(fitted;
    ncols = 2,
    kwargs_layout = (size=(1600, 1000),)
)
```

### 10.5 Disabling Shared Axes

```julia
# Each subplot gets its own axis range
plot_fits(fitted;
    shared_x_axis = false,
    shared_y_axis = false
)
```

### 10.6 Subplot-Level Customization

```julia
# Override subplot options
plot_fits(fitted;
    kwargs_subplot = (
        grid = false,              # No gridlines
        legend = :bottomright,     # Force legend position
        xscale = :log10,           # Log scale x-axis
    )
)
```

### 10.7 Comprehensive Diagnostic Panel

```julia
# Individual fits
p1 = plot_fits(fitted; save_path="panel_a.pdf")

# Residual diagnostics -- 4 display types
p2 = plot_residuals(fitted; display_type=:scatter, save_path="panel_b.pdf")
p3 = plot_residuals(fitted; display_type=:hist_kde, save_path="panel_c.pdf")

# QQ plots
p4 = plot_qq_pearson_residuals(fitted; save_path="panel_d.pdf")
p5 = plot_qq_random_effects(fitted; save_path="panel_e.pdf")

# VPC
p6 = plot_vpc(fitted; n_simulations=500, save_path="panel_f.pdf")

# Random effects
p7 = plot_re_distribution_sampling(fitted; save_path="panel_g.pdf")
p8 = plot_re_bivariate_kdes(fitted; save_path="panel_h.pdf")
```


### 10.9 Multistart Analysis

```julia
# Waterfall plot of objective values
plot_waterfall_multistart(fitted; save_path="waterfall.pdf")

# Parameter variation across fits
plot_parameter_summaries(fitted; mode=:mean, save_path="param_summary.pdf")

# Violin plots comparing models
plot_violin_marginal_random_effects([fitted1, fitted2];
    model_labels=["FOCEI", "Laplace"],
    save_path="re_violins.pdf"
)
```

### 10.10 Color Scheme Summary

| Role | Default Color | Hex |
|------|--------------|-----|
| Observed data / primary lines | Blue | `#0173B2` |
| Model predictions / fitted lines | Green | `#029E73` |
| Density / highlights / observed percentiles | Orange | `#DE8F05` |
| Reference lines / text / dark elements | Dark blue-gray | `#2C3E50` |
| Light background | Medium gray | `#7F8C8D` |
| Error | Red | `#D55E00` |
| Confidence intervals | Light blue | `#56B4E9` |

### 10.11 Quick Reference Table -- All Plotting Functions

| Function | Input Type | Subplots | Key Feature |
|----------|-----------|----------|-------------|
| `plot_fits` | `FittedModel` | Per individual | Data + predictions overlay, optional density heatmap |
| `plot_data` | data model | Per individual | Raw observed data only |
| `plot_re_distribution_sampling` | `FittedModel` | Per RE component | Marginal KDE density with mu/sigma |
| `plot_re_bivariate_kdes` | `FittedModel` | Per RE pair | Bivariate contour (viridis) |
| `plot_qq_random_effects` | `FittedModel` | Per RE component | Empirical Bayes vs theoretical QQ |
| `plot_residuals` | `FittedModel` | Per observable | 4 display types: scatter/histogram/kde/hist_kde |
| `plot_qq_pearson_residuals` | `FittedModel` | Per observable | Pearson residuals vs reference dist QQ |
| `plot_vpc`  | `FittedODEModel` | Per observable | Prediction intervals + observed percentiles |
| `plot_observation_distribution` | `FittedModel` | Single | Full PDF/PMF at one observation |
| `plot_observation_distributions` | `FittedModel` | Per observation | Grid of PDF/PMF for multiple observations |
| `plot_waterfall_multistart` | `FittedModel` | Single | Sorted objective values (line+markers) |
| `plot_parameter_summaries` | `FittedModel` | Single | Parameter values across fits |
| `plot_violin_marginal_random_effects` | `Vector{FittedModel}` | Per RE param | Violin plots across fits/models |
### 10.12 Discrete Distribution Detection Helpers

These internal utilities determine rendering strategy for observation distributions:

| Function | Returns | Purpose |
|----------|---------|---------|
| `is_discrete_distribution(d)` | `Bool` | `true` for `DiscreteDistribution` subtypes |
| `is_bernoulli_distribution(d)` | `Bool` | `true` for `Bernoulli` only |
| `is_bernoulli_like(values)` | `Bool` | `true` if all values are 0 or 1 |
| `get_discrete_support(d, coverage=0.99)` | `Vector{Int}` | Integer support covering given probability mass |
| `get_bernoulli_ylims(values)` | `(-0.05, 1.05)` | Fixed y-limits for binary data |
