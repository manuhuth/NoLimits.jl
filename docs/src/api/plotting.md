# Plotting and Diagnostics API

Every `plot_*` function. All of them require a Makie backend to be loaded.

!!! note "Some entries need an optional dependency"
    Parts of this API are implemented in package extensions and become available only once the
    corresponding package is loaded. Calling one without its package raises an error naming what
    to install. See [Optional Dependencies](../installation.md#Optional-Dependencies).

## Core Plots

```@docs
PlotStyle
PlotCache
build_plot_cache
plot_data
plot_fits
plot_fits_comparison
```

## Visual Predictive Checks

```@docs
plot_vpc
```

## Residual Diagnostics

```@docs
get_residuals
plot_residuals
plot_residual_distribution
plot_residual_qq
plot_residual_pit
plot_residual_acf
```

## Random-Effects Diagnostics

```@docs
plot_random_effects_pdf
plot_random_effects_scatter
plot_random_effect_pairplot
plot_random_effect_distributions
plot_random_effect_pit
plot_random_effect_standardized
plot_random_effect_standardized_scatter
```

## Observation Distributions

```@docs
plot_observation_distributions
plot_hidden_states
plot_emission_distributions
```

## Uncertainty Quantification Plots

```@docs
plot_uq_distributions
```

## Multistart Plots

```@docs
plot_multistart_waterfall
plot_multistart_fixed_effect_variability
```

## Goodness-of-Fit and Diagnostic Plots

```@docs
plot_dv_pred
plot_dv_ipred
plot_wres_pred
plot_shrinkage
plot_observed_profiles
plot_em_trajectories
```

## Where to go next

- [Plotting guide](../plotting/index.md) - the prose behind these names.
- [Distributions & Utilities](distributions.md) - the next API page.
- [API overview](../api.md) - all seven reference pages.
