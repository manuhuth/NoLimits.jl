# Uncertainty, Simulation, and Identifiability API

Standard errors, profiles, posterior summaries, simulation, and identifiability analysis.

!!! note "Some entries need an optional dependency"
    Parts of this API are implemented in package extensions and become available only once the
    corresponding package is loaded. Calling one without its package raises an error naming what
    to install. See [Optional Dependencies](../installation.md#Optional-Dependencies).

## Uncertainty Quantification

```@docs
compute_uq
UQResult
UQIntervals
get_uq_backend
get_uq_source_method
get_uq_parameter_names
get_uq_estimates
get_uq_intervals
get_uq_vcov
get_uq_draws
get_uq_diagnostics
```

## Data Simulation

```@docs
simulate_data
simulate_data_model
```

## Identifiability Analysis

```@docs
identifiability_report
IdentifiabilityReport
NullDirection
RandomEffectInformation
```

## Where to go next

- [Uncertainty Quantification guide](../uncertainty-quantification/index.md) - the prose behind these names.
- [Plotting & Diagnostics](plotting.md) - the next API page.
- [API overview](../api.md) - all seven reference pages.
