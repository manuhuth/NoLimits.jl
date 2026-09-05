# Estimation API

Method objects, the `fit_model` entry point, result accessors, and serialization.

!!! note "Some entries need an optional dependency"
    Parts of this API are implemented in package extensions and become available only once the
    corresponding package is loaded. Calling one without its package raises an error naming what
    to install. See [Optional Dependencies](../installation.md#Optional-Dependencies).

## Base Types

```@docs
FittingMethod
MethodResult
FitResult
FitSummary
FitDiagnostics
FitParameters
```

## Fitting Interface

```@docs
fit_model
```

## Methods

```@docs
MLE
MAP
Laplace
FOCEI
GHQuadrature
MCEM
MCEM_MCMC
MCEM_IS
SAEM
SaemixMH
AdaptiveNoLimitsMH
MCMC
Pooled
PooledMap
Multistart
```

## Result Types

```@docs
StandardOptimizationResult
MCMCResult
MultistartFitResult
```

## Fit Result Accessors

```@docs
get_params
get_objective
get_converged
get_diagnostics
get_summary
get_method
get_result
get_data_model
get_iterations
get_raw
get_notes
get_chain
get_observed
get_sampler
get_random_effects
get_loglikelihood
```

## Multistart Accessors

```@docs
get_multistart_results
get_multistart_errors
get_multistart_starts
get_multistart_failed_results
get_multistart_failed_starts
get_multistart_best_index
get_multistart_best
```

## Cross-Validation

See the [Cross-Validation](../estimation/cv.md) page for the full API.

## Fit Summaries

```@docs
FitResultSummary
UQResultSummary
```

## Utilities

```@docs
default_bounds_from_start
```

## Variational Inference

```@docs
VI
VIResult
get_variational_posterior
get_vi_state
get_vi_trace
```

## Estimation and Random-Effects Helpers

```@docs
sample_random_effects
sample_posterior
reestimate_ebes
get_marginal_likelihood
get_loglikelihood_quadrature
compute_shrinkage
compare_parameters
MCIntegrator
get_laplace_random_effects
```

## Serialization

```@docs
save_fit
load_fit
```

## Where to go next

- [Estimation guide](../estimation/index.md) - the prose behind these names.
- [Uncertainty & Simulation](uncertainty.md) - the next API page.
- [API overview](../api.md) - all seven reference pages.
