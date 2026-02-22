# API Reference

This page documents the complete public API of NoLimits.jl. Each entry is rendered from
the docstring attached to the corresponding function, type, or macro.

## Model Building

### Macros

```@docs
@Model
@helpers
@fixedEffects
@covariates
@randomEffects
@preDifferentialEquation
@DifferentialEquation
@initialDE
@formulas
```

### Parameter Types

```@docs
RealNumber
RealVector
RealPSDMatrix
RealDiagonalMatrix
ProbabilityVector
DiscreteTransitionMatrix
ContinuousTransitionMatrix
NNParameters
NPFParameter
SoftTreeParameters
SplineParameters
Priorless
```

### Covariate Types

```@docs
Covariate
CovariateVector
ConstantCovariate
ConstantCovariateVector
DynamicCovariate
DynamicCovariateVector
```

### Random Effects

```@docs
RandomEffect
```

### Model Struct and Solver Configuration

```@docs
Model
ODESolverConfig
set_solver_config
get_model_funs
get_helper_funs
get_solver_config
```

### Model Component Structs

These structs hold the parsed, compiled form of each model block. They are constructed
automatically by the block macros and stored inside `Model`.

```@docs
FixedEffects
Covariates
finalize_covariates
RandomEffects
PreDifferentialEquation
DifferentialEquation
InitialDE
get_initialde_builder
Formulas
get_formulas_builders
```

## Data Binding

### DataModel

```@docs
DataModel
```

### DataModel Accessors

```@docs
get_individuals
get_individual
get_batches
get_batch_ids
get_primary_id
get_df
get_model
get_row_groups
get_re_group_info
get_re_indices
```

### Summaries

```@docs
ModelSummary
DataModelSummary
DescriptiveStats
summarize
```

## Estimation

### Base Types

```@docs
FittingMethod
MethodResult
FitResult
FitSummary
FitDiagnostics
FitParameters
```

### Fitting Interface

```@docs
fit_model
```

### Methods

```@docs
MLE
MAP
Laplace
LaplaceMAP
MCEM
SAEM
MCMC
Multistart
```

### Result Types

```@docs
MLEResult
MAPResult
LaplaceResult
LaplaceMAPResult
MCEMResult
SAEMResult
MCMCResult
MultistartFitResult
```

### Fit Result Accessors

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

### Multistart Accessors

```@docs
get_multistart_results
get_multistart_errors
get_multistart_starts
get_multistart_failed_results
get_multistart_failed_starts
get_multistart_best_index
get_multistart_best
```

### Fit Summaries

```@docs
FitResultSummary
UQResultSummary
```

### Utilities

```@docs
default_bounds_from_start
```

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

## Plotting and Diagnostics

### Core Plots

```@docs
PlotStyle
PlotCache
build_plot_cache
plot_data
plot_fits
plot_fits_comparison
```

### Visual Predictive Checks

```@docs
plot_vpc
```

### Residual Diagnostics

```@docs
get_residuals
plot_residuals
plot_residual_distribution
plot_residual_qq
plot_residual_pit
plot_residual_acf
```

### Random-Effects Diagnostics

```@docs
plot_random_effects_pdf
plot_random_effects_scatter
plot_random_effect_pairplot
plot_random_effect_distributions
plot_random_effect_pit
plot_random_effect_standardized
plot_random_effect_standardized_scatter
```

### Observation Distributions

```@docs
plot_observation_distributions
```

### Uncertainty Quantification Plots

```@docs
plot_uq_distributions
```

### Multistart Plots

```@docs
plot_multistart_waterfall
plot_multistart_fixed_effect_variability
```

## Distributions

### Hidden Markov Models

```@docs
DiscreteTimeDiscreteStatesHMM
probabilities_hidden_states
posterior_hidden_states
ContinuousTimeDiscreteStatesHMM
```

### Normalizing Flows

```@docs
AbstractNormalizingFlow
NormalizingPlanarFlow
```

## Utilities

### Soft Decision Trees

```@docs
SoftTree
SoftTreeParams
init_params
destructure_params
```

### B-Splines

```@docs
bspline_basis
bspline_eval
```
