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
RealLiePSDMatrix
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
ClosedFormPlan
set_solver_config
get_model_funs
get_helper_funs
get_solver_config
get_source
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
get_closed_form_plan
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
FOCEI
GHQuadrature
MCEM
MCEM_MCMC
MCEM_IS
SAEM
MCMC
Pooled
PooledMap
Multistart
```

### Result Types

```@docs
StandardOptimizationResult
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

### Cross-Validation

See the [Cross-Validation](estimation/cv.md) page for the full API.

### Fit Summaries

```@docs
FitResultSummary
UQResultSummary
```

### Utilities

```@docs
default_bounds_from_start
```

## Method-Developer API

Primitives for building new estimation methods on top of NoLimits.jl. See the
[Method-Developer API](method-developer-api.md) page for the contracts and worked examples.

### Contracts and transforms

```@docs
symmetrize_psd_parameters
logabsdetjac
```

### Random-effect batching

```@docs
build_re_batch_infos
REBatchInfo
get_batch_individuals
get_batch_re_info
get_batch_re_dim
build_eta_individual
eta_from_modes
build_likelihood_cache
```

### Density primitives

```@docs
solve_individual
obs_distributions
hmm_filter_step!
conditional_loglikelihood
complete_data_loglikelihood
re_logprior
complete_data_loglikelihood_gradient
complete_data_loglikelihood_hessian
```

### Posterior, empirical Bayes, and sampling

```@docs
empirical_bayes
empirical_bayes_covariance
laplace_marginal
ghq_marginal
sample_random_effect_draws
RandomEffectPosteriorSample
EBEOptions
```

### Curvature seam

```@docs
AbstractCurvature
ExactHessianCurvature
FisherInformationCurvature
inner_curvature
CurvatureWorkspace
```

### Fitting drivers and objective factory

```@docs
fit_method
fit_fixed_effects
fit_laplace_family
free_parameter_layout
NLFreeLayout
resolve_fitted_parameters
build_fit_result
uq_family
```

### Fit context (convenience layer)

```@docs
FitContext
build_fit_context
get_batch_infos
initial_parameters
optimize_parameters
```

### Sparse-grid quadrature

```@docs
GHQuadratureNodes
build_sparse_grid
get_sparse_grid
build_tensor_product_grid
get_anisotropic_grid
n_ghq_points
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
plot_hidden_states
plot_emission_distributions
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

### Markov Models

NoLimits supports both hidden-state and observed-state Markov outcome models, in discrete
and continuous time.

#### Hidden Markov models

The state is latent and drives an emission distribution.

```@docs
DiscreteTimeDiscreteStatesHMM
ContinuousTimeDiscreteStatesHMM
MVDiscreteTimeDiscreteStatesHMM
MVContinuousTimeDiscreteStatesHMM
probabilities_hidden_states
posterior_hidden_states
```

#### Observed-state Markov models

The state itself is the observation; `coarsed` wraps an observed-state model for
set-valued (ambiguous) observations.

```@docs
DiscreteTimeObservedStatesMarkovModel
ContinuousTimeObservedStatesMarkovModel
CoarsedObservedStatesMarkovModel
coarsed
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

## Additional Exported API

Exported, stable API that was missing from the sections above.

### Variational Inference

```@docs
VI
VIResult
get_variational_posterior
get_vi_state
get_vi_trace
```

### Goodness-of-Fit and Diagnostic Plots

```@docs
plot_dv_pred
plot_dv_ipred
plot_wres_pred
plot_shrinkage
plot_observed_profiles
plot_em_trajectories
```

### Estimation and Random-Effects Helpers

```@docs
sample_random_effects
sample_posterior
reestimate_ebes
get_loglikelihood_quadrature
compute_shrinkage
compare_parameters
MCIntegrator
get_laplace_random_effects
```

### Serialization

```@docs
save_fit
load_fit
```

### Model Display

```@docs
show_equations
```
