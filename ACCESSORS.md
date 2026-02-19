Accessors

This document lists the public accessor functions for fit results. The goal is
to avoid direct dot access on fitted objects in user-facing code.

By default, fit results store the DataModel (use `fit_model(...; store_data_model=false)`
to disable this and require passing `dm` into accessors).

Core accessors (FitResult)
- get_method(res) -> FittingMethod
- get_result(res) -> MethodResult
- get_summary(res) -> FitSummary
- get_diagnostics(res) -> FitDiagnostics
- get_params(res; scale=:transformed|:untransformed|:both)
- get_objective(res)
- get_converged(res)
- get_data_model(res) -> DataModel (stored by default)

Multistart accessors
- get_multistart_results(res) -> Vector{FitResult} (successful runs)
- get_multistart_errors(res) -> Vector{Any} (errors for failed runs)
- get_multistart_starts(res) -> Vector{ComponentArray} (starts for successful runs)
- get_multistart_failed_results(res) -> Vector{Any} (failed runs; results are `nothing`)
- get_multistart_failed_starts(res) -> Vector{ComponentArray} (starts for failed runs)
- get_multistart_best_index(res) -> Int (index into successful runs)
- get_multistart_best(res) -> FitResult

All core accessors also work on MultistartFitResult and will route to the best run.

Method-specific accessors
- MCMC
  - get_chain(res)
  - get_observed(res)
  - get_sampler(res)
  - get_n_samples(res)
- Optimization-based methods (MLE/MAP/Laplace/LaplaceMAP/MCEM/SAEM)
  - get_iterations(res)
  - get_raw(res)
  - get_notes(res)

Random-effects accessors
- get_random_effects(dm, res; constants_re=NamedTuple(), flatten=true, include_constants=true)
  - Supported for Laplace, LaplaceMAP, MCEM, and SAEM (EB point estimates).
- get_random_effects(res; constants_re=..., flatten=..., include_constants=...)
  - Uses the stored DataModel; errors if the result does not carry one.

Log-likelihood accessor
- get_loglikelihood(dm, res; constants_re=NamedTuple(), ode_args=(), ode_kwargs=NamedTuple(),
  serialization=EnsembleSerial())
  - Supported for MLE, MAP, Laplace, LaplaceMAP, MCEM, and SAEM.
  - For Laplace/LaplaceMAP/MCEM/SAEM, EB modes are used as random effects.
- get_loglikelihood(res; constants_re=..., ode_args=..., ode_kwargs=..., serialization=...)
  - Uses the stored DataModel; errors if the result does not carry one.

Unsupported access
- Accessors throw an error when the method does not define the requested field
  (for example, get_chain on MLE, or get_loglikelihood on MCMC).
