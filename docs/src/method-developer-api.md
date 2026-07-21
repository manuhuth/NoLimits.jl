# Method-Developer API

This page is for researchers who want to implement a *new* estimation method for nonlinear
mixed-effects models on top of NoLimits.jl, rather than use one of the built-in fitters. It
exposes the statistical building blocks the built-in methods are made of - the complete-data
likelihood, the random-effect prior, the joint and marginal densities, empirical-Bayes modes
and covariances, and posterior sampling - as composable, documented, semver-stable functions.

For using the package to fit models, see the [Quickstart](quickstart.md) and
[Estimation](estimation/index.md) pages. For a step-by-step walkthrough that builds two working
estimators from these primitives - a Monte-Carlo EM and a closed-form-posterior EM - see the
[Building Custom Estimators](tutorials/building-custom-estimators.md) tutorial. For contributing
to the package internals, see the [Developers Guide](developers-guide.md).

## Two contracts

Every function here obeys two conventions:

1. **Natural-scale parameters.** Fixed effects `θ` and random effects `η` are passed as
   natural-scale (constrained, human-readable) `ComponentArray`s. The unconstrained optimizer
   scale is reached only through the transform primitives (`ForwardTransform`,
   `InverseTransform`, `apply_inv_jacobian_T`, and the change-of-variables companion
   `logabsdetjac`). PSD-matrix parameters are symmetrized for you at each public call.
2. **Batches are the random-effect currency.** Individuals that share a random-effect level
   (for example a crossed `ID` x `SITE` design) are grouped into a *batch* by a transitive
   union-find. The canonical per-batch random-effect argument `b` is a flat natural-scale
   vector of length `get_batch_re_dim(batch)`; `build_re_batch_infos(dm, constants_re)` returns
   the batch descriptors, and `build_eta_individual` / `eta_from_modes` map `b` back to per-
   individual `η`. Independent-subject models are simply the special case of one individual per
   batch.

## The primitives

| Layer | Functions |
|---|---|
| Data / batching | `build_re_batch_infos`, `get_batch_individuals`, `get_batch_re_info`, `get_batch_re_dim`, `build_eta_individual`, `eta_from_modes`, `build_likelihood_cache` |
| Forward map | `solve_individual`, `obs_distributions`, `hmm_filter_step!` |
| Densities | `conditional_loglikelihood`, `complete_data_loglikelihood`, `re_logprior`, `complete_data_loglikelihood_gradient`, `complete_data_loglikelihood_hessian` |
| Posterior / empirical Bayes | `empirical_bayes`, `posterior_moments`, `sample_random_effect_draws` |
| Marginals | `laplace_marginal`, `ghq_marginal` |
| Curvature seam | `AbstractCurvature`, `ExactHessianCurvature`, `FisherInformationCurvature`, `inner_curvature` |
| Fitting drivers | `fit_method`, `fit_fixed_effects`, `fit_laplace_family` |
| Transforms | `ForwardTransform`, `InverseTransform`, `apply_inv_jacobian_T`, `logabsdetjac` |

Full signatures are in the [API reference](api.md). The identity `complete_data_loglikelihood ==
conditional_loglikelihood + re_logprior` holds at batch scale, and `posterior_moments` returns
the Laplace covariance `Σ = (−H)⁻¹` at the empirical-Bayes mode.

## Building a new fitting method

A new estimator is a `struct MyMethod <: FittingMethod` plus one method,
`fit_method(dm, ::MyMethod, ...)`, which the public `fit_model` dispatches to. Implementing it
means `fit_model`, `Multistart`, and every result accessor (`get_params`, `get_objective`,
`get_random_effects`, `get_loglikelihood`, uncertainty quantification) work automatically,
because the shared drivers below build the same result types the built-in methods use.

One contract to honour: `Multistart` and `pooled_init` deliver their starting points through the
`theta_0_untransformed` keyword of `fit_method`. The shared drivers handle it for you; a
hand-rolled `fit_method` that swallows it in `kwargs...` silently ignores every start, so accept
it and use it as the initial `θ` (see the tutorial's `ClosedFormEM`).

### The quick path: `FitContext`

For a hand-rolled iterative method, `build_fit_context(dm; constants_re)` performs the setup
every fitting loop needs, once: the random-effect batch structure (`build_re_batch_infos`), the
constant-RE cache, and the likelihood evaluation cache (`build_likelihood_cache`). The context
is θ-independent - build it once per fit and reuse it across all iterations; parameters flow
through every call. With a context, the primitives lose their cache keywords and address batches
by index, `optimize_parameters` runs an M-step from an objective written purely in natural-scale
parameters (the transformed-scale round trip and PSD symmetrisation are handled), and
`build_fit_result(ctx, …)` fills `eb_modes` automatically:

```julia
function NoLimits.fit_method(dm, m::MyEM, args...; theta_0_untransformed = nothing, kwargs...)
    ctx = build_fit_context(dm)
    θ = something(theta_0_untransformed, initial_parameters(ctx))
    for _ in 1:m.n_iter
        pm = posterior_moments(ctx, θ)                     # E-step at the current θ
        θ, _ = optimize_parameters(ctx; θ_start = θ) do θn  # M-step, natural scale
            -sum(complete_data_loglikelihood(ctx, bi, θn, pm[bi][1]) +
                 0.5 * tr(pm[bi][2] * complete_data_loglikelihood_hessian(ctx, bi, θn, pm[bi][1]))
                 for bi in eachindex(get_batch_infos(ctx)))
        end
    end
    return build_fit_result(ctx, m, θ; kind = :frequentist_re,
        objective = -laplace_marginal(ctx, θ))
end
```

Every context call is a thin cover forwarding to the corresponding cache-explicit primitive
with the context's stored objects - results are identical, and the explicit layer below remains
the full-control path (own caches per thread, `BatchThetaContext` amortisation, custom bounds).
Note that the population primitives called with a bare `dm` (e.g. `posterior_moments(dm, θ)`)
rebuild the caches on every call - inside a loop, prefer the context forms or the explicit
layer. See the [Building Custom Estimators](tutorials/building-custom-estimators.md) tutorial
for the full walkthrough.

### A fixed-effects method

Delegate to `fit_fixed_effects` with a natural-scale `objective_term` added to the negative
log-likelihood. Here is a ridge-penalized MLE:

```julia
using NoLimits, Optimization, OptimizationOptimJL

struct RidgeMLE{O, K, A, L, U} <: FittingMethod
    λ::Float64
    optimizer::O
    optim_kwargs::K
    adtype::A
    lb::L
    ub::U
    ignore_model_bounds::Bool
end
RidgeMLE(; λ = 1.0, optimizer = LBFGS(), optim_kwargs = (;),
    adtype = AutoForwardDiff(), lb = nothing, ub = nothing, ignore_model_bounds = false) =
    RidgeMLE(λ, optimizer, optim_kwargs, adtype, lb, ub, ignore_model_bounds)

function NoLimits.fit_method(dm, m::RidgeMLE, args...; constants = NamedTuple(),
        penalty = NamedTuple(), extra_objective = nothing, kwargs...)
    ridge(θu) = m.λ * sum(abs2, θu)          # natural-scale penalty
    return fit_fixed_effects(dm, m; objective_term = ridge,
        constants = constants, penalty = penalty, kwargs...)
end

res = fit_model(dm, RidgeMLE(λ = 2.0))       # get_params/get_objective/... all work
```

### A random-effects method

A new marginal method is a custom curvature plugged into `fit_laplace_family`. A curvature is a
`struct <: AbstractCurvature` with one method, `inner_curvature`, returning the raw
`H = ∇²_b log f`; everything downstream (empirical-Bayes solve, the marginal, the analytic
θ-gradient, threading) is inherited. Here is a diagonal-Hessian Laplace variant:

```julia
using NoLimits, LinearAlgebra

struct DiagonalCurvature <: AbstractCurvature end
function NoLimits.inner_curvature(::DiagonalCurvature, dm, batch, θ, b, cc, cache, ws;
        ctx = "", tctx = nothing)
    H = inner_curvature(ExactHessianCurvature(), dm, batch, θ, b, cc, cache, ws;
        ctx = ctx, tctx = tctx)
    return Matrix(Diagonal(diag(H)))
end

struct DiagonalLaplace <: FittingMethod
    base::Laplace
end
DiagonalLaplace(; kwargs...) = DiagonalLaplace(Laplace(; kwargs...))

function NoLimits.fit_method(dm, m::DiagonalLaplace, args...; kwargs...)
    fit_kwargs = NamedTuple(kwargs)
    return fit_laplace_family(dm, m.base, DiagonalCurvature(), args, fit_kwargs, _ -> nothing;
        nan_recovery = m.base.nan_recovery, kwargs...)
end

res = fit_model(dm, DiagonalLaplace())       # get_random_effects/get_loglikelihood all work
```

Because `fit_laplace_family` returns a `FrequentistREResult`, the method is first-class across the
random-effects accessors without any further wiring.

### A Bayesian method

A Bayesian estimator produces a posterior *chain* rather than a point estimate, so it packages
its result with the chain method of `build_fit_result`. The estimator brings its own chain (from
whatever sampler it runs); `build_fit_result` wraps it in the same result a built-in `MCMC` fit
returns, so `get_chain`, chain-based uncertainty (`compute_uq(res; method=:chain)`),
posterior-predictive plotting, and `summarize` (which reports `inference: bayesian`) all work -
the method keeps its own type.

```julia
struct MyBayes <: FittingMethod end

function NoLimits.fit_method(dm, m::MyBayes, args...; kwargs...)
    chain = run_my_sampler(dm)                      # your sampler returns an MCMCChains.Chains
    return build_fit_result(dm, m, chain; sampler = :my_sampler, n_samples = size(chain, 1))
end

res = fit_model(dm, MyBayes())                       # get_chain / compute_uq(:chain) / summarize
```

As with the built-in `MCMC` fit, the point-estimate `get_params` slot is empty; the posterior
summaries (medians, credible intervals) are computed from the chain by `summarize` and
`compute_uq`.

## Assembling an objective directly

For a method that does not fit the fixed-effects or marginal-RE templates, build the objective
from the density primitives. A sketch of a from-scratch empirical-Bayes objective:

```julia
θ = get_θ0_untransformed(get_fixed(get_model(dm)))
_, batches, const_cache = build_re_batch_infos(dm, NamedTuple())
cache = build_likelihood_cache(dm)

# per-batch empirical-Bayes mode and Laplace marginal at θ
b_stars = empirical_bayes(dm, θ)
marginal = sum(laplace_marginal(dm, θ, batches[i], b_stars[i];
                                const_cache = const_cache, cache = cache)
               for i in eachindex(batches))
```

`complete_data_loglikelihood_gradient` / `complete_data_loglikelihood_hessian` give the inner `∇_b` / `∇²_b`,
and `sample_random_effect_draws` draws from the random-effect posterior (Laplace-Gaussian importance sampling by
default, or `method = :mcmc` for a Turing sampler).

## Stability

The names on this page are the semver-stable method-developer API: a breaking change to any of
them requires a major version bump. Underscore-prefixed symbols are internal and may change at
any release.
