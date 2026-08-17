# Turing-free core of the sampling estimators. `MCMC`/`VI` and their result types are
# referenced all over the package (uq, serialization, cv, plotting, summaries), so they
# live here; everything that actually talks to Turing/DynamicPPL/AdvancedVI lives in
# NoLimitsTuringExt and is reached through the `_require_ext` gates at the bottom.

import MCMCChains
using SciMLBase
using ComponentArrays
using Distributions
using Bijectors
using Random

export MCMC
export MCMCResult
export VI
export VIResult

function _warn_if_scaled_params(fe::FixedEffects; method_name::AbstractString = "MCMC")
    specs = get_transforms(fe).forward.specs
    ignored = Symbol[]
    for spec in specs
        if spec.kind != :identity
            push!(ignored, spec.name)
        end
    end
    isempty(ignored) ||
        @debug "$(method_name) uses priors on the natural scale; parameter scale settings are ignored during sampling." ignored_parameters = ignored
    return nothing
end

"""
    MCMC(; sampler, turing_kwargs, adtype, progress) <: FittingMethod

Bayesian sampling via Turing.jl for models with or without random effects.
All free fixed effects and random effects must have prior distributions.

# Keyword Arguments
- `sampler`: Turing-compatible sampler. `nothing` (the default) resolves to `NUTS(0.75)`
  once Turing is loaded.
- `turing_kwargs::NamedTuple = NamedTuple()`: keyword arguments forwarded to `Turing.sample`
  (e.g. `n_samples`, `n_adapt`).
- `adtype`: automatic-differentiation backend. `nothing` (the default) resolves to
  `AutoForwardDiff()` once Turing is loaded.
- `progress::Bool = false`: whether to display a progress bar during sampling.
"""
struct MCMC{S, K, A, P} <: FittingMethod
    sampler::S
    turing_kwargs::K
    adtype::A
    progress::P
end

# `n_samples`/`n_adapt` are forwarded verbatim to the sampler, where a non-positive
# value either errors late or silently produces an empty chain (#229).
function _check_turing_kwargs(what::AbstractString, turing_kwargs)
    turing_kwargs isa NamedTuple || return nothing
    if haskey(turing_kwargs, :n_samples)
        n = turing_kwargs.n_samples
        (n isa Integer && n >= 1) ||
            error("$what: turing_kwargs.n_samples must be an integer ≥ 1. Got: $(repr(n))")
    end
    if haskey(turing_kwargs, :n_adapt)
        n = turing_kwargs.n_adapt
        (n isa Integer && n >= 0) ||
            error("$what: turing_kwargs.n_adapt must be an integer ≥ 0. Got: $(repr(n))")
    end
    return nothing
end

function MCMC(;
        sampler = nothing,
        turing_kwargs = NamedTuple(),
        adtype = nothing,
        progress = false
    )
    turing_kwargs = _as_namedtuple(turing_kwargs)
    _check_turing_kwargs("MCMC", turing_kwargs)
    return MCMC(sampler, turing_kwargs, adtype, progress)
end

"""
    MCMCResult{C, S, A, N, O} <: MethodResult

Method-specific result from a [`MCMC`](@ref) fit. Stores the MCMCChains chain,
sampler, number of samples, optional notes, and observed data columns.
"""
struct MCMCResult{C, S, A, N, O} <: MethodResult
    chain::C
    sampler::S
    n_samples::A
    notes::N
    observed::O
end

get_chain(res::MCMCResult) = res.chain

"""
    build_fit_result(dm, method, chain::MCMCChains.Chains; sampler, n_samples, n_adapt=0,
                     observed=<observed columns of dm>, notes=NamedTuple(),
                     store_data_model=true, fit_args=(), fit_kwargs=NamedTuple()) -> FitResult

Bayesian counterpart of [`build_fit_result`](@ref): package a posterior `chain` from a custom
Bayesian estimator into the same first-class `FitResult` a built-in `MCMC` fit returns, so
`get_chain`, `get_observed`, chain-based uncertainty (`compute_uq(res; method=:chain)`),
posterior-predictive plotting, and `summarize` (which reports `inference: bayesian`) all work.
The estimator brings its own `chain`; this only packages it - it does not run a sampler.

Mirrors the built-in MCMC path exactly: the point-estimate slot is filled with the
posterior mean of the fixed effects (richer posterior summaries come from the chain via
`summarize`/`compute_uq`), and `observed` defaults to the model's observed-outcome
columns. Pass `n_adapt` when the chain still contains adaptation draws - summaries and
chain UQ drop that many warm-up rows by default. `fit_kwargs` (e.g. `(constants = …,)`) is stored on the result so `compute_uq`
resolves the same settings the fit used. Dispatch is on the `chain` argument, so the
frequentist `build_fit_result(dm, method, θ; kind=…)` is unaffected.
"""
function build_fit_result(
        dm::DataModel, method::FittingMethod, chain::MCMCChains.Chains;
        sampler, n_samples::Integer, n_adapt::Integer = 0,
        observed = get_df(dm)[:, get_obs_cols(dm)],
        notes = NamedTuple(),
        store_data_model::Bool = true,
        fit_args::Tuple = (), fit_kwargs = NamedTuple()
    )
    n_samples >= 1 ||
        throw(ArgumentError("build_fit_result: `n_samples` must be ≥ 1. Got: $(n_samples)"))
    n_adapt >= 0 ||
        throw(ArgumentError("build_fit_result: `n_adapt` must be ≥ 0. Got: $(n_adapt)"))
    n_adapt < size(chain, 1) ||
        throw(ArgumentError("build_fit_result: `n_adapt` = $(n_adapt) discards every one of the $(size(chain, 1)) draws in `chain`; it must be smaller."))
    result = MCMCResult(chain, sampler, n_samples, notes, observed)
    summary = FitSummary(
        _mcmc_objective(chain, n_adapt), missing,
        FitParameters(ComponentArray(), ComponentArray()), notes
    )
    diagnostics = FitDiagnostics(
        (;), (sampler = sampler,), (n_samples = n_samples, n_adapt = n_adapt), notes
    )
    res = FitResult(
        method, result, summary, diagnostics,
        store_data_model ? dm : nothing, fit_args, fit_kwargs
    )
    return _with_posterior_params(res, dm; rng = Random.default_rng())
end

# Objective for a posterior chain: the mean negative log posterior density over the
# post-warmup draws, so `get_objective` keeps the "lower is better" sign convention of
# the optimization methods. NaN when the sampler records no log-density column.
function _mcmc_objective(chain::MCMCChains.Chains, n_adapt::Integer)
    internals = MCMCChains.names(chain, :internals)
    key = :lp in internals ? :lp : (:logjoint in internals ? :logjoint : nothing)
    key === nothing && return NaN
    arr = Array(getfield(MCMCChains.get(chain, key), key))
    vals = ndims(arr) == 1 ? reshape(arr, :, 1) : arr
    n_iter = size(vals, 1)
    rows = (min(Int(n_adapt), n_iter - 1) + 1):n_iter
    finite = filter(isfinite, vec(vals[rows, :]))
    return isempty(finite) ? NaN : -mean(finite)
end

# Turing's `~` takes a single distribution, so a per-element prior vector (accepted on
# NN/SoftTree/NPF blocks) has to become one product distribution -- the same conversion
# `_logprior_eval` already does on the MAP path.
_turing_prior(prior, name::Symbol) = prior
function _turing_prior(prior::AbstractVector{<:Distribution}, name::Symbol)
    return product_distribution(prior)
end
_turing_prior(prior::MatrixDistribution, name::Symbol) = _SafeMatrixPrior(prior, name)

# A proposal can push a matrix-variate prior's argument onto the positive-definite
# boundary, where Distributions' `logpdf` throws a factorization error instead of
# returning -Inf and kills the sampler. Wrapping keeps the throw inside NoLimits and
# scores the draw -Inf so it is rejected — the same contract as `_mcmc_re_dist`.
struct _SafeMatrixPrior{D <: MatrixDistribution} <: ContinuousMatrixDistribution
    d::D
    name::Symbol
end

Base.size(p::_SafeMatrixPrior) = size(p.d)
Distributions.insupport(p::_SafeMatrixPrior, x::AbstractMatrix{<:Real}) = insupport(p.d, x)
function Distributions._rand!(rng::AbstractRNG, p::_SafeMatrixPrior, x::AbstractMatrix)
    return Distributions._rand!(rng, p.d, x)
end
Bijectors.bijector(p::_SafeMatrixPrior) = Bijectors.bijector(p.d)

# Linking is the one thing the wrapper cannot inherit: Bijectors dispatches the
# constrained↔unconstrained maps on the concrete distribution type. `VectorBijectors` is
# the Bijectors 0.16 interface; on 0.15 `bijector` above is the whole story.
if isdefined(Bijectors, :VectorBijectors)
    for f in (:from_linked_vec, :to_linked_vec, :linked_vec_length, :linked_optic_vec)
        @eval function Bijectors.VectorBijectors.$f(p::_SafeMatrixPrior)
            return Bijectors.VectorBijectors.$f(p.d)
        end
    end
end

function Distributions._logpdf(p::_SafeMatrixPrior, x::AbstractMatrix{<:Real})
    try
        return logpdf(p.d, x)
    catch err
        _is_numeric_error(err) || rethrow(err)
        if !Threads.atomic_cas!(_WARNED_NUMERIC_ERROR, false, true)
            @warn "A numeric error ($(nameof(typeof(err)))) was raised while evaluating " *
                "the prior of $(p.name) (a $(nameof(typeof(p.d)))); rejecting this " *
                "proposal. Warned once per fit."
        end
        return convert(float(eltype(x)), -Inf)
    end
end

"""
    VI(; turing_kwargs=NamedTuple()) <: FittingMethod

Variational inference via Turing/AdvancedVI for **fixed-effects-only** models.
All free fixed effects must have prior distributions.

`turing_kwargs` controls VI behavior and is forwarded to `Turing.vi` after removing
NoLimits-managed keys:
- `max_iter::Int` (default: `1000`)
- `family::Symbol` (`:meanfield` or `:fullrank`, default: `:meanfield`)
- `q_init` (optional custom variational family)
- `adtype` (default: `Turing.AutoForwardDiff()`)
- `progress` / `show_progress` (default: `false`)
- `convergence_window`, `convergence_rtol`, `convergence_atol` (NoLimits convergence rule)

!!! note
    VI is not supported for models with random effects. Use `MCMC` for full Bayesian
    inference on mixed-effects models.
"""
struct VI{K} <: FittingMethod
    turing_kwargs::K
end

function VI(; turing_kwargs = NamedTuple())
    turing_kwargs = _as_namedtuple(turing_kwargs)
    _check_turing_kwargs("VI", turing_kwargs)
    return VI(turing_kwargs)
end

"""
    VIResult{Q, T, S, N, O, C, M} <: MethodResult

Method-specific result from a [`VI`](@ref) fit. Stores the variational posterior,
optimization trace/state, ELBO summary, and observed data.
"""
struct VIResult{Q, T, S, N, O, C, M} <: MethodResult
    posterior::Q
    trace::T
    state::S
    n_iter::Int
    max_iter::Int
    final_elbo::Float64
    converged::Bool
    notes::N
    observed::O
    coord_names::C
    model::M       # DynamicPPL model used for VI; needed to unlink posterior draws to
    # natural space. `nothing` after deserialization (draws stay linked).
end

get_variational_posterior(res::VIResult) = res.posterior
get_vi_trace(res::VIResult) = res.trace
get_vi_state(res::VIResult) = res.state

# Sampler kind is recorded when a fit is serialized; the Turing sampler types that
# refine this live in the extension.
@inline _mcmc_sampler_kind(sampler) = :other

# Filled in by NoLimitsTuringExt.
function _mcmc_fit_impl end
function _vi_fit_impl end
function _vi_unlink_draws end

"""
    sample_posterior(res::VIResult; n_draws, rng, return_names)

Draw `n_draws` parameter vectors from a variational posterior, mapped back to the
natural (constrained) scale. A deserialized result carries no model, so its draws are
returned on the linked scale unchanged - that path needs no Turing.
"""
function sample_posterior(
        res::VIResult; n_draws::Int = 1000,
        rng::AbstractRNG = Random.default_rng(), return_names::Bool = false
    )
    n_draws >= 1 || error("n_draws must be >= 1.")
    raw = rand(rng, res.posterior, n_draws)
    mat = raw isa AbstractVector ? reshape(raw, :, 1) : Matrix(raw)
    linked = Matrix(permutedims(mat))
    draws = if res.model === nothing
        linked
    else
        _require_ext(:NoLimitsTuringExt, :Turing, "Unlinking variational draws")
        _vi_unlink_draws(res, linked)
    end
    if return_names
        return (draws = draws, names = res.coord_names)
    end
    return draws
end

function _fit_model(dm::DataModel, method::MCMC, args...; kwargs...)
    _require_ext(:NoLimitsTuringExt, :Turing, "Fitting with `MCMC`")
    return _mcmc_fit_impl(dm, method, args...; kwargs...)
end

function _fit_model(dm::DataModel, method::VI, args...; kwargs...)
    _require_ext(:NoLimitsTuringExt, :Turing, "Fitting with `VI`")
    return _vi_fit_impl(dm, method, args...; kwargs...)
end
