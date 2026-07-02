export FittingMethod
export MethodResult
export FitResult
export FitSummary
export FitDiagnostics
export FitParameters
export fit_model
export get_summary
export get_params
export get_random_effects
export sample_random_effects
export reestimate_ebes
export get_diagnostics
export get_result
export get_method
export get_objective
export get_converged
export get_data_model
export get_chain
export get_iterations
export get_raw
export get_notes
export get_closed_form_mstep_used
export get_observed
export get_sampler
export get_n_samples
export get_variational_posterior
export get_vi_trace
export get_vi_state
export sample_posterior
export get_loglikelihood
export get_loglikelihood_quadrature
export get_laplace_random_effects
export get_re_covariate_usage
export compute_shrinkage
export loglikelihood
export build_ll_cache
export MCIntegrator

using StatsFuns
using SpecialFunctions
using DataFrames
using Random
using SciMLBase

"""
    FittingMethod

Abstract base type for all estimation methods. Concrete subtypes include
[`MLE`](@ref), [`MAP`](@ref), [`MCMC`](@ref), [`Laplace`](@ref),
[`SAEM`](@ref), [`MCEM`](@ref), and [`Multistart`](@ref).
"""
abstract type FittingMethod end

"""
    _SavedFittingMethod <: FittingMethod

Lightweight stub that records which fitting method was used when a [`FitResult`](@ref)
is loaded from disk via [`load_fit`](@ref). Replaces the full method struct (which
contains optimizer closures that cannot be serialized) on save.

`get_method(res).kind` returns a Symbol such as `:mle`, `:map`, `:laplace`, etc.
"""
struct _SavedFittingMethod <: FittingMethod
    kind::Symbol   # :mle :map :laplace
    # :mcem :saem :mcmc :vi :ghquadrature
end

function Base.show(io::IO, m::_SavedFittingMethod)
    print(io,
        "_SavedFittingMethod(:$(m.kind)) [loaded from disk; original optimizer not stored]")
end

"""
    MethodResult

Abstract base type for the method-specific result structs stored inside
[`FitResult`](@ref). Each [`FittingMethod`](@ref) subtype has a corresponding
`MethodResult` subtype.
"""
abstract type MethodResult end

struct EBEOptions{O, K, A, T}
    optimizer::O
    optim_kwargs::K
    adtype::A
    grad_tol::T
    multistart_n::Int
    multistart_k::Int
    max_rounds::Int
    sampling::Symbol
end

struct EBERescueOptions{T}
    enabled::Bool
    multistart_n::Int
    multistart_k::Int
    max_rounds::Int
    grad_tol::T
    sampling::Symbol
end

"""
    MCIntegrator(; n_samples, mode, sampler, n_warmup, rng)

Configuration for Monte Carlo marginal log-likelihood integration used by
[`get_loglikelihood_quadrature`](@ref) as a primary integrator or fallback.

# Keyword Arguments
- `n_samples::Int = 1000`: number of Monte Carlo samples.
- `mode::Symbol = :turing`: integration mode.
  - `:turing` — uses Turing MCMC to fit a Gaussian proposal `q` to the posterior
    `p(b|y,θ)`, then draws fresh IID samples from `q` and applies an IS correction.
    More accurate than `:prior` at the same sample count; default.
  - `:prior` — draws `b_r ~ p(b | θ)` from the prior and estimates
    `log p(y|θ) ≈ logsumexp(log p(y|b_r,θ)) - log(n_samples)`.
    Simple fallback; higher variance when the posterior is narrow relative to the prior.
- `sampler`: Turing-compatible sampler used when `mode = :turing`
  (e.g. `MH()`, `NUTS(0.65)`, `AdaptiveNoLimitsMH()`). Ignored for `:prior`.
- `n_warmup::Int = 500`: number of MCMC warmup steps. Ignored for `:prior`.
- `rng::Union{Nothing, AbstractRNG} = nothing`: random number generator.
  `nothing` (default) means the RNG is inherited from `get_loglikelihood_quadrature`'s
  `seed` argument, ensuring reproducibility without setting it explicitly here.
  Pass an explicit RNG to pin reproducibility independently of the caller.
"""
struct MCIntegrator{S, R}
    n_samples::Int
    mode::Symbol
    sampler::S
    n_warmup::Int
    rng::R   # Union{Nothing, AbstractRNG}
end

function MCIntegrator(;
        n_samples::Int = 1000,
        mode::Symbol = :turing,
        sampler = nothing,
        n_warmup::Int = 500,
        rng::Union{Nothing, AbstractRNG} = nothing
)
    mode in (:prior, :turing) ||
        error("MCIntegrator mode must be :prior or :turing, got :$(mode).")
    n_samples > 0 || error("MCIntegrator n_samples must be > 0.")
    n_warmup >= 0 || error("MCIntegrator n_warmup must be ≥ 0.")
    return MCIntegrator(n_samples, mode, sampler, n_warmup, rng)
end

@inline _default_ebe_grad_tol(dm::DataModel) = dm.model.de.de === nothing ? 1e-4 : 1e-2

@inline function _resolve_multistart_sampling(sampling, what::AbstractString)
    (sampling === :lhs || sampling === :random) || error("$(what) must be :lhs or :random.")
    return sampling
end

@inline function _resolve_ebe_grad_tol(grad_tol, dm::DataModel)
    if grad_tol isa Symbol
        grad_tol === :auto || error("EBE grad_tol must be numeric or :auto.")
        return _default_ebe_grad_tol(dm)
    end
    return grad_tol
end

@inline function _resolve_ebe_options(ebe::EBEOptions, dm::DataModel)
    grad_tol = _resolve_ebe_grad_tol(ebe.grad_tol, dm)
    sampling = _resolve_multistart_sampling(ebe.sampling, "EBE multistart sampling")
    return EBEOptions(ebe.optimizer, ebe.optim_kwargs, ebe.adtype, grad_tol,
        ebe.multistart_n, ebe.multistart_k, ebe.max_rounds, sampling)
end

@inline function _resolve_ebe_rescue_options(
        rescue::Union{Nothing, EBERescueOptions}, ebe_grad_tol, dm::DataModel)
    rescue === nothing && return nothing
    grad_tol = rescue.grad_tol
    if grad_tol isa Symbol
        grad_tol === :auto || error("EBE rescue grad_tol must be numeric or :auto.")
        grad_tol = ebe_grad_tol
    end
    sampling = _resolve_multistart_sampling(
        rescue.sampling, "EBE rescue multistart sampling")
    return EBERescueOptions(rescue.enabled, rescue.multistart_n, rescue.multistart_k,
        rescue.max_rounds, grad_tol, sampling)
end

"""
    FitParameters{T, U}

Stores parameter estimates on both the transformed (optimization) and untransformed
(natural) scales as `ComponentArray`s.

# Fields
- `transformed::T`: parameter vector on the optimization scale.
- `untransformed::U`: parameter vector on the natural scale.
"""
struct FitParameters{T, U}
    transformed::T
    untransformed::U
end

"""
    FitSummary{O, C, P, N}

High-level summary of a fitting result.

# Fields
- `objective::O`: the final objective value (negative log-likelihood, negative log-posterior, etc.).
- `converged::C`: convergence flag (`true` / `false` / `nothing` for MCMC).
- `params::P`: a [`FitParameters`](@ref) struct with parameter estimates.
- `notes::N`: method-specific string notes or `nothing`.
"""
struct FitSummary{O, C, P, N}
    objective::O
    converged::C
    params::P
    notes::N
end

"""
    FitDiagnostics{T, O, X, N}

Diagnostic information for a fitting run.

# Fields
- `timing::T`: elapsed time in seconds.
- `optimizer::O`: optimizer-specific diagnostic (e.g. Optim.jl result).
- `convergence::X`: convergence-related metadata.
- `notes::N`: additional string notes.
"""
struct FitDiagnostics{T, O, X, N}
    timing::T
    optimizer::O
    convergence::X
    notes::N
end

"""
    FitResult{M, R, S, D, DM, A, K}

Unified result wrapper returned by [`fit_model`](@ref). Contains the fitting method,
method-specific result, summary, diagnostics, and optionally the `DataModel`.

Use accessor functions rather than accessing fields directly:
[`get_summary`](@ref), [`get_diagnostics`](@ref), [`get_result`](@ref),
[`get_method`](@ref), [`get_objective`](@ref), [`get_converged`](@ref),
[`get_params`](@ref), [`get_data_model`](@ref).
"""
struct FitResult{M <: FittingMethod, R <: MethodResult, S, D, DM, A, K}
    method::M
    result::R
    summary::S
    diagnostics::D
    data_model::DM
    fit_args::A
    fit_kwargs::K
end

"""
    get_summary(res::FitResult) -> FitSummary

Return the [`FitSummary`](@ref) containing objective, convergence flag, and parameters.
"""
get_summary(res::FitResult) = res.summary

"""
    get_diagnostics(res::FitResult) -> FitDiagnostics

Return the [`FitDiagnostics`](@ref) with timing, optimizer, and convergence details.
"""
get_diagnostics(res::FitResult) = res.diagnostics

"""
    get_result(res::FitResult) -> MethodResult

Return the method-specific [`MethodResult`](@ref) subtype (e.g. `MLEResult`, `MCMCResult`).
"""
get_result(res::FitResult) = res.result

"""
    get_method(res::FitResult) -> FittingMethod

Return the [`FittingMethod`](@ref) used to produce this result.
"""
get_method(res::FitResult) = res.method

"""
    get_objective(res::FitResult) -> Real

Return the final objective value (e.g. negative log-likelihood for MLE).
"""
get_objective(res::FitResult) = res.summary.objective

"""
    get_converged(res::FitResult) -> Bool or Nothing

Return the convergence flag. `true` indicates successful convergence, `false` indicates
failure, and `nothing` is returned for methods that do not track convergence (e.g. MCMC).
"""
get_converged(res::FitResult) = res.summary.converged

"""
    get_data_model(res::FitResult) -> DataModel or Nothing

Return the [`DataModel`](@ref) stored in the fit result, or `nothing` if the result
was created with `store_data_model=false`.
"""
get_data_model(res::FitResult) = res.data_model
get_fit_args(res::FitResult) = res.fit_args
get_fit_kwargs(res::FitResult) = res.fit_kwargs

# Stored fit keyword with a default — shared by the RE/loglikelihood accessors,
# plotting, and the UQ entry points.
@inline function _fit_kw(res::FitResult, key::Symbol, default)
    return haskey(res.fit_kwargs, key) ? getfield(res.fit_kwargs, key) : default
end

function _nl_fmt_compact_value(x)
    x === nothing && return "-"
    x isa Missing && return "-"
    if x isa Real
        xv = Float64(x)
        isfinite(xv) || return "-"
        ax = abs(xv)
        if ax >= 1e4 || (ax > 0 && ax < 1e-3)
            return string(round(xv; sigdigits = 4))
        end
        return string(round(xv; digits = 4))
    end
    return string(x)
end

_nl_method_name(m) = nameof(typeof(m))
_nl_method_name(m::_SavedFittingMethod) = Symbol(uppercase(string(m.kind)) * "_loaded")

function _nl_fitresult_show_line(res::FitResult)
    method_name = _nl_method_name(res.method)
    objective_str = _nl_fmt_compact_value(get_objective(res))
    converged = get_converged(res)
    n_params = try
        length(get_params(res; scale = :untransformed))
    catch
        "?"
    end
    dm_state = res.data_model === nothing ? "not_stored" : "stored"
    return "FitResult(method=$(method_name), objective=$(objective_str), converged=$(converged), n_params=$(n_params), data_model=$(dm_state))"
end

Base.show(io::IO, res::FitResult) = print(io, _nl_fitresult_show_line(res))
function Base.show(io::IO, ::MIME"text/plain", res::FitResult)
    print(io, _nl_fitresult_show_line(res))
end

function _res_constants_re(res::FitResult, constants_re::NamedTuple)
    isempty(constants_re) || return constants_re
    if haskey(res.fit_kwargs, :constants_re)
        return getfield(res.fit_kwargs, :constants_re)
    end
    return constants_re
end

"""
    get_params(res::FitResult; scale=:both) -> FitParameters or ComponentArray

Return the estimated parameter vector.

# Keyword Arguments
- `scale::Symbol = :both`: which scale to return.
  - `:both` — a [`FitParameters`](@ref) struct with both scales.
  - `:transformed` — the optimization-scale `ComponentArray`.
  - `:untransformed` — the natural-scale `ComponentArray`.
"""
function get_params(res::FitResult; scale::Symbol = :both)
    params = res.summary.params
    scale === :both && return params
    scale === :transformed && return params.transformed
    scale === :untransformed && return params.untransformed
    error("scale must be :both, :transformed, or :untransformed.")
end

"""
    get_chain(res::FitResult) -> MCMCChains.Chains

Return the MCMC chain. Only valid for results produced by [`MCMC`](@ref).
"""
function get_chain(res::FitResult)
    return get_chain(res.result)
end

get_chain(::MethodResult) = error("Chain access not supported for this method.")

"""
    get_iterations(res::FitResult) -> Int

Return the number of optimizer iterations. Valid for optimization-based methods
(MLE, MAP, Laplace, MCEM, SAEM).
"""
get_iterations(res::FitResult) = get_iterations(res.result)

"""
    get_raw(res::FitResult)

Return the raw method-specific result object (e.g. the Optim.jl result for MLE/MAP).
"""
get_raw(res::FitResult) = get_raw(res.result)

"""
    get_notes(res::FitResult) -> String or Nothing

Return any method-specific string notes attached to the result.
"""
get_notes(res::FitResult) = get_notes(res.result)

"""
    get_closed_form_mstep_used(res::FitResult) -> Bool

Return `true` when the fitting run used any closed-form M-step updates.

Currently this is method-specific metadata populated by methods that support
closed-form M-step paths (e.g. SAEM). Methods without this concept return
`false`.
"""
get_closed_form_mstep_used(res::FitResult) = get_closed_form_mstep_used(res.result)

"""
    get_observed(res::FitResult)

Return the observed data used during MCMC sampling. Only valid for MCMC results.
"""
get_observed(res::FitResult) = get_observed(res.result)

"""
    get_sampler(res::FitResult)

Return the sampler object (e.g. `NUTS`) used for MCMC. Only valid for MCMC results.
"""
get_sampler(res::FitResult) = get_sampler(res.result)

"""
    get_n_samples(res::FitResult) -> Int

Return the number of MCMC samples drawn. Only valid for MCMC results.
"""
get_n_samples(res::FitResult) = get_n_samples(res.result)

"""
    get_variational_posterior(res::FitResult)

Return the variational posterior object for VI fits.
"""
get_variational_posterior(res::FitResult) = get_variational_posterior(res.result)

"""
    get_vi_trace(res::FitResult)

Return per-iteration VI trace information.
"""
get_vi_trace(res::FitResult) = get_vi_trace(res.result)

"""
    get_vi_state(res::FitResult)

Return the final VI optimizer state.
"""
get_vi_state(res::FitResult) = get_vi_state(res.result)

"""
    sample_posterior(res::FitResult; n_draws, rng)

Draw posterior samples from methods that expose a posterior sampler (e.g. VI).
"""
sample_posterior(res::FitResult; kwargs...) = sample_posterior(res.result; kwargs...)

@inline function _maxabs(v::AbstractVector)
    m = zero(eltype(v))
    @inbounds for i in eachindex(v)
        ai = abs(v[i])
        if ai > m
            m = ai
        end
    end
    return m
end

@inline function _maxabsdiff(a::AbstractVector, b::AbstractVector)
    length(a) == length(b) || throw(DimensionMismatch("vectors must have equal length"))
    m = zero(promote_type(eltype(a), eltype(b)))
    @inbounds for i in eachindex(a, b)
        d = abs(a[i] - b[i])
        if d > m
            m = d
        end
    end
    return m
end

@inline function _spawn_child_rngs(rng::AbstractRNG, n::Int)
    n <= 0 && return Random.Xoshiro[]
    seeds = rand(rng, UInt64, n)
    return [Random.Xoshiro(seeds[i]) for i in 1:n]
end

function get_iterations(res::MethodResult)
    hasproperty(res, :iterations) ? res.iterations :
    error("iterations not available for this method.")
end
function get_raw(res::MethodResult)
    hasproperty(res, :raw) ? res.raw :
    error("raw result not available for this method.")
end
function get_notes(res::MethodResult)
    hasproperty(res, :notes) ? res.notes :
    error("notes not available for this method.")
end
get_closed_form_mstep_used(::MethodResult) = false
function get_observed(res::MethodResult)
    hasproperty(res, :observed) ? res.observed :
    error("observed data not available for this method.")
end
function get_sampler(res::MethodResult)
    hasproperty(res, :sampler) ? res.sampler :
    error("sampler not available for this method.")
end
function get_n_samples(res::MethodResult)
    hasproperty(res, :n_samples) ? res.n_samples :
    error("n_samples not available for this method.")
end
function get_variational_posterior(::MethodResult)
    error("Variational posterior access not supported for this method.")
end
get_vi_trace(::MethodResult) = error("VI trace access not supported for this method.")
get_vi_state(::MethodResult) = error("VI state access not supported for this method.")
function sample_posterior(::MethodResult; kwargs...)
    error("Posterior sampling not supported for this method.")
end

function _re_dataframes_from_bstars(dm::DataModel,
        batch_infos::Vector,
        bstars::Vector;
        constants_re::NamedTuple = NamedTuple(),
        flatten::Bool = true,
        include_constants::Bool = true)
    cache = dm.re_group_info.laplace_cache
    cache === nothing && return NamedTuple()
    re_names = cache.re_names
    isempty(re_names) && return NamedTuple()
    length(bstars) == length(batch_infos) ||
        error("EB modes do not match number of batches.")
    re_groups = get_re_groups(dm.model.random.random)
    fixed_maps = _normalize_constants_re(dm, constants_re)
    const_cache = _build_constants_cache(dm, fixed_maps)

    # collect free EB values by level
    re_level_vals = Dict{Symbol, Dict{Int, Any}}()
    for re in re_names
        re_level_vals[re] = Dict{Int, Any}()
    end
    for (bi, info) in enumerate(batch_infos)
        b = bstars[bi]
        for (ri, re) in enumerate(re_names)
            rei = info.re_info[ri]
            for lvl_id in rei.map.levels
                v = _re_value_from_b(rei, lvl_id, b)
                v === nothing && continue
                re_level_vals[re][lvl_id] = v
            end
        end
    end

    # build output DataFrames per RE
    out_pairs = Pair{Symbol, Any}[]
    for (ri, re) in enumerate(re_names)
        col = getfield(re_groups, re)
        levels_all = cache.re_index[ri].levels
        level_ids = collect(1:length(levels_all))
        free_vals = re_level_vals[re]
        const_mask = const_cache.is_const[ri]
        const_scalars = const_cache.scalar_vals[ri]
        const_vectors = const_cache.vector_vals[ri]
        is_scalar = cache.is_scalar[ri]

        # determine dimension
        dim = 1
        for info in batch_infos
            rei = info.re_info[ri]
            if rei.dim > 0
                dim = rei.dim
                break
            end
        end
        dim == 0 && (dim = 1)

        rows = Any[]
        vals_flat = Vector{Vector{Any}}()
        for lvl_id in level_ids
            v = nothing
            if include_constants && const_mask[lvl_id]
                v = is_scalar ? const_scalars[lvl_id] : const_vectors[lvl_id]
            elseif haskey(free_vals, lvl_id)
                v = free_vals[lvl_id]
            end
            v === nothing && continue
            push!(rows, levels_all[lvl_id])
            if flatten
                if v isa Number
                    push!(vals_flat, [v])
                else
                    push!(vals_flat, collect(vec(v)))
                end
            else
                push!(vals_flat, [v])
            end
        end

        if flatten
            names = flatten_re_names(re, zeros(dim))
            df = DataFrame(col => rows)
            for j in 1:length(names)
                df[!, names[j]] = [vals_flat[i][j] for i in 1:length(vals_flat)]
            end
            push!(out_pairs, re => df)
        else
            push!(out_pairs,
                re => DataFrame(col => rows, :value => [v[1] for v in vals_flat]))
        end
    end
    return NamedTuple(out_pairs)
end

"""
    get_laplace_random_effects(dm::DataModel, res::FitResult; constants_re = NamedTuple(),
                               flatten = true, include_constants = true)
    get_laplace_random_effects(res::FitResult; kwargs...)

Empirical-Bayes (EB) random-effects estimates from a `Laplace` or `GHQuadrature` fit,
returned as a `NamedTuple` of `DataFrame`s (one per random effect). The single-argument
form uses the `DataModel` stored on `res`.

# Keyword Arguments
- `constants_re::NamedTuple = NamedTuple()`: random-effect levels held fixed during fitting.
- `flatten::Bool = true`: split vector-valued random effects into one row per component.
- `include_constants::Bool = true`: include levels fixed via `constants_re` in the output.
"""
function get_laplace_random_effects(dm::DataModel,
        res::FitResult;
        constants_re::NamedTuple = NamedTuple(),
        flatten::Bool = true,
        include_constants::Bool = true)
    (res.result isa LaplaceResult || res.result isa GHQuadratureResult) ||
        error("Laplace-style random-effects accessor requires a Laplace or GHQuadrature fit result.")
    constants_re = _res_constants_re(res, constants_re)
    re_names = get_re_names(dm.model.random.random)
    isempty(re_names) && return NamedTuple()
    _, batch_infos, _ = _build_laplace_batch_infos(dm, constants_re)
    bstars = res.result.eb_modes
    return _re_dataframes_from_bstars(dm, batch_infos, bstars; constants_re = constants_re,
        flatten = flatten, include_constants = include_constants)
end

function get_laplace_random_effects(res::FitResult;
        constants_re::NamedTuple = NamedTuple(),
        flatten::Bool = true,
        include_constants::Bool = true)
    dm = res.data_model
    dm === nothing &&
        error("This fit result does not store a DataModel; call get_laplace_random_effects(dm, res) instead.")
    return get_laplace_random_effects(
        dm, res; constants_re = constants_re, flatten = flatten,
        include_constants = include_constants)
end

function _eta_from_eb(dm::DataModel,
        batch_infos::Vector,
        bstars::Vector,
        const_cache,
        θ::ComponentArray)
    if const_cache isa NamedTuple
        const_cache = _build_constants_cache(dm, const_cache)
    end
    η_vec = Vector{ComponentArray}(undef, length(dm.individuals))
    for (bi, info) in enumerate(batch_infos)
        b = bstars[bi]
        for i in info.inds
            nt = _build_eta_ind(dm, i, info, b, const_cache, θ)
            η_vec[i] = ComponentArray(nt)
        end
    end
    return η_vec
end

function _compute_mcmc_candidates(dm::DataModel,
        batch_infos::Vector,
        const_cache,
        θu::ComponentArray,
        ll_cache,
        sampler,
        n_samples::Int,
        n_adapt::Int,
        rng::AbstractRNG,
        active_batch_indices = nothing)
    re_names = get_re_names(dm.model.random.random)
    ll_local = ll_cache isa AbstractVector ? ll_cache[1] : ll_cache
    turing_kwargs = (n_samples = n_samples, n_adapt = n_adapt, progress = false)
    active_set = active_batch_indices === nothing ? nothing : Set(active_batch_indices)
    return map(enumerate(batch_infos)) do (bi, info)
        if info.n_b == 0 || (active_set !== nothing && !(bi ∈ active_set))
            return Matrix{Float64}(undef, 0, 0)
        end
        samples, _, _ = _mcem_sample_batch(dm, info, θu, const_cache, ll_local,
            sampler, turing_kwargs, rng,
            re_names, false, nothing)
        samples
    end
end

function _compute_bstars(dm::DataModel,
        θu::ComponentArray,
        constants_re::NamedTuple,
        ll_cache,
        ebe::EBEOptions,
        rng::AbstractRNG;
        rescue::Union{Nothing, EBERescueOptions} = nothing,
        progress::Bool = false,
        progress_desc::AbstractString = "Final EBE",
        mcmc_candidates_by_batch::Union{Nothing, Vector} = nothing,
        active_batch_indices::Union{Nothing, AbstractVector{Int}} = nothing)
    ebe = _resolve_ebe_options(ebe, dm)
    rescue = _resolve_ebe_rescue_options(rescue, ebe.grad_tol, dm)
    _, batch_infos, const_cache = _build_laplace_batch_infos(dm, constants_re)
    T = eltype(θu)
    n_batches = length(batch_infos)
    ebe_cache = _init_laplace_eval_cache(n_batches, T)
    ll_cache_local = ll_cache isa Vector ? ll_cache[1] : ll_cache
    ebe_serialization = ll_cache isa Vector ? SciMLBase.EnsembleThreads() :
                        SciMLBase.EnsembleSerial()
    active_set = active_batch_indices === nothing ? nothing : Set(active_batch_indices)

    function _batch_grad_norms()
        norms = Vector{Float64}(undef, n_batches)
        for (bi, info) in enumerate(batch_infos)
            if info.n_b == 0 || (active_set !== nothing && !(bi ∈ active_set))
                norms[bi] = 0.0
                continue
            end
            b = ebe_cache.bstar_cache.b_star[bi]
            if isempty(b)
                norms[bi] = Inf
                continue
            end
            g, _ = _laplace_gradb_cached!(
                ebe_cache, bi, dm, info, θu, const_cache, ll_cache_local, b)
            gn = maximum(abs, g)
            norms[bi] = isfinite(gn) ? Float64(gn) : Inf
        end
        return norms
    end

    bstars = _laplace_get_bstar!(ebe_cache, dm, batch_infos, θu, const_cache, ll_cache;
        optimizer = ebe.optimizer,
        optim_kwargs = ebe.optim_kwargs,
        adtype = ebe.adtype,
        grad_tol = ebe.grad_tol,
        multistart = LaplaceMultistartOptions(
            ebe.multistart_n, ebe.multistart_k, ebe.grad_tol, ebe.max_rounds, ebe.sampling),
        rng = rng,
        serialization = ebe_serialization,
        progress = progress,
        progress_desc = "$(progress_desc) (pass 1)",
        mcmc_candidates_by_batch = mcmc_candidates_by_batch,
        active_batches = active_set)

    if rescue !== nothing && rescue.enabled && n_batches > 0
        norms_before = _batch_grad_norms()
        rescue_tol = Float64(rescue.grad_tol)
        if any(>(rescue_tol), norms_before)
            bstars = _laplace_get_bstar!(
                ebe_cache, dm, batch_infos, θu, const_cache, ll_cache;
                optimizer = ebe.optimizer,
                optim_kwargs = ebe.optim_kwargs,
                adtype = ebe.adtype,
                grad_tol = ebe.grad_tol,
                theta_tol = -one(eltype(θu)),
                multistart = LaplaceMultistartOptions(
                    rescue.multistart_n, rescue.multistart_k,
                    rescue.grad_tol, rescue.max_rounds, rescue.sampling),
                rng = rng,
                serialization = ebe_serialization,
                progress = progress,
                progress_desc = "$(progress_desc) (rescue)",
                active_batches = active_set)
            norms_after = _batch_grad_norms()
            if any(>(rescue_tol), norms_after)
                @warn "Final EBE rescue multistart did not satisfy the EBE gradient tolerance for all batches." max_grad_before=maximum(norms_before) max_grad_after=maximum(norms_after) grad_tol=rescue_tol multistart_n=rescue.multistart_n multistart_k=rescue.multistart_k max_rounds=rescue.max_rounds
            end
        end
    end

    return bstars, batch_infos
end

"""
    get_random_effects(dm::DataModel, res::FitResult; constants_re, flatten,
                       include_constants) -> NamedTuple
    get_random_effects(res::FitResult; constants_re, flatten, include_constants) -> NamedTuple

Return empirical Bayes (EB) random-effect estimates as a `NamedTuple` of `DataFrame`s,
one per random effect.

Supported methods: `Laplace`, `MCEM`, `SAEM`, `GHQuadrature`.

# Keyword Arguments
- `constants_re::NamedTuple = NamedTuple()`: fix random effects at given values (natural scale).
- `flatten::Bool = true`: if `true`, expand vector random effects to individual columns.
- `include_constants::Bool = true`: if `true`, include constant random effects in the output.
"""
function get_random_effects(dm::DataModel,
        res::FitResult;
        constants_re::NamedTuple = NamedTuple(),
        flatten::Bool = true,
        include_constants::Bool = true)
    constants_re = _res_constants_re(res, constants_re)
    if res.result isa LaplaceResult || res.result isa GHQuadratureResult
        return get_laplace_random_effects(
            dm, res; constants_re = constants_re, flatten = flatten,
            include_constants = include_constants)
    end
    if res.result isa MCEMResult
        θu = get_params(res; scale = :untransformed)
        bstars = res.result.eb_modes
        if bstars === nothing
            # Only the recompute path needs the fit kwargs and a likelihood cache
            # (an O(rows) build) — stored modes skip both.
            ode_args = _fit_kw(res, :ode_args, ())
            ode_kwargs = _fit_kw(res, :ode_kwargs, NamedTuple())
            serialization = _fit_kw(res, :serialization, EnsembleThreads())
            rng = _fit_kw(res, :rng, Random.default_rng())
            ll_cache = build_ll_cache(dm; ode_args = ode_args, ode_kwargs = ode_kwargs,
                serialization = serialization, force_saveat = true)
            bstars, batch_infos = _compute_bstars(
                dm, θu, constants_re, ll_cache, res.method.ebe, rng;
                rescue = res.method.ebe_rescue)
        else
            _, batch_infos, _ = _build_laplace_batch_infos(dm, constants_re)
        end
        return _re_dataframes_from_bstars(
            dm, batch_infos, bstars; constants_re = constants_re,
            flatten = flatten, include_constants = include_constants)
    end
    if res.result isa SAEMResult
        θu = get_params(res; scale = :untransformed)
        constants_re = _saem_anneal_constants_re(
            dm, θu, _saem_anneal_names(res), constants_re)
        bstars = res.result.eb_modes
        if bstars === nothing
            # Only the recompute path needs the fit kwargs and a likelihood cache
            # (an O(rows) build) — stored modes skip both.
            ode_args = _fit_kw(res, :ode_args, ())
            ode_kwargs = _fit_kw(res, :ode_kwargs, NamedTuple())
            serialization = _fit_kw(res, :serialization, EnsembleThreads())
            rng = _fit_kw(res, :rng, Random.default_rng())
            ll_cache = build_ll_cache(dm; ode_args = ode_args, ode_kwargs = ode_kwargs,
                serialization = serialization, force_saveat = true)
            # When the result was loaded from disk, fall back to defaults from a vanilla SAEM().
            _saem_opts = res.method isa _SavedFittingMethod ? SAEM().saem : res.method.saem
            ebe = EBEOptions(_saem_opts.ebe_optimizer,
                _saem_opts.ebe_optim_kwargs, _saem_opts.ebe_adtype,
                _saem_opts.ebe_grad_tol, _saem_opts.ebe_multistart_n, _saem_opts.ebe_multistart_k,
                _saem_opts.ebe_multistart_max_rounds, _saem_opts.ebe_multistart_sampling)
            bstars, batch_infos = _compute_bstars(dm, θu, constants_re, ll_cache, ebe, rng;
                rescue = _saem_opts.ebe_rescue)
        else
            _, batch_infos, _ = _build_laplace_batch_infos(dm, constants_re)
        end
        return _re_dataframes_from_bstars(
            dm, batch_infos, bstars; constants_re = constants_re,
            flatten = flatten, include_constants = include_constants)
    end
    if res.result isa PooledResult
        return _pooled_re_dataframes(dm, res.result.eta_vec; flatten = flatten)
    end
    error("Random-effects access not supported for this method.")
end

function get_random_effects(res::FitResult;
        constants_re::NamedTuple = NamedTuple(),
        flatten::Bool = true,
        include_constants::Bool = true)
    dm = res.data_model
    dm === nothing &&
        error("This fit result does not store a DataModel; call get_random_effects(dm, res) instead.")
    return get_random_effects(dm, res; constants_re = constants_re, flatten = flatten,
        include_constants = include_constants)
end

"""
    get_random_effects(res::FitResult, re::Symbol; kwargs...) -> Vector

    get_random_effects(dm::DataModel, res::FitResult, re::Symbol; kwargs...) -> Vector

Return the empirical Bayes estimates for a single random effect `re` as a plain vector,
ordered by individual index in `dm`.
"""
function get_random_effects(dm::DataModel, res::FitResult, re::Symbol;
        constants_re::NamedTuple = NamedTuple(),
        include_constants::Bool = true)
    nt = get_random_effects(dm, res; constants_re = constants_re, flatten = true,
        include_constants = include_constants)
    haskey(nt, re) || error("Random effect :$(re) not found. Available: $(keys(nt)).")
    df = getfield(nt, re)
    id_col = dm.config.primary_id
    val_cols = [c for c in propertynames(df) if c != id_col]
    length(val_cols) == 1 ||
        error("Random effect :$(re) is multivariate ($(length(val_cols)) components); use get_random_effects(res) to access the full DataFrame.")
    val_col = val_cols[1]
    id_order = [dm.df[dm.row_groups.obs_rows[i][1], id_col]
                for i in 1:length(dm.individuals)]
    id_to_val = Dict(row[id_col] => row[val_col] for row in eachrow(df))
    return [id_to_val[id] for id in id_order]
end

function get_random_effects(res::FitResult, re::Symbol;
        constants_re::NamedTuple = NamedTuple(),
        include_constants::Bool = true)
    dm = res.data_model
    dm === nothing &&
        error("This fit result does not store a DataModel; call get_random_effects(dm, res, re) instead.")
    return get_random_effects(
        dm, res, re; constants_re = constants_re, include_constants = include_constants)
end

function _resolve_bstars_for_re(dm::DataModel, res::FitResult, constants_re::NamedTuple)
    if res.result isa LaplaceResult || res.result isa GHQuadratureResult
        θu = get_params(res; scale = :untransformed)
        ode_args = _fit_kw(res, :ode_args, ())
        ode_kwargs = _fit_kw(res, :ode_kwargs, NamedTuple())
        serialization = _fit_kw(res, :serialization, EnsembleThreads())
        ll_cache = build_ll_cache(dm; ode_args = ode_args, ode_kwargs = ode_kwargs,
            serialization = serialization, force_saveat = true)
        _, batch_infos, const_cache = _build_laplace_batch_infos(dm, constants_re)
        return res.result.eb_modes, batch_infos, θu, const_cache, ll_cache, constants_re
    end
    if res.result isa MCEMResult
        θu = get_params(res; scale = :untransformed)
        ode_args = _fit_kw(res, :ode_args, ())
        ode_kwargs = _fit_kw(res, :ode_kwargs, NamedTuple())
        serialization = _fit_kw(res, :serialization, EnsembleThreads())
        rng = _fit_kw(res, :rng, Random.default_rng())
        ll_cache = build_ll_cache(dm; ode_args = ode_args, ode_kwargs = ode_kwargs,
            serialization = serialization, force_saveat = true)
        bstars = res.result.eb_modes
        if bstars === nothing
            bstars, batch_infos = _compute_bstars(
                dm, θu, constants_re, ll_cache, res.method.ebe, rng;
                rescue = res.method.ebe_rescue)
            _, _, const_cache = _build_laplace_batch_infos(dm, constants_re)
        else
            _, batch_infos, const_cache = _build_laplace_batch_infos(dm, constants_re)
        end
        return bstars, batch_infos, θu, const_cache, ll_cache, constants_re
    end
    if res.result isa SAEMResult
        θu = get_params(res; scale = :untransformed)
        constants_re = _saem_anneal_constants_re(
            dm, θu, _saem_anneal_names(res), constants_re)
        ode_args = _fit_kw(res, :ode_args, ())
        ode_kwargs = _fit_kw(res, :ode_kwargs, NamedTuple())
        serialization = _fit_kw(res, :serialization, EnsembleThreads())
        rng = _fit_kw(res, :rng, Random.default_rng())
        ll_cache = build_ll_cache(dm; ode_args = ode_args, ode_kwargs = ode_kwargs,
            serialization = serialization, force_saveat = true)
        bstars = res.result.eb_modes
        if bstars === nothing
            _saem_opts = res.method isa _SavedFittingMethod ? SAEM().saem : res.method.saem
            ebe = EBEOptions(_saem_opts.ebe_optimizer,
                _saem_opts.ebe_optim_kwargs, _saem_opts.ebe_adtype,
                _saem_opts.ebe_grad_tol, _saem_opts.ebe_multistart_n, _saem_opts.ebe_multistart_k,
                _saem_opts.ebe_multistart_max_rounds, _saem_opts.ebe_multistart_sampling)
            bstars, batch_infos = _compute_bstars(dm, θu, constants_re, ll_cache, ebe, rng;
                rescue = _saem_opts.ebe_rescue)
            _, _, const_cache = _build_laplace_batch_infos(dm, constants_re)
        else
            _, batch_infos, const_cache = _build_laplace_batch_infos(dm, constants_re)
        end
        return bstars, batch_infos, θu, const_cache, ll_cache, constants_re
    end
    error("Random-effects access not supported for this method.")
end

function _bstars_to_re_df(dm::DataModel, batch_infos, bstars_per_sample::Vector,
        constants_re::NamedTuple, flatten::Bool, include_constants::Bool)
    n_samples = length(bstars_per_sample)
    n_samples == 0 && return NamedTuple()
    sample_dfs = Vector{NamedTuple}(undef, n_samples)
    for s in 1:n_samples
        sample_dfs[s] = _re_dataframes_from_bstars(dm, batch_infos, bstars_per_sample[s];
            constants_re = constants_re,
            flatten = flatten,
            include_constants = include_constants)
    end
    isempty(sample_dfs[1]) && return NamedTuple()
    re_keys = keys(sample_dfs[1])
    out_pairs = Pair{Symbol, Any}[]
    for k in re_keys
        df_list = DataFrame[]
        for s in 1:n_samples
            df = copy(sample_dfs[s][k])
            insertcols!(df, 1, :sample => fill(s, nrow(df)))
            push!(df_list, df)
        end
        push!(out_pairs, k => vcat(df_list...))
    end
    return NamedTuple(out_pairs)
end

# Laplace-approximation path: bstars come from the EB modes, conditional
# covariance is (-H)^{-1}. Used for Laplace, GHQuadrature.
# Returns raw bstars_per_sample (Vector{Vector{Any}}) without the DataFrame conversion.
# Used directly by fit_cv for CV evaluation.
function _sample_laplace_bstars_raw(dm::DataModel, batch_infos, bstars, θu, const_cache,
        ll_cache_local;
        n_samples::Int, rng::AbstractRNG, jitter::Real = 1e-8)
    n_batches = length(batch_infos)
    chols = Vector{Any}(undef, n_batches)
    for (bi, info) in enumerate(batch_infos)
        if info.n_b == 0
            chols[bi] = nothing
            continue
        end
        H = _laplace_hessian_b(dm, info, θu, bstars[bi], const_cache, ll_cache_local,
            nothing, bi; ctx = "sample_random_effects")
        chol, _ = _laplace_cholesky_negH(H; jitter = jitter, max_tries = 8,
            growth = 10.0, adaptive = true,
            scale_factor = 1e-6)
        (chol === nothing || chol.info != 0) &&
            error("Failed to compute conditional covariance for batch $bi: " *
                  "negative Hessian is not positive definite even with jitter.")
        chols[bi] = chol
    end
    bstars_per_sample = Vector{Vector{Any}}(undef, n_samples)
    for s in 1:n_samples
        sampled = Vector{Any}(undef, n_batches)
        for bi in 1:n_batches
            info = batch_infos[bi]
            b0 = bstars[bi]
            if info.n_b == 0
                sampled[bi] = b0
                continue
            end
            z = randn(rng, eltype(b0), info.n_b)
            sampled[bi] = b0 .+ (chols[bi].U \ z)
        end
        bstars_per_sample[s] = sampled
    end
    return bstars_per_sample
end

function _sample_re_laplace_path(dm::DataModel, res::FitResult, constants_re::NamedTuple,
        bstars, batch_infos, θu, const_cache, ll_cache;
        n_samples::Int, rng::AbstractRNG, flatten::Bool,
        include_constants::Bool, jitter::Real)
    ll_cache_local = ll_cache isa Vector ? ll_cache[1] : ll_cache
    bstars_per_sample = _sample_laplace_bstars_raw(
        dm, batch_infos, bstars, θu, const_cache,
        ll_cache_local;
        n_samples = n_samples, rng = rng, jitter = jitter)
    return _bstars_to_re_df(dm, batch_infos, bstars_per_sample,
        constants_re, flatten, include_constants)
end

# MCMC path: draw n_samples from the exact conditional p(b | y, θ̂) using
# the same MCMC kernel that drives the E-step (Turing sampler for MCEM,
# SaemixMH or Turing for SAEM). Each call to _mcem_sample_batch advances
# the chain by one sweep and we record the state after every sweep.
# Returns raw bstars_per_sample (Vector{Vector{Any}}) without the DataFrame conversion.
# Used directly by fit_cv for CV evaluation.
function _sample_mcmc_bstars_raw(dm::DataModel, batch_infos, bstars, θu, const_cache,
        ll_cache_local, re_names, sampler, base_turing_kwargs;
        n_samples::Int, n_adapt::Int, rng::AbstractRNG,
        warm_start::Bool)
    n_batches = length(batch_infos)
    tkwargs = merge(base_turing_kwargs, (n_samples = 1, n_adapt = n_adapt))
    haskey(tkwargs, :progress) || (tkwargs = merge(tkwargs, (progress = false,)))
    haskey(tkwargs, :verbose) || (tkwargs = merge(tkwargs, (verbose = false,)))
    batch_rngs = _spawn_child_rngs(rng, n_batches)
    last_params = Vector{Any}(undef, n_batches)
    for bi in 1:n_batches
        info = batch_infos[bi]
        last_params[bi] = info.n_b == 0 ? nothing :
                          _b_to_last_params(bstars[bi], info, re_names)
    end
    bstars_per_sample = Vector{Vector{Any}}(undef, n_samples)
    tkwargs_noadapt = merge(tkwargs, (n_adapt = 0,))
    for s in 1:n_samples
        sampled = Vector{Any}(undef, n_batches)
        sweep_kwargs = s == 1 ? tkwargs : tkwargs_noadapt
        for bi in 1:n_batches
            info = batch_infos[bi]
            if info.n_b == 0
                sampled[bi] = bstars[bi]
                continue
            end
            samples_mat, lastp, lastb = _mcem_sample_batch(
                dm, info, θu, const_cache, ll_cache_local, sampler, sweep_kwargs,
                batch_rngs[bi], re_names, warm_start, last_params[bi])
            last_params[bi] = lastp
            sampled[bi] = size(samples_mat, 2) > 0 ? samples_mat[:, end] : copy(lastb)
        end
        bstars_per_sample[s] = sampled
    end
    return bstars_per_sample
end

function _sample_re_mcmc_path(dm::DataModel, res::FitResult, constants_re::NamedTuple,
        bstars, batch_infos, θu, const_cache, ll_cache,
        sampler, base_turing_kwargs;
        n_samples::Int, n_adapt::Int, rng::AbstractRNG,
        warm_start::Bool, flatten::Bool, include_constants::Bool)
    ll_cache_local = ll_cache isa Vector ? ll_cache[1] : ll_cache
    re_names = get_re_names(dm.model.random.random)
    bstars_per_sample = _sample_mcmc_bstars_raw(dm, batch_infos, bstars, θu, const_cache,
        ll_cache_local, re_names, sampler,
        base_turing_kwargs;
        n_samples = n_samples, n_adapt = n_adapt,
        rng = rng, warm_start = warm_start)
    return _bstars_to_re_df(dm, batch_infos, bstars_per_sample,
        constants_re, flatten, include_constants)
end

"""
    sample_random_effects(dm::DataModel, res::FitResult; n_samples, rng,
                          constants_re, flatten, include_constants,
                          jitter, n_adapt, warm_start) -> NamedTuple
    sample_random_effects(res::FitResult; ...) -> NamedTuple

Draw samples from the conditional posterior of the random effects
``p(b_i \\mid y_i, \\hat{\\theta})`` for each individual.

The sampling mechanism matches the one the fitting method itself uses for the
conditional distribution, so the draws are consistent with the E-step:

- `Laplace`, `GHQuadrature`: Gaussian Laplace
  approximation centered at the EBE mode ``b^\\star`` with covariance
  ``(-H)^{-1}``, where ``H`` is the Hessian of the joint log-density at ``b^\\star``.
- `MCEM`, `SAEM`: MCMC samples drawn from the exact conditional with the same
  sampler used in the E-step (Turing sampler for MCEM, `SaemixMH` or Turing
  sampler for SAEM), seeded at the EBE mode.

For batched random effects (e.g. blockwise MCEM/SAEM), the batch is sampled
jointly and split per individual.

# Keyword Arguments
- `n_samples::Int = 100`: number of conditional posterior draws per individual.
- `rng::AbstractRNG = Random.default_rng()`: random source for the draws.
- `constants_re::NamedTuple = NamedTuple()`: fix random effects at given values.
- `flatten::Bool = true`: expand vector random effects to individual columns.
- `include_constants::Bool = true`: include constant random effects in the output.
- `jitter::Real = 1e-8`: initial Cholesky jitter (Laplace path only).
- `n_adapt::Int = 200`: MCMC adaptation steps for the first sweep (MCMC path only).
- `warm_start::Bool = true`: seed the MCMC chain at the EB mode (MCMC path only).
- `sampler = nothing`: override the MCMC sampler used for MCEM/SAEM. Defaults to
  the sampler stored on the fit method, or to `SaemixMH()` when the result was
  loaded from disk (where the original sampler is not serialized).
- `turing_kwargs::NamedTuple = NamedTuple()`: extra kwargs passed through to the
  MCMC sampler. Merged on top of the method's stored `turing_kwargs` when those
  are available.

The returned `NamedTuple` mirrors [`get_random_effects`](@ref), with one extra
leading `:sample` integer column (1..`n_samples`) so each individual appears
`n_samples` times per `DataFrame`.
"""
function sample_random_effects(dm::DataModel,
        res::FitResult;
        n_samples::Int = 100,
        rng::AbstractRNG = Random.default_rng(),
        constants_re::NamedTuple = NamedTuple(),
        flatten::Bool = true,
        include_constants::Bool = true,
        jitter::Real = 1e-8,
        n_adapt::Int = 200,
        warm_start::Bool = true,
        sampler = nothing,
        turing_kwargs::NamedTuple = NamedTuple())
    n_samples >= 1 || error("n_samples must be >= 1.")
    constants_re = _res_constants_re(res, constants_re)

    bstars, batch_infos, θu, const_cache, ll_cache, constants_re = _resolve_bstars_for_re(
        dm, res, constants_re)
    isempty(batch_infos) && return NamedTuple()

    if res.result isa LaplaceResult || res.result isa GHQuadratureResult
        return _sample_re_laplace_path(dm, res, constants_re,
            bstars, batch_infos, θu, const_cache, ll_cache;
            n_samples = n_samples, rng = rng,
            flatten = flatten,
            include_constants = include_constants,
            jitter = jitter)
    end

    # MCMC path for MCEM / SAEM. Prefer the sampler stored on the method;
    # fall back to SaemixMH() when the result was loaded from disk or the
    # method has no usable MCMC sampler (e.g. pure-IS MCEM).
    method_sampler, method_tkwargs = if res.method isa _SavedFittingMethod
        (nothing, NamedTuple())
    elseif res.result isa MCEMResult
        es = _mcmc_e_step(res.method.e_step)
        es === nothing ? (nothing, NamedTuple()) : (es.sampler, es.turing_kwargs)
    elseif res.result isa SAEMResult
        (res.method.saem.sampler, res.method.saem.turing_kwargs)
    else
        error("Random-effects sampling not supported for this method.")
    end

    final_sampler = sampler !== nothing ? sampler :
                    (method_sampler !== nothing ? method_sampler : SaemixMH())
    base_tkwargs = merge(method_tkwargs, turing_kwargs)

    return _sample_re_mcmc_path(dm, res, constants_re,
        bstars, batch_infos, θu, const_cache, ll_cache,
        final_sampler, base_tkwargs;
        n_samples = n_samples, n_adapt = n_adapt, rng = rng,
        warm_start = warm_start, flatten = flatten,
        include_constants = include_constants)
end

function sample_random_effects(res::FitResult;
        n_samples::Int = 100,
        rng::AbstractRNG = Random.default_rng(),
        constants_re::NamedTuple = NamedTuple(),
        flatten::Bool = true,
        include_constants::Bool = true,
        jitter::Real = 1e-8,
        n_adapt::Int = 200,
        warm_start::Bool = true,
        sampler = nothing,
        turing_kwargs::NamedTuple = NamedTuple())
    dm = res.data_model
    dm === nothing &&
        error("This fit result does not store a DataModel; call sample_random_effects(dm, res) instead.")
    return sample_random_effects(dm, res; n_samples = n_samples, rng = rng,
        constants_re = constants_re, flatten = flatten,
        include_constants = include_constants,
        jitter = jitter, n_adapt = n_adapt, warm_start = warm_start,
        sampler = sampler, turing_kwargs = turing_kwargs)
end

"""
    reestimate_ebes(dm::DataModel, res::FitResult; kwargs...) -> FitResult
    reestimate_ebes(res::FitResult; kwargs...) -> FitResult

Re-estimate empirical Bayes estimates (EBEs) from a fitted model, ignoring any EBEs
stored in the result. Returns a new `FitResult` with `eb_modes` replaced by the
freshly-optimized modes. Use `get_random_effects` on the returned result to obtain the
corresponding `NamedTuple` of `DataFrame`s.

Supported methods: `Laplace`, `MCEM`, `SAEM`.

# Keyword Arguments
- `ebe_optimizer`: inner optimizer for EBE mode-finding (default: `LBFGS` with backtracking).
- `ebe_optim_kwargs::NamedTuple`: extra kwargs forwarded to `Optimization.solve`.
- `ebe_adtype`: automatic differentiation type (default: `AutoForwardDiff()`).
- `ebe_grad_tol`: gradient convergence tolerance (default: `:auto`).
- `ebe_multistart_n::Int`: number of multistart candidates (default: `50`).
- `ebe_multistart_k::Int`: k-means clusters for multistart initialization (default: `1`).
- `ebe_multistart_max_rounds::Int`: maximum multistart refinement rounds (default: `5`).
- `ebe_multistart_sampling::Symbol`: candidate sampling strategy — `:lhs`, `:random`, or `:mcmc`
  (default: `:lhs`). With `:mcmc`, `ebe_multistart_n` conditional posterior samples are drawn
  via MCMC and used as starting points; `ebe_mcmc_sampler` and `ebe_mcmc_n_adapt` control
  the sampler and burn-in length.
- `ebe_mcmc_sampler`: MCMC sampler for conditional posterior sampling (default: `SaemixMH()`).
  Only used when `ebe_multistart_sampling=:mcmc`.
- `ebe_mcmc_n_adapt::Int`: number of MCMC burn-in steps before collecting candidates
  (default: `50`). Only used when `ebe_multistart_sampling=:mcmc`.
- `ebe_rescue_on_high_grad::Bool`: run a rescue pass (LHS) when any EBE mode has a high
  gradient norm (default: `false`).
- `ebe_rescue_multistart_n::Int`: multistart candidates for the rescue pass (default: `128`).
- `ebe_rescue_multistart_k::Int`: k-means clusters for rescue (default: `32`).
- `ebe_rescue_max_rounds::Int`: refinement rounds for rescue (default: `8`).
- `ebe_rescue_grad_tol`: gradient tolerance for the rescue pass (default: matches `ebe_grad_tol`).
- `ebe_rescue_multistart_sampling::Symbol`: sampling for rescue (default: `:lhs`).
- `constants_re::NamedTuple`: fix specific RE levels at given values (natural scale).
- `individuals`: `nothing` (all) or a vector of primary IDs. When provided, only batches
  containing at least one listed individual are re-optimized; the remaining batches retain
  the `eb_modes` already stored in `res`. Co-batch individuals are always re-optimized.
- `ode_args::Tuple`, `ode_kwargs::NamedTuple`: forwarded to the ODE solver.
- `rng::AbstractRNG`: random number generator.
- `progress::Bool`: show progress bar (default: `false`).
"""
function reestimate_ebes(dm::DataModel,
        res::FitResult;
        ebe_optimizer = OptimizationOptimJL.LBFGS(linesearch = LineSearches.BackTracking(maxstep = 1.0)),
        ebe_optim_kwargs::NamedTuple = NamedTuple(),
        ebe_adtype = Optimization.AutoForwardDiff(),
        ebe_grad_tol = :auto,
        ebe_multistart_n::Int = 50,
        ebe_multistart_k::Int = 1,
        ebe_multistart_max_rounds::Int = 5,
        ebe_multistart_sampling::Symbol = :lhs,
        ebe_mcmc_sampler = SaemixMH(),
        ebe_mcmc_n_adapt::Int = 50,
        ebe_rescue_on_high_grad::Bool = false,
        ebe_rescue_multistart_n::Int = 128,
        ebe_rescue_multistart_k::Int = 32,
        ebe_rescue_max_rounds::Int = 8,
        ebe_rescue_grad_tol = ebe_grad_tol,
        ebe_rescue_multistart_sampling::Symbol = :lhs,
        constants_re::NamedTuple = NamedTuple(),
        individuals = nothing,
        ode_args::Tuple = (),
        ode_kwargs::NamedTuple = NamedTuple(),
        rng::AbstractRNG = Random.default_rng(),
        progress::Bool = false)
    supported = res.result isa LaplaceResult ||
                res.result isa MCEMResult || res.result isa SAEMResult
    supported || error("reestimate_ebes is not supported for this fitting method.")
    sampling_sym = ebe_multistart_sampling == :mcmc ? :lhs : ebe_multistart_sampling
    ebe = EBEOptions(ebe_optimizer, ebe_optim_kwargs, ebe_adtype, ebe_grad_tol,
        ebe_multistart_n, ebe_multistart_k, ebe_multistart_max_rounds,
        sampling_sym)
    ebe_rescue = EBERescueOptions(ebe_rescue_on_high_grad, ebe_rescue_multistart_n,
        ebe_rescue_multistart_k, ebe_rescue_max_rounds,
        ebe_rescue_grad_tol, ebe_rescue_multistart_sampling)
    θu = get_params(res; scale = :untransformed)
    constants_re = _res_constants_re(res, constants_re)
    if res.result isa SAEMResult
        constants_re = _saem_anneal_constants_re(
            dm, θu, _saem_anneal_names(res), constants_re)
    end
    ll_cache = build_ll_cache(
        dm; ode_args = ode_args, ode_kwargs = ode_kwargs, force_saveat = true)
    # Compute batch structure once; derive active batch set if individuals are specified.
    _, batch_infos_pre, const_cache_pre = _build_laplace_batch_infos(dm, constants_re)
    active_batch_indices = if individuals !== nothing
        ind_indices = Set(dm.id_index[id] for id in individuals if haskey(dm.id_index, id))
        findall(bi -> any(i ∈ ind_indices for i in batch_infos_pre[bi].inds),
            eachindex(batch_infos_pre))
    else
        nothing
    end
    mcmc_candidates = nothing
    if ebe_multistart_sampling == :mcmc
        mcmc_candidates = _compute_mcmc_candidates(
            dm, batch_infos_pre, const_cache_pre, θu,
            ll_cache, ebe_mcmc_sampler,
            ebe_multistart_n, ebe_mcmc_n_adapt, rng,
            active_batch_indices)
    end
    bstars, batch_infos = _compute_bstars(dm, θu, constants_re, ll_cache, ebe, rng;
        rescue = ebe_rescue, progress = progress,
        mcmc_candidates_by_batch = mcmc_candidates,
        active_batch_indices = active_batch_indices)
    new_eb_modes = if individuals !== nothing
        existing = res.result.eb_modes
        if existing !== nothing && length(existing) == length(bstars)
            active_set = Set(active_batch_indices)
            merged = copy(existing)
            for bi in active_set
                merged[bi] = bstars[bi]
            end
            merged
        else
            bstars
        end
    else
        bstars
    end
    return _with_result(res, _with_eb_modes(res.result, new_eb_modes))
end

function reestimate_ebes(res::FitResult; kwargs...)
    dm = res.data_model
    dm === nothing &&
        error("This fit result does not store a DataModel; call reestimate_ebes(dm, res) instead.")
    return reestimate_ebes(dm, res; kwargs...)
end

function _with_result(res::FitResult, new_result)
    return FitResult(res.method, new_result, res.summary, res.diagnostics,
        res.data_model, res.fit_args, res.fit_kwargs)
end

"""
    get_loglikelihood(dm::DataModel, res::FitResult; constants_re, ode_args,
                      ode_kwargs, serialization) -> Real
    get_loglikelihood(res::FitResult; constants_re, ode_args, ode_kwargs,
                      serialization) -> Real

Compute the marginal log-likelihood at the estimated parameter values.

For MLE/MAP results, evaluates the population log-likelihood. For Laplace-style
results, evaluates using the EB modes stored in the result.

# Keyword Arguments
- `constants_re::NamedTuple = NamedTuple()`: random effects fixed at given values.
- `ode_args::Tuple = ()`: additional positional arguments for the ODE solver.
- `ode_kwargs::NamedTuple = NamedTuple()`: additional keyword arguments for the ODE solver.
- `serialization = EnsembleThreads()`: parallelisation strategy.
"""
function get_loglikelihood(dm::DataModel,
        res::FitResult;
        constants_re::NamedTuple = NamedTuple(),
        ode_args::Tuple = (),
        ode_kwargs::NamedTuple = NamedTuple(),
        serialization::SciMLBase.EnsembleAlgorithm = EnsembleThreads())
    constants_re = _res_constants_re(res, constants_re)
    θu = get_params(res; scale = :untransformed)
    if res.result isa MLEResult || res.result isa MAPResult
        return loglikelihood(dm, θu, ComponentArray(); ode_args = ode_args,
            ode_kwargs = ode_kwargs, serialization = serialization)
    elseif res.result isa LaplaceResult
        pairing, batch_infos, const_cache = _build_laplace_batch_infos(dm, constants_re)
        bstars = res.result.eb_modes
        length(bstars) == length(batch_infos) ||
            error("Laplace-style EB modes do not match number of batches.")
        η_vec = _eta_from_eb(dm, batch_infos, bstars, const_cache, θu)
        return loglikelihood(dm, θu, η_vec; ode_args = ode_args,
            ode_kwargs = ode_kwargs, serialization = serialization)
    elseif res.result isa GHQuadratureResult
        # Re-evaluate the sparse-grid marginal log-likelihood at the estimated θ.
        level = res.method.level
        ll_cache = build_ll_cache(
            dm; ode_args = ode_args, ode_kwargs = ode_kwargs, force_saveat = true)
        _, batch_infos, const_cache = _build_laplace_batch_infos(dm, constants_re)
        θu_re = _symmetrize_psd_params(θu, dm.model.fixed.fixed)
        total = 0.0
        for info in batch_infos
            bll = _ghq_batch_ll(dm, info, θu_re, const_cache, ll_cache, level)
            bll == -Inf && return -Inf
            total += bll
        end
        return total
    elseif res.result isa PooledResult
        return loglikelihood(dm, θu, res.result.eta_vec; ode_args = ode_args,
            ode_kwargs = ode_kwargs, serialization = serialization)
    else
        error("loglikelihood accessor not supported for this method.")
    end
end

function get_loglikelihood(res::FitResult;
        constants_re::NamedTuple = NamedTuple(),
        ode_args::Tuple = (),
        ode_kwargs::NamedTuple = NamedTuple(),
        serialization::SciMLBase.EnsembleAlgorithm = EnsembleThreads())
    dm = res.data_model
    dm === nothing &&
        error("This fit result does not store a DataModel; call get_loglikelihood(dm, res) instead.")
    return get_loglikelihood(dm, res; constants_re = constants_re, ode_args = ode_args,
        ode_kwargs = ode_kwargs, serialization = serialization)
end

function _default_ebe_options()
    EBEOptions(
        OptimizationOptimJL.LBFGS(linesearch = LineSearches.BackTracking(maxstep = 1.0)),
        NamedTuple(),
        Optimization.AutoForwardDiff(),
        :auto,
        50,
        10,
        1,
        :lhs
    )
end

"""
    get_loglikelihood_quadrature(dm, res; level=3, constants_re, ode_args, ode_kwargs,
                                  serialization, ebe_options, rng, jitter,
                                  mc_integrator, fallback) -> Float64

Compute the marginal log-likelihood using **Adaptive Gauss-Hermite Quadrature** (AGHQ),
with optional Monte Carlo sampling as the primary method or as a fallback.

Unlike `get_loglikelihood`, which plugs in the EBE point estimate for Laplace/SAEM/MCEM
methods, this function integrates over each batch's random effects.

The integration measure for AGHQ is

    b = b* + S * z,  z ~ N(0, I)

where **b*** is the empirical-Bayes mode and **S = chol(-H)^{-1}** is derived from
the Hessian H of log p(b | y, θ) at b*. The log-correction

    logcorrection(z) = log p(b* + Sz | θ) + log|det(S)| + ½‖z‖² + ½d·log(2π)

accounts for the prior, Jacobian, and Gaussian quadrature measure.

# Supported methods
Laplace, SAEM, MCEM, GHQuadrature.

# Not supported
- **MCMC**: raises an error.
- **MLE/MAP**: raises an error directing the user to `get_loglikelihood`, which already
  returns the exact marginal log-likelihood for these methods (no random effects).

# Keyword Arguments
- `level`: Smolyak accuracy level (default 3). Same as in `GHQuadrature`.
- `constants_re`: fixes specific RE levels on the natural scale.
- `ode_args`, `ode_kwargs`: forwarded to the ODE solver.
- `serialization`: `EnsembleThreads()` (default) or `EnsembleSerial()`.
- `ebe_options::Union{Nothing, EBEOptions}`: EBE optimizer options used when stored
  modes are unavailable. `nothing` uses the same defaults as `Laplace()`.
- `rng`: random number generator for EBE multistart (if needed).
- `jitter`: initial jitter for Cholesky of negative Hessian (default 1e-6).
- `mc_integrator::Union{Nothing, MCIntegrator}`: if not `nothing`, use Monte Carlo
  sampling for **all** batches instead of AGHQ. See [`MCIntegrator`](@ref).
- `fallback::Union{Nothing, MCIntegrator}`: what to do when the Cholesky of `-H` fails
  for a batch. `nothing` raises an error (old behavior). An `MCIntegrator` falls back to
  sampling for that batch and issues a warning. Default: `MCIntegrator()` (Turing-based sampling,
  1000 samples).
"""
function get_loglikelihood_quadrature(dm::DataModel,
        res::FitResult;
        level::Union{Int, NamedTuple} = 3,
        constants_re::NamedTuple = NamedTuple(),
        ode_args::Tuple = (),
        ode_kwargs::NamedTuple = NamedTuple(),
        serialization::SciMLBase.EnsembleAlgorithm = EnsembleThreads(),
        ebe_options::Union{Nothing, EBEOptions} = nothing,
        seed::Int = 0,
        rng::AbstractRNG = Random.Xoshiro(seed),
        jitter::Float64 = 1e-6,
        mc_integrator::Union{Nothing, MCIntegrator} = nothing,
        fallback::Union{Nothing, MCIntegrator} = MCIntegrator())
    if res.result isa MCMCResult
        error("get_loglikelihood_quadrature: MCMC results are not supported.")
    end
    if res.result isa MLEResult || res.result isa MAPResult
        error("get_loglikelihood_quadrature: MLE/MAP models have no random effects. " *
              "Use get_loglikelihood instead, which already returns the exact marginal log-likelihood.")
    end
    if res.result isa PooledResult
        error("get_loglikelihood_quadrature: Pooled/PooledMap results have fixed RE. " *
              "Use get_loglikelihood instead.")
    end

    constants_re = _res_constants_re(res, constants_re)
    θu = get_params(res; scale = :untransformed)
    # Upcast to Float64 if needed (SAEM stores Float32; Hessian computation requires Float64)
    θu = eltype(θu) === Float64 ? θu : ComponentArray(Float64.(θu), getaxes(θu))
    θu_re = _symmetrize_psd_params(θu, dm.model.fixed.fixed)
    # For SAEM with anneal_to_fixed, rebuild the annealed constants_re from the final θ
    # so that _build_laplace_batch_infos sees the correct n_b (matching stored eb_modes).
    if res.result isa SAEMResult &&
       hasproperty(res.result.notes, :anneal_to_fixed) &&
       !isempty(res.result.notes.anneal_to_fixed)
        constants_re = _saem_anneal_constants_re(
            dm, θu_re, res.result.notes.anneal_to_fixed,
            constants_re)
    end

    _, batch_infos, const_cache = _build_laplace_batch_infos(dm, constants_re)
    ll_cache = build_ll_cache(
        dm; ode_args = ode_args, ode_kwargs = ode_kwargs, force_saveat = true)

    # Resolve EBE modes: use stored ones if available and matching, else compute.
    # Not needed when mc_integrator is set (pure MC path skips EBE/Hessian entirely).
    bstars = if mc_integrator === nothing
        if hasproperty(res.result, :eb_modes) &&
           res.result.eb_modes !== nothing &&
           length(res.result.eb_modes) == length(batch_infos)
            res.result.eb_modes
        else
            ebe = ebe_options === nothing ? _default_ebe_options() : ebe_options
            bstars_new, _ = _compute_bstars(dm, θu, constants_re, ll_cache, ebe, rng)
            bstars_new
        end
    else
        nothing
    end

    if mc_integrator === nothing && _any_batch_too_large(dm, batch_infos, level, 10_000)
        @warn "get_loglikelihood_quadrature: one or more batches have > 10,000 quadrature nodes. " *
              "Consider reducing `level` or checking your RE batch structure."
    end

    total = 0.0
    for (bi, info) in enumerate(batch_infos)
        if info.n_b == 0
            s = 0.0
            empty_b = Float64[]
            for i in info.inds
                η_i = _build_eta_ind(dm, i, info, empty_b, const_cache, θu_re)
                lli = _loglikelihood_individual(dm, i, θu_re, η_i, ll_cache)
                !isfinite(lli) && return -Inf
                s += lli
            end
            total += s
        else
            bll = if mc_integrator !== nothing
                # MC for all batches: skip AGHQ entirely
                _batch_loglik_from_mc(
                    dm, info, θu_re, const_cache, ll_cache, mc_integrator, rng)
            else
                b_star = bstars[bi]
                re_measure = build_centered_re_measure(
                    b_star, info, bi, θu_re, const_cache, dm, ll_cache;
                    jitter = jitter)
                if re_measure !== nothing
                    sgrid = level isa Int ? get_sparse_grid(info.n_b, level) :
                            _build_anisotropic_batch_grid(dm, info, level)
                    batch_loglik_ghq(
                        dm, info, θu_re, re_measure, sgrid, const_cache, ll_cache)
                elseif fallback !== nothing
                    @warn "get_loglikelihood_quadrature: Cholesky of -H failed for batch $bi " *
                          "(b* may not be a true mode or posterior is near-flat). " *
                          "Falling back to $(fallback.mode) MC sampling with $(fallback.n_samples) samples."
                    _batch_loglik_from_mc(
                        dm, info, θu_re, const_cache, ll_cache, fallback, rng)
                else
                    error("get_loglikelihood_quadrature: Cholesky of -H failed for batch $bi. " *
                          "Pass fallback=MCIntegrator(...) to use sampling as fallback, " *
                          "or increase `jitter`.")
                end
            end
            bll == -Inf && return -Inf
            total += bll
        end
    end
    return total
end

function get_loglikelihood_quadrature(res::FitResult;
        level::Union{Int, NamedTuple} = 3,
        constants_re::NamedTuple = NamedTuple(),
        ode_args::Tuple = (),
        ode_kwargs::NamedTuple = NamedTuple(),
        serialization::SciMLBase.EnsembleAlgorithm = EnsembleThreads(),
        ebe_options::Union{Nothing, EBEOptions} = nothing,
        seed::Int = 0,
        rng::AbstractRNG = Random.Xoshiro(seed),
        jitter::Float64 = 1e-6,
        mc_integrator::Union{Nothing, MCIntegrator} = nothing,
        fallback::Union{Nothing, MCIntegrator} = MCIntegrator())
    dm = res.data_model
    dm === nothing &&
        error("This fit result does not store a DataModel; call get_loglikelihood_quadrature(dm, res) instead.")
    return get_loglikelihood_quadrature(
        dm, res; level = level, constants_re = constants_re,
        ode_args = ode_args, ode_kwargs = ode_kwargs,
        serialization = serialization, ebe_options = ebe_options,
        seed = seed, rng = rng, jitter = jitter,
        mc_integrator = mc_integrator, fallback = fallback)
end

function get_re_covariate_usage(res::FitResult; dm::Union{Nothing, DataModel} = nothing)
    dm === nothing && (dm = res.data_model)
    dm === nothing &&
        error("This fit result does not store a DataModel; pass dm=... to get_re_covariate_usage.")
    return get_re_covariate_usage(dm)
end

"""
    fit_model(dm::DataModel, method::FittingMethod; constants, penalty,
              ode_args, ode_kwargs, serialization, rng,
              theta_0_untransformed, store_data_model) -> FitResult

Fit a model to data using the specified estimation method.

# Arguments
- `dm::DataModel`: the data model.
- `method::FittingMethod`: estimation method (e.g. `MLE()`, `Laplace()`, `MCMC(...)`).

# Keyword Arguments
- `constants::NamedTuple = NamedTuple()`: fix named parameters at given values on the
  natural scale. Fixed parameters are removed from the optimizer state.
- `penalty::NamedTuple = NamedTuple()`: add per-parameter quadratic penalties on the
  natural scale (not available for MCMC).
- `ode_args::Tuple = ()`: extra positional arguments forwarded to the ODE solver.
- `ode_kwargs::NamedTuple = NamedTuple()`: extra keyword arguments forwarded to the ODE solver.
- `serialization = EnsembleThreads()`: parallelisation strategy.
- `rng = Random.default_rng()`: random number generator (used by MCMC/SAEM/MCEM).
- `theta_0_untransformed::Union{Nothing, ComponentArray} = nothing`: custom starting
  point on the natural scale; defaults to the model's declared initial values.
- `store_data_model::Bool = true`: whether to store a reference to `dm` in the result.
- `pooled_init = false`: warm-start the fit from a quick [`Pooled`](@ref) pre-fit.
  `true` runs `Pooled(optim_kwargs=(; maxiters=50))`; alternatively pass your own
  `Pooled`/`PooledMap` instance for full control. The pooled pre-fit starts from
  `theta_0_untransformed` (or the model's initial values) and its estimate becomes the
  starting point of the actual fit. Inside [`Multistart`](@ref) the pre-fit runs once
  per start. Requires a model with random effects; not available when `method` itself
  is `Pooled`/`PooledMap`.
- `fit_options_pooled_init::NamedTuple = NamedTuple()`: extra keyword arguments for the
  pooled pre-fit (same keywords as `fit_model`, e.g. `constants`, `penalty`,
  `serialization`). By default the pre-fit inherits `constants`, `penalty`, `ode_args`,
  `ode_kwargs`, `serialization`, and `rng` from the main call; entries here override.
"""
function fit_model(dm::DataModel, method::FittingMethod, args...;
        store_data_model::Bool = true,
        pooled_init = false,
        fit_options_pooled_init::NamedTuple = NamedTuple(),
        kwargs...)
    if pooled_init === false
        isempty(fit_options_pooled_init) ||
            @warn "fit_options_pooled_init is ignored because pooled_init is false."
        return _fit_model(
            dm, method, args...; store_data_model = store_data_model, kwargs...)
    end
    θ_init = _pooled_init_theta(dm, method, pooled_init, fit_options_pooled_init,
        NamedTuple(kwargs))
    kw = merge(NamedTuple(kwargs), (theta_0_untransformed = θ_init,))
    return _fit_model(dm, method, args...; store_data_model = store_data_model, kw...)
end

# Type-stable varying-covariate row construction. Building rows from
# `Pair{Symbol, Any}` vectors gives every row the abstract type `NamedTuple`, which
# forces dynamic dispatch at each `calculate_formulas_obs` call site (and breaks
# Enzyme reverse mode). Going through the NamedTuple structure keeps row types concrete.
@inline _vary_value_at(v::AbstractVector, idx::Int) = v[idx]
@inline _vary_value_at(v::NamedTuple, idx::Int) = _covariate_vector(map(x -> x[idx], v))

@inline function _vary_row(vary::NamedTuple, dyn::NamedTuple, t_obs, idx::Int)
    t_val = hasproperty(vary, :t) ? vary.t[idx] : t_obs[idx]
    rest = Base.structdiff(vary, NamedTuple{(:t,)})
    vals = map(v -> _vary_value_at(v, idx), rest)
    return merge((t = t_val,), vals, dyn)
end

function _varying_at(dm::DataModel, ind::Individual, idx::Int, t_obs)
    return _vary_row(ind.series.vary, ind.series.dyn, t_obs, idx)
end

@inline function _needs_rowwise_random_effects(
        dm::DataModel, idx::Int; obs_only::Bool = true)
    dm.model.de.de !== nothing && return false
    re_names = get_re_names(dm.model.random.random)
    isempty(re_names) && return false
    info = dm.re_group_info.index_by_individual
    for re in re_names
        re_info = getfield(info, re)
        positions = obs_only ? re_info.unique_pos_obs[idx] : re_info.unique_pos_all[idx]
        isempty(positions) && continue
        first_pos = positions[1]
        @inbounds for k in 2:length(positions)
            positions[k] != first_pos && return true
        end
    end
    return false
end

@inline function _row_random_effects_at(dm::DataModel,
        idx::Int,
        row_idx::Int,
        η_ind::NamedTuple,
        rowwise_re::Bool;
        obs_only::Bool = true)
    return _row_random_effects_at(
        dm, idx, row_idx, ComponentArray(η_ind), rowwise_re; obs_only = obs_only)
end

function _row_random_effects_at(dm::DataModel,
        idx::Int,
        row_idx::Int,
        η_ind::ComponentArray,
        rowwise_re::Bool;
        obs_only::Bool = true)
    rowwise_re || return η_ind
    re_names = get_re_names(dm.model.random.random)
    isempty(re_names) && return η_ind
    ind = dm.individuals[idx]
    info = dm.re_group_info.index_by_individual
    nt_pairs = Pair{Symbol, Any}[]
    for re in re_names
        η_re = getproperty(η_ind, re)
        nlevels = length(getfield(ind.re_groups, re))
        if nlevels <= 1
            push!(nt_pairs, re => η_re)
            continue
        end
        re_info = getfield(info, re)
        positions = obs_only ? re_info.unique_pos_obs[idx] : re_info.unique_pos_all[idx]
        push!(nt_pairs, re => η_re[positions[row_idx]])
    end
    return ComponentArray(NamedTuple(nt_pairs))
end

# ---------------------------------------------------------------------------------
# Per-individual rowwise-η template: the row-η ComponentArray has the SAME axes for
# every row of a given individual (only the selected values differ), so hot row
# loops derive the axes once — via the generic boxing path above on the first
# iterated row — and then fill a fresh flat vector per row. This replaces the
# per-row `Pair{Symbol,Any}[]` + `ComponentArray(NamedTuple)` construction
# (measured ~1.9 KB / 1.8 μs per row) with a single typed-vector fill.
# ---------------------------------------------------------------------------------
struct _RowREFill{A}
    axs::A
    len::Int
end

function _row_re_template(dm::DataModel, idx::Int, first_row::Int,
        η_ind::ComponentArray; obs_only::Bool = true)
    proto = _row_random_effects_at(dm, idx, first_row, η_ind, true; obs_only = obs_only)
    return _RowREFill(getaxes(proto), length(proto))
end

function _row_random_effects_fill(dm::DataModel, idx::Int, row_idx::Int,
        η_ind::ComponentArray, tmpl::_RowREFill; obs_only::Bool = true)
    re_names = get_re_names(dm.model.random.random)
    ind = dm.individuals[idx]
    info = dm.re_group_info.index_by_individual
    vals = Vector{eltype(η_ind)}(undef, tmpl.len)
    pos = 1
    for re in re_names
        η_re = getproperty(η_ind, re)
        nlevels = length(getfield(ind.re_groups, re))
        sel = if nlevels <= 1
            η_re
        else
            re_info = getfield(info, re)
            positions = obs_only ? re_info.unique_pos_obs[idx] :
                        re_info.unique_pos_all[idx]
            η_re[positions[row_idx]]
        end
        if sel isa Number
            vals[pos] = sel
            pos += 1
        else
            for k in eachindex(sel)
                vals[pos] = sel[k]
                pos += 1
            end
        end
    end
    return ComponentArray(vals, tmpl.axs)
end

mutable struct _LLCache{H, M, S, A, O, K, P, V, SA}
    helpers::H
    model_funs::M
    solver_cfg::S
    alg::A
    ode_args::O
    ode_kwargs::K
    prob_templates::P
    vary_cache::V
    saveat_cache::SA
end

# `_is_hmm_dist` (the 7 HMM-family outcome types) is defined in
# distributions/outcomes/_HMMSimulationUtils.jl.
@inline function _row_has_hmm_dist(obs, obs_cols)
    for col in obs_cols
        _is_hmm_dist(getproperty(obs, col)) && return true
    end
    return false
end

@inline function _ll_saveat(cache::_LLCache, idx::Int, ind::Individual)
    cache.saveat_cache === nothing && return ind.saveat
    return cache.saveat_cache[idx]
end

# Per-individual solver options (`saveat`/`save_everystep`/`dense` and `callback`)
# are baked into the cached problem TEMPLATE's kwargs instead of the solve call:
# a rooted object (saveat Vector, callback) inside the solve-kwargs NamedTuple trips
# Enzyme's reverse rule-argument classification, which either asserts
# (`roots_activep != activep`) or — in deeper call contexts — silently corrupts the
# adjoint gradient by ~1%. With the options on the problem, the per-call solve
# kwargs stay scalar-only and the rule classifies correctly. All of these options
# are evaluation-constant per individual, so template caching remains sound; call
# kwargs still take precedence over problem kwargs, so `ode_kwargs` overrides keep
# working (overriding `saveat` through `ode_kwargs` reintroduces the rooted-kwargs
# hazard for Enzyme reverse, but is unchanged for FD/forward use).
#
# These are function barriers for the same reason as before: `ind.saveat` /
# `ind.callbacks` come out of the abstractly-typed `Individual` storage, so the
# kwargs NamedTuples must be built behind a dispatch boundary to stay concrete
# (keeps the Bool method of `_ode_normalize_verbose` statically dead).
@inline function _ll_prob_kwargs(cb, saveat_use)
    base = saveat_use === nothing ? (dense = true,) :
           (saveat = saveat_use, save_everystep = false, dense = false)
    return cb === nothing ? base : merge(base, (callback = cb,))
end

function _ll_build_prob_template(f!_use, u0, tspan, p_flat, cb, saveat_use)
    kw = _ll_prob_kwargs(cb, saveat_use)
    return ODEProblem{true, SciMLBase.FullSpecialize}(f!_use, u0, tspan, p_flat; kw...)
end

function _ll_ode_solve_baked(cache::_LLCache, prob)
    solve_kwargs = _ode_solve_kwargs(
        cache.solver_cfg.kwargs, cache.ode_kwargs, NamedTuple())
    return solve(prob, cache.alg, cache.ode_args...; solve_kwargs...)
end

# Shared DE-solve scaffolding for the per-individual evaluators
# (`_loglikelihood_individual`, `_resid_stats_individual*`, the FOCEI obs
# collectors, and the CV row collector). `pre` is the precomputed preDE
# NamedTuple — row-constant, reused for the compile context and the initial
# state instead of being re-derived inside each call. Returns the sol_accessors
# NamedTuple, or `nothing` when the solve failed.
function _ll_solve_de(dm::DataModel, idx::Int, θ, η_ind, cache::_LLCache, pre)
    model = dm.model
    ind = dm.individuals[idx]
    const_cov = ind.const_cov
    pc = (;
        fixed_effects = θ,
        random_effects = η_ind,
        constant_covariates = const_cov,
        varying_covariates = merge((t = ind.series.vary.t[1],), ind.series.dyn),
        helpers = cache.helpers,
        model_funs = cache.model_funs,
        preDE = pre
    )
    compiled = get_de_compiler(model.de.de)(pc)
    u0 = _initial_state_with_pre(model, θ, η_ind, const_cov, pre)
    cb = nothing
    infusion_rates = nothing
    if ind.callbacks !== nothing
        _apply_initial_events!(u0, ind.callbacks)
        cb = ind.callbacks.callback
        infusion_rates = ind.callbacks.infusion_rates
    end
    # Solve parameters travel as a flat numeric Vector (DERHSFlat adapter):
    # plain-vector `prob.p` is the only carrier Enzyme's reverse adjoint route
    # handles correctly. Same generated RHS kernel — numerically equivalent
    # (ulp-level fp-contraction differences from the changed inlining context
    # can shift adaptive step sequences by ~1e-15; well inside solver tolerance).
    # `funs` (interpolants/model funs/helpers) are evaluation-constant per
    # individual, so caching the adapter inside the problem template is sound.
    layout, plen = _flat_layout(compiled.vars)
    f!_use = _with_infusion(
        DERHSFlat(get_de_f!(model.de.de), layout, compiled.funs), infusion_rates)
    # T must cover the vars eltype too (η/θ can enter the RHS without entering
    # u0) — pack once with the promoted type, reuse for template and remake.
    T = promote_type(eltype(θ), eltype(η_ind), eltype(u0))
    p_flat = _flat_pack(compiled.vars, layout, plen, T)
    prob = cache.prob_templates === nothing ? nothing : cache.prob_templates[idx]
    if prob === nothing
        saveat_use = _ll_saveat(cache, idx, ind)
        prob = _ll_build_prob_template(f!_use, u0, ind.tspan, p_flat, cb, saveat_use)
        if cache.prob_templates !== nothing
            cache.prob_templates[idx] = prob
        end
    end

    u0_T = eltype(u0) === T ? u0 : T.(u0)
    prob = remake(prob; u0 = u0_T, p = p_flat)
    sol = _ll_ode_solve_baked(cache, prob)
    SciMLBase.successful_retcode(sol) || return nothing
    return get_de_accessors_builder(model.de.de)(sol, compiled)
end

function _loglikelihood_individual(dm::DataModel, idx::Int, θ, η_ind, cache::_LLCache)
    model = dm.model
    ind = dm.individuals[idx]
    obs_rows = dm.row_groups.obs_rows[idx]
    const_cov = ind.const_cov
    obs_series = ind.series.obs
    vary_cache = cache.vary_cache === nothing ? nothing : cache.vary_cache[idx]
    if η_ind isa NamedTuple
        η_ind = ComponentArray(η_ind)
    end

    sol_accessors = nothing
    pre = nothing
    if model.de.de !== nothing
        pre = calculate_prede(model, θ, η_ind, const_cov)
        sol_accessors = _ll_solve_de(dm, idx, θ, η_ind, cache, pre)
        sol_accessors === nothing && return -Inf
    end

    # Rowwise-varying random effects (multiple RE levels within one individual;
    # only possible for non-DE models) need a per-row η and keep the original
    # per-row path. The common case hoists all row-constant context (preDE,
    # formula ctx, vary rows) out of the loop and runs the rows behind a function
    # barrier, so every row evaluation is statically dispatched and
    # allocation-free instead of boxing through `Any`-inferred dynamic calls.
    if _needs_rowwise_random_effects(dm, idx; obs_only = true)
        return _loglikelihood_individual_rowwise(dm, idx, θ, η_ind, cache, sol_accessors)
    end
    pre === nothing && (pre = calculate_prede(model, θ, η_ind, const_cov))
    ctx = (; fixed_effects = θ, random_effects = η_ind, prede = pre,
        helpers = cache.helpers, model_funs = cache.model_funs)
    vrows = vary_cache !== nothing ? vary_cache :
            _build_vary_cache_individual(ind.series.vary, ind.series.dyn,
        _get_col(dm.df, dm.config.time_col)[obs_rows],
        length(obs_rows))
    sol_acc = sol_accessors === nothing ? NamedTuple() : sol_accessors
    T_el = promote_type(eltype(θ), eltype(η_ind))
    return _ll_rows_obs(dm, idx, θ, η_ind, cache, sol_accessors,
        model.formulas.obs, ctx, sol_acc, const_cov, obs_series,
        vrows, dm.config.obs_cols, T_el)
end

# Row loop of `_loglikelihood_individual` behind a function barrier: every
# argument arrives concretely typed, so the formulas RGF call, the observation
# field accesses, and the logpdf evaluations all dispatch statically.
# HMM-free fast path. Filtering state created lazily inside this loop would be
# loop-carried (phi nodes reverse-mode Enzyme cannot invert), so rows are processed
# assuming no HMM outcome; the first row that produces one hands all remaining rows
# to `_loglikelihood_rows_hmm`, which allocates its state once up front. Non-HMM
# models never allocate any filtering state.
function _ll_rows_obs(dm::DataModel, idx::Int, θ, η_ind, cache::_LLCache, sol_accessors,
        obs_f::F, ctx::C, sol_acc::SA, const_cov::CC, obs_series::OS,
        vrows::Vector{V}, obs_cols, ::Type{T}) where {F, C, SA, CC, OS, V, T}
    ll = zero(T)
    for i in eachindex(vrows)
        obs = obs_f(ctx, sol_acc, const_cov, vrows[i])
        if _row_has_hmm_dist(obs, obs_cols)
            ll_hmm = _loglikelihood_rows_hmm(dm, idx, θ, η_ind, cache, sol_accessors, i)
            isfinite(ll_hmm) || return T(-Inf)
            return ll + ll_hmm
        end
        for col in obs_cols
            y = getfield(obs_series, col)[i]
            y === missing && continue
            dist = getproperty(obs, col)
            v = _fast_logpdf(dist, y)
            v === nothing && (v = logpdf(dist, y))
            if !isfinite(v)
                return T(-Inf)
            end
            ll += v
        end
    end
    return ll
end

# Original per-row path for individuals whose random effects vary across rows
# (multiple RE levels within one individual; non-DE models only). η must be
# re-selected per row, so the formula context cannot be hoisted.
function _loglikelihood_individual_rowwise(dm::DataModel, idx::Int, θ, η_ind,
        cache::_LLCache, sol_accessors)
    model = dm.model
    ind = dm.individuals[idx]
    obs_rows = dm.row_groups.obs_rows[idx]
    T_el = promote_type(eltype(θ), eltype(η_ind))
    isempty(obs_rows) && return zero(T_el)
    const_cov = ind.const_cov
    obs_series = ind.series.obs
    obs_cols = dm.config.obs_cols
    vary_cache = cache.vary_cache === nothing ? nothing : cache.vary_cache[idx]
    t_obs = vary_cache === nothing ? _get_col(dm.df, dm.config.time_col)[obs_rows] : nothing
    # `_row_re_template` derives its axes from the deliberately-boxing
    # `_row_random_effects_at`, so `row_tmpl` is statically `_RowREFill{Any}`.
    # Pass it into the row loop behind a function barrier (`tmpl::TM` type
    # parameter) so the axes become concrete there: `_row_random_effects_fill`
    # then returns a concrete ComponentVector and the formulas/logpdf calls
    # dispatch statically — mirroring the `_ll_rows_obs` fast path. (Without the
    # barrier every per-row `calculate_formulas_obs`/`getproperty`/`logpdf` boxed
    # through `Any`.)
    row_tmpl = _row_re_template(dm, idx, 1, η_ind; obs_only = true)
    return _ll_rows_obs_rowwise(dm, idx, model, θ, η_ind, cache, sol_accessors,
        const_cov, obs_series, obs_cols, vary_cache, ind, t_obs, row_tmpl, T_el)
end

# Row loop of `_loglikelihood_individual_rowwise` behind a function barrier: with
# `tmpl::TM` arriving as a concrete type parameter the per-row η ComponentVector is
# concrete, so the formulas RGF call, observation field accesses, and logpdf
# evaluations dispatch statically (same role as `_ll_rows_obs`, but with per-row η
# re-selection retained). HMM-free fast path; the first HMM-producing row hands the
# remainder to `_loglikelihood_rows_hmm`.
function _ll_rows_obs_rowwise(dm::DataModel, idx::Int, model::M, θ, η_ind,
        cache::_LLCache, sol_accessors, const_cov::CC, obs_series::OS, obs_cols,
        vary_cache, ind, t_obs, tmpl::TM, ::Type{T}) where {M, CC, OS, TM, T}
    obs_rows = dm.row_groups.obs_rows[idx]
    ll = zero(T)
    for i in eachindex(obs_rows)
        vary = vary_cache === nothing ? _varying_at(dm, ind, i, t_obs) : vary_cache[i]
        η_row = _row_random_effects_fill(dm, idx, i, η_ind, tmpl; obs_only = true)
        obs = sol_accessors === nothing ?
              calculate_formulas_obs(model, θ, η_row, const_cov, vary) :
              calculate_formulas_obs(model, θ, η_row, const_cov, vary, sol_accessors)
        if _row_has_hmm_dist(obs, obs_cols)
            ll_hmm = _loglikelihood_rows_hmm(dm, idx, θ, η_ind, cache, sol_accessors, i)
            isfinite(ll_hmm) || return T(-Inf)
            return ll + ll_hmm
        end
        for col in obs_cols
            y = getfield(obs_series, col)[i]
            y === missing && continue
            dist = getproperty(obs, col)
            v = _fast_logpdf(dist, y)
            v === nothing && (v = logpdf(dist, y))
            if !isfinite(v)
                return T(-Inf)
            end
            ll += v
        end
    end
    return ll
end

# HMM-capable continuation of `_loglikelihood_individual`, entered at the first row
# whose formulas produce an HMM-family outcome distribution. Forward-filtering state is
# allocated once up front (assigned-once slots — no loop-carried phi nodes) and the
# per-column prior store is positional instead of a `Dict{Symbol, Any}`: both keep the
# function reverse-mode-AD-friendly and avoid hashing on the HMM hot path.
function _loglikelihood_rows_hmm(dm::DataModel, idx::Int, θ, η_ind, cache::_LLCache,
        sol_accessors, i_start::Int)
    model = dm.model
    ind = dm.individuals[idx]
    obs_rows = dm.row_groups.obs_rows[idx]
    const_cov = ind.const_cov
    obs_series = ind.series.obs
    vary_cache = cache.vary_cache === nothing ? nothing : cache.vary_cache[idx]
    obs_cols = dm.config.obs_cols
    rowwise_re = _needs_rowwise_random_effects(dm, idx; obs_only = true)
    T_hmm = promote_type(eltype(θ), eltype(η_ind))
    n_cols = length(obs_cols)
    hmm_seen = falses(n_cols)
    hmm_init = Vector{Vector{T_hmm}}(undef, n_cols)
    hmm_has_prior = falses(n_cols)
    hmm_priors = Vector{Vector{T_hmm}}(undef, n_cols)
    # Row-constant formula context, hoisted like in `_loglikelihood_individual`
    # (assigned once before the loop — no loop-carried state added). The rowwise
    # path keeps the per-row η selection and the dynamic formulas entry point.
    pre = rowwise_re ? nothing : calculate_prede(model, θ, η_ind, const_cov)
    ctx = rowwise_re ? nothing :
          (; fixed_effects = θ, random_effects = η_ind, prede = pre,
        helpers = cache.helpers, model_funs = cache.model_funs)
    sol_acc = sol_accessors === nothing ? NamedTuple() : sol_accessors
    t_obs = vary_cache === nothing ? _get_col(dm.df, dm.config.time_col)[obs_rows] : nothing
    row_tmpl = rowwise_re ? _row_re_template(dm, idx, i_start, η_ind; obs_only = true) :
               nothing
    ll = zero(T_hmm)
    for i in i_start:length(obs_rows)
        vary = vary_cache === nothing ? _varying_at(dm, ind, i, t_obs) : vary_cache[i]
        obs = if rowwise_re
            η_row = _row_random_effects_fill(dm, idx, i, η_ind, row_tmpl; obs_only = true)
            sol_accessors === nothing ?
            calculate_formulas_obs(model, θ, η_row, const_cov, vary) :
            calculate_formulas_obs(model, θ, η_row, const_cov, vary, sol_accessors)
        else
            model.formulas.obs(ctx, sol_acc, const_cov, vary)
        end
        for (j, col) in Base.pairs(obs_cols)
            y = getfield(obs_series, col)[i]
            dist = getproperty(obs, col)
            if _is_hmm_dist(dist)
                if !hmm_seen[j]
                    init_probs = dist isa CoarsedObservedStatesMarkovModel ?
                                 dist.base_dist.initial_dist.p : dist.initial_dist.p
                    buf = Vector{T_hmm}(undef, length(init_probs))
                    copyto!(buf, init_probs)
                    hmm_init[j] = buf
                    hmm_seen[j] = true
                end
                init_p = hmm_init[j]
                # check_args=false inside `_hmm_pin_initial_probs`: degenerate
                # posteriors must not throw — the isfinite guard below catches
                # NaN logpdf.
                dist_up = _hmm_pin_initial_probs(dist, init_p)
                prior = hmm_has_prior[j] ? hmm_priors[j] : nothing
                dist_use = try
                    _hmm_with_prior(dist_up, prior)
                catch err
                    if err isa DomainError || err isa ArgumentError
                        return -Inf
                    end
                    rethrow(err)
                end
                if y === missing
                    hmm_priors[j] = try
                        probabilities_hidden_states(dist_use)
                    catch err
                        if err isa DomainError || err isa ArgumentError
                            return -Inf
                        end
                        rethrow(err)
                    end
                    hmm_has_prior[j] = true
                    continue
                end
                # Combined accessor: continuous-time families propagate the hidden
                # state once and reuse it for both the likelihood and the posterior
                # (was two independent `exp(QΔt)` propagations per observed row). The
                # returned pair is bit-identical to separate logpdf/posterior calls.
                lp_post = try
                    _hmm_logpdf_and_posterior(dist_use, y)
                catch err
                    if err isa DomainError || err isa ArgumentError
                        return -Inf
                    end
                    rethrow(err)
                end
                v = lp_post[1]
                if !isfinite(v)
                    return -Inf
                end
                hmm_priors[j] = lp_post[2]
                hmm_has_prior[j] = true
            else
                y === missing && continue
                v = _fast_logpdf(dist, y)
                v === nothing && (v = logpdf(dist, y))
                if !isfinite(v)
                    return -Inf
                end
            end
            ll += v
        end
    end
    return ll
end

function _resid_stats_individual(dm::DataModel, idx::Int, θ, η_ind, cache::_LLCache)
    model = dm.model
    ind = dm.individuals[idx]
    obs_rows = dm.row_groups.obs_rows[idx]
    const_cov = ind.const_cov
    obs_series = ind.series.obs
    vary_cache = cache.vary_cache === nothing ? nothing : cache.vary_cache[idx]
    if η_ind isa NamedTuple
        η_ind = ComponentArray(η_ind)
    end

    sol_accessors = nothing
    pre = nothing
    if model.de.de !== nothing
        pre = calculate_prede(model, θ, η_ind, const_cov)
        sol_accessors = _ll_solve_de(dm, idx, θ, η_ind, cache, pre)
        sol_accessors === nothing &&
            return (zero(promote_type(eltype(θ), Float64)), 0, false)
    end

    resid_ss = zero(promote_type(eltype(θ), Float64))
    resid_n = 0
    obs_cols = dm.config.obs_cols
    time_col = vary_cache === nothing ? _get_col(dm.df, dm.config.time_col)[obs_rows] :
               nothing
    rowwise_re = _needs_rowwise_random_effects(dm, idx; obs_only = true)
    # Row-constant formula context, hoisted as in `_loglikelihood_individual`;
    # the rowwise path keeps the per-row η selection.
    pre === nothing && !rowwise_re && (pre = calculate_prede(model, θ, η_ind, const_cov))
    ctx = rowwise_re ? nothing :
          (; fixed_effects = θ, random_effects = η_ind, prede = pre,
        helpers = cache.helpers, model_funs = cache.model_funs)
    sol_acc = sol_accessors === nothing ? NamedTuple() : sol_accessors
    row_tmpl = if rowwise_re && !isempty(obs_rows)
        η_ind isa NamedTuple && (η_ind = ComponentArray(η_ind))
        _row_re_template(dm, idx, 1, η_ind; obs_only = true)
    else
        nothing
    end
    for i in eachindex(obs_rows)
        vary = vary_cache === nothing ? _varying_at(dm, ind, i, time_col) : vary_cache[i]
        obs = if rowwise_re
            η_row = _row_random_effects_fill(dm, idx, i, η_ind, row_tmpl; obs_only = true)
            sol_accessors === nothing ?
            calculate_formulas_obs(model, θ, η_row, const_cov, vary) :
            calculate_formulas_obs(model, θ, η_row, const_cov, vary, sol_accessors)
        else
            model.formulas.obs(ctx, sol_acc, const_cov, vary)
        end
        for col in obs_cols
            dist = getproperty(obs, col)
            dist isa Normal || return (resid_ss, resid_n, false)
            y = getfield(obs_series, col)[i]
            y === missing && continue
            resid = y - dist.μ
            resid_ss += resid * resid
            resid_n += 1
        end
    end
    return (resid_ss, resid_n, true)
end

@inline function _fast_logpdf(dist::Normal, y)
    σ = dist.σ
    σ > 0 || return -Inf
    z = (y - dist.μ) / σ
    return -log(σ) - 0.5 * log(2π) - 0.5 * z * z
end

@inline function _fast_logpdf(dist::LogNormal, y)
    y > 0 || return -Inf
    σ = dist.σ
    σ > 0 || return -Inf
    ly = log(y)
    z = (ly - dist.μ) / σ
    return -ly - log(σ) - 0.5 * log(2π) - 0.5 * z * z
end

@inline function _fast_logpdf(dist::Bernoulli, y)
    p = dist.p
    (p >= 0 && p <= 1) || return -Inf
    y == 1 ? log(p) : y == 0 ? log1p(-p) : -Inf
end

@inline function _fast_logpdf(dist::Poisson, y)
    λ = dist.λ
    λ >= 0 || return -Inf
    y < 0 && return -Inf
    y_int = floor(Int, y)
    y_int == y || return -Inf
    return y_int * log(λ) - λ - SpecialFunctions.logfactorial(y_int)
end

@inline _fast_logpdf(::Any, ::Any) = nothing

# Function barrier: `dm.individuals` has abstract eltype, so `vary`/`dyn` are only
# concretely typed once they cross this call. Inside, the comprehension infers a
# concrete row type, giving each individual a `Vector{<concrete NamedTuple>}`.
function _build_vary_cache_individual(vary::NamedTuple, dyn::NamedTuple, t_obs,
        n_rows::Int)
    # Each thread cache must own its dynamic-covariate interpolants. A DataInterpolations
    # interpolant mutates an internal search guesser (`idx_prev`) on every evaluation, so
    # sharing one interpolant across threads (they all reference the single DataModel's
    # `series.dyn`) is a data race that makes threaded results nondeterministic. deepcopy
    # gives this cache its own interpolants; skipped when there are no dynamic covariates
    # (the common case), so non-dynamic models pay nothing.
    dyn_local = isempty(dyn) ? dyn : map(deepcopy, dyn)
    return [_vary_row(vary, dyn_local, t_obs, j) for j in 1:n_rows]
end

function _build_vary_cache(dm::DataModel)
    return map(eachindex(dm.individuals)) do i
        ind = dm.individuals[i]
        obs_rows = dm.row_groups.obs_rows[i]
        t_obs = _get_col(dm.df, dm.config.time_col)[obs_rows]
        _build_vary_cache_individual(ind.series.vary, ind.series.dyn, t_obs,
            length(obs_rows))
    end
end

function build_ll_cache(dm::DataModel;
        ode_args::Tuple = (),
        ode_kwargs::NamedTuple = NamedTuple(),
        serialization::SciMLBase.EnsembleAlgorithm = EnsembleSerial(),
        force_saveat::Bool = false,
        nthreads::Int = 1)
    if serialization isa SciMLBase.EnsembleThreads && nthreads == 1
        nthreads = Threads.maxthreadid()
    end
    nthreads <= 1 && return _build_ll_cache_single(
        dm; ode_args = ode_args, ode_kwargs = ode_kwargs, force_saveat = force_saveat)
    return [_build_ll_cache_single(dm; ode_args = ode_args, ode_kwargs = ode_kwargs,
                force_saveat = force_saveat) for _ in 1:nthreads]
end

function _build_ll_cache_single(dm::DataModel;
        ode_args::Tuple = (),
        ode_kwargs::NamedTuple = NamedTuple(),
        force_saveat::Bool = false)
    solver_cfg = get_solver_config(dm.model)
    alg = solver_cfg.alg === nothing ? Tsit5() : solver_cfg.alg
    prob_templates = dm.model.de.de === nothing ? nothing :
                     Vector{Any}(undef, length(dm.individuals))
    if prob_templates !== nothing
        fill!(prob_templates, nothing)
    end
    vary_cache = _build_vary_cache(dm)
    saveat_cache = _build_fit_saveat_cache(dm, force_saveat)
    return _LLCache(get_helper_funs(dm.model),
        get_model_funs(dm.model),
        solver_cfg,
        alg,
        ode_args,
        ode_kwargs,
        prob_templates,
        vary_cache,
        saveat_cache)
end

function _build_fit_saveat_cache(dm::DataModel, force_saveat::Bool)
    (!force_saveat || dm.model.de.de === nothing) && return nothing
    state_names = get_de_states(dm.model.de.de)
    signal_names = get_de_signals(dm.model.de.de)
    time_offsets, requires_dense = get_formulas_time_offsets(
        dm.model.formulas.formulas, state_names, signal_names)
    requires_dense && return nothing
    out = Vector{Any}(undef, length(dm.individuals))
    for i in eachindex(dm.individuals)
        ind = dm.individuals[i]
        if ind.saveat === nothing
            rows = dm.row_groups.rows[i]
            obs_rows = dm.row_groups.obs_rows[i]
            tvals = _get_col(dm.df, dm.config.time_col)[obs_rows]
            if !isempty(time_offsets)
                expanded = Float64[]
                for t in tvals
                    for off in time_offsets
                        push!(expanded, t + off)
                    end
                end
                tvals = expanded
            end
            if dm.config.evid_col !== nothing
                evid = _get_col(dm.df, dm.config.evid_col)[rows]
                evt_idx = findall(!=(0), evid)
                if !isempty(evt_idx)
                    tvals = vcat(tvals, _get_col(dm.df, dm.config.time_col)[rows][evt_idx])
                end
            end
            out[i] = sort(unique(tvals))
        else
            out[i] = ind.saveat
        end
    end
    return out
end

function loglikelihood(dm::DataModel, θ::ComponentArray, η;
        ode_args::Tuple = (),
        ode_kwargs::NamedTuple = NamedTuple(),
        serialization::SciMLBase.EnsembleAlgorithm = EnsembleThreads(),
        cache = nothing)
    # Fresh bindings throughout: `θ` and `η` are captured by the threaded closure
    # below, and reassigning a captured variable boxes it (`Core.Box`) — which made
    # every `η[i]` lookup and the whole serial loop dynamically typed (measured
    # ~1 KB/individual of boxing overhead on a 16 B/individual evaluation).
    θs = _symmetrize_psd_params(θ, dm.model.fixed.fixed)
    # A shared NamedTuple η would otherwise be converted to a ComponentArray inside
    # EVERY `_loglikelihood_individual` call (measured 15× on simple models) —
    # convert once up front. (Vector-of-NamedTuple elements are used once each, so
    # they are left to the per-call guard.)
    ηs = η isa NamedTuple ? ComponentArray(η) : η
    cache_use = cache === nothing ?
                build_ll_cache(dm; ode_args = ode_args, ode_kwargs = ode_kwargs,
        serialization = serialization) : cache
    n = length(dm.individuals)
    η_eltype = ηs isa Vector ? (isempty(ηs) ? Float64 : eltype(first(ηs))) : eltype(ηs)
    T = promote_type(eltype(θs), η_eltype)
    if serialization isa SciMLBase.EnsembleThreads
        nthreads = Threads.maxthreadid()
        caches = if cache_use isa Vector
            length(cache_use) < nthreads &&
                throw(ArgumentError("Threaded loglikelihood requires at least $(nthreads) cache entries, got $(length(cache_use))."))
            cache_use
        elseif cache_use isa _LLCache
            build_ll_cache(
                dm; ode_args = ode_args, ode_kwargs = ode_kwargs, nthreads = nthreads)
        else
            built = build_ll_cache(
                dm; ode_args = ode_args, ode_kwargs = ode_kwargs, nthreads = nthreads)
            built isa Vector ? built : [built]
        end
        by_individual = Vector{T}(undef, n)
        bad = Threads.Atomic{Bool}(false)
        # Chunk-indexed cache assignment: each task owns cache slot `c` for its whole
        # stride. Indexing by `Threads.threadid()` is unsafe under task migration
        # (`@threads :dynamic` may move a task between threads at yield points,
        # letting two tasks share one cache slot).
        n_chunks = length(caches)
        Threads.@threads for c in 1:n_chunks
            cache_c = caches[c]
            for i in c:n_chunks:n
                bad[] && break
                η_ind = ηs isa Vector ? ηs[i] : ηs
                lli = _loglikelihood_individual(dm, i, θs, η_ind, cache_c)
                if !isfinite(lli)
                    bad[] = true
                else
                    by_individual[i] = lli
                end
            end
        end
        bad[] && return -Inf
        ll = zero(T)
        @inbounds for i in 1:n
            ll += by_individual[i]
        end
        return ll
    else
        # zero(T), not zero(eltype(θs)): a Dual-η/Float64-θ evaluation would
        # otherwise carry a loop-wide Union accumulator (the threaded branch
        # promotes the same way).
        ll = zero(T)
        for i in 1:n
            η_ind = ηs isa Vector ? ηs[i] : ηs
            lli = _loglikelihood_individual(dm, i, θs, η_ind, cache_use)
            !isfinite(lli) && return -Inf
            ll += lli
        end
        return ll
    end
end

# Compile-time check whether any fixed effect is a RealPSDMatrix: lets
# `_symmetrize_psd_params` return immediately for the (common) no-PSD case
# instead of running a boxing `getfield(params, name)` loop on every call.
@generated function _has_psd_params(::NamedTuple{names, T}) where {names, T}
    return any(p -> p <: RealPSDMatrix, T.parameters) ? :(true) : :(false)
end

function _symmetrize_psd_params(θ::ComponentArray, fe::FixedEffects)
    params = get_params(fe)
    _has_psd_params(params) || return θ
    θsym = θ
    for name in get_names(fe)
        p = getfield(params, name)
        if p isa RealPSDMatrix
            A = getproperty(θsym, name)
            if A isa AbstractMatrix
                Asym = 0.5 .* (A .+ A')
                if θsym === θ
                    # Flat data copy + axes rewrap instead of deepcopy:
                    # `setproperty!` writes values, not containers, so copying
                    # the underlying vector is sufficient (and much cheaper).
                    θsym = ComponentArray(copy(ComponentArrays.getdata(θ)), getaxes(θ))
                end
                setproperty!(θsym, name, Asym)
            end
        end
    end
    return θsym
end

# ─────────────────────────────────────────────────────────────────────────────
# Shared closed-form M-step helpers (SAEM and MCEM).
#
# The canonical implementations live in saem.jl (_saem_* names) and are
# resolved at call-time (after the full module is loaded).  These thin
# wrappers give MCEM a stable, non-SAEM-prefixed API to call.
# ─────────────────────────────────────────────────────────────────────────────

@inline _autodetect_gaussian_re(dm::DataModel, fixed_names) = _saem_autodetect_gaussian_re(
    dm, fixed_names)

@inline _re_family_map(dm::DataModel) = _saem_re_family_map(dm)

@inline _builtin_re_targets(re_cov_params::NamedTuple, re_mean_params::NamedTuple) = _saem_builtin_re_targets(
    re_cov_params, re_mean_params)

@inline _outcome_targets(obs_cols, resid_var_param) = _saem_outcome_targets(
    obs_cols, resid_var_param)

@inline _builtin_closed_form_eligibility(dm::DataModel,
re_cov_params::NamedTuple,
re_mean_params::NamedTuple,
resid_var_param,
hmm_emission_params::NamedTuple = NamedTuple()) = _saem_builtin_closed_form_eligibility(
    dm, re_cov_params, re_mean_params,
    resid_var_param, hmm_emission_params)

@inline _builtin_updates_from_stats(dm::DataModel,
θ::ComponentArray,
stats,
resid_var_param,
hmm_emission_params::NamedTuple,
re_cov_params::NamedTuple,
re_mean_params::NamedTuple) = _saem_builtin_updates_from_smoothed_stats(
    dm, θ, stats, resid_var_param,
    hmm_emission_params,
    re_cov_params, re_mean_params)

@inline _clamp_constants_to_bounds(constants::NamedTuple, fe::FixedEffects) = _saem_clamp_constants_to_bounds(
    constants, fe)

function _log_closed_form_plan(method_name::String,
        elig,
        re_cov_params::NamedTuple,
        re_mean_params::NamedTuple,
        resid_var_param,
        has_custom_closed_form::Bool,
        base_free_names::Vector{Symbol})
    cf_syms = Symbol[]
    for v in values(re_cov_params)
        _saem_collect_target_symbols!(cf_syms, v)
    end
    for v in values(re_mean_params)
        _saem_collect_target_symbols!(cf_syms, v)
    end
    _saem_collect_target_symbols!(cf_syms, resid_var_param)
    cf_set = Set(cf_syms)
    if has_custom_closed_form
        union!(cf_set, base_free_names)
    end
    numeric_params = [n for n in base_free_names if !(n in cf_set)]
    if isempty(numeric_params)
        @info "$(method_name): all parameters handled by closed-form M-step; numeric optimizer inactive."
    else
        @info "$(method_name): closed-form M-step active. Numerically optimized parameters: $(numeric_params)"
    end
    return nothing
end

"""
    _compute_obs_fe_syms(model) -> Set{Symbol}

Return the set of fixed-effect names that appear in any observation-side block:
`@formulas`, `@preDifferentialEquation`, `@DifferentialEquation`, or `@initialDE`.

A fixed-effect name NOT in this set appears only in `@randomEffects` distribution
expressions and therefore belongs entirely to Q2 (the RE-distribution term of the
complete-data log-likelihood), enabling a cheaper M-step that skips ODE evaluation.
"""
function _compute_obs_fe_syms(model)
    fe_names = Set(get_names(model.fixed.fixed))
    syms = Set{Symbol}()
    # @formulas — FormulasIR.var_syms contains all variable symbols
    union!(syms, filter(∈(fe_names), model.formulas.formulas.ir.var_syms))
    de = model.de
    if de !== nothing
        # @preDifferentialEquation
        if de.prede !== nothing
            union!(syms, filter(∈(fe_names), de.prede.meta.syms))
        end
        # @DifferentialEquation — var_syms are non-state/signal variable references,
        # fun_syms are function-call heads (e.g. model functions used in RHS)
        if de.de !== nothing
            union!(syms, filter(∈(fe_names), de.de.meta.var_syms))
            union!(syms, filter(∈(fe_names), de.de.meta.fun_syms))
        end
        # @initialDE
        if de.initial !== nothing
            union!(syms, filter(∈(fe_names), de.initial.ir.var_syms))
            union!(syms, filter(∈(fe_names), de.initial.ir.prop_syms))
        end
    end
    return syms
end

"""
    _partition_q1_q2_names(model, free_names) -> (q1=Vector{Symbol}, q2=Vector{Symbol})

Partition `free_names` into Q1 parameters (appear in obs-side blocks, require ODE
evaluation) and Q2 parameters (appear only in `@randomEffects` distribution expressions,
no ODE required).

Returns a NamedTuple `(q1=..., q2=...)` preserving the original order of `free_names`.
"""
function _partition_q1_q2_names(model, free_names::Vector{Symbol})
    obs_fe = _compute_obs_fe_syms(model)
    re_syms_map = get_re_syms(model.random.random)
    re_fe_syms = Set{Symbol}()
    for (_, syms) in Base.pairs(re_syms_map)
        union!(re_fe_syms, syms)
    end
    # Q2 candidates: in RE distribution expressions but not in any obs-side block
    q2_candidates = setdiff(re_fe_syms, obs_fe)
    q2_free = [n for n in free_names if n ∈ q2_candidates]
    q1_free = [n for n in free_names if n ∉ q2_candidates]
    return (q1 = q1_free, q2 = q2_free)
end

"""
    compute_shrinkage(res::FitResult; dm, constants_re) -> NamedTuple

Compute eta shrinkage for every scalar random effect in the fitted model.

Shrinkage is defined as `1 - SD(eta) / omega`, where `eta_i = f(EBE_i) - mu_i` is the
individual-level random-effect residual, `mu_i` is the covariate-adjusted population mean
for individual `i`, and `omega` is the estimated standard deviation of the RE distribution.
For `LogNormal` random effects the transformation is `f(x) = log(x)`; for `Normal` it is
the identity; for all other distributions the linear deviation from the population mean is
used.

A value near 0 indicates that EBEs carry individual information. Values above 0.3–0.4
signal that EBEs are pulled toward the population mean and should not be interpreted as
individual estimates.

# Keyword Arguments
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
- `constants_re::NamedTuple = NamedTuple()`: random effects fixed at given values.

# Returns
A `NamedTuple` mapping each RE name to a `NamedTuple` with fields
`shrinkage`, `eta_std`, and `sigma`.
"""
function compute_shrinkage(res::FitResult;
        dm::Union{Nothing, DataModel} = nothing,
        constants_re::NamedTuple = NamedTuple())
    dm_use = dm !== nothing ? dm : get_data_model(res)
    dm_use === nothing &&
        error("This fit result does not store a DataModel; pass dm=... explicitly.")

    θ = get_params(res; scale = :untransformed)
    constants_re = _res_constants_re(res, constants_re)
    re_names = get_re_names(dm_use.model.random.random)
    dists_builder = get_create_random_effect_distribution(dm_use.model.random.random)
    model_funs = get_model_funs(dm_use.model)
    helpers = get_helper_funs(dm_use.model)

    pairs = Pair{Symbol, NamedTuple}[]
    for re in re_names
        # get_random_effects(res, re) returns a vector ordered by individual index
        ebes = try
            get_random_effects(dm_use, res, re; constants_re = constants_re)
        catch
            continue
        end
        length(ebes) == length(dm_use.individuals) || continue

        etas = Float64[]
        sigma = NaN
        valid = true

        for i in eachindex(dm_use.individuals)
            ind = dm_use.individuals[i]
            ebe = Float64(ebes[i])
            isfinite(ebe) || continue

            dists = dists_builder(θ, ind.const_cov, model_funs, helpers)
            hasfield(typeof(dists), re) || (valid = false; break)
            dist_i = getfield(dists, re)

            local eta_i::Float64
            if dist_i isa LogNormal
                eta_i = log(ebe) - dist_i.μ
                sigma = dist_i.σ
            elseif dist_i isa Normal
                eta_i = ebe - dist_i.μ
                sigma = dist_i.σ
            else
                eta_i = ebe - mean(dist_i)
                sigma = std(dist_i)
            end
            push!(etas, eta_i)
        end

        valid && !isempty(etas) && isfinite(sigma) && sigma > 0 || continue
        eta_s = std(etas)
        shrink = 1.0 - eta_s / sigma
        push!(pairs, re => (; shrinkage = shrink, eta_std = eta_s, sigma = sigma))
    end
    return NamedTuple(pairs)
end
