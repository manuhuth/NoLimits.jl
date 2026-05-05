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
[`LaplaceMAP`](@ref), [`SAEM`](@ref), [`MCEM`](@ref), and [`Multistart`](@ref).
"""
abstract type FittingMethod end

"""
    _SavedFittingMethod <: FittingMethod

Lightweight stub that records which fitting method was used when a [`FitResult`](@ref)
is loaded from disk via [`load_fit`](@ref). Replaces the full method struct (which
contains optimiser closures that cannot be serialised) on save.

`get_method(res).kind` returns a Symbol such as `:mle`, `:map`, `:laplace`, etc.
"""
struct _SavedFittingMethod <: FittingMethod
    kind::Symbol   # :mle :map :laplace :laplacemap
                   # :mcem :saem :mcmc :vi :ghquadrature :ghquadraturemap
end

Base.show(io::IO, m::_SavedFittingMethod) =
    print(io, "_SavedFittingMethod(:$(m.kind)) [loaded from disk; original optimiser not stored]")

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
    n_samples :: Int
    mode      :: Symbol
    sampler   :: S
    n_warmup  :: Int
    rng       :: R   # Union{Nothing, AbstractRNG}
end

function MCIntegrator(;
    n_samples :: Int                            = 1000,
    mode      :: Symbol                         = :turing,
    sampler                                     = nothing,
    n_warmup  :: Int                            = 500,
    rng       :: Union{Nothing, AbstractRNG}    = nothing,
)
    mode in (:prior, :turing) || error("MCIntegrator mode must be :prior or :turing, got :$(mode).")
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

@inline function _resolve_ebe_rescue_options(rescue::Union{Nothing, EBERescueOptions}, ebe_grad_tol, dm::DataModel)
    rescue === nothing && return nothing
    grad_tol = rescue.grad_tol
    if grad_tol isa Symbol
        grad_tol === :auto || error("EBE rescue grad_tol must be numeric or :auto.")
        grad_tol = ebe_grad_tol
    end
    sampling = _resolve_multistart_sampling(rescue.sampling, "EBE rescue multistart sampling")
    return EBERescueOptions(rescue.enabled, rescue.multistart_n, rescue.multistart_k, rescue.max_rounds, grad_tol, sampling)
end

"""
    FitParameters{T, U}

Stores parameter estimates on both the transformed (optimisation) and untransformed
(natural) scales as `ComponentArray`s.

# Fields
- `transformed::T`: parameter vector on the optimisation scale.
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
struct FitResult{M<:FittingMethod, R<:MethodResult, S, D, DM, A, K}
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

function _nl_fmt_compact_value(x)
    x === nothing && return "-"
    x isa Missing && return "-"
    if x isa Real
        xv = Float64(x)
        isfinite(xv) || return "-"
        ax = abs(xv)
        if ax >= 1e4 || (ax > 0 && ax < 1e-3)
            return string(round(xv; sigdigits=4))
        end
        return string(round(xv; digits=4))
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
        length(get_params(res; scale=:untransformed))
    catch
        "?"
    end
    dm_state = res.data_model === nothing ? "not_stored" : "stored"
    return "FitResult(method=$(method_name), objective=$(objective_str), converged=$(converged), n_params=$(n_params), data_model=$(dm_state))"
end

Base.show(io::IO, res::FitResult) = print(io, _nl_fitresult_show_line(res))
Base.show(io::IO, ::MIME"text/plain", res::FitResult) = print(io, _nl_fitresult_show_line(res))

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
  - `:transformed` — the optimisation-scale `ComponentArray`.
  - `:untransformed` — the natural-scale `ComponentArray`.
"""
function get_params(res::FitResult; scale::Symbol=:both)
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

Return the number of optimiser iterations. Valid for optimisation-based methods
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

get_iterations(res::MethodResult) = hasproperty(res, :iterations) ? res.iterations :
    error("iterations not available for this method.")
get_raw(res::MethodResult) = hasproperty(res, :raw) ? res.raw :
    error("raw result not available for this method.")
get_notes(res::MethodResult) = hasproperty(res, :notes) ? res.notes :
    error("notes not available for this method.")
get_closed_form_mstep_used(::MethodResult) = false
get_observed(res::MethodResult) = hasproperty(res, :observed) ? res.observed :
    error("observed data not available for this method.")
get_sampler(res::MethodResult) = hasproperty(res, :sampler) ? res.sampler :
    error("sampler not available for this method.")
get_n_samples(res::MethodResult) = hasproperty(res, :n_samples) ? res.n_samples :
    error("n_samples not available for this method.")
get_variational_posterior(::MethodResult) = error("Variational posterior access not supported for this method.")
get_vi_trace(::MethodResult) = error("VI trace access not supported for this method.")
get_vi_state(::MethodResult) = error("VI state access not supported for this method.")
sample_posterior(::MethodResult; kwargs...) = error("Posterior sampling not supported for this method.")

function _re_dataframes_from_bstars(dm::DataModel,
                                    batch_infos::Vector,
                                    bstars::Vector;
                                    constants_re::NamedTuple=NamedTuple(),
                                    flatten::Bool=true,
                                    include_constants::Bool=true)
    cache = dm.re_group_info.laplace_cache
    cache === nothing && return NamedTuple()
    re_names = cache.re_names
    isempty(re_names) && return NamedTuple()
    length(bstars) == length(batch_infos) || error("EB modes do not match number of batches.")
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
        is_scalar = cache.is_scalar[ri] || cache.dims[ri] == 1

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
            cols = Dict{Symbol, Any}()
            cols[col] = rows
            for j in 1:length(names)
                cols[names[j]] = [vals_flat[i][j] for i in 1:length(vals_flat)]
            end
            push!(out_pairs, re => DataFrame(cols))
        else
            push!(out_pairs, re => DataFrame(col => rows, :value => [v[1] for v in vals_flat]))
        end
    end
    return NamedTuple(out_pairs)
end

function get_laplace_random_effects(dm::DataModel,
                                    res::FitResult;
                                    constants_re::NamedTuple=NamedTuple(),
                                    flatten::Bool=true,
                                    include_constants::Bool=true)
    (res.result isa LaplaceResult || res.result isa LaplaceMAPResult || res.result isa GHQuadratureResult || res.result isa GHQuadratureMAPResult) ||
        error("Laplace-style random-effects accessor requires a Laplace, LaplaceMAP, GHQuadrature, or GHQuadratureMAP fit result.")
    constants_re = _res_constants_re(res, constants_re)
    re_names = get_re_names(dm.model.random.random)
    isempty(re_names) && return NamedTuple()
    _, batch_infos, _ = _build_laplace_batch_infos(dm, constants_re)
    bstars = res.result.eb_modes
    return _re_dataframes_from_bstars(dm, batch_infos, bstars; constants_re=constants_re,
                                      flatten=flatten, include_constants=include_constants)
end

function get_laplace_random_effects(res::FitResult;
                                    constants_re::NamedTuple=NamedTuple(),
                                    flatten::Bool=true,
                                    include_constants::Bool=true)
    dm = res.data_model
    dm === nothing && error("This fit result does not store a DataModel; call get_laplace_random_effects(dm, res) instead.")
    return get_laplace_random_effects(dm, res; constants_re=constants_re, flatten=flatten, include_constants=include_constants)
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

function _compute_bstars(dm::DataModel,
                         θu::ComponentArray,
                         constants_re::NamedTuple,
                         ll_cache,
                         ebe::EBEOptions,
                         rng::AbstractRNG;
                         rescue::Union{Nothing, EBERescueOptions}=nothing,
                         progress::Bool=false,
                         progress_desc::AbstractString="Final EBE")
    ebe = _resolve_ebe_options(ebe, dm)
    rescue = _resolve_ebe_rescue_options(rescue, ebe.grad_tol, dm)
    _, batch_infos, const_cache = _build_laplace_batch_infos(dm, constants_re)
    T = eltype(θu)
    n_batches = length(batch_infos)
    bstar_cache = _LaplaceBStarCache([Vector{T}() for _ in 1:n_batches], falses(n_batches))
    grad_cache = _LaplaceGradCache([Vector{T}() for _ in 1:n_batches],
                                   fill(T(NaN), n_batches),
                                   [Vector{T}() for _ in 1:n_batches],
                                   falses(n_batches))
    ad_cache = _init_laplace_ad_cache(n_batches)
    hess_cache = _init_laplace_hess_cache(T, n_batches)
    ebe_cache = _LaplaceCache(nothing, bstar_cache, grad_cache, ad_cache, hess_cache)
    ll_cache_local = ll_cache isa Vector ? ll_cache[1] : ll_cache
    ebe_serialization = ll_cache isa Vector ? SciMLBase.EnsembleThreads() : SciMLBase.EnsembleSerial()

    function _batch_grad_norms()
        norms = Vector{Float64}(undef, n_batches)
        for (bi, info) in enumerate(batch_infos)
            if info.n_b == 0
                norms[bi] = 0.0
                continue
            end
            b = ebe_cache.bstar_cache.b_star[bi]
            if isempty(b)
                norms[bi] = Inf
                continue
            end
            g, _ = _laplace_gradb_cached!(ebe_cache, bi, dm, info, θu, const_cache, ll_cache_local, b)
            gn = maximum(abs, g)
            norms[bi] = isfinite(gn) ? Float64(gn) : Inf
        end
        return norms
    end

    bstars = _laplace_get_bstar!(ebe_cache, dm, batch_infos, θu, const_cache, ll_cache;
                                 optimizer=ebe.optimizer,
                                 optim_kwargs=ebe.optim_kwargs,
                                 adtype=ebe.adtype,
                                 grad_tol=ebe.grad_tol,
                                 multistart=LaplaceMultistartOptions(ebe.multistart_n, ebe.multistart_k, ebe.grad_tol, ebe.max_rounds, ebe.sampling),
                                 rng=rng,
                                 serialization=ebe_serialization,
                                 progress=progress,
                                 progress_desc="$(progress_desc) (pass 1)")

    if rescue !== nothing && rescue.enabled && n_batches > 0
        norms_before = _batch_grad_norms()
        rescue_tol = Float64(rescue.grad_tol)
        if any(>(rescue_tol), norms_before)
            bstars = _laplace_get_bstar!(ebe_cache, dm, batch_infos, θu, const_cache, ll_cache;
                                         optimizer=ebe.optimizer,
                                         optim_kwargs=ebe.optim_kwargs,
                                         adtype=ebe.adtype,
                                         grad_tol=ebe.grad_tol,
                                         theta_tol=-one(eltype(θu)),
                                         multistart=LaplaceMultistartOptions(rescue.multistart_n, rescue.multistart_k, rescue.grad_tol, rescue.max_rounds, rescue.sampling),
                                         rng=rng,
                                         serialization=ebe_serialization,
                                         progress=progress,
                                         progress_desc="$(progress_desc) (rescue)")
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

Supported methods: `Laplace`, `LaplaceMAP`, `MCEM`, `SAEM`, `GHQuadrature`, `GHQuadratureMAP`.

# Keyword Arguments
- `constants_re::NamedTuple = NamedTuple()`: fix random effects at given values (natural scale).
- `flatten::Bool = true`: if `true`, expand vector random effects to individual columns.
- `include_constants::Bool = true`: if `true`, include constant random effects in the output.
"""
function get_random_effects(dm::DataModel,
                            res::FitResult;
                            constants_re::NamedTuple=NamedTuple(),
                            flatten::Bool=true,
                            include_constants::Bool=true)
    constants_re = _res_constants_re(res, constants_re)
    if res.result isa LaplaceResult || res.result isa LaplaceMAPResult || res.result isa GHQuadratureResult || res.result isa GHQuadratureMAPResult
        return get_laplace_random_effects(dm, res; constants_re=constants_re, flatten=flatten, include_constants=include_constants)
    end
    if res.result isa MCEMResult
        θu = get_params(res; scale=:untransformed)
        ode_args = haskey(res.fit_kwargs, :ode_args) ? getfield(res.fit_kwargs, :ode_args) : ()
        ode_kwargs = haskey(res.fit_kwargs, :ode_kwargs) ? getfield(res.fit_kwargs, :ode_kwargs) : NamedTuple()
        serialization = haskey(res.fit_kwargs, :serialization) ? getfield(res.fit_kwargs, :serialization) : EnsembleThreads()
        rng = haskey(res.fit_kwargs, :rng) ? getfield(res.fit_kwargs, :rng) : Random.default_rng()
        ll_cache = build_ll_cache(dm; ode_args=ode_args, ode_kwargs=ode_kwargs, serialization=serialization, force_saveat=true)
        bstars = res.result.eb_modes
        if bstars === nothing
            bstars, batch_infos = _compute_bstars(dm, θu, constants_re, ll_cache, res.method.ebe, rng;
                                                  rescue=res.method.ebe_rescue)
        else
            _, batch_infos, _ = _build_laplace_batch_infos(dm, constants_re)
        end
        return _re_dataframes_from_bstars(dm, batch_infos, bstars; constants_re=constants_re,
                                          flatten=flatten, include_constants=include_constants)
    end
    if res.result isa SAEMResult
        θu = get_params(res; scale=:untransformed)
        constants_re = _saem_anneal_constants_re(dm, θu, _saem_anneal_names(res), constants_re)
        ode_args = haskey(res.fit_kwargs, :ode_args) ? getfield(res.fit_kwargs, :ode_args) : ()
        ode_kwargs = haskey(res.fit_kwargs, :ode_kwargs) ? getfield(res.fit_kwargs, :ode_kwargs) : NamedTuple()
        serialization = haskey(res.fit_kwargs, :serialization) ? getfield(res.fit_kwargs, :serialization) : EnsembleThreads()
        rng = haskey(res.fit_kwargs, :rng) ? getfield(res.fit_kwargs, :rng) : Random.default_rng()
        ll_cache = build_ll_cache(dm; ode_args=ode_args, ode_kwargs=ode_kwargs, serialization=serialization, force_saveat=true)
        bstars = res.result.eb_modes
        if bstars === nothing
            # When the result was loaded from disk, fall back to defaults from a vanilla SAEM().
            _saem_opts = res.method isa _SavedFittingMethod ? SAEM().saem : res.method.saem
            ebe = EBEOptions(_saem_opts.ebe_optimizer, _saem_opts.ebe_optim_kwargs, _saem_opts.ebe_adtype,
                             _saem_opts.ebe_grad_tol, _saem_opts.ebe_multistart_n, _saem_opts.ebe_multistart_k,
                             _saem_opts.ebe_multistart_max_rounds, _saem_opts.ebe_multistart_sampling)
            bstars, batch_infos = _compute_bstars(dm, θu, constants_re, ll_cache, ebe, rng;
                                                  rescue=_saem_opts.ebe_rescue)
        else
            _, batch_infos, _ = _build_laplace_batch_infos(dm, constants_re)
        end
        return _re_dataframes_from_bstars(dm, batch_infos, bstars; constants_re=constants_re,
                                          flatten=flatten, include_constants=include_constants)
    end
    error("Random-effects access not supported for this method.")
end

function get_random_effects(res::FitResult;
                            constants_re::NamedTuple=NamedTuple(),
                            flatten::Bool=true,
                            include_constants::Bool=true)
    dm = res.data_model
    dm === nothing && error("This fit result does not store a DataModel; call get_random_effects(dm, res) instead.")
    return get_random_effects(dm, res; constants_re=constants_re, flatten=flatten, include_constants=include_constants)
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
                           constants_re::NamedTuple=NamedTuple(),
                           ode_args::Tuple=(),
                           ode_kwargs::NamedTuple=NamedTuple(),
                           serialization::SciMLBase.EnsembleAlgorithm=EnsembleThreads())
    constants_re = _res_constants_re(res, constants_re)
    θu = get_params(res; scale=:untransformed)
    if res.result isa MLEResult || res.result isa MAPResult
        return loglikelihood(dm, θu, ComponentArray(); ode_args=ode_args, ode_kwargs=ode_kwargs, serialization=serialization)
    elseif res.result isa LaplaceResult || res.result isa LaplaceMAPResult || res.result isa GHQuadratureMAPResult
        pairing, batch_infos, const_cache = _build_laplace_batch_infos(dm, constants_re)
        bstars = res.result.eb_modes
        length(bstars) == length(batch_infos) || error("Laplace-style EB modes do not match number of batches.")
        η_vec = _eta_from_eb(dm, batch_infos, bstars, const_cache, θu)
        return loglikelihood(dm, θu, η_vec; ode_args=ode_args, ode_kwargs=ode_kwargs, serialization=serialization)
    elseif res.result isa GHQuadratureResult || res.result isa GHQuadratureMAPResult
        # Re-evaluate the sparse-grid marginal log-likelihood at the estimated θ.
        level = res.method.level
        ll_cache = build_ll_cache(dm; ode_args=ode_args, ode_kwargs=ode_kwargs, force_saveat=true)
        _, batch_infos, const_cache = _build_laplace_batch_infos(dm, constants_re)
        θu_re = _symmetrize_psd_params(θu, dm.model.fixed.fixed)
        total = 0.0
        for info in batch_infos
            bll = _ghq_batch_ll(dm, info, θu_re, const_cache, ll_cache, level)
            bll == -Inf && return -Inf
            total += bll
        end
        return total
    else
        error("loglikelihood accessor not supported for this method.")
    end
end

function get_loglikelihood(res::FitResult;
                           constants_re::NamedTuple=NamedTuple(),
                           ode_args::Tuple=(),
                           ode_kwargs::NamedTuple=NamedTuple(),
                           serialization::SciMLBase.EnsembleAlgorithm=EnsembleThreads())
    dm = res.data_model
    dm === nothing && error("This fit result does not store a DataModel; call get_loglikelihood(dm, res) instead.")
    return get_loglikelihood(dm, res; constants_re=constants_re, ode_args=ode_args, ode_kwargs=ode_kwargs, serialization=serialization)
end

function _default_ebe_options()
    EBEOptions(
        OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking(maxstep=1.0)),
        NamedTuple(),
        Optimization.AutoForwardDiff(),
        :auto,
        50,
        10,
        1,
        :lhs,
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
Laplace, LaplaceMAP, SAEM, MCEM, GHQuadrature, GHQuadratureMAP.

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
  for a batch. `nothing` raises an error (old behaviour). An `MCIntegrator` falls back to
  sampling for that batch and issues a warning. Default: `MCIntegrator()` (prior sampling,
  1000 samples).
"""
function get_loglikelihood_quadrature(dm::DataModel,
                                       res::FitResult;
                                       level::Union{Int, NamedTuple}=3,
                                       constants_re::NamedTuple=NamedTuple(),
                                       ode_args::Tuple=(),
                                       ode_kwargs::NamedTuple=NamedTuple(),
                                       serialization::SciMLBase.EnsembleAlgorithm=EnsembleThreads(),
                                       ebe_options::Union{Nothing, EBEOptions}=nothing,
                                       seed::Int=0,
                                       rng::AbstractRNG=Random.Xoshiro(seed),
                                       jitter::Float64=1e-6,
                                       mc_integrator::Union{Nothing, MCIntegrator}=nothing,
                                       fallback::Union{Nothing, MCIntegrator}=MCIntegrator())
    if res.result isa MCMCResult
        error("get_loglikelihood_quadrature: MCMC results are not supported.")
    end
    if res.result isa MLEResult || res.result isa MAPResult
        error("get_loglikelihood_quadrature: MLE/MAP models have no random effects. " *
              "Use get_loglikelihood instead, which already returns the exact marginal log-likelihood.")
    end

    constants_re = _res_constants_re(res, constants_re)
    θu = get_params(res; scale=:untransformed)
    θu_re = _symmetrize_psd_params(θu, dm.model.fixed.fixed)

    _, batch_infos, const_cache = _build_laplace_batch_infos(dm, constants_re)
    ll_cache = build_ll_cache(dm; ode_args=ode_args, ode_kwargs=ode_kwargs, force_saveat=true)

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
                _batch_loglik_from_mc(dm, info, θu_re, const_cache, ll_cache, mc_integrator, rng)
            else
                b_star = bstars[bi]
                re_measure = build_centered_re_measure(b_star, info, bi, θu_re, const_cache, dm, ll_cache;
                                                        jitter=jitter)
                if re_measure !== nothing
                    sgrid = level isa Int ? get_sparse_grid(info.n_b, level) :
                                            _build_anisotropic_batch_grid(dm, info, level)
                    batch_loglik_ghq(dm, info, θu_re, re_measure, sgrid, const_cache, ll_cache)
                elseif fallback !== nothing
                    @warn "get_loglikelihood_quadrature: Cholesky of -H failed for batch $bi " *
                          "(b* may not be a true mode or posterior is near-flat). " *
                          "Falling back to $(fallback.mode) MC sampling with $(fallback.n_samples) samples."
                    _batch_loglik_from_mc(dm, info, θu_re, const_cache, ll_cache, fallback, rng)
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
                                       level::Union{Int, NamedTuple}=3,
                                       constants_re::NamedTuple=NamedTuple(),
                                       ode_args::Tuple=(),
                                       ode_kwargs::NamedTuple=NamedTuple(),
                                       serialization::SciMLBase.EnsembleAlgorithm=EnsembleThreads(),
                                       ebe_options::Union{Nothing, EBEOptions}=nothing,
                                       seed::Int=0,
                                       rng::AbstractRNG=Random.Xoshiro(seed),
                                       jitter::Float64=1e-6,
                                       mc_integrator::Union{Nothing, MCIntegrator}=nothing,
                                       fallback::Union{Nothing, MCIntegrator}=MCIntegrator())
    dm = res.data_model
    dm === nothing && error("This fit result does not store a DataModel; call get_loglikelihood_quadrature(dm, res) instead.")
    return get_loglikelihood_quadrature(dm, res; level=level, constants_re=constants_re,
                                        ode_args=ode_args, ode_kwargs=ode_kwargs,
                                        serialization=serialization, ebe_options=ebe_options,
                                        seed=seed, rng=rng, jitter=jitter,
                                        mc_integrator=mc_integrator, fallback=fallback)
end

function get_re_covariate_usage(res::FitResult; dm::Union{Nothing, DataModel}=nothing)
    dm === nothing && (dm = res.data_model)
    dm === nothing && error("This fit result does not store a DataModel; pass dm=... to get_re_covariate_usage.")
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
  natural scale. Fixed parameters are removed from the optimiser state.
- `penalty::NamedTuple = NamedTuple()`: add per-parameter quadratic penalties on the
  natural scale (not available for MCMC).
- `ode_args::Tuple = ()`: extra positional arguments forwarded to the ODE solver.
- `ode_kwargs::NamedTuple = NamedTuple()`: extra keyword arguments forwarded to the ODE solver.
- `serialization = EnsembleThreads()`: parallelisation strategy.
- `rng = Random.default_rng()`: random number generator (used by MCMC/SAEM/MCEM).
- `theta_0_untransformed::Union{Nothing, ComponentArray} = nothing`: custom starting
  point on the natural scale; defaults to the model's declared initial values.
- `store_data_model::Bool = true`: whether to store a reference to `dm` in the result.
"""
function fit_model(dm::DataModel, method::FittingMethod, args...; store_data_model::Bool=true, kwargs...)
    return _fit_model(dm, method, args...; store_data_model=store_data_model, kwargs...)
end

function _varying_at(dm::DataModel, ind::Individual, idx::Int, t_obs)
    pairs = Pair{Symbol, Any}[]
    vary = ind.series.vary
    if hasproperty(vary, :t)
        push!(pairs, :t => getproperty(vary, :t)[idx])
    else
        push!(pairs, :t => t_obs[idx])
    end
    for name in keys(vary)
        name == :t && continue
        v = getfield(vary, name)
        if v isa AbstractVector
            push!(pairs, name => v[idx])
        elseif v isa NamedTuple
            sub = NamedTuple{keys(v)}(Tuple(getfield(v, k)[idx] for k in keys(v)))
            push!(pairs, name => sub)
        end
    end
    return merge(NamedTuple(pairs), ind.series.dyn)
end

@inline function _needs_rowwise_random_effects(dm::DataModel, idx::Int; obs_only::Bool=true)
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
                                        obs_only::Bool=true)
    return _row_random_effects_at(dm, idx, row_idx, ComponentArray(η_ind), rowwise_re; obs_only=obs_only)
end

function _row_random_effects_at(dm::DataModel,
                                idx::Int,
                                row_idx::Int,
                                η_ind::ComponentArray,
                                rowwise_re::Bool;
                                obs_only::Bool=true)
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

@inline function _ll_saveat(cache::_LLCache, idx::Int, ind::Individual)
    cache.saveat_cache === nothing && return ind.saveat
    return cache.saveat_cache[idx]
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
    if model.de.de !== nothing
        pre = calculate_prede(model, θ, η_ind, const_cov)
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
        u0 = calculate_initial_state(model, θ, η_ind, const_cov)
        cb = nothing
        infusion_rates = nothing
        if ind.callbacks !== nothing
            _apply_initial_events!(u0, ind.callbacks)
            cb = ind.callbacks.callback
            infusion_rates = ind.callbacks.infusion_rates
        end
        f!_use = _with_infusion(get_de_f!(model.de.de), infusion_rates)
        prob = cache.prob_templates === nothing ? nothing : cache.prob_templates[idx]
        if prob === nothing
            prob = ODEProblem{true, SciMLBase.FullSpecialize}(f!_use, u0, ind.tspan, compiled)
            if cache.prob_templates !== nothing
                cache.prob_templates[idx] = prob
            end
        end

        T = promote_type(eltype(θ), eltype(η_ind), eltype(u0))
        u0_T = eltype(u0) === T ? u0 : T.(u0)
        prob = remake(prob; u0 = u0_T, p = compiled)
        saveat_use = _ll_saveat(cache, idx, ind)
        sol = if saveat_use === nothing
            solve_kwargs = _ode_solve_kwargs(cache.solver_cfg.kwargs,
                                             cache.ode_kwargs,
                                             (dense=true,))
            cb === nothing ?
                solve(prob, cache.alg, cache.ode_args...; solve_kwargs...) :
                solve(prob, cache.alg, cache.ode_args...; solve_kwargs..., callback=cb)
        else
            solve_kwargs = _ode_solve_kwargs(cache.solver_cfg.kwargs,
                                             cache.ode_kwargs,
                                             (saveat=saveat_use, save_everystep=false, dense=false))
            cb === nothing ?
                solve(prob, cache.alg, cache.ode_args...; solve_kwargs...) :
                solve(prob, cache.alg, cache.ode_args...; solve_kwargs..., callback=cb)
        end
        SciMLBase.successful_retcode(sol) || return -Inf
        sol_accessors = get_de_accessors_builder(model.de.de)(sol, compiled)
    end

    T_el = promote_type(eltype(θ), eltype(η_ind))
    ll = zero(T_el)
    obs_cols = dm.config.obs_cols
    rowwise_re = _needs_rowwise_random_effects(dm, idx; obs_only=true)
    hmm_priors = nothing  # lazily initialised on first HMM observation to avoid Dict alloc for non-HMM models
    hmm_seen = nothing
    hmm_init = nothing
    T_hmm = T_el
    for i in eachindex(obs_rows)
        vary = vary_cache === nothing ? _varying_at(dm, ind, i, _get_col(dm.df, dm.config.time_col)[obs_rows]) : vary_cache[i]
        η_row = _row_random_effects_at(dm, idx, i, η_ind, rowwise_re; obs_only=true)
        obs = sol_accessors === nothing ?
              calculate_formulas_obs(model, θ, η_row, const_cov, vary) :
              calculate_formulas_obs(model, θ, η_row, const_cov, vary, sol_accessors)
        for (j, col) in pairs(obs_cols)
            y = getfield(obs_series, col)[i]
            dist = getproperty(obs, col)
            if dist isa ContinuousTimeDiscreteStatesHMM || dist isa DiscreteTimeDiscreteStatesHMM ||
               dist isa MVContinuousTimeDiscreteStatesHMM || dist isa MVDiscreteTimeDiscreteStatesHMM ||
               dist isa DiscreteTimeObservedStatesMarkovModel || dist isa ContinuousTimeObservedStatesMarkovModel ||
               dist isa CoarsedObservedStatesMarkovModel
                if hmm_seen === nothing
                    hmm_init = Vector{Vector{T_hmm}}(undef, length(obs_cols))
                    hmm_seen = falses(length(obs_cols))
                end
                hs = hmm_seen::BitVector
                hi = hmm_init::Vector{Vector{T_hmm}}
                if !hs[j]
                    init_probs = dist isa CoarsedObservedStatesMarkovModel ?
                                 dist.base_dist.initial_dist.p : dist.initial_dist.p
                    buf = Vector{T_hmm}(undef, length(init_probs))
                    copyto!(buf, init_probs)
                    hi[j] = buf
                    hs[j] = true
                end
                init_p = hi[j]
                # Use check_args=false so that degenerate posteriors (NaN/Inf from
                # zero-probability observations at extreme parameter values) do not
                # throw a DomainError.  The isfinite guard below catches NaN logpdf.
                dist_up = if dist isa ContinuousTimeDiscreteStatesHMM
                    ContinuousTimeDiscreteStatesHMM(dist.transition_matrix, dist.emission_dists,
                                                    Distributions.Categorical(init_p; check_args=false), dist.Δt;
                                                    propagation_mode=dist.propagation_mode)
                elseif dist isa MVContinuousTimeDiscreteStatesHMM
                    MVContinuousTimeDiscreteStatesHMM(dist.transition_matrix, dist.emission_dists,
                                                      Distributions.Categorical(init_p; check_args=false), dist.Δt;
                                                      propagation_mode=dist.propagation_mode)
                elseif dist isa MVDiscreteTimeDiscreteStatesHMM
                    MVDiscreteTimeDiscreteStatesHMM(dist.transition_matrix, dist.emission_dists,
                                                    Distributions.Categorical(init_p; check_args=false))
                elseif dist isa DiscreteTimeObservedStatesMarkovModel
                    DiscreteTimeObservedStatesMarkovModel(dist.transition_matrix,
                                                          Distributions.Categorical(init_p; check_args=false),
                                                          dist.state_labels)
                elseif dist isa ContinuousTimeObservedStatesMarkovModel
                    ContinuousTimeObservedStatesMarkovModel(dist.transition_matrix,
                                                            Distributions.Categorical(init_p; check_args=false),
                                                            dist.Δt, dist.state_labels;
                                                            propagation_mode=dist.propagation_mode)
                elseif dist isa CoarsedObservedStatesMarkovModel &&
                       dist.base_dist isa DiscreteTimeObservedStatesMarkovModel
                    base_dist = dist.base_dist
                    coarsed(DiscreteTimeObservedStatesMarkovModel(
                        base_dist.transition_matrix,
                        Distributions.Categorical(init_p; check_args=false),
                        base_dist.state_labels
                    ))
                elseif dist isa CoarsedObservedStatesMarkovModel &&
                       dist.base_dist isa ContinuousTimeObservedStatesMarkovModel
                    base_dist = dist.base_dist
                    coarsed(ContinuousTimeObservedStatesMarkovModel(
                        base_dist.transition_matrix,
                        Distributions.Categorical(init_p; check_args=false),
                        base_dist.Δt,
                        base_dist.state_labels;
                        propagation_mode=base_dist.propagation_mode
                    ))
                else
                    DiscreteTimeDiscreteStatesHMM(dist.transition_matrix, dist.emission_dists,
                                                  Distributions.Categorical(init_p; check_args=false))
                end
                if hmm_priors === nothing
                    hmm_priors = Dict{Symbol, Any}()
                end
                prior = get(hmm_priors::Dict{Symbol, Any}, col, nothing)
                dist_use = try
                    _hmm_with_prior(dist_up, prior)
                catch err
                    if err isa DomainError || err isa ArgumentError
                        return -Inf
                    end
                    rethrow(err)
                end
                if y === missing
                    (hmm_priors::Dict{Symbol, Any})[col] = try
                        probabilities_hidden_states(dist_use)
                    catch err
                        if err isa DomainError || err isa ArgumentError
                            return -Inf
                        end
                        rethrow(err)
                    end
                    continue
                end
                v = try
                    logpdf(dist_use, y)
                catch err
                    if err isa DomainError || err isa ArgumentError
                        return -Inf
                    end
                    rethrow(err)
                end
                if !isfinite(v)
                    return -Inf
                end
                (hmm_priors::Dict{Symbol, Any})[col] = try
                    posterior_hidden_states(dist_use, y)
                catch err
                    if err isa DomainError || err isa ArgumentError
                        return -Inf
                    end
                    rethrow(err)
                end
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

    sol_accessors = nothing
    if model.de.de !== nothing
        pre = calculate_prede(model, θ, η_ind, const_cov)
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
        u0 = calculate_initial_state(model, θ, η_ind, const_cov)
        cb = nothing
        infusion_rates = nothing
        if ind.callbacks !== nothing
            _apply_initial_events!(u0, ind.callbacks)
            cb = ind.callbacks.callback
            infusion_rates = ind.callbacks.infusion_rates
        end
        f!_use = _with_infusion(get_de_f!(model.de.de), infusion_rates)
        prob = cache.prob_templates === nothing ? nothing : cache.prob_templates[idx]
        if prob === nothing
            prob = ODEProblem{true, SciMLBase.FullSpecialize}(f!_use, u0, ind.tspan, compiled)
            if cache.prob_templates !== nothing
                cache.prob_templates[idx] = prob
            end
        end

        T = promote_type(eltype(θ), eltype(η_ind), eltype(u0))
        u0_T = eltype(u0) === T ? u0 : T.(u0)
        prob = remake(prob; u0 = u0_T, p = compiled)
        saveat_use = _ll_saveat(cache, idx, ind)
        sol = if saveat_use === nothing
            solve_kwargs = _ode_solve_kwargs(cache.solver_cfg.kwargs,
                                             cache.ode_kwargs,
                                             (dense=true,))
            cb === nothing ?
                solve(prob, cache.alg, cache.ode_args...; solve_kwargs...) :
                solve(prob, cache.alg, cache.ode_args...; solve_kwargs..., callback=cb)
        else
            solve_kwargs = _ode_solve_kwargs(cache.solver_cfg.kwargs,
                                             cache.ode_kwargs,
                                             (saveat=saveat_use, save_everystep=false, dense=false))
            cb === nothing ?
                solve(prob, cache.alg, cache.ode_args...; solve_kwargs...) :
                solve(prob, cache.alg, cache.ode_args...; solve_kwargs..., callback=cb)
        end
        SciMLBase.successful_retcode(sol) || return (zero(promote_type(eltype(θ), Float64)), 0, false)
        sol_accessors = get_de_accessors_builder(model.de.de)(sol, compiled)
    end

    resid_ss = zero(promote_type(eltype(θ), Float64))
    resid_n = 0
    obs_cols = dm.config.obs_cols
    time_col = _get_col(dm.df, dm.config.time_col)[obs_rows]
    rowwise_re = _needs_rowwise_random_effects(dm, idx; obs_only=true)
    for i in eachindex(obs_rows)
        vary = vary_cache === nothing ? _varying_at(dm, ind, i, time_col) : vary_cache[i]
        η_row = _row_random_effects_at(dm, idx, i, η_ind, rowwise_re; obs_only=true)
        obs = sol_accessors === nothing ?
              calculate_formulas_obs(model, θ, η_row, const_cov, vary) :
              calculate_formulas_obs(model, θ, η_row, const_cov, vary, sol_accessors)
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

function _resid_stats_individual_cols(dm::DataModel, idx::Int, θ, η_ind, cache::_LLCache)
    model = dm.model
    ind = dm.individuals[idx]
    obs_rows = dm.row_groups.obs_rows[idx]
    const_cov = ind.const_cov
    obs_series = ind.series.obs
    vary_cache = cache.vary_cache === nothing ? nothing : cache.vary_cache[idx]

    sol_accessors = nothing
    if model.de.de !== nothing
        pre = calculate_prede(model, θ, η_ind, const_cov)
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
        u0 = calculate_initial_state(model, θ, η_ind, const_cov)
        cb = nothing
        infusion_rates = nothing
        if ind.callbacks !== nothing
            _apply_initial_events!(u0, ind.callbacks)
            cb = ind.callbacks.callback
            infusion_rates = ind.callbacks.infusion_rates
        end
        f!_use = _with_infusion(get_de_f!(model.de.de), infusion_rates)
        prob = cache.prob_templates === nothing ? nothing : cache.prob_templates[idx]
        if prob === nothing
            prob = ODEProblem{true, SciMLBase.FullSpecialize}(f!_use, u0, ind.tspan, compiled)
            if cache.prob_templates !== nothing
                cache.prob_templates[idx] = prob
            end
        end

        T = promote_type(eltype(θ), eltype(η_ind), eltype(u0))
        u0_T = eltype(u0) === T ? u0 : T.(u0)
        prob = remake(prob; u0 = u0_T, p = compiled)
        saveat_use = _ll_saveat(cache, idx, ind)
        sol = if saveat_use === nothing
            solve_kwargs = _ode_solve_kwargs(cache.solver_cfg.kwargs,
                                             cache.ode_kwargs,
                                             (dense=true,))
            cb === nothing ?
                solve(prob, cache.alg, cache.ode_args...; solve_kwargs...) :
                solve(prob, cache.alg, cache.ode_args...; solve_kwargs..., callback=cb)
        else
            solve_kwargs = _ode_solve_kwargs(cache.solver_cfg.kwargs,
                                             cache.ode_kwargs,
                                             (saveat=saveat_use, save_everystep=false, dense=false))
            cb === nothing ?
                solve(prob, cache.alg, cache.ode_args...; solve_kwargs...) :
                solve(prob, cache.alg, cache.ode_args...; solve_kwargs..., callback=cb)
        end
        SciMLBase.successful_retcode(sol) || return (zeros(promote_type(eltype(θ), Float64), length(dm.config.obs_cols)),
                                                     zeros(Int, length(dm.config.obs_cols)),
                                                     false)
        sol_accessors = get_de_accessors_builder(model.de.de)(sol, compiled)
    end

    Tss = promote_type(eltype(θ), Float64)
    obs_cols = dm.config.obs_cols
    resid_ss = zeros(Tss, length(obs_cols))
    resid_n = zeros(Int, length(obs_cols))
    time_col = _get_col(dm.df, dm.config.time_col)[obs_rows]
    rowwise_re = _needs_rowwise_random_effects(dm, idx; obs_only=true)
    for i in eachindex(obs_rows)
        vary = vary_cache === nothing ? _varying_at(dm, ind, i, time_col) : vary_cache[i]
        η_row = _row_random_effects_at(dm, idx, i, η_ind, rowwise_re; obs_only=true)
        obs = sol_accessors === nothing ?
              calculate_formulas_obs(model, θ, η_row, const_cov, vary) :
              calculate_formulas_obs(model, θ, η_row, const_cov, vary, sol_accessors)
        for (j, col) in pairs(obs_cols)
            dist = getproperty(obs, col)
            dist isa Normal || return (resid_ss, resid_n, false)
            y = getfield(obs_series, col)[i]
            y === missing && continue
            resid = y - dist.μ
            resid_ss[j] += resid * resid
            resid_n[j] += 1
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

function _build_vary_cache(dm::DataModel)
    n = length(dm.individuals)
    caches = Vector{Vector{NamedTuple}}(undef, n)
    for i in 1:n
        ind = dm.individuals[i]
        obs_rows = dm.row_groups.obs_rows[i]
        t_obs = _get_col(dm.df, dm.config.time_col)[obs_rows]
        vary = ind.series.vary
        dyn = ind.series.dyn
        cache_i = Vector{NamedTuple}(undef, length(obs_rows))
        for j in eachindex(obs_rows)
            pairs = Pair{Symbol, Any}[]
            if hasproperty(vary, :t)
                push!(pairs, :t => getproperty(vary, :t)[j])
            else
                push!(pairs, :t => t_obs[j])
            end
            for name in keys(vary)
                name == :t && continue
                v = getfield(vary, name)
                if v isa AbstractVector
                    push!(pairs, name => v[j])
                elseif v isa NamedTuple
                    sub = NamedTuple{keys(v)}(Tuple(getfield(v, k)[j] for k in keys(v)))
                    push!(pairs, name => sub)
                end
            end
            cache_i[j] = merge(NamedTuple(pairs), dyn)
        end
        caches[i] = cache_i
    end
    return caches
end

function build_ll_cache(dm::DataModel;
                        ode_args::Tuple=(),
                        ode_kwargs::NamedTuple=NamedTuple(),
                        serialization::SciMLBase.EnsembleAlgorithm=EnsembleSerial(),
                        force_saveat::Bool=false,
                        nthreads::Int=1)
    if serialization isa SciMLBase.EnsembleThreads && nthreads == 1
        nthreads = Threads.maxthreadid()
    end
    nthreads <= 1 && return _build_ll_cache_single(dm; ode_args=ode_args, ode_kwargs=ode_kwargs, force_saveat=force_saveat)
    return [_build_ll_cache_single(dm; ode_args=ode_args, ode_kwargs=ode_kwargs, force_saveat=force_saveat) for _ in 1:nthreads]
end

function _build_ll_cache_single(dm::DataModel;
                                ode_args::Tuple=(),
                                ode_kwargs::NamedTuple=NamedTuple(),
                                force_saveat::Bool=false)
    solver_cfg = get_solver_config(dm.model)
    alg = solver_cfg.alg === nothing ? Tsit5() : solver_cfg.alg
    prob_templates = dm.model.de.de === nothing ? nothing : Vector{Any}(undef, length(dm.individuals))
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
    time_offsets, requires_dense = get_formulas_time_offsets(dm.model.formulas.formulas, state_names, signal_names)
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
                       ode_args::Tuple=(),
                       ode_kwargs::NamedTuple=NamedTuple(),
                       serialization::SciMLBase.EnsembleAlgorithm=EnsembleThreads(),
                       cache=nothing)
    θ = _symmetrize_psd_params(θ, dm.model.fixed.fixed)
    if cache === nothing
        cache = build_ll_cache(dm; ode_args=ode_args, ode_kwargs=ode_kwargs, serialization=serialization)
    end
    n = length(dm.individuals)
    if serialization isa SciMLBase.EnsembleThreads
        nthreads = Threads.maxthreadid()
        caches = if cache isa Vector
            length(cache) < nthreads &&
                throw(ArgumentError("Threaded loglikelihood requires at least $(nthreads) cache entries, got $(length(cache))."))
            cache
        elseif cache isa _LLCache
            build_ll_cache(dm; ode_args=ode_args, ode_kwargs=ode_kwargs, nthreads=nthreads)
        else
            built = build_ll_cache(dm; ode_args=ode_args, ode_kwargs=ode_kwargs, nthreads=nthreads)
            built isa Vector ? built : [built]
        end
        η_eltype = η isa Vector ? (isempty(η) ? Float64 : eltype(first(η))) : eltype(η)
        T = promote_type(eltype(θ), η_eltype)
        by_individual = Vector{T}(undef, n)
        bad = Threads.Atomic{Bool}(false)
        Threads.@threads for i in 1:n
            bad[] && continue
            tid = Threads.threadid()
            η_ind = η isa Vector ? η[i] : η
            lli = _loglikelihood_individual(dm, i, θ, η_ind, caches[tid])
            if !isfinite(lli)
                bad[] = true
            else
                by_individual[i] = lli
            end
        end
        bad[] && return -Inf
        ll = zero(T)
        @inbounds for i in 1:n
            ll += by_individual[i]
        end
        return ll
    else
        ll = zero(eltype(θ))
        for i in 1:n
            η_ind = η isa Vector ? η[i] : η
            lli = _loglikelihood_individual(dm, i, θ, η_ind, cache)
            !isfinite(lli) && return -Inf
            ll += lli
        end
        return ll
    end
end

function _symmetrize_psd_params(θ::ComponentArray, fe::FixedEffects)
    params = get_params(fe)
    θsym = θ
    for name in get_names(fe)
        p = getfield(params, name)
        if p isa RealPSDMatrix
            A = getproperty(θsym, name)
            if A isa AbstractMatrix
                Asym = 0.5 .* (A .+ A')
                if θsym === θ
                    θsym = deepcopy(θ)
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

@inline _autodetect_gaussian_re(dm::DataModel, fixed_names) =
    _saem_autodetect_gaussian_re(dm, fixed_names)

@inline _re_family_map(dm::DataModel) = _saem_re_family_map(dm)

@inline _builtin_re_targets(re_cov_params::NamedTuple, re_mean_params::NamedTuple) =
    _saem_builtin_re_targets(re_cov_params, re_mean_params)

@inline _outcome_targets(obs_cols, resid_var_param) =
    _saem_outcome_targets(obs_cols, resid_var_param)

@inline _builtin_closed_form_eligibility(dm::DataModel,
                                          re_cov_params::NamedTuple,
                                          re_mean_params::NamedTuple,
                                          resid_var_param,
                                          hmm_emission_params::NamedTuple=NamedTuple()) =
    _saem_builtin_closed_form_eligibility(dm, re_cov_params, re_mean_params,
                                          resid_var_param, hmm_emission_params)

@inline _builtin_updates_from_stats(dm::DataModel,
                                     θ::ComponentArray,
                                     stats,
                                     resid_var_param,
                                     hmm_emission_params::NamedTuple,
                                     re_cov_params::NamedTuple,
                                     re_mean_params::NamedTuple) =
    _saem_builtin_updates_from_smoothed_stats(dm, θ, stats, resid_var_param,
                                               hmm_emission_params,
                                               re_cov_params, re_mean_params)

@inline _clamp_constants_to_bounds(constants::NamedTuple, fe::FixedEffects) =
    _saem_clamp_constants_to_bounds(constants, fe)

function _log_closed_form_plan(method_name::String,
                                elig,
                                re_cov_params::NamedTuple,
                                re_mean_params::NamedTuple,
                                resid_var_param,
                                has_custom_closed_form::Bool,
                                base_free_names::Vector{Symbol})
    cf_syms = Symbol[]
    for v in values(re_cov_params);     _saem_collect_target_symbols!(cf_syms, v); end
    for v in values(re_mean_params);    _saem_collect_target_symbols!(cf_syms, v); end
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
