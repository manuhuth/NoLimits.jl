export FOCEI
export FOCEIResult
export FOCEIMAP
export FOCEIMAPResult
export FOCEIInformationOptions
export focei_information_opg

using ForwardDiff
using LinearAlgebra
using Optimization
using OptimizationOptimJL
using OptimizationBBO
using LineSearches
using SciMLBase
using ComponentArrays
using Random
using Distributions

"""
    FOCEIInformationOptions{T, C}

Configuration for the information matrix computation used in the FOCEI outer objective.

# Fields
- `mode::Symbol` - Strategy for computing the log-determinant of the information matrix.
  Options: `:fisher_common` (default), `:fisher_individual`, `:opg`, `:custom`.
- `jitter::T` - Initial diagonal jitter added when the information matrix is not positive
  definite.
- `max_tries::Int` - Maximum number of jitter retries before falling back.
- `growth::T` - Multiplicative factor by which jitter grows on each retry.
- `adaptive::Bool` - Whether to use adaptive jitter (scaled to matrix diagonal).
- `scale_factor::T` - Scale factor for adaptive jitter.
- `fallback_to_laplace::Bool` - If `true`, fall back to the Laplace log-determinant when
  the information matrix is indefinite after all retries.
- `mode_sensitivity::Symbol` - Method used to compute the Hessian at the empirical Bayes
  mode. Options: `:exact_hessian` (default) or `:focei_info`.
- `custom_information::C` - Custom information function; required when `mode == :custom`.
  Signature: `(dm, batch_info, θ, b, const_cache, ll_cache) -> matrix`.

Construct via the keyword constructors of `FOCEI` or `FOCEIMAP` rather than directly.
"""
struct FOCEIInformationOptions{T, C}
    mode::Symbol
    jitter::T
    max_tries::Int
    growth::T
    adaptive::Bool
    scale_factor::T
    fallback_to_laplace::Bool
    mode_sensitivity::Symbol
    custom_information::C
end

FOCEIInformationOptions(mode, jitter, max_tries, growth, adaptive, scale_factor, fallback_to_laplace, mode_sensitivity) =
    FOCEIInformationOptions(mode, jitter, max_tries, growth, adaptive, scale_factor, fallback_to_laplace, mode_sensitivity, nothing)

"""
    FOCEI(; optimizer, optim_kwargs, adtype, inner_options, info_options,
            fallback_hessian_options, cache_options, multistart_options,
            inner_optimizer, inner_kwargs, inner_adtype, inner_grad_tol,
            multistart_n, multistart_k, multistart_grad_tol,
            multistart_max_rounds, multistart_sampling,
            info_mode, info_jitter, info_max_tries, info_jitter_growth,
            info_adaptive_jitter, info_jitter_scale, info_custom,
            fallback_to_laplace, mode_sensitivity,
            fallback_jitter, fallback_max_tries, fallback_jitter_growth,
            fallback_adaptive_jitter, fallback_jitter_scale,
            fallback_use_trace_logdet_grad, fallback_use_hutchinson,
            fallback_hutchinson_n, theta_tol, lb, ub) <: FittingMethod

First Order Conditional Estimation with Interaction (FOCEI) for models with random effects.

FOCEI extends the Laplace approximation by evaluating the information matrix at the
empirical Bayes mode of each individual/batch, giving a second-order correction to the
marginal log-likelihood. The outer objective (over fixed effects) is optimized with
gradient-based methods; the inner problem (EBE mode finding) uses a nested optimizer.

Use `FOCEIMAP` instead to also add the log-prior of fixed effects to the objective.

# Key Keyword Arguments
- `optimizer` - Outer optimizer (default: `LBFGS` with backtracking line search).
- `optim_kwargs` - Extra `NamedTuple` passed to `Optimization.solve`.
- `adtype` - AD backend for outer gradients (default: `AutoForwardDiff()`).
- `info_mode::Symbol` - Information matrix strategy: `:fisher_common` (default),
  `:fisher_individual`, `:opg`, or `:custom`.
- `fallback_to_laplace::Bool` - Fall back to Laplace log-determinant when the information
  matrix is indefinite (default: `true`).
- `mode_sensitivity::Symbol` - Hessian method at the EBE mode: `:exact_hessian`
  (default) or `:focei_info`.
- `inner_optimizer` - Optimizer for the inner EBE problem (default: `LBFGS`).
- `multistart_n::Int` - Number of inner multistart candidates (default: `50`).
- `multistart_k::Int` - Number of top candidates carried forward (default: `10`).
- `theta_tol::Float64` - Cache tolerance: reuse EBE modes when fixed effects change by
  less than this amount (default: `0.0`, always recompute).
- `lb`, `ub` - Lower/upper bounds for the outer optimizer (default: `nothing`).
- `info_custom` - Custom information function; required when `info_mode=:custom`.
"""
struct FOCEI{O, K, A, IO, FO, HO, CO, MS, L, U} <: FittingMethod
    optimizer::O
    optim_kwargs::K
    adtype::A
    inner::IO
    info::FO
    fallback_hessian::HO
    cache::CO
    multistart::MS
    lb::L
    ub::U
end

"""
    FOCEIMAP(; <same keyword arguments as FOCEI>) <: FittingMethod

FOCEI with MAP regularization for models with random effects.

Identical to `FOCEI` but adds the log-prior of free fixed effects to the outer objective,
making it a maximum a posteriori (MAP) variant. Requires prior distributions on at least
one free fixed effect.

See `FOCEI` for the full list of keyword arguments and field descriptions.
"""
struct FOCEIMAP{O, K, A, IO, FO, HO, CO, MS, L, U} <: FittingMethod
    optimizer::O
    optim_kwargs::K
    adtype::A
    inner::IO
    info::FO
    fallback_hessian::HO
    cache::CO
    multistart::MS
    lb::L
    ub::U
end

FOCEI(; optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
      optim_kwargs=NamedTuple(),
      adtype=Optimization.AutoForwardDiff(),
      inner_options=nothing,
      info_options=nothing,
      fallback_hessian_options=nothing,
      cache_options=nothing,
      multistart_options=nothing,
      inner_optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
      inner_kwargs=NamedTuple(),
      inner_adtype=Optimization.AutoForwardDiff(),
      inner_grad_tol=:auto,
      multistart_n=50,
      multistart_k=10,
      multistart_grad_tol=inner_grad_tol,
      multistart_max_rounds=1,
      multistart_sampling=:lhs,
      info_mode=:fisher_common,
      info_jitter=1e-8,
      info_max_tries=6,
      info_jitter_growth=10.0,
      info_adaptive_jitter=true,
      info_jitter_scale=1e-8,
      info_custom=nothing,
      fallback_to_laplace=true,
      mode_sensitivity=:exact_hessian,
      fallback_jitter=1e-6,
      fallback_max_tries=6,
      fallback_jitter_growth=10.0,
      fallback_adaptive_jitter=true,
      fallback_jitter_scale=1e-6,
      fallback_use_trace_logdet_grad=true,
      fallback_use_hutchinson=false,
      fallback_hutchinson_n=8,
      theta_tol=0.0,
      lb=nothing,
      ub=nothing) = begin
    inner = inner_options === nothing ? LaplaceInnerOptions(inner_optimizer, inner_kwargs, inner_adtype, inner_grad_tol) : inner_options
    (mode_sensitivity == :exact_hessian || mode_sensitivity == :focei_info) ||
        error("mode_sensitivity must be :exact_hessian or :focei_info.")
    info = info_options === nothing ? FOCEIInformationOptions(info_mode, info_jitter, info_max_tries, info_jitter_growth,
                                                              info_adaptive_jitter, info_jitter_scale, fallback_to_laplace,
                                                              mode_sensitivity, info_custom) : info_options
    info.mode == :custom && info.custom_information === nothing &&
        error("FOCEI info_mode=:custom requires `info_custom` (or `info_options.custom_information`) with signature (dm, batch_info, θ, b, const_cache, ll_cache) -> information_matrix.")
    fallback_hess = fallback_hessian_options === nothing ?
                    LaplaceHessianOptions(fallback_jitter, fallback_max_tries, fallback_jitter_growth,
                                          fallback_adaptive_jitter, fallback_jitter_scale,
                                          fallback_use_trace_logdet_grad, fallback_use_hutchinson, fallback_hutchinson_n) :
                    fallback_hessian_options
    cache = cache_options === nothing ? LaplaceCacheOptions(theta_tol) : cache_options
    ms = multistart_options === nothing ? LaplaceMultistartOptions(multistart_n, multistart_k, multistart_grad_tol, multistart_max_rounds, multistart_sampling) : multistart_options
    FOCEI(optimizer, optim_kwargs, adtype, inner, info, fallback_hess, cache, ms, lb, ub)
end

FOCEIMAP(; optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
         optim_kwargs=NamedTuple(),
         adtype=Optimization.AutoForwardDiff(),
         inner_options=nothing,
         info_options=nothing,
         fallback_hessian_options=nothing,
         cache_options=nothing,
         multistart_options=nothing,
         inner_optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
         inner_kwargs=NamedTuple(),
         inner_adtype=Optimization.AutoForwardDiff(),
         inner_grad_tol=:auto,
         multistart_n=50,
         multistart_k=10,
         multistart_grad_tol=inner_grad_tol,
         multistart_max_rounds=1,
         multistart_sampling=:lhs,
         info_mode=:fisher_common,
         info_jitter=1e-8,
         info_max_tries=6,
         info_jitter_growth=10.0,
         info_adaptive_jitter=true,
         info_jitter_scale=1e-8,
         info_custom=nothing,
         fallback_to_laplace=true,
         mode_sensitivity=:exact_hessian,
         fallback_jitter=1e-6,
         fallback_max_tries=6,
         fallback_jitter_growth=10.0,
         fallback_adaptive_jitter=true,
         fallback_jitter_scale=1e-6,
         fallback_use_trace_logdet_grad=true,
         fallback_use_hutchinson=false,
         fallback_hutchinson_n=8,
         theta_tol=0.0,
         lb=nothing,
         ub=nothing) = begin
    inner = inner_options === nothing ? LaplaceInnerOptions(inner_optimizer, inner_kwargs, inner_adtype, inner_grad_tol) : inner_options
    (mode_sensitivity == :exact_hessian || mode_sensitivity == :focei_info) ||
        error("mode_sensitivity must be :exact_hessian or :focei_info.")
    info = info_options === nothing ? FOCEIInformationOptions(info_mode, info_jitter, info_max_tries, info_jitter_growth,
                                                              info_adaptive_jitter, info_jitter_scale, fallback_to_laplace,
                                                              mode_sensitivity, info_custom) : info_options
    info.mode == :custom && info.custom_information === nothing &&
        error("FOCEIMAP info_mode=:custom requires `info_custom` (or `info_options.custom_information`) with signature (dm, batch_info, θ, b, const_cache, ll_cache) -> information_matrix.")
    fallback_hess = fallback_hessian_options === nothing ?
                    LaplaceHessianOptions(fallback_jitter, fallback_max_tries, fallback_jitter_growth,
                                          fallback_adaptive_jitter, fallback_jitter_scale,
                                          fallback_use_trace_logdet_grad, fallback_use_hutchinson, fallback_hutchinson_n) :
                    fallback_hessian_options
    cache = cache_options === nothing ? LaplaceCacheOptions(theta_tol) : cache_options
    ms = multistart_options === nothing ? LaplaceMultistartOptions(multistart_n, multistart_k, multistart_grad_tol, multistart_max_rounds, multistart_sampling) : multistart_options
    FOCEIMAP(optimizer, optim_kwargs, adtype, inner, info, fallback_hess, cache, ms, lb, ub)
end

"""
    FOCEIResult{S, O, I, R, N, B} <: MethodResult

Result returned by `fit_model` with the `FOCEI` method.

# Fields
- `solution::S` - Optimal fixed-effects parameter vector on the transformed scale
  (`ComponentArray`).
- `objective::O` - Final FOCEI objective value (negative marginal log-likelihood).
- `iterations::I` - Number of outer optimizer iterations.
- `raw::R` - Raw result from `Optimization.solve`.
- `notes::N` - `NamedTuple` of diagnostic counters, including fallback statistics.
- `eb_modes::B` - Cached empirical Bayes modes for each batch at the optimum.

Access via `get_params`, `get_objective`, `get_iterations`, `get_raw`, `get_notes`,
and `get_random_effects` rather than direct field access.
"""
struct FOCEIResult{S, O, I, R, N, B} <: MethodResult
    solution::S
    objective::O
    iterations::I
    raw::R
    notes::N
    eb_modes::B
end

"""
    FOCEIMAPResult{S, O, I, R, N, B} <: MethodResult

Result returned by `fit_model` with the `FOCEIMAP` method.

Identical in structure to `FOCEIResult`; the objective additionally includes the
log-prior contribution of fixed effects.

# Fields
- `solution::S` - Optimal fixed-effects parameter vector on the transformed scale.
- `objective::O` - Final FOCEIMAP objective value.
- `iterations::I` - Number of outer optimizer iterations.
- `raw::R` - Raw result from `Optimization.solve`.
- `notes::N` - `NamedTuple` of diagnostic counters, including fallback statistics.
- `eb_modes::B` - Cached empirical Bayes modes for each batch at the optimum.

Access via `get_params`, `get_objective`, `get_iterations`, `get_raw`, `get_notes`,
and `get_random_effects` rather than direct field access.
"""
struct FOCEIMAPResult{S, O, I, R, N, B} <: MethodResult
    solution::S
    objective::O
    iterations::I
    raw::R
    notes::N
    eb_modes::B
end

mutable struct _FOCEIFallbackTracker
    total::Threads.Atomic{Int}
    info_logdet::Threads.Atomic{Int}
    mode_hessian::Threads.Atomic{Int}
end

_FOCEIFallbackTracker() = _FOCEIFallbackTracker(Threads.Atomic{Int}(0),
                                                Threads.Atomic{Int}(0),
                                                Threads.Atomic{Int}(0))

@inline function _focei_note_fallback!(tracker::Union{Nothing, _FOCEIFallbackTracker}, kind::Symbol)
    tracker === nothing && return
    Threads.atomic_add!(tracker.total, 1)
    if kind === :info_logdet
        Threads.atomic_add!(tracker.info_logdet, 1)
    elseif kind === :mode_hessian
        Threads.atomic_add!(tracker.mode_hessian, 1)
    end
    return
end

@inline function _focei_fallback_notes(tracker::_FOCEIFallbackTracker)
    return (focei_fallback_total=tracker.total[],
            focei_fallback_info_logdet=tracker.info_logdet[],
            focei_fallback_mode_hessian=tracker.mode_hessian[])
end

@inline function _focei_allfinite(A)
    @inbounds for v in A
        isfinite(v) || return false
    end
    return true
end

@inline _focei_nan(::Type{T}) where {T} = oftype(zero(T), NaN)

@inline function _focei_rank1_upper!(A::AbstractMatrix, v::AbstractVector)
    n = length(v)
    @inbounds for j in 1:n
        vj = v[j]
        for i in 1:j
            A[i, j] += v[i] * vj
        end
    end
    return A
end

@inline function _focei_rank1_upper_scaled!(A::AbstractMatrix, v::AbstractVector, w)
    n = length(v)
    @inbounds for j in 1:n
        vj = v[j]
        for i in 1:j
            A[i, j] += w * v[i] * vj
        end
    end
    return A
end

@inline function _focei_fill_symmetric_from_upper!(A::AbstractMatrix)
    n = size(A, 1)
    @inbounds for j in 1:n
        for i in (j + 1):n
            A[i, j] = A[j, i]
        end
    end
    return A
end

@inline function _focei_prior_score_analytic(dist, x::Number)
    if dist isa Normal
        σ = dist.σ
        σ > 0 || return nothing
        return (dist.μ - x) / (σ * σ)
    elseif dist isa LogNormal
        μ = dist.μ
        σ = dist.σ
        T = promote_type(typeof(x), typeof(μ), typeof(σ))
        (isfinite(x) && isfinite(μ) && isfinite(σ) && σ > 0 && x > 0) || return _focei_nan(T)
        σ2 = σ * σ
        return -inv(x) - (log(x) - μ) / (σ2 * x)
        
    elseif dist isa Exponential
        θ = dist.θ
        T = promote_type(typeof(x), typeof(θ))
        (isfinite(x) && isfinite(θ) && θ > 0 && x >= 0) || return _focei_nan(T)
        return -inv(θ)
    end
    return nothing
end

@inline function _focei_prior_score_analytic(dist, x::AbstractVector)
    if dist isa MvNormal
        length(dist) == length(x) || return nothing
        return invcov(dist) * (mean(dist) .- x)
    end
    return nothing
end

@inline function _focei_prior_score(dist, x::Number)
    s = _focei_prior_score_analytic(dist, x)
    return s === nothing ? ForwardDiff.derivative(v -> logpdf(dist, v), x) : s
end

@inline function _focei_prior_score(dist, x::AbstractVector)
    s = _focei_prior_score_analytic(dist, x)
    return s === nothing ? ForwardDiff.gradient(v -> logpdf(dist, v), x) : s
end

@inline function _focei_add_prior_fisher_common!(I::AbstractMatrix, dist, r)
    d = length(r)
    if d == 1 && dist isa Normal
        σ = dist.σ
        (σ > 0 && isfinite(σ)) || return :invalid_params
        I[first(r), first(r)] += inv(σ * σ)
        return :ok
    elseif d == 1 && dist isa LogNormal
        μ = dist.μ
        σ = dist.σ
        (isfinite(μ) && isfinite(σ) && σ > 0) || return :invalid_params
        σ2 = σ * σ
        w = exp(-2 * μ + 2 * σ2) * (1 + inv(σ2))
        (isfinite(w) && w > 0) || return :invalid_params
        I[first(r), first(r)] += w
        return :ok
    elseif d == 1 && dist isa Exponential
        θ = dist.θ
        (isfinite(θ) && θ > 0) || return :invalid_params
        I[first(r), first(r)] += inv(θ * θ)
        return :ok
    elseif dist isa MvNormal
        length(dist) == d || return :unsupported
        P = invcov(dist)
        _focei_allfinite(P) || return :invalid_params
        r0 = first(r) - 1
        @inbounds for j in 1:d
            gj = r0 + j
            for i in 1:j
                gi = r0 + i
                I[gi, gj] += P[i, j]
            end
        end
        return :ok
    end
    return :unsupported
end

@inline function _focei_push_common_params!(vals::AbstractVector, weights::Union{Nothing, AbstractVector}, dist)
    if dist isa Normal
        μ = dist.μ
        σ = dist.σ
        (isfinite(μ) && isfinite(σ) && σ > 0) || return :invalid
        invσ2 = inv(σ * σ)
        push!(vals, μ)
        push!(vals, σ)
        if weights !== nothing
            push!(weights, invσ2)
            push!(weights, 2 * invσ2)
        end
        return :ok
    elseif dist isa LogNormal
        μ = dist.μ
        σ = dist.σ
        (isfinite(μ) && isfinite(σ) && σ > 0) || return :invalid
        invσ2 = inv(σ * σ)
        push!(vals, μ)
        push!(vals, σ)
        if weights !== nothing
            push!(weights, invσ2)
            push!(weights, 2 * invσ2)
        end
        return :ok
    elseif dist isa Bernoulli
        p = dist.p
        (isfinite(p) && p > 0 && p < 1) || return :invalid
        push!(vals, p)
        if weights !== nothing
            push!(weights, inv(p * (1 - p)))
        end
        return :ok
    elseif dist isa Poisson
        λ = dist.λ
        (isfinite(λ) && λ > 0) || return :invalid
        push!(vals, λ)
        if weights !== nothing
            push!(weights, inv(λ))
        end
        return :ok
    elseif dist isa Exponential
        θ = dist.θ
        (isfinite(θ) && θ > 0) || return :invalid
        push!(vals, θ)
        if weights !== nothing
            push!(weights, inv(θ * θ))
        end
        return :ok
    elseif dist isa Geometric
        p = dist.p
        (isfinite(p) && p > 0 && p < 1) || return :invalid
        push!(vals, p)
        if weights !== nothing
            push!(weights, inv(p * p * (1 - p)))
        end
        return :ok
    elseif dist isa Binomial
        n = dist.n
        p = dist.p
        (isfinite(n) && n > 0 && isfinite(p) && p > 0 && p < 1) || return :invalid
        push!(vals, p)
        if weights !== nothing
            push!(weights, n * inv(p * (1 - p)))
        end
        return :ok
    end
    return :unsupported
end

@inline function _focei_fisher_common_error_message(; reason::Symbol,
                                                    distribution=nothing,
                                                    observable=nothing,
                                                    re_name=nothing,
                                                    level_id=nothing,
                                                    individual_index=nothing,
                                                    observation_row=nothing,
                                                    note=nothing)
    io = IOBuffer()
    print(io, "FOCEI(info_mode=:fisher_common) failed to build an analytic expected-information matrix. ")
    if reason == :unsupported_outcome_distribution
        print(io, "Unsupported outcome distribution ")
        distribution !== nothing && print(io, "`", distribution, "`")
        observable !== nothing && print(io, " for observable `", observable, "`")
        individual_index !== nothing && print(io, " (individual index ", individual_index, ")")
        observation_row !== nothing && print(io, " at data row ", observation_row)
        print(io, ". Supported outcome distributions are: Normal, LogNormal, Bernoulli, Poisson, Exponential, Geometric, Binomial. ")
    elseif reason == :unsupported_re_distribution
        print(io, "Unsupported random-effects distribution ")
        distribution !== nothing && print(io, "`", distribution, "`")
        re_name !== nothing && print(io, " for random effect `", re_name, "`")
        level_id !== nothing && print(io, " (level `", level_id, "`)")
        print(io, ". Supported random-effects distributions are: Normal, LogNormal, Exponential, and MvNormal. ")
    elseif reason == :custom_invalid_output
        print(io, "The `info_custom` callback returned an invalid matrix. ")
    else
        print(io, "Could not evaluate analytic information terms. ")
    end
    note !== nothing && print(io, note, " ")
    print(io, "Automatic OPG fallback has been removed. ")
    print(io, "Provide a custom information approximation via `FOCEI(info_mode=:custom, info_custom=(dm, batch_info, θ, b, const_cache, ll_cache) -> I)` ")
    print(io, "or `FOCEIMAP(info_mode=:custom, info_custom=...)`.")
    return String(take!(io))
end

@inline function _focei_error_fisher_common(; kwargs...)
    error(_focei_fisher_common_error_message(; kwargs...))
end

function _focei_information_from_custom(dm::DataModel,
                                        batch_info::_LaplaceBatchInfo,
                                        θ::ComponentArray,
                                        b,
                                        const_cache::LaplaceConstantsCache,
                                        ll_cache::_LLCache,
                                        info_opts::FOCEIInformationOptions)
    info_fun = info_opts.custom_information
    info_fun === nothing &&
        error("FOCEI info_mode=:custom requires `info_custom` (or `info_options.custom_information`) with signature (dm, batch_info, θ, b, const_cache, ll_cache) -> information_matrix.")
    I_raw = info_fun(dm, batch_info, θ, b, const_cache, ll_cache)
    I_raw === nothing &&
        _focei_error_fisher_common(reason=:custom_invalid_output,
                                   note="`info_custom` returned `nothing`.")
    I = Matrix(I_raw)
    nb = batch_info.n_b
    size(I) == (nb, nb) ||
        _focei_error_fisher_common(reason=:custom_invalid_output,
                                   note="`info_custom` returned size $(size(I)); expected ($(nb), $(nb)).")
    _focei_allfinite(I) ||
        _focei_error_fisher_common(reason=:custom_invalid_output,
                                   note="`info_custom` returned non-finite entries.")
    return (I .+ I') ./ 2
end

function _focei_collect_common_params_individual(dm::DataModel,
                                                 idx::Int,
                                                 θ,
                                                 η_ind,
                                                 cache::_LLCache;
                                                 with_weights::Bool)
    model = dm.model
    ind = dm.individuals[idx]
    obs_rows = dm.row_groups.obs_rows[idx]
    const_cov = ind.const_cov
    vary_cache = cache.vary_cache === nothing ? nothing : cache.vary_cache[idx]
    if η_ind isa NamedTuple
        η_ind = ComponentArray(η_ind)
    end

    T = promote_type(eltype(θ), eltype(η_ind), Float64)
    vals = T[]
    weights = with_weights ? T[] : nothing

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

        Ts = promote_type(eltype(θ), eltype(η_ind), eltype(u0))
        prob = remake(prob; u0 = Ts.(u0), p = compiled)
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
        SciMLBase.successful_retcode(sol) ||
            return (ok=false,
                    reason=:ode_solve_failure,
                    individual_index=idx,
                    note="ODE solve failed while evaluating analytic information terms.")
        sol_accessors = get_de_accessors_builder(model.de.de)(sol, compiled)
    end

    t_obs = _get_col(dm.df, dm.config.time_col)[obs_rows]
    for i in eachindex(obs_rows)
        vary = vary_cache === nothing ? _varying_at(dm, ind, i, t_obs) : vary_cache[i]
        obs = sol_accessors === nothing ?
              calculate_formulas_obs(model, θ, η_ind, const_cov, vary) :
              calculate_formulas_obs(model, θ, η_ind, const_cov, vary, sol_accessors)
        for col in dm.config.obs_cols
            dist = getproperty(obs, col)
            status = _focei_push_common_params!(vals, weights, dist)
            if status === :ok
                continue
            end
            return (ok=false,
                    reason=status === :unsupported ? :unsupported_outcome_distribution : :invalid_outcome_parameters,
                    distribution=typeof(dist),
                    observable=col,
                    individual_index=idx,
                    observation_row=obs_rows[i])
        end
    end
    return with_weights ? (ok=true, values=vals, weights=weights::Vector{T}) : (ok=true, values=vals)
end

@inline function _focei_individual_score(dm::DataModel,
                                         i::Int,
                                         batch_info::_LaplaceBatchInfo,
                                         θ::ComponentArray,
                                         b,
                                         const_cache::LaplaceConstantsCache,
                                         ll_cache::_LLCache)
    f_i = bb -> begin
        η_ind = _build_eta_ind(dm, i, batch_info, bb, const_cache, θ)
        _loglikelihood_individual(dm, i, θ, η_ind, ll_cache)
    end
    lli = f_i(b)
    isfinite(lli) || return nothing
    gi = ForwardDiff.gradient(f_i, b)
    _focei_allfinite(gi) || return nothing
    return gi
end

function _focei_add_expected_common_data_info!(I::AbstractMatrix,
                                               dm::DataModel,
                                               i::Int,
                                               batch_info::_LaplaceBatchInfo,
                                               θ::ComponentArray,
                                               b,
                                               const_cache::LaplaceConstantsCache,
                                               ll_cache::_LLCache)
    η_ind = _build_eta_ind(dm, i, batch_info, b, const_cache, θ)
    info0 = _focei_collect_common_params_individual(dm, i, θ, η_ind, ll_cache; with_weights=true)
    info0.ok || return info0
    m = length(info0.values)
    m == length(info0.weights) || return (ok=false, reason=:internal_shape_mismatch, individual_index=i)
    m == 0 && return (ok=true,)

    fail_ref = Ref{Any}(nothing)
    fvals = bb -> begin
        η_bb = _build_eta_ind(dm, i, batch_info, bb, const_cache, θ)
        vals_info = _focei_collect_common_params_individual(dm, i, θ, η_bb, ll_cache; with_weights=false)
        if !vals_info.ok || length(vals_info.values) != m
            fail_ref[] = vals_info
            Tnan = promote_type(eltype(θ), eltype(bb), Float64)
            out = Vector{Tnan}(undef, m)
            fill!(out, _focei_nan(Tnan))
            return out
        end
        return vals_info.values
    end

    J = ForwardDiff.jacobian(fvals, b)
    (size(J, 1) == m && size(J, 2) == length(b)) ||
        return (ok=false, reason=:jacobian_shape_mismatch, individual_index=i)
    if !_focei_allfinite(J)
        fail = fail_ref[]
        return fail === nothing ? (ok=false, reason=:nonfinite_jacobian, individual_index=i) : fail
    end
    @inbounds for k in 1:m
        wk = info0.weights[k]
        isfinite(wk) || return (ok=false, reason=:nonfinite_weight, individual_index=i)
        _focei_rank1_upper_scaled!(I, view(J, k, :), wk)
    end
    return (ok=true,)
end

"""
    focei_information_opg(dm, batch_info, θ, b, const_cache, ll_cache)
        -> Matrix

Compute the FOCEI information matrix for a batch using the outer-product-of-gradients
(OPG) estimator.

Each individual's score vector is computed at the empirical Bayes mode `b`, then summed
as rank-1 updates to form the information matrix. The prior contribution of each random
effect level is added analytically.

This function matches the signature required by `FOCEIInformationOptions` when
`info_mode=:custom`.

# Arguments
- `dm::DataModel` - The data model.
- `batch_info` - Internal batch descriptor (`_LaplaceBatchInfo`).
- `θ::ComponentArray` - Fixed-effects parameter vector (transformed scale).
- `b` - Empirical Bayes mode vector for this batch.
- `const_cache::LaplaceConstantsCache` - Precomputed per-individual constants.
- `ll_cache::_LLCache` - Log-likelihood cache holding model functions and helpers.

# Returns
A square `Matrix` of size `(n_b, n_b)` where `n_b` is the total number of random effect
dimensions in this batch. Returns a `NaN`-filled matrix if any individual score is
unavailable.
"""
function focei_information_opg(dm::DataModel,
                               batch_info::_LaplaceBatchInfo,
                               θ::ComponentArray,
                               b,
                               const_cache::LaplaceConstantsCache,
                               ll_cache::_LLCache)
    nb = batch_info.n_b
    T = promote_type(eltype(θ), eltype(b))
    I = zeros(T, nb, nb)
    nb == 0 && return I

    # Data contribution via OPG scores.
    for i in batch_info.inds
        gi = _focei_individual_score(dm, i, batch_info, θ, b, const_cache, ll_cache)
        gi === nothing && return fill(_focei_nan(T), nb, nb)
        _focei_rank1_upper!(I, gi)
    end

    # Prior contribution per RE level.
    re_cache = dm.re_group_info.laplace_cache
    re_names = re_cache.re_names
    θ_re = _symmetrize_psd_params(θ, dm.model.fixed.fixed)
    dists_builder = get_create_random_effect_distribution(dm.model.random.random)
    model_funs = ll_cache.model_funs
    helpers = ll_cache.helpers

    for (ri, re) in enumerate(re_names)
        info = batch_info.re_info[ri]
        isempty(info.map.levels) && continue
        for (li, level_id) in enumerate(info.map.levels)
            rep_idx = info.reps[li]
            const_cov = dm.individuals[rep_idx].const_cov
            dists = dists_builder(θ_re, const_cov, model_funs, helpers)
            dist = getproperty(dists, re)
            r = info.ranges[li]
            if info.is_scalar || info.dim == 1
                idx = first(r)
                s = _focei_prior_score(dist, b[idx])
                isfinite(s) || return fill(_focei_nan(T), nb, nb)
                I[idx, idx] += s * s
            else
                v = view(b, r)
                s = _focei_prior_score(dist, v)
                _focei_allfinite(s) || return fill(_focei_nan(T), nb, nb)
                _focei_rank1_upper!(view(I, r, r), s)
            end
        end
    end
    return _focei_fill_symmetric_from_upper!(I)
end

function _focei_information_matrix(dm::DataModel,
                                   batch_info::_LaplaceBatchInfo,
                                   θ::ComponentArray,
                                   b,
                                   const_cache::LaplaceConstantsCache,
                                   ll_cache::_LLCache,
                                   info_opts::FOCEIInformationOptions)
    mode = info_opts.mode
    (mode == :fisher_common || mode == :custom) ||
        error("FOCEI info_mode must be :fisher_common or :custom.")
    mode == :custom &&
        return _focei_information_from_custom(dm, batch_info, θ, b, const_cache, ll_cache, info_opts)
    nb = batch_info.n_b
    T = promote_type(eltype(θ), eltype(b))
    I = zeros(T, nb, nb)
    nb == 0 && return I

    # Data contribution per individual.
    for i in batch_info.inds
        ok = _focei_add_expected_common_data_info!(I, dm, i, batch_info, θ, b, const_cache, ll_cache)
        if !ok.ok
            reason = get(ok, :reason, :unknown)
            if reason == :unsupported_outcome_distribution
                _focei_error_fisher_common(reason=reason,
                                           distribution=get(ok, :distribution, nothing),
                                           observable=get(ok, :observable, nothing),
                                           individual_index=get(ok, :individual_index, nothing),
                                           observation_row=get(ok, :observation_row, nothing),
                                           note=get(ok, :note, nothing))
            end
            fill!(I, _focei_nan(T))
            return I
        end
    end

    # Prior contribution per RE level.
    re_cache = dm.re_group_info.laplace_cache
    re_names = re_cache.re_names
    θ_re = _symmetrize_psd_params(θ, dm.model.fixed.fixed)
    dists_builder = get_create_random_effect_distribution(dm.model.random.random)
    model_funs = ll_cache.model_funs
    helpers = ll_cache.helpers

    for (ri, re) in enumerate(re_names)
        info = batch_info.re_info[ri]
        isempty(info.map.levels) && continue
        for (li, level_id) in enumerate(info.map.levels)
            rep_idx = info.reps[li]
            const_cov = dm.individuals[rep_idx].const_cov
            dists = dists_builder(θ_re, const_cov, model_funs, helpers)
            dist = getproperty(dists, re)
            r = info.ranges[li]
            status = _focei_add_prior_fisher_common!(I, dist, r)
            if status === :ok
                continue
            end
            if status === :invalid_params
                fill!(I, _focei_nan(T))
                return I
            end
            _focei_error_fisher_common(reason=:unsupported_re_distribution,
                                       distribution=typeof(dist),
                                       re_name=re,
                                       level_id=level_id)
        end
    end
    return _focei_fill_symmetric_from_upper!(I)
end

function _focei_cholesky_info(I_mat::AbstractMatrix;
                              jitter=1e-8,
                              max_tries=6,
                              growth=10.0,
                              adaptive=false,
                              scale_factor=0.0)
    I_sym = Symmetric(I_mat)
    chol = nothing
    jit = jitter
    if adaptive
        scale = mean(abs.(diag(I_sym)))
        scale = isfinite(scale) ? scale : one(real(eltype(I_sym)))
        jit = max(jit, scale_factor * scale)
    end
    for _ in 1:max_tries
        chol = cholesky(I_sym + jit * LinearAlgebra.I, check=false)
        chol.info == 0 && return (chol, jit)
        jit *= growth
    end
    return (chol, jit)
end

function _focei_logdet_information(dm::DataModel,
                                   batch_info::_LaplaceBatchInfo,
                                   θ::ComponentArray,
                                   b,
                                   const_cache::LaplaceConstantsCache,
                                   ll_cache::_LLCache,
                                   info_opts::FOCEIInformationOptions)
    infT = convert(eltype(θ), Inf)
    if batch_info.n_b == 0
        T = eltype(θ)
        I = zeros(T, 0, 0)
        return (zero(T), I, nothing)
    end
    I_mat = _focei_information_matrix(dm, batch_info, θ, b, const_cache, ll_cache, info_opts)
    _focei_allfinite(I_mat) || return (infT, I_mat, nothing)
    chol, _ = _focei_cholesky_info(I_mat;
                                   jitter=info_opts.jitter,
                                   max_tries=info_opts.max_tries,
                                   growth=info_opts.growth,
                                   adaptive=info_opts.adaptive,
                                   scale_factor=info_opts.scale_factor)
    chol === nothing && return (infT, I_mat, nothing)
    chol.info == 0 || return (infT, I_mat, chol)
    logdet = 2 * sum(log, diag(chol.U))
    return (logdet, I_mat, chol)
end

function _focei_grad_batch(dm::DataModel,
                           batch_info::_LaplaceBatchInfo,
                           θ::ComponentArray,
                           b,
                           const_cache::LaplaceConstantsCache,
                           ll_cache::_LLCache,
                           ad_cache::Union{Nothing, LaplaceADCache},
                           bi::Int;
                           info_opts::FOCEIInformationOptions,
                           fallback_hessian::LaplaceHessianOptions,
                           fallback_tracker::Union{Nothing, _FOCEIFallbackTracker}=nothing,
                           rng::AbstractRNG=Random.default_rng())
    nb = batch_info.n_b
    if nb == 0
        Tθ = eltype(θ)
        logf = _laplace_logf_batch(dm, batch_info, θ, b, const_cache, ll_cache)
        grad = ForwardDiff.gradient(θv -> _laplace_logf_batch(dm, batch_info, θv, b, const_cache, ll_cache), θ)
        grad = grad isa ComponentArray ? grad : ComponentArray(grad, getaxes(θ))
        return (logf=logf, logdet=zero(Tθ), grad=grad)
    end

    infT = convert(eltype(θ), Inf)
    logf = _laplace_logf_batch(dm, batch_info, θ, b, const_cache, ll_cache)
    logf == -Inf && return (logf=-Inf, logdet=infT, grad=ComponentArray(zeros(eltype(θ), length(θ)), getaxes(θ)))
    logdet, I_mat, chol = _focei_logdet_information(dm, batch_info, θ, b, const_cache, ll_cache, info_opts)
    if logdet == Inf
        if info_opts.fallback_to_laplace
            _focei_note_fallback!(fallback_tracker, :info_logdet)
            return _laplace_grad_batch(dm, batch_info, θ, b, const_cache, ll_cache, ad_cache, bi;
                                       jitter=fallback_hessian.jitter,
                                       max_tries=fallback_hessian.max_tries,
                                       growth=fallback_hessian.growth,
                                       adaptive=fallback_hessian.adaptive,
                                       scale_factor=fallback_hessian.scale_factor,
                                       use_trace_logdet_grad=fallback_hessian.use_trace_logdet_grad,
                                       use_hutchinson=fallback_hessian.use_hutchinson,
                                       hutchinson_n=fallback_hessian.hutchinson_n,
                                       rng=rng)
        end
        return (logf=-Inf, logdet=infT, grad=ComponentArray(zeros(eltype(θ), length(θ)), getaxes(θ)))
    end

    axs = getaxes(θ)
    θ_vec = θ
    nθ = length(θ_vec)
    nb = length(b)
    buf = ad_cache === nothing ?
          _LaplaceGradBuffers(Vector{eltype(θ_vec)}(undef, nθ),
                              Matrix{eltype(θ_vec)}(undef, nb, nθ),
                              Vector{eltype(θ_vec)}(undef, nθ),
                              Vector{eltype(θ_vec)}(undef, nb),
                              Matrix{eltype(θ_vec)}(undef, _ntri(nb), nθ),
                              Matrix{eltype(θ_vec)}(undef, _ntri(nb), nb),
                              nθ, nb,
                              Vector{eltype(θ_vec)}(undef, nb),
                              Any[], Any[], Any[], Any[], Any[], Any[], Any[]) :
          _get_grad_buffers!(ad_cache, bi, eltype(θ_vec), nθ, nb, true)

    # envelope term
    logf_θ = _LaplaceLogfTheta(dm, batch_info, b, const_cache, ll_cache)
    cfg = _get_fd_cfg!(buf.grad_logf_cfg, logf_θ, θ_vec, () -> ForwardDiff.GradientConfig(logf_θ, θ_vec))
    ForwardDiff.gradient!(buf.grad_logf, logf_θ, θ_vec, cfg)
    grad_logf = buf.grad_logf

    # g(b, θ) = ∂ logf / ∂b
    gθ!(out, θv) = begin
        fb = bv -> _laplace_logf_batch(dm, batch_info, θv, bv, const_cache, ll_cache)
        cfg = _get_fd_cfg!(buf.gθ_grad_cfg, fb, b, () -> ForwardDiff.GradientConfig(fb, b))
        ForwardDiff.gradient!(out, fb, b, cfg)
        return out
    end
    cfg = _get_fd_cfg!(buf.Gθ_cfg, gθ!, θ_vec, () -> ForwardDiff.JacobianConfig(gθ!, buf.gradb_tmp, θ_vec))
    ForwardDiff.jacobian!(buf.Gθ, gθ!, buf.gradb_tmp, θ_vec, cfg)
    Gθ = buf.Gθ

    n = nb
    Ainv = chol \ Matrix{eltype(I_mat)}(LinearAlgebra.I, n, n)
    weights = Vector{eltype(I_mat)}(undef, _ntri(n))
    _vech_weights!(weights, Ainv)

    # ∂ logdet(I) / ∂θ
    fθ = θv -> begin
        Iθ = _focei_information_matrix(dm, batch_info, θv, b, const_cache, ll_cache, info_opts)
        _vech(Iθ)
    end
    cfg = _get_fd_cfg!(buf.Jθ_cfg, fθ, θ_vec, () -> ForwardDiff.JacobianConfig(fθ, θ_vec))
    ForwardDiff.jacobian!(buf.Jθ, fθ, θ_vec, cfg)
    grad_logdet_θ = buf.Jθ' * weights

    # ∂ logdet(I) / ∂b
    fb = bv -> begin
        Ib = _focei_information_matrix(dm, batch_info, θ, bv, const_cache, ll_cache, info_opts)
        _vech(Ib)
    end
    cfg = _get_fd_cfg!(buf.Jb_cfg, fb, b, () -> ForwardDiff.JacobianConfig(fb, b))
    ForwardDiff.jacobian!(buf.Jb, fb, b, cfg)
    grad_logdet_b = buf.Jb' * weights

    # Mode sensitivity uses the true b-Hessian of logf (second-order only):
    # db*/dθ = (-∂²logf/∂b²)^{-1} * ∂²logf/∂b∂θ
    mode_sens = info_opts.mode_sensitivity
    (mode_sens == :exact_hessian || mode_sens == :focei_info) ||
        error("FOCEI mode_sensitivity must be :exact_hessian or :focei_info.")
    chol_mode = nothing
    if mode_sens == :exact_hessian
        H = _laplace_hessian_b(dm, batch_info, θ, b, const_cache, ll_cache, ad_cache, bi; ctx="focei_dbdθ")
        chol_mode, _ = _laplace_cholesky_negH(H;
                                              jitter=fallback_hessian.jitter,
                                              max_tries=fallback_hessian.max_tries,
                                              growth=fallback_hessian.growth,
                                              adaptive=fallback_hessian.adaptive,
                                              scale_factor=fallback_hessian.scale_factor)
        if chol_mode === nothing || chol_mode.info != 0
            if info_opts.fallback_to_laplace
                _focei_note_fallback!(fallback_tracker, :mode_hessian)
                return _laplace_grad_batch(dm, batch_info, θ, b, const_cache, ll_cache, ad_cache, bi;
                                           jitter=fallback_hessian.jitter,
                                           max_tries=fallback_hessian.max_tries,
                                           growth=fallback_hessian.growth,
                                           adaptive=fallback_hessian.adaptive,
                                           scale_factor=fallback_hessian.scale_factor,
                                           use_trace_logdet_grad=fallback_hessian.use_trace_logdet_grad,
                                           use_hutchinson=fallback_hessian.use_hutchinson,
                                           hutchinson_n=fallback_hessian.hutchinson_n,
                                           rng=rng)
            end
            return (logf=-Inf, logdet=infT, grad=ComponentArray(zeros(eltype(θ), length(θ)), getaxes(θ)))
        end
    else
        chol_mode = chol
    end
    dbdθ = chol_mode \ Gθ
    corr = vec(grad_logdet_b' * dbdθ)
    grad = grad_logf .- 0.5 .* (grad_logdet_θ .+ corr)
    return (logf=logf, logdet=logdet, grad=ComponentArray(grad, axs))
end

function _focei_objective_and_grad(dm::DataModel,
                                   batch_infos::Vector{_LaplaceBatchInfo},
                                   θ::ComponentArray,
                                   const_cache::LaplaceConstantsCache,
                                   ll_cache,
                                   ebe_cache::_LaplaceCache;
                                   inner::LaplaceInnerOptions,
                                   info_opts::FOCEIInformationOptions,
                                   fallback_hessian::LaplaceHessianOptions,
                                   cache_opts::LaplaceCacheOptions,
                                   multistart::LaplaceMultistartOptions,
                                   rng::AbstractRNG,
                                   fallback_tracker::Union{Nothing, _FOCEIFallbackTracker}=nothing,
                                   serialization::SciMLBase.EnsembleAlgorithm=EnsembleSerial())
    inner_opts = _resolve_inner_options(inner, dm)
    multistart_opts = _resolve_multistart_options(multistart, inner_opts)

    bstars = _laplace_get_bstar!(ebe_cache, dm, batch_infos, θ, const_cache, ll_cache;
                                 optimizer=inner_opts.optimizer,
                                 optim_kwargs=inner_opts.kwargs,
                                 adtype=inner_opts.adtype,
                                 grad_tol=inner_opts.grad_tol,
                                 theta_tol=cache_opts.theta_tol,
                                 multistart=multistart_opts,
                                 rng=rng,
                                 serialization=serialization)

    infT = convert(eltype(θ), Inf)
    grad = zeros(eltype(θ), length(θ))
    axs = getaxes(θ)
    batch_rngs = _laplace_thread_rngs(rng, length(batch_infos))
    if serialization isa SciMLBase.EnsembleThreads
        nthreads = Threads.maxthreadid()
        caches = _laplace_thread_caches(dm, ll_cache, nthreads)
        obj_by_batch = Vector{Float64}(undef, length(batch_infos))
        grad_by_batch = Matrix{eltype(θ)}(undef, length(θ), length(batch_infos))
        bad = Threads.Atomic{Bool}(false)
        Threads.@threads for bi in eachindex(batch_infos)
            bad[] && continue
            tid = Threads.threadid()
            info = batch_infos[bi]
            b = bstars[bi]
            res = _focei_grad_batch(dm, info, θ, b, const_cache, caches[tid], ebe_cache.ad_cache, bi;
                                    info_opts=info_opts,
                                    fallback_hessian=fallback_hessian,
                                    fallback_tracker=fallback_tracker,
                                    rng=batch_rngs[bi])
            if res.logf == -Inf
                bad[] = true
                continue
            end
            obj_by_batch[bi] = res.logf + 0.5 * info.n_b * log(2π) - 0.5 * res.logdet
            @views grad_by_batch[:, bi] .= res.grad
        end
        bad[] && return (infT, ComponentArray(grad, axs), bstars)
        total = 0.0
        @inbounds for bi in eachindex(batch_infos)
            total += obj_by_batch[bi]
            @views grad .+= grad_by_batch[:, bi]
        end
        return (-total, ComponentArray(-grad, axs), bstars)
    else
        total = 0.0
        ll_cache_local = ll_cache isa Vector ? ll_cache[1] : ll_cache
        for (bi, info) in enumerate(batch_infos)
            b = bstars[bi]
            res = _focei_grad_batch(dm, info, θ, b, const_cache, ll_cache_local, ebe_cache.ad_cache, bi;
                                    info_opts=info_opts,
                                    fallback_hessian=fallback_hessian,
                                    fallback_tracker=fallback_tracker,
                                    rng=batch_rngs[bi])
            res.logf == -Inf && return (infT, ComponentArray(grad, axs), bstars)
            total += res.logf + 0.5 * info.n_b * log(2π) - 0.5 * res.logdet
            grad .+= res.grad
        end
        return (-total, ComponentArray(-grad, axs), bstars)
    end
end

function _focei_objective_only(dm::DataModel,
                               batch_infos::Vector{_LaplaceBatchInfo},
                               θ::ComponentArray,
                               const_cache::LaplaceConstantsCache,
                               ll_cache,
                               ebe_cache::_LaplaceCache;
                               inner::LaplaceInnerOptions,
                               info_opts::FOCEIInformationOptions,
                               fallback_hessian::LaplaceHessianOptions,
                               cache_opts::LaplaceCacheOptions,
                               multistart::LaplaceMultistartOptions,
                               rng::AbstractRNG,
                               fallback_tracker::Union{Nothing, _FOCEIFallbackTracker}=nothing,
                               serialization::SciMLBase.EnsembleAlgorithm=EnsembleSerial())
    inner_opts = _resolve_inner_options(inner, dm)
    multistart_opts = _resolve_multistart_options(multistart, inner_opts)

    bstars = _laplace_get_bstar!(ebe_cache, dm, batch_infos, θ, const_cache, ll_cache;
                                 optimizer=inner_opts.optimizer,
                                 optim_kwargs=inner_opts.kwargs,
                                 adtype=inner_opts.adtype,
                                 grad_tol=inner_opts.grad_tol,
                                 theta_tol=cache_opts.theta_tol,
                                 multistart=multistart_opts,
                                 rng=rng,
                                 serialization=serialization)
    infT = convert(eltype(θ), Inf)
    if serialization isa SciMLBase.EnsembleThreads
        nthreads = Threads.maxthreadid()
        caches = _laplace_thread_caches(dm, ll_cache, nthreads)
        obj_by_batch = Vector{Float64}(undef, length(batch_infos))
        bad = Threads.Atomic{Bool}(false)
        Threads.@threads for bi in eachindex(batch_infos)
            bad[] && continue
            tid = Threads.threadid()
            info = batch_infos[bi]
            b = bstars[bi]
            logf = _laplace_logf_batch(dm, info, θ, b, const_cache, caches[tid])
            logf == -Inf && (bad[] = true; continue)
            logdet, _, _ = _focei_logdet_information(dm, info, θ, b, const_cache, caches[tid], info_opts)
            if logdet == Inf && info_opts.fallback_to_laplace
                _focei_note_fallback!(fallback_tracker, :info_logdet)
                logdet, _, _ = _laplace_logdet_negH(dm, info, θ, b, const_cache, caches[tid], nothing, bi;
                                                    jitter=fallback_hessian.jitter,
                                                    max_tries=fallback_hessian.max_tries,
                                                    growth=fallback_hessian.growth,
                                                    adaptive=fallback_hessian.adaptive,
                                                    scale_factor=fallback_hessian.scale_factor,
                                                    hess_cache=nothing,
                                                    use_cache=false)
            end
            logdet == Inf && (bad[] = true; continue)
            obj_by_batch[bi] = logf + 0.5 * info.n_b * log(2π) - 0.5 * logdet
        end
        bad[] && return infT
        total = 0.0
        @inbounds for bi in eachindex(batch_infos)
            total += obj_by_batch[bi]
        end
    else
        total = 0.0
        ll_cache_local = ll_cache isa Vector ? ll_cache[1] : ll_cache
        for (bi, info) in enumerate(batch_infos)
            b = bstars[bi]
            logf = _laplace_logf_batch(dm, info, θ, b, const_cache, ll_cache_local)
            logf == -Inf && return infT
            logdet, _, _ = _focei_logdet_information(dm, info, θ, b, const_cache, ll_cache_local, info_opts)
            if logdet == Inf && info_opts.fallback_to_laplace
                _focei_note_fallback!(fallback_tracker, :info_logdet)
                logdet, _, _ = _laplace_logdet_negH(dm, info, θ, b, const_cache, ll_cache_local, ebe_cache.ad_cache, bi;
                                                    jitter=fallback_hessian.jitter,
                                                    max_tries=fallback_hessian.max_tries,
                                                    growth=fallback_hessian.growth,
                                                    adaptive=fallback_hessian.adaptive,
                                                    scale_factor=fallback_hessian.scale_factor,
                                                    hess_cache=ebe_cache.hess_cache,
                                                    use_cache=false)
            end
            logdet == Inf && return infT
            total += logf + 0.5 * info.n_b * log(2π) - 0.5 * logdet
        end
    end
    return -total
end

function _fit_model(dm::DataModel, method::FOCEI, args...;
                    constants::NamedTuple=NamedTuple(),
                    constants_re::NamedTuple=NamedTuple(),
                    penalty::NamedTuple=NamedTuple(),
                    ode_args::Tuple=(),
                    ode_kwargs::NamedTuple=NamedTuple(),
                    serialization::SciMLBase.EnsembleAlgorithm=EnsembleSerial(),
                    rng::AbstractRNG=Random.default_rng(),
                    theta_0_untransformed::Union{Nothing, ComponentArray}=nothing,
                    store_data_model::Bool=true)
    fit_kwargs = (constants=constants,
                  constants_re=constants_re,
                  penalty=penalty,
                  ode_args=ode_args,
                  ode_kwargs=ode_kwargs,
                  serialization=serialization,
                  rng=rng,
                  theta_0_untransformed=theta_0_untransformed,
                  store_data_model=store_data_model)
    re_names = get_re_names(dm.model.random.random)
    isempty(re_names) && error("FOCEI requires random effects. Use MLE/MAP for fixed-effects models.")
    fe = dm.model.fixed.fixed
    fixed_names = get_names(fe)
    isempty(fixed_names) && error("FOCEI requires at least one fixed effect.")
    fixed_set = Set(fixed_names)
    for name in keys(constants)
        name in fixed_set || error("Unknown constant parameter $(name).")
    end
    all(name in keys(constants) for name in fixed_names) &&
        error("FOCEI requires at least one free fixed effect. Remove constants or specify a fixed effect or random effect.")

    free_names = [n for n in fixed_names if !(n in keys(constants))]
    θ0_u = get_θ0_untransformed(fe)
    if theta_0_untransformed !== nothing
        for n in fixed_names
            hasproperty(theta_0_untransformed, n) || error("theta_0_untransformed is missing parameter $(n).")
        end
        θ0_u = theta_0_untransformed
    end

    transform = get_transform(fe)
    θ0_t = transform(θ0_u)
    inv_transform = get_inverse_transform(fe)
    θ_const_u = deepcopy(θ0_u)
    _apply_constants!(θ_const_u, constants)
    θ_const_t = transform(θ_const_u)
    inner_opts = _resolve_inner_options(method.inner, dm)
    multistart_opts = _resolve_multistart_options(method.multistart, inner_opts)

    _, batch_infos, const_cache = _build_laplace_batch_infos(dm, constants_re)
    ll_cache = serialization isa SciMLBase.EnsembleThreads ?
               build_ll_cache(dm; ode_args=ode_args, ode_kwargs=ode_kwargs, nthreads=Threads.maxthreadid(), force_saveat=true) :
               build_ll_cache(dm; ode_args=ode_args, ode_kwargs=ode_kwargs, force_saveat=true)
    n_batches = length(batch_infos)
    Tθ = eltype(θ0_t)
    bstar_cache = _LaplaceBStarCache([Vector{Tθ}() for _ in 1:n_batches], falses(n_batches))
    grad_cache = _LaplaceGradCache([Vector{Tθ}() for _ in 1:n_batches],
                                   fill(Tθ(NaN), n_batches),
                                   [Vector{Tθ}() for _ in 1:n_batches],
                                   falses(n_batches))
    ad_cache = _init_laplace_ad_cache(n_batches)
    hess_cache = _init_laplace_hess_cache(Tθ, n_batches)
    ebe_cache = _LaplaceCache(nothing, bstar_cache, grad_cache, ad_cache, hess_cache)

    θ0_free_t = θ0_t[free_names]
    axs_free = getaxes(θ0_free_t)
    axs_full = getaxes(θ_const_t)
    T0 = eltype(θ0_free_t)
    obj_cache = _LaplaceObjCache{T0, ComponentArray}(nothing,
                                                     T0(Inf),
                                                     ComponentArray(zeros(T0, length(θ0_free_t)), axs_free),
                                                     false)
    fallback_tracker = _FOCEIFallbackTracker()

    function obj_only(θt, p)
        θt_free = θt isa ComponentArray ? θt : ComponentArray(θt, axs_free)
        T = eltype(θt_free)
        infT = convert(T, Inf)
        θt_full = ComponentArray(T.(θ_const_t), axs_full)
        for name in free_names
            setproperty!(θt_full, name, getproperty(θt_free, name))
        end
        θu = inv_transform(θt_full)
        obj = _focei_objective_only(dm, batch_infos, θu, const_cache, ll_cache, ebe_cache;
                                    inner=inner_opts,
                                    info_opts=method.info,
                                    fallback_hessian=method.fallback_hessian,
                                    cache_opts=method.cache,
                                    multistart=multistart_opts,
                                    rng=rng,
                                    fallback_tracker=fallback_tracker,
                                    serialization=serialization)
        obj == Inf && return infT
        obj += _penalty_value(θu, penalty)
        _laplace_obj_cache_set_obj!(obj_cache, θt_free, obj)
        return obj
    end

    function obj_grad(θt, p)
        θt_free = θt isa ComponentArray ? θt : ComponentArray(θt, axs_free)
        θt_vec = θt_free
        cached = _laplace_obj_cache_lookup(obj_cache, θt_vec, method.cache.theta_tol)
        if cached !== nothing
            return cached
        end
        T = eltype(θt_free)
        infT = convert(T, Inf)
        θt_full = ComponentArray(T.(θ_const_t), axs_full)
        for name in free_names
            setproperty!(θt_full, name, getproperty(θt_free, name))
        end
        θu = inv_transform(θt_full)
        obj, grad_full, _ = _focei_objective_and_grad(dm, batch_infos, θu, const_cache, ll_cache, ebe_cache;
                                                      inner=inner_opts,
                                                      info_opts=method.info,
                                                      fallback_hessian=method.fallback_hessian,
                                                      cache_opts=method.cache,
                                                      multistart=multistart_opts,
                                                      rng=rng,
                                                      fallback_tracker=fallback_tracker,
                                                      serialization=serialization)
        obj == Inf && return (infT, ComponentArray(zeros(T, length(θt_free)), axs_free))
        grad_u = grad_full
        grad_t_ca = apply_inv_jacobian_T(inv_transform, θt_full, grad_u)
        grad_free = similar(θt_free)
        for name in free_names
            setproperty!(grad_free, name, getproperty(grad_t_ca, name))
        end
        obj += _penalty_value(θu, penalty)
        _laplace_obj_cache_set_obj_grad!(obj_cache, θt_free, obj, grad_free)
        return (obj, grad_free)
    end

    optf = OptimizationFunction(obj_only,
                                method.adtype;
                                grad = (G, θt, p) -> begin
                                    grad = obj_grad(θt, p)[2]
                                    G .= grad
                                end)
    lower_t, upper_t = get_bounds_transformed(fe)
    lower_t_free = lower_t[free_names]
    upper_t_free = upper_t[free_names]
    use_bounds = !(all(isinf, lower_t_free) && all(isinf, upper_t_free))
    user_bounds = method.lb !== nothing || method.ub !== nothing
    if user_bounds && !isempty(keys(constants))
        @info "Bounds for constant parameters are ignored." constants=collect(keys(constants))
    end
    if user_bounds
        lb = method.lb
        ub = method.ub
        if lb isa ComponentArray
            lb = lb[free_names]
        end
        if ub isa ComponentArray
            ub = ub[free_names]
        end
    else
        lb = lower_t_free
        ub = upper_t_free
    end
    use_bounds = use_bounds || user_bounds
    if parentmodule(typeof(method.optimizer)) === OptimizationBBO && !use_bounds
        error("BlackBoxOptim methods require finite bounds. Add lower/upper bounds in @fixedEffects (on transformed scale) or pass them via FOCEI(lb=..., ub=...). A quick helper is default_bounds_from_start(dm; margin=...).")
    end
    if parentmodule(typeof(method.optimizer)) === OptimizationBBO
        θ0_init = collect(θ0_free_t)
    else
        θ0_init = θ0_free_t
    end
    prob = use_bounds ? OptimizationProblem(optf, θ0_init; lb=lb, ub=ub) :
                        OptimizationProblem(optf, θ0_init)
    sol = Optimization.solve(prob, method.optimizer; method.optim_kwargs...)

    θ_hat_t_raw = sol.u
    θ_hat_t_free = θ_hat_t_raw isa ComponentArray ? θ_hat_t_raw : ComponentArray(θ_hat_t_raw, axs_free)
    T = eltype(θ_hat_t_free)
    θ_hat_t = ComponentArray(T.(θ_const_t), axs_full)
    for name in free_names
        setproperty!(θ_hat_t, name, getproperty(θ_hat_t_free, name))
    end
    θ_hat_u = inv_transform(θ_hat_t)

    # Evaluate fallback use at the reported optimum.
    fallback_tracker_solution = _FOCEIFallbackTracker()
    _focei_objective_and_grad(dm, batch_infos, θ_hat_u, const_cache, ll_cache, ebe_cache;
                              inner=inner_opts,
                              info_opts=method.info,
                              fallback_hessian=method.fallback_hessian,
                              cache_opts=method.cache,
                              multistart=multistart_opts,
                              rng=rng,
                              fallback_tracker=fallback_tracker_solution,
                              serialization=serialization)
    notes_total = _focei_fallback_notes(fallback_tracker)
    notes_sol = _focei_fallback_notes(fallback_tracker_solution)
    notes = merge(notes_total,
                  (focei_fallback_at_solution=notes_sol.focei_fallback_total > 0,
                   focei_fallback_total_at_solution=notes_sol.focei_fallback_total,
                   focei_fallback_info_logdet_at_solution=notes_sol.focei_fallback_info_logdet,
                   focei_fallback_mode_hessian_at_solution=notes_sol.focei_fallback_mode_hessian))
    if notes.focei_fallback_total > 0
        @warn "FOCEI fell back to Laplace derivatives in some evaluations." focei_fallback_total=notes.focei_fallback_total focei_fallback_info_logdet=notes.focei_fallback_info_logdet focei_fallback_mode_hessian=notes.focei_fallback_mode_hessian focei_fallback_at_solution=notes.focei_fallback_at_solution
    end

    summary = FitSummary(sol.objective, sol.retcode == SciMLBase.ReturnCode.Success,
                         FitParameters(θ_hat_t, θ_hat_u),
                         notes)
    diagnostics = FitDiagnostics((;), (optimizer=method.optimizer,), (retcode=sol.retcode,), NamedTuple())
    niter = hasproperty(sol, :stats) && hasproperty(sol.stats, :iterations) ? sol.stats.iterations : missing
    raw = hasproperty(sol, :original) ? sol.original : sol
    result = FOCEIResult(sol, sol.objective, niter, raw, notes, ebe_cache.bstar_cache.b_star)
    return FitResult(method, result, summary, diagnostics, store_data_model ? dm : nothing, args, fit_kwargs)
end

function _fit_model(dm::DataModel, method::FOCEIMAP, args...;
                    constants::NamedTuple=NamedTuple(),
                    constants_re::NamedTuple=NamedTuple(),
                    penalty::NamedTuple=NamedTuple(),
                    ode_args::Tuple=(),
                    ode_kwargs::NamedTuple=NamedTuple(),
                    serialization::SciMLBase.EnsembleAlgorithm=EnsembleSerial(),
                    rng::AbstractRNG=Random.default_rng(),
                    theta_0_untransformed::Union{Nothing, ComponentArray}=nothing,
                    store_data_model::Bool=true)
    fit_kwargs = (constants=constants,
                  constants_re=constants_re,
                  penalty=penalty,
                  ode_args=ode_args,
                  ode_kwargs=ode_kwargs,
                  serialization=serialization,
                  rng=rng,
                  theta_0_untransformed=theta_0_untransformed,
                  store_data_model=store_data_model)
    re_names = get_re_names(dm.model.random.random)
    isempty(re_names) && error("FOCEIMAP requires random effects. Use MAP for fixed-effects models.")
    fe = dm.model.fixed.fixed
    priors = get_priors(fe)
    fixed_names = get_names(fe)
    isempty(fixed_names) && error("FOCEIMAP requires at least one fixed effect.")
    for name in fixed_names
        hasproperty(priors, name) || error("FOCEIMAP requires priors on all fixed effects. Missing prior for $(name).")
        getfield(priors, name) isa Priorless && error("FOCEIMAP requires priors on all fixed effects. Priorless for $(name).")
    end

    fixed_set = Set(fixed_names)
    for name in keys(constants)
        name in fixed_set || error("Unknown constant parameter $(name).")
    end
    all(name in keys(constants) for name in fixed_names) &&
        error("FOCEIMAP requires at least one free fixed effect. Remove constants or specify a fixed effect or random effect.")

    free_names = [n for n in fixed_names if !(n in keys(constants))]
    θ0_u = get_θ0_untransformed(fe)
    if theta_0_untransformed !== nothing
        for n in fixed_names
            hasproperty(theta_0_untransformed, n) || error("theta_0_untransformed is missing parameter $(n).")
        end
        θ0_u = theta_0_untransformed
    end

    transform = get_transform(fe)
    θ0_t = transform(θ0_u)
    inv_transform = get_inverse_transform(fe)
    θ_const_u = deepcopy(θ0_u)
    _apply_constants!(θ_const_u, constants)
    θ_const_t = transform(θ_const_u)
    inner_opts = _resolve_inner_options(method.inner, dm)
    multistart_opts = _resolve_multistart_options(method.multistart, inner_opts)

    _, batch_infos, const_cache = _build_laplace_batch_infos(dm, constants_re)
    ll_cache = serialization isa SciMLBase.EnsembleThreads ?
               build_ll_cache(dm; ode_args=ode_args, ode_kwargs=ode_kwargs, nthreads=Threads.maxthreadid(), force_saveat=true) :
               build_ll_cache(dm; ode_args=ode_args, ode_kwargs=ode_kwargs, force_saveat=true)
    n_batches = length(batch_infos)
    Tθ = eltype(θ0_t)
    bstar_cache = _LaplaceBStarCache([Vector{Tθ}() for _ in 1:n_batches], falses(n_batches))
    grad_cache = _LaplaceGradCache([Vector{Tθ}() for _ in 1:n_batches],
                                   fill(Tθ(NaN), n_batches),
                                   [Vector{Tθ}() for _ in 1:n_batches],
                                   falses(n_batches))
    ad_cache = _init_laplace_ad_cache(n_batches)
    hess_cache = _init_laplace_hess_cache(Tθ, n_batches)
    ebe_cache = _LaplaceCache(nothing, bstar_cache, grad_cache, ad_cache, hess_cache)

    θ0_free_t = θ0_t[free_names]
    axs_free = getaxes(θ0_free_t)
    axs_full = getaxes(θ_const_t)
    T0 = eltype(θ0_free_t)
    obj_cache = _LaplaceObjCache{T0, ComponentArray}(nothing,
                                                     T0(Inf),
                                                     ComponentArray(zeros(T0, length(θ0_free_t)), axs_free),
                                                     false)
    fallback_tracker = _FOCEIFallbackTracker()

    function obj_only(θt, p)
        θt_free = θt isa ComponentArray ? θt : ComponentArray(θt, axs_free)
        T = eltype(θt_free)
        infT = convert(T, Inf)
        θt_full = ComponentArray(T.(θ_const_t), axs_full)
        for name in free_names
            setproperty!(θt_full, name, getproperty(θt_free, name))
        end
        θu = inv_transform(θt_full)
        obj = _focei_objective_only(dm, batch_infos, θu, const_cache, ll_cache, ebe_cache;
                                    inner=inner_opts,
                                    info_opts=method.info,
                                    fallback_hessian=method.fallback_hessian,
                                    cache_opts=method.cache,
                                    multistart=multistart_opts,
                                    rng=rng,
                                    fallback_tracker=fallback_tracker,
                                    serialization=serialization)
        obj == Inf && return infT
        lp = logprior(fe, θu)
        lp == -Inf && return infT
        obj += -lp + _penalty_value(θu, penalty)
        _laplace_obj_cache_set_obj!(obj_cache, θt_free, obj)
        return obj
    end

    function obj_grad(θt, p)
        θt_free = θt isa ComponentArray ? θt : ComponentArray(θt, axs_free)
        θt_vec = θt_free
        cached = _laplace_obj_cache_lookup(obj_cache, θt_vec, method.cache.theta_tol)
        if cached !== nothing
            return cached
        end
        T = eltype(θt_free)
        infT = convert(T, Inf)
        θt_full = ComponentArray(T.(θ_const_t), axs_full)
        for name in free_names
            setproperty!(θt_full, name, getproperty(θt_free, name))
        end
        θu = inv_transform(θt_full)
        obj, grad_full, _ = _focei_objective_and_grad(dm, batch_infos, θu, const_cache, ll_cache, ebe_cache;
                                                      inner=inner_opts,
                                                      info_opts=method.info,
                                                      fallback_hessian=method.fallback_hessian,
                                                      cache_opts=method.cache,
                                                      multistart=multistart_opts,
                                                      rng=rng,
                                                      fallback_tracker=fallback_tracker,
                                                      serialization=serialization)
        obj == Inf && return (infT, ComponentArray(zeros(T, length(θt_free)), axs_free))
        lp = logprior(fe, θu)
        lp == -Inf && return (infT, ComponentArray(zeros(T, length(θt_free)), axs_free))
        penalty_val = _penalty_value(θu, penalty)
        obj += -lp + penalty_val

        lp_grad = ForwardDiff.gradient(x -> logprior(fe, x), θu)
        pen_grad = ForwardDiff.gradient(x -> _penalty_value(x, penalty), θu)
        grad_u = grad_full .- lp_grad .+ pen_grad

        grad_t_ca = apply_inv_jacobian_T(inv_transform, θt_full, grad_u)
        grad_free = similar(θt_free)
        for name in free_names
            setproperty!(grad_free, name, getproperty(grad_t_ca, name))
        end
        _laplace_obj_cache_set_obj_grad!(obj_cache, θt_free, obj, grad_free)
        return (obj, grad_free)
    end

    optf = OptimizationFunction(obj_only,
                                method.adtype;
                                grad = (G, θt, p) -> begin
                                    grad = obj_grad(θt, p)[2]
                                    G .= grad
                                end)
    lower_t, upper_t = get_bounds_transformed(fe)
    lower_t_free = lower_t[free_names]
    upper_t_free = upper_t[free_names]
    use_bounds = !(all(isinf, lower_t_free) && all(isinf, upper_t_free))
    user_bounds = method.lb !== nothing || method.ub !== nothing
    if user_bounds && !isempty(keys(constants))
        @info "Bounds for constant parameters are ignored." constants=collect(keys(constants))
    end
    if user_bounds
        lb = method.lb
        ub = method.ub
        if lb isa ComponentArray
            lb = lb[free_names]
        end
        if ub isa ComponentArray
            ub = ub[free_names]
        end
    else
        lb = lower_t_free
        ub = upper_t_free
    end
    use_bounds = use_bounds || user_bounds
    if parentmodule(typeof(method.optimizer)) === OptimizationBBO && !use_bounds
        error("BlackBoxOptim methods require finite bounds. Add lower/upper bounds in @fixedEffects (on transformed scale) or pass them via FOCEIMAP(lb=..., ub=...). A quick helper is default_bounds_from_start(dm; margin=...).")
    end
    if parentmodule(typeof(method.optimizer)) === OptimizationBBO
        θ0_init = collect(θ0_free_t)
    else
        θ0_init = θ0_free_t
    end
    prob = use_bounds ? OptimizationProblem(optf, θ0_init; lb=lb, ub=ub) :
                        OptimizationProblem(optf, θ0_init)
    sol = Optimization.solve(prob, method.optimizer; method.optim_kwargs...)

    θ_hat_t_raw = sol.u
    θ_hat_t_free = θ_hat_t_raw isa ComponentArray ? θ_hat_t_raw : ComponentArray(θ_hat_t_raw, axs_free)
    T = eltype(θ_hat_t_free)
    θ_hat_t = ComponentArray(T.(θ_const_t), axs_full)
    for name in free_names
        setproperty!(θ_hat_t, name, getproperty(θ_hat_t_free, name))
    end
    θ_hat_u = inv_transform(θ_hat_t)

    # Evaluate fallback use at the reported optimum.
    fallback_tracker_solution = _FOCEIFallbackTracker()
    _focei_objective_and_grad(dm, batch_infos, θ_hat_u, const_cache, ll_cache, ebe_cache;
                              inner=inner_opts,
                              info_opts=method.info,
                              fallback_hessian=method.fallback_hessian,
                              cache_opts=method.cache,
                              multistart=multistart_opts,
                              rng=rng,
                              fallback_tracker=fallback_tracker_solution,
                              serialization=serialization)
    notes_total = _focei_fallback_notes(fallback_tracker)
    notes_sol = _focei_fallback_notes(fallback_tracker_solution)
    notes = merge(notes_total,
                  (focei_fallback_at_solution=notes_sol.focei_fallback_total > 0,
                   focei_fallback_total_at_solution=notes_sol.focei_fallback_total,
                   focei_fallback_info_logdet_at_solution=notes_sol.focei_fallback_info_logdet,
                   focei_fallback_mode_hessian_at_solution=notes_sol.focei_fallback_mode_hessian))
    if notes.focei_fallback_total > 0
        @warn "FOCEIMAP fell back to Laplace derivatives in some evaluations." focei_fallback_total=notes.focei_fallback_total focei_fallback_info_logdet=notes.focei_fallback_info_logdet focei_fallback_mode_hessian=notes.focei_fallback_mode_hessian focei_fallback_at_solution=notes.focei_fallback_at_solution
    end

    summary = FitSummary(sol.objective, sol.retcode == SciMLBase.ReturnCode.Success,
                         FitParameters(θ_hat_t, θ_hat_u),
                         notes)
    diagnostics = FitDiagnostics((;), (optimizer=method.optimizer,), (retcode=sol.retcode,), NamedTuple())
    niter = hasproperty(sol, :stats) && hasproperty(sol.stats, :iterations) ? sol.stats.iterations : missing
    raw = hasproperty(sol, :original) ? sol.original : sol
    result = FOCEIMAPResult(sol, sol.objective, niter, raw, notes, ebe_cache.bstar_cache.b_star)
    return FitResult(method, result, summary, diagnostics, store_data_model ? dm : nothing, args, fit_kwargs)
end
