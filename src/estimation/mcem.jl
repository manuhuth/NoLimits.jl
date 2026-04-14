export MCEM
export MCEM_MCMC
export MCEM_IS
export MCEMResult

using Optimization
using OptimizationOptimJL
using SciMLBase
using ComponentArrays
using Random
using LineSearches
using OptimizationBBO
using Turing
using DynamicPPL
using Distributions
using LinearAlgebra
using ProgressMeter
using Statistics
import ForwardDiff

const _MCEM_MODEL_CACHE = Dict{Tuple{Tuple{Vararg{Symbol}}}, Symbol}()
const _MCEM_MODEL_CACHE_LOCK = ReentrantLock()

struct _MCEMQCache{T}
    partial_obj::Vector{T}
end

function _init_mcem_q_cache(::Type{T}, serialization) where {T}
    return _MCEMQCache{T}(Vector{T}())
end

@inline function _mcem_thread_caches(dm::DataModel, ll_cache, nthreads::Int)
    if ll_cache isa Vector
        return ll_cache
    elseif ll_cache isa _LLCache
        caches = build_ll_cache(dm;
                                ode_args=ll_cache.ode_args,
                                ode_kwargs=ll_cache.ode_kwargs,
                                force_saveat=ll_cache.saveat_cache !== nothing,
                                nthreads=nthreads)
        return caches isa Vector ? caches : [caches]
    else
        caches = build_ll_cache(dm; nthreads=nthreads)
        return caches isa Vector ? caches : [caches]
    end
end

@inline function _mcem_thread_rngs(rng::AbstractRNG, nthreads::Int)
    return _spawn_child_rngs(rng, nthreads)
end

# ---------------------------------------------------------------------------
# E-step strategy types
# ---------------------------------------------------------------------------

"""
    AbstractMCEMEStep

Abstract supertype for MCEM E-step strategies. Concrete subtypes:
- [`MCEM_MCMC`](@ref) — Turing.jl MCMC sampler (default)
- [`MCEM_IS`](@ref)   — Importance Sampling with an adaptive proposal
"""
abstract type AbstractMCEMEStep end

"""
    MCEM_MCMC(; sampler, turing_kwargs, sample_schedule, warm_start)

MCMC-based E-step for [`MCEM`](@ref). Wraps a Turing.jl-compatible sampler.

# Keyword Arguments
- `sampler`: Turing-compatible sampler. Defaults to `NUTS(0.75)`.
- `turing_kwargs::NamedTuple`: forwarded to `Turing.sample`.
- `sample_schedule`: samples per E-step — `Int`, `Vector{Int}`, or `Function(iter)->Int`.
- `warm_start::Bool = true`: initialise sampler from previous iteration's modes.
"""
struct MCEM_MCMC{S, K, SS} <: AbstractMCEMEStep
    sampler::S
    turing_kwargs::K
    sample_schedule::SS
    warm_start::Bool
end

function MCEM_MCMC(; sampler=Turing.NUTS(0.75),
                     turing_kwargs=NamedTuple(),
                     sample_schedule=250,
                     warm_start::Bool=true)
    return MCEM_MCMC(sampler, turing_kwargs, sample_schedule, warm_start)
end

"""
    MCEM_IS(; n_samples, proposal, adapt, warm_start_mcmc_iters, mcmc_warmup)

Importance Sampling E-step for [`MCEM`](@ref).

Draws `n_samples` random-effect vectors from a proposal distribution `q(b)` and
reweights them by `log p(y, b | θ_k) - log q(b)` to form a self-normalised
Monte Carlo approximation of the Q-function.

# Keyword Arguments
- `n_samples::Int = 500`: number of IS draws per E-step.
- `proposal`: proposal mode, one of:
  - `:prior` — draw each RE level from its current prior `p(b | θ_k)`.
  - `:gaussian` (default) — block-diagonal Gaussian in bijected space, adapted
    from previous iteration's samples via Haario-Welford statistics.
  - `Function` — user-supplied function with signature
    `(θ, batch_info, re_dists, rng, n_samples) -> (samples::Matrix, log_qs::Vector)`.
- `adapt::Bool = true`: update Gaussian proposal blocks after each IS iteration.
- `warm_start_mcmc_iters::Int = 0`: run this many MCMC iterations before switching
  to IS. If > 0, `mcmc_warmup` must be provided (or defaults are used).
- `mcmc_warmup`: an [`MCEM_MCMC`](@ref) for the warm-up phase, or `nothing` to use
  default MCMC options.
"""
struct MCEM_IS{PR, WM} <: AbstractMCEMEStep
    n_samples::Int
    proposal::PR
    adapt::Bool
    warm_start_mcmc_iters::Int
    mcmc_warmup::WM   # MCEM_MCMC or nothing
end

function MCEM_IS(; n_samples::Int=500,
                   proposal=:gaussian,
                   adapt::Bool=true,
                   warm_start_mcmc_iters::Int=0,
                   mcmc_warmup=nothing)
    n_samples > 0 || throw(ArgumentError("n_samples must be > 0"))
    warm_start_mcmc_iters >= 0 || throw(ArgumentError("warm_start_mcmc_iters must be ≥ 0"))
    if warm_start_mcmc_iters > 0 && mcmc_warmup === nothing
        mcmc_warmup = MCEM_MCMC()
    end
    return MCEM_IS(n_samples, proposal, adapt, warm_start_mcmc_iters, mcmc_warmup)
end

# Backward-compatible internal alias (kept for reference in constructor logic)
struct MCEMOptions{S, K, SS, W, V, P}
    sampler::S
    turing_kwargs::K
    sample_schedule::SS
    warm_start::W
    verbose::V
    progress::P
end

struct EMOptions{T}
    maxiters::Int
    rtol_theta::T
    atol_theta::T
    rtol_Q::T
    atol_Q::T
    consecutive_params::Int
end

"""
    MCEM(; optimizer, optim_kwargs, adtype, e_step,
           sampler, turing_kwargs, sample_schedule, warm_start,
           verbose, progress, maxiters, rtol_theta, atol_theta, rtol_Q,
           atol_Q, consecutive_params, ebe_optimizer, ebe_optim_kwargs, ebe_adtype,
           ebe_grad_tol, ebe_multistart_n, ebe_multistart_k, ebe_multistart_max_rounds,
           ebe_multistart_sampling, ebe_rescue_on_high_grad, ebe_rescue_multistart_n,
           ebe_rescue_multistart_k, ebe_rescue_max_rounds, ebe_rescue_grad_tol,
           ebe_rescue_multistart_sampling, lb, ub) <: FittingMethod

Monte Carlo Expectation-Maximisation for random-effects models. At each EM iteration the
E-step draws random effects; the M-step maximises the Monte Carlo Q-function over the
fixed effects.

# Keyword Arguments
- `optimizer`: M-step Optimization.jl optimiser. Defaults to `LBFGS` with backtracking.
- `optim_kwargs::NamedTuple = NamedTuple()`: keyword arguments for the M-step `solve`.
- `adtype`: AD backend for the M-step. Defaults to `AutoForwardDiff()`.
- `e_step`: E-step strategy. Either [`MCEM_MCMC`](@ref) or [`MCEM_IS`](@ref). When
  omitted, defaults to `MCEM_MCMC` constructed from the legacy keyword arguments below.
- `sampler`: (legacy) Turing sampler; used when `e_step` is not provided.
- `turing_kwargs::NamedTuple`: (legacy) forwarded to `Turing.sample`.
- `sample_schedule::Int = 250`: (legacy) MCMC samples per E-step iteration.
- `warm_start::Bool = true`: (legacy) initialise sampler from previous iteration's modes.
- `verbose::Bool = false`: print per-iteration diagnostics.
- `progress::Bool = true`: show a progress bar.
- `maxiters::Int = 100`: maximum number of EM iterations.
- `rtol_theta`, `atol_theta`: relative/absolute convergence tolerance on fixed effects.
- `rtol_Q`, `atol_Q`: relative/absolute convergence tolerance on the Q-function.
- `consecutive_params::Int = 3`: consecutive iterations satisfying tolerance to converge.
- `ebe_optimizer`, `ebe_optim_kwargs`, `ebe_adtype`, `ebe_grad_tol`: EBE inner optimiser.
- `ebe_multistart_n`, `ebe_multistart_k`, `ebe_multistart_max_rounds`,
  `ebe_multistart_sampling`: multistart settings for EBE mode computation.
- `ebe_rescue_on_high_grad`, `ebe_rescue_multistart_n`, `ebe_rescue_multistart_k`,
  `ebe_rescue_max_rounds`, `ebe_rescue_grad_tol`, `ebe_rescue_multistart_sampling`:
  rescue multistart settings when an EBE mode has a high gradient norm.
- `lb`, `ub`: bounds on the transformed fixed-effect scale, or `nothing`.
"""
struct MCEM{O, K, A, ES, EO, EB, ER, L, U} <: FittingMethod
    optimizer::O
    optim_kwargs::K
    adtype::A
    e_step::ES           # AbstractMCEMEStep (MCEM_MCMC or MCEM_IS)
    em::EO
    ebe::EB
    ebe_rescue::ER
    lb::L
    ub::U
    verbose::Bool
    progress::Bool
end

MCEM(; optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
     optim_kwargs=NamedTuple(),
     adtype=Optimization.AutoForwardDiff(),
     e_step=nothing,
     sampler=Turing.NUTS(0.75),
     turing_kwargs=NamedTuple(),
     sample_schedule=250,
     warm_start=true,
     verbose=false,
     progress=true,
     maxiters=100,
     rtol_theta=1e-4,
     atol_theta=1e-6,
     rtol_Q=1e-4,
     atol_Q=1e-6,
     consecutive_params=3,
     ebe_optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking(maxstep=1.0)),
     ebe_optim_kwargs=NamedTuple(),
     ebe_adtype=Optimization.AutoForwardDiff(),
     ebe_grad_tol=:auto,
     ebe_multistart_n=50,
     ebe_multistart_k=10,
     ebe_multistart_max_rounds=5,
     ebe_multistart_sampling=:lhs,
     ebe_rescue_on_high_grad=true,
     ebe_rescue_multistart_n=128,
     ebe_rescue_multistart_k=32,
     ebe_rescue_max_rounds=8,
     ebe_rescue_grad_tol=ebe_grad_tol,
     ebe_rescue_multistart_sampling=ebe_multistart_sampling,
     lb=nothing,
     ub=nothing,
     ) = begin
    # When e_step is not provided, build MCEM_MCMC from legacy kwargs (backward compat)
    e_step_actual = if e_step === nothing
        MCEM_MCMC(sampler, turing_kwargs, sample_schedule, warm_start)
    else
        e_step
    end
    em = EMOptions(maxiters, rtol_theta, atol_theta, rtol_Q, atol_Q, consecutive_params)
    ebe = EBEOptions(ebe_optimizer, ebe_optim_kwargs, ebe_adtype, ebe_grad_tol,
                     ebe_multistart_n, ebe_multistart_k, ebe_multistart_max_rounds,
                     ebe_multistart_sampling)
    ebe_rescue = EBERescueOptions(ebe_rescue_on_high_grad, ebe_rescue_multistart_n,
                                   ebe_rescue_multistart_k, ebe_rescue_max_rounds,
                                   ebe_rescue_grad_tol, ebe_rescue_multistart_sampling)
    MCEM(optimizer, optim_kwargs, adtype, e_step_actual, em, ebe, ebe_rescue,
         lb, ub, verbose, progress)
end

"""
    MCEMResult{S, O, I, R, N, B} <: MethodResult

Method-specific result from a [`MCEM`](@ref) fit. Stores the solution, objective value,
iteration count, raw solver result, optional notes, and final empirical-Bayes mode
estimates for each individual.
"""
struct MCEMResult{S, O, I, R, N, B} <: MethodResult
    solution::S
    objective::O
    iterations::I
    raw::R
    notes::N
    eb_modes::B
end

mutable struct _MCEMDiagnostics{T}
    θ_hist::Vector{AbstractVector{T}}
    Q_hist::Vector{T}
    dθ_abs::Vector{T}
    dθ_rel::Vector{T}
    dQ_abs::Vector{T}
    dQ_rel::Vector{T}
    samples::Vector{Int}
    ess_hist::Vector{Float64}   # mean ESS across batches; NaN for MCMC iterations
end

function _mcem_schedule(samplespec, iter::Int)
    samplespec === nothing && return 0
    samplespec isa Integer && return samplespec
    samplespec isa AbstractVector && return samplespec[min(iter, length(samplespec))]
    return samplespec(iter)
end

function _build_mcem_batch_model(re_names::Vector{Symbol})
    key = (Tuple(re_names),)
    lock(_MCEM_MODEL_CACHE_LOCK)
    if haskey(_MCEM_MODEL_CACHE, key)
        val = _MCEM_MODEL_CACHE[key]
        unlock(_MCEM_MODEL_CACHE_LOCK)
        return val
    end
    unlock(_MCEM_MODEL_CACHE_LOCK)
    fname = gensym(:_mcem_batch_model)
    sample_blocks = Expr(:block)
    re_val_syms = Symbol[]
    for (ri, re) in enumerate(re_names)
        re_q = QuoteNode(re)
        meta_sym = Symbol(re, :_meta)
        levels_sym = Symbol(re, :_levels)
        reps_sym = Symbol(re, :_reps)
        ranges_sym = Symbol(re, :_ranges)
        vals_sym = Symbol(re, :_vals)
        push!(re_val_syms, vals_sym)
        meta_get = :(info.re_info[$ri])
        levels_get = :(getproperty(getproperty($meta_sym, :map), :levels))
        reps_get = :(getproperty($meta_sym, :reps))
        ranges_get = :(getproperty($meta_sym, :ranges))
        is_scalar = :(getproperty($meta_sym, :is_scalar))
        scalar_block = quote
            local nlvls = length($levels_sym)
            if nlvls > 0
                const_cov = dm.individuals[$reps_sym[1]].const_cov
                dists = dists_builder(θ_re, const_cov, model_funs, helpers)
                dist = getproperty(dists, $re_q)
                if _has_anneal && haskey(anneal_sds, $re_q)
                    dist = _saem_apply_anneal_dist(dist, getfield(anneal_sds, $re_q))
                end
                v1 = ($(Symbol(re, :_v1)) ~ dist)
                local $vals_sym = Vector{typeof(v1)}(undef, nlvls)
                $vals_sym[1] = v1
                for j in 2:nlvls
                    const_cov = dm.individuals[$reps_sym[j]].const_cov
                    dists = dists_builder(θ_re, const_cov, model_funs, helpers)
                    dist = getproperty(dists, $re_q)
                    if _has_anneal && haskey(anneal_sds, $re_q)
                        dist = _saem_apply_anneal_dist(dist, getfield(anneal_sds, $re_q))
                    end
                    $vals_sym[j] ~ dist
                end
            else
                local $vals_sym = Vector{Float64}(undef, 0)
            end
        end
        vector_block = quote
            local nlvls = length($levels_sym)
            if nlvls > 0
                const_cov = dm.individuals[$reps_sym[1]].const_cov
                dists = dists_builder(θ_re, const_cov, model_funs, helpers)
                dist = getproperty(dists, $re_q)
                if _has_anneal && haskey(anneal_sds, $re_q)
                    dist = _saem_apply_anneal_dist(dist, getfield(anneal_sds, $re_q))
                end
                v1 = ($(Symbol(re, :_v1)) ~ dist)
                local $vals_sym = Vector{typeof(v1)}(undef, nlvls)
                $vals_sym[1] = v1
                for j in 2:nlvls
                    const_cov = dm.individuals[$reps_sym[j]].const_cov
                    dists = dists_builder(θ_re, const_cov, model_funs, helpers)
                    dist = getproperty(dists, $re_q)
                    if _has_anneal && haskey(anneal_sds, $re_q)
                        dist = _saem_apply_anneal_dist(dist, getfield(anneal_sds, $re_q))
                    end
                    $vals_sym[j] ~ dist
                end
            else
                local $vals_sym = Vector{Vector{Float64}}(undef, 0)
            end
        end
        push!(sample_blocks.args, :(local $meta_sym = $meta_get))
        push!(sample_blocks.args, :(local $levels_sym = $levels_get))
        push!(sample_blocks.args, :(local $reps_sym = $reps_get))
        push!(sample_blocks.args, :(local $ranges_sym = $ranges_get))
        push!(sample_blocks.args, :(if $is_scalar
                                         $scalar_block
                                     else
                                         $vector_block
                                     end))
    end
    re_samples_expr = Expr(:call,
                           Expr(:curly, :NamedTuple, Expr(:tuple, QuoteNode.(re_names)...)),
                           Expr(:tuple, re_val_syms...))

    ex = quote
        @model function $(fname)(dm, info, θ, const_cache, cache, anneal_sds=NamedTuple())
            θ_re = _symmetrize_psd_params(θ, dm.model.fixed.fixed)
            dists_builder = get_create_random_effect_distribution(dm.model.random.random)
            model_funs = cache.model_funs
            helpers = cache.helpers
            _has_anneal = !isempty(anneal_sds)

            $sample_blocks
            re_samples = $re_samples_expr

            Tb = eltype(θ)
            for re in $re_names
                vals = getproperty(re_samples, re)
                if !isempty(vals)
                    v1 = vals[1]
                    Tb = v1 isa AbstractVector ? eltype(v1) : typeof(v1)
                    break
                end
            end

            nb = info.n_b
            b = Vector{Tb}(undef, nb)
            for (ri, re) in enumerate($re_names)
                meta = info.re_info[ri]
                levels = getproperty(getproperty(meta, :map), :levels)
                ranges = meta.ranges
                vals = getproperty(re_samples, re)
                for (li, _) in enumerate(levels)
                    r = ranges[li]
                    if meta.is_scalar
                        b[first(r)] = vals[li]
                    else
                        b[r] .= vals[li]
                    end
                end
            end

            ll = zero(Tb)
            for i in info.inds
                η_ind = _build_eta_ind(dm, i, info, b, const_cache, θ)
                lli = _loglikelihood_individual(dm, i, θ, η_ind, cache)
                if !isfinite(lli)
                    ll = -Inf
                    break
                end
                ll += lli
            end
            Turing.@addlogprob! ll
        end
    end
    Core.eval(@__MODULE__, ex)
    lock(_MCEM_MODEL_CACHE_LOCK)
    _MCEM_MODEL_CACHE[key] = fname
    unlock(_MCEM_MODEL_CACHE_LOCK)
    return fname
end

function _extract_b_samples(chain, info::_LaplaceBatchInfo, re_names::Vector{Symbol})
    nb = info.n_b
    nb == 0 && return (zeros(Float64, 0, 0), Float64[], Float64[])
    names = MCMCChains.names(chain, :parameters)
    name_to_idx = Dict{String, Int}()
    for (i, n) in enumerate(names)
        name_to_idx[String(n)] = i
    end
    arr = Array(chain)
    if ndims(arr) == 1
        iters = size(arr, 1)
        nparams = 1
        nchains = 1
        arr = reshape(arr, iters, nparams, 1)
    elseif ndims(arr) == 2
        iters, nparams = size(arr)
        nchains = 1
        arr = reshape(arr, iters, nparams, 1)
    end
    iters, _, nchains = size(arr)
    ns = iters * nchains
    T = eltype(arr)
    samples = Matrix{T}(undef, nb, ns)

    col = 1
    for c in 1:nchains
        for t in 1:iters
            b = view(samples, :, col)
            for (ri, re) in enumerate(re_names)
                meta = info.re_info[ri]
                levels = getproperty(getproperty(meta, :map), :levels)
                ranges = meta.ranges
                for li in eachindex(levels)
                    r = ranges[li]
                    if meta.is_scalar
                        key = string(re, "_vals[", li, "]")
                        idx = get(name_to_idx, key, 0)
                        if idx == 0 && li == 1
                            key = string(re, "_v1")
                            idx = get(name_to_idx, key, 0)
                        end
                        idx == 0 && error("Missing MCMC parameter $(key).")
                        b[first(r)] = arr[t, idx, c]
                    else
                        for k in 1:length(r)
                            key = string(re, "_vals[", li, "][", k, "]")
                            idx = get(name_to_idx, key, 0)
                            if idx == 0 && li == 1
                                key = string(re, "_v1[", k, "]")
                                idx = get(name_to_idx, key, 0)
                            end
                            idx == 0 && error("Missing MCMC parameter $(key).")
                            b[r[k]] = arr[t, idx, c]
                        end
                    end
                end
            end
            col += 1
        end
    end
    sym_names = Symbol.(String.(names))
    vals = Tuple(arr[end, i, end] for i in 1:length(sym_names))
    last_params = NamedTuple{Tuple(sym_names)}(vals)
    last_b = vec(samples[:, end])
    return (samples, last_params, last_b)
end

function _re_dists_for_info(dm::DataModel,
                            info::_LaplaceBatchInfo,
                            θ::ComponentArray,
                            cache::_LLCache)
    model_funs = cache.model_funs
    helpers = cache.helpers
    θ_re = _symmetrize_psd_params(θ, dm.model.fixed.fixed)
    dists_builder = get_create_random_effect_distribution(dm.model.random.random)
    cache_re = dm.re_group_info.laplace_cache
    re_names = cache_re.re_names
    pairs = Vector{Pair{Symbol, Vector{Distributions.Distribution}}}(undef, length(re_names))
    for (ri, re) in enumerate(re_names)
        info_re = info.re_info[ri]
        levels = getproperty(getproperty(info_re, :map), :levels)
        dists = Vector{Distributions.Distribution}(undef, length(levels))
        for (li, level) in enumerate(levels)
            rep_idx = info_re.reps[li]
            const_cov = dm.individuals[rep_idx].const_cov
            d = dists_builder(θ_re, const_cov, model_funs, helpers)
            dists[li] = getproperty(d, re)
        end
        pairs[ri] = re => dists
    end
    return NamedTuple(pairs)
end

function _filter_b_samples_by_prior(dm::DataModel,
                                    info::_LaplaceBatchInfo,
                                    θ::ComponentArray,
                                    const_cache::LaplaceConstantsCache,
                                    cache::_LLCache,
                                    samples::AbstractMatrix)
    size(samples, 2) == 0 && return samples
    dists_by_re = _re_dists_for_info(dm, info, θ, cache)
    keep = trues(size(samples, 2))
    re_names = dm.re_group_info.laplace_cache.re_names
    for s in 1:size(samples, 2)
        b = view(samples, :, s)
        ok = true
        for (ri, re) in enumerate(re_names)
            info_re = info.re_info[ri]
            isempty(info_re.map.levels) && continue
            dists = getproperty(dists_by_re, re)
            for (li, level_id) in enumerate(info_re.map.levels)
                v = _re_value_from_b(info_re, level_id, b)
                v === nothing && continue
                lp = logpdf(dists[li], v)
                if !isfinite(lp)
                    ok = false
                    break
                end
            end
            ok || break
        end
        keep[s] = ok
    end
    all(keep) && return samples
    idx = findall(keep)
    return samples[:, idx]
end

function _mcem_sample_batch(dm, info, θ, const_cache, cache, sampler, turing_kwargs, rng,
                            re_names, warm_start, last_params;
                            anneal_sds::NamedTuple=NamedTuple(),
                            outer_iter::Int=1)
    if sampler isa SaemixMH
        error("SAEM internal error: SaemixMH dispatched to generic _mcem_sample_batch with cache type $(typeof(cache)). This usually means a threaded SAEM path passed a cache vector instead of a per-thread cache.")
    end
    nb = info.n_b
    if nb == 0
        return (zeros(eltype(θ), 0, 0), Float64[], eltype(θ)[])
    end
    fname = _build_mcem_batch_model(re_names)
    model_fn = Base.invokelatest(getfield, @__MODULE__, fname)
    model = Base.invokelatest(model_fn, dm, info, θ, const_cache, cache, anneal_sds)
    n_samples = get(turing_kwargs, :n_samples, 100)
    n_adapt = get(turing_kwargs, :n_adapt, 50)
    tkwargs = Base.structdiff(turing_kwargs, (n_samples=0, n_adapt=0))
    haskey(tkwargs, :progress) || (tkwargs = merge(tkwargs, (progress=false,)))
    haskey(tkwargs, :verbose) || (tkwargs = merge(tkwargs, (verbose=false,)))
    chain = if warm_start && last_params isa NamedTuple && !isempty(last_params)
        init = DynamicPPL.InitFromParams(last_params)
        Base.invokelatest(Turing.sample, rng, model, sampler, n_samples;
                          adapt=n_adapt, initial_params=init, tkwargs...)
    else
        Base.invokelatest(Turing.sample, rng, model, sampler, n_samples;
                          adapt=n_adapt, tkwargs...)
    end
    samples, lastp, lastb = _extract_b_samples(chain, info, re_names)
    samples = _filter_b_samples_by_prior(dm, info, θ, const_cache, cache, samples)
    if size(samples, 2) == 0
        return (zeros(eltype(θ), nb, 0), lastp, zeros(eltype(θ), nb))
    end
    lastb = vec(samples[:, end])
    return (samples, lastp, lastb)
end

# ---------------------------------------------------------------------------
# Importance Sampling E-step infrastructure
# ---------------------------------------------------------------------------

# Initialize one _REAdaptBlock per RE group for a given batch.
# Reuses _amh_init_cov (adaptive_mh.jl) for prior-based initial covariance.
# lp_offset and level_inds are set to dummy values — they are only used by
# _amh_step_block! (MH path), not by _amh_haario_update! (reused here).
function _is_init_proposal_blocks(dm::DataModel,
                                   info::_LaplaceBatchInfo,
                                   θ::ComponentArray,
                                   cache::_LLCache,
                                   re_names::Vector{Symbol},
                                   re_types::NamedTuple;
                                   init_scale::Float64=1.0,
                                   eps_reg::Float64=1e-6)
    dists_builder = get_create_random_effect_distribution(dm.model.random.random)
    model_funs    = cache.model_funs
    helpers       = cache.helpers
    blocks = Vector{_REAdaptBlock}(undef, length(re_names))
    for (ri, re) in enumerate(re_names)
        re_info  = info.re_info[ri]
        dim      = re_info.dim
        n_levels = length(re_info.map.levels)
        re_type  = getproperty(re_types, re)
        # Get representative distribution from first active level
        dist = if n_levels > 0
            rep_idx  = re_info.reps[1]
            const_cov = dm.individuals[rep_idx].const_cov
            dists = dists_builder(θ, const_cov, model_funs, helpers)
            getproperty(dists, re)
        else
            nothing
        end
        C = dist !== nothing ? _amh_init_cov(dist, dim, init_scale, eps_reg) :
                               (init_scale + eps_reg) .* Matrix{Float64}(I(max(dim, 1)))
        blocks[ri] = _REAdaptBlock(
            re, re_type, dim, n_levels, ri,
            0,           # lp_offset — unused for IS
            Vector{Int}[],  # level_inds — unused for IS
            C,
            _amh_chol_L(C, dim),
            zeros(Float64, dim),
            zeros(Float64, dim, dim),
            0,
        )
    end
    return blocks
end

# Draw n_samples from the prior proposal for a single batch.
# Returns (samples::Matrix{Float64}(n_b, n_samples), log_qs::Vector{Float64}).
function _is_prior_sample_batch(dm::DataModel,
                                 info::_LaplaceBatchInfo,
                                 θ::ComponentArray,
                                 const_cache::LaplaceConstantsCache,
                                 cache::_LLCache,
                                 rng::AbstractRNG,
                                 n_samples::Int,
                                 re_names::Vector{Symbol})
    nb = info.n_b
    if nb == 0
        return (zeros(Float64, 0, n_samples), zeros(Float64, n_samples))
    end
    dists_builder = get_create_random_effect_distribution(dm.model.random.random)
    model_funs    = cache.model_funs
    helpers       = cache.helpers
    samples = Matrix{Float64}(undef, nb, n_samples)
    log_qs  = zeros(Float64, n_samples)
    for m in 1:n_samples
        b = view(samples, :, m)
        lq = 0.0
        for (ri, re) in enumerate(re_names)
            re_info = info.re_info[ri]
            levels  = re_info.map.levels
            isempty(levels) && continue
            for (li, _) in enumerate(levels)
                r         = re_info.ranges[li]
                rep_idx   = re_info.reps[li]
                const_cov = dm.individuals[rep_idx].const_cov
                dists     = dists_builder(θ, const_cov, model_funs, helpers)
                dist      = getproperty(dists, re)
                draw      = rand(rng, dist)
                if re_info.is_scalar
                    b[first(r)] = draw isa AbstractVector ? draw[1] : Float64(draw)
                    lq += logpdf(dist, b[first(r)])
                else
                    b[r] .= draw
                    lq += logpdf(dist, Vector{Float64}(view(b, r)))
                end
            end
        end
        log_qs[m] = lq
    end
    return (samples, log_qs)
end

# Build a b vector at the prior mean (or median fallback) for each RE level in a batch.
# Used to initialize b_current and Turing's starting state before the main SAEM/MCEM loop.
function _re_prior_mean_b(dm::DataModel,
                          info::_LaplaceBatchInfo,
                          θ::ComponentArray,
                          const_cache::LaplaceConstantsCache,
                          cache::_LLCache,
                          re_names::Vector{Symbol})
    nb = info.n_b
    b  = zeros(Float64, nb)
    nb == 0 && return b
    dists_by_re = _re_dists_for_info(dm, info, θ, cache)
    for (ri, re) in enumerate(re_names)
        meta   = info.re_info[ri]
        levels = meta.map.levels
        isempty(levels) && continue
        dists  = getproperty(dists_by_re, re)
        for (li, _) in enumerate(levels)
            r    = meta.ranges[li]
            dist = dists[li]
            # Try mean, then median — both are generic Distributions.jl API
            val = try
                m = Distributions.mean(dist)
                all(isfinite, m isa Number ? (m,) : m) ? m : nothing
            catch
                nothing
            end
            if val === nothing
                val = try
                    m = Distributions.median(dist)
                    all(isfinite, m isa Number ? (m,) : m) ? m : nothing
                catch
                    nothing
                end
            end
            val === nothing && continue   # leave b at 0 for this level
            if meta.is_scalar
                b[first(r)] = val isa Number ? Float64(val) : Float64(val[1])
            else
                v = val isa AbstractVector ? val : [val]
                for k in eachindex(r)
                    k > length(v) && break
                    b[r[k]] = Float64(v[k])
                end
            end
        end
    end
    return b
end

# Convert a flat b vector to the NamedTuple format expected by Turing as `initial_params`.
# Key names match those produced by `_extract_b_samples`: level 1 → `η_v1` (or `η_v1[k]`
# for vectors), level li>1 → `η_vals[li]` (or `η_vals[li][k]` for vectors).
function _b_to_last_params(b::AbstractVector,
                            info::_LaplaceBatchInfo,
                            re_names::Vector{Symbol})
    pairs = Pair{Symbol, Float64}[]
    for (ri, re) in enumerate(re_names)
        meta   = info.re_info[ri]
        levels = meta.map.levels
        for (li, _) in enumerate(levels)
            r = meta.ranges[li]
            if meta.is_scalar
                key = li == 1 ? Symbol(re, :_v1) : Symbol(string(re, "_vals[", li, "]"))
                push!(pairs, key => Float64(b[first(r)]))
            else
                for k in eachindex(r)
                    key = li == 1 ?
                        Symbol(string(re, "_v1[", k, "]")) :
                        Symbol(string(re, "_vals[", li, "][", k, "]"))
                    push!(pairs, key => Float64(b[r[k]]))
                end
            end
        end
    end
    isempty(pairs) && return NamedTuple()
    ks = Tuple(first.(pairs))
    vs = Tuple(last.(pairs))
    return NamedTuple{ks}(vs)
end

# Draw n_samples from the adaptive Gaussian proposal in bijected z-space.
# If blocks have not yet accumulated enough samples, falls back to prior-like
# sampling from the initial covariance (blocks.μ_run starts at zero = prior mean
# in z-space for symmetric distributions like Normal).
function _is_gaussian_sample_batch(dm::DataModel,
                                    info::_LaplaceBatchInfo,
                                    rng::AbstractRNG,
                                    n_samples::Int,
                                    re_names::Vector{Symbol},
                                    re_types::NamedTuple,
                                    blocks::Vector{_REAdaptBlock})
    nb = info.n_b
    if nb == 0
        return (zeros(Float64, 0, n_samples), zeros(Float64, n_samples))
    end
    samples = Matrix{Float64}(undef, nb, n_samples)
    log_qs  = zeros(Float64, n_samples)
    for m in 1:n_samples
        b  = view(samples, :, m)
        lq = 0.0
        for (ri, re) in enumerate(re_names)
            re_info = info.re_info[ri]
            levels  = re_info.map.levels
            isempty(levels) && continue
            block   = blocks[ri]
            re_type = getproperty(re_types, re)
            L       = block.C_chol_L
            μ       = block.μ_run
            dim     = block.dim
            for li in eachindex(levels)
                r = re_info.ranges[li]
                # Draw z ~ N(μ, C) in bijected space
                z = _amh_propose(rng, μ, L)
                # Back-transform to natural space
                η = _amh_bij_inverse(re_type, dim == 1 ? z[1] : z)
                if re_info.is_scalar
                    η_val = η isa AbstractVector ? η[1] : Float64(η)
                    b[first(r)] = η_val
                    z_val = dim == 1 ? z[1] : z[1]
                    lq += logpdf(Normal(μ[1], sqrt(max(block.C[1, 1], 1e-14))), z_val) +
                           _bij_log_jac_forward(re_type, z_val)
                else
                    b[r] .= η
                    # Multivariate Gaussian log-density: -0.5*(z-μ)'C⁻¹(z-μ) - 0.5*logdet(2πC)
                    zv   = Vector{Float64}(z)
                    Cfull = Symmetric(block.C .+ 1e-14 .* I(dim))
                    lq += logpdf(MvNormal(μ, Cfull), zv) +
                           _bij_log_jac_forward(re_type, zv)
                end
            end
        end
        log_qs[m] = lq
    end
    return (samples, log_qs)
end

# Compute log-normalized IS weights for a batch.
# log_ws[m] = log p(y, b_m | θ_k) - log_qs[m], then log-normalized (logsumexp = 0).
# Returns (log_ws::Vector{Float64}, ess::Float64).
function _is_compute_log_weights(dm::DataModel,
                                  info::_LaplaceBatchInfo,
                                  θ::ComponentArray,
                                  const_cache::LaplaceConstantsCache,
                                  cache::_LLCache,
                                  samples::AbstractMatrix,
                                  log_qs::AbstractVector{Float64})
    n = size(samples, 2)
    if n == 0
        return (Float64[], NaN)
    end
    log_ws_unnorm = Vector{Float64}(undef, n)
    for m in 1:n
        b      = view(samples, :, m)
        log_f  = _laplace_logf_batch(dm, info, θ, b, const_cache, cache)
        log_ws_unnorm[m] = isfinite(log_f) ? log_f - log_qs[m] : -Inf
    end
    # logsumexp normalization
    lmax = maximum(log_ws_unnorm)
    if !isfinite(lmax)
        @warn "MCEM IS: all samples have -Inf log-weight in a batch — using uniform weights"
        fill!(log_ws_unnorm, -log(Float64(n)))
        return (log_ws_unnorm, 1.0)
    end
    log_sum = lmax + log(sum(exp(lw - lmax) for lw in log_ws_unnorm))
    log_ws  = log_ws_unnorm .- log_sum
    # ESS = 1 / sum(w_m^2)  in log domain: -logsumexp(2 .* log_ws)
    lmax2   = 2.0 * maximum(log_ws)
    ess     = exp(-(lmax2 + log(sum(exp(2.0 * lw - lmax2) for lw in log_ws))))
    return (log_ws, ess)
end

# Update Gaussian proposal blocks from IS or MCMC samples.
# When log_ws is provided (IS case) the weighted posterior mean and covariance are
# estimated directly; otherwise uniform weights (MCMC posterior samples) are assumed.
# Pooling across all levels within the batch is intentional — they share the proposal.
function _is_update_blocks!(blocks::Vector{_REAdaptBlock},
                             samples::AbstractMatrix,
                             info::_LaplaceBatchInfo,
                             re_names::Vector{Symbol},
                             re_types::NamedTuple,
                             adapt_start::Int,
                             eps_reg::Float64;
                             log_ws::Union{Nothing, AbstractVector{Float64}}=nothing)
    n_samp = size(samples, 2)
    n_samp == 0 && return
    # Build sample weights (normalized, sum to 1 per batch)
    ws = if log_ws === nothing
        fill(1.0 / Float64(n_samp), n_samp)   # uniform for MCMC samples
    else
        exp.(log_ws)                            # already log-normalized → sum ≈ 1
    end
    for (ri, re) in enumerate(re_names)
        re_info = info.re_info[ri]
        levels  = re_info.map.levels
        isempty(levels) && continue
        block   = blocks[ri]
        re_type = getproperty(re_types, re)
        dim     = block.dim
        n_lev   = length(levels)
        n_pts   = n_lev * n_samp
        # Collect z-vectors (in bijected space) and pooled weights
        zs        = Matrix{Float64}(undef, dim, n_pts)
        ws_pooled = Vector{Float64}(undef, n_pts)
        k = 0
        for m in 1:n_samp
            b = view(samples, :, m)
            for li in eachindex(levels)
                k += 1
                r   = re_info.ranges[li]
                η   = re_info.is_scalar ? b[first(r)] : Vector{Float64}(view(b, r))
                z   = _amh_bij_forward(re_type, η)
                z_v = dim == 1 ? Float64(z isa AbstractVector ? z[1] : z) : nothing
                if dim == 1
                    zs[1, k] = z_v
                else
                    zs[:, k] .= Vector{Float64}(z)
                end
                ws_pooled[k] = ws[m]
            end
        end
        # Normalize pooled weights (levels share weight proportionally within sample)
        w_sum = sum(ws_pooled)
        w_sum < 1e-300 && continue
        ws_pooled ./= w_sum
        # Weighted mean in z-space
        μ = vec(zs * ws_pooled)
        # Weighted covariance
        dz = zs .- μ                         # dim × n_pts deviations
        C  = Symmetric((dz .* ws_pooled') * dz')  # dim × dim
        # Regularize and re-Cholesky
        C_reg = Symmetric(Matrix(C) + eps_reg * I(dim))
        try
            L = cholesky(C_reg).L
            block.μ_run    .= μ
            block.C        .= Matrix(C_reg)
            block.C_chol_L .= Matrix(L)
            block.n_samples += n_pts
        catch
            # Cholesky failed (degenerate weights) — keep current block
        end
    end
end

# Unified IS E-step dispatcher for a single batch.
# Returns (samples, log_ws, ess) — log_ws are log-normalized weights.
function _is_sample_batch(dm::DataModel,
                           info::_LaplaceBatchInfo,
                           θ::ComponentArray,
                           const_cache::LaplaceConstantsCache,
                           cache::_LLCache,
                           rng::AbstractRNG,
                           re_names::Vector{Symbol},
                           re_types::NamedTuple,
                           e_step::MCEM_IS,
                           blocks::Vector{_REAdaptBlock})
    n_samples = e_step.n_samples
    proposal  = e_step.proposal
    # Choose sampling strategy
    samples, log_qs = if proposal === :prior
        _is_prior_sample_batch(dm, info, θ, const_cache, cache, rng, n_samples, re_names)
    elseif proposal === :gaussian
        # Fall back to prior-based initial covariance until blocks have data
        if isempty(blocks) || blocks[1].n_samples < 2
            _is_prior_sample_batch(dm, info, θ, const_cache, cache, rng, n_samples,
                                    re_names)
        else
            _is_gaussian_sample_batch(dm, info, rng, n_samples, re_names, re_types, blocks)
        end
    elseif proposal isa Function
        re_dists = _re_dists_for_info(dm, info, θ, cache)
        proposal(θ, info, re_dists, rng, n_samples)
    else
        error("MCEM_IS: unknown proposal type $(repr(proposal)). Use :prior, :gaussian, or a Function.")
    end
    log_ws, ess = _is_compute_log_weights(dm, info, θ, const_cache, cache, samples, log_qs)
    return (samples, log_ws, ess)
end

# Helper: is the current iteration still in the MCMC warm-up phase?
_use_mcmc_this_iter(::Int, ::MCEM_MCMC) = true
_use_mcmc_this_iter(iter::Int, es::MCEM_IS) = iter <= es.warm_start_mcmc_iters

# Helper: get the MCMC e_step options for warm-up (or the method itself for MCEM_MCMC)
_mcmc_e_step(es::MCEM_MCMC) = es
_mcmc_e_step(es::MCEM_IS)   = es.mcmc_warmup

function _mcem_Q_array(dm::DataModel,
                       batch_infos::Vector{_LaplaceBatchInfo},
                       θ::ComponentArray,
                       const_cache::LaplaceConstantsCache,
                       ll_cache,
                       samples_by_batch::AbstractVector{<:AbstractMatrix},
                       weights_by_batch::Union{Nothing, AbstractVector{<:AbstractVector}}=nothing;
                       serialization::SciMLBase.EnsembleAlgorithm=EnsembleSerial(),
                       q_cache::Union{Nothing, _MCEMQCache}=nothing)
    total = zero(eltype(θ))
    if serialization isa SciMLBase.EnsembleThreads
        nthreads = Threads.maxthreadid()
        caches = _mcem_thread_caches(dm, ll_cache, nthreads)
        Tθ = eltype(θ)
        use_cache = q_cache !== nothing && eltype(q_cache.partial_obj) === Tθ
        partial_obj = use_cache ? q_cache.partial_obj : Vector{Tθ}()
        if length(partial_obj) != length(batch_infos)
            resize!(partial_obj, length(batch_infos))
        end
        fill!(partial_obj, zero(Tθ))
        bad = Threads.Atomic{Bool}(false)
        Threads.@threads for bi in eachindex(batch_infos)
            bad[] && continue
            tid = Threads.threadid()
            info = batch_infos[bi]
            samples = samples_by_batch[bi]
            if size(samples, 2) == 0
                continue
            end
            ws = weights_by_batch === nothing ? nothing : weights_by_batch[bi]
            acc = zero(eltype(θ))
            for s in 1:size(samples, 2)
                b = view(samples, :, s)
                logf = _laplace_logf_batch(dm, info, θ, b, const_cache, caches[tid])
                !isfinite(logf) && (bad[] = true; break)
                w = ws === nothing ? one(eltype(θ)) : eltype(θ)(ws[s])
                acc += w * logf
            end
            bad[] && continue
            # For MCMC (uniform weights) divide by count; for IS weights already sum to 1
            if ws === nothing
                acc /= size(samples, 2)
            end
            partial_obj[bi] = acc
        end
        bad[] && return Inf
        @inbounds for bi in eachindex(batch_infos)
            total += partial_obj[bi]
        end
    else
        ll_cache_local = ll_cache isa Vector ? ll_cache[1] : ll_cache
        for (bi, info) in enumerate(batch_infos)
            samples = samples_by_batch[bi]
            if size(samples, 2) == 0
                continue
            end
            ws = weights_by_batch === nothing ? nothing : weights_by_batch[bi]
            acc = zero(eltype(θ))
            for s in 1:size(samples, 2)
                b = view(samples, :, s)
                logf = _laplace_logf_batch(dm, info, θ, b, const_cache, ll_cache_local)
                !isfinite(logf) && return Inf
                w = ws === nothing ? one(eltype(θ)) : eltype(θ)(ws[s])
                acc += w * logf
            end
            if ws === nothing
                acc /= size(samples, 2)
            end
            total += acc
        end
    end
    return total
end

function _mcem_Q(dm::DataModel,
                 batch_infos::Vector{_LaplaceBatchInfo},
                 θ::ComponentArray,
                 const_cache::LaplaceConstantsCache,
                 ll_cache,
                 samples_by_batch::AbstractVector{<:AbstractMatrix},
                 weights_by_batch::Union{Nothing, AbstractVector{<:AbstractVector}}=nothing;
                 serialization::SciMLBase.EnsembleAlgorithm=EnsembleSerial(),
                 q_cache::Union{Nothing, _MCEMQCache}=nothing)
    return _mcem_Q_array(dm, batch_infos, θ, const_cache, ll_cache, samples_by_batch,
                         weights_by_batch; serialization=serialization, q_cache=q_cache)
end

function _mcem_Q(dm::DataModel,
                 batch_infos::Vector{_LaplaceBatchInfo},
                 θ::ComponentVector,
                 const_cache::LaplaceConstantsCache,
                 ll_cache,
                 samples_by_batch::AbstractVector{<:AbstractMatrix},
                 weights_by_batch::Union{Nothing, AbstractVector{<:AbstractVector}}=nothing;
                 serialization::SciMLBase.EnsembleAlgorithm=EnsembleSerial(),
                 q_cache::Union{Nothing, _MCEMQCache}=nothing)
    return _mcem_Q_array(dm, batch_infos, θ, const_cache, ll_cache, samples_by_batch,
                         weights_by_batch; serialization=serialization, q_cache=q_cache)
end


function _fit_model(dm::DataModel, method::MCEM, args...;
                    constants::NamedTuple=NamedTuple(),
                    constants_re::NamedTuple=NamedTuple(),
                    penalty::NamedTuple=NamedTuple(),
                    ode_args::Tuple=(),
                    ode_kwargs::NamedTuple=NamedTuple(),
                    serialization::SciMLBase.EnsembleAlgorithm=EnsembleSerial(),
                    rng::AbstractRNG=Xoshiro(0),
                    theta_0_untransformed::Union{Nothing, ComponentArray}=nothing,
                    store_eb_modes::Bool=true,
                    store_data_model::Bool=true)
    fit_kwargs = (constants=constants,
                  constants_re=constants_re,
                  penalty=penalty,
                  ode_args=ode_args,
                  ode_kwargs=ode_kwargs,
                  serialization=serialization,
                  rng=rng,
                  theta_0_untransformed=theta_0_untransformed,
                  store_eb_modes=store_eb_modes,
                  store_data_model=store_data_model)
    re_names = get_re_names(dm.model.random.random)
    isempty(re_names) && error("MCEM requires random effects. Use MLE/MAP for fixed-effects models.")
    fe = dm.model.fixed.fixed
    priors = get_priors(fe)
    if any(!(getfield(priors, k) isa Priorless) for k in keys(priors))
        @info "MCEM ignores fixed-effect priors. Use MAP/MCMC for prior-aware inference."
    end

    fixed_names = get_names(fe)
    isempty(fixed_names) && error("MCEM requires at least one fixed effect.")
    fixed_set = Set(fixed_names)
    for name in keys(constants)
        name in fixed_set || error("Unknown constant parameter $(name).")
    end
    all(name in keys(constants) for name in fixed_names) &&
        error("MCEM requires at least one free fixed effect. Remove constants or specify a fixed effect or random effect.")

    free_names = [n for n in fixed_names if !(n in keys(constants))]
    θ0_u = get_θ0_untransformed(fe)
    if theta_0_untransformed !== nothing
        for n in fixed_names
            hasproperty(theta_0_untransformed, n) || error("theta_0_untransformed is missing parameter $(n).")
        end
        θ0_u = theta_0_untransformed
    end
    transform = get_transform(fe)
    inv_transform = get_inverse_transform(fe)
    θ0_t = transform(θ0_u)
    θ_const_u = deepcopy(θ0_u)
    _apply_constants!(θ_const_u, constants)
    θ_const_t = transform(θ_const_u)

    constants_re = _normalize_constants_re(dm, constants_re)
    const_cache = _build_constants_cache(dm, constants_re)
    pairing, batch_infos, _ = _build_laplace_batch_infos(dm, constants_re)
    ll_cache = serialization isa SciMLBase.EnsembleThreads ?
               build_ll_cache(dm; ode_args=ode_args, ode_kwargs=ode_kwargs, nthreads=Threads.maxthreadid(), force_saveat=true) :
               build_ll_cache(dm; ode_args=ode_args, ode_kwargs=ode_kwargs, force_saveat=true)

    θt_free = ComponentArray(NamedTuple{Tuple(free_names)}(Tuple(getproperty(θ0_t, n) for n in free_names)))
    axs_free = getaxes(θt_free)
    axs_full = getaxes(θ_const_t)
    T0 = eltype(θt_free)

    diag = _MCEMDiagnostics{T0}(Vector{AbstractVector{T0}}(),
                               Vector{T0}(),
                               Vector{T0}(),
                               Vector{T0}(),
                               Vector{T0}(),
                               Vector{T0}(),
                               Int[],
                               Float64[])

    q_cache = _init_mcem_q_cache(T0, serialization)

    last_params = Vector{Union{Nothing, NamedTuple, AbstractVector, _AdaptiveMHState, _SaemixMHState}}(undef, length(batch_infos))
    fill!(last_params, nothing)
    batch_rngs = _mcem_thread_rngs(rng, length(batch_infos))

    # Pre-loop initialization: seed last_params from prior mean so that the first MCMC chain
    # (when warm_start=true) starts at a valid, finite-likelihood point — analogous to saemix's
    # initialization filter, but checking the full joint density via _laplace_logf_batch.
    let cache_init = ll_cache isa Vector ? ll_cache[1] : ll_cache
        for bi in 1:length(batch_infos)
            info = batch_infos[bi]
            info.n_b == 0 && continue
            b_init = _re_prior_mean_b(dm, info, θ0_u, const_cache, cache_init, re_names)
            logf   = _laplace_logf_batch(dm, info, θ0_u, b_init, const_cache, cache_init)
            if !isfinite(logf)
                b_init = nothing
                for _ in 1:10
                    samp, _ = _is_prior_sample_batch(dm, info, θ0_u, const_cache, cache_init,
                                                     rng, 1, re_names)
                    b_cand = samp[:, 1]
                    if isfinite(_laplace_logf_batch(dm, info, θ0_u, b_cand, const_cache, cache_init))
                        b_init = b_cand
                        break
                    end
                end
                if b_init === nothing
                    error("MCEM: Cannot find valid initial random effects for batch $bi after 10 tries. " *
                          "Initial fixed-effect parameters likely produce -Inf log-likelihood. " *
                          "Try different starting values.")
                end
            end
            last_params[bi] = _b_to_last_params(b_init, info, re_names)
        end
    end

    # IS proposal blocks — one Vector{_REAdaptBlock} per batch (only for MCEM_IS)
    re_types = get_re_types(dm.model.random.random)
    ll_cache_single = ll_cache isa Vector ? ll_cache[1] : ll_cache
    proposal_blocks = if method.e_step isa MCEM_IS
        [_is_init_proposal_blocks(dm, batch_infos[bi], θ0_u, ll_cache_single, re_names,
                                   re_types) for bi in eachindex(batch_infos)]
    else
        nothing
    end

    θ_prev = copy(θt_free)
    Q_prev = T0(Inf)
    θu_last = θ0_u   # tracks the full untransformed params at the end of each iteration
    param_streak = 0
    q_streak = 0
    converged = false
    progress_bar = ProgressMeter.Progress(method.em.maxiters; desc="MCEM",
                                           enabled=method.progress)

    for iter in 1:method.em.maxiters
        θt_curr = θ_prev isa ComponentArray ? θ_prev : ComponentArray(θ_prev, axs_free)
        θt_full_curr = ComponentArray(T0.(θ_const_t), axs_full)

        for name in free_names
            setproperty!(θt_full_curr, name, getproperty(θt_curr, name))
        end
        θu_curr = inv_transform(θt_full_curr)

        use_mcmc = _use_mcmc_this_iter(iter, method.e_step)
        samples_by_batch  = Vector{Matrix{T0}}(undef, length(batch_infos))
        weights_by_batch  = use_mcmc ? nothing :
                            Vector{Vector{Float64}}(undef, length(batch_infos))
        ess_by_batch      = use_mcmc ? nothing : Vector{Float64}(undef, length(batch_infos))

        if use_mcmc
            # --- MCMC E-step (MCEM_MCMC or warm-up phase of MCEM_IS) ---
            mcmc_es = _mcmc_e_step(method.e_step)
            S = _mcem_schedule(mcmc_es.sample_schedule, iter)
            if S <= 0
                S = get(mcmc_es.turing_kwargs, :n_samples, 100)
            end
            tkwargs = merge(mcmc_es.turing_kwargs, (n_samples=S,))

            if serialization isa SciMLBase.EnsembleThreads
                nthreads = Threads.maxthreadid()
                caches = _mcem_thread_caches(dm, ll_cache, nthreads)
                Threads.@threads for bi in eachindex(batch_infos)
                    info = batch_infos[bi]
                    samples, lastp, _ = _mcem_sample_batch(
                        dm, info, θu_curr, const_cache, caches[Threads.threadid()],
                        mcmc_es.sampler, tkwargs, batch_rngs[bi],
                        re_names, mcmc_es.warm_start, last_params[bi];
                        outer_iter=iter)
                    samples_by_batch[bi] = samples
                    last_params[bi] = lastp
                end
            else
                for (bi, info) in enumerate(batch_infos)
                    samples, lastp, _ = _mcem_sample_batch(
                        dm, info, θu_curr, const_cache, ll_cache,
                        mcmc_es.sampler, tkwargs, batch_rngs[bi],
                        re_names, mcmc_es.warm_start, last_params[bi];
                        outer_iter=iter)
                    samples_by_batch[bi] = samples
                    last_params[bi] = lastp
                end
            end

            # If this is the last MCMC warm-up iteration, seed IS proposal blocks
            if method.e_step isa MCEM_IS &&
               iter == method.e_step.warm_start_mcmc_iters &&
               method.e_step.adapt
                for bi in eachindex(batch_infos)
                    _is_update_blocks!(proposal_blocks[bi], samples_by_batch[bi],
                                       batch_infos[bi], re_names, re_types, 2, 1e-6)
                end
            end
            push!(diag.ess_hist, NaN)
            S_diag = S
        else
            # --- IS E-step ---
            is_es = method.e_step
            if serialization isa SciMLBase.EnsembleThreads
                nthreads = Threads.maxthreadid()
                caches = _mcem_thread_caches(dm, ll_cache, nthreads)
                Threads.@threads for bi in eachindex(batch_infos)
                    info = batch_infos[bi]
                    samps, log_ws, ess = _is_sample_batch(
                        dm, info, θu_curr, const_cache, caches[Threads.threadid()],
                        batch_rngs[bi], re_names, re_types, is_es, proposal_blocks[bi])
                    samples_by_batch[bi] = samps
                    weights_by_batch[bi] = exp.(log_ws)
                    ess_by_batch[bi] = ess
                    if is_es.adapt
                        _is_update_blocks!(proposal_blocks[bi], samps, info,
                                           re_names, re_types, 2, 1e-6; log_ws=log_ws)
                    end
                end
            else
                for (bi, info) in enumerate(batch_infos)
                    samps, log_ws, ess = _is_sample_batch(
                        dm, info, θu_curr, const_cache, ll_cache_single,
                        batch_rngs[bi], re_names, re_types, is_es, proposal_blocks[bi])
                    samples_by_batch[bi] = samps
                    weights_by_batch[bi] = exp.(log_ws)
                    ess_by_batch[bi] = ess
                    if is_es.adapt
                        _is_update_blocks!(proposal_blocks[bi], samps, info,
                                           re_names, re_types, 2, 1e-6; log_ws=log_ws)
                    end
                end
            end
            mean_ess = mean(filter(isfinite, ess_by_batch))
            if mean_ess < 0.1 * is_es.n_samples
                @warn "MCEM IS: low effective sample size" iter=iter ess=mean_ess n_samples=is_es.n_samples
            end
            push!(diag.ess_hist, mean_ess)
            S_diag = is_es.n_samples
        end

        # ── M-step ───────────────────────────────────────────────────────────
        θu_new = nothing
        Q_new  = T0(Inf)

        θt_free_iter = ComponentArray(NamedTuple{Tuple(free_names)}(
            Tuple(getproperty(θt_full_curr, n) for n in free_names)))
        axs_free_iter = getaxes(θt_free_iter)

        obj_cache = (θ=Ref{Any}(nothing), obj=Ref{Any}(nothing))
        function obj_only(θt, p)
            any(isnan.(θt)) && return Inf
            θt_free_loc = θt isa ComponentArray ? θt : ComponentArray(θt, axs_free_iter)
            θt_vec = θt_free_loc
            use_cache = !(eltype(θt_free_loc) <: ForwardDiff.Dual)
            if use_cache && obj_cache.θ[] !== nothing && length(obj_cache.θ[]) == length(θt_vec)
                _maxabsdiff(θt_vec, obj_cache.θ[]) == 0.0 && return obj_cache.obj[]
            end
            T = eltype(θt_free_loc)
            θt_full_loc = ComponentArray(T.(θ_const_t), axs_full)
            for name in free_names
                setproperty!(θt_full_loc, name, getproperty(θt_free_loc, name))
            end
            θu = inv_transform(θt_full_loc)
            Q = _mcem_Q(dm, batch_infos, θu, const_cache, ll_cache, samples_by_batch,
                        weights_by_batch; serialization=serialization, q_cache=q_cache)
            !isfinite(Q) && return Inf
            obj = -Q + _penalty_value(θu, penalty)
            !isfinite(obj) && return Inf
            if use_cache
                obj_cache.θ[] = copy(θt_vec)
                obj_cache.obj[] = obj
            end
            return obj
        end

        optf = OptimizationFunction(obj_only, method.adtype)
        lower_t, upper_t = get_bounds_transformed(fe)

        lower_t_free_iter = ComponentArray(NamedTuple{Tuple(free_names)}(Tuple(getproperty(lower_t, n) for n in free_names)))
        upper_t_free_iter = ComponentArray(NamedTuple{Tuple(free_names)}(Tuple(getproperty(upper_t, n) for n in free_names)))
        lower_t_free_vec = collect(lower_t_free_iter)
        upper_t_free_vec = collect(upper_t_free_iter)
        use_bounds = !(all(isinf, lower_t_free_vec) && all(isinf, upper_t_free_vec))
        user_bounds = method.lb !== nothing || method.ub !== nothing
        if user_bounds && !isempty(keys(constants))
            @info "Bounds for constant parameters are ignored." constants=collect(keys(constants))
        end
        if user_bounds
            lb_m = method.lb
            ub_m = method.ub
            if lb_m isa ComponentArray
                lb_m = ComponentArray(NamedTuple{Tuple(free_names)}(Tuple(getproperty(lb_m, n) for n in free_names)))
            end
            if ub_m isa ComponentArray
                ub_m = ComponentArray(NamedTuple{Tuple(free_names)}(Tuple(getproperty(ub_m, n) for n in free_names)))
            end
            lb_m = lb_m === nothing ? lower_t_free_vec : collect(lb_m)
            ub_m = ub_m === nothing ? upper_t_free_vec : collect(ub_m)
        else
            lb_m = lower_t_free_vec
            ub_m = upper_t_free_vec
        end
        use_bounds = use_bounds || user_bounds
        if parentmodule(typeof(method.optimizer)) === OptimizationBBO && !use_bounds
            error("BlackBoxOptim methods require finite bounds. Add lower/upper bounds in @fixedEffects (on transformed scale) or pass them via MCEM(lb=..., ub=...). A quick helper is default_bounds_from_start(dm; margin=...).")
        end
        θ0_init = collect(θt_free_iter)
        prob = use_bounds ? OptimizationProblem(optf, θ0_init; lb=lb_m, ub=ub_m) :
                            OptimizationProblem(optf, θ0_init)

        sol = Optimization.solve(prob, method.optimizer; method.optim_kwargs...)

        if any(!isfinite, sol.u)
            @warn "MCEM M-step iter $iter: optimizer returned non-finite parameters; skipping update."
            θu_new = θu_curr
            Q_new  = Q_prev
        else
            θ_hat_t_raw  = sol.u
            θ_hat_t_free = θ_hat_t_raw isa ComponentArray ? θ_hat_t_raw :
                            ComponentArray(θ_hat_t_raw, axs_free_iter)

            θt_full_new = ComponentArray(eltype(θ_hat_t_free).(θ_const_t), axs_full)
            for name in free_names
                setproperty!(θt_full_new, name, getproperty(θ_hat_t_free, name))
            end
            θu_new = inv_transform(θt_full_new)
            Q_new = _mcem_Q(dm, batch_infos, θu_new, const_cache, ll_cache, samples_by_batch,
                            weights_by_batch; serialization=serialization, q_cache=q_cache)
            Q_new = Q_new == Inf ? T0(Inf) : Q_new
        end

        # Update θt_free to reflect full current state (use full axes for tracking)
        θt_full_new_full = transform(θu_new)
        θt_free = ComponentArray(NamedTuple{Tuple(free_names)}(
            Tuple(getproperty(θt_full_new_full, n) for n in free_names)))
        θ_prev_new = copy(θt_free)

        dθ_abs = _maxabsdiff(θ_prev_new, θ_prev)
        dθ_rel = dθ_abs / max(T0(1.0), _maxabs(θ_prev))
        dQ_abs = abs(Q_new - Q_prev)
        dQ_rel = dQ_abs / max(T0(1.0), abs(Q_prev))

        push!(diag.θ_hist, θ_prev_new)
        push!(diag.Q_hist, Q_new)
        push!(diag.dθ_abs, dθ_abs)
        push!(diag.dθ_rel, dθ_rel)
        push!(diag.dQ_abs, dQ_abs)
        push!(diag.dQ_rel, dQ_rel)
        push!(diag.samples, S_diag)

        if method.verbose
            @info "MCEM iteration" iter=iter samples=S_diag Q=Q_new dθ_abs=dθ_abs dθ_rel=dθ_rel dQ_abs=dQ_abs dQ_rel=dQ_rel
        end
        ProgressMeter.next!(progress_bar;
                            showvalues=[(:iter, iter), (:samples, S_diag), (:Q, Q_new)])

        θ_prev = θ_prev_new
        Q_prev = Q_new
        θu_last = θu_new

        if dθ_abs <= method.em.atol_theta && dθ_rel <= method.em.rtol_theta
            param_streak += 1
        else
            param_streak = 0
        end
        if isfinite(dQ_abs) && isfinite(dQ_rel) &&
           dQ_abs <= method.em.atol_Q && dQ_rel <= method.em.rtol_Q
            q_streak += 1
        else
            q_streak = 0
        end
        if param_streak >= method.em.consecutive_params &&
           q_streak >= method.em.consecutive_params
            converged = true
            break
        end
    end
    ProgressMeter.finish!(progress_bar)

    # Reconstruct final parameter estimates from the last full θu
    θ_hat_u = θu_last
    θ_hat_t = transform(θ_hat_u)

    summary = FitSummary(Q_prev, converged,
                         FitParameters(θ_hat_t, θ_hat_u),
                         NamedTuple())
    diagnostics = FitDiagnostics((;), (optimizer=method.optimizer,),
                                 (em_iters=length(diag.Q_hist),
                                  dθ_abs=isempty(diag.dθ_abs) ? T0(NaN) : diag.dθ_abs[end],
                                  dQ_abs=isempty(diag.dQ_abs) ? T0(NaN) : diag.dQ_abs[end]),
                                 NamedTuple())
    eb_modes = store_eb_modes ? _compute_bstars(dm, θ_hat_u, constants_re, ll_cache, method.ebe, rng;
                                                rescue=method.ebe_rescue,
                                                progress=method.progress,
                                                progress_desc="MCEM Final EBE")[1] : nothing

    notes = (diagnostics=diag,)

    result = MCEMResult(nothing, Q_prev, length(diag.Q_hist), nothing, notes, eb_modes)
    return FitResult(method, result, summary, diagnostics, store_data_model ? dm : nothing, args, fit_kwargs)
end
