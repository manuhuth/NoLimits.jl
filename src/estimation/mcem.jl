export MCEM
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
        return build_ll_cache(dm;
                              ode_args=ll_cache.ode_args,
                              ode_kwargs=ll_cache.ode_kwargs,
                              force_saveat=ll_cache.saveat_cache !== nothing,
                              nthreads=nthreads)
    else
        return build_ll_cache(dm; nthreads=nthreads)
    end
end

@inline function _mcem_thread_rngs(rng::AbstractRNG, nthreads::Int)
    return _spawn_child_rngs(rng, nthreads)
end

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
    MCEM(; optimizer, optim_kwargs, adtype, sampler, turing_kwargs, sample_schedule,
           warm_start, verbose, progress, maxiters, rtol_theta, atol_theta, rtol_Q,
           atol_Q, consecutive_params, ebe_optimizer, ebe_optim_kwargs, ebe_adtype,
           ebe_grad_tol, ebe_multistart_n, ebe_multistart_k, ebe_multistart_max_rounds,
           ebe_multistart_sampling, ebe_rescue_on_high_grad, ebe_rescue_multistart_n,
           ebe_rescue_multistart_k, ebe_rescue_max_rounds, ebe_rescue_grad_tol,
           ebe_rescue_multistart_sampling, lb, ub) <: FittingMethod

Monte Carlo Expectation-Maximisation for random-effects models. At each EM iteration the
E-step draws random effects from the posterior using a Turing.jl sampler; the M-step
maximises the Monte Carlo Q-function over the fixed effects.

# Keyword Arguments
- `optimizer`: M-step Optimization.jl optimiser. Defaults to `LBFGS` with backtracking.
- `optim_kwargs::NamedTuple = NamedTuple()`: keyword arguments for the M-step `solve`.
- `adtype`: AD backend for the M-step. Defaults to `AutoForwardDiff()`.
- `sampler`: Turing-compatible sampler for the E-step. Defaults to `NUTS(0.75)`.
- `turing_kwargs::NamedTuple = NamedTuple()`: keyword arguments forwarded to `Turing.sample`.
- `sample_schedule::Int = 250`: number of MCMC samples per E-step iteration.
- `warm_start::Bool = true`: initialise sampler from the previous iteration's modes.
- `verbose::Bool = false`: print per-iteration diagnostics.
- `progress::Bool = true`: show a progress bar.
- `maxiters::Int = 100`: maximum number of EM iterations.
- `rtol_theta`, `atol_theta`: relative/absolute convergence tolerance on fixed effects.
- `rtol_Q`, `atol_Q`: relative/absolute convergence tolerance on the Q-function.
- `consecutive_params::Int = 3`: number of consecutive iterations satisfying the tolerance
  required to declare convergence.
- `ebe_optimizer`, `ebe_optim_kwargs`, `ebe_adtype`, `ebe_grad_tol`: EBE inner optimiser
  settings used to compute mode starting points.
- `ebe_multistart_n`, `ebe_multistart_k`, `ebe_multistart_max_rounds`,
  `ebe_multistart_sampling`: multistart settings for EBE mode computation.
- `ebe_rescue_on_high_grad`, `ebe_rescue_multistart_n`, `ebe_rescue_multistart_k`,
  `ebe_rescue_max_rounds`, `ebe_rescue_grad_tol`, `ebe_rescue_multistart_sampling`:
  rescue multistart settings when an EBE mode has a high gradient norm.
- `lb`, `ub`: bounds on the transformed fixed-effect scale, or `nothing`.
"""
struct MCEM{O, K, A, MO, EO, EB, ER, L, U} <: FittingMethod
    optimizer::O
    optim_kwargs::K
    adtype::A
    mcmc::MO
    em::EO
    ebe::EB
    ebe_rescue::ER
    lb::L
    ub::U
end

MCEM(; optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
     optim_kwargs=NamedTuple(),
     adtype=Optimization.AutoForwardDiff(),
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
     ebe_optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
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
     ub=nothing) = begin
    mcmc = MCEMOptions(sampler, turing_kwargs, sample_schedule, warm_start, verbose, progress)
    em = EMOptions(maxiters, rtol_theta, atol_theta, rtol_Q, atol_Q, consecutive_params)
    ebe = EBEOptions(ebe_optimizer, ebe_optim_kwargs, ebe_adtype, ebe_grad_tol, ebe_multistart_n, ebe_multistart_k, ebe_multistart_max_rounds, ebe_multistart_sampling)
    ebe_rescue = EBERescueOptions(ebe_rescue_on_high_grad, ebe_rescue_multistart_n, ebe_rescue_multistart_k, ebe_rescue_max_rounds, ebe_rescue_grad_tol, ebe_rescue_multistart_sampling)
    MCEM(optimizer, optim_kwargs, adtype, mcmc, em, ebe, ebe_rescue, lb, ub)
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
                v1 = ($(Symbol(re, :_v1)) ~ dist)
                local $vals_sym = Vector{typeof(v1)}(undef, nlvls)
                $vals_sym[1] = v1
                for j in 2:nlvls
                    const_cov = dm.individuals[$reps_sym[j]].const_cov
                    dists = dists_builder(θ_re, const_cov, model_funs, helpers)
                    dist = getproperty(dists, $re_q)
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
                v1 = ($(Symbol(re, :_v1)) ~ dist)
                local $vals_sym = Vector{typeof(v1)}(undef, nlvls)
                $vals_sym[1] = v1
                for j in 2:nlvls
                    const_cov = dm.individuals[$reps_sym[j]].const_cov
                    dists = dists_builder(θ_re, const_cov, model_funs, helpers)
                    dist = getproperty(dists, $re_q)
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
        @model function $(fname)(dm, info, θ, const_cache, cache)
            θ_re = _symmetrize_psd_params(θ, dm.model.fixed.fixed)
            dists_builder = get_create_random_effect_distribution(dm.model.random.random)
            model_funs = cache.model_funs
            helpers = cache.helpers

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
                if lli == -Inf
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
                            re_names, warm_start, last_params)
    nb = info.n_b
    if nb == 0
        return (zeros(eltype(θ), 0, 0), Float64[], eltype(θ)[])
    end
    fname = _build_mcem_batch_model(re_names)
    model_fn = Base.invokelatest(getfield, @__MODULE__, fname)
    model = Base.invokelatest(model_fn, dm, info, θ, const_cache, cache)
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

function _mcem_Q_array(dm::DataModel,
                       batch_infos::Vector{_LaplaceBatchInfo},
                       θ::ComponentArray,
                       const_cache::LaplaceConstantsCache,
                       ll_cache,
                       samples_by_batch::AbstractVector{<:AbstractMatrix};
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
            acc = zero(eltype(θ))
            for s in 1:size(samples, 2)
                b = view(samples, :, s)
                logf = _laplace_logf_batch(dm, info, θ, b, const_cache, caches[tid])
                !isfinite(logf) && (bad[] = true; break)
                acc += logf
            end
            bad[] && continue
            acc /= size(samples, 2)
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
            acc = zero(eltype(θ))
            for s in 1:size(samples, 2)
                b = view(samples, :, s)
                logf = _laplace_logf_batch(dm, info, θ, b, const_cache, ll_cache_local)
                !isfinite(logf) && return Inf
                acc += logf
            end
            acc /= size(samples, 2)
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
                 samples_by_batch::AbstractVector{<:AbstractMatrix};
                 serialization::SciMLBase.EnsembleAlgorithm=EnsembleSerial(),
                 q_cache::Union{Nothing, _MCEMQCache}=nothing)
    return _mcem_Q_array(dm, batch_infos, θ, const_cache, ll_cache, samples_by_batch;
                         serialization=serialization, q_cache=q_cache)
end

function _mcem_Q(dm::DataModel,
                 batch_infos::Vector{_LaplaceBatchInfo},
                 θ::ComponentVector,
                 const_cache::LaplaceConstantsCache,
                 ll_cache,
                 samples_by_batch::AbstractVector{<:AbstractMatrix};
                 serialization::SciMLBase.EnsembleAlgorithm=EnsembleSerial(),
                 q_cache::Union{Nothing, _MCEMQCache}=nothing)
    return _mcem_Q_array(dm, batch_infos, θ, const_cache, ll_cache, samples_by_batch;
                         serialization=serialization, q_cache=q_cache)
end


function _fit_model(dm::DataModel, method::MCEM, args...;
                    constants::NamedTuple=NamedTuple(),
                    constants_re::NamedTuple=NamedTuple(),
                    penalty::NamedTuple=NamedTuple(),
                    ode_args::Tuple=(),
                    ode_kwargs::NamedTuple=NamedTuple(),
                    serialization::SciMLBase.EnsembleAlgorithm=EnsembleSerial(),
                    rng::AbstractRNG=Random.default_rng(),
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
                               Int[])

    q_cache = _init_mcem_q_cache(T0, serialization)

    last_params = Vector{Union{Nothing, NamedTuple, AbstractVector}}(undef, length(batch_infos))
    fill!(last_params, nothing)
    batch_rngs = _mcem_thread_rngs(rng, length(batch_infos))
    θ_prev = copy(θt_free)
    Q_prev = T0(Inf)
    param_streak = 0
    q_streak = 0
    converged = false
    progress = ProgressMeter.Progress(method.em.maxiters; desc="MCEM", enabled=method.mcmc.progress)

    for iter in 1:method.em.maxiters
        θt_curr = θ_prev isa ComponentArray ? θ_prev : ComponentArray(θ_prev, axs_free)
        θt_full_curr = ComponentArray(T0.(θ_const_t), axs_full)

        for name in free_names
            setproperty!(θt_full_curr, name, getproperty(θt_curr, name))
        end
        θu_curr = inv_transform(θt_full_curr)

        S = _mcem_schedule(method.mcmc.sample_schedule, iter)
        if S <= 0
            S = get(method.mcmc.turing_kwargs, :n_samples, 100)
        end
        tkwargs = merge(method.mcmc.turing_kwargs, (n_samples=S,))

        samples_by_batch = Vector{Matrix{T0}}(undef, length(batch_infos))
        if serialization isa SciMLBase.EnsembleThreads
            nthreads = Threads.maxthreadid()
            caches = _mcem_thread_caches(dm, ll_cache, nthreads)
            Threads.@threads for bi in eachindex(batch_infos)
                info = batch_infos[bi]
                samples, lastp, _ = _mcem_sample_batch(dm, info, θu_curr, const_cache, caches[Threads.threadid()],
                                                       method.mcmc.sampler, tkwargs, batch_rngs[bi],
                                                       re_names, method.mcmc.warm_start, last_params[bi])
                samples_by_batch[bi] = samples
                last_params[bi] = lastp
            end
        else
            for (bi, info) in enumerate(batch_infos)
                samples, lastp, _ = _mcem_sample_batch(dm, info, θu_curr, const_cache, ll_cache,
                                                       method.mcmc.sampler, tkwargs, batch_rngs[bi],
                                                       re_names, method.mcmc.warm_start, last_params[bi])
                samples_by_batch[bi] = samples
                last_params[bi] = lastp
            end
        end

        obj_cache = (θ=Ref{Any}(nothing), obj=Ref{Any}(nothing))
        function obj_only(θt, p)
            if any(isnan.(θt))
                return Inf
            end

            θt_free = θt isa ComponentArray ? θt : ComponentArray(θt, axs_free)
            θt_vec = θt_free
            use_cache = !(eltype(θt_free) <: ForwardDiff.Dual)
            if use_cache && obj_cache.θ[] !== nothing && length(obj_cache.θ[]) == length(θt_vec)
                maxdiff = _maxabsdiff(θt_vec, obj_cache.θ[])
                if maxdiff == 0.0
                    return obj_cache.obj[]
                end
            end
            T = eltype(θt_free)
            θt_full = ComponentArray(T.(θ_const_t), axs_full)

            for name in free_names
                setproperty!(θt_full, name, getproperty(θt_free, name))
            end

            θu = inv_transform(θt_full)

            Q = _mcem_Q(dm, batch_infos, θu, const_cache, ll_cache, samples_by_batch;
                        serialization=serialization, q_cache=q_cache)

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

        lower_t_free = ComponentArray(NamedTuple{Tuple(free_names)}(Tuple(getproperty(lower_t, n) for n in free_names)))
        upper_t_free = ComponentArray(NamedTuple{Tuple(free_names)}(Tuple(getproperty(upper_t, n) for n in free_names)))
        lower_t_free_vec = collect(lower_t_free)
        upper_t_free_vec = collect(upper_t_free)
        use_bounds = !(all(isinf, lower_t_free_vec) && all(isinf, upper_t_free_vec))
        user_bounds = method.lb !== nothing || method.ub !== nothing
        if user_bounds && !isempty(keys(constants))
            @info "Bounds for constant parameters are ignored." constants=collect(keys(constants))
        end
        if user_bounds
            lb = method.lb
            ub = method.ub
            if lb isa ComponentArray
                lb = ComponentArray(NamedTuple{Tuple(free_names)}(Tuple(getproperty(lb, n) for n in free_names)))
            end
            if ub isa ComponentArray
                ub = ComponentArray(NamedTuple{Tuple(free_names)}(Tuple(getproperty(ub, n) for n in free_names)))
            end
            lb = lb === nothing ? lower_t_free_vec : collect(lb)
            ub = ub === nothing ? upper_t_free_vec : collect(ub)
        else
            lb = lower_t_free_vec
            ub = upper_t_free_vec
        end
        use_bounds = use_bounds || user_bounds
        if parentmodule(typeof(method.optimizer)) === OptimizationBBO && !use_bounds
            error("BlackBoxOptim methods require finite bounds. Add lower/upper bounds in @fixedEffects (on transformed scale) or pass them via MCEM(lb=..., ub=...). A quick helper is default_bounds_from_start(dm; margin=...).")
        end
        θ0_init = collect(θt_free)
        prob = use_bounds ? OptimizationProblem(optf, θ0_init; lb=lb, ub=ub) :
                            OptimizationProblem(optf, θ0_init)


        sol = Optimization.solve(prob, method.optimizer; method.optim_kwargs...)

        θ_hat_t_raw = sol.u
        θ_hat_t_free = θ_hat_t_raw isa ComponentArray ? θ_hat_t_raw : ComponentArray(θ_hat_t_raw, axs_free)
        θt_free = θ_hat_t_free
        θ_prev_new = copy(θt_free)

        θt_full = ComponentArray(eltype(θt_free).(θ_const_t), axs_full)
        for name in free_names
            setproperty!(θt_full, name, getproperty(θt_free, name))
        end
        θu_new = inv_transform(θt_full)
        Q_new = _mcem_Q(dm, batch_infos, θu_new, const_cache, ll_cache, samples_by_batch;
                        serialization=serialization, q_cache=q_cache)
        Q_new = Q_new == Inf ? T0(Inf) : Q_new

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
        push!(diag.samples, S)

        if method.mcmc.verbose
            @info "MCEM iteration" iter=iter samples=S Q=Q_new dθ_abs=dθ_abs dθ_rel=dθ_rel dQ_abs=dQ_abs dQ_rel=dQ_rel
        end
        ProgressMeter.next!(progress; showvalues=[(:iter, iter), (:samples, S), (:Q, Q_new)])

        θ_prev = θ_prev_new
        Q_prev = Q_new

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
    ProgressMeter.finish!(progress)

    θ_hat_t_free = θ_prev isa ComponentArray ? θ_prev : ComponentArray(θ_prev, axs_free)
    θ_hat_t = ComponentArray(eltype(θ_hat_t_free).(θ_const_t), axs_full)
    for name in free_names
        setproperty!(θ_hat_t, name, getproperty(θ_hat_t_free, name))
    end
    θ_hat_u = inv_transform(θ_hat_t)

    summary = FitSummary(Q_prev, converged,
                         FitParameters(θ_hat_t, θ_hat_u),
                         NamedTuple())
    diagnostics = FitDiagnostics((;), (optimizer=method.optimizer,),
                                 (em_iters=length(diag.Q_hist), dθ_abs=diag.dθ_abs[end], dQ_abs=diag.dQ_abs[end]),
                                 NamedTuple())
    eb_modes = store_eb_modes ? _compute_bstars(dm, θ_hat_u, constants_re, ll_cache, method.ebe, rng;
                                                rescue=method.ebe_rescue,
                                                progress=method.mcmc.progress,
                                                progress_desc="MCEM Final EBE")[1] : nothing
    result = MCEMResult(nothing, Q_prev, length(diag.Q_hist), nothing, (diagnostics=diag,), eb_modes)
    return FitResult(method, result, summary, diagnostics, store_data_model ? dm : nothing, args, fit_kwargs)
end
