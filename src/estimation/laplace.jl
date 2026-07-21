export Laplace
export FrequentistREResult
export NewtonInner

using ForwardDiff
using LinearAlgebra
using Optimization
using OptimizationOptimJL
using OptimizationBBO
using OptimizationNLopt
using LineSearches
using SciMLBase
using ComponentArrays
using Random
using Distributions
using StaticArrays
using ProgressMeter

struct LaplaceADCache{G, H, O, B}
    grad_cfg::G
    hess_cfg::H
    optf::O
    buffers::B
end

struct _LaplaceLogfBatch{DM, INFO, TH, CC, CA, CX}
    dm::DM
    info::INFO
    θ::TH
    const_cache::CC
    cache::CA
    tctx::CX   # _LaplaceThetaCtx built from θ, or nothing
end

function _LaplaceLogfBatch(dm, info, θ, const_cache, cache)
    _LaplaceLogfBatch(dm, info, θ, const_cache, cache, nothing)
end

@inline function (f::_LaplaceLogfBatch)(b)
    return _laplace_logf_batch(f.dm, f.info, f.θ, b, f.const_cache, f.cache; tctx = f.tctx)
end

struct _LaplaceLogfBatchParam{DM, INFO, CC, CA}
    dm::DM
    info::INFO
    const_cache::CC
    cache::CA
end

# `p` is either θ itself or a `(θ, tctx)` tuple (the θ-context is θ-only work
# hoisted out of the inner solve's many objective evaluations; see
# `_LaplaceThetaCtx`). The branch is resolved at compile time per p-type.
@inline function (f::_LaplaceLogfBatchParam)(b, p)
    if p isa Tuple
        return -_laplace_logf_batch(
            f.dm, f.info, p[1], b, f.const_cache, f.cache; tctx = p[2])
    end
    return -_laplace_logf_batch(f.dm, f.info, p, b, f.const_cache, f.cache)
end

struct _LaplaceLogfTheta{DM, INFO, B, CC, CA}
    dm::DM
    info::INFO
    b::B
    const_cache::CC
    cache::CA
end

@inline function (f::_LaplaceLogfTheta)(θ)
    return _laplace_logf_batch(f.dm, f.info, θ, f.b, f.const_cache, f.cache)
end

struct _LaplaceLogdetTheta{DM, INFO, B, CC, CA, H, A}
    dm::DM
    info::INFO
    b::B
    const_cache::CC
    cache::CA
    hess::H
    ad_cache::A
    bi::Int
end

@inline function (f::_LaplaceLogdetTheta)(θ)
    return _laplace_logdet_negH(
        f.dm, f.info, θ, f.b, f.const_cache, f.cache, f.ad_cache, f.bi;
        jitter = f.hess.jitter,
        max_tries = f.hess.max_tries,
        growth = f.hess.growth,
        adaptive = f.hess.adaptive,
        scale_factor = f.hess.scale_factor,
        ctx = "grad_logdet_θ")[1]
end

struct _LaplaceLogdetB{DM, INFO, TH, CC, CA, H, A}
    dm::DM
    info::INFO
    θ::TH
    const_cache::CC
    cache::CA
    hess::H
    ad_cache::A
    bi::Int
end

@inline function (f::_LaplaceLogdetB)(b)
    return _laplace_logdet_negH(
        f.dm, f.info, f.θ, b, f.const_cache, f.cache, f.ad_cache, f.bi;
        jitter = f.hess.jitter,
        max_tries = f.hess.max_tries,
        growth = f.hess.growth,
        adaptive = f.hess.adaptive,
        scale_factor = f.hess.scale_factor,
        ctx = "grad_logdet_b")[1]
end

# Exception classes that signal a numerically-degenerate point (rather than a
# programming error): callers treat these as objective = Inf / cache-fallback
# during optimizer exploration instead of rethrowing.
@inline function _is_numeric_error(err)
    return err isa LinearAlgebra.PosDefException ||
           err isa LinearAlgebra.SingularException ||
           err isa DomainError || err isa ArgumentError
end

# ── SAEM anneal-to-fixed helpers ─────────────────────────────────────────────

function _saem_anneal_names(res::FitResult)
    get_result(res) isa SAEMResult || return ()
    if get_method(res) isa _SavedFittingMethod
        notes = res.result.notes
        if notes isa NamedTuple && hasproperty(notes, :anneal_to_fixed)
            return notes.anneal_to_fixed
        end
        return ()
    end
    return get_method(res).saem.anneal_to_fixed
end

# For each annealed RE, compute the prior-distribution mean per group level at
# the final θu, and pin those levels via constants_re so the EBE optimizer
# excludes them entirely.  User-supplied constants_re entries take precedence.
function _saem_anneal_constants_re(dm::DataModel,
        θu::ComponentArray,
        anneal_names,
        constants_re::NamedTuple)
    isempty(anneal_names) && return constants_re
    dists_builder = create_random_effect_distribution(get_random(get_model(dm)))
    helpers = get_helper_funs(get_model(dm))
    model_funs = get_model_funs(get_model(dm))

    level_means = Dict{Symbol, Dict{Any, Any}}()
    for name in anneal_names
        level_means[Symbol(name)] = Dict{Any, Any}()
    end

    for ind in get_individuals(dm)
        for name in anneal_names
            sym = Symbol(name)
            # re_groups stores a Vector of unique levels for that individual
            gvec = getfield(get_re_groups(ind), sym)
            for gv in gvec
                haskey(level_means[sym], gv) && continue
                dists = dists_builder(θu, get_const_cov(ind), model_funs, helpers)
                dist = getfield(dists, sym)
                level_means[sym][gv] = Distributions.mean(dist)
            end
        end
    end

    pairs = Pair{Symbol, Any}[k => v for (k, v) in Base.pairs(constants_re)]
    for name in anneal_names
        sym = Symbol(name)
        haskey(constants_re, sym) && continue   # user-supplied wins
        push!(pairs, sym => level_means[sym])
    end
    return NamedTuple(pairs)
end

struct _LaplaceBStarCache{T, B}
    b_star::Vector{Vector{T}}
    has_bstar::B
end

struct _LaplaceGradCache{T, L, G, V}
    last_b::L
    last_logf::Vector{T}
    last_gradb::G
    last_valid::V
end

# Typed key→config stores for the per-batch ForwardDiff config caches. Iterating
# the old `Vector{Any}` of `(key, cfg)` pairs boxed every key on retrieval
# (measured ~270 B / 180 ns per warm hit, with several hits per
# `_laplace_grad_batch` call). With concretely-typed keys the scan is
# allocation-free; the config itself stays `Any` because its concrete type
# depends on the runtime chunk size — its single dynamic use site is the
# ForwardDiff entry call, which is the specializing barrier anyway.
const _FDCfgStore = Vector{Tuple{Tuple{DataType, DataType, Int}, Any}}
const _FDHessCfgStore = Vector{Tuple{Tuple{DataType, DataType}, Any}}

mutable struct _LaplaceGradBuffers{T}
    grad_logf::Vector{T}
    Gθ::Matrix{T}
    grad_logdet_θ::Vector{T}
    grad_logdet_b::Vector{T}
    Jθ::Matrix{T}
    Jb::Matrix{T}
    nθ::Int
    nb::Int
    gradb_tmp::Vector{T}
    grad_logf_cfg::_FDCfgStore
    Gθ_cfg::_FDCfgStore
    gθ_grad_cfg::_FDCfgStore
    logdet_θ_cfg::_FDCfgStore
    logdet_b_cfg::_FDCfgStore
    Jθ_cfg::_FDCfgStore
    Jb_cfg::_FDCfgStore
end

struct _LaplaceHessCache{T, L, H, C, V}
    last_b::L
    last_logdet::Vector{T}
    last_H::H
    last_chol::C
    last_valid::V
end

mutable struct _LaplaceCache{T, B, G, A, H}
    θ_cache::Union{Nothing, AbstractVector{T}}
    bstar_cache::B
    grad_cache::G
    ad_cache::A
    hess_cache::H
end

function _LaplaceCache(θ_cache::AbstractVector{T},
        bstar_cache::_LaplaceBStarCache{T, B},
        grad_cache::_LaplaceGradCache{T, L, G, V},
        ad_cache::A,
        hess_cache::Union{Nothing, _LaplaceHessCache}) where {T, B, L, G, V, A}
    _LaplaceCache{
        T, typeof(bstar_cache), typeof(grad_cache), typeof(ad_cache), typeof(hess_cache)}(
        θ_cache, bstar_cache, grad_cache, ad_cache, hess_cache)
end

function _LaplaceCache(θ_cache::Nothing,
        bstar_cache::_LaplaceBStarCache{T, B},
        grad_cache::_LaplaceGradCache{T, L, G, V},
        ad_cache::A,
        hess_cache::Union{Nothing, _LaplaceHessCache}) where {T, B, L, G, V, A}
    _LaplaceCache{
        T, typeof(bstar_cache), typeof(grad_cache), typeof(ad_cache), typeof(hess_cache)}(
        θ_cache, bstar_cache, grad_cache, ad_cache, hess_cache)
end

function _init_laplace_ad_cache(n::Int)
    grad_cfg = Vector{Any}(undef, n)
    hess_cfg = Vector{Any}(undef, n)
    optf = Vector{Any}(undef, n)
    buffers = Vector{Any}(undef, n)
    fill!(grad_cfg, nothing)
    fill!(hess_cfg, nothing)
    fill!(optf, nothing)
    fill!(buffers, nothing)
    return LaplaceADCache(grad_cfg, hess_cfg, optf, buffers)
end

function _get_fd_cfg!(store::_FDCfgStore, f, x, builder::Function)
    key = (typeof(f), eltype(x), length(x))
    cfg = nothing
    if isempty(store)
        cfg = builder()
        push!(store, (key, cfg))
        return cfg
    end
    for (k, c) in store
        if k == key
            return c
        end
    end
    cfg = builder()
    push!(store, (key, cfg))
    return cfg
end

_ntri(n::Int) = n * (n + 1) ÷ 2

function _vech!(out::AbstractVector, H::AbstractMatrix)
    n = size(H, 1)
    idx = 1
    @inbounds for j in 1:n
        for i in 1:j
            out[idx] = H[i, j]
            idx += 1
        end
    end
    return out
end

function _vech(H::AbstractMatrix)
    out = Vector{eltype(H)}(undef, _ntri(size(H, 1)))
    return _vech!(out, H)
end

function _vech_weights!(out::AbstractVector, Ainv::AbstractMatrix)
    n = size(Ainv, 1)
    idx = 1
    @inbounds for j in 1:n
        for i in 1:j
            out[idx] = i == j ? Ainv[i, i] : 2 * Ainv[i, j]
            idx += 1
        end
    end
    return out
end

function _get_grad_buffers!(
        ad_cache::LaplaceADCache, bi::Int, T::Type, nθ::Int, nb::Int, use_trace::Bool)
    entry = ad_cache.buffers[bi]
    if entry === nothing || !(entry isa _LaplaceGradBuffers) || entry.nθ != nθ ||
       entry.nb != nb || !(eltype(entry.grad_logf) === T)
        grad_logf = Vector{T}(undef, nθ)
        Gθ = Matrix{T}(undef, nb, nθ)
        grad_logdet_θ = Vector{T}(undef, nθ)
        grad_logdet_b = Vector{T}(undef, nb)
        ntri = _ntri(nb)
        Jθ = use_trace ? Matrix{T}(undef, ntri, nθ) : Matrix{T}(undef, 0, 0)
        Jb = use_trace ? Matrix{T}(undef, ntri, nb) : Matrix{T}(undef, 0, 0)
        gradb_tmp = Vector{T}(undef, nb)
        entry = _LaplaceGradBuffers(grad_logf, Gθ, grad_logdet_θ, grad_logdet_b, Jθ, Jb,
            nθ, nb, gradb_tmp,
            _FDCfgStore(), _FDCfgStore(), _FDCfgStore(), _FDCfgStore(),
            _FDCfgStore(), _FDCfgStore(), _FDCfgStore())
        ad_cache.buffers[bi] = entry
    elseif use_trace && (size(entry.Jθ, 1) == 0 || size(entry.Jb, 1) == 0)
        ntri = _ntri(nb)
        entry.Jθ = Matrix{T}(undef, ntri, nθ)
        entry.Jb = Matrix{T}(undef, ntri, nb)
    end
    # The slot is an untyped `Vector{Any}` element — assert the concrete buffer
    # type here so every downstream field access dispatches statically.
    return entry::_LaplaceGradBuffers{T}
end

function _init_laplace_hess_cache(T::Type, n::Int)
    last_b = [Vector{T}() for _ in 1:n]
    last_logdet = fill(T(NaN), n)
    last_H = [Matrix{T}(undef, 0, 0) for _ in 1:n]
    last_chol = Vector{Any}(undef, n)
    fill!(last_chol, nothing)
    last_valid = falses(n)
    return _LaplaceHessCache(last_b, last_logdet, last_H, last_chol, last_valid)
end

# Fresh full EBE cache (b*/gradient/AD/Hessian sub-caches) — shared by every
# fit entry point that runs the Laplace-family inner problem.
function _init_laplace_eval_cache(n_batches::Int, T::Type{<:Real})
    bstar_cache = _LaplaceBStarCache([Vector{T}() for _ in 1:n_batches], falses(n_batches))
    grad_cache = _LaplaceGradCache([Vector{T}() for _ in 1:n_batches],
        fill(T(NaN), n_batches),
        [Vector{T}() for _ in 1:n_batches],
        falses(n_batches))
    ad_cache = _init_laplace_ad_cache(n_batches)
    hess_cache = _init_laplace_hess_cache(T, n_batches)
    return _LaplaceCache(nothing, bstar_cache, grad_cache, ad_cache, hess_cache)
end

function _re_start_value(dist, dim::Int, T)
    if dim == 1
        v = 0.0
        ok = false
        try
            v = mean(dist)
            ok = true
        catch
        end
        if !ok
            try
                v = median(dist)
                ok = true
            catch
            end
        end
        # A length-1 multivariate distribution (e.g. 1-D MvNormal) returns a
        # length-1 vector mean/median; the flat b-vector slot is scalar, so
        # reduce it to its single element.
        ok && v isa AbstractVector && (v = first(v))
        return ok ? T(v) : T(0.0)
    else
        v = nothing
        ok = false
        try
            v = mean(dist)
            ok = true
        catch
        end
        if !ok
            try
                v = median(dist)
                ok = true
            catch
            end
        end
        if ok && v isa AbstractVector && length(v) == dim
            return T.(v)
        else
            return zeros(T, dim)
        end
    end
end

@inline function _laplace_lhs_unit(n::Int, rng::AbstractRNG)
    n <= 0 && return Float64[]
    u = ((0:(n - 1)) .+ rand(rng, n)) ./ n
    return u[Random.randperm(rng, n)]
end

@inline function _laplace_lhs_draws_univariate(dist, n::Int, rng::AbstractRNG)
    n <= 0 && return Any[]
    dist isa Distributions.UnivariateDistribution || return nothing
    try
        Distributions.quantile(dist, 0.5)
    catch
        return nothing
    end
    u = _laplace_lhs_unit(n, rng)
    out = Vector{Any}(undef, n)
    @inbounds for i in eachindex(u)
        out[i] = Distributions.quantile(dist, u[i])
    end
    return out
end

@inline function _laplace_marginal_mvnormal(dist, i::Int)
    if dist isa Distributions.AbstractMvNormal
        μ = Distributions.mean(dist)
        Σ = Distributions.cov(dist)
        return Normal(μ[i], sqrt(Σ[i, i]))
    elseif dist isa Distributions.MvLogNormal
        μ = Distributions.mean(dist.normal)
        Σ = Distributions.cov(dist.normal)
        return LogNormal(μ[i], sqrt(Σ[i, i]))
    elseif dist isa Distributions.MvLogitNormal
        # MvLogitNormal: outer dim = d+1 (simplex), inner MvNormal dim = d.
        # The i-th ALR coordinate (for i ≤ d) is Normal(μ[i], sqrt(Σ[i,i])).
        # The reference category (i = d+1) has no simple marginal.
        i <= length(dist.normal) || return nothing
        μ = Distributions.mean(dist.normal)
        Σ = Distributions.cov(dist.normal)
        return Normal(μ[i], sqrt(Σ[i, i]))
    end
    return nothing
end

function _laplace_lhs_draws_mvnormal(dist, n::Int, rng::AbstractRNG, dim::Int)
    n <= 0 && return Any[]
    first_m = _laplace_marginal_mvnormal(dist, 1)
    first_m === nothing && return nothing
    draws_by_dim = Vector{Vector{Any}}(undef, dim)
    for j in 1:dim
        mj = _laplace_marginal_mvnormal(dist, j)
        mj === nothing && return nothing
        dj = _laplace_lhs_draws_univariate(mj, n, rng)
        dj === nothing && return nothing
        draws_by_dim[j] = dj
    end
    out = Vector{Vector{Any}}(undef, n)
    for i in 1:n
        vi = Vector{Any}(undef, dim)
        for j in 1:dim
            vi[j] = draws_by_dim[j][i]
        end
        out[i] = vi
    end
    return out
end

# Robustness wrapper: RE-distribution construction can throw on a numerically
# degenerate θ the outer optimizer steps into (e.g. a singular Ω, even on the
# cholesky scale via exp-underflow). Fall back to a zero start — the subsequent
# logf evaluation returns -Inf and the optimizer backtracks instead of the whole
# fit crashing (mirrors the `_laplace_logf_batch` exception policy).
function _laplace_default_b0(dm::DataModel,
        batch_info::REBatchInfo,
        θ::ComponentArray,
        const_cache::REConstantsCache,
        cache::_LLCache;
        tctx = nothing)
    try
        return _laplace_default_b0_impl(dm, batch_info, θ, const_cache, cache; tctx = tctx)
    catch err
        if _is_numeric_error(err)
            return zeros(eltype(θ), max(get_n_b(batch_info), 0))
        end
        rethrow(err)
    end
end

function _laplace_default_b0_impl(dm::DataModel,
        batch_info::REBatchInfo,
        θ::ComponentArray,
        const_cache::REConstantsCache,
        cache::_LLCache;
        tctx = nothing)
    nb = get_n_b(batch_info)
    nb == 0 && return Float64[]
    T = eltype(θ)
    b0 = zeros(T, nb)
    model_funs = cache.model_funs
    helpers = cache.helpers
    dists_builder = create_random_effect_distribution(get_random(get_model(dm)))
    θ_re = tctx === nothing ? _symmetrize_psd_params(θ, get_fixed(get_model(dm))) :
           tctx.θ_re
    cache = get_laplace_cache(get_re_group_info(dm))
    re_names = get_re_names(cache)
    for (ri, re) in enumerate(re_names)
        info = get_re_info(batch_info)[ri]
        isempty(get_levels(get_re_map(info))) && continue
        for (li, _) in enumerate(get_levels(get_re_map(info)))
            rep_idx = get_reps(info)[li]
            dists = tctx === nothing ?
                    dists_builder(
                θ_re, get_const_cov(get_individuals(dm)[rep_idx]), model_funs, helpers) :
                    tctx.dists[ri][li]
            dist = getproperty(dists, re)
            start = _re_start_value(dist, get_dim(info), T)
            r = get_ranges(info)[li]
            if get_is_scalar(info) || get_dim(info) == 1
                b0[first(r)] = start
            else
                b0[r] .= start
            end
        end
    end
    return b0
end

function _laplace_sample_b0s(dm::DataModel,
        batch_info::REBatchInfo,
        θ::ComponentArray,
        const_cache::REConstantsCache,
        cache::_LLCache,
        rng::AbstractRNG,
        n::Int,
        sampling::Symbol;
        tctx = nothing)
    n <= 0 && return Vector{Vector{eltype(θ)}}()
    _resolve_multistart_sampling(sampling, "inner multistart sampling")
    T = eltype(θ)
    nb = get_n_b(batch_info)
    nb == 0 && return [T[] for _ in 1:n]
    # On a numerically degenerate θ (see `_laplace_default_b0`) keep the zero
    # candidates instead of crashing — the logf screening discards bad starts.
    try
        return _laplace_sample_b0s_impl(
            dm, batch_info, θ, const_cache, cache, rng, n, sampling; tctx = tctx)
    catch err
        if _is_numeric_error(err)
            return [zeros(T, nb) for _ in 1:n]
        end
        rethrow(err)
    end
end

function _laplace_sample_b0s_impl(dm::DataModel,
        batch_info::REBatchInfo,
        θ::ComponentArray,
        const_cache::REConstantsCache,
        cache::_LLCache,
        rng::AbstractRNG,
        n::Int,
        sampling::Symbol;
        tctx = nothing)
    T = eltype(θ)
    nb = get_n_b(batch_info)
    b0s = [zeros(T, nb) for _ in 1:n]
    model_funs = cache.model_funs
    helpers = cache.helpers
    dists_builder = create_random_effect_distribution(get_random(get_model(dm)))
    θ_re = tctx === nothing ? _symmetrize_psd_params(θ, get_fixed(get_model(dm))) :
           tctx.θ_re
    cache = get_laplace_cache(get_re_group_info(dm))
    re_names = get_re_names(cache)
    for (ri, re) in enumerate(re_names)
        info = get_re_info(batch_info)[ri]
        isempty(get_levels(get_re_map(info))) && continue
        for (li, _) in enumerate(get_levels(get_re_map(info)))
            rep_idx = get_reps(info)[li]
            dists = tctx === nothing ?
                    dists_builder(
                θ_re, get_const_cov(get_individuals(dm)[rep_idx]), model_funs, helpers) :
                    tctx.dists[ri][li]
            dist = getproperty(dists, re)
            r = get_ranges(info)[li]
            dim = length(r)
            if get_is_scalar(info)
                draws = sampling === :lhs ? _laplace_lhs_draws_univariate(dist, n, rng) :
                        nothing
                draws === nothing && (draws = [rand(rng, dist) for _ in 1:n])
                for i in 1:n
                    v = draws[i]
                    if v isa AbstractVector
                        length(v) == 1 ||
                            error("Expected scalar random effect for $(re), got vector of length $(length(v)).")
                        v = v[1]
                    end
                    b0s[i][first(r)] = T(v)
                end
            else
                draws = (sampling === :lhs &&
                         (dist isa Distributions.AbstractMvNormal ||
                          dist isa Distributions.MvLogNormal ||
                          dist isa Distributions.MvLogitNormal)) ?
                        _laplace_lhs_draws_mvnormal(dist, n, rng, dim) : nothing
                draws === nothing && (draws = [rand(rng, dist) for _ in 1:n])
                for i in 1:n
                    v = draws[i]
                    if v isa Number
                        b0s[i][r] .= T(v)
                    else
                        vv = vec(v)
                        length(vv) == dim ||
                            error("Expected random effect draw for $(re) with length $(dim), got $(length(vv)).")
                        b0s[i][r] .= T.(vv)
                    end
                end
            end
        end
    end
    return b0s
end

@inline function _laplace_gradb_cached!(cache::_LaplaceCache,
        bi::Int,
        dm::DataModel,
        info::REBatchInfo,
        θ::ComponentArray,
        const_cache::REConstantsCache,
        ll_cache::_LLCache,
        b;
        tctx = nothing)
    grad_cache = cache.grad_cache
    if grad_cache.last_valid[bi]
        b_last = grad_cache.last_b[bi]
        if length(b_last) == length(b)
            same = true
            @inbounds for i in eachindex(b_last)
                if b_last[i] != b[i]
                    same = false
                    break
                end
            end
            if same
                return (grad_cache.last_gradb[bi], grad_cache.last_logf[bi])
            end
        end
    end
    f = _LaplaceLogfBatch(dm, info, θ, const_cache, ll_cache, tctx)
    logf = f(b)
    entry = cache.ad_cache.grad_cfg[bi]
    if entry === nothing
        entry = _FDCfgStore()
        cache.ad_cache.grad_cfg[bi] = entry
    end
    entry = entry::_FDCfgStore
    cfg = _get_fd_cfg!(entry, f, b, () -> ForwardDiff.GradientConfig(f, b))
    gradb = grad_cache.last_gradb[bi]
    if length(gradb) != length(b)
        gradb = similar(b)
        grad_cache.last_gradb[bi] = gradb
    end
    ForwardDiff.gradient!(gradb, f, b, cfg)
    b_last = grad_cache.last_b[bi]
    if length(b_last) != length(b)
        resize!(b_last, length(b))
    end
    copyto!(b_last, b)
    grad_cache.last_logf[bi] = logf
    grad_cache.last_valid[bi] = true
    return (gradb, logf)
end

# Evaluates log p(η_const | θ) for all constant RE levels in the batch.
# Each unique constant level contributes exactly once (deduplicated via `seen`).
function _const_re_prior_logf(dm::DataModel,
        batch_info::REBatchInfo,
        θ_re::ComponentArray,
        const_cache::REConstantsCache,
        cache::_LLCache;
        anneal_sds::NamedTuple = NamedTuple())
    isempty(const_cache.is_const) && return zero(eltype(θ_re))
    T = eltype(θ_re)
    ll = zero(T)
    re_cache = get_laplace_cache(get_re_group_info(dm))
    dists_builder = create_random_effect_distribution(get_random(get_model(dm)))
    has_anneal = !isempty(anneal_sds)
    for (ri, re) in enumerate(get_re_names(re_cache))
        const_mask = const_cache.is_const[ri]
        any(const_mask) || continue
        seen = falses(length(const_mask))
        for i in get_inds(batch_info)
            for id in get_ind_level_ids(re_cache)[i][ri]
                (const_mask[id] && !seen[id]) || continue
                seen[id] = true
                dists = dists_builder(θ_re, get_const_cov(get_individuals(dm)[i]),
                    cache.model_funs, cache.helpers)
                v = get_is_scalar(re_cache)[ri] ?
                    T(const_cache.scalar_vals[ri][id]) :
                    T.(const_cache.vector_vals[ri][id])
                lp = if has_anneal && haskey(anneal_sds, re)
                    dist = _saem_apply_anneal_dist(getproperty(dists, re),
                        getfield(anneal_sds, re))
                    convert(T, logpdf(dist, v))
                else
                    _logpdf_re_static(keys(dists), values(dists), re, v, T)
                end
                isfinite(lp) || return T(-Inf)
                ll += lp
            end
        end
    end
    return ll
end

# Wrapper around the batch log-density. An invalid RE covariance — e.g. a degenerate
# matrix-exp (`:expm`) draw the optimizer/UQ steps into — makes distribution construction
# throw (`PosDefException` etc.) rather than return a finite density. Treat such numeric
# failures as a -Inf log-density so callers backtrack instead of crashing. The try/catch is
# free on the non-throwing path, so the hot-path performance of the impl is preserved.
function _laplace_logf_batch(
        dm::DataModel, batch_info::REBatchInfo, θ::ComponentArray,
        b, const_cache::REConstantsCache, cache::_LLCache;
        anneal_sds::NamedTuple = NamedTuple(), tctx = nothing)
    T = promote_type(eltype(θ), eltype(b))
    try
        # The RE-prior terms box through the RE-distribution RGF (`dists_builder` returns
        # `Any`, a codegen-level limit shared with `calculate_prede`), so the impl infers
        # `Any`. Pin the batch log-density to `T` here so the differentiated M-step Q
        # objective (`acc += w*logf` in the Q-core) stays concrete. AD-safe: `convert` is
        # identity when the value is already `T`, an AD-correct zero-partial lift otherwise.
        return convert(T,
            _laplace_logf_batch_impl(dm, batch_info, θ, b, const_cache,
                cache; anneal_sds = anneal_sds, tctx = tctx))::T
    catch err
        if _is_numeric_error(err)
            return convert(T, -Inf)::T
        end
        rethrow(err)
    end
end

# Type-stable logpdf for one RE picked out of the dists NamedTuple by name.
# `getproperty(dists, re)` with a runtime Symbol makes the subsequent `logpdf` a
# dynamic call, which routes Enzyme through its runtime-activity rules — and those
# have no forward-mode implementation for the BLAS/LAPACK kernels inside e.g.
# MvNormal's logpdf (`trtrs`: "Runtime Activity not yet implemented for
# Forward-Mode BLAS calls"). Recursive tuple peeling keeps every logpdf call
# statically dispatched; recursion depth = number of REs (small).
function _logpdf_re_static(ks::Tuple, ds::Tuple, re::Symbol, v, ::Type{T}) where {T}
    if first(ks) === re
        return convert(T, logpdf(first(ds), v))
    end
    return _logpdf_re_static(Base.tail(ks), Base.tail(ds), re, v, T)
end
function _logpdf_re_static(::Tuple{}, ::Tuple{}, re::Symbol, v, ::Type{T}) where {T}
    throw(ArgumentError("Random effect $(re) not found in distributions NamedTuple."))
end

# θ-only work shared across the many b-evaluations of one batch at a fixed θ:
# the symmetrized parameter vector and the per-(RE, free-level) distribution
# NamedTuples (full builder outputs, so `_logpdf_re_static` keeps its static
# dispatch). Optionally carries the prior Hessian Λ = -∇²_b log π(b), which is
# constant in b when every RE distribution is Gaussian — FOCEI reuses it across
# its negH builds (b-paths only; θ-dual paths rebuild so ∂Λ/∂θ stays exact).
#
# Correctness contract: a context is valid only for the θ it was built from and
# must not cross a θ change — every use below builds it locally inside a scope
# where θ is fixed (one inner solve, one gradient, one Hessian, one batch term).
struct BatchThetaContext{TH, D, L}
    θ_re::TH
    dists::D          # Vector (per RE) of Vector (per free level) of builder NamedTuples
    prior_hess::L     # nothing, or cached Λ (all-Gaussian REs only)
end
const _LaplaceThetaCtx = BatchThetaContext

function build_batch_theta_context(dm::DataModel,
        batch_info::REBatchInfo,
        θ::ComponentArray,
        cache::_LLCache)
    θ_re = _symmetrize_psd_params(θ, get_fixed(get_model(dm)))
    builder = create_random_effect_distribution(get_random(get_model(dm)))
    model_funs = cache.model_funs
    helpers = cache.helpers
    dists = [[builder(θ_re, get_const_cov(get_individuals(dm)[get_reps(rinfo)[li]]),
                  model_funs, helpers)
              for li in eachindex(get_levels(get_re_map(rinfo)))]
             for rinfo in get_re_info(batch_info)]
    return _LaplaceThetaCtx(θ_re, dists, nothing)
end
const _build_theta_ctx = build_batch_theta_context

@inline function _re_all_gaussian(dm::DataModel)
    types = get_re_types(get_random(get_model(dm)))
    return all(s -> s === :Normal || s === :MvNormal, values(types))
end

# Attach the (b-constant, Gaussian-RE) prior Hessian to a context. Computed at
# `b` via the same ForwardDiff path FOCEI uses, so the cached matrix is
# bit-identical to a per-call evaluation; for Gaussian REs the prior is
# quadratic in b, hence the Hessian is the same at every b and its ∂/∂b is
# exactly zero — matching the dual partials the per-call path would produce.
function _ctx_with_prior_hess(tctx::_LaplaceThetaCtx,
        dm::DataModel,
        batch_info::REBatchInfo,
        const_cache::REConstantsCache,
        cache::_LLCache,
        b)
    _re_all_gaussian(dm) || return tctx
    Λ = -ForwardDiff.hessian(
        _FOCEIPriorLogf(dm, batch_info, tctx.θ_re, const_cache, cache), b)
    return _LaplaceThetaCtx(tctx.θ_re, tctx.dists, Λ)
end

# Exception-safe variant of `_build_theta_ctx`: a degenerate θ falls back to
# `nothing` (the per-call path, whose logf wrapper turns the same exceptions
# into -Inf), instead of letting the throw escape a path that was previously
# protected only inside the evaluation.
function _safe_theta_ctx(dm::DataModel,
        batch_info::REBatchInfo,
        θ::ComponentArray,
        cache::_LLCache)
    try
        return _build_theta_ctx(dm, batch_info, θ, cache)
    catch err
        if _is_numeric_error(err)
            return nothing
        end
        rethrow(err)
    end
end

# Context for one objective/gradient batch term; FOCEI additionally gets the
# cached prior Hessian when the REs are all Gaussian. (`hmode` is untyped here
# because `_HessMode` is defined further down this file.)
function _objective_theta_ctx(dm::DataModel,
        batch_info::REBatchInfo,
        θ::ComponentArray,
        const_cache::REConstantsCache,
        cache::_LLCache,
        b,
        hmode)
    tctx = _safe_theta_ctx(dm, batch_info, θ, cache)
    tctx === nothing && return nothing
    hmode isa _FOCEIHess || return tctx
    return try
        _ctx_with_prior_hess(tctx, dm, batch_info, const_cache, cache, b)
    catch err
        if _is_numeric_error(err)
            tctx
        else
            rethrow(err)
        end
    end
end

# Free-level RE prior term log p(η_free | θ_re) for a batch. Shared by the Laplace/FOCEI/SAEM
# joint (`_laplace_logf_batch_impl`), the Q2 M-step (`_re_logpdf_batch`), and the AGHQ
# logcorrection (`_re_prior_logf_batch`). Uses `_logpdf_re_static` so the logpdf stays
# statically dispatched (Enzyme forward-mode BLAS safe); the anneal branch keeps dynamic
# getproperty, matching the previous per-function behaviour.
function _re_free_prior_logf(dm::DataModel, batch_info::REBatchInfo, θ_re::ComponentArray,
        b, cache::_LLCache; anneal_sds::NamedTuple = NamedTuple(), tctx = nothing)
    T_ll = promote_type(eltype(θ_re), eltype(b))
    ll = zero(T_ll)
    model_funs = cache.model_funs
    helpers = cache.helpers
    dists_builder = create_random_effect_distribution(get_random(get_model(dm)))
    re_cache = get_laplace_cache(get_re_group_info(dm))
    re_names = get_re_names(re_cache)
    has_anneal = !isempty(anneal_sds)
    for (ri, re) in enumerate(re_names)
        info = get_re_info(batch_info)[ri]
        isempty(get_levels(get_re_map(info))) && continue
        for (li, level_id) in enumerate(get_levels(get_re_map(info)))
            rep_idx = get_reps(info)[li]
            dists = tctx === nothing ?
                    dists_builder(θ_re, get_const_cov(get_individuals(dm)[rep_idx]),
                model_funs, helpers) : tctx.dists[ri][li]
            v = _re_value_from_b(info, level_id, b)
            v === nothing && continue
            lp = if has_anneal && haskey(anneal_sds, re)
                dist = _saem_apply_anneal_dist(getproperty(dists, re),
                    getfield(anneal_sds, re))
                convert(T_ll, logpdf(dist, v))
            else
                _logpdf_re_static(keys(dists), values(dists), re, v, T_ll)
            end
            isfinite(lp) || return T_ll(-Inf)
            ll += lp
        end
    end
    return ll
end

function _laplace_logf_batch_impl(dm::DataModel,
        batch_info::REBatchInfo,
        θ::ComponentArray,
        b,
        const_cache::REConstantsCache,
        cache::_LLCache;
        anneal_sds::NamedTuple = NamedTuple(),
        tctx = nothing)
    T_ll = promote_type(eltype(θ), eltype(b))
    θ_re = tctx === nothing ? _symmetrize_psd_params(θ, get_fixed(get_model(dm))) :
           tctx.θ_re
    # random-effects prior term (free levels)
    ll = _re_free_prior_logf(dm, batch_info, θ_re, b, cache;
        anneal_sds = anneal_sds, tctx = tctx)
    !isfinite(ll) && return T_ll(-Inf)
    # constant-RE prior term
    const_ll = _const_re_prior_logf(
        dm, batch_info, θ_re, const_cache, cache; anneal_sds = anneal_sds)
    !isfinite(const_ll) && return T_ll(-Inf)
    ll += const_ll
    # likelihood term
    for i in get_inds(batch_info)
        η_ind = _build_eta_ind(dm, i, batch_info, b, const_cache, θ_re)
        lli = _loglikelihood_individual(dm, i, θ_re, η_ind, cache)
        !isfinite(lli) && return -Inf
        ll += lli
    end
    return ll
end

# RE-prior-only counterpart to _laplace_logf_batch.
# Evaluates log p(η | θ_re) for all free RE levels in the batch — no ODE, no observation
# likelihood.  Used by the Q2 M-step optimizer to update parameters that appear only in
# random-effect distribution expressions.
function _re_logpdf_batch(dm::DataModel,
        batch_info::REBatchInfo,
        θ::ComponentArray,
        b,
        const_cache::REConstantsCache,
        cache::_LLCache;
        anneal_sds::NamedTuple = NamedTuple(),
        tctx = nothing)
    T_ll = promote_type(eltype(θ), eltype(b))
    θ_re = tctx === nothing ? _symmetrize_psd_params(θ, get_fixed(get_model(dm))) :
           tctx.θ_re
    ll = _re_free_prior_logf(dm, batch_info, θ_re, b, cache;
        anneal_sds = anneal_sds, tctx = tctx)
    !isfinite(ll) && return T_ll(-Inf)
    const_ll = _const_re_prior_logf(
        dm, batch_info, θ_re, const_cache, cache; anneal_sds = anneal_sds)
    !isfinite(const_ll) && return T_ll(-Inf)
    # Pin to `T_ll`: the prior terms box through the RE-distribution RGF (`Any`), and the
    # Q2 M-step optimizer differentiates this objective. AD-safe convert (see above).
    return convert(T_ll, ll + const_ll)::T_ll
end

# Returns only the RE prior log-density term for a batch (no observation likelihood).
# Used by build_centered_re_measure for the AGHQ logcorrection.
function _re_prior_logf_batch(dm::DataModel,
        batch_info::REBatchInfo,
        θ::ComponentArray,
        b,
        const_cache::REConstantsCache,
        ll_cache::_LLCache)
    θ_re = _symmetrize_psd_params(θ, get_fixed(get_model(dm)))
    return _re_free_prior_logf(dm, batch_info, θ_re, b, ll_cache)
end

"""
    NewtonInner(; max_dim=32, maxiters=100, g_abstol=1e-8, alpha_min=1e-4)

Opt-in inner-EBE optimizer for [`Laplace`](@ref)/[`FOCEI`](@ref) (and the
`ebe_optimizer` options): a damped Newton ascent on the batch log-joint, built
on the same AD gradient/Hessian and adaptive-jitter Cholesky machinery the
Laplace approximation already uses. Pass it as
`Laplace(inner_optimizer=NewtonInner())`.

The default inner optimizer is unchanged (LBFGS via Optimization.jl); this is
an option because a different optimizer can reach a (numerically) different
conditional mode on hard problems. Newton converges quadratically from the
warm starts the outer optimization provides and avoids the per-solve
`OptimizationProblem` setup overhead.

- `max_dim`: batches with more than `max_dim` random-effect dimensions fall
  back to the default LBFGS path.
- `maxiters`: maximum Newton iterations.
- `g_abstol`: gradient sup-norm convergence tolerance (matches the Optim.jl
  default of `1e-8`).
- `alpha_min`: smallest backtracking step before the solve is declared stalled
  (a stalled or failed Newton solve automatically falls back to the default
  LBFGS path from the best point found, so robustness is never worse than the
  default).
"""
struct NewtonInner
    max_dim::Int
    maxiters::Int
    g_abstol::Float64
    alpha_min::Float64
end

function NewtonInner(; max_dim::Int = 32, maxiters::Int = 100, g_abstol::Float64 = 1e-8,
        alpha_min::Float64 = 1e-4)
    NewtonInner(max_dim, maxiters, g_abstol, alpha_min)
end

# Minimal solution wrapper for the Newton path, duck-typed for the accessors
# `_laplace_sol_logf` / `_laplace_sol_grad_norm` and the `sol.u` reads in
# `_laplace_compute_bstar_batch!`.
struct _NewtonSol{U}
    u::U
    logf::Float64
    g_norm::Float64
    converged::Bool
end

# Damped Newton ascent on b ↦ logf(b) at fixed (Float64) θ. Direction
# Δ = (-H + λI)⁻¹ g via the adaptive-jitter Cholesky (λ escalation doubles as
# Levenberg-style globalization away from the mode), Armijo backtracking on
# logf. Returns a `_NewtonSol`; `converged=false` signals the caller to fall
# back to the default quasi-Newton path.
function _newton_inner_solve(dm::DataModel,
        batch_info::REBatchInfo,
        θ_val::ComponentArray,
        const_cache::REConstantsCache,
        cache::_LLCache,
        ad_cache::LaplaceADCache,
        bi::Int,
        b0::Vector{Float64},
        newton::NewtonInner;
        tctx = nothing)
    nb = length(b0)
    # Exception-safe: a degenerate θ falls back to the per-call path inside the
    # logf wrapper (-Inf) instead of throwing out of the inner solve.
    tctx === nothing && (tctx = _safe_theta_ctx(dm, batch_info, θ_val, cache))
    f = _LaplaceLogfBatch(dm, batch_info, θ_val, const_cache, cache, tctx)
    b = copy(b0)
    fval = f(b)
    isfinite(fval) || return _NewtonSol(b, -Inf, Inf, false)
    entry = ad_cache.grad_cfg[bi]
    if entry === nothing
        entry = _FDCfgStore()
        ad_cache.grad_cfg[bi] = entry
    end
    cfg = _get_fd_cfg!(entry::_FDCfgStore, f, b, () -> ForwardDiff.GradientConfig(f, b))
    g = Vector{Float64}(undef, nb)
    ForwardDiff.gradient!(g, f, b, cfg)
    b_try = similar(b)
    for _ in 1:(newton.maxiters)
        gn = maximum(abs, g)
        isfinite(gn) || return _NewtonSol(b, Float64(fval), Inf, false)
        gn <= newton.g_abstol && return _NewtonSol(b, Float64(fval), gn, true)
        H = _laplace_hessian_b(dm, batch_info, θ_val, b, const_cache, cache, ad_cache, bi;
            ctx = "newton_inner", tctx = tctx)
        chol, _ = _laplace_cholesky_negH(H; jitter = 1e-6, max_tries = 8, growth = 10.0,
            adaptive = true, scale_factor = 1e-6)
        (chol === nothing || chol.info != 0) &&
            return _NewtonSol(b, Float64(fval), gn, false)
        Δ = chol \ g
        slope = dot(g, Δ)
        (isfinite(slope) && slope > 0) || return _NewtonSol(b, Float64(fval), gn, false)
        α = 1.0
        accepted = false
        while α >= newton.alpha_min
            @. b_try = b + α * Δ
            f_try = f(b_try)
            if isfinite(f_try) && f_try >= fval + 1e-4 * α * slope
                b, b_try = b_try, b
                fval = f_try
                accepted = true
                break
            end
            α /= 2
        end
        accepted || return _NewtonSol(b, Float64(fval), gn, false)
        ForwardDiff.gradient!(g, f, b, cfg)
    end
    gn = maximum(abs, g)
    return _NewtonSol(b, Float64(fval), isfinite(gn) ? gn : Inf, false)
end

@inline _laplace_sol_logf(sol::_NewtonSol) = sol.logf
@inline _laplace_sol_grad_norm(sol::_NewtonSol) = sol.g_norm

function _laplace_solve_batch!(dm::DataModel,
        batch_info::REBatchInfo,
        θ::ComponentArray,
        const_cache::REConstantsCache,
        cache::_LLCache,
        ad_cache::LaplaceADCache,
        bi::Int,
        b0;
        optimizer = OptimizationOptimJL.LBFGS(linesearch = LineSearches.BackTracking(maxstep = 1.0)),
        optim_kwargs::NamedTuple = NamedTuple(),
        adtype = Optimization.AutoForwardDiff(),
        tctx = nothing)
    nb = get_n_b(batch_info)
    nb == 0 && return Float64[]
    θ_val = _laplace_floatize(θ)
    T = eltype(θ_val)
    b0_use = if b0 isa AbstractVector
        if length(b0) == nb && eltype(b0) === T
            b0
        else
            out = Vector{T}(undef, nb)
            @inbounds for i in 1:nb
                out[i] = b0[i]
            end
            out
        end
    else
        zeros(T, nb)
    end
    optimizer_use = optimizer
    if optimizer isa NewtonInner
        if nb <= optimizer.max_dim
            nsol = _newton_inner_solve(dm, batch_info, θ_val, const_cache, cache,
                ad_cache, bi, collect(Float64, b0_use), optimizer;
                tctx = tctx)
            nsol.converged && return nsol
            # Stalled/failed Newton: continue into the default quasi-Newton path
            # from the best point found, so the option is never less robust
            # than the default.
            b0_use = nsol.u
        end
        optimizer_use = OptimizationOptimJL.LBFGS(linesearch = LineSearches.BackTracking(maxstep = 1.0))
    end
    opt_entry = ad_cache.optf[bi]
    if opt_entry === nothing || opt_entry[1] != typeof(adtype)
        f = _LaplaceLogfBatchParam(dm, batch_info, const_cache, cache)
        optf = OptimizationFunction(f, adtype)
        opt_entry = (typeof(adtype), optf)
        ad_cache.optf[bi] = opt_entry
    end
    optf = opt_entry[2]
    p = tctx === nothing ? θ_val : (θ_val, tctx)
    prob = OptimizationProblem(optf, b0_use, p)
    sol = solve(prob, optimizer_use; optim_kwargs...)
    return sol
end

@inline function _laplace_sol_logf(sol)
    hasproperty(sol, :objective) || return NaN
    f = -getproperty(sol, :objective)
    return f isa Real ? Float64(f) : NaN
end

@inline function _laplace_sol_grad_norm(sol)
    SciMLBase.successful_retcode(sol) || return Inf
    hasproperty(sol, :original) || return NaN
    raw = getproperty(sol, :original)
    hasproperty(raw, :g_residual) || return NaN
    gres = getproperty(raw, :g_residual)
    if gres isa Real
        g = abs(gres)
        return isfinite(g) ? Float64(g) : Inf
    elseif gres isa AbstractArray
        g = maximum(abs, gres)
        return isfinite(g) ? Float64(g) : Inf
    end
    return NaN
end

@inline function _laplace_store_bstar!(cache::_LaplaceCache, bi::Int, b)
    slot = cache.bstar_cache.b_star[bi]
    nb = length(b)
    if length(slot) == nb
        copyto!(slot, b)
    else
        cache.bstar_cache.b_star[bi] = collect(b)
    end
    cache.bstar_cache.has_bstar[bi] = true
    return cache.bstar_cache.b_star[bi]
end

function _laplace_compute_bstar_batch!(cache::_LaplaceCache,
        bi::Int,
        dm::DataModel,
        info::REBatchInfo,
        θ_val::ComponentArray,
        const_cache::REConstantsCache,
        ll_cache::_LLCache;
        optimizer = OptimizationOptimJL.LBFGS(linesearch = LineSearches.BackTracking(maxstep = 1.0)),
        optim_kwargs::NamedTuple = NamedTuple(),
        adtype = Optimization.AutoForwardDiff(),
        grad_tol = 1e-6,
        multistart = LaplaceMultistartOptions(0, 0, grad_tol, 5, :lhs),
        rng::AbstractRNG = Random.default_rng(),
        mcmc_candidates::Union{Nothing, AbstractMatrix} = nothing)
    nb = get_n_b(info)
    if nb == 0
        b_slot = cache.bstar_cache.b_star[bi]
        if !isempty(b_slot)
            resize!(b_slot, 0)
        end
        cache.bstar_cache.has_bstar[bi] = true
        return nothing
    end
    # θ-only work (symmetrize + RE-distribution table) hoisted once per batch
    # per θ-evaluation — every logf/gradient/Hessian/solve below reuses it.
    tctx = try
        _build_theta_ctx(dm, info, θ_val, ll_cache)
    catch err
        if _is_numeric_error(err)
            nothing   # degenerate θ: fall back to the per-call path (returns -Inf downstream)
        else
            rethrow(err)
        end
    end
    b0_default = _laplace_default_b0(dm, info, θ_val, const_cache, ll_cache; tctx = tctx)
    b_start = b0_default
    warm_start_usable = false
    if cache.bstar_cache.has_bstar[bi]
        b_prev = cache.bstar_cache.b_star[bi]
        if length(b_prev) == nb
            f_prev = _laplace_logf_batch(
                dm, info, θ_val, b_prev, const_cache, ll_cache; tctx = tctx)
            f_def = _laplace_logf_batch(
                dm, info, θ_val, b0_default, const_cache, ll_cache; tctx = tctx)
            warm_start_usable = isfinite(f_prev)
            if f_prev > f_def
                b_start = b_prev
            end
        end
    end
    g0, f_start = _laplace_gradb_cached!(
        cache, bi, dm, info, θ_val, const_cache, ll_cache, b_start; tctx = tctx)
    if !isfinite(f_start)
        best_f = -Inf
        best_b = b_start
        n_try = multistart.n > 0 ? multistart.n : 0
        b_tries = _laplace_sample_b0s(dm, info, θ_val, const_cache, ll_cache, rng,
            n_try, multistart.sampling; tctx = tctx)
        for b_try in b_tries
            f_try = _laplace_logf_batch(
                dm, info, θ_val, b_try, const_cache, ll_cache; tctx = tctx)
            if f_try > best_f
                best_f = f_try
                best_b = b_try
            end
        end
        if isfinite(best_f)
            b_start = best_b
        end
        g0, _ = _laplace_gradb_cached!(
            cache, bi, dm, info, θ_val, const_cache, ll_cache, b_start; tctx = tctx)
    end
    g0_norm = maximum(abs, g0)
    if !isfinite(g0_norm)
        g0_norm = Inf
    end
    b_best = b_start
    best_f = isfinite(f_start) ? f_start : -Inf
    g_best_norm = g0_norm
    sol_best = try
        _laplace_solve_batch!(
            dm, info, θ_val, const_cache, ll_cache, cache.ad_cache, bi, b_start;
            optimizer = optimizer,
            optim_kwargs = optim_kwargs,
            adtype = adtype,
            tctx = tctx)
    catch err
        if err isa DomainError || err isa ArgumentError || err isa ErrorException
            nothing
        else
            rethrow(err)
        end
    end
    if sol_best !== nothing
        b_best = sol_best.u
        best_f = _laplace_sol_logf(sol_best)
        if !isfinite(best_f)
            best_f = _laplace_logf_batch(
                dm, info, θ_val, b_best, const_cache, ll_cache; tctx = tctx)
            isfinite(best_f) || (best_f = -Inf)
        end
        g_best_norm = _laplace_sol_grad_norm(sol_best)
        if !isfinite(g_best_norm)
            g_best, _ = _laplace_gradb_cached!(
                cache, bi, dm, info, θ_val, const_cache, ll_cache, b_best; tctx = tctx)
            g_best_norm = maximum(abs, g_best)
        end
        if !isfinite(g_best_norm)
            g_best_norm = Inf
        end
    end
    use_mcmc = mcmc_candidates !== nothing && size(mcmc_candidates, 2) > 0
    # The inner multistart fires on cold starts (no usable previous mode for this
    # batch), on failed/non-finite solves, or when MCMC candidates were explicitly
    # provided. With a finite warm start the LBFGS refinement tracks the
    # conditional mode as θ moves through the outer line search; prior-drawn
    # candidates were measured to consume the bulk of every ODE objective
    # evaluation without improving the result (the gradient tolerance often sits
    # at the ODE solver-noise floor). The thorough multistart still runs wherever
    # no warm start exists — the first outer evaluation, the final-EBE pass,
    # rescue passes, and `reestimate_ebes` (all of which build fresh b*-caches).
    multistart_allowed = use_mcmc || !warm_start_usable ||
                         !isfinite(best_f) || !isfinite(g_best_norm)
    if multistart_allowed && g_best_norm > multistart.grad_tol && multistart.k > 0 &&
       (multistart.n > 0 || use_mcmc)
        n_mcmc_stored = use_mcmc ? size(mcmc_candidates, 2) : 0
        n = use_mcmc ? max(multistart.n, n_mcmc_stored) : multistart.n
        k = multistart.k
        max_rounds = use_mcmc ? 1 : max(1, multistart.max_rounds)
        for round in 1:max_rounds
            k = min(k, n)
            if use_mcmc
                mcmc_b0s = [mcmc_candidates[:, i] for i in 1:n_mcmc_stored]
                n_lhs = max(0, multistart.n - n_mcmc_stored)
                lhs_b0s = n_lhs > 0 ?
                          _laplace_sample_b0s(dm, info, θ_val, const_cache, ll_cache, rng,
                    n_lhs, multistart.sampling; tctx = tctx) :
                          Vector{Vector{eltype(θ_val)}}()
                b0s = vcat(mcmc_b0s, lhs_b0s)
                n = length(b0s)
                k = min(k, n)
            else
                b0s = _laplace_sample_b0s(dm, info, θ_val, const_cache, ll_cache,
                    rng, n, multistart.sampling; tctx = tctx)
            end
            vals = Vector{Tuple{Float64, Vector{eltype(θ_val)}}}(undef, n)
            for s in 1:n
                b0 = b0s[s]
                f0 = _laplace_logf_batch(
                    dm, info, θ_val, b0, const_cache, ll_cache; tctx = tctx)
                isfinite(f0) || (f0 = -Inf)
                vals[s] = (f0, b0)
            end
            idxs = k < n ? partialsortperm(vals, 1:k; by = x -> x[1], rev = true) : (1:n)
            best_after = b_best
            best_after_f = best_f
            best_after_g_norm = g_best_norm
            for s in idxs
                b0 = vals[s][2]
                sol_try = try
                    _laplace_solve_batch!(
                        dm, info, θ_val, const_cache, ll_cache, cache.ad_cache, bi, b0;
                        optimizer = optimizer,
                        optim_kwargs = optim_kwargs,
                        adtype = adtype,
                        tctx = tctx)
                catch err
                    if err isa DomainError || err isa ArgumentError ||
                       err isa ErrorException
                        nothing
                    else
                        rethrow(err)
                    end
                end
                sol_try === nothing && continue
                b_try = sol_try.u
                f_try = _laplace_sol_logf(sol_try)
                if !isfinite(f_try)
                    f_try = _laplace_logf_batch(
                        dm, info, θ_val, b_try, const_cache, ll_cache; tctx = tctx)
                end
                isfinite(f_try) || (f_try = -Inf)
                if f_try > best_after_f
                    best_after = b_try
                    best_after_f = f_try
                    g_try_norm = _laplace_sol_grad_norm(sol_try)
                    if !isfinite(g_try_norm)
                        g_try, _ = _laplace_gradb_cached!(
                            cache, bi, dm, info, θ_val, const_cache,
                            ll_cache, b_try; tctx = tctx)
                        g_try_norm = maximum(abs, g_try)
                    end
                    best_after_g_norm = isfinite(g_try_norm) ? g_try_norm : Inf
                end
            end
            b_best = best_after
            best_f = best_after_f
            g_best_norm = best_after_g_norm
            if g_best_norm <= multistart.grad_tol
                break
            end
            use_mcmc && break
            n *= 2
            k = min(k * 2, n)
        end
    end
    _laplace_store_bstar!(cache, bi, b_best)
    return nothing
end

function _laplace_get_bstar!(cache::_LaplaceCache,
        dm::DataModel,
        batch_infos::Vector{REBatchInfo},
        θ::ComponentArray,
        const_cache::REConstantsCache,
        ll_cache;
        optimizer = OptimizationOptimJL.LBFGS(linesearch = LineSearches.BackTracking(maxstep = 1.0)),
        optim_kwargs::NamedTuple = NamedTuple(),
        adtype = Optimization.AutoForwardDiff(),
        grad_tol = 1e-6,
        theta_tol = 0.0,
        multistart = LaplaceMultistartOptions(0, 0, grad_tol, 5, :lhs),
        rng::AbstractRNG = Random.default_rng(),
        serialization::SciMLBase.EnsembleAlgorithm = EnsembleThreads(),
        progress::Bool = false,
        progress_desc::AbstractString = "EBE",
        mcmc_candidates_by_batch::Union{Nothing, Vector} = nothing,
        active_batches::Union{Nothing, AbstractSet{Int}} = nothing)
    θ_vec = θ
    if cache.θ_cache !== nothing && length(cache.θ_cache) == length(θ_vec)
        maxdiff = _maxabsdiff(θ_vec, cache.θ_cache)
        if isfinite(maxdiff) && maxdiff <= theta_tol
            return cache.bstar_cache.b_star
        end
    end
    if cache.grad_cache.last_valid !== nothing
        fill!(cache.grad_cache.last_valid, false)
    end
    total_inds = sum(
        max(0, get_n_b(info))
        for (bi, info) in enumerate(batch_infos)
        if active_batches === nothing || bi ∈ active_batches;
        init = 0)
    p = ProgressMeter.Progress(
        total_inds; desc = progress_desc, enabled = progress && total_inds > 0)
    θ_val = _laplace_floatize(θ)
    batch_rngs = _laplace_thread_rngs(rng, length(batch_infos))
    use_threaded = serialization isa SciMLBase.EnsembleThreads ||
                   ll_cache isa AbstractVector
    if use_threaded
        nthreads = Threads.maxthreadid()
        caches = _laplace_thread_caches(dm, ll_cache, nthreads)
        ind_counter = Threads.Atomic{Int}(0)
        # Chunk-indexed cache assignment: each task owns cache slot `c` for its whole
        # stride. Indexing by `Threads.threadid()` is unsafe under task migration
        # (`@threads :dynamic` can move a task between threads at yield points — and the
        # inner `solve(...)` yields — so two tasks may share one cache slot and race on
        # its mutable buffers, making the EB modes and the whole fit nondeterministic
        # across processes).
        n_chunks = length(caches)
        Threads.@threads for c in 1:n_chunks
            cache_c = caches[c]
            for bi in c:n_chunks:length(batch_infos)
                active_batches !== nothing && !(bi ∈ active_batches) && continue
                info = batch_infos[bi]
                _laplace_compute_bstar_batch!(
                    cache, bi, dm, info, θ_val, const_cache, cache_c;
                    optimizer = optimizer,
                    optim_kwargs = optim_kwargs,
                    adtype = adtype,
                    grad_tol = grad_tol,
                    multistart = multistart,
                    rng = batch_rngs[bi],
                    mcmc_candidates = mcmc_candidates_by_batch === nothing ? nothing :
                                      mcmc_candidates_by_batch[bi])
                new_count = Threads.atomic_add!(ind_counter, max(0, get_n_b(info))) +
                            max(0, get_n_b(info))
                ProgressMeter.update!(p, new_count)
            end
        end
    else
        ll_cache_local = ll_cache isa AbstractVector ? ll_cache[1] : ll_cache
        ind_done = 0
        for (bi, info) in enumerate(batch_infos)
            active_batches !== nothing && !(bi ∈ active_batches) && continue
            _laplace_compute_bstar_batch!(
                cache, bi, dm, info, θ_val, const_cache, ll_cache_local;
                optimizer = optimizer,
                optim_kwargs = optim_kwargs,
                adtype = adtype,
                grad_tol = grad_tol,
                multistart = multistart,
                rng = batch_rngs[bi],
                mcmc_candidates = mcmc_candidates_by_batch === nothing ? nothing :
                                  mcmc_candidates_by_batch[bi])
            ind_done += max(0, get_n_b(info))
            ProgressMeter.update!(p, ind_done)
        end
    end
    ProgressMeter.finish!(p)
    if !(eltype(θ_vec) <: ForwardDiff.Dual)
        cache.θ_cache = copy(θ_vec)
    end
    return cache.bstar_cache.b_star
end

function _laplace_hessian_b(dm::DataModel,
        batch_info::REBatchInfo,
        θ::ComponentArray,
        b,
        const_cache::REConstantsCache,
        cache::_LLCache,
        ad_cache::Union{Nothing, LaplaceADCache},
        bi::Int;
        ctx::AbstractString = "",
        tctx = nothing)
    # The θ-context is used only when the caller passes one (from a call site
    # that amortizes it over many evaluations — the inner solve, the
    # objective/gradient batch terms, Q sample loops). Building one here
    # unconditionally was measured to cost more than it saves for the common
    # single-level batch, where the Hessian re-evaluates f only a couple of
    # times.
    f = _LaplaceLogfBatch(dm, batch_info, θ, const_cache, cache, tctx)
    H = Matrix{promote_type(eltype(θ), eltype(b))}(undef, length(b), length(b))
    cfg = nothing
    if ad_cache === nothing
        cfg = ForwardDiff.HessianConfig(f, b)
    else
        entry = ad_cache.hess_cfg[bi]
        ftype = typeof(f)
        if entry === nothing
            cfg = ForwardDiff.HessianConfig(f, b)
            ad_cache.hess_cfg[bi] = _FDHessCfgStore([((ftype, eltype(b)), cfg)])
        else
            entry = entry::_FDHessCfgStore
            found = false
            for (k, c) in entry
                if k == (ftype, eltype(b))
                    cfg = c
                    found = true
                    break
                end
            end
            if !found
                cfg = ForwardDiff.HessianConfig(f, b)
                push!(entry, ((ftype, eltype(b)), cfg))
            end
        end
    end
    ForwardDiff.hessian!(H, f, b, cfg)
    return H
end

# Pluggable curvature for the inner Hessian of the log-joint - the public method-developer
# seam. A curvature returns `H = ∇²_b log f` (typically negative-definite); the downstream
# `negH = -H` / Cholesky / trace-gradient machinery is identical for every curvature.
# `ExactHessianCurvature` is the full second-order AD Hessian (Laplace default);
# `FisherInformationCurvature` (focei.jl) plugs in the Gauss-Newton / expected-information
# Hessian. Add a new curvature by implementing `inner_curvature(::YourCurvature, …)`.
"""
    AbstractCurvature

Curvature seam for marginal-likelihood methods. A curvature supplies the inner Hessian
`H = ∇²_b log f(b)` used to assemble the Laplace/FOCEI marginal; everything else (the
empirical-Bayes solve, the log-determinant, the θ-gradient, threading) is shared.
`ExactHessianCurvature` is the Laplace default; `FisherInformationCurvature` is the FOCEI/FOCE
expected-information form. Define a new marginal method by subtyping this and implementing
`inner_curvature`, then delegating `fit_method` to `fit_laplace_family`.
"""
abstract type AbstractCurvature end
const _HessMode = AbstractCurvature

"""
    ExactHessianCurvature() <: AbstractCurvature

Curvature that uses the full second-order automatic-differentiation Hessian `H = ∇²_b log f` -
the exact Laplace approximation, and the default curvature.
"""
struct ExactHessianCurvature <: AbstractCurvature end
const _ExactHess = ExactHessianCurvature

"""
    CurvatureWorkspace(; ad_cache=nothing, bi=1)

Opaque handle passed to `inner_curvature`, bundling the internal AD cache and the batch index.
It is kept off the public signature so the API does not commit to those caching
representations; a custom curvature forwards it unchanged when delegating to the built-in
`inner_curvature(ExactHessianCurvature(), …)`.
"""
struct CurvatureWorkspace{A}
    ad_cache::A
    bi::Int
end
CurvatureWorkspace(; ad_cache = nothing, bi::Int = 1) = CurvatureWorkspace(ad_cache, bi)

"""
    inner_curvature(curvature, dm, batch_info, θ, b, const_cache, cache, ws;
                    ctx="", tctx=nothing) -> Matrix

Return the inner Hessian `H = ∇²_b log f(b)` for one batch under `curvature` (negative-definite
near a mode; the posterior precision is `-H`). This is the single method a new
`AbstractCurvature` implements; `ExactHessianCurvature` and `FisherInformationCurvature` are the
reference implementations. `ws` is an opaque `CurvatureWorkspace`.
"""
@inline function inner_curvature(::ExactHessianCurvature, dm::DataModel,
        batch_info::REBatchInfo, θ, b, const_cache::REConstantsCache, cache::_LLCache,
        ws::CurvatureWorkspace; ctx::AbstractString = "", tctx = nothing)
    return _laplace_hessian_b(dm, batch_info, θ, b, const_cache, cache,
        ws.ad_cache, ws.bi; ctx = ctx, tctx = tctx)
end

# Internal shim: the fit machinery calls `_build_hess_b(mode, …, ad_cache, bi)`; route it
# through the public `inner_curvature` seam so third-party curvatures are picked up too.
@inline _build_hess_b(mode::AbstractCurvature, dm::DataModel, batch_info::REBatchInfo, θ, b,
const_cache::REConstantsCache, cache::_LLCache,
ad_cache::Union{Nothing, LaplaceADCache}, bi::Int;
ctx::AbstractString = "", tctx = nothing) = inner_curvature(
    mode, dm, batch_info, θ, b, const_cache, cache,
    CurvatureWorkspace(ad_cache, bi); ctx = ctx, tctx = tctx)

function _laplace_cholesky_negH(
        H::AbstractMatrix; jitter = 1e-6, max_tries = 6, growth = 10.0,
        adaptive = false, scale_factor = 0.0)
    n = size(H, 1)
    Hneg = Symmetric(-H)
    chol = nothing
    jit = jitter
    if adaptive
        scale = mean(abs.(diag(Hneg)))
        scale = isfinite(scale) ? scale : one(real(eltype(Hneg)))
        jit = max(jit, scale_factor * scale)
    end
    for _ in 1:max_tries
        chol = cholesky(Hneg + jit * I, check = false)
        chol.info == 0 && return (chol, jit)
        jit *= growth
    end
    return (chol, jit)
end

function _laplace_logdet_negH(dm::DataModel,
        batch_info::REBatchInfo,
        θ::ComponentArray,
        b,
        const_cache::REConstantsCache,
        cache::_LLCache,
        ad_cache::Union{Nothing, LaplaceADCache},
        bi::Int;
        jitter = 1e-6,
        max_tries = 6,
        growth = 10.0,
        adaptive = false,
        scale_factor = 0.0,
        ctx::AbstractString = "",
        hess_cache::Union{Nothing, _LaplaceHessCache} = nothing,
        use_cache::Bool = false,
        hmode::_HessMode = _ExactHess(),
        tctx = nothing)
    if get_n_b(batch_info) == 0
        T = eltype(θ)
        H = zeros(T, 0, 0)
        return (zero(T), H, nothing)
    end
    if use_cache && hess_cache !== nothing
        if hess_cache.last_valid[bi]
            b_last = hess_cache.last_b[bi]
            if length(b_last) == length(b)
                same = true
                @inbounds for i in eachindex(b_last)
                    if b_last[i] != b[i]
                        same = false
                        break
                    end
                end
                if same
                    return (hess_cache.last_logdet[bi],
                        hess_cache.last_H[bi], hess_cache.last_chol[bi])
                end
            end
        end
    end
    H = _build_hess_b(hmode, dm, batch_info, θ, b, const_cache,
        cache, ad_cache, bi; ctx = ctx, tctx = tctx)
    chol, _ = _laplace_cholesky_negH(
        H; jitter = jitter, max_tries = max_tries, growth = growth,
        adaptive = adaptive, scale_factor = scale_factor)
    infT = convert(eltype(H), Inf)
    chol === nothing && return (infT, H, nothing)
    chol.info == 0 || return (infT, H, chol)
    logdet = 2 * sum(log, diag(chol.U))
    if use_cache && hess_cache !== nothing
        b_last = hess_cache.last_b[bi]
        if length(b_last) != length(b)
            resize!(b_last, length(b))
        end
        copyto!(b_last, b)
        hess_cache.last_logdet[bi] = logdet
        hess_cache.last_H[bi] = H
        hess_cache.last_chol[bi] = chol
        hess_cache.last_valid[bi] = true
    end
    return (logdet, H, chol)
end

function _laplace_grad_batch(dm::DataModel,
        batch_info::REBatchInfo,
        θ::ComponentArray,
        b,
        const_cache::REConstantsCache,
        cache::_LLCache,
        ad_cache::Union{Nothing, LaplaceADCache},
        bi::Int;
        jitter = 1e-6,
        max_tries = 6,
        growth = 10.0,
        adaptive = false,
        scale_factor = 0.0,
        use_trace_logdet_grad::Bool = true,
        use_hutchinson::Bool = false,
        hutchinson_n::Int = 8,
        rng::AbstractRNG = Random.default_rng(),
        hmode::_HessMode = _ExactHess())
    nb = get_n_b(batch_info)
    if nb == 0
        logf = _laplace_logf_batch(dm, batch_info, θ, b, const_cache, cache)
        grad = ForwardDiff.gradient(
            θv -> _laplace_logf_batch(dm, batch_info, θv, b, const_cache, cache), θ)
        grad = grad isa ComponentArray ? grad : ComponentArray(grad, getaxes(θ))
        return (logf = logf, logdet = 0.0, grad = grad)
    end

    # θ is the (Float64) evaluation point for everything below except the
    # explicitly θ-differentiated closures (logf_θ, gθ!, fθ — those see dual θ
    # and rebuild their own context). For FOCEI with all-Gaussian REs the prior
    # Hessian Λ is constant in b, so it is computed once here and reused by the
    # b-paths (base logdet and the fb Jacobian); its ∂/∂b contribution is
    # exactly zero either way.
    tctx0 = try
        _build_theta_ctx(dm, batch_info, θ, cache)
    catch err
        if _is_numeric_error(err)
            nothing
        else
            rethrow(err)
        end
    end
    tctx = if tctx0 !== nothing && hmode isa _FOCEIHess
        try
            _ctx_with_prior_hess(tctx0, dm, batch_info, const_cache, cache, b)
        catch err
            if _is_numeric_error(err)
                tctx0
            else
                rethrow(err)
            end
        end
    else
        tctx0
    end
    logf = _laplace_logf_batch(dm, batch_info, θ, b, const_cache, cache; tctx = tctx)
    logdet, H, chol = _laplace_logdet_negH(
        dm, batch_info, θ, b, const_cache, cache, ad_cache, bi;
        jitter = jitter, max_tries = max_tries, growth = growth,
        adaptive = adaptive, scale_factor = scale_factor, ctx = "logdet", hmode = hmode, tctx = tctx)
    infT = convert(eltype(θ), Inf)
    logdet == Inf && return (
        logf = -Inf, logdet = infT, grad = ComponentArray(zeros(length(θ)), getaxes(θ)))

    axs = getaxes(θ)
    θ_vec = θ
    nθ = length(θ_vec)
    nb = length(b)
    buf = ad_cache === nothing ?
          _LaplaceGradBuffers(Vector{eltype(θ_vec)}(undef, nθ),
        Matrix{eltype(θ_vec)}(undef, nb, nθ),
        Vector{eltype(θ_vec)}(undef, nθ),
        Vector{eltype(θ_vec)}(undef, nb),
        use_trace_logdet_grad ? Matrix{eltype(θ_vec)}(undef, _ntri(nb), nθ) :
        Matrix{eltype(θ_vec)}(undef, 0, 0),
        use_trace_logdet_grad ? Matrix{eltype(θ_vec)}(undef, _ntri(nb), nb) :
        Matrix{eltype(θ_vec)}(undef, 0, 0),
        nθ, nb,
        Vector{eltype(θ_vec)}(undef, nb),
        _FDCfgStore(), _FDCfgStore(), _FDCfgStore(), _FDCfgStore(),
        _FDCfgStore(), _FDCfgStore(), _FDCfgStore()) :
          _get_grad_buffers!(ad_cache, bi, eltype(θ_vec), nθ, nb, use_trace_logdet_grad)
    # envelope term
    logf_θ = _LaplaceLogfTheta(dm, batch_info, b, const_cache, cache)
    cfg = _get_fd_cfg!(
        buf.grad_logf_cfg, logf_θ, θ_vec, () -> ForwardDiff.GradientConfig(logf_θ, θ_vec))
    ForwardDiff.gradient!(buf.grad_logf, logf_θ, θ_vec, cfg)
    grad_logf = buf.grad_logf
    # g(b, θ) = ∂ logf / ∂b
    function gθ!(out, θv)
        fb = bv -> _laplace_logf_batch(dm, batch_info, θv, bv, const_cache, cache)
        cfg = _get_fd_cfg!(buf.gθ_grad_cfg, fb, b, () -> ForwardDiff.GradientConfig(fb, b))
        ForwardDiff.gradient!(out, fb, b, cfg)
        return out
    end
    cfg = _get_fd_cfg!(
        buf.Gθ_cfg, gθ!, θ_vec, () -> ForwardDiff.JacobianConfig(gθ!, buf.gradb_tmp, θ_vec))
    ForwardDiff.jacobian!(buf.Gθ, gθ!, buf.gradb_tmp, θ_vec, cfg)
    Gθ = buf.Gθ

    # logdet(-H) gradients w.r.t θ and b (b fixed)
    if use_hutchinson
        n = nb
        Ainv = chol \ Matrix{eltype(H)}(I, n, n)
        ns = max(1, hutchinson_n)
        grad_logdet_θ = zeros(eltype(θ_vec), nθ)
        grad_logdet_b = zeros(eltype(θ_vec), nb)
        for s in 1:ns
            z = rand(rng, (-1, 1), n)
            Az = Ainv * z
            fθ = θv -> begin
                Hθ = _build_hess_b(hmode, dm, batch_info, θv, b, const_cache,
                    cache, nothing, bi; ctx = "hutch_dH_dθ")
                Hθ * Az
            end
            cfg = _get_fd_cfg!(
                buf.logdet_θ_cfg, fθ, θ_vec, () -> ForwardDiff.JacobianConfig(fθ, θ_vec))
            Jθv = ForwardDiff.jacobian(fθ, θ_vec, cfg)
            grad_logdet_θ .+= -(Jθv' * z)
            fb = bv -> begin
                Hb = _build_hess_b(hmode, dm, batch_info, θ, bv, const_cache, cache,
                    nothing, bi; ctx = "hutch_dH_db", tctx = tctx)
                Hb * Az
            end
            cfg = _get_fd_cfg!(
                buf.logdet_b_cfg, fb, b, () -> ForwardDiff.JacobianConfig(fb, b))
            Jbv = ForwardDiff.jacobian(fb, b, cfg)
            grad_logdet_b .+= -(Jbv' * z)
        end
        grad_logdet_θ ./= ns
        grad_logdet_b ./= ns
    elseif use_trace_logdet_grad
        n = nb
        Ainv = chol \ Matrix{eltype(H)}(I, n, n)
        weights = Vector{eltype(H)}(undef, _ntri(n))
        _vech_weights!(weights, Ainv)
        fθ = θv -> begin
            Hθ = _build_hess_b(hmode, dm, batch_info, θv, b, const_cache,
                cache, nothing, bi; ctx = "trace_dH_dθ")
            _vech(Hθ)
        end
        cfg = _get_fd_cfg!(
            buf.Jθ_cfg, fθ, θ_vec, () -> ForwardDiff.JacobianConfig(fθ, θ_vec))
        ForwardDiff.jacobian!(buf.Jθ, fθ, θ_vec, cfg)
        grad_logdet_θ = -(buf.Jθ' * weights)

        fb = bv -> begin
            Hb = _build_hess_b(hmode, dm, batch_info, θ, bv, const_cache, cache,
                nothing, bi; ctx = "trace_dH_db", tctx = tctx)
            _vech(Hb)
        end
        cfg = _get_fd_cfg!(buf.Jb_cfg, fb, b, () -> ForwardDiff.JacobianConfig(fb, b))
        ForwardDiff.jacobian!(buf.Jb, fb, b, cfg)
        grad_logdet_b = -(buf.Jb' * weights)
    else
        logdet_θ = _LaplaceLogdetTheta(dm,
            batch_info,
            b,
            const_cache,
            cache,
            (; jitter = jitter, max_tries = max_tries, growth = growth,
                adaptive = adaptive, scale_factor = scale_factor),
            ad_cache, bi)
        cfg = _get_fd_cfg!(buf.logdet_θ_cfg, logdet_θ, θ_vec,
            () -> ForwardDiff.GradientConfig(logdet_θ, θ_vec))
        ForwardDiff.gradient!(buf.grad_logdet_θ, logdet_θ, θ_vec, cfg)
        grad_logdet_θ = buf.grad_logdet_θ

        logdet_b = _LaplaceLogdetB(dm,
            batch_info,
            θ,
            const_cache,
            cache,
            (; jitter = jitter, max_tries = max_tries, growth = growth,
                adaptive = adaptive, scale_factor = scale_factor),
            ad_cache, bi)
        cfg = _get_fd_cfg!(
            buf.logdet_b_cfg, logdet_b, b, () -> ForwardDiff.GradientConfig(logdet_b, b))
        ForwardDiff.gradient!(buf.grad_logdet_b, logdet_b, b, cfg)
        grad_logdet_b = buf.grad_logdet_b
    end

    # db*/dθ = (-H)^{-1} * ∂g/∂θ
    #Hneg = Symmetric(-H)
    dbdθ = chol \ Gθ
    corr = vec(grad_logdet_b' * dbdθ)
    grad = grad_logf .- 0.5 .* (grad_logdet_θ .+ corr)
    return (logf = logf, logdet = logdet, grad = ComponentArray(grad, axs))
end

struct LaplaceInnerOptions{O, K, A, T}
    optimizer::O
    kwargs::K
    adtype::A
    grad_tol::T
end

struct LaplaceMultistartOptions{T}
    n::Int
    k::Int
    grad_tol::T
    max_rounds::Int
    sampling::Symbol
end

struct LaplaceHessianOptions{T}
    jitter::T
    max_tries::Int
    growth::T
    adaptive::Bool
    scale_factor::T
    use_trace_logdet_grad::Bool
    use_hutchinson::Bool
    hutchinson_n::Int
end

struct LaplaceCacheOptions{T}
    theta_tol::T
end

@inline _default_inner_grad_tol(dm::DataModel) = get_de(get_model(dm)) === nothing ? 1e-8 :
                                                 1e-2

@inline function _resolve_inner_options(inner::LaplaceInnerOptions, dm::DataModel)
    gt = inner.grad_tol
    if gt isa Symbol
        gt === :auto || error("inner_grad_tol must be numeric or :auto.")
        return LaplaceInnerOptions(
            inner.optimizer, inner.kwargs, inner.adtype, _default_inner_grad_tol(dm))
    end
    return inner
end

@inline function _resolve_multistart_options(
        multistart::LaplaceMultistartOptions, inner::LaplaceInnerOptions)
    gt = multistart.grad_tol
    sampling = _resolve_multistart_sampling(multistart.sampling, "multistart_sampling")
    if gt isa Symbol
        gt === :auto || error("multistart_grad_tol must be numeric or :auto.")
        return LaplaceMultistartOptions(
            multistart.n, multistart.k, inner.grad_tol, multistart.max_rounds, sampling)
    end
    return LaplaceMultistartOptions(
        multistart.n, multistart.k, gt, multistart.max_rounds, sampling)
end

"""
    Laplace(; optimizer, optim_kwargs, adtype, inner_options, hessian_options,
              cache_options, multistart_options, inner_optimizer, inner_kwargs,
              inner_adtype, inner_grad_tol, multistart_n, multistart_k,
              multistart_grad_tol, multistart_max_rounds, multistart_sampling,
              jitter, max_tries, jitter_growth, adaptive_jitter, jitter_scale,
              use_trace_logdet_grad, use_hutchinson, hutchinson_n, theta_tol,
              lb, ub) <: FittingMethod

Laplace approximation with Empirical Bayes Estimates (EBE) for random-effects models.
The outer optimizer maximizes the Laplace-approximated marginal likelihood over the
fixed effects, while the inner optimizer computes per-individual MAP estimates of the
random effects.

# Keyword Arguments
- `optimizer`: outer Optimization.jl optimizer. Defaults to `NLopt.LN_BOBYQA()`
  (derivative-free; requires a stopping criterion, supplied by the default `maxiters`).
- `optim_kwargs::NamedTuple = (; maxiters = 1000)`: keyword arguments for the outer `solve` call.
- `adtype`: AD backend for the outer optimizer. Defaults to `AutoForwardDiff()`.
- `inner_optimizer`: inner optimizer for computing EBE modes. Defaults to `LBFGS`.
- `inner_kwargs::NamedTuple = NamedTuple()`: keyword arguments for the inner `solve` call.
- `inner_adtype`: AD backend for the inner optimizer. Defaults to `AutoForwardDiff()`.
- `inner_grad_tol`: gradient tolerance for inner convergence (`:auto` chooses automatically).
- `multistart_n::Int = 50`: number of random starts for the inner EBE multistart.
- `multistart_k::Int = 10`: number of best starts to refine in the inner multistart.
- `multistart_grad_tol`: gradient tolerance for multistart refinement.
- `multistart_max_rounds::Int = 1`: maximum multistart refinement rounds.
- `multistart_sampling::Symbol = :lhs`: inner multistart sampling strategy (`:lhs` or `:random`).
- `jitter::Float64 = 1e-6`: initial diagonal jitter added to ensure Hessian PD.
- `max_tries::Int = 6`: maximum attempts to regularize the Hessian.
- `jitter_growth::Float64 = 10.0`: multiplicative growth factor for jitter on each retry.
- `adaptive_jitter::Bool = true`: whether to adapt jitter magnitude based on scale.
- `jitter_scale::Float64 = 1e-6`: scale for the adaptive jitter.
- `use_trace_logdet_grad::Bool = true`: use trace estimator for log-determinant gradient.
- `use_hutchinson::Bool = false`: use Hutchinson estimator instead of Cholesky for log-det.
- `hutchinson_n::Int = 8`: number of Rademacher vectors for the Hutchinson estimator.
- `theta_tol::Float64 = 0.0`: fixed-effect change tolerance for EBE caching.
- `lb`, `ub`: bounds on the transformed fixed-effect scale, or `nothing`.
"""
struct Laplace{O, K, A, IO, HO, CO, MS, L, U} <: FittingMethod
    optimizer::O
    optim_kwargs::K
    adtype::A
    inner::IO
    hessian::HO
    cache::CO
    multistart::MS
    lb::L
    ub::U
    ignore_model_bounds::Bool
    nan_recovery::Symbol   # :backtrack (default; NaN grad → non-finite objective), :fd (finite-difference gradient fallback), or :nan (propagate NaN to the optimizer)
end

function Laplace(;
        optimizer = NLopt.LN_BOBYQA(),
        optim_kwargs = (; maxiters = 1000),
        adtype = Optimization.AutoForwardDiff(),
        inner_options = nothing,
        hessian_options = nothing,
        cache_options = nothing,
        multistart_options = nothing,
        inner_optimizer = OptimizationOptimJL.LBFGS(linesearch = LineSearches.BackTracking(maxstep = 1.0)),
        inner_kwargs = NamedTuple(),
        inner_adtype = Optimization.AutoForwardDiff(),
        inner_grad_tol = :auto,
        multistart_n = 50,
        multistart_k = 10,
        multistart_grad_tol = inner_grad_tol,
        multistart_max_rounds = 1,
        multistart_sampling = :lhs,
        jitter = 1e-6,
        max_tries = 6,
        jitter_growth = 10.0,
        adaptive_jitter = true,
        jitter_scale = 1e-6,
        use_trace_logdet_grad = true,
        use_hutchinson = false,
        hutchinson_n = 8,
        theta_tol = 0.0,
        lb = nothing,
        ub = nothing,
        ignore_model_bounds = false,
        nan_recovery = :backtrack)
    inner = inner_options === nothing ?
            LaplaceInnerOptions(
        inner_optimizer, inner_kwargs, inner_adtype, inner_grad_tol) : inner_options
    hess = hessian_options === nothing ?
           LaplaceHessianOptions(
        jitter, max_tries, jitter_growth, adaptive_jitter, jitter_scale,
        use_trace_logdet_grad, use_hutchinson, hutchinson_n) : hessian_options
    cache = cache_options === nothing ? LaplaceCacheOptions(theta_tol) : cache_options
    ms = multistart_options === nothing ?
         LaplaceMultistartOptions(multistart_n, multistart_k, multistart_grad_tol,
        multistart_max_rounds, multistart_sampling) : multistart_options
    Laplace(optimizer, optim_kwargs, adtype, inner, hess, cache,
        ms, lb, ub, ignore_model_bounds, nan_recovery)
end

# FrequentistREResult is a StandardOptimizationResult{:frequentist_re} alias + constructor (see common.jl).

mutable struct _LaplaceObjCache{T, G}
    θ::Union{Nothing, AbstractVector{T}}
    obj::T
    grad::G
    has_grad::Bool
end

@inline function _laplace_obj_cache_set_obj!(cache::_LaplaceObjCache{T}, θ, obj) where {T}
    eltype(θ) === T || return nothing
    cache.θ = copy(θ)
    cache.obj = obj
    cache.has_grad = false
    return nothing
end

@inline function _laplace_obj_cache_set_obj_grad!(
        cache::_LaplaceObjCache{T}, θ, obj, grad) where {T}
    eltype(θ) === T || return nothing
    cache.θ = copy(θ)
    cache.obj = obj
    cache.grad = grad
    cache.has_grad = true
    return nothing
end

@inline function _laplace_obj_cache_lookup(
        cache::_LaplaceObjCache{T}, θ, theta_tol) where {T}
    eltype(θ) === T || return nothing
    cache.θ === nothing && return nothing
    cache.has_grad || return nothing
    maxdiff = _maxabsdiff(θ, cache.θ)
    if isfinite(maxdiff) && maxdiff <= theta_tol
        return (cache.obj, cache.grad)
    end
    return nothing
end

@inline function _laplace_obj_cache_lookup_obj(
        cache::_LaplaceObjCache{T}, θ, theta_tol) where {T}
    eltype(θ) === T || return nothing
    cache.θ === nothing && return nothing
    maxdiff = _maxabsdiff(θ, cache.θ)
    if isfinite(maxdiff) && maxdiff <= theta_tol
        return cache.obj
    end
    return nothing
end

@inline function _laplace_thread_caches(dm::DataModel, ll_cache, nthreads::Int)
    if ll_cache isa Vector
        return ll_cache
    end
    caches = if ll_cache isa _LLCache
        build_ll_cache(dm;
            ode_args = ll_cache.ode_args,
            ode_kwargs = ll_cache.ode_kwargs,
            force_saveat = ll_cache.saveat_cache !== nothing,
            nthreads = nthreads)
    else
        build_ll_cache(dm; nthreads = nthreads)
    end
    return caches isa Vector ? caches : [deepcopy(caches) for _ in 1:nthreads]
end

@inline function _laplace_thread_rngs(rng::AbstractRNG, nthreads::Int)
    return _spawn_child_rngs(rng, nthreads)
end

function _laplace_objective_and_grad(dm::DataModel,
        batch_infos::Vector{REBatchInfo},
        θ::ComponentArray,
        const_cache::REConstantsCache,
        ll_cache,
        ebe_cache::_LaplaceCache;
        inner::LaplaceInnerOptions,
        hessian::LaplaceHessianOptions,
        cache_opts::LaplaceCacheOptions,
        multistart::LaplaceMultistartOptions,
        rng::AbstractRNG,
        serialization::SciMLBase.EnsembleAlgorithm = EnsembleThreads(),
        hmode::_HessMode = _ExactHess())
    inner_opts = _resolve_inner_options(inner, dm)
    multistart_opts = _resolve_multistart_options(multistart, inner_opts)

    bstars = _laplace_get_bstar!(ebe_cache, dm, batch_infos, θ, const_cache, ll_cache;
        optimizer = inner_opts.optimizer,
        optim_kwargs = inner_opts.kwargs,
        adtype = inner_opts.adtype,
        grad_tol = inner_opts.grad_tol,
        theta_tol = cache_opts.theta_tol,
        multistart = multistart_opts,
        rng = rng,
        serialization = serialization)

    infT = convert(eltype(θ), Inf)
    grad = zeros(eltype(θ), length(θ))
    axs = getaxes(θ)
    batch_rngs = _laplace_thread_rngs(rng, length(batch_infos))
    if serialization isa SciMLBase.EnsembleThreads
        nthreads = Threads.maxthreadid()
        caches = _laplace_thread_caches(dm, ll_cache, nthreads)
        obj_by_batch = Vector{eltype(θ)}(undef, length(batch_infos))
        grad_by_batch = Matrix{eltype(θ)}(undef, length(θ), length(batch_infos))
        bad = Threads.Atomic{Bool}(false)
        # Chunk-indexed cache assignment (see _laplace_get_bstar!): each task owns cache
        # slot `c`; `caches[Threads.threadid()]` is unsafe under task migration.
        n_chunks = length(caches)
        Threads.@threads for c in 1:n_chunks
            cache_c = caches[c]
            for bi in c:n_chunks:length(batch_infos)
                bad[] && break
                info = batch_infos[bi]
                b = bstars[bi]
                res = _laplace_grad_batch(
                    dm, info, θ, b, const_cache, cache_c, ebe_cache.ad_cache, bi;
                    jitter = hessian.jitter,
                    max_tries = hessian.max_tries,
                    growth = hessian.growth,
                    adaptive = hessian.adaptive,
                    scale_factor = hessian.scale_factor,
                    use_trace_logdet_grad = hessian.use_trace_logdet_grad,
                    use_hutchinson = hessian.use_hutchinson,
                    hutchinson_n = hessian.hutchinson_n,
                    rng = batch_rngs[bi],
                    hmode = hmode)
                if res.logf == -Inf
                    bad[] = true
                    break
                end
                obj_by_batch[bi] = res.logf + 0.5 * get_n_b(info) * log(2π) -
                                   0.5 * res.logdet
                @views grad_by_batch[:, bi] .= res.grad
            end
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
        for (bi, info) in enumerate(batch_infos)
            b = bstars[bi]
            res = _laplace_grad_batch(
                dm, info, θ, b, const_cache, ll_cache, ebe_cache.ad_cache, bi;
                jitter = hessian.jitter,
                max_tries = hessian.max_tries,
                growth = hessian.growth,
                adaptive = hessian.adaptive,
                scale_factor = hessian.scale_factor,
                use_trace_logdet_grad = hessian.use_trace_logdet_grad,
                use_hutchinson = hessian.use_hutchinson,
                hutchinson_n = hessian.hutchinson_n,
                rng = batch_rngs[bi],
                hmode = hmode)
            res.logf == -Inf && return (infT, ComponentArray(grad, axs), bstars)
            total += res.logf + 0.5 * get_n_b(info) * log(2π) - 0.5 * res.logdet
            grad .+= res.grad
        end
        return (-total, ComponentArray(-grad, axs), bstars)
    end
end

function _laplace_objective_only(dm::DataModel,
        batch_infos::Vector{REBatchInfo},
        θ::ComponentArray,
        const_cache::REConstantsCache,
        ll_cache,
        ebe_cache::_LaplaceCache;
        inner::LaplaceInnerOptions,
        hessian::LaplaceHessianOptions,
        cache_opts::LaplaceCacheOptions,
        multistart::LaplaceMultistartOptions,
        rng::AbstractRNG,
        serialization::SciMLBase.EnsembleAlgorithm = EnsembleThreads(),
        hmode::_HessMode = _ExactHess())
    inner_opts = _resolve_inner_options(inner, dm)
    multistart_opts = _resolve_multistart_options(multistart, inner_opts)

    bstars = _laplace_get_bstar!(ebe_cache, dm, batch_infos, θ, const_cache, ll_cache;
        optimizer = inner_opts.optimizer,
        optim_kwargs = inner_opts.kwargs,
        adtype = inner_opts.adtype,
        grad_tol = inner_opts.grad_tol,
        theta_tol = cache_opts.theta_tol,
        multistart = multistart_opts,
        rng = rng,
        serialization = serialization)
    infT = convert(eltype(θ), Inf)
    use_cache = false
    if ebe_cache.θ_cache !== nothing && length(ebe_cache.θ_cache) == length(θ)
        maxdiff = _maxabsdiff(θ, ebe_cache.θ_cache)
        use_cache = isfinite(maxdiff) && maxdiff <= cache_opts.theta_tol
    end
    if serialization isa SciMLBase.EnsembleThreads
        nthreads = Threads.maxthreadid()
        caches = _laplace_thread_caches(dm, ll_cache, nthreads)
        obj_by_batch = Vector{eltype(θ)}(undef, length(batch_infos))
        bad = Threads.Atomic{Bool}(false)
        # Chunk-indexed cache assignment (see _laplace_get_bstar!): each task owns cache
        # slot `c`; `caches[Threads.threadid()]` is unsafe under task migration.
        n_chunks = length(caches)
        Threads.@threads for c in 1:n_chunks
            cache_c = caches[c]
            for bi in c:n_chunks:length(batch_infos)
                bad[] && break
                info = batch_infos[bi]
                b = bstars[bi]
                tctx = _objective_theta_ctx(dm, info, θ, const_cache, cache_c, b, hmode)
                logf = _laplace_logf_batch(
                    dm, info, θ, b, const_cache, cache_c; tctx = tctx)
                logf == -Inf && (bad[] = true; break)
                logdet, _, _ = _laplace_logdet_negH(
                    dm, info, θ, b, const_cache, cache_c, nothing, bi;
                    jitter = hessian.jitter,
                    max_tries = hessian.max_tries,
                    growth = hessian.growth,
                    adaptive = hessian.adaptive,
                    scale_factor = hessian.scale_factor,
                    hess_cache = ebe_cache.hess_cache,
                    use_cache = use_cache,
                    hmode = hmode,
                    tctx = tctx)
                logdet == Inf && (bad[] = true; break)
                obj_by_batch[bi] = logf + 0.5 * get_n_b(info) * log(2π) - 0.5 * logdet
            end
        end
        bad[] && return infT
        total = 0.0
        @inbounds for bi in eachindex(batch_infos)
            total += obj_by_batch[bi]
        end
    else
        # `ll_cache` may be a vector of per-thread caches even on this serial path (e.g.
        # the Wald-UQ forces a serial logdet/Hessian for reproducibility while keeping the
        # EB solve threaded via the vector cache). Use a single cache slot here.
        ll_cache_use = ll_cache isa AbstractVector ? ll_cache[1] : ll_cache
        total = 0.0
        for (bi, info) in enumerate(batch_infos)
            b = bstars[bi]
            tctx = _objective_theta_ctx(dm, info, θ, const_cache, ll_cache_use, b, hmode)
            logf = _laplace_logf_batch(
                dm, info, θ, b, const_cache, ll_cache_use; tctx = tctx)
            logf == -Inf && return infT
            logdet, _, _ = _laplace_logdet_negH(
                dm, info, θ, b, const_cache, ll_cache_use, ebe_cache.ad_cache, bi;
                jitter = hessian.jitter,
                max_tries = hessian.max_tries,
                growth = hessian.growth,
                adaptive = hessian.adaptive,
                scale_factor = hessian.scale_factor,
                hess_cache = ebe_cache.hess_cache,
                use_cache = use_cache,
                hmode = hmode,
                tctx = tctx)
            logdet == Inf && return infT
            total += logf + 0.5 * get_n_b(info) * log(2π) - 0.5 * logdet
        end
    end
    return -total
end

function _fit_model(dm::DataModel, method::Laplace, args...;
        constants::NamedTuple = NamedTuple(),
        constants_re::NamedTuple = NamedTuple(),
        penalty::NamedTuple = NamedTuple(),
        extra_objective = nothing,
        ode_args::Tuple = (),
        ode_kwargs::NamedTuple = NamedTuple(),
        serialization::SciMLBase.EnsembleAlgorithm = EnsembleThreads(),
        rng::AbstractRNG = Random.default_rng(),
        theta_0_untransformed::Union{Nothing, ComponentArray} = nothing,
        store_data_model::Bool = true)
    fit_kwargs = (constants = constants,
        constants_re = constants_re,
        penalty = penalty,
        ode_args = ode_args,
        ode_kwargs = ode_kwargs,
        serialization = serialization,
        rng = rng,
        theta_0_untransformed = theta_0_untransformed,
        store_data_model = store_data_model)
    re_names = get_re_names(get_random(get_model(dm)))
    isempty(re_names) &&
        error("Laplace requires random effects. Use MLE/MAP for fixed-effects models.")
    fe = get_fixed(get_model(dm))
    fixed_names = get_names(fe)
    isempty(fixed_names) && error("Laplace requires at least one fixed effect.")
    fixed_set = Set(fixed_names)
    for name in keys(constants)
        name in fixed_set || error("Unknown constant parameter $(name).")
    end
    all(name in keys(constants) for name in fixed_names) &&
        error("Laplace requires at least one free fixed effect. Remove constants or specify a fixed effect or random effect.")

    return _fit_laplace_family(dm, method, _ExactHess(), args, fit_kwargs,
        _ -> nothing;
        nan_recovery = method.nan_recovery, allow_bbo = true,
        constants = constants, constants_re = constants_re, penalty = penalty,
        ode_args = ode_args, ode_kwargs = ode_kwargs, serialization = serialization,
        rng = rng, theta_0_untransformed = theta_0_untransformed,
        store_data_model = store_data_model, extra_objective = extra_objective)
end

# Shared optimizer driver for the Laplace family: exact-Hessian Laplace and
# FOCEI/FOCE run the same marginal-likelihood machinery and differ only in the
# inner Hessian mode (`hmode`), the gradient NaN-recovery policy, BlackBoxOptim
# support, and an optional post-transform validation hook (FOCEI's closed-form
# Fisher-information outcome check). Method-specific validation and error
# messages stay in the `_fit_model` wrappers.
function _fit_laplace_family(dm::DataModel, method, hmode::_HessMode, args, fit_kwargs,
        validate_post_transform::V;
        nan_recovery::Symbol,
        allow_bbo::Bool,
        constants::NamedTuple, constants_re::NamedTuple, penalty::NamedTuple,
        ode_args::Tuple, ode_kwargs::NamedTuple,
        serialization::SciMLBase.EnsembleAlgorithm, rng::AbstractRNG,
        theta_0_untransformed::Union{Nothing, ComponentArray},
        store_data_model::Bool, extra_objective = nothing) where {V}
    fe = get_fixed(get_model(dm))
    layout = free_parameter_layout(fe; constants = constants,
        theta0_untransformed = theta_0_untransformed)
    free_names = layout.free_names
    inv_transform = layout.inv_transform

    validate_post_transform(dm)

    inner_opts = _resolve_inner_options(method.inner, dm)
    multistart_opts = _resolve_multistart_options(method.multistart, inner_opts)

    _, batch_infos, const_cache = _build_re_batch_infos(dm, constants_re)
    ll_cache = build_ll_cache(dm; ode_args = ode_args, ode_kwargs = ode_kwargs,
        serialization = serialization, force_saveat = true)
    n_batches = length(batch_infos)
    Tθ = eltype(layout.θ0_free_t)
    ebe_cache = _init_laplace_eval_cache(n_batches, Tθ)

    θ0_free_t = layout.θ0_free_t
    axs_free = layout.axs
    axs_full = layout.axs_full
    θ_const_t_vec = layout.θ_const_t_vec
    free_idx = layout.free_idx
    has_penalty = !isempty(keys(penalty))
    # Optional user-supplied objective term, a function of the natural-scale θu.
    # Mirrors `penalty` exactly: added to both obj_only and obj_grad, no-op when
    # `extra_objective === nothing` (default) so existing fits are unaffected.
    has_extra = extra_objective !== nothing
    T0 = eltype(θ0_free_t)
    obj_cache = _LaplaceObjCache{T0, ComponentArray}(nothing,
        T0(Inf),
        ComponentArray(zeros(T0, length(θ0_free_t)), axs_free),
        false)

    function obj_only(θt, p)
        θt_free = θt isa ComponentArray ? θt : ComponentArray(θt, axs_free)
        cached_obj = _laplace_obj_cache_lookup_obj(
            obj_cache, θt_free, method.cache.theta_tol)
        cached_obj !== nothing && return cached_obj
        T = eltype(θt_free)
        infT = convert(T, Inf)
        θt_full = _merge_free_into_full(θ_const_t_vec, free_idx, θt_free, axs_full)
        θu = inv_transform(θt_full)
        obj = _laplace_objective_only(
            dm, batch_infos, θu, const_cache, ll_cache, ebe_cache;
            inner = inner_opts,
            hessian = method.hessian,
            cache_opts = method.cache,
            multistart = multistart_opts,
            rng = rng,
            serialization = serialization,
            hmode = hmode)
        !isfinite(obj) && return infT
        has_penalty && (obj += _penalty_value(θu, penalty))
        has_extra && (obj += extra_objective(θu))
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
        θt_full = _merge_free_into_full(θ_const_t_vec, free_idx, θt_free, axs_full)
        θu = inv_transform(θt_full)
        obj, grad_full, bstars = _laplace_objective_and_grad(
            dm, batch_infos, θu, const_cache, ll_cache, ebe_cache;
            inner = inner_opts,
            hessian = method.hessian,
            cache_opts = method.cache,
            multistart = multistart_opts,
            rng = rng,
            serialization = serialization,
            hmode = hmode)
        !isfinite(obj) && return (infT, ComponentArray(zeros(T, length(θt_free)), axs_free))
        grad_u = grad_full
        if has_penalty
            # The objective includes the penalty, so the analytic gradient must too
            # (mirrors focei.jl; the :fd recovery below already differentiates
            # `obj_only`, which is penalized).
            obj += _penalty_value(θu, penalty)
            grad_u = grad_u .+ ForwardDiff.gradient(x -> _penalty_value(x, penalty), θu)
        end
        if has_extra
            # Same treatment as the penalty: value into both objectives, gradient
            # into the analytic gradient (the :fd recovery differentiates obj_only,
            # which already includes the extra term).
            obj += extra_objective(θu)
            grad_u = grad_u .+ ForwardDiff.gradient(extra_objective, θu)
        end
        grad_t_ca = apply_inv_jacobian_T(inv_transform, θt_full, grad_u)
        grad_free = similar(θt_free)
        for name in free_names
            setproperty!(grad_free, name, getproperty(grad_t_ca, name))
        end
        if any(isnan, grad_free)
            if nan_recovery === :fd
                for i in eachindex(grad_free)
                    ε = max(1e-5, 1e-5 * abs(θt_free[i]))
                    θp = copy(θt_free)
                    θp[i] += ε
                    θm = copy(θt_free)
                    θm[i] -= ε
                    fp = obj_only(θp, nothing)
                    fm = obj_only(θm, nothing)
                    grad_free[i] = (isfinite(fp) && isfinite(fm)) ? (fp - fm) / (2ε) :
                                   zero(T)
                end
            elseif nan_recovery !== :nan
                # :backtrack (default) — treat NaN gradient as non-finite objective to force backtracking
                return (infT, ComponentArray(zeros(T, length(θt_free)), axs_free))
            end
            # :nan — no recovery; the NaN gradient is passed to the optimizer unchanged
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
    lb, ub, use_bounds, θ0_init = _resolve_optim_bounds(
        fe, free_names, θ0_free_t, method.optimizer, method.lb, method.ub, constants;
        ignore_model_bounds = method.ignore_model_bounds, allow_bbo = allow_bbo,
        method_label = "Laplace")
    prob = use_bounds ? OptimizationProblem(optf, θ0_init; lb = lb, ub = ub) :
           OptimizationProblem(optf, θ0_init)
    sol = Optimization.solve(prob, method.optimizer; method.optim_kwargs...)

    summary = FitSummary(sol.objective, sol.retcode == SciMLBase.ReturnCode.Success,
        resolve_fitted_parameters(layout, sol.u), NamedTuple())
    diagnostics = FitDiagnostics(
        (;), (optimizer = method.optimizer,), (retcode = sol.retcode,), NamedTuple())
    niter = hasproperty(sol, :stats) && hasproperty(sol.stats, :iterations) ?
            sol.stats.iterations : missing
    raw = hasproperty(sol, :original) ? sol.original : sol
    result = FrequentistREResult(
        sol, sol.objective, niter, raw, NamedTuple(), ebe_cache.bstar_cache.b_star)
    return FitResult(method, result, summary, diagnostics,
        store_data_model ? dm : nothing, args, fit_kwargs)
end

@inline _laplace_value(x) = x
@inline _laplace_value(x::ForwardDiff.Dual) = ForwardDiff.value(x)

function _laplace_floatize(θ::ComponentArray)
    eltype(θ) === Float64 && return θ
    vals = map(_laplace_value, θ)
    return ComponentArray(Float64.(vals), getaxes(θ))
end

function _with_eb_modes(result::FrequentistREResult, eb_modes)
    return FrequentistREResult(result.solution, result.objective, result.iterations,
        result.raw, result.notes, eb_modes)
end
