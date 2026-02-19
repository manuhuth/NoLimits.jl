export Laplace
export LaplaceResult
export LaplaceMAP
export LaplaceMAPResult

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
using StaticArrays
using ProgressMeter


struct LaplaceConstantsCache{M, S, V}
    is_const::M
    scalar_vals::S
    vector_vals::V
end

struct LaplaceADCache{G, H, O, B}
    grad_cfg::G
    hess_cfg::H
    optf::O
    buffers::B
end

struct _LaplaceLogfBatch{DM, INFO, TH, CC, CA}
    dm::DM
    info::INFO
    θ::TH
    const_cache::CC
    cache::CA
end

@inline function (f::_LaplaceLogfBatch)(b)
    return _laplace_logf_batch(f.dm, f.info, f.θ, b, f.const_cache, f.cache)
end

struct _LaplaceLogfBatchParam{DM, INFO, CC, CA}
    dm::DM
    info::INFO
    const_cache::CC
    cache::CA
end

@inline function (f::_LaplaceLogfBatchParam)(b, θ)
    return -_laplace_logf_batch(f.dm, f.info, θ, b, f.const_cache, f.cache)
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
    return _laplace_logdet_negH(f.dm, f.info, θ, f.b, f.const_cache, f.cache, f.ad_cache, f.bi;
                                jitter=f.hess.jitter,
                                max_tries=f.hess.max_tries,
                                growth=f.hess.growth,
                                adaptive=f.hess.adaptive,
                                scale_factor=f.hess.scale_factor,
                                ctx="grad_logdet_θ")[1]
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
    return _laplace_logdet_negH(f.dm, f.info, f.θ, b, f.const_cache, f.cache, f.ad_cache, f.bi;
                                jitter=f.hess.jitter,
                                max_tries=f.hess.max_tries,
                                growth=f.hess.growth,
                                adaptive=f.hess.adaptive,
                                scale_factor=f.hess.scale_factor,
                                ctx="grad_logdet_b")[1]
end

function _normalize_constants_re(dm::DataModel, constants_re::NamedTuple)
    isempty(constants_re) && return NamedTuple()
    re_names = get_re_names(dm.model.random.random)
    isempty(re_names) && return NamedTuple()
    values = dm.re_group_info.values
    pairs = Pair{Symbol, Any}[]
    for re in re_names
        haskey(constants_re, re) || continue
        spec = getfield(constants_re, re)
        if spec isa NamedTuple
            spec = spec
        elseif spec isa AbstractDict
            spec = (; spec...)
        elseif spec isa Base.Iterators.Pairs
            spec = NamedTuple(spec)
        elseif spec isa Pair
            spec = NamedTuple((spec,))
        else
            error("constants_re for $(re) must be a NamedTuple of level => value.")
        end
        vals = getfield(values, re)
        dict = Dict{Any, Any}()
        col = getfield(get_re_groups(dm.model.random.random), re)
        for (k, v) in Base.pairs(spec)
            matched = false
            for gv in vals
                if gv == k ||
                   (gv isa AbstractString && k isa Symbol && Symbol(gv) == k) ||
                   (gv isa Symbol && k isa AbstractString && Symbol(k) == gv)
                    dict[gv] = v
                    matched = true
                    break
                end
            end
            matched || error("constants_re for $(re) includes level $(k) not found in column $(col). The value must be present in that column.")
        end
        push!(pairs, re => dict)
    end
    return NamedTuple(pairs)
end

function _build_constants_cache(dm::DataModel, constants_re::NamedTuple)
    cache = dm.re_group_info.laplace_cache
    cache === nothing && return LaplaceConstantsCache(BitVector[], Vector{Vector{Float64}}(), Vector{Vector{Vector{Float64}}}())
    re_names = cache.re_names
    nre = length(re_names)
    is_const = Vector{BitVector}(undef, nre)
    scalar_vals = Vector{Vector{Float64}}(undef, nre)
    vector_vals = Vector{Vector{Vector{Float64}}}(undef, nre)
    for (ri, re) in enumerate(re_names)
        levels = cache.re_index[ri].levels
        is_const[ri] = falses(length(levels))
        if cache.is_scalar[ri] || cache.dims[ri] == 1
            scalar_vals[ri] = Vector{Float64}(undef, length(levels))
            vector_vals[ri] = Vector{Vector{Float64}}(undef, 0)
        else
            scalar_vals[ri] = Float64[]
            vector_vals[ri] = Vector{Vector{Float64}}(undef, length(levels))
        end
    end
    for (ri, re) in enumerate(re_names)
        haskey(constants_re, re) || continue
        cmap = getfield(constants_re, re)
        idx_map = cache.re_index[ri].level_to_index
        for (k, v) in pairs(cmap)
            idx = get(idx_map, k, 0)
            idx == 0 && error("constants_re for $(re) includes level $(k) not found in column $(getfield(get_re_groups(dm.model.random.random), re)). The value must be present in that column.")
            is_const[ri][idx] = true
            if cache.is_scalar[ri] || cache.dims[ri] == 1
                v isa Number || error("constants_re for $(re) level $(k) must be a scalar number.")
                scalar_vals[ri][idx] = Float64(v)
            else
                v isa AbstractVector || error("constants_re for $(re) level $(k) must be a vector.")
                length(v) == cache.dims[ri] || error("constants_re for $(re) level $(k) must have length $(cache.dims[ri]).")
                vector_vals[ri][idx] = Float64.(v)
            end
        end
    end
    return LaplaceConstantsCache(is_const, scalar_vals, vector_vals)
end

function _build_laplace_batches(dm::DataModel, const_cache::LaplaceConstantsCache)
    cache = dm.re_group_info.laplace_cache
    cache === nothing && return dm.pairing
    re_names = cache.re_names
    isempty(re_names) && return dm.pairing
    n = length(dm.individuals)
    n == 0 && return PairingInfo(Int[], Vector{Vector{Int}}())
    uf = UnionFind(n)
    for ri in eachindex(re_names)
        const_mask = const_cache.is_const[ri]
        seen = zeros(Int, length(const_mask))
        for i in 1:n
            ids = cache.ind_level_ids[i][ri]
            for id in ids
                const_mask[id] && continue
                first = seen[id]
                if first != 0
                    _uf_union!(uf, i, first)
                else
                    seen[id] = i
                end
            end
        end
    end
    batch_ids = [_uf_find(uf, i) for i in 1:n]
    batches_dict = Dict{Int, Vector{Int}}()
    for i in 1:n
        push!(get!(batches_dict, batch_ids[i], Int[]), i)
    end
    batches = collect(values(batches_dict))
    sort!(batches, by = b -> length(b))
    return PairingInfo(batch_ids, batches)
end


struct _LaplaceREMap{L, M}
    levels::L
    level_to_index::M
end

struct _LaplaceREInfo{A, R, P}
    map::A
    ranges::R
    reps::P
    dim::Int
    is_scalar::Bool
end

struct _LaplaceBatchInfo{B, R}
    inds::B
    re_info::R
    n_b::Int
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
    grad_logf_cfg::Vector{Any}
    Gθ_cfg::Vector{Any}
    gθ_grad_cfg::Vector{Any}
    logdet_θ_cfg::Vector{Any}
    logdet_b_cfg::Vector{Any}
    Jθ_cfg::Vector{Any}
    Jb_cfg::Vector{Any}
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

_LaplaceCache(θ_cache::AbstractVector{T},
              bstar_cache::_LaplaceBStarCache{T, B},
              grad_cache::_LaplaceGradCache{T, L, G, V},
              ad_cache::A,
              hess_cache::Union{Nothing, _LaplaceHessCache}) where {T, B, L, G, V, A} =
    _LaplaceCache{T, typeof(bstar_cache), typeof(grad_cache), typeof(ad_cache), typeof(hess_cache)}(θ_cache, bstar_cache, grad_cache, ad_cache, hess_cache)

_LaplaceCache(θ_cache::Nothing,
              bstar_cache::_LaplaceBStarCache{T, B},
              grad_cache::_LaplaceGradCache{T, L, G, V},
              ad_cache::A,
              hess_cache::Union{Nothing, _LaplaceHessCache}) where {T, B, L, G, V, A} =
    _LaplaceCache{T, typeof(bstar_cache), typeof(grad_cache), typeof(ad_cache), typeof(hess_cache)}(θ_cache, bstar_cache, grad_cache, ad_cache, hess_cache)

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

function _get_fd_cfg!(store::Vector{Any}, f, x, builder::Function)
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

function _get_grad_buffers!(ad_cache::LaplaceADCache, bi::Int, T::Type, nθ::Int, nb::Int, use_trace::Bool)
    entry = ad_cache.buffers[bi]
    if entry === nothing || !(entry isa _LaplaceGradBuffers) || entry.nθ != nθ || entry.nb != nb || !(eltype(entry.grad_logf) === T)
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
                                    Any[], Any[], Any[], Any[], Any[], Any[], Any[])
        ad_cache.buffers[bi] = entry
    elseif use_trace && (size(entry.Jθ, 1) == 0 || size(entry.Jb, 1) == 0)
        ntri = _ntri(nb)
        entry.Jθ = Matrix{T}(undef, ntri, nθ)
        entry.Jb = Matrix{T}(undef, ntri, nb)
    end
    return entry
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

function _build_laplace_batch_infos(dm::DataModel, constants_re::NamedTuple)
    constants_re = _normalize_constants_re(dm, constants_re)
    const_cache = _build_constants_cache(dm, constants_re)
    pairing = _build_laplace_batches(dm, const_cache)
    cache = dm.re_group_info.laplace_cache
    cache === nothing && return pairing, _LaplaceBatchInfo[], const_cache
    re_names = cache.re_names
    isempty(re_names) && return pairing, _LaplaceBatchInfo[], const_cache
    batch_infos = Vector{_LaplaceBatchInfo}(undef, length(pairing.batches))
    for (bi, inds) in enumerate(pairing.batches)
        total_dim = 0
        re_info = Vector{_LaplaceREInfo}(undef, length(re_names))
        for (ri, re) in enumerate(re_names)
            nlevels = length(cache.re_index[ri].levels)
            level_to_index = zeros(Int, nlevels)
            levels = Int[]
            reps = Int[]
            for i in inds
                ids = cache.ind_level_ids[i][ri]
                for id in ids
                    const_cache.is_const[ri][id] && continue
                    if level_to_index[id] == 0
                        push!(levels, id)
                        push!(reps, i)
                        level_to_index[id] = length(levels)
                    end
                end
            end
            ranges = Vector{UnitRange{Int}}(undef, length(levels))
            dim = cache.dims[ri]
            is_scalar = cache.is_scalar[ri]
            for li in eachindex(levels)
                ranges[li] = total_dim + 1:total_dim + dim
                total_dim += dim
            end
            map = _LaplaceREMap(levels, level_to_index)
            re_info[ri] = _LaplaceREInfo(map, ranges, reps, dim, is_scalar)
        end
        batch_infos[bi] = _LaplaceBatchInfo(inds, re_info, total_dim)
    end
    return pairing, batch_infos, const_cache
end


@inline function _re_value_from_b(info::_LaplaceREInfo, level_id::Int, b)
    idx = info.map.level_to_index[level_id]
    idx == 0 && return nothing
    r = info.ranges[idx]
    if info.is_scalar || length(r) == 1
        return b[first(r)]
    else
        return view(b, r)
    end
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
    dist isa Distributions.AbstractMvNormal || return nothing
    μ = Distributions.mean(dist)
    Σ = Distributions.cov(dist)
    return Normal(μ[i], sqrt(Σ[i, i]))
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

function _laplace_default_b0(dm::DataModel,
                             batch_info::_LaplaceBatchInfo,
                             θ::ComponentArray,
                             const_cache::LaplaceConstantsCache,
                             cache::_LLCache)
    nb = batch_info.n_b
    nb == 0 && return Float64[]
    T = eltype(θ)
    b0 = zeros(T, nb)
    model_funs = cache.model_funs
    helpers = cache.helpers
    dists_builder = get_create_random_effect_distribution(dm.model.random.random)
    θ_re = _symmetrize_psd_params(θ, dm.model.fixed.fixed)
    cache = dm.re_group_info.laplace_cache
    re_names = cache.re_names
    for (ri, re) in enumerate(re_names)
        info = batch_info.re_info[ri]
        isempty(info.map.levels) && continue
        for (li, _) in enumerate(info.map.levels)
            rep_idx = info.reps[li]
            const_cov = dm.individuals[rep_idx].const_cov
            dists = dists_builder(θ_re, const_cov, model_funs, helpers)
            dist = getproperty(dists, re)
            start = _re_start_value(dist, info.dim, T)
            r = info.ranges[li]
            if info.is_scalar || info.dim == 1
                b0[first(r)] = start
            else
                b0[r] .= start
            end
        end
    end
    return b0
end

function _laplace_sample_b0s(dm::DataModel,
                             batch_info::_LaplaceBatchInfo,
                             θ::ComponentArray,
                             const_cache::LaplaceConstantsCache,
                             cache::_LLCache,
                             rng::AbstractRNG,
                             n::Int,
                             sampling::Symbol)
    n <= 0 && return Vector{Vector{eltype(θ)}}()
    _resolve_multistart_sampling(sampling, "inner multistart sampling")
    T = eltype(θ)
    nb = batch_info.n_b
    nb == 0 && return [T[] for _ in 1:n]
    b0s = [zeros(T, nb) for _ in 1:n]
    model_funs = cache.model_funs
    helpers = cache.helpers
    dists_builder = get_create_random_effect_distribution(dm.model.random.random)
    θ_re = _symmetrize_psd_params(θ, dm.model.fixed.fixed)
    cache = dm.re_group_info.laplace_cache
    re_names = cache.re_names
    for (ri, re) in enumerate(re_names)
        info = batch_info.re_info[ri]
        isempty(info.map.levels) && continue
        for (li, _) in enumerate(info.map.levels)
            rep_idx = info.reps[li]
            const_cov = dm.individuals[rep_idx].const_cov
            dists = dists_builder(θ_re, const_cov, model_funs, helpers)
            dist = getproperty(dists, re)
            r = info.ranges[li]
            dim = length(r)
            if info.is_scalar || info.dim == 1
                draws = sampling === :lhs ? _laplace_lhs_draws_univariate(dist, n, rng) : nothing
                draws === nothing && (draws = [rand(rng, dist) for _ in 1:n])
                for i in 1:n
                    v = draws[i]
                    if v isa AbstractVector
                        length(v) == 1 || error("Expected scalar random effect for $(re), got vector of length $(length(v)).")
                        v = v[1]
                    end
                    b0s[i][first(r)] = T(v)
                end
            else
                draws = (sampling === :lhs && dist isa Distributions.AbstractMvNormal) ?
                        _laplace_lhs_draws_mvnormal(dist, n, rng, dim) : nothing
                draws === nothing && (draws = [rand(rng, dist) for _ in 1:n])
                for i in 1:n
                    v = draws[i]
                    if v isa Number
                        b0s[i][r] .= T(v)
                    else
                        vv = vec(v)
                        length(vv) == dim || error("Expected random effect draw for $(re) with length $(dim), got $(length(vv)).")
                        b0s[i][r] .= T.(vv)
                    end
                end
            end
        end
    end
    return b0s
end

function _laplace_sample_b0(dm::DataModel,
                            batch_info::_LaplaceBatchInfo,
                            θ::ComponentArray,
                            const_cache::LaplaceConstantsCache,
                            cache::_LLCache,
                            rng::AbstractRNG)
    draws = _laplace_sample_b0s(dm, batch_info, θ, const_cache, cache, rng, 1, :random)
    return isempty(draws) ? eltype(θ)[] : draws[1]
end

@inline function _laplace_gradb_cached!(cache::_LaplaceCache,
                                       bi::Int,
                                       dm::DataModel,
                                       info::_LaplaceBatchInfo,
                                       θ::ComponentArray,
                                       const_cache::LaplaceConstantsCache,
                                       ll_cache::_LLCache,
                                       b)
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
    f = _LaplaceLogfBatch(dm, info, θ, const_cache, ll_cache)
    logf = f(b)
    entry = cache.ad_cache.grad_cfg[bi]
    if entry === nothing
        entry = Any[]
        cache.ad_cache.grad_cfg[bi] = entry
    end
    entry = entry::Vector{Any}
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

function _build_eta_ind(dm::DataModel,
                        ind_idx::Int,
                        batch_info::_LaplaceBatchInfo,
                        b,
                        const_cache::LaplaceConstantsCache,
                        θ::ComponentArray)
    cache = dm.re_group_info.laplace_cache
    re_names = cache.re_names
    pairs = Pair{Symbol, Any}[]
    T = eltype(b)
    for (ri, re) in enumerate(re_names)
        info = batch_info.re_info[ri]
        ids = cache.ind_level_ids[ind_idx][ri]
        const_mask = const_cache.is_const[ri]
        const_scalars = const_cache.scalar_vals[ri]
        const_vectors = const_cache.vector_vals[ri]
        if length(ids) == 1
            id = ids[1]
            if const_mask[id]
                if info.is_scalar || info.dim == 1
                    v = const_scalars[id]
                    push!(pairs, re => T(v))
                else
                    v = const_vectors[id]
                    push!(pairs, re => T.(v))
                end
            else
                v = _re_value_from_b(info, id, b)
                v === nothing && error("Missing random effect value for $(re) level $(cache.re_index[ri].levels[id]).")
                if info.is_scalar || info.dim == 1
                    push!(pairs, re => T(v))
                else
                    push!(pairs, re => Vector{T}(v))
                end
            end
        else
            if info.is_scalar || info.dim == 1
                vals = Vector{T}(undef, length(ids))
            else
                vals = Vector{Vector{T}}(undef, length(ids))
            end
            for (gi, id) in pairs(ids)
                if const_mask[id]
                    if info.is_scalar || info.dim == 1
                        v = const_scalars[id]
                        vals[gi] = T(v)
                    else
                        v = const_vectors[id]
                        vals[gi] = T.(v)
                    end
                else
                    v = _re_value_from_b(info, id, b)
                    v === nothing && error("Missing random effect value for $(re) level $(cache.re_index[ri].levels[id]).")
                    if info.is_scalar || info.dim == 1
                        vals[gi] = T(v)
                    else
                        vals[gi] = Vector{T}(v)
                    end
                end
            end
            push!(pairs, re => vals)
        end
    end
    nt = NamedTuple(pairs)
    return ComponentArray(nt)
end

function _laplace_logf_batch(dm::DataModel,
                             batch_info::_LaplaceBatchInfo,
                             θ::ComponentArray,
                             b,
                             const_cache::LaplaceConstantsCache,
                             cache::_LLCache)
    ll = 0.0
    model_funs = cache.model_funs
    helpers = cache.helpers
    θ_re = _symmetrize_psd_params(θ, dm.model.fixed.fixed)
    dists_builder = get_create_random_effect_distribution(dm.model.random.random)
    # random-effects prior term (free levels only)
    re_cache = dm.re_group_info.laplace_cache
    re_names = re_cache.re_names
    for (ri, re) in enumerate(re_names)
        info = batch_info.re_info[ri]
        isempty(info.map.levels) && continue
        for (li, level_id) in enumerate(info.map.levels)
            rep_idx = info.reps[li]
            const_cov = dm.individuals[rep_idx].const_cov
            dists = dists_builder(θ_re, const_cov, model_funs, helpers)
            dist = getproperty(dists, re)
            v = _re_value_from_b(info, level_id, b)
            v === nothing && continue
            lp = logpdf(dist, v)
            isfinite(lp) || return -Inf
            ll += lp
        end
    end
    # likelihood term
    for i in batch_info.inds
        η_ind = _build_eta_ind(dm, i, batch_info, b, const_cache, θ_re)
        lli = _loglikelihood_individual(dm, i, θ_re, η_ind, cache)
        lli == -Inf && return -Inf
        ll += lli
    end
    return ll
end

function _laplace_solve_batch!(dm::DataModel,
                               batch_info::_LaplaceBatchInfo,
                               θ::ComponentArray,
                               const_cache::LaplaceConstantsCache,
                               cache::_LLCache,
                               ad_cache::LaplaceADCache,
                               bi::Int,
                               b0;
                               optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
                               optim_kwargs::NamedTuple=NamedTuple(),
                               adtype=Optimization.AutoForwardDiff())
    nb = batch_info.n_b
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
    opt_entry = ad_cache.optf[bi]
    if opt_entry === nothing || opt_entry[1] != typeof(adtype)
        f = _LaplaceLogfBatchParam(dm, batch_info, const_cache, cache)
        optf = OptimizationFunction(f, adtype)
        opt_entry = (typeof(adtype), optf)
        ad_cache.optf[bi] = opt_entry
    end
    optf = opt_entry[2]
    prob = OptimizationProblem(optf, b0_use, θ_val)
    sol = solve(prob, optimizer; optim_kwargs...)
    # Inner EBE optimization failures are handled by caller fallback logic.
    SciMLBase.successful_retcode(sol) || return nothing
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
                                       info::_LaplaceBatchInfo,
                                       θ_val::ComponentArray,
                                       const_cache::LaplaceConstantsCache,
                                       ll_cache::_LLCache;
                                       optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
                                       optim_kwargs::NamedTuple=NamedTuple(),
                                       adtype=Optimization.AutoForwardDiff(),
                                       grad_tol=1e-6,
                                       multistart=LaplaceMultistartOptions(0, 0, grad_tol, 5, :lhs),
                                       rng::AbstractRNG=Random.default_rng())
    nb = info.n_b
    if nb == 0
        b_slot = cache.bstar_cache.b_star[bi]
        if !isempty(b_slot)
            resize!(b_slot, 0)
        end
        cache.bstar_cache.has_bstar[bi] = true
        return nothing
    end
    b0_default = _laplace_default_b0(dm, info, θ_val, const_cache, ll_cache)
    b_start = b0_default
    if cache.bstar_cache.has_bstar[bi]
        b_prev = cache.bstar_cache.b_star[bi]
        if length(b_prev) == nb
            f_prev = _laplace_logf_batch(dm, info, θ_val, b_prev, const_cache, ll_cache)
            f_def = _laplace_logf_batch(dm, info, θ_val, b0_default, const_cache, ll_cache)
            if f_prev > f_def
                b_start = b_prev
            end
        end
    end
    g0, f_start = _laplace_gradb_cached!(cache, bi, dm, info, θ_val, const_cache, ll_cache, b_start)
    if !isfinite(f_start)
        best_f = -Inf
        best_b = b_start
        n_try = multistart.n > 0 ? multistart.n : 0
        b_tries = _laplace_sample_b0s(dm, info, θ_val, const_cache, ll_cache, rng, n_try, multistart.sampling)
        for b_try in b_tries
            f_try = _laplace_logf_batch(dm, info, θ_val, b_try, const_cache, ll_cache)
            if f_try > best_f
                best_f = f_try
                best_b = b_try
            end
        end
        if isfinite(best_f)
            b_start = best_b
        end
        g0, _ = _laplace_gradb_cached!(cache, bi, dm, info, θ_val, const_cache, ll_cache, b_start)
    end
    g0_norm = maximum(abs, g0)
    if !isfinite(g0_norm)
        g0_norm = Inf
    end
    b_best = b_start
    best_f = isfinite(f_start) ? f_start : -Inf
    g_best_norm = g0_norm
    sol_best = try
        _laplace_solve_batch!(dm, info, θ_val, const_cache, ll_cache, cache.ad_cache, bi, b_start;
                              optimizer=optimizer,
                              optim_kwargs=optim_kwargs,
                              adtype=adtype)
    catch err
        if err isa DomainError || err isa ArgumentError
            nothing
        else
            rethrow(err)
        end
    end
    if sol_best !== nothing
        b_best = sol_best.u
        best_f = _laplace_sol_logf(sol_best)
        if !isfinite(best_f)
            best_f = _laplace_logf_batch(dm, info, θ_val, b_best, const_cache, ll_cache)
            isfinite(best_f) || (best_f = -Inf)
        end
        g_best_norm = _laplace_sol_grad_norm(sol_best)
        if !isfinite(g_best_norm)
            g_best, _ = _laplace_gradb_cached!(cache, bi, dm, info, θ_val, const_cache, ll_cache, b_best)
            g_best_norm = maximum(abs, g_best)
        end
        if !isfinite(g_best_norm)
            g_best_norm = Inf
        end
    else
        _laplace_store_bstar!(cache, bi, b_best)
        return nothing
    end
    if g_best_norm > multistart.grad_tol && multistart.n > 0 && multistart.k > 0
        n = multistart.n
        k = multistart.k
        max_rounds = max(1, multistart.max_rounds)
        for round in 1:max_rounds
            k = min(k, n)
            b0s = _laplace_sample_b0s(dm, info, θ_val, const_cache, ll_cache, rng, n, multistart.sampling)
            vals = Vector{Tuple{Float64, Vector{eltype(θ_val)}}}(undef, n)
            for s in 1:n
                b0 = b0s[s]
                f0 = _laplace_logf_batch(dm, info, θ_val, b0, const_cache, ll_cache)
                isfinite(f0) || (f0 = -Inf)
                vals[s] = (f0, b0)
            end
            idxs = k < n ? partialsortperm(vals, 1:k; by=x -> x[1], rev=true) : (1:n)
            best_after = b_best
            best_after_f = best_f
            best_after_g_norm = g_best_norm
            for s in idxs
                b0 = vals[s][2]
                sol_try = try
                    _laplace_solve_batch!(dm, info, θ_val, const_cache, ll_cache, cache.ad_cache, bi, b0;
                                          optimizer=optimizer,
                                          optim_kwargs=optim_kwargs,
                                          adtype=adtype)
                catch err
                    if err isa DomainError || err isa ArgumentError
                        nothing
                    else
                        rethrow(err)
                    end
                end
                sol_try === nothing && continue
                b_try = sol_try.u
                f_try = _laplace_sol_logf(sol_try)
                if !isfinite(f_try)
                    f_try = _laplace_logf_batch(dm, info, θ_val, b_try, const_cache, ll_cache)
                end
                isfinite(f_try) || (f_try = -Inf)
                if f_try > best_after_f
                    best_after = b_try
                    best_after_f = f_try
                    g_try_norm = _laplace_sol_grad_norm(sol_try)
                    if !isfinite(g_try_norm)
                        g_try, _ = _laplace_gradb_cached!(cache, bi, dm, info, θ_val, const_cache, ll_cache, b_try)
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
            n *= 2
            k = min(k * 2, n)
        end
        #if g_best_norm > multistart.grad_tol
        #    @info "Laplace EBE multistart did not reach grad_tol; consider increasing multistart_n/k." batch=bi grad_norm=g_best_norm
        #end
    end
    _laplace_store_bstar!(cache, bi, b_best)
    return nothing
end

function _laplace_get_bstar!(cache::_LaplaceCache,
                             dm::DataModel,
                             batch_infos::Vector{_LaplaceBatchInfo},
                             θ::ComponentArray,
                             const_cache::LaplaceConstantsCache,
                             ll_cache;
                             optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
                             optim_kwargs::NamedTuple=NamedTuple(),
                             adtype=Optimization.AutoForwardDiff(),
                             grad_tol=1e-6,
                             theta_tol=0.0,
                             multistart=LaplaceMultistartOptions(0, 0, grad_tol, 5, :lhs),
                             rng::AbstractRNG=Random.default_rng(),
                             serialization::SciMLBase.EnsembleAlgorithm=EnsembleSerial(),
                             progress::Bool=false,
                             progress_desc::AbstractString="EBE")
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
    p = ProgressMeter.Progress(length(batch_infos); desc=progress_desc, enabled=progress && !isempty(batch_infos))
    θ_val = _laplace_floatize(θ)
    batch_rngs = _laplace_thread_rngs(rng, length(batch_infos))
    use_threaded = serialization isa SciMLBase.EnsembleThreads || ll_cache isa AbstractVector
    if use_threaded
        nthreads = Threads.maxthreadid()
        caches = _laplace_thread_caches(dm, ll_cache, nthreads)
        Threads.@threads for bi in eachindex(batch_infos)
            info = batch_infos[bi]
            tid = Threads.threadid()
            _laplace_compute_bstar_batch!(cache, bi, dm, info, θ_val, const_cache, caches[tid];
                                          optimizer=optimizer,
                                          optim_kwargs=optim_kwargs,
                                          adtype=adtype,
                                          grad_tol=grad_tol,
                                          multistart=multistart,
                                          rng=batch_rngs[bi])
            ProgressMeter.next!(p)
        end
    else
        ll_cache_local = ll_cache isa AbstractVector ? ll_cache[1] : ll_cache
        for (bi, info) in enumerate(batch_infos)
            _laplace_compute_bstar_batch!(cache, bi, dm, info, θ_val, const_cache, ll_cache_local;
                                          optimizer=optimizer,
                                          optim_kwargs=optim_kwargs,
                                          adtype=adtype,
                                          grad_tol=grad_tol,
                                          multistart=multistart,
                                          rng=batch_rngs[bi])
            ProgressMeter.next!(p)
        end
    end
    ProgressMeter.finish!(p)
    cache.θ_cache = copy(θ_vec)
    return cache.bstar_cache.b_star
end

function _laplace_hessian_b(dm::DataModel,
                            batch_info::_LaplaceBatchInfo,
                            θ::ComponentArray,
                            b,
                            const_cache::LaplaceConstantsCache,
                            cache::_LLCache,
                            ad_cache::Union{Nothing, LaplaceADCache},
                            bi::Int;
                            ctx::AbstractString="")
    f = _LaplaceLogfBatch(dm, batch_info, θ, const_cache, cache)
    H = Matrix{promote_type(eltype(θ), eltype(b))}(undef, length(b), length(b))
    cfg = nothing
    if ad_cache === nothing
        cfg = ForwardDiff.HessianConfig(f, b)
    else
        entry = ad_cache.hess_cfg[bi]
        ftype = typeof(f)
        if entry === nothing
            cfg = ForwardDiff.HessianConfig(f, b)
            ad_cache.hess_cfg[bi] = Any[((ftype, eltype(b)), cfg)]
        else
            entry = entry::Vector{Any}
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

function _laplace_cholesky_negH(H::AbstractMatrix; jitter=1e-6, max_tries=6, growth=10.0,
                                adaptive=false, scale_factor=0.0)
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
        chol = cholesky(Hneg + jit * I, check=false)
        chol.info == 0 && return (chol, jit)
        jit *= growth
    end
    return (chol, jit)
end

function _laplace_logdet_negH(dm::DataModel,
                              batch_info::_LaplaceBatchInfo,
                              θ::ComponentArray,
                              b,
                              const_cache::LaplaceConstantsCache,
                              cache::_LLCache,
                              ad_cache::Union{Nothing, LaplaceADCache},
                              bi::Int;
                              jitter=1e-6,
                              max_tries=6,
                              growth=10.0,
                              adaptive=false,
                              scale_factor=0.0,
                              ctx::AbstractString="",
                              hess_cache::Union{Nothing, _LaplaceHessCache}=nothing,
                              use_cache::Bool=false)
    if batch_info.n_b == 0
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
                    return (hess_cache.last_logdet[bi], hess_cache.last_H[bi], hess_cache.last_chol[bi])
                end
            end
        end
    end
    H = _laplace_hessian_b(dm, batch_info, θ, b, const_cache, cache, ad_cache, bi; ctx=ctx)
    chol, _ = _laplace_cholesky_negH(H; jitter=jitter, max_tries=max_tries, growth=growth,
                                     adaptive=adaptive, scale_factor=scale_factor)
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
                             batch_info::_LaplaceBatchInfo,
                             θ::ComponentArray,
                             b,
                             const_cache::LaplaceConstantsCache,
                             cache::_LLCache,
                             ad_cache::Union{Nothing, LaplaceADCache},
                             bi::Int;
                             jitter=1e-6,
                             max_tries=6,
                             growth=10.0,
                             adaptive=false,
                             scale_factor=0.0,
                             use_trace_logdet_grad::Bool=true,
                             use_hutchinson::Bool=false,
                             hutchinson_n::Int=8,
                             rng::AbstractRNG=Random.default_rng())
    nb = batch_info.n_b
    if nb == 0
        logf = _laplace_logf_batch(dm, batch_info, θ, b, const_cache, cache)
        grad = ForwardDiff.gradient(θv -> _laplace_logf_batch(dm, batch_info, θv, b, const_cache, cache), θ)
        grad = grad isa ComponentArray ? grad : ComponentArray(grad, getaxes(θ))
        return (logf=logf, logdet=0.0, grad=grad)
    end

    logf = _laplace_logf_batch(dm, batch_info, θ, b, const_cache, cache)
    logdet, H, chol = _laplace_logdet_negH(dm, batch_info, θ, b, const_cache, cache, ad_cache, bi; jitter=jitter, max_tries=max_tries, growth=growth,
                                           adaptive=adaptive, scale_factor=scale_factor, ctx="logdet")
    infT = convert(eltype(θ), Inf)
    logdet == Inf && return (logf=-Inf, logdet=infT, grad=ComponentArray(zeros(length(θ)), getaxes(θ)))

    axs = getaxes(θ)
    θ_vec = θ
    nθ = length(θ_vec)
    nb = length(b)
    buf = ad_cache === nothing ?
          _LaplaceGradBuffers(Vector{eltype(θ_vec)}(undef, nθ),
                              Matrix{eltype(θ_vec)}(undef, nb, nθ),
                              Vector{eltype(θ_vec)}(undef, nθ),
                              Vector{eltype(θ_vec)}(undef, nb),
                              use_trace_logdet_grad ? Matrix{eltype(θ_vec)}(undef, _ntri(nb), nθ) : Matrix{eltype(θ_vec)}(undef, 0, 0),
                              use_trace_logdet_grad ? Matrix{eltype(θ_vec)}(undef, _ntri(nb), nb) : Matrix{eltype(θ_vec)}(undef, 0, 0),
                              nθ, nb,
                              Vector{eltype(θ_vec)}(undef, nb),
                              Any[], Any[], Any[], Any[], Any[], Any[], Any[]) :
          _get_grad_buffers!(ad_cache, bi, eltype(θ_vec), nθ, nb, use_trace_logdet_grad)
    # envelope term
    logf_θ = _LaplaceLogfTheta(dm, batch_info, b, const_cache, cache)
    cfg = _get_fd_cfg!(buf.grad_logf_cfg, logf_θ, θ_vec, () -> ForwardDiff.GradientConfig(logf_θ, θ_vec))
    ForwardDiff.gradient!(buf.grad_logf, logf_θ, θ_vec, cfg)
    grad_logf = buf.grad_logf
    # g(b, θ) = ∂ logf / ∂b
    gθ!(out, θv) = begin
        fb = bv -> _laplace_logf_batch(dm, batch_info, θv, bv, const_cache, cache)
        cfg = _get_fd_cfg!(buf.gθ_grad_cfg, fb, b, () -> ForwardDiff.GradientConfig(fb, b))
        ForwardDiff.gradient!(out, fb, b, cfg)
        return out
    end
    cfg = _get_fd_cfg!(buf.Gθ_cfg, gθ!, θ_vec, () -> ForwardDiff.JacobianConfig(gθ!, buf.gradb_tmp, θ_vec))
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
                Hθ = _laplace_hessian_b(dm, batch_info, θv, b, const_cache, cache, nothing, bi; ctx="hutch_dH_dθ")
                Hθ * Az
            end
            cfg = _get_fd_cfg!(buf.logdet_θ_cfg, fθ, θ_vec, () -> ForwardDiff.JacobianConfig(fθ, θ_vec))
            Jθv = ForwardDiff.jacobian(fθ, θ_vec, cfg)
            grad_logdet_θ .+= -(Jθv' * z)
            fb = bv -> begin
                Hb = _laplace_hessian_b(dm, batch_info, θ, bv, const_cache, cache, nothing, bi; ctx="hutch_dH_db")
                Hb * Az
            end
            cfg = _get_fd_cfg!(buf.logdet_b_cfg, fb, b, () -> ForwardDiff.JacobianConfig(fb, b))
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
            Hθ = _laplace_hessian_b(dm, batch_info, θv, b, const_cache, cache, nothing, bi; ctx="trace_dH_dθ")
            _vech(Hθ)
        end
        cfg = _get_fd_cfg!(buf.Jθ_cfg, fθ, θ_vec, () -> ForwardDiff.JacobianConfig(fθ, θ_vec))
        ForwardDiff.jacobian!(buf.Jθ, fθ, θ_vec, cfg)
        grad_logdet_θ = -(buf.Jθ' * weights)

        fb = bv -> begin
            Hb = _laplace_hessian_b(dm, batch_info, θ, bv, const_cache, cache, nothing, bi; ctx="trace_dH_db")
            _vech(Hb)
        end
        cfg = _get_fd_cfg!(buf.Jb_cfg, fb, b, () -> ForwardDiff.JacobianConfig(fb, b))
        ForwardDiff.jacobian!(buf.Jb, fb, b, cfg)
        grad_logdet_b = -(buf.Jb' * weights)
    else
        logdet_θ = _LaplaceLogdetTheta(dm, batch_info, b, const_cache, cache, (; jitter=jitter, max_tries=max_tries, growth=growth,
                                                                              adaptive=adaptive, scale_factor=scale_factor),
                                      ad_cache, bi)
        cfg = _get_fd_cfg!(buf.logdet_θ_cfg, logdet_θ, θ_vec, () -> ForwardDiff.GradientConfig(logdet_θ, θ_vec))
        ForwardDiff.gradient!(buf.grad_logdet_θ, logdet_θ, θ_vec, cfg)
        grad_logdet_θ = buf.grad_logdet_θ

        logdet_b = _LaplaceLogdetB(dm, batch_info, θ, const_cache, cache, (; jitter=jitter, max_tries=max_tries, growth=growth,
                                                                           adaptive=adaptive, scale_factor=scale_factor),
                                   ad_cache, bi)
        cfg = _get_fd_cfg!(buf.logdet_b_cfg, logdet_b, b, () -> ForwardDiff.GradientConfig(logdet_b, b))
        ForwardDiff.gradient!(buf.grad_logdet_b, logdet_b, b, cfg)
        grad_logdet_b = buf.grad_logdet_b
    end

    # db*/dθ = (-H)^{-1} * ∂g/∂θ
    #Hneg = Symmetric(-H)
    dbdθ = chol \ Gθ
    corr = vec(grad_logdet_b' * dbdθ)
    grad = grad_logf .- 0.5 .* (grad_logdet_θ .+ corr)
    return (logf=logf, logdet=logdet, grad=ComponentArray(grad, axs))
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

@inline _default_inner_grad_tol(dm::DataModel) = dm.model.de.de === nothing ? 1e-8 : 1e-2

@inline function _resolve_inner_options(inner::LaplaceInnerOptions, dm::DataModel)
    gt = inner.grad_tol
    if gt isa Symbol
        gt === :auto || error("inner_grad_tol must be numeric or :auto.")
        return LaplaceInnerOptions(inner.optimizer, inner.kwargs, inner.adtype, _default_inner_grad_tol(dm))
    end
    return inner
end

@inline function _resolve_multistart_options(multistart::LaplaceMultistartOptions, inner::LaplaceInnerOptions)
    gt = multistart.grad_tol
    sampling = _resolve_multistart_sampling(multistart.sampling, "multistart_sampling")
    if gt isa Symbol
        gt === :auto || error("multistart_grad_tol must be numeric or :auto.")
        return LaplaceMultistartOptions(multistart.n, multistart.k, inner.grad_tol, multistart.max_rounds, sampling)
    end
    return LaplaceMultistartOptions(multistart.n, multistart.k, gt, multistart.max_rounds, sampling)
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
The outer optimiser maximises the Laplace-approximated marginal likelihood over the
fixed effects, while the inner optimiser computes per-individual MAP estimates of the
random effects.

# Keyword Arguments
- `optimizer`: outer Optimization.jl optimiser. Defaults to `LBFGS` with backtracking.
- `optim_kwargs::NamedTuple = NamedTuple()`: keyword arguments for the outer `solve` call.
- `adtype`: AD backend for the outer optimiser. Defaults to `AutoForwardDiff()`.
- `inner_optimizer`: inner optimiser for computing EBE modes. Defaults to `LBFGS`.
- `inner_kwargs::NamedTuple = NamedTuple()`: keyword arguments for the inner `solve` call.
- `inner_adtype`: AD backend for the inner optimiser. Defaults to `AutoForwardDiff()`.
- `inner_grad_tol`: gradient tolerance for inner convergence (`:auto` chooses automatically).
- `multistart_n::Int = 50`: number of random starts for the inner EBE multistart.
- `multistart_k::Int = 10`: number of best starts to refine in the inner multistart.
- `multistart_grad_tol`: gradient tolerance for multistart refinement.
- `multistart_max_rounds::Int = 1`: maximum multistart refinement rounds.
- `multistart_sampling::Symbol = :lhs`: inner multistart sampling strategy (`:lhs` or `:random`).
- `jitter::Float64 = 1e-6`: initial diagonal jitter added to ensure Hessian PD.
- `max_tries::Int = 6`: maximum attempts to regularise the Hessian.
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
end

Laplace(; optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
        optim_kwargs=NamedTuple(),
        adtype=Optimization.AutoForwardDiff(),
        inner_options=nothing,
        hessian_options=nothing,
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
        jitter=1e-6,
        max_tries=6,
        jitter_growth=10.0,
        adaptive_jitter=true,
        jitter_scale=1e-6,
        use_trace_logdet_grad=true,
        use_hutchinson=false,
        hutchinson_n=8,
        theta_tol=0.0,
        lb=nothing,
        ub=nothing) = begin
    inner = inner_options === nothing ? LaplaceInnerOptions(inner_optimizer, inner_kwargs, inner_adtype, inner_grad_tol) : inner_options
        hess = hessian_options === nothing ? LaplaceHessianOptions(jitter, max_tries, jitter_growth, adaptive_jitter, jitter_scale, use_trace_logdet_grad, use_hutchinson, hutchinson_n) : hessian_options
    cache = cache_options === nothing ? LaplaceCacheOptions(theta_tol) : cache_options
    ms = multistart_options === nothing ? LaplaceMultistartOptions(multistart_n, multistart_k, multistart_grad_tol, multistart_max_rounds, multistart_sampling) : multistart_options
    Laplace(optimizer, optim_kwargs, adtype, inner, hess, cache, ms, lb, ub)
end

"""
    LaplaceResult{S, O, I, R, N, B} <: MethodResult

Method-specific result from a [`Laplace`](@ref) fit. Stores the solution, objective value,
iteration count, raw solver result, optional notes, and empirical-Bayes mode estimates
for each individual.
"""
struct LaplaceResult{S, O, I, R, N, B} <: MethodResult
    solution::S
    objective::O
    iterations::I
    raw::R
    notes::N
    eb_modes::B
end

"""
    LaplaceMAP(; optimizer, optim_kwargs, adtype, inner_options, hessian_options,
                 cache_options, multistart_options, inner_optimizer, inner_kwargs,
                 inner_adtype, inner_grad_tol, multistart_n, multistart_k,
                 multistart_grad_tol, multistart_max_rounds, multistart_sampling,
                 jitter, max_tries, jitter_growth, adaptive_jitter, jitter_scale,
                 use_trace_logdet_grad, use_hutchinson, hutchinson_n, theta_tol,
                 lb, ub) <: FittingMethod

Laplace approximation with MAP-regularised fixed effects for random-effects models.
Identical to [`Laplace`](@ref) but adds the log-prior of the fixed effects to the
outer objective, giving a MAP estimate of the fixed effects rather than MLE.
Requires prior distributions on at least one free fixed effect.

See [`Laplace`](@ref) for a description of all keyword arguments. The only difference
in defaults is `multistart_max_rounds = 5`.
"""
struct LaplaceMAP{O, K, A, IO, HO, CO, MS, L, U} <: FittingMethod
    optimizer::O
    optim_kwargs::K
    adtype::A
    inner::IO
    hessian::HO
    cache::CO
    multistart::MS
    lb::L
    ub::U
end

LaplaceMAP(; optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
           optim_kwargs=NamedTuple(),
           adtype=Optimization.AutoForwardDiff(),
           inner_options=nothing,
           hessian_options=nothing,
           cache_options=nothing,
           multistart_options=nothing,
           inner_optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
           inner_kwargs=NamedTuple(),
           inner_adtype=Optimization.AutoForwardDiff(),
           inner_grad_tol=:auto,
           multistart_n=50,
           multistart_k=10,
           multistart_grad_tol=inner_grad_tol,
           multistart_max_rounds=5,
           multistart_sampling=:lhs,
           jitter=1e-6,
           max_tries=6,
           jitter_growth=10.0,
           adaptive_jitter=true,
           jitter_scale=1e-6,
           use_trace_logdet_grad=true,
           use_hutchinson=false,
           hutchinson_n=8,
           theta_tol=0.0,
           lb=nothing,
           ub=nothing) = begin
    inner = inner_options === nothing ? LaplaceInnerOptions(inner_optimizer, inner_kwargs, inner_adtype, inner_grad_tol) : inner_options
    hess = hessian_options === nothing ? LaplaceHessianOptions(jitter, max_tries, jitter_growth, adaptive_jitter, jitter_scale, use_trace_logdet_grad, use_hutchinson, hutchinson_n) : hessian_options
    cache = cache_options === nothing ? LaplaceCacheOptions(theta_tol) : cache_options
    ms = multistart_options === nothing ? LaplaceMultistartOptions(multistart_n, multistart_k, multistart_grad_tol, multistart_max_rounds, multistart_sampling) : multistart_options
    LaplaceMAP(optimizer, optim_kwargs, adtype, inner, hess, cache, ms, lb, ub)
end

"""
    LaplaceMAPResult{S, O, I, R, N, B} <: MethodResult

Method-specific result from a [`LaplaceMAP`](@ref) fit. Stores the solution, objective
value, iteration count, raw solver result, optional notes, and empirical-Bayes mode
estimates for each individual.
"""
struct LaplaceMAPResult{S, O, I, R, N, B} <: MethodResult
    solution::S
    objective::O
    iterations::I
    raw::R
    notes::N
    eb_modes::B
end

mutable struct _LaplaceObjCache{T, G}
    θ::Union{Nothing, AbstractVector{T}}
    obj::T
    grad::G
    has_grad::Bool
end

@inline function _laplace_obj_cache_set_obj!(cache::_LaplaceObjCache, θ, obj)
    cache.θ = copy(θ)
    cache.obj = obj
    cache.has_grad = false
    return nothing
end

@inline function _laplace_obj_cache_set_obj_grad!(cache::_LaplaceObjCache, θ, obj, grad)
    cache.θ = copy(θ)
    cache.obj = obj
    cache.grad = grad
    cache.has_grad = true
    return nothing
end

@inline function _laplace_obj_cache_lookup(cache::_LaplaceObjCache, θ, theta_tol)
    cache.θ === nothing && return nothing
    cache.has_grad || return nothing
    maxdiff = _maxabsdiff(θ, cache.θ)
    if isfinite(maxdiff) && maxdiff <= theta_tol
        return (cache.obj, cache.grad)
    end
    return nothing
end

@inline function _laplace_obj_cache_lookup_obj(cache::_LaplaceObjCache, θ, theta_tol)
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
                       ode_args=ll_cache.ode_args,
                       ode_kwargs=ll_cache.ode_kwargs,
                       force_saveat=ll_cache.saveat_cache !== nothing,
                       nthreads=nthreads)
    else
        build_ll_cache(dm; nthreads=nthreads)
    end
    return caches isa Vector ? caches : [deepcopy(caches) for _ in 1:nthreads]
end

@inline function _laplace_thread_rngs(rng::AbstractRNG, nthreads::Int)
    return _spawn_child_rngs(rng, nthreads)
end

function _laplace_objective_and_grad(dm::DataModel,
                                     batch_infos::Vector{_LaplaceBatchInfo},
                                     θ::ComponentArray,
                                     const_cache::LaplaceConstantsCache,
                                     ll_cache,
                                     ebe_cache::_LaplaceCache;
                                     inner::LaplaceInnerOptions,
                                     hessian::LaplaceHessianOptions,
                                     cache_opts::LaplaceCacheOptions,
                                     multistart::LaplaceMultistartOptions,
                                     rng::AbstractRNG,
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
            res = _laplace_grad_batch(dm, info, θ, b, const_cache, caches[tid], ebe_cache.ad_cache, bi;
                                      jitter=hessian.jitter,
                                      max_tries=hessian.max_tries,
                                      growth=hessian.growth,
                                      adaptive=hessian.adaptive,
                                      scale_factor=hessian.scale_factor,
                                      use_trace_logdet_grad=hessian.use_trace_logdet_grad,
                                      use_hutchinson=hessian.use_hutchinson,
                                      hutchinson_n=hessian.hutchinson_n,
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
        for (bi, info) in enumerate(batch_infos)
            b = bstars[bi]
            res = _laplace_grad_batch(dm, info, θ, b, const_cache, ll_cache, ebe_cache.ad_cache, bi;
                                      jitter=hessian.jitter,
                                      max_tries=hessian.max_tries,
                                      growth=hessian.growth,
                                      adaptive=hessian.adaptive,
                                      scale_factor=hessian.scale_factor,
                                      use_trace_logdet_grad=hessian.use_trace_logdet_grad,
                                      use_hutchinson=hessian.use_hutchinson,
                                      hutchinson_n=hessian.hutchinson_n,
                                      rng=batch_rngs[bi])
            res.logf == -Inf && return (infT, ComponentArray(grad, axs), bstars)
            total += res.logf + 0.5 * info.n_b * log(2π) - 0.5 * res.logdet
            grad .+= res.grad
        end
        return (-total, ComponentArray(-grad, axs), bstars)
    end
end

function _laplace_objective_only(dm::DataModel,
                                 batch_infos::Vector{_LaplaceBatchInfo},
                                 θ::ComponentArray,
                                 const_cache::LaplaceConstantsCache,
                                 ll_cache,
                                 ebe_cache::_LaplaceCache;
                                 inner::LaplaceInnerOptions,
                                 hessian::LaplaceHessianOptions,
                                 cache_opts::LaplaceCacheOptions,
                                 multistart::LaplaceMultistartOptions,
                                 rng::AbstractRNG,
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
    use_cache = false
    if ebe_cache.θ_cache !== nothing && length(ebe_cache.θ_cache) == length(θ)
        maxdiff = _maxabsdiff(θ, ebe_cache.θ_cache)
        use_cache = isfinite(maxdiff) && maxdiff <= cache_opts.theta_tol
    end
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
            logdet, _, _ = _laplace_logdet_negH(dm, info, θ, b, const_cache, caches[tid], nothing, bi;
                                                jitter=hessian.jitter,
                                                max_tries=hessian.max_tries,
                                                growth=hessian.growth,
                                                adaptive=hessian.adaptive,
                                                scale_factor=hessian.scale_factor,
                                                hess_cache=ebe_cache.hess_cache,
                                                use_cache=use_cache)
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
        for (bi, info) in enumerate(batch_infos)
            b = bstars[bi]
            logf = _laplace_logf_batch(dm, info, θ, b, const_cache, ll_cache)
            logf == -Inf && return infT
            logdet, _, _ = _laplace_logdet_negH(dm, info, θ, b, const_cache, ll_cache, ebe_cache.ad_cache, bi;
                                                jitter=hessian.jitter,
                                                max_tries=hessian.max_tries,
                                                growth=hessian.growth,
                                                adaptive=hessian.adaptive,
                                                scale_factor=hessian.scale_factor,
                                                hess_cache=ebe_cache.hess_cache,
                                                use_cache=use_cache)
            logdet == Inf && return infT
            total += logf + 0.5 * info.n_b * log(2π) - 0.5 * logdet
        end
    end
    return -total
end

function _fit_model(dm::DataModel, method::Laplace, args...;
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
    isempty(re_names) && error("Laplace requires random effects. Use MLE/MAP for fixed-effects models.")
    fe = dm.model.fixed.fixed
    fixed_names = get_names(fe)
    isempty(fixed_names) && error("Laplace requires at least one fixed effect.")
    fixed_set = Set(fixed_names)
    for name in keys(constants)
        name in fixed_set || error("Unknown constant parameter $(name).")
    end
    all(name in keys(constants) for name in fixed_names) &&
        error("Laplace requires at least one free fixed effect. Remove constants or specify a fixed effect or random effect.")

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

    pairing, batch_infos, const_cache = _build_laplace_batch_infos(dm, constants_re)
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

    function obj_only(θt, p)
        θt_free = θt isa ComponentArray ? θt : ComponentArray(θt, axs_free)
        cached_obj = _laplace_obj_cache_lookup_obj(obj_cache, θt_free, method.cache.theta_tol)
        cached_obj !== nothing && return cached_obj
        T = eltype(θt_free)
        infT = convert(T, Inf)
        θt_full = ComponentArray(T.(θ_const_t), axs_full)
        for name in free_names
            setproperty!(θt_full, name, getproperty(θt_free, name))
        end
        θu = inv_transform(θt_full)
        obj = _laplace_objective_only(dm, batch_infos, θu, const_cache, ll_cache, ebe_cache;
                                      inner=inner_opts,
                                      hessian=method.hessian,
                                      cache_opts=method.cache,
                                      multistart=multistart_opts,
                                      rng=rng,
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
        obj, grad_full, bstars = _laplace_objective_and_grad(dm, batch_infos, θu, const_cache, ll_cache, ebe_cache;
                                                             inner=inner_opts,
                                                             hessian=method.hessian,
                                                             cache_opts=method.cache,
                                                             multistart=multistart_opts,
                                                             rng=rng,
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
        error("BlackBoxOptim methods require finite bounds. Add lower/upper bounds in @fixedEffects (on transformed scale) or pass them via Laplace(lb=..., ub=...). A quick helper is default_bounds_from_start(dm; margin=...).")
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

    summary = FitSummary(sol.objective, sol.retcode == SciMLBase.ReturnCode.Success,
                         FitParameters(θ_hat_t, θ_hat_u),
                         NamedTuple())
    diagnostics = FitDiagnostics((;), (optimizer=method.optimizer,), (retcode=sol.retcode,), NamedTuple())
    niter = hasproperty(sol, :stats) && hasproperty(sol.stats, :iterations) ? sol.stats.iterations : missing
    raw = hasproperty(sol, :original) ? sol.original : sol
    result = LaplaceResult(sol, sol.objective, niter, raw, NamedTuple(), ebe_cache.bstar_cache.b_star)
    return FitResult(method, result, summary, diagnostics, store_data_model ? dm : nothing, args, fit_kwargs)
end

function _fit_model(dm::DataModel, method::LaplaceMAP, args...;
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
    isempty(re_names) && error("LaplaceMAP requires random effects. Use MAP for fixed-effects models.")
    fe = dm.model.fixed.fixed
    priors = get_priors(fe)
    fixed_names = get_names(fe)
    isempty(fixed_names) && error("LaplaceMAP requires at least one fixed effect.")
    for name in fixed_names
        hasproperty(priors, name) || error("LaplaceMAP requires priors on all fixed effects. Missing prior for $(name).")
        getfield(priors, name) isa Priorless && error("LaplaceMAP requires priors on all fixed effects. Priorless for $(name).")
    end

    fixed_set = Set(fixed_names)
    for name in keys(constants)
        name in fixed_set || error("Unknown constant parameter $(name).")
    end
    all(name in keys(constants) for name in fixed_names) &&
        error("LaplaceMAP requires at least one free fixed effect. Remove constants or specify a fixed effect or random effect.")

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

    pairing, batch_infos, const_cache = _build_laplace_batch_infos(dm, constants_re)
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

    function obj_only(θt, p)
        θt_free = θt isa ComponentArray ? θt : ComponentArray(θt, axs_free)
        cached_obj = _laplace_obj_cache_lookup_obj(obj_cache, θt_free, method.cache.theta_tol)
        cached_obj !== nothing && return cached_obj
        T = eltype(θt_free)
        infT = convert(T, Inf)
        θt_full = ComponentArray(T.(θ_const_t), axs_full)
        for name in free_names
            setproperty!(θt_full, name, getproperty(θt_free, name))
        end
        θu = inv_transform(θt_full)
        obj = _laplace_objective_only(dm, batch_infos, θu, const_cache, ll_cache, ebe_cache;
                                      inner=inner_opts,
                                      hessian=method.hessian,
                                      cache_opts=method.cache,
                                      multistart=multistart_opts,
                                      rng=rng,
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
        obj, grad_full, bstars = _laplace_objective_and_grad(dm, batch_infos, θu, const_cache, ll_cache, ebe_cache;
                                                             inner=inner_opts,
                                                             hessian=method.hessian,
                                                             cache_opts=method.cache,
                                                             multistart=multistart_opts,
                                                             rng=rng,
                                                             serialization=serialization)
        obj == Inf && return (infT, ComponentArray(zeros(T, length(θt_free)), axs_free))
        lp = logprior(fe, θu)
        lp == -Inf && return (infT, ComponentArray(zeros(T, length(θt_free)), axs_free))
        penalty_val = _penalty_value(θu, penalty)
        obj += -lp + penalty_val

        grad_u = grad_full
        lp_grad = ForwardDiff.gradient(x -> logprior(fe, x), θu)
        pen_grad = ForwardDiff.gradient(x -> _penalty_value(x, penalty), θu)
        grad_u = grad_u .- lp_grad .+ pen_grad

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
        error("BlackBoxOptim methods require finite bounds. Add lower/upper bounds in @fixedEffects (on transformed scale) or pass them via LaplaceMAP(lb=..., ub=...). A quick helper is default_bounds_from_start(dm; margin=...).")
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

    summary = FitSummary(sol.objective, sol.retcode == SciMLBase.ReturnCode.Success,
                         FitParameters(θ_hat_t, θ_hat_u),
                         NamedTuple())
    diagnostics = FitDiagnostics((;), (optimizer=method.optimizer,), (retcode=sol.retcode,), NamedTuple())
    niter = hasproperty(sol, :stats) && hasproperty(sol.stats, :iterations) ? sol.stats.iterations : missing
    raw = hasproperty(sol, :original) ? sol.original : sol
    result = LaplaceMAPResult(sol, sol.objective, niter, raw, NamedTuple(), ebe_cache.bstar_cache.b_star)
    return FitResult(method, result, summary, diagnostics, store_data_model ? dm : nothing, args, fit_kwargs)
end
import ForwardDiff

@inline _laplace_value(x) = x
@inline _laplace_value(x::ForwardDiff.Dual) = ForwardDiff.value(x)

function _laplace_floatize(θ::ComponentArray)
    eltype(θ) === Float64 && return θ
    vals = map(_laplace_value, θ)
    return ComponentArray(Float64.(vals), getaxes(θ))
end
