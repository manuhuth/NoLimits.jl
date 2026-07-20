# Shared random-effects batching, constant-RE caching, and per-individual η
# assembly. Used by every random-effects estimator (Laplace, FOCEI, SAEM, MCEM,
# GHQuadrature, the MH samplers, kernel) plus CV, prediction, and plotting — hence
# its own file rather than living inside laplace.jl.

using ComponentArrays

struct REConstantsCache{M, S, V}
    is_const::M
    scalar_vals::S
    vector_vals::V
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
        if spec isa NamedTuple || spec isa AbstractDict
            # Keep as-is; Base.pairs works for any key type. Avoid (; spec...)
            # which fails when keys are not Symbols (e.g. integer group levels).
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
            matched ||
                error("constants_re for $(re) includes level $(k) not found in column $(col). The value must be present in that column.")
        end
        push!(pairs, re => dict)
    end
    return NamedTuple(pairs)
end

function _build_constants_cache(dm::DataModel, constants_re::NamedTuple)
    cache = dm.re_group_info.laplace_cache
    cache === nothing && return REConstantsCache(
        BitVector[], Vector{Vector{Float64}}(), Vector{Vector{Vector{Float64}}}())
    re_names = cache.re_names
    nre = length(re_names)
    is_const = Vector{BitVector}(undef, nre)
    scalar_vals = Vector{Vector{Float64}}(undef, nre)
    vector_vals = Vector{Vector{Vector{Float64}}}(undef, nre)
    for (ri, re) in enumerate(re_names)
        levels = cache.re_index[ri].levels
        is_const[ri] = falses(length(levels))
        if cache.is_scalar[ri]
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
            idx == 0 &&
                error("constants_re for $(re) includes level $(k) not found in column $(getfield(get_re_groups(dm.model.random.random), re)). The value must be present in that column.")
            is_const[ri][idx] = true
            if cache.is_scalar[ri]
                v isa Number ||
                    error("constants_re for $(re) level $(k) must be a scalar number.")
                scalar_vals[ri][idx] = Float64(v)
            else
                v isa AbstractVector ||
                    error("constants_re for $(re) level $(k) must be a vector.")
                length(v) == cache.dims[ri] ||
                    error("constants_re for $(re) level $(k) must have length $(cache.dims[ri]).")
                vector_vals[ri][idx] = Float64.(v)
            end
        end
    end
    return REConstantsCache(is_const, scalar_vals, vector_vals)
end
function _build_re_batches(dm::DataModel, const_cache::REConstantsCache)
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

struct _REMap{L, M}
    levels::L
    level_to_index::M
end

struct RELevelInfo{A, R, P}
    map::A
    ranges::R
    reps::P
    dim::Int
    is_scalar::Bool
end
const _REInfo = RELevelInfo

struct REBatchInfo{B, R}
    inds::B
    re_info::R
    n_b::Int
end
function build_re_batch_infos(dm::DataModel, constants_re::NamedTuple)
    constants_re = _normalize_constants_re(dm, constants_re)
    const_cache = _build_constants_cache(dm, constants_re)
    pairing = _build_re_batches(dm, const_cache)
    cache = dm.re_group_info.laplace_cache
    cache === nothing && return pairing, REBatchInfo[], const_cache
    re_names = cache.re_names
    isempty(re_names) && return pairing, REBatchInfo[], const_cache
    batch_infos = Vector{REBatchInfo}(undef, length(pairing.batches))
    # ponytail: one shared level→batch-local-index map per RE instead of a fresh
    # length-nlevels array per batch. Level ids partition across batches (union-find
    # groups every individual sharing a level), so batches fill disjoint slots and
    # never collide. Drops construction from O(n_batches × nlevels) to O(nlevels).
    shared_lti = [zeros(Int, length(cache.re_index[ri].levels))
                  for ri in eachindex(re_names)]
    for (bi, inds) in enumerate(pairing.batches)
        total_dim = 0
        re_info = Vector{_REInfo}(undef, length(re_names))
        for (ri, re) in enumerate(re_names)
            level_to_index = shared_lti[ri]
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
                ranges[li] = (total_dim + 1):(total_dim + dim)
                total_dim += dim
            end
            map = _REMap(levels, level_to_index)
            re_info[ri] = _REInfo(map, ranges, reps, dim, is_scalar)
        end
        # Narrow `re_info` to a concrete eltype before storing it. The `undef`
        # scratch buffer above has the abstract `_REInfo` eltype, which makes
        # `REBatchInfo.re_info` abstract — so every `batch_info.re_info[ri]`
        # field access on the per-row EBE hot path (`_build_eta_ind_fast`,
        # `_laplace_logf_batch`, the grad/Hessian assembly) boxes and dynamic-
        # dispatches. All entries share one concrete type, so `identity.` narrows
        # the eltype with no data copy (same element objects); it auto-falls back to
        # an abstract eltype if a future config ever mixes element types. The outer
        # `batch_infos::Vector{REBatchInfo}` stays abstract on purpose (the
        # per-batch dispatch is amortized once per batch, and ~25 signatures across
        # the estimators annotate `::Vector{REBatchInfo}`).
        batch_infos[bi] = REBatchInfo(inds, identity.(re_info), total_dim)
    end
    return pairing, batch_infos, const_cache
end
const _build_re_batch_infos = build_re_batch_infos

@inline function random_effect_value(info::RELevelInfo, level_id::Int, b)
    idx = info.map.level_to_index[level_id]
    idx == 0 && return nothing
    r = info.ranges[idx]
    # Only genuinely univariate REs collapse to a scalar value. A length-1
    # multivariate RE (e.g. 1-D MvNormal) must stay a 1-vector so that its
    # logpdf/mean operate on a vector.
    if info.is_scalar
        return b[first(r)]
    else
        return view(b, r)
    end
end
const _re_value_from_b = random_effect_value

function build_eta_individual(dm::DataModel,
        ind_idx::Int,
        batch_info::REBatchInfo,
        b,
        const_cache::REConstantsCache,
        θ::ComponentArray)
    cache = dm.re_group_info.laplace_cache
    template = cache.eta_template
    if template !== nothing
        return _build_eta_ind_fast(template, ind_idx, batch_info, b, const_cache, cache)
    end
    # Slow path: heterogeneous case (some individuals have multiple levels per RE group).
    re_names = cache.re_names
    nt_pairs = Pair{Symbol, Any}[]
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
                if info.is_scalar
                    v = const_scalars[id]
                    push!(nt_pairs, re => T(v))
                else
                    v = const_vectors[id]
                    push!(nt_pairs, re => T.(v))
                end
            else
                v = _re_value_from_b(info, id, b)
                v === nothing &&
                    error("Missing random effect value for $(re) level $(cache.re_index[ri].levels[id]).")
                if info.is_scalar
                    push!(nt_pairs, re => T(v))
                else
                    push!(nt_pairs, re => Vector{T}(v))
                end
            end
        else
            if info.is_scalar
                vals = Vector{T}(undef, length(ids))
            else
                vals = Vector{Vector{T}}(undef, length(ids))
            end
            for (gi, id) in pairs(ids)
                if const_mask[id]
                    if info.is_scalar
                        v = const_scalars[id]
                        vals[gi] = T(v)
                    else
                        v = const_vectors[id]
                        vals[gi] = T.(v)
                    end
                else
                    v = _re_value_from_b(info, id, b)
                    v === nothing &&
                        error("Missing random effect value for $(re) level $(cache.re_index[ri].levels[id]).")
                    if info.is_scalar
                        vals[gi] = T(v)
                    else
                        vals[gi] = Vector{T}(v)
                    end
                end
            end
            push!(nt_pairs, re => vals)
        end
    end
    nt = NamedTuple(nt_pairs)
    return ComponentArray(nt)
end
const _build_eta_ind = build_eta_individual

# In-place variant for hot loops that evaluate many η per batch (e.g. one per
# quadrature node): writes into a caller-owned buffer and wraps it with the
# template axes. The returned ComponentArray aliases `vals`, so callers must
# consume it before the next call reuses the buffer.
function _build_eta_ind_fast!(vals::Vector{T},
        template::ComponentArray{Float64},
        ind_idx::Int,
        batch_info::REBatchInfo,
        b,
        const_cache::REConstantsCache,
        cache) where {T}
    re_names = cache.re_names
    out_pos = 1
    for (ri, re) in enumerate(re_names)
        info = batch_info.re_info[ri]
        id = cache.ind_level_ids[ind_idx][ri][1]
        const_mask = const_cache.is_const[ri]
        if const_mask[id]
            if info.is_scalar
                @inbounds vals[out_pos] = T(const_cache.scalar_vals[ri][id])
                out_pos += 1
            else
                cv = const_cache.vector_vals[ri][id]
                d = info.dim
                @inbounds for k in 1:d
                    vals[out_pos + k - 1] = T(cv[k])
                end
                out_pos += d
            end
        else
            b_idx = info.map.level_to_index[id]
            b_idx == 0 &&
                error("Missing random effect value for $(re) level $(cache.re_index[ri].levels[id]).")
            r = info.ranges[b_idx]
            if info.is_scalar
                @inbounds vals[out_pos] = b[first(r)]
                out_pos += 1
            else
                d = info.dim
                r_start = first(r)
                @inbounds for k in 1:d
                    vals[out_pos + k - 1] = b[r_start + k - 1]
                end
                out_pos += d
            end
        end
    end
    return ComponentArray(vals, getaxes(template))
end

# Fast path for `_build_eta_ind`: used when every individual has exactly one RE level
# per RE group (the common case, e.g. `column=:ID`). Avoids Pair{Symbol,Any}[] boxing
# by filling a flat Vector{T} and wrapping it with pre-computed axes.
function _build_eta_ind_fast(template::ComponentArray{Float64},
        ind_idx::Int,
        batch_info::REBatchInfo,
        b,
        const_cache::REConstantsCache,
        cache)
    vals = Vector{eltype(b)}(undef, length(template))
    return _build_eta_ind_fast!(
        vals, template, ind_idx, batch_info, b, const_cache, cache)
end

# ── Accessors for the RE-batch structs (used across the estimators) ───────────
@inline get_inds(info::REBatchInfo) = info.inds
@inline get_re_info(info::REBatchInfo) = info.re_info
@inline get_n_b(info::REBatchInfo) = info.n_b

@inline get_re_map(ri::_REInfo) = ri.map
@inline get_ranges(ri::_REInfo) = ri.ranges
@inline get_reps(ri::_REInfo) = ri.reps
@inline get_dim(ri::_REInfo) = ri.dim
@inline get_is_scalar(ri::_REInfo) = ri.is_scalar

@inline get_levels(m::_REMap) = m.levels
@inline get_level_to_index(m::_REMap) = m.level_to_index

# ── Public batch/level accessors (developer API; thin covers over the internals) ──
@inline get_batch_individuals(info::REBatchInfo) = get_inds(info)
@inline get_batch_re_info(info::REBatchInfo) = get_re_info(info)
@inline get_batch_re_dim(info::REBatchInfo) = get_n_b(info)
@inline get_re_levels(li::RELevelInfo) = get_levels(get_re_map(li))
@inline get_re_ranges(li::RELevelInfo) = get_ranges(li)
@inline get_re_reps(li::RELevelInfo) = get_reps(li)
@inline get_re_dim(li::RELevelInfo) = get_dim(li)
@inline get_re_is_scalar(li::RELevelInfo) = get_is_scalar(li)

export build_re_dists

# One-shot: build the RE distribution NamedTuple for one (θ, const_cov). For loops that build
# many dists at a fixed θ, hoist `create_random_effect_distribution(get_random(model))` once
# instead of calling this per iteration.
@inline function build_re_dists(model, θ, const_cov)
    return create_random_effect_distribution(get_random(model))(
        θ, const_cov, get_model_funs(model), get_helper_funs(model))
end
