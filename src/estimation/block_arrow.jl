# Block-arrow batches (#408). With nested random effects (e.g. subject-level REs inside
# experiment-level REs) one batch holds all individuals of an experiment. Its log-joint is
# a sum of per-individual terms, each depending on the individual's own levels and the
# shared ones, so -∇²_b log f is block diagonal over the own blocks plus a dense coupling
# to the shared block. Derivatives are taken per individual on single-individual
# sub-batches (the existing batch functions, unchanged) and assembled; the Newton step,
# log-determinant, inverse blocks and db*/dθ go through the Schur complement on the
# shared block. Per-individual results land in their own slots and are reduced serially
# in index order, so the result does not depend on who filled the slots (`pool`).

# A batch is structured when it has >= 2 individuals and its shared block (indices touched
# by >= 2 individuals) is non-empty and at most half the batch dimension; other batches
# (single individuals, crossed designs) keep the dense path.
struct _BAStructure{I}
    info::REBatchInfo             # parent batch (identity key)
    subs::Vector{I}               # one single-individual sub-batch per individual
    maps::Vector{Vector{Int}}     # sub-batch position -> parent b index
    lidx::Vector{Vector{Vector{Int}}}  # per RE: sub-batch level -> parent level index
    own::Vector{Vector{Int}}      # sub-batch positions of the individual's own indices
    ownglob::Vector{Vector{Int}}  # their parent b indices
    shl::Vector{Vector{Int}}      # sub-batch positions of shared indices
    shg::Vector{Vector{Int}}      # their positions in the shared block
    shared::Vector{Int}           # parent b indices of the shared block
    max_dim::Int                  # largest sub-batch or shared-block dimension
end

# Sub-batch of individual `i`: every free level `i` touches (so η and the FOCE prior mean
# are complete), with the parent's representative per level. `_re_free_prior_logf` counts
# a level's prior only in the sub-batch of its representative, so the prior terms of all
# sub-batches sum to the parent's. `level_to_index` is a Dict: the parent's array is
# shared across batches and must not be written.
function _ba_sub_batch(dm::DataModel, info::REBatchInfo, i::Int, const_cache::REConstantsCache)
    rc = get_laplace_cache(get_re_group_info(dm))
    glob = Int[]
    pos = 0
    re_info = Vector{Any}(undef, length(get_re_info(info)))
    lidx = Vector{Vector{Int}}(undef, length(re_info))
    for (ri, pinfo) in enumerate(get_re_info(info))
        plti = get_level_to_index(get_re_map(pinfo))
        levels, reps, ranges, pl = Int[], Int[], UnitRange{Int}[], Int[]
        lti = Dict{Int, Int}()
        d = get_dim(pinfo)
        for id in get_ind_level_ids(rc)[i][ri]
            (const_cache.is_const[ri][id] || haskey(lti, id)) && continue
            pidx = plti[id]
            push!(levels, id)
            push!(pl, pidx)
            push!(reps, get_reps(pinfo)[pidx])
            lti[id] = length(levels)
            push!(ranges, (pos + 1):(pos + d))
            pos += d
            append!(glob, get_ranges(pinfo)[pidx])
        end
        re_info[ri] = RELevelInfo(_REMap(levels, lti), ranges, reps, d, get_is_scalar(pinfo))
        lidx[ri] = pl
    end
    return REBatchInfo([i], identity.(re_info), pos), glob, lidx
end

function _ba_structure(dm::DataModel, info::REBatchInfo, const_cache::REConstantsCache)
    inds = get_inds(info)
    nb = get_n_b(info)
    length(inds) >= 2 || return nothing
    built = [_ba_sub_batch(dm, info, i, const_cache) for i in inds]
    subs = [s for (s, _, _) in built]
    maps = [m for (_, m, _) in built]
    count = zeros(Int, nb)
    for m in maps, j in m
        count[j] += 1
    end
    shared = findall(>=(2), count)
    ns = length(shared)
    (ns >= 1 && 2 * ns <= nb) || return nothing
    shpos = zeros(Int, nb)
    shpos[shared] .= 1:ns
    own = [findall(j -> count[j] == 1, m) for m in maps]
    shl = [findall(j -> count[j] >= 2, m) for m in maps]
    return _BAStructure(
        info, subs, maps, [l for (_, _, l) in built], own, [m[o] for (m, o) in zip(maps, own)], shl,
        [shpos[m[s]] for (m, s) in zip(maps, shl)], shared,
        max(ns, maximum(length, maps))
    )
end

# Structures of `batch_infos`, memoized in the EBE cache (keyed by the batch vector).
function _ba_structures!(
        ebe::_LaplaceCache, dm::DataModel, batch_infos::Vector{REBatchInfo},
        const_cache::REConstantsCache
    )
    s = ebe.structures
    s !== nothing && s[1] === batch_infos && return s[2]
    sts = Union{Nothing, _BAStructure}[_ba_structure(dm, info, const_cache) for info in batch_infos]
    ebe.structures = (batch_infos, sts)
    return sts
end

@inline function _ba_get(ebe::_LaplaceCache, bi::Int, info::REBatchInfo)
    s = ebe.structures
    s === nothing && return nothing
    st = s[2][bi]
    return st !== nothing && st.info === info ? st : nothing
end

# f(k, cache) for every individual k: serially on `cache`, or on the task-owned slots of
# `pool` (within-batch threading). `f` may write only slot k of its outputs and must not
# assign a name that is also a local of the enclosing function (that local would be one
# variable shared by all tasks).
function _ba_foreach(f, n::Int, cache, pool)
    if pool === nothing
        for k in 1:n
            f(k, cache)
        end
    else
        try
            _greedy_foreach((k, c) -> f(k, pool[c]), n, length(pool))
        catch err
            # rethrow the task's own exception, as the serial loop would raise it
            err isa CompositeException || rethrow()
            e = first(err.exceptions)
            throw(e isa TaskFailedException ? e.task.exception : e)
        end
    end
    return nothing
end

# Factorization of A = -H = [blockdiag(A_oo) B; Bᵀ D]: Cholesky of each own block,
# M_k = A_oo,k \ B_k and of the Schur complement S = D - Σ B_kᵀ M_k.
struct _BAFactor{T, C, SC}
    chols::Vector{C}
    M::Vector{Matrix{T}}
    B::Vector{Matrix{T}}
    S::SC
    logdet::T
end

# `nothing` when A + jit·I is not positive definite (or not finite): A is positive definite
# iff every own block and the Schur complement are, so this is the un-jittered
# definiteness test of `negH_definite_without_jitter` at jit = 0.
function _ba_factor(st::_BAStructure, Hs::AbstractVector, jit::Real = 0)
    T = eltype(first(Hs))
    ns = length(st.shared)
    n = length(Hs)
    D = zeros(T, ns, ns)
    chols = Vector{Cholesky{T, Matrix{T}}}(undef, n)
    Ms = Vector{Matrix{T}}(undef, n)
    Bs = Vector{Matrix{T}}(undef, n)
    ld = zero(T)
    for k in 1:n
        H = Hs[k]
        all(isfinite, H) || return nothing
        A = (H .+ H') ./ -2
        o, sl, sg = st.own[k], st.shl[k], st.shg[k]
        Aoo = A[o, o]
        iszero(jit) || (Aoo += jit * I)
        c = cholesky(Symmetric(Aoo), check = false)
        issuccess(c) || return nothing
        Bk = zeros(T, length(o), ns)
        Bk[:, sg] = A[o, sl]
        Mk = c \ Bk
        @views D[sg, sg] .+= A[sl, sl]
        D .-= Bk' * Mk
        chols[k], Ms[k], Bs[k] = c, Mk, Bk
        ld += 2 * sum(log, diag(c.U); init = zero(T))
    end
    iszero(jit) || (D += jit * I)
    cS = cholesky(Symmetric(D), check = false)
    issuccess(cS) || return nothing
    ld += 2 * sum(log, diag(cS.U))
    return _BAFactor(chols, Ms, Bs, cS, ld)
end

# Newton-step factorization: un-jittered first, then the adaptive jitter of the dense
# Newton loops on the whole diagonal, scaled by mean|diag(A)|.
function _ba_factor_jittered(
        st::_BAStructure, Hs::AbstractVector; jitter = 1.0e-6, max_tries = 8,
        growth = 10.0, scale_factor = 1.0e-6
    )
    F = _ba_factor(st, Hs)
    F === nothing || return F
    dsum = 0.0
    dsh = zeros(length(st.shared))
    for k in eachindex(Hs)
        H = Hs[k]
        for p in st.own[k]
            dsum += abs(H[p, p])
        end
        for (p, q) in zip(st.shl[k], st.shg[k])
            dsh[q] -= H[p, p]
        end
    end
    scale = (dsum + sum(abs, dsh)) / sum(length, st.own; init = length(st.shared))
    scale = isfinite(scale) ? scale : 1.0
    jit = max(jitter, scale_factor * scale)
    for _ in 1:max_tries
        F = _ba_factor(st, Hs, jit)
        F === nothing || return F
        jit *= growth
    end
    return nothing
end

# A \ R for a vector or matrix R in parent b coordinates.
function _ba_solve(st::_BAStructure, F::_BAFactor, R::AbstractVecOrMat)
    Rm = R isa AbstractVector ? reshape(R, :, 1) : R
    T = promote_type(eltype(Rm), eltype(F.logdet))
    X = zeros(T, size(Rm))
    rs = Matrix{T}(Rm[st.shared, :])
    ys = Vector{Matrix{T}}(undef, length(F.chols))
    for k in eachindex(F.chols)
        ys[k] = F.chols[k] \ Rm[st.ownglob[k], :]
        rs .-= F.B[k]' * ys[k]
    end
    xs = F.S \ rs
    X[st.shared, :] = xs
    for k in eachindex(F.chols)
        X[st.ownglob[k], :] = ys[k] .- F.M[k] * xs
    end
    return R isa AbstractVector ? vec(X) : X
end

# Block of A⁻¹ on individual k's sub-batch positions, from A⁻¹_ss = S⁻¹,
# A⁻¹_os = -M S⁻¹ and A⁻¹_oo = A_oo⁻¹ + M S⁻¹ Mᵀ.
function _ba_inverse_block(st::_BAStructure, F::_BAFactor, Sinv::AbstractMatrix, k::Int)
    o, sl, sg = st.own[k], st.shl[k], st.shg[k]
    m = length(st.maps[k])
    W = zeros(eltype(Sinv), m, m)
    MS = F.M[k] * Sinv
    W[o, o] = inv(F.chols[k]) .+ MS * F.M[k]'
    Wos = -MS[:, sg]
    W[o, sl] = Wos
    W[sl, o] = Wos'
    W[sl, sl] = Sinv[sg, sg]
    return W
end

# θ-context of sub-batch k: the parent's builder outputs (`pctx`, built once per θ before
# the per-individual loop, so a shared level is built once, not in every sub-batch) indexed
# by the sub-batch's levels. `nothing` when the parent context failed.
function _ba_sub_ctx(st::_BAStructure, pctx, k::Int)
    pctx === nothing && return nothing
    dists = [pctx.dists[ri][l] for (ri, l) in enumerate(st.lidx[k])]
    return _LaplaceThetaCtx(pctx.θ_re, dists, nothing)
end

# Per-individual θ-contexts for one inner solve at fixed θ; `pctx`: the parent's context.
function _ba_theta_ctxs(
        st::_BAStructure, curvature, dm::DataModel, θ, const_cache, cache, pool, b;
        pctx = _safe_theta_ctx(dm, st.info, θ, cache)
    )
    ctxs = Vector{Any}(undef, length(st.subs))
    _ba_foreach(length(st.subs), cache, pool) do k, c
        sub = st.subs[k]
        t = _ba_sub_ctx(st, pctx, k)
        ctxs[k] = _newton_curvature_ctx(curvature, t, dm, sub, const_cache, c, b[st.maps[k]])
    end
    return ctxs
end

function _ba_logf(st::_BAStructure, dm::DataModel, θ, b, const_cache, cache, pool, ctxs)
    n = length(st.subs)
    vals = Vector{Float64}(undef, n)
    _ba_foreach(n, cache, pool) do k, c
        vals[k] = _laplace_logf_batch(
            dm, st.subs[k], θ, b[st.maps[k]], const_cache, c; tctx = ctxs[k]
        )
    end
    fval = 0.0
    for k in 1:n
        fval += vals[k]
    end
    return fval
end

# log f, ∇_b log f and the per-individual curvature blocks at b (Float64 θ).
function _ba_grad_hess(
        st::_BAStructure, curvature, dm::DataModel, θ, b, const_cache, cache, pool, ctxs
    )
    n = length(st.subs)
    vals = Vector{Float64}(undef, n)
    gs = Vector{Vector{Float64}}(undef, n)
    Hs = Vector{Matrix{Float64}}(undef, n)
    _ba_foreach(n, cache, pool) do k, c
        sub = st.subs[k]
        bk = b[st.maps[k]]
        f = _LaplaceLogfBatch(dm, sub, θ, const_cache, c, ctxs[k])
        if curvature isa ExactHessianCurvature
            r = ForwardDiff.hessian!(ForwardDiff.DiffResults.HessianResult(bk), f, bk)
            vals[k] = ForwardDiff.DiffResults.value(r)
            gs[k] = copy(ForwardDiff.DiffResults.gradient(r))
            Hs[k] = copy(ForwardDiff.DiffResults.hessian(r))
        else
            vals[k] = f(bk)
            gk = similar(bk)
            Hk = _newton_grad_hess!(
                gk, curvature, f, ForwardDiff.GradientConfig(f, bk), dm, sub, θ, bk,
                const_cache, c, ctxs[k]
            )
            Hk === nothing && (
                Hk = _build_hess_b(
                    curvature, dm, sub, θ, bk, const_cache, c, nothing, 1;
                    ctx = "newton_inner", tctx = ctxs[k]
                )
            )
            gs[k], Hs[k] = gk, Hk
        end
    end
    fval = 0.0
    grad = zeros(length(b))
    for k in 1:n
        fval += vals[k]
        @views grad[st.maps[k]] .+= gs[k]
    end
    return fval, grad, Hs
end

# Damped Newton ascent of `_newton_exact_loop` / `_newton_inner_loop` on a structured
# batch: same tests, backtracking and fallback signal, with the step from the Schur solve.
function _ba_newton_solve(
        st::_BAStructure, dm::DataModel, θ::ComponentArray, const_cache, cache, pool,
        b0::Vector{Float64}, newton::NewtonInner; f0_min::Union{Nothing, Float64} = nothing,
        pctx = _safe_theta_ctx(dm, st.info, θ, cache)
    )
    curv = newton.curvature
    ctxs = _ba_theta_ctxs(st, curv, dm, θ, const_cache, cache, pool, b0; pctx = pctx)
    gh(b) = _ba_grad_hess(st, curv, dm, θ, b, const_cache, cache, pool, ctxs)
    b = copy(b0)
    fval, g, Hs = gh(b)
    f0_min === nothing || (isfinite(fval) && fval > f0_min) || return nothing
    isfinite(fval) || return _NewtonSol(b, -Inf, Inf, false)
    for _ in 1:(newton.maxiters)
        gn = maximum(abs, g)
        isfinite(gn) || return _NewtonSol(b, fval, Inf, false)
        gn <= newton.g_abstol && return _NewtonSol(b, fval, gn, true)
        F = _ba_factor_jittered(st, Hs)
        F === nothing && return _NewtonSol(b, fval, gn, false)
        Δ = _ba_solve(st, F, g)
        slope = dot(g, Δ)
        (isfinite(slope) && slope > 0) || return _NewtonSol(b, fval, gn, false)
        α = 1.0
        accepted = false
        Δmax = maximum(abs, Δ)
        while α >= newton.alpha_min
            α * Δmax <= newton.x_abstol && return _NewtonSol(b, fval, gn, true)
            b_try = b .+ α .* Δ
            if α == 1.0
                f_try, g_try, H_try = gh(b_try)
                gn_try = maximum(abs, g_try)
                # full step at the mode up to solver noise in log f (see `_newton_exact_loop`)
                isfinite(f_try) && f_try >= fval - 1.0e-8 * (1 + abs(fval)) &&
                    gn_try <= newton.g_abstol &&
                    return _NewtonSol(b_try, f_try, gn_try, true)
            else
                f_try = _ba_logf(st, dm, θ, b_try, const_cache, cache, pool, ctxs)
            end
            if isfinite(f_try) && f_try >= fval + 1.0e-4 * α * slope
                b = b_try
                if α == 1.0
                    fval, g, Hs = f_try, g_try, H_try
                else
                    fval, g, Hs = gh(b)
                end
                accepted = true
                break
            end
            α /= 2
        end
        accepted || return _NewtonSol(b, fval, gn, false)
    end
    gn = maximum(abs, g)
    return _NewtonSol(b, fval, isfinite(gn) ? gn : Inf, false)
end

# log f(b) and logdet(-H(b)) of a structured batch (Float64 θ); logdet = Inf when -H is
# not positive definite without jitter, logf = -Inf when log f is not finite.
function _ba_objective_batch(
        st::_BAStructure, dm::DataModel, θ::ComponentArray, b, const_cache, cache, pool,
        hmode
    )
    n = length(st.subs)
    vals = Vector{Float64}(undef, n)
    Hs = Vector{Matrix{Float64}}(undef, n)
    pctx = _safe_theta_ctx(dm, st.info, θ, cache)
    _ba_foreach(n, cache, pool) do k, c
        sub = st.subs[k]
        bk = b[st.maps[k]]
        tctx = _objective_theta_ctx(
            dm, sub, θ, const_cache, c, bk, hmode; tctx = _ba_sub_ctx(st, pctx, k)
        )
        vals[k] = _laplace_logf_batch(dm, sub, θ, bk, const_cache, c; tctx = tctx)
        isfinite(vals[k]) || return
        Hs[k] = _build_hess_b(
            hmode, dm, sub, θ, bk, const_cache, c, nothing, 1; ctx = "logdet", tctx = tctx
        )
    end
    logf = 0.0
    for k in 1:n
        isfinite(vals[k]) || return (logf = -Inf, logdet = Inf)
        logf += vals[k]
    end
    F = _ba_factor(st, Hs)
    return (logf = logf, logdet = F === nothing ? Inf : F.logdet)
end

# `_laplace_grad_batch` (one-pass trace-gradient branch) on a structured batch. Returns
# `nothing` when a sub-batch has no one-pass derivatives; the caller then takes the dense
# path.
function _ba_grad_batch(
        st::_BAStructure, dm::DataModel, θ::ComponentArray, b, const_cache, cache, pool,
        bi::Int; hmode::_HessMode = _ExactHess(), bstar_cache = nothing
    )
    n = length(st.subs)
    nb, nθ = length(b), length(θ)
    axs = getaxes(θ)
    split = _theta_split_obs_re(dm, θ)
    vals = Vector{Float64}(undef, n)
    Hs = Vector{Matrix{Float64}}(undef, n)
    gls = Vector{Vector{Float64}}(undef, n)
    ds = Vector{Union{Nothing, _OnePassDerivs}}(nothing, n)
    pctx = _safe_theta_ctx(dm, st.info, θ, cache)
    _ba_foreach(n, cache, pool) do k, c
        sub = st.subs[k]
        bk = b[st.maps[k]]
        tctx = _objective_theta_ctx(
            dm, sub, θ, const_cache, c, bk, hmode; tctx = _ba_sub_ctx(st, pctx, k)
        )
        vals[k] = _laplace_logf_batch(dm, sub, θ, bk, const_cache, c; tctx = tctx)
        isfinite(vals[k]) || return
        dk = _onepass_logdet_derivatives(
            hmode, dm, sub, θ, bk, const_cache, c, tctx; split = split
        )
        ds[k] = dk
        dk === nothing && return
        gls[k] = dk.grad_logf !== nothing ? dk.grad_logf : ForwardDiff.gradient(
                _LaplaceLogfTheta(dm, sub, bk, const_cache, c), θ
            )
        Hs[k] = hmode isa ExactHessianCurvature ? dk.Hex : _build_hess_b(
                hmode, dm, sub, θ, bk, const_cache, c, nothing, 1;
                ctx = "logdet", tctx = tctx
            )
    end
    fail = (logf = -Inf, logdet = Inf, grad = ComponentArray(zeros(nθ), axs))
    logf = 0.0
    for k in 1:n
        isfinite(vals[k]) || return fail
        ds[k] === nothing && return nothing
        logf += vals[k]
    end
    F = _ba_factor(st, Hs)
    F === nothing && return fail
    grad_logf = zeros(nθ)
    Gθ = zeros(nb, nθ)
    gld_θ = zeros(nθ)
    gld_b = zeros(nb)
    Sinv = inv(F.S)
    for k in 1:n
        d = ds[k]
        m = st.maps[k]
        grad_logf .+= gls[k]
        @views Gθ[m, :] .+= d.Gθ
        w = _vech_weights!(Vector{Float64}(undef, _ntri(length(m))), _ba_inverse_block(st, F, Sinv, k))
        gld_θ .-= d.Jθ' * w
        @views gld_b[m] .-= d.Jb' * w
    end
    # db*/dθ needs the exact Hessian (see `_laplace_grad_batch`)
    Fb = F
    if !(hmode isa ExactHessianCurvature)
        Fx = _ba_factor(st, [d.Hex for d in ds])
        Fx === nothing || (Fb = Fx)
    end
    dbdθ = _ba_solve(st, Fb, Gθ)
    _laplace_store_predictor!(bstar_cache, bi, θ, b, dbdθ)
    corr = vec(gld_b' * dbdθ)
    grad = grad_logf .- 0.5 .* (gld_θ .+ corr)
    return (logf = logf, logdet = F.logdet, grad = ComponentArray(grad, axs))
end
