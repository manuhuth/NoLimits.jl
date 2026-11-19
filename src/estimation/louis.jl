# Louis (1982) observed information from E-step draws (#441). The information is formed
# on the Wald coordinates (free fixed effects, transformed scale) so compute_uq can use it
# in place of the FOCEI/Laplace marginal Hessian.

# x (free, transformed) -> natural-scale θ, the same map as the Wald path.
struct _LouisThetaMap{AF, TC, AX, IT}
    axs_free::AF
    θ_const_t::TC
    axs_full::AX
    free_names::Vector{Symbol}
    inv_transform::IT
end
(m::_LouisThetaMap)(x) = _theta_u_from_free_t(
    x, m.axs_free, m.θ_const_t, m.axs_full, m.free_names, m.inv_transform
)

# The complete-data log-likelihood of a batch splits into the RE prior log p(b | θ), cheap and
# free of the model, and the observation term log p(y | b, θ). Each part is differentiated
# per draw in the natural parameters it depends on: parameters that enter only the RE
# distributions (the SAEM Q2 set: mu-referenced means, Ω) never reach the model. The chain
# rule to the Wald coordinates x is applied once per E-step to the averaged moments:
# ∇_x = Jᵀ∇_θ and ∇²_x = Jᵀ∇²_θ J + Σ_k ∂ℓ/∂θ_k ∇²_x θ_k, with J = ∂θ/∂x.

# One part of the batch log-likelihood as a function of the natural coordinates pos of θ.
struct _LouisPartLogf{F, D, C, L, AX}
    fun::F
    dm::D
    info::REBatchInfo
    const_cache::C
    cache::Base.RefValue{L}
    b::Vector{Float64}
    u::Vector{Float64}
    pos::Vector{Int}
    axs::AX
end
function (f::_LouisPartLogf)(z)
    uu = eltype(z).(f.u)
    uu[f.pos] .= z
    return f.fun(f.dm, f.info, ComponentArray(uu, f.axs), f.b, f.const_cache, f.cache[])
end

# Per-batch functors, ForwardDiff configs and buffers (one set per batch, so configs are
# built once and threads never share one). Natural free coordinates F (length q); the RE
# part uses loc_re ⊂ 1:q, the observation part loc_obs ⊂ 1:q (each nothing when empty). fu is
# the unweighted observation part, nothing without likelihood_weights.
struct _LouisWork{FR, RR, RC, FO, OR, OC, FU, GC}
    posF::Vector{Int}
    loc_re::Vector{Int}
    loc_obs::Vector{Int}
    fr::FR
    rres::RR
    rcfg::RC
    fo::FO
    ores::OR
    ocfg::OC
    fu::FU
    gcfg::GC
    gbuf::Vector{Vector{Float64}}
    gw::Vector{Vector{Float64}}
    gu::Vector{Vector{Float64}}
    Hw::Vector{Matrix{Float64}}
    Δwn::Matrix{Float64}
    Δun::Matrix{Float64}
    Sbn::Array{Float64, 3}
    Δw::Matrix{Float64}
    Δu::Matrix{Float64}
    ok::Vector{Bool}
end

# natural flat positions of the given fixed effects
function _louis_positions(u::ComponentArray, names)
    ix = ComponentArray(collect(1.0:length(u)), getaxes(u))
    return Int[Int(i) for n in names for i in vec(collect(getproperty(ix, n)))]
end

function _louis_work(dm::DataModel, batch_infos, map::_LouisThetaMap, const_cache, ll_cache, x::Vector{Float64})
    c1 = ll_cache isa Vector ? ll_cache[1] : ll_cache
    nb, p = length(batch_infos), length(x)
    model = get_model(dm)
    θu = map(x)
    u, axs = Vector{Float64}(collect(θu)), getaxes(θu)
    names = map.free_names
    re_syms = reduce(union, values(get_re_syms(get_random(model))); init = Set{Symbol}())
    p1 = _partition_q1_q2_names(model, names).q1
    posF = _louis_positions(θu, names)
    loc(ns) = [findfirst(==(k), posF) for k in _louis_positions(θu, ns)]
    loc_re, loc_obs = loc([n for n in names if n in re_syms]), loc(p1)
    q = length(posF)
    mk(fun, d, info, l) = _LouisPartLogf(fun, d, info, const_cache, Ref(c1), zeros(get_n_b(info)), copy(u), posF[l], axs)
    zr, zo = zeros(length(loc_re)), zeros(length(loc_obs))
    has_re = !isempty(loc_re)
    fr = has_re ? [mk(_re_logpdf_batch, dm, info, loc_re) for info in batch_infos] : nothing
    rres = has_re ? [ForwardDiff.DiffResults.HessianResult(zr) for _ in 1:nb] : nothing
    rcfg = has_re ? [ForwardDiff.HessianConfig(fr[bi], rres[bi], zr) for bi in 1:nb] : nothing
    has_obs = !isempty(loc_obs)
    fo = has_obs ? [mk(_mcem_obs_logf_batch, dm, info, loc_obs) for info in batch_infos] : nothing
    ores = has_obs ? [ForwardDiff.DiffResults.HessianResult(zo) for _ in 1:nb] : nothing
    ocfg = has_obs ? [ForwardDiff.HessianConfig(fo[bi], ores[bi], zo) for bi in 1:nb] : nothing
    weighted = get_likelihood_weights(dm) !== nothing && has_obs
    fu = weighted ? [mk(_mcem_obs_logf_batch, _sampling_dm(dm), info, loc_obs) for info in batch_infos] : nothing
    gcfg = weighted ? [ForwardDiff.GradientConfig(fu[bi], zo) for bi in 1:nb] : nothing
    gw = [zeros(q) for _ in 1:nb]
    Δwn, Δw = zeros(q, nb), zeros(p, nb)
    return _LouisWork(
        posF, loc_re, loc_obs, fr, rres, rcfg, fo, ores, ocfg, fu, gcfg,
        [zeros(length(loc_obs)) for _ in 1:nb], gw, weighted ? [zeros(q) for _ in 1:nb] : gw,
        [zeros(q, q) for _ in 1:nb], Δwn, weighted ? zeros(q, nb) : Δwn, zeros(q, q, nb),
        Δw, weighted ? zeros(p, nb) : Δw, fill(true, nb)
    )
end

@inline function _louis_add!(g, H, loc, r)
    gp, Hp = ForwardDiff.DiffResults.gradient(r), ForwardDiff.DiffResults.hessian(r)
    @inbounds for (a, i) in enumerate(loc)
        g[i] += gp[a]
        for (c, j) in enumerate(loc)
            H[j, i] += Hp[c, a]
        end
    end
    return nothing
end

# Natural-coordinate moments of batch bi over its draws (weights ws normalized; nothing =
# uniform): E[∇ℓ_w], E[∇ℓ_u], E[∇²ℓ_w + ∇ℓ_w ∇ℓ_uᵀ]. False when any draw is non-finite.
function _louis_batch!(W::_LouisWork, bi::Int, u::Vector{Float64}, draws, ws, cache)
    q = length(W.posF)
    weighted = W.fu !== nothing
    parts = tuple(
        (W.fr === nothing ? () : (W.fr[bi],))..., (W.fo === nothing ? () : (W.fo[bi],))...,
        (weighted ? (W.fu[bi],) : ())...
    )
    for f in parts
        f.cache[] = cache
        copyto!(f.u, u)
    end
    zr, zo = u[W.posF[W.loc_re]], u[W.posF[W.loc_obs]]
    Δw = view(W.Δwn, :, bi)
    Δu = view(W.Δun, :, bi)
    S = view(W.Sbn, :, :, bi)
    fill!(Δw, 0.0)
    fill!(Δu, 0.0)
    fill!(S, 0.0)
    g, gu, H = W.gw[bi], W.gu[bi], W.Hw[bi]
    n = length(draws)
    for (s, b) in enumerate(draws)
        fill!(g, 0.0)
        fill!(H, 0.0)
        for f in parts
            copyto!(f.b, b)
        end
        if W.fr !== nothing
            r = ForwardDiff.hessian!(W.rres[bi], W.fr[bi], zr, W.rcfg[bi])
            isfinite(ForwardDiff.DiffResults.value(r)) || return false
            _louis_add!(g, H, W.loc_re, r)
        end
        if W.fo !== nothing
            ro = ForwardDiff.hessian!(W.ores[bi], W.fo[bi], zo, W.ocfg[bi])
            isfinite(ForwardDiff.DiffResults.value(ro)) || return false
            if weighted
                gu .= g
                gou = ForwardDiff.gradient!(W.gbuf[bi], W.fu[bi], zo, W.gcfg[bi])
                @inbounds for (a, i) in enumerate(W.loc_obs)
                    gu[i] += gou[a]
                end
            end
            _louis_add!(g, H, W.loc_obs, ro)
        end
        (all(isfinite, g) && all(isfinite, H) && all(isfinite, gu)) || return false
        w = ws === nothing ? 1.0 / n : Float64(ws[s])
        @inbounds for j in 1:q
            weighted && (Δu[j] += w * gu[j])
            Δw[j] += w * g[j]
            for i in 1:q
                S[i, j] += w * (H[i, j] + g[i] * gu[j])
            end
        end
    end
    return true
end

# Moments of every batch at x; draws_of(bi) / ws_of(bi) give a batch's draws and weights.
# Returns Σ_b w_b E_b[∇²ℓ_w + ∇ℓ_w∇ℓ_uᵀ] in x and fills W.Δw / W.Δu, or nothing when a
# batch is non-finite.
function _louis_eval!(
        W::_LouisWork, dm::DataModel, batch_infos, map::_LouisThetaMap, x::Vector{Float64},
        draws_of::F, ws_of::G, ll_cache, serialization
    ) where {F, G}
    u = Vector{Float64}(collect(map(x)))
    _saem_foreach_batch(dm, ll_cache, length(batch_infos), serialization) do bi, cache
        info = batch_infos[bi]
        if iszero(_batch_weight(dm, get_inds(info)))
            W.Δwn[:, bi] .= 0.0
            W.Δun[:, bi] .= 0.0
            W.Sbn[:, :, bi] .= 0.0
            W.ok[bi] = true
        else
            draws = get_n_b(info) == 0 ? (Float64[],) : draws_of(bi)
            W.ok[bi] = _louis_batch!(W, bi, u, draws, get_n_b(info) == 0 ? nothing : ws_of(bi), cache)
        end
    end
    all(W.ok) || return nothing
    q, p = length(W.posF), length(x)
    Sn, dbar = zeros(q, q), zeros(q)
    for (bi, info) in enumerate(batch_infos)
        wb = _batch_weight(dm, get_inds(info))
        Sn .+= wb .* view(W.Sbn, :, :, bi)
        dbar .+= wb .* view(W.Δwn, :, bi)
    end
    θF(y) = collect(map(y))[W.posF]
    J = ForwardDiff.jacobian(θF, x)
    T = reshape(ForwardDiff.jacobian(y -> vec(ForwardDiff.jacobian(θF, y)), x), q, p, p)
    mul!(W.Δw, J', W.Δwn)
    W.Δu === W.Δw || mul!(W.Δu, J', W.Δun)
    S = J' * Sn * J
    for k in 1:q
        S .+= dbar[k] .* view(T, k, :, :)
    end
    return S
end

# A = -Σ w_b E[∇²ℓ_w + ∇ℓ_w∇ℓ_uᵀ] + Σ w_b Δw_b Δu_bᵀ (the Louis information when v ≡ 1),
# B = Σ w_b² Δw_b Δw_bᵀ (sandwich meat, same convention as the Wald sandwich).
function _louis_matrices(S::AbstractMatrix, Δw::AbstractMatrix, Δu::AbstractMatrix, dm::DataModel, batch_infos)
    A = -Matrix{Float64}(S)
    B = zeros(size(A))
    for (bi, info) in enumerate(batch_infos)
        wb = _batch_weight(dm, get_inds(info))
        iszero(wb) && continue
        dw = view(Δw, :, bi)
        A .+= wb .* (dw * view(Δu, :, bi)')
        B .+= wb^2 .* (dw * dw')
    end
    return A, B
end

# louis keyword -> :posterior, :online or :off.
function _louis_mode(label::AbstractString, louis, louis_draws::Int)
    louis === false && return :off
    (louis isa Symbol && louis in (:posterior, :online)) ||
        throw(ArgumentError("$label: louis must be :posterior, :online or false. Got: $(repr(louis))"))
    louis === :posterior && louis_draws < 1 &&
        throw(ArgumentError("$label: louis_draws must be ≥ 1 for louis = :posterior. Got: $louis_draws"))
    return louis
end

_louis_note(A, B, coords, n, status::Symbol) = (; A = A, B = B, coords = coords, n = n, status = status)
_louis_note(status::Symbol) = (; A = nothing, B = nothing, coords = nothing, n = 0, status = status)

# Fewest averaged SAEM iterations for the Louis information; below it compute_uq falls
# back to the FOCEI/Laplace marginal Hessian.
const _LOUIS_MIN_ITERS = 20

# Louis state of one fit: its own likelihood cache (the fit's caches are never touched),
# work buffers and the running averages of S and the per-batch scores.
function _louis_init(
        dm::DataModel, batch_infos, map::_LouisThetaMap, const_cache, x::Vector{Float64};
        ode_args, ode_kwargs, serialization
    )
    cache = build_ll_cache(
        dm; ode_args = ode_args, ode_kwargs = ode_kwargs,
        serialization = serialization, force_saveat = true
    )
    W = _louis_work(dm, batch_infos, map, const_cache, cache, x)
    p, nb = length(x), length(batch_infos)
    Δw = zeros(p, nb)
    return (;
        cache = cache, W = W, map = map, S = zeros(p, p), Δw = Δw,
        Δu = W.Δu === W.Δw ? Δw : zeros(p, nb), n = Ref(0), nfail = Ref(0),
    )
end

# Equal-weight running mean over the E-steps passed in (Polyak averaging, see SAEM docs).
function _louis_average!(st, dm::DataModel, batch_infos, x::Vector{Float64}, draws_of, ws_of, serialization)
    S = _louis_eval!(st.W, dm, batch_infos, st.map, x, draws_of, ws_of, st.cache, serialization)
    S === nothing && (st.nfail[] += 1; return st)
    γ = 1.0 / (st.n[] += 1)
    st.S .+= γ .* (S .- st.S)
    st.Δw .+= γ .* (st.W.Δw .- st.Δw)
    st.Δu === st.Δw || (st.Δu .+= γ .* (st.W.Δu .- st.Δu))
    return st
end

function _louis_finish(st, dm::DataModel, batch_infos, coords::Vector{Symbol}; min_n::Int = 1)
    if st.n[] < min_n
        return _louis_note(st.n[] == 0 && st.nfail[] > 0 ? :nonfinite : :too_few_iterations)
    end
    A, B = _louis_matrices(st.S, st.Δw, st.Δu, dm, batch_infos)
    (all(isfinite, A) && all(isfinite, B)) || return _louis_note(:nonfinite)
    return _louis_note(A, B, coords, st.n[], :ok)
end

# SAEM post-fit Louis phase (as nlmixr2 covMethod = "sa"): θ frozen at θ̂, copies of the
# last chains and per-batch RNGs continue for n_iter E-steps, each averaged with equal
# weight. The fit's chains, RNG streams and caches are untouched.
function _saem_louis_postfit!(
        st, dm::DataModel, dm_s::DataModel, batch_infos, θu, x::Vector{Float64}, const_cache,
        sampler, tkwargs, batch_rngs, re_names, warm_start, last_chain_params, b_chains,
        nc::Int, serialization, n_iter::Int, iter::Int
    )
    rngs = [copy(r) for r in batch_rngs]
    bch = deepcopy(b_chains)
    lcp = if sampler isa SaemixMH
        deepcopy(last_chain_params)
    else
        # a Turing sampler pays a full sample() setup per call; the native kernel continues
        # the chains from their last draws instead
        nbmax = maximum((get_n_b(info) for info in batch_infos); init = 1)
        sampler, tkwargs = SaemixMH(), (n_samples = max(1, cld(nbmax, 3)),)
        [Any[Vector{Float64}(bch[bi][c]) for c in 1:nc] for bi in eachindex(bch)]
    end
    all_b = collect(1:length(batch_infos))
    for k in 1:n_iter
        _saem_sample_batches!(
            dm_s, batch_infos, all_b, θu, const_cache, st.cache, sampler, tkwargs, rngs,
            re_names, warm_start, lcp, bch, nc, serialization; outer_iter = iter + k
        )
        _louis_average!(st, dm, batch_infos, x, bi -> bch[bi], _ -> nothing, serialization)
    end
    return st
end

# MCEM: the final draws target p(η | y, θ_{K-1}); self-normalized weights move them to θ̂.
function _louis_reweight(
        dm_s::DataModel, batch_infos, θ_from, θ_to, samples_by_batch, ws_by_batch,
        const_cache, ll_cache, serialization
    )
    out = Vector{Vector{Float64}}(undef, length(batch_infos))
    _saem_foreach_batch(dm_s, ll_cache, length(batch_infos), serialization) do bi, cache
        info = batch_infos[bi]
        smp = samples_by_batch[bi]
        if get_n_b(info) == 0 || size(smp, 2) == 0
            out[bi] = Float64[]
            return
        end
        ws = ws_by_batch === nothing ? nothing : ws_by_batch[bi]
        lw = [
            (ws === nothing ? 0.0 : log(ws[s])) +
                _laplace_logf_batch(dm_s, info, θ_to, view(smp, :, s), const_cache, cache) -
                _laplace_logf_batch(dm_s, info, θ_from, view(smp, :, s), const_cache, cache)
                for s in axes(smp, 2)
        ]
        w = exp.(lw .- maximum(lw))
        out[bi] = w ./ sum(w)
    end
    return out
end

# MCEM louis_draws: fresh draws at θ̂ from copies of the per-batch RNGs and sampler state.
function _mcem_louis_draws(
        dm_s::DataModel, batch_infos, θu, const_cache, cache, e_step, use_mcmc::Bool,
        batch_rngs, re_names, re_types, last_params, proposal_blocks, samples_by_batch,
        n_draws::Int, iter::Int
    )
    nb = length(batch_infos)
    smp = Vector{Matrix{Float64}}(undef, nb)
    ws = use_mcmc ? nothing : Vector{Vector{Float64}}(undef, nb)
    for bi in 1:nb
        info = batch_infos[bi]
        rng = copy(batch_rngs[bi])
        if get_n_b(info) == 0
            smp[bi] = zeros(0, 0)
            ws === nothing || (ws[bi] = Float64[])
        elseif use_mcmc
            es = _mcmc_e_step(e_step)
            # Turing samplers: the native kernel continues from the last fit draw (see SAEM)
            native = es.sampler isa SaemixMH
            thin = native ? _mcem_thin_of(es) : 1
            tk = native ? merge(es.turing_kwargs, (n_samples = n_draws * thin,)) : (n_samples = n_draws,)
            s, _, _ = _mcem_sample_batch(
                dm_s, info, θu, const_cache, cache, native ? es.sampler : SaemixMH(), tk, rng,
                re_names, true, native ? deepcopy(last_params[bi]) : Vector{Float64}(samples_by_batch[bi][:, end]);
                outer_iter = iter
            )
            smp[bi] = _mcem_thin(s, thin)
        else
            s, lw, _ = _is_sample_batch(
                dm_s, info, θu, const_cache, cache, rng, re_names, re_types, e_step,
                deepcopy(proposal_blocks[bi]); n_samples = n_draws
            )
            smp[bi] = s
            w = exp.(lw .- maximum(lw))
            ws[bi] = w ./ sum(w)
        end
    end
    return smp, ws
end

# Stored Louis information of an SAEM/MCEM fit, or `nothing` for the marginal-Hessian path:
# an explicit re_approx / re_approx_method or changed constants select that path.
function _louis_for_uq(res, source_method, re_approx, re_approx_method, constants, constants_re, coords)
    source_method isa Union{SAEM, MCEM} || return nothing
    notes = get_notes(res)
    (notes isa NamedTuple && hasproperty(notes, :louis)) || return nothing
    lou = notes.louis
    (re_approx === :auto && re_approx_method === nothing) || return nothing
    (constants === nothing && constants_re === nothing) || return nothing
    if lou.status !== :ok
        lou.status === :off ||
            @warn "Louis information unavailable for this fit ($(lou.status)); Wald UQ uses the FOCEI/Laplace marginal Hessian instead." status = lou.status
        return nothing
    end
    lou.coords == coords || return nothing
    return lou
end
