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

# Complete-data log-likelihood of one batch as a function of x at the draw held in `b`.
# One instance per batch, so its ForwardDiff config is built once and reused.
struct _LouisLogf{D, M, C, L}
    dm::D
    info::REBatchInfo
    map::M
    const_cache::C
    cache::Base.RefValue{L}
    b::Vector{Float64}
end
(f::_LouisLogf)(x) = _laplace_logf_batch(f.dm, f.info, f.map(x), f.b, f.const_cache, f.cache[])

# Per-batch buffers of the current E-step moments. `fu` is `nothing` without
# likelihood_weights: the unweighted score is then the weighted one and Δu aliases Δw.
struct _LouisWork{FW, FU, HR, HC, GC}
    fw::Vector{FW}
    fu::FU
    hres::Vector{HR}
    hcfg::Vector{HC}
    gbuf::Vector{Vector{Float64}}
    gcfg::GC
    Δw::Matrix{Float64}
    Δu::Matrix{Float64}
    Sb::Array{Float64, 3}
    ok::Vector{Bool}
end

function _louis_work(dm::DataModel, batch_infos, map::_LouisThetaMap, const_cache, ll_cache, x::Vector{Float64})
    c1 = ll_cache isa Vector ? ll_cache[1] : ll_cache
    nb = length(batch_infos)
    p = length(x)
    mk(d, info) = _LouisLogf(d, info, map, const_cache, Ref(c1), zeros(get_n_b(info)))
    fw = [mk(dm, info) for info in batch_infos]
    hres = [ForwardDiff.DiffResults.HessianResult(x) for _ in 1:nb]
    hcfg = [ForwardDiff.HessianConfig(fw[bi], hres[bi], x) for bi in 1:nb]
    weighted = get_likelihood_weights(dm) !== nothing
    fu = weighted ? [mk(_sampling_dm(dm), info) for info in batch_infos] : nothing
    gcfg = weighted ? [ForwardDiff.GradientConfig(fu[bi], x) for bi in 1:nb] : nothing
    Δw = zeros(p, nb)
    return _LouisWork(
        fw, fu, hres, hcfg, [zeros(p) for _ in 1:nb], gcfg,
        Δw, weighted ? zeros(p, nb) : Δw, zeros(p, p, nb), fill(true, nb)
    )
end

# E[∇ℓ_w], E[∇ℓ_u] and E[∇²ℓ_w + ∇ℓ_w ∇ℓ_uᵀ] of batch bi over its draws (weights `ws`,
# normalized; `nothing` = uniform). False when any draw is non-finite.
function _louis_batch!(W::_LouisWork, bi::Int, x::Vector{Float64}, draws, ws, cache)
    p = length(x)
    fw = W.fw[bi]
    fw.cache[] = cache
    Δw = view(W.Δw, :, bi)
    Δu = view(W.Δu, :, bi)
    S = view(W.Sb, :, :, bi)
    fill!(Δw, 0.0)
    fill!(Δu, 0.0)
    fill!(S, 0.0)
    n = length(draws)
    for (s, b) in enumerate(draws)
        copyto!(fw.b, b)
        r = ForwardDiff.hessian!(W.hres[bi], fw, x, W.hcfg[bi])
        g = ForwardDiff.DiffResults.gradient(r)
        H = ForwardDiff.DiffResults.hessian(r)
        (isfinite(ForwardDiff.DiffResults.value(r)) && all(isfinite, g) && all(isfinite, H)) ||
            return false
        gu = g
        if W.fu !== nothing
            fu = W.fu[bi]
            fu.cache[] = cache
            copyto!(fu.b, b)
            gu = ForwardDiff.gradient!(W.gbuf[bi], fu, x, W.gcfg[bi])
            all(isfinite, gu) || return false
        end
        w = ws === nothing ? 1.0 / n : Float64(ws[s])
        @inbounds for j in 1:p
            W.fu === nothing || (Δu[j] += w * gu[j])
            Δw[j] += w * g[j]
            for i in 1:p
                S[i, j] += w * (H[i, j] + g[i] * gu[j])
            end
        end
    end
    return true
end

# Moments of every batch at x; `draws_of(bi)` / `ws_of(bi)` give a batch's draws and
# weights. Returns Σ_b w_b S_b, or `nothing` when a batch is non-finite.
function _louis_eval!(
        W::_LouisWork, dm::DataModel, batch_infos, x::Vector{Float64},
        draws_of::F, ws_of::G, ll_cache, serialization
    ) where {F, G}
    _saem_foreach_batch(dm, ll_cache, length(batch_infos), serialization) do bi, cache
        info = batch_infos[bi]
        if iszero(_batch_weight(dm, get_inds(info)))
            W.Δw[:, bi] .= 0.0
            W.Δu[:, bi] .= 0.0
            W.Sb[:, :, bi] .= 0.0
            W.ok[bi] = true
        else
            draws = get_n_b(info) == 0 ? (Float64[],) : draws_of(bi)
            W.ok[bi] = _louis_batch!(W, bi, x, draws, get_n_b(info) == 0 ? nothing : ws_of(bi), cache)
        end
    end
    all(W.ok) || return nothing
    S = zeros(size(W.Sb, 1), size(W.Sb, 2))
    for (bi, info) in enumerate(batch_infos)
        S .+= _batch_weight(dm, get_inds(info)) .* view(W.Sb, :, :, bi)
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
        cache = cache, W = W, S = zeros(p, p), Δw = Δw,
        Δu = W.Δu === W.Δw ? Δw : zeros(p, nb), n = Ref(0), nfail = Ref(0),
    )
end

# Equal-weight running mean over the E-steps passed in (Polyak averaging, see SAEM docs).
function _louis_average!(st, dm::DataModel, batch_infos, x::Vector{Float64}, draws_of, ws_of, serialization)
    S = _louis_eval!(st.W, dm, batch_infos, x, draws_of, ws_of, st.cache, serialization)
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
    lcp = deepcopy(last_chain_params)
    bch = deepcopy(b_chains)
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
        batch_rngs, re_names, re_types, last_params, proposal_blocks, n_draws::Int, iter::Int
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
            thin = _mcem_thin_of(es)
            tk = merge(es.turing_kwargs, (n_samples = n_draws * thin,))
            s, _, _ = _mcem_sample_batch(
                dm_s, info, θu, const_cache, cache, es.sampler, tk, rng, re_names,
                es.warm_start, deepcopy(last_params[bi]); outer_iter = iter
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
