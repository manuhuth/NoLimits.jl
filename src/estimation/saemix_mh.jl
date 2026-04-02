export SaemixMH

using Distributions
using LinearAlgebra
using Random

# ---------------------------------------------------------------------------
# Public sampler type
# ---------------------------------------------------------------------------

"""
    SaemixMH(; n_kern1=2, n_kern2=2, target_accept=0.44, adapt_rate=0.7)

Saemix-style lightweight MH sampler for the SAEM/MCEM E-step.

Implements the two main MCMC kernels used by the saemix R package:

- **Kernel 1** (`n_kern1` steps): independent proposal from the current RE prior
  `p(η|θ)`.  Acceptance uses only the *likelihood ratio* — no prior term because
  the proposal equals the prior.  Efficient when the posterior is close to the
  prior (early SAEM iterations or weakly informative data).

- **Kernel 2** (`n_kern2` steps): per-level coordinate-wise random-walk in the
  natural parameter space.  Acceptance uses the full log-joint ratio.  Scale is
  adapted via Robbins-Monro to reach `target_accept`.

Unlike `MH()` and `AdaptiveNoLimitsMH()`, `SaemixMH` does **not** go through
Turing.jl.  It operates directly on the flat b-vector using `_loglikelihood_individual`,
eliminating interpreter and compilation overhead.

# Keyword Arguments

- `n_kern1::Int = 2`: prior-proposal steps per E-step call.
- `n_kern2::Int = 2`: per-level RW steps per E-step call.
- `target_accept::Float64 = 0.44`: target acceptance rate for kernel-2 adaptation.
- `adapt_rate::Float64 = 0.7`: Robbins-Monro exponent for kernel-2 scale updates.

# Example

```julia
res = fit_model(dm, SAEM(;
    sampler    = SaemixMH(),
    mcmc_steps = 1,          # 1 sweep = n_kern1 + n_kern2 total MH steps
    maxiters   = 300,
    builtin_stats = :closed_form,
    re_cov_params = (; η = :τ)
))
```
"""
struct SaemixMH
    n_kern1       :: Int
    n_kern2       :: Int
    target_accept :: Float64
    adapt_rate    :: Float64
end

function SaemixMH(; n_kern1::Int=2, n_kern2::Int=2,
                    target_accept::Float64=0.44,
                    adapt_rate::Float64=0.7)
    return SaemixMH(n_kern1, n_kern2, target_accept, adapt_rate)
end

# ---------------------------------------------------------------------------
# Per-level pre-computed metadata
# ---------------------------------------------------------------------------

# One entry per free RE level in the batch.  Built once during init and reused
# across SAEM iterations.
struct _SaemixMHLevel
    ri        :: Int              # RE group index (1-based into re_names)
    li        :: Int              # level index within that RE group
    range     :: UnitRange{Int}   # slice of flat b for this level
    rep_ind   :: Int              # representative individual index (const_cov access)
    dim       :: Int              # dimensionality (1 for scalar RE)
    group_pos :: Vector{Int}      # positions into batch_info.inds that use this level
end

# ---------------------------------------------------------------------------
# State (preserved across SAEM iterations)
# ---------------------------------------------------------------------------

mutable struct _SaemixMHState
    b          :: Vector{Float64}         # current flat b (all free RE levels)
    b_prop     :: Vector{Float64}         # preallocated proposal buffer (same length as b)
    indiv_ll   :: Vector{Float64}         # obs log-lik per individual (indexed by pos in inds)
    level_plp  :: Vector{Float64}         # prior log-pdf per free level
    levels     :: Vector{_SaemixMHLevel}  # pre-built level metadata
    dom        :: Vector{Float64}         # per-(level,dim) adaptive RW scale for kernel-2
    n_accept2  :: Vector{Int}             # per-(level,dim) kernel-2 accepts
    n_total2   :: Vector{Int}             # per-(level,dim) kernel-2 totals
    n_accept   :: Int
    n_total    :: Int
end

# ---------------------------------------------------------------------------
# Init helpers
# ---------------------------------------------------------------------------

# Draw an initial b from the RE priors.
function _saemixmh_init_b!(b::Vector{Float64},
                            levels::Vector{_SaemixMHLevel},
                            dm::DataModel,
                            θ::ComponentArray,
                            const_cache::LaplaceConstantsCache,
                            cache::_LLCache,
                            rng::AbstractRNG,
                            re_names::Vector{Symbol})
    θ_re = _symmetrize_psd_params(θ, dm.model.fixed.fixed)
    dists_builder = get_create_random_effect_distribution(dm.model.random.random)
    model_funs = cache.model_funs
    helpers    = cache.helpers
    for lv in levels
        re    = re_names[lv.ri]
        const_cov = dm.individuals[lv.rep_ind].const_cov
        dists = dists_builder(θ_re, const_cov, model_funs, helpers)
        dist  = getproperty(dists, re)
        rng_draw = rand(rng, dist)
        r = lv.range
        if lv.dim == 1
            b[first(r)] = Float64(rng_draw isa AbstractVector ? rng_draw[1] : rng_draw)
        else
            for k in 1:lv.dim
                b[r[k]] = Float64(rng_draw isa AbstractVector ? rng_draw[k] : rng_draw)
            end
        end
    end
    return b
end

# Compute initial per-individual obs log-likelihood.
function _saemixmh_init_indiv_ll!(indiv_ll::Vector{Float64},
                                   dm::DataModel,
                                   batch_info::_LaplaceBatchInfo,
                                   θ::ComponentArray,
                                   const_cache::LaplaceConstantsCache,
                                   cache::_LLCache,
                                   b::Vector{Float64})
    θ_re = _symmetrize_psd_params(θ, dm.model.fixed.fixed)
    for (pos, i) in enumerate(batch_info.inds)
        η_ind = _build_eta_ind(dm, i, batch_info, b, const_cache, θ_re)
        indiv_ll[pos] = _loglikelihood_individual(dm, i, θ_re, η_ind, cache)
    end
    return indiv_ll
end

function _saemixmh_level_plp!(level_plp::Vector{Float64},
                               levels::Vector{_SaemixMHLevel},
                               dm::DataModel,
                               θ::ComponentArray,
                               const_cache::LaplaceConstantsCache,
                               cache::_LLCache,
                               b::Vector{Float64};
                               anneal_sds::NamedTuple=NamedTuple())
    θ_re = _symmetrize_psd_params(θ, dm.model.fixed.fixed)
    dists_builder = get_create_random_effect_distribution(dm.model.random.random)
    model_funs = cache.model_funs
    helpers    = cache.helpers
    has_anneal = !isempty(anneal_sds)
    re_names   = dm.re_group_info.laplace_cache.re_names
    for (k, lv) in enumerate(levels)
        re    = re_names[lv.ri]
        const_cov = dm.individuals[lv.rep_ind].const_cov
        dists = dists_builder(θ_re, const_cov, model_funs, helpers)
        dist  = getproperty(dists, re)
        if has_anneal && haskey(anneal_sds, re)
            dist = Normal(mean(dist), getfield(anneal_sds, re))
        end
        r = lv.range
        if lv.dim == 1
            level_plp[k] = logpdf(dist, b[first(r)])
        else
            level_plp[k] = logpdf(dist, view(b, r))
        end
    end
    return level_plp
end

# Build the _SaemixMHLevel list from batch_info.
function _saemixmh_build_levels(dm::DataModel,
                                 batch_info::_LaplaceBatchInfo,
                                 re_names::Vector{Symbol})
    cache = dm.re_group_info.laplace_cache
    levels = _SaemixMHLevel[]
    for (ri, re) in enumerate(re_names)
        info = batch_info.re_info[ri]
        isempty(info.map.levels) && continue
        for li in eachindex(info.map.levels)
            l_id   = info.map.levels[li]
            range  = info.ranges[li]
            rep    = info.reps[li]
            dim    = info.dim

            # which individuals (by position in batch_info.inds) use this level?
            group_pos = Int[]
            for (pos, i) in enumerate(batch_info.inds)
                ids_for_i = cache.ind_level_ids[i][ri]
                if l_id in ids_for_i
                    push!(group_pos, pos)
                end
            end

            push!(levels, _SaemixMHLevel(ri, li, range, rep, dim, group_pos))
        end
    end
    return levels
end

# Build initial adaptive RW scales for kernel-2.
# Use sqrt(variance of prior) per dimension, scaled by 2.38/sqrt(d).
function _saemixmh_init_dom(levels::Vector{_SaemixMHLevel},
                              dm::DataModel,
                              θ::ComponentArray,
                              const_cache::LaplaceConstantsCache,
                              cache::_LLCache,
                              re_names::Vector{Symbol})
    θ_re = _symmetrize_psd_params(θ, dm.model.fixed.fixed)
    dists_builder = get_create_random_effect_distribution(dm.model.random.random)
    model_funs = cache.model_funs
    helpers    = cache.helpers
    n_dims = sum(lv.dim for lv in levels; init=0)
    dom = ones(Float64, n_dims)
    offset = 0
    for lv in levels
        re    = re_names[lv.ri]
        const_cov = dm.individuals[lv.rep_ind].const_cov
        dists = dists_builder(θ_re, const_cov, model_funs, helpers)
        dist  = getproperty(dists, re)
        d = lv.dim
        scale_factor = 2.38 / sqrt(Float64(d))
        if d == 1
            v = var(dist)
            dom[offset + 1] = isfinite(v) && v > 0 ? scale_factor * sqrt(v) : 1.0
        else
            V = cov(dist)
            for k in 1:d
                v = V[k, k]
                dom[offset + k] = isfinite(v) && v > 0 ? scale_factor * sqrt(v) : 1.0
            end
        end
        offset += d
    end
    return dom
end

function _saemixmh_init_state(dm::DataModel,
                               batch_info::_LaplaceBatchInfo,
                               θ::ComponentArray,
                               const_cache::LaplaceConstantsCache,
                               cache::_LLCache,
                               rng::AbstractRNG,
                               re_names::Vector{Symbol})
    nb     = batch_info.n_b
    n_inds = length(batch_info.inds)
    levels = _saemixmh_build_levels(dm, batch_info, re_names)
    b      = zeros(Float64, nb)
    _saemixmh_init_b!(b, levels, dm, θ, const_cache, cache, rng, re_names)
    indiv_ll  = Vector{Float64}(undef, n_inds)
    _saemixmh_init_indiv_ll!(indiv_ll, dm, batch_info, θ, const_cache, cache, b)
    level_plp = Vector{Float64}(undef, length(levels))
    _saemixmh_level_plp!(level_plp, levels, dm, θ, const_cache, cache, b)
    dom = _saemixmh_init_dom(levels, dm, θ, const_cache, cache, re_names)
    n_dims = length(dom)
    return _SaemixMHState(b, similar(b), indiv_ll, level_plp, levels,
                           dom, zeros(Int, n_dims), zeros(Int, n_dims),
                           0, 0)
end

# ---------------------------------------------------------------------------
# Kernel 1: prior-proposal MH (likelihood-ratio acceptance)
# ---------------------------------------------------------------------------

function _saemixmh_kern1!(state::_SaemixMHState,
                           dm::DataModel,
                           batch_info::_LaplaceBatchInfo,
                           θ::ComponentArray,
                           const_cache::LaplaceConstantsCache,
                           cache::_LLCache,
                           rng::AbstractRNG,
                           re_names::Vector{Symbol},
                           n_steps::Int,
                           anneal_sds::NamedTuple)
    n_steps == 0 && return
    θ_re = _symmetrize_psd_params(θ, dm.model.fixed.fixed)
    dists_builder = get_create_random_effect_distribution(dm.model.random.random)
    model_funs = cache.model_funs
    helpers    = cache.helpers
    has_anneal = !isempty(anneal_sds)
    b          = state.b
    b_prop     = state.b_prop
    indiv_ll   = state.indiv_ll

    for _ in 1:n_steps
        for (k, lv) in enumerate(state.levels)
            re    = re_names[lv.ri]
            const_cov = dm.individuals[lv.rep_ind].const_cov
            dists = dists_builder(θ_re, const_cov, model_funs, helpers)
            dist  = getproperty(dists, re)
            if has_anneal && haskey(anneal_sds, re)
                dist = Normal(mean(dist), getfield(anneal_sds, re))
            end

            # Propose η* ~ prior (or annealed prior)
            eta_star = rand(rng, dist)

            # Build proposed b: copy current b, overwrite this level's range
            copyto!(b_prop, b)
            r = lv.range
            if lv.dim == 1
                b_prop[first(r)] = Float64(eta_star isa AbstractVector ? eta_star[1] : eta_star)
            else
                for j in 1:lv.dim
                    b_prop[r[j]] = Float64(eta_star isa AbstractVector ? eta_star[j] : eta_star)
                end
            end

            # Accept / reject (likelihood ratio only — prior terms cancel)
            state.n_total += 1
            if length(lv.group_pos) == 1
                # Fast path: single individual — no temporary vector allocation
                pos    = lv.group_pos[1]
                i      = batch_info.inds[pos]
                η_ind  = _build_eta_ind(dm, i, batch_info, b_prop, const_cache, θ_re)
                ll_new = _loglikelihood_individual(dm, i, θ_re, η_ind, cache)
                if log(rand(rng)) < ll_new - indiv_ll[pos]
                    copyto!(b, b_prop)
                    state.level_plp[k] = logpdf(dist, eta_star)
                    indiv_ll[pos]      = ll_new
                    state.n_accept     += 1
                end
            else
                Δ_ll   = 0.0
                new_ll = Vector{Float64}(undef, length(lv.group_pos))
                for (gi, pos) in enumerate(lv.group_pos)
                    i      = batch_info.inds[pos]
                    η_ind  = _build_eta_ind(dm, i, batch_info, b_prop, const_cache, θ_re)
                    ll_new = _loglikelihood_individual(dm, i, θ_re, η_ind, cache)
                    new_ll[gi] = ll_new
                    Δ_ll      += ll_new - indiv_ll[pos]
                end
                if log(rand(rng)) < Δ_ll
                    copyto!(b, b_prop)
                    state.level_plp[k] = logpdf(dist, eta_star)
                    for (gi, pos) in enumerate(lv.group_pos)
                        indiv_ll[pos] = new_ll[gi]
                    end
                    state.n_accept += 1
                end
            end
        end
    end
    return
end

# ---------------------------------------------------------------------------
# Kernel 2: per-dimension adaptive random-walk MH (full log-joint acceptance)
# ---------------------------------------------------------------------------

function _saemixmh_kern2!(state::_SaemixMHState,
                           dm::DataModel,
                           batch_info::_LaplaceBatchInfo,
                           θ::ComponentArray,
                           const_cache::LaplaceConstantsCache,
                           cache::_LLCache,
                           rng::AbstractRNG,
                           re_names::Vector{Symbol},
                           n_steps::Int,
                           target_accept::Float64,
                           adapt_rate::Float64,
                           anneal_sds::NamedTuple)
    n_steps == 0 && return
    θ_re = _symmetrize_psd_params(θ, dm.model.fixed.fixed)
    dists_builder = get_create_random_effect_distribution(dm.model.random.random)
    model_funs = cache.model_funs
    helpers    = cache.helpers
    has_anneal = !isempty(anneal_sds)
    b          = state.b
    b_prop     = state.b_prop
    indiv_ll   = state.indiv_ll

    for _ in 1:n_steps
        dim_offset = 0
        for (k, lv) in enumerate(state.levels)
            re    = re_names[lv.ri]
            const_cov = dm.individuals[lv.rep_ind].const_cov
            dists = dists_builder(θ_re, const_cov, model_funs, helpers)
            dist  = getproperty(dists, re)
            if has_anneal && haskey(anneal_sds, re)
                dist = Normal(mean(dist), getfield(anneal_sds, re))
            end

            r = lv.range

            # Per-dimension coordinate-wise proposal
            for d in 1:lv.dim
                dom_idx = dim_offset + d
                # Propose: η*[d] = η[d] + dom[dom_idx] * ε
                copyto!(b_prop, b)
                b_prop[r[d]] = b[r[d]] + state.dom[dom_idx] * randn(rng)

                # New prior lp for this level (using b_prop)
                new_plp = if lv.dim == 1
                    logpdf(dist, b_prop[first(r)])
                else
                    logpdf(dist, view(b_prop, r))
                end

                # Full log-joint ratio; fast path avoids temporary vector allocation
                state.n_total  += 1
                state.n_total2[dom_idx] += 1
                if length(lv.group_pos) == 1
                    # Fast path: single individual
                    pos    = lv.group_pos[1]
                    i      = batch_info.inds[pos]
                    η_ind  = _build_eta_ind(dm, i, batch_info, b_prop, const_cache, θ_re)
                    ll_new = _loglikelihood_individual(dm, i, θ_re, η_ind, cache)
                    Δ = new_plp - state.level_plp[k] + (ll_new - indiv_ll[pos])
                    if log(rand(rng)) < Δ
                        copyto!(b, b_prop)
                        state.level_plp[k]       = new_plp
                        indiv_ll[pos]            = ll_new
                        state.n_accept           += 1
                        state.n_accept2[dom_idx] += 1
                    end
                else
                    Δ_ll   = 0.0
                    new_ll = Vector{Float64}(undef, length(lv.group_pos))
                    for (gi, pos) in enumerate(lv.group_pos)
                        i      = batch_info.inds[pos]
                        η_ind  = _build_eta_ind(dm, i, batch_info, b_prop, const_cache, θ_re)
                        ll_new = _loglikelihood_individual(dm, i, θ_re, η_ind, cache)
                        new_ll[gi] = ll_new
                        Δ_ll      += ll_new - indiv_ll[pos]
                    end
                    Δ = new_plp - state.level_plp[k] + Δ_ll
                    if log(rand(rng)) < Δ
                        copyto!(b, b_prop)
                        state.level_plp[k] = new_plp
                        for (gi, pos) in enumerate(lv.group_pos)
                            indiv_ll[pos] = new_ll[gi]
                        end
                        state.n_accept           += 1
                        state.n_accept2[dom_idx] += 1
                    end
                end
            end
            dim_offset += lv.dim
        end
    end

    # Robbins-Monro scale update (after all kernel-2 steps)
    for dom_idx in eachindex(state.dom)
        n2 = state.n_total2[dom_idx]
        n2 == 0 && continue
        accept_rate = state.n_accept2[dom_idx] / n2
        state.dom[dom_idx] *= (1.0 + adapt_rate * (accept_rate - target_accept))
        state.dom[dom_idx] = max(state.dom[dom_idx], 1e-8)
        # Reset counters for next iteration (Robbins-Monro: use per-iter rate)
        state.n_accept2[dom_idx] = 0
        state.n_total2[dom_idx]  = 0
    end
    return
end

# ---------------------------------------------------------------------------
# _mcem_sample_batch dispatch
# ---------------------------------------------------------------------------

function _mcem_sample_batch(dm::DataModel,
                             info::_LaplaceBatchInfo,
                             θ::ComponentArray,
                             const_cache::LaplaceConstantsCache,
                             cache::_LLCache,
                             sampler::SaemixMH,
                             turing_kwargs,
                             rng::AbstractRNG,
                             re_names::Vector{Symbol},
                             warm_start,
                             last_params;
                             anneal_sds::NamedTuple=NamedTuple())
    nb = info.n_b
    if nb == 0
        return (zeros(eltype(θ), 0, 0), nothing, eltype(θ)[])
    end

    # Initialise or warm-start state
    state = if warm_start && last_params isa _SaemixMHState
        last_params
    else
        _saemixmh_init_state(dm, info, θ, const_cache, cache, rng, re_names)
    end

    # Re-sync indiv_ll and level_plp when θ has changed (e.g. M-step update).
    # We check by recomputing — cheap compared to MCMC itself.
    _saemixmh_init_indiv_ll!(state.indiv_ll, dm, info, θ, const_cache, cache, state.b)
    _saemixmh_level_plp!(state.level_plp, state.levels, dm, θ, const_cache, cache,
                          state.b; anneal_sds=anneal_sds)

    # Run n_samples sweeps: each sweep = n_kern1 kernel-1 steps + n_kern2 kernel-2 steps
    n_sweeps = max(1, get(turing_kwargs, :n_samples, 1))
    for _ in 1:n_sweeps
        _saemixmh_kern1!(state, dm, info, θ, const_cache, cache, rng, re_names,
                          sampler.n_kern1, anneal_sds)
        _saemixmh_kern2!(state, dm, info, θ, const_cache, cache, rng, re_names,
                          sampler.n_kern2, sampler.target_accept, sampler.adapt_rate,
                          anneal_sds)
    end

    b_out = copy(state.b)
    # Return: (single-column sample matrix, updated state, last b)
    return (reshape(b_out, nb, 1), state, b_out)
end
