export SaemixMH

using Distributions
using LinearAlgebra
using Random

# ---------------------------------------------------------------------------
# Public sampler type
# ---------------------------------------------------------------------------

"""
    SaemixMH(; n_kern1=2, n_kern2=2, n_kern3=2,
               proba_mcmc=0.4, stepsize_rw=0.4, rw_init=0.5,
               target_accept=nothing, adapt_rate=nothing)

Saemix-style lightweight MH sampler for the SAEM/MCEM E-step.

Implements the first three MCMC kernels used by the saemix R package:

- **Kernel 1** (`n_kern1` steps): independent proposal from the current RE prior
  `p(η|θ)`.  Acceptance uses only the *likelihood ratio* — no prior term because
  the proposal equals the prior.  Efficient when the posterior is close to the
  prior (early SAEM iterations or weakly informative data).

- **Kernel 2** (`n_kern2` steps): per-level coordinate-wise random-walk in the
  natural parameter space. Acceptance uses the full log-joint ratio. Proposal
  scales follow saemix's `domega2[:, 1]` adaptation rule.

- **Kernel 3** (`n_kern3` steps): block random-walk kernel with the same
  iteration-dependent block-size schedule as saemix. Proposal scales follow
  saemix's `domega2[:, nrs2]` adaptation rule for the active block size.

Unlike `MH()` and `AdaptiveNoLimitsMH()`, `SaemixMH` does **not** go through
Turing.jl.  It operates directly on the flat b-vector using `_loglikelihood_individual`,
eliminating interpreter and compilation overhead.

# Keyword Arguments

- `n_kern1::Int = 2`: prior-proposal steps per E-step call.
- `n_kern2::Int = 2`: per-level RW steps per E-step call.
- `n_kern3::Int = 2`: block RW steps per E-step call, matching saemix default.
- `proba_mcmc::Float64 = 0.4`: saemix acceptance-rate target.
- `stepsize_rw::Float64 = 0.4`: saemix multiplicative adaptation step size.
- `rw_init::Float64 = 0.5`: saemix initial RW scale multiplier.

Backward-compatible aliases:

- `target_accept` maps to `proba_mcmc` when provided.
- `adapt_rate` maps to `stepsize_rw` when provided.

# Example

```julia
res = fit_model(dm, SAEM(;
    sampler    = SaemixMH(),
    mcmc_steps = 1,          # 1 sweep applies kernels 1-3 once
    maxiters   = 300,
    builtin_stats = :closed_form,
    re_cov_params = (; η = :τ)
))
```
"""
struct SaemixMH
    n_kern1       :: Int
    n_kern2       :: Int
    n_kern3       :: Int
    proba_mcmc    :: Float64
    stepsize_rw   :: Float64
    rw_init       :: Float64
end

function SaemixMH(; n_kern1::Int=2,
                    n_kern2::Int=2,
                    n_kern3::Int=2,
                    proba_mcmc::Float64=0.4,
                    stepsize_rw::Float64=0.4,
                    rw_init::Float64=0.5,
                    target_accept=nothing,
                    adapt_rate=nothing)
    p_mcmc = isnothing(target_accept) ? proba_mcmc : Float64(target_accept)
    step_rw = isnothing(adapt_rate) ? stepsize_rw : Float64(adapt_rate)
    return SaemixMH(n_kern1, n_kern2, n_kern3, p_mcmc, step_rw, rw_init)
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
    re_type   :: Symbol           # distribution type symbol (for bijection dispatch)
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
    domega2    :: Vector{Matrix{Float64}} # per-level saemix RW scales, indexed by (dim, block_size)
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
            dist = _saem_apply_anneal_dist(dist, getfield(anneal_sds, re))
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
    cache    = dm.re_group_info.laplace_cache
    re_types = get_re_types(dm.model.random.random)
    levels   = _SaemixMHLevel[]
    for (ri, re) in enumerate(re_names)
        info    = batch_info.re_info[ri]
        re_type = get(re_types, re, :Unknown)
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

            push!(levels, _SaemixMHLevel(ri, li, range, rep, dim, group_pos, re_type))
        end
    end
    return levels
end

# Return the standard deviation in the bijected (z) space for a scalar RE level.
# Used to initialise domega2 in the correct units — the bijection is what kernels
# 2 and 3 now propose in, so the initial step size should live in that space.
function _saemixmh_zspace_std(dist, re_type::Symbol)::Float64
    if re_type == :LogNormal
        # z = log(η) ~ Normal(μ, σ)  →  z-space std = σ
        s = try Float64(dist.σ) catch; NaN end
        return isfinite(s) && s > 0 ? s : 1.0
    elseif re_type == :Exponential
        # z = log(η), log(Exp(θ)) has variance π²/6 regardless of θ
        return Float64(π) / sqrt(6.0)
    elseif re_type == :Beta
        # z = logit(η), delta-method: Var[logit(η)] ≈ (α+β+1)/(α*β)
        α = try Float64(dist.α) catch; NaN end
        β = try Float64(dist.β) catch; NaN end
        if isfinite(α) && isfinite(β) && α > 0 && β > 0
            return sqrt((α + β + 1.0) / (α * β))
        end
        return 1.0
    else
        # Normal and identity fallback: proposal is in natural space
        v = try Float64(var(dist)) catch; NaN end
        return isfinite(v) && v > 0 ? sqrt(v) : 1.0
    end
end

function _saemixmh_init_domega2(levels::Vector{_SaemixMHLevel},
                                 dm::DataModel,
                                 θ::ComponentArray,
                                 const_cache::LaplaceConstantsCache,
                                 cache::_LLCache,
                                 re_names::Vector{Symbol},
                                 rw_init::Float64)
    θ_re = _symmetrize_psd_params(θ, dm.model.fixed.fixed)
    dists_builder = get_create_random_effect_distribution(dm.model.random.random)
    model_funs = cache.model_funs
    helpers    = cache.helpers
    domega2 = Vector{Matrix{Float64}}(undef, length(levels))
    for (k, lv) in enumerate(levels)
        re    = re_names[lv.ri]
        const_cov = dm.individuals[lv.rep_ind].const_cov
        dists = dists_builder(θ_re, const_cov, model_funs, helpers)
        dist  = getproperty(dists, re)
        d = lv.dim
        scales = Vector{Float64}(undef, d)
        if d == 1
            scales[1] = rw_init * _saemixmh_zspace_std(dist, lv.re_type)
        else
            # MvNormal (identity bijection): use diagonal of covariance in natural space
            V = cov(dist)
            for j in 1:d
                v = V[j, j]
                scales[j] = isfinite(v) && v > 0 ? rw_init * sqrt(v) : rw_init
            end
        end
        domega2[k] = repeat(reshape(scales, d, 1), 1, d)
    end
    return domega2
end

function _saemixmh_init_state(dm::DataModel,
                               batch_info::_LaplaceBatchInfo,
                               θ::ComponentArray,
                               const_cache::LaplaceConstantsCache,
                               cache::_LLCache,
                               rng::AbstractRNG,
                               re_names::Vector{Symbol},
                               rw_init::Float64)
    nb     = batch_info.n_b
    n_inds = length(batch_info.inds)
    levels = _saemixmh_build_levels(dm, batch_info, re_names)
    b      = zeros(Float64, nb)
    _saemixmh_init_b!(b, levels, dm, θ, const_cache, cache, rng, re_names)
    indiv_ll  = Vector{Float64}(undef, n_inds)
    _saemixmh_init_indiv_ll!(indiv_ll, dm, batch_info, θ, const_cache, cache, b)
    level_plp = Vector{Float64}(undef, length(levels))
    _saemixmh_level_plp!(level_plp, levels, dm, θ, const_cache, cache, b)
    domega2 = _saemixmh_init_domega2(levels, dm, θ, const_cache, cache, re_names, rw_init)
    return _SaemixMHState(b, similar(b), indiv_ll, level_plp, levels,
                          domega2, 0, 0)
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
                dist = _saem_apply_anneal_dist(dist, getfield(anneal_sds, re))
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
                           proba_mcmc::Float64,
                           stepsize_rw::Float64,
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

    for (k, lv) in enumerate(state.levels)
        nbc2 = zeros(Int, lv.dim)
        nt2  = zeros(Int, lv.dim)
        domega2_k = state.domega2[k]
        for _ in 1:n_steps
            re    = re_names[lv.ri]
            const_cov = dm.individuals[lv.rep_ind].const_cov
            dists = dists_builder(θ_re, const_cov, model_funs, helpers)
            dist  = getproperty(dists, re)
            if has_anneal && haskey(anneal_sds, re)
                dist = _saem_apply_anneal_dist(dist, getfield(anneal_sds, re))
            end

            r = lv.range

            # Per-dimension coordinate-wise proposal (in bijected z-space)
            for d in 1:lv.dim
                copyto!(b_prop, b)
                z_curr_d  = _amh_bij_forward(lv.re_type, b[r[d]])
                z_prop_d  = z_curr_d + domega2_k[d, 1] * randn(rng)
                b_prop[r[d]] = _amh_bij_inverse(lv.re_type, z_prop_d)
                log_jac_d = _amh_bij_log_jac(lv.re_type, z_prop_d, z_curr_d)

                # New prior lp for this level (using b_prop)
                new_plp = if lv.dim == 1
                    logpdf(dist, b_prop[first(r)])
                else
                    logpdf(dist, view(b_prop, r))
                end

                # Full log-joint ratio; fast path avoids temporary vector allocation
                state.n_total  += 1
                nt2[d] += 1
                if length(lv.group_pos) == 1
                    # Fast path: single individual
                    pos    = lv.group_pos[1]
                    i      = batch_info.inds[pos]
                    η_ind  = _build_eta_ind(dm, i, batch_info, b_prop, const_cache, θ_re)
                    ll_new = _loglikelihood_individual(dm, i, θ_re, η_ind, cache)
                    Δ = new_plp - state.level_plp[k] + (ll_new - indiv_ll[pos]) + log_jac_d
                    if log(rand(rng)) < Δ
                        copyto!(b, b_prop)
                        state.level_plp[k]       = new_plp
                        indiv_ll[pos]            = ll_new
                        state.n_accept           += 1
                        nbc2[d] += 1
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
                    Δ = new_plp - state.level_plp[k] + Δ_ll + log_jac_d
                    if log(rand(rng)) < Δ
                        copyto!(b, b_prop)
                        state.level_plp[k] = new_plp
                        for (gi, pos) in enumerate(lv.group_pos)
                            indiv_ll[pos] = new_ll[gi]
                        end
                        state.n_accept += 1
                        nbc2[d] += 1
                    end
                end
            end
        end
        for d in 1:lv.dim
            nt2[d] == 0 && continue
            accept_rate = nbc2[d] / nt2[d]
            domega2_k[d, 1] *= (1.0 + stepsize_rw * (accept_rate - proba_mcmc))
            domega2_k[d, 1] = max(domega2_k[d, 1], 1e-8)
        end
    end
    return
end

@inline function _saemixmh_nrs2(dim::Int, outer_iter::Int)
    dim == 1 && return 1
    return mod(outer_iter, dim - 1) + 2
end

function _saemixmh_kern3!(state::_SaemixMHState,
                           dm::DataModel,
                           batch_info::_LaplaceBatchInfo,
                           θ::ComponentArray,
                           const_cache::LaplaceConstantsCache,
                           cache::_LLCache,
                           rng::AbstractRNG,
                           re_names::Vector{Symbol},
                           n_steps::Int,
                           outer_iter::Int,
                           proba_mcmc::Float64,
                           stepsize_rw::Float64,
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

    for (k, lv) in enumerate(state.levels)
        d = lv.dim
        nrs2 = _saemixmh_nrs2(d, outer_iter)
        nbc2 = zeros(Int, d)
        nt2  = zeros(Int, d)
        VK = vcat(collect(1:d), collect(1:d))
        domega2_k = state.domega2[k]
        for _ in 1:n_steps
            re    = re_names[lv.ri]
            const_cov = dm.individuals[lv.rep_ind].const_cov
            dists = dists_builder(θ_re, const_cov, model_funs, helpers)
            dist  = getproperty(dists, re)
            if has_anneal && haskey(anneal_sds, re)
                dist = _saem_apply_anneal_dist(dist, getfield(anneal_sds, re))
            end
            r = lv.range
            if nrs2 < d
                vk = vcat(0, randperm(rng, d - 1)[1:(nrs2 - 1)])
                nb_iter2 = d
            else
                vk = collect(0:(d - 1))
                nb_iter2 = 1
            end
            for k2 in 1:nb_iter2
                vk2 = VK[k2 .+ vk]
                copyto!(b_prop, b)
                log_jac_block = 0.0
                for coord in vk2
                    z_curr_c = _amh_bij_forward(lv.re_type, b[r[coord]])
                    z_prop_c = z_curr_c + domega2_k[coord, nrs2] * randn(rng)
                    b_prop[r[coord]] = _amh_bij_inverse(lv.re_type, z_prop_c)
                    log_jac_block   += _amh_bij_log_jac(lv.re_type, z_prop_c, z_curr_c)
                    nt2[coord] += 1
                end

                new_plp = if d == 1
                    logpdf(dist, b_prop[first(r)])
                else
                    logpdf(dist, view(b_prop, r))
                end

                state.n_total += 1
                if length(lv.group_pos) == 1
                    pos    = lv.group_pos[1]
                    i      = batch_info.inds[pos]
                    η_ind  = _build_eta_ind(dm, i, batch_info, b_prop, const_cache, θ_re)
                    ll_new = _loglikelihood_individual(dm, i, θ_re, η_ind, cache)
                    Δ = new_plp - state.level_plp[k] + (ll_new - indiv_ll[pos]) + log_jac_block
                    if log(rand(rng)) < Δ
                        copyto!(b, b_prop)
                        state.level_plp[k] = new_plp
                        indiv_ll[pos] = ll_new
                        state.n_accept += 1
                        for coord in vk2
                            nbc2[coord] += 1
                        end
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
                    Δ = new_plp - state.level_plp[k] + Δ_ll + log_jac_block
                    if log(rand(rng)) < Δ
                        copyto!(b, b_prop)
                        state.level_plp[k] = new_plp
                        for (gi, pos) in enumerate(lv.group_pos)
                            indiv_ll[pos] = new_ll[gi]
                        end
                        state.n_accept += 1
                        for coord in vk2
                            nbc2[coord] += 1
                        end
                    end
                end
            end
        end
        for coord in 1:d
            nt2[coord] == 0 && continue
            accept_rate = nbc2[coord] / nt2[coord]
            domega2_k[coord, nrs2] *= (1.0 + stepsize_rw * (accept_rate - proba_mcmc))
            domega2_k[coord, nrs2] = max(domega2_k[coord, nrs2], 1e-8)
        end
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
                             anneal_sds::NamedTuple=NamedTuple(),
                             outer_iter::Int=1)
    nb = info.n_b
    if nb == 0
        return (zeros(eltype(θ), 0, 0), nothing, eltype(θ)[])
    end

    # Initialise or warm-start state
    state = if warm_start && last_params isa _SaemixMHState
        last_params
    else
        _saemixmh_init_state(dm, info, θ, const_cache, cache, rng, re_names, sampler.rw_init)
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
                          sampler.n_kern2, sampler.proba_mcmc, sampler.stepsize_rw,
                          anneal_sds)
        _saemixmh_kern3!(state, dm, info, θ, const_cache, cache, rng, re_names,
                          sampler.n_kern3, outer_iter, sampler.proba_mcmc,
                          sampler.stepsize_rw, anneal_sds)
    end

    b_out = copy(state.b)
    # Return: (single-column sample matrix, updated state, last b)
    return (reshape(b_out, nb, 1), state, b_out)
end
