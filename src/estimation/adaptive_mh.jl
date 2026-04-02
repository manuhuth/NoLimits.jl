export AdaptiveNoLimitsMH

using Distributions
using LinearAlgebra
using Random

# ---------------------------------------------------------------------------
# Public sampler type
# ---------------------------------------------------------------------------

"""
    AdaptiveNoLimitsMH(; adapt_start, init_scale, eps_reg)

Adaptive Metropolis-Hastings sampler for the SAEM/MCEM E-step.

Implements the Haario et al. (2001) Adaptive Metropolis algorithm with
per-RE-name proposal covariance. Samples are collected from all active
levels of each random effect and pooled together for covariance estimation,
so the proposal adapts quickly even with few individuals per batch.

The sampler works in the natural proposal space for each distribution family:

| Distribution      | Proposal space | Bijection     |
|-------------------|----------------|---------------|
| `Normal`          | η ∈ ℝ          | identity      |
| `MvNormal`        | η ∈ ℝ^d        | identity      |
| `LogNormal`       | z = log(η)     | log / exp     |
| `Exponential`     | z = log(η)     | log / exp     |
| `Beta`            | z = logit(η)   | logit / sigmoid |
| `NormalizingPlanarFlow` | η ∈ ℝ^d  | identity      |

Per-individual log-likelihoods are cached and updated incrementally: each
per-level proposal recomputes only the log-likelihood of the individual(s)
that belong to that RE level, not the full batch. This makes the per-step
cost O(1) in the number of individuals rather than O(N²).

Adaptation state (running covariance, mean, and sample count) persists across
SAEM iterations via the warm-start mechanism. Unknown distribution types fall
back to the identity bijection; bounded-support distributions will use
implicit rejection (proposals outside the support yield `logpdf = -Inf`).

# Keyword arguments
- `adapt_start::Int = 50`: pooled sample count before Haario updates begin.
- `init_scale::Float64 = 1.0`: multiplier on the prior-based initial proposal
  covariance.
- `eps_reg::Float64 = 1e-6`: regularisation added to the diagonal of the
  proposal covariance to ensure positive-definiteness.

# Usage
```julia
res = fit_model(dm, SAEM(sampler=AdaptiveNoLimitsMH()))
```
"""
struct AdaptiveNoLimitsMH
    adapt_start :: Int
    init_scale  :: Float64
    eps_reg     :: Float64
end

function AdaptiveNoLimitsMH(; adapt_start::Int    = 50,
                               init_scale::Float64 = 1.0,
                               eps_reg::Float64    = 1e-6)
    adapt_start >= 0 || throw(ArgumentError("adapt_start must be ≥ 0"))
    init_scale  >  0 || throw(ArgumentError("init_scale must be > 0"))
    eps_reg     >  0 || throw(ArgumentError("eps_reg must be > 0"))
    return AdaptiveNoLimitsMH(adapt_start, init_scale, eps_reg)
end

# ---------------------------------------------------------------------------
# Internal state types
# ---------------------------------------------------------------------------

mutable struct _REAdaptBlock
    re_name    :: Symbol
    re_type    :: Symbol          # constructor symbol from get_re_types, e.g. :Normal
    dim        :: Int             # dimension per level
    n_levels   :: Int             # active levels in this batch
    ri         :: Int             # index into re_names / re_info
    lp_offset  :: Int             # offset into re_lp vector (lp_offset+li = index for level li)
    level_inds :: Vector{Vector{Int}}  # level_inds[li] = batch-local ind indices for level li
    C          :: Matrix{Float64} # d×d proposal covariance
    C_chol_L   :: Matrix{Float64} # cached lower-triangular Cholesky factor of C
    μ_run      :: Vector{Float64} # pooled running mean (d)
    S_run      :: Matrix{Float64} # pooled running scatter (d×d, Welford)
    n_samples  :: Int             # total pooled sample count (across levels)
end

mutable struct _AdaptiveMHState
    b_current :: Vector{Float64}
    ind_ll    :: Vector{Float64}  # ind_ll[j] = loglik for batch individual j
    re_lp     :: Vector{Float64}  # re_lp[block.lp_offset+li] = log-prior for level li
    logp      :: Float64          # ≈ sum(ind_ll) + sum(re_lp); recomputed after warm-start
    blocks    :: Vector{_REAdaptBlock}
    n_accept  :: Int
    n_total   :: Int
end

# ---------------------------------------------------------------------------
# Bijections: η (natural space) ↔ z (proposal space)
#
# MH acceptance log-ratio for a symmetric proposal in z-space:
#   log α = logf(b_prop) - logf(b_curr) + _amh_bij_log_jac(re_type, z_new, z_old)
#
# The log-Jacobian correction is:
#   log|J_f(η_old)| - log|J_f(η_new)|
# where f is the forward bijection f: η → z.
# ---------------------------------------------------------------------------

# Normal: identity (η ∈ ℝ, no bijection needed)
@inline _amh_bij_forward(::Val{:Normal}, η::Real)           = Float64(η)
@inline _amh_bij_inverse(::Val{:Normal}, z::Real)           = Float64(z)
@inline _amh_bij_log_jac(::Val{:Normal}, ::Real, ::Real)    = 0.0

# MvNormal: identity (η ∈ ℝ^d, no bijection needed)
# When input is already Vector{Float64}, return it directly to avoid an unnecessary copy.
@inline _amh_bij_forward(::Val{:MvNormal}, η::Vector{Float64}) = η
@inline _amh_bij_forward(::Val{:MvNormal}, η::AbstractVector)  = Vector{Float64}(η)
@inline _amh_bij_inverse(::Val{:MvNormal}, z::Vector{Float64}) = z
@inline _amh_bij_inverse(::Val{:MvNormal}, z::AbstractVector)  = Vector{Float64}(z)
@inline _amh_bij_log_jac(::Val{:MvNormal}, ::AbstractVector, ::AbstractVector) = 0.0

# LogNormal: log bijection (η > 0)
# J_f(η) = 1/η  →  log|J_f| = -log(η) = -z
# Correction: -z_old - (-z_new) = z_new - z_old
@inline _amh_bij_forward(::Val{:LogNormal}, η::Real) = log(max(Float64(η), 1e-300))
@inline _amh_bij_inverse(::Val{:LogNormal}, z::Real) = exp(Float64(z))
@inline _amh_bij_log_jac(::Val{:LogNormal}, z_new::Real, z_old::Real) =
    Float64(z_new) - Float64(z_old)

# Exponential: same log bijection (η > 0)
@inline _amh_bij_forward(::Val{:Exponential}, η::Real) = log(max(Float64(η), 1e-300))
@inline _amh_bij_inverse(::Val{:Exponential}, z::Real) = exp(Float64(z))
@inline _amh_bij_log_jac(::Val{:Exponential}, z_new::Real, z_old::Real) =
    Float64(z_new) - Float64(z_old)

# Beta: logit bijection (η ∈ (0,1))
# J_f(η) = 1/(η(1-η))  →  log|J_f| = -log(η(1-η))
# Correction: -log(η_old(1-η_old)) - (-log(η_new(1-η_new)))
#           = log(η_new(1-η_new)) - log(η_old(1-η_old))
@inline function _amh_bij_forward(::Val{:Beta}, η::Real)
    ηf = clamp(Float64(η), 1e-15, 1.0 - 1e-15)
    return log(ηf) - log1p(-ηf)
end
@inline function _amh_bij_inverse(::Val{:Beta}, z::Real)
    return 1.0 / (1.0 + exp(-Float64(z)))
end
@inline function _amh_bij_log_jac(::Val{:Beta}, z_new::Real, z_old::Real)
    η_new = _amh_bij_inverse(Val(:Beta), z_new)
    η_old = _amh_bij_inverse(Val(:Beta), z_old)
    return log(max(η_new * (1.0 - η_new), 1e-300)) -
           log(max(η_old * (1.0 - η_old), 1e-300))
end

# NormalizingPlanarFlow: identity (output ∈ ℝ^d; user chose η-space)
@inline _amh_bij_forward(::Val{:NormalizingPlanarFlow}, η::Vector{Float64}) = η
@inline _amh_bij_forward(::Val{:NormalizingPlanarFlow}, η::AbstractVector)  = Vector{Float64}(η)
@inline _amh_bij_inverse(::Val{:NormalizingPlanarFlow}, z::Vector{Float64}) = z
@inline _amh_bij_inverse(::Val{:NormalizingPlanarFlow}, z::AbstractVector)  = Vector{Float64}(z)
@inline _amh_bij_log_jac(
    ::Val{:NormalizingPlanarFlow}, ::AbstractVector, ::AbstractVector) = 0.0

# Fallback: identity for unknown / unbounded distributions
@inline _amh_bij_forward(::Val, η) = η
@inline _amh_bij_inverse(::Val, z) = z
@inline _amh_bij_log_jac(::Val, z_new, z_old) = 0.0

# Symbol-based dispatch wrappers
@inline _amh_bij_forward(s::Symbol, η)            = _amh_bij_forward(Val(s), η)
@inline _amh_bij_inverse(s::Symbol, z)            = _amh_bij_inverse(Val(s), z)
@inline _amh_bij_log_jac(s::Symbol, z_new, z_old) = _amh_bij_log_jac(Val(s), z_new, z_old)

# ---------------------------------------------------------------------------
# Initial proposal covariance from prior distribution
# ---------------------------------------------------------------------------

function _amh_init_cov(dist::Normal, ::Int,
                        init_scale::Float64, eps_reg::Float64)
    v = 2.38^2 * init_scale * dist.σ^2 + eps_reg
    C = Matrix{Float64}(undef, 1, 1)
    C[1, 1] = v
    return C
end

function _amh_init_cov(dist::AbstractMvNormal, dim::Int,
                        init_scale::Float64, eps_reg::Float64)
    λ = 2.38^2 / dim * init_scale
    Ω = Matrix{Float64}(cov(dist))
    return λ .* Ω .+ eps_reg .* Matrix{Float64}(I(dim))
end

function _amh_init_cov(dist::LogNormal, ::Int,
                        init_scale::Float64, eps_reg::Float64)
    # In log-space the prior is Normal(μ, σ), so proposal variance = σ²
    v = 2.38^2 * init_scale * dist.σ^2 + eps_reg
    C = Matrix{Float64}(undef, 1, 1)
    C[1, 1] = v
    return C
end

function _amh_init_cov(::Exponential, ::Int,
                        init_scale::Float64, eps_reg::Float64)
    # Var[log(Exponential)] = π²/6 (log-exponential / Gumbel variance)
    v = 2.38^2 * init_scale * π^2 / 6.0 + eps_reg
    C = Matrix{Float64}(undef, 1, 1)
    C[1, 1] = v
    return C
end

function _amh_init_cov(dist::Beta, ::Int,
                        init_scale::Float64, eps_reg::Float64)
    α, β = dist.α, dist.β
    σ²_logit = (α + β + 1.0) / (α * β)   # delta-method: Var[logit(η)]
    v = 2.38^2 * init_scale * σ²_logit + eps_reg
    C = Matrix{Float64}(undef, 1, 1)
    C[1, 1] = v
    return C
end

function _amh_init_cov(::Any, dim::Int, init_scale::Float64, eps_reg::Float64)
    # Fallback: isotropic identity scaled by prior-independent constant
    λ = 2.38^2 / max(dim, 1) * init_scale
    return (λ + eps_reg) .* Matrix{Float64}(I(dim))
end

# ---------------------------------------------------------------------------
# Cholesky helper — compute lower factor of proposal covariance
# ---------------------------------------------------------------------------

@inline function _amh_chol_L(C::Matrix{Float64}, dim::Int) :: Matrix{Float64}
    if dim == 1
        L = Matrix{Float64}(undef, 1, 1)
        L[1, 1] = sqrt(max(C[1, 1], 1e-14))
        return L
    else
        return Matrix{Float64}(cholesky(Symmetric(C .+ 1e-14 .* I(dim))).L)
    end
end

# ---------------------------------------------------------------------------
# Proposal generation in z-space  (uses cached L, not C directly)
# ---------------------------------------------------------------------------

@inline function _amh_propose(rng::AbstractRNG, z_curr::Real, L::Matrix{Float64})
    return z_curr + L[1, 1] * randn(rng)
end

function _amh_propose(rng::AbstractRNG, z_curr::AbstractVector{<:Real},
                       L::Matrix{Float64})
    d = length(z_curr)
    return z_curr .+ L * randn(rng, d)
end

# ---------------------------------------------------------------------------
# Haario AM covariance update (pooled across levels of the same RE name)
# ---------------------------------------------------------------------------

function _amh_haario_update!(block::_REAdaptBlock, z::Real,
                              adapt_start::Int, eps_reg::Float64)
    block.n_samples += 1
    n = block.n_samples
    # Welford online mean and scatter (always updated)
    δ = Float64(z) - block.μ_run[1]
    block.μ_run[1] += δ / n
    δ2 = Float64(z) - block.μ_run[1]
    block.S_run[1, 1] += δ * δ2
    # Update proposal covariance once we have enough pooled samples
    if n >= max(adapt_start + 1, 2)
        C_new = 2.38^2 * block.S_run[1, 1] / (n - 1) + eps_reg
        block.C[1, 1] = max(C_new, eps_reg)
        block.C_chol_L[1, 1] = sqrt(block.C[1, 1])
    end
end

function _amh_haario_update!(block::_REAdaptBlock, z::AbstractVector{<:Real},
                              adapt_start::Int, eps_reg::Float64)
    block.n_samples += 1
    n  = block.n_samples
    d  = block.dim
    zf = Vector{Float64}(z)
    # Welford update
    δ = zf .- block.μ_run
    block.μ_run .+= δ ./ n
    δ2 = zf .- block.μ_run
    for j in 1:d, i in 1:d
        block.S_run[i, j] += δ[i] * δ2[j]
    end
    if n >= max(adapt_start + 1, 2)
        λ   = 2.38^2 / d
        Id  = Matrix{Float64}(I(d))
        @. block.C = λ * block.S_run / (n - 1) + eps_reg * Id
        block.C_chol_L .= _amh_chol_L(block.C, d)
    end
end

# ---------------------------------------------------------------------------
# Build per-level individual index: level_inds[li] = batch-local ind indices
# ---------------------------------------------------------------------------

function _amh_build_level_inds(info::_LaplaceBatchInfo, ri::Int,
                                 laplace_cache) :: Vector{Vector{Int}}
    re_info  = info.re_info[ri]
    n_levels = length(re_info.map.levels)
    level_inds = [Int[] for _ in 1:n_levels]
    for (j, ind_global) in enumerate(info.inds)
        ids_j = laplace_cache.ind_level_ids[ind_global][ri]
        for id in ids_j
            li = re_info.map.level_to_index[id]
            li == 0 && continue   # constant level — not in b
            push!(level_inds[li], j)
        end
    end
    return level_inds
end

# ---------------------------------------------------------------------------
# State initialisation
# ---------------------------------------------------------------------------

function _amh_init_state(dm::DataModel, info::_LaplaceBatchInfo,
                          θ::ComponentArray, re_names::Vector{Symbol},
                          const_cache::LaplaceConstantsCache, cache::_LLCache,
                          sampler::AdaptiveNoLimitsMH, rng::AbstractRNG)
    nb            = info.n_b
    b_current     = zeros(Float64, nb)
    dists_builder = get_create_random_effect_distribution(dm.model.random.random)
    re_types      = get_re_types(dm.model.random.random)
    model_funs    = cache.model_funs
    helpers       = cache.helpers
    θ_re          = _symmetrize_psd_params(θ, dm.model.fixed.fixed)
    laplace_cache = dm.re_group_info.laplace_cache
    blocks        = _REAdaptBlock[]
    lp_offset     = 0

    for (ri, re_name) in enumerate(re_names)
        re_info  = info.re_info[ri]
        levels   = re_info.map.levels
        n_levels = length(levels)
        dim      = re_info.dim
        n_levels == 0 && continue

        # Sample from prior to initialise b
        for (li, _) in enumerate(levels)
            const_cov = dm.individuals[re_info.reps[li]].const_cov
            dists     = dists_builder(θ_re, const_cov, model_funs, helpers)
            dist      = getproperty(dists, re_name)
            val       = rand(rng, dist)
            r         = re_info.ranges[li]
            if re_info.is_scalar
                v = val isa AbstractVector ? val[1] : val
                b_current[first(r)] = Float64(v)
            else
                b_current[r] .= Float64.(val)
            end
        end

        # Representative distribution for initial proposal covariance
        const_cov_rep = dm.individuals[re_info.reps[1]].const_cov
        dists_rep     = dists_builder(θ_re, const_cov_rep, model_funs, helpers)
        dist_rep      = getproperty(dists_rep, re_name)
        re_type       = get(re_types, re_name, :Unknown)
        C             = _amh_init_cov(dist_rep, dim, sampler.init_scale, sampler.eps_reg)

        # Precompute per-level individual indices (for incremental log-joint)
        level_inds = _amh_build_level_inds(info, ri, laplace_cache)

        push!(blocks, _REAdaptBlock(
            re_name, re_type, dim, n_levels, ri,
            lp_offset, level_inds,
            C,
            _amh_chol_L(C, dim),
            zeros(Float64, dim),
            zeros(Float64, dim, dim),
            0,
        ))
        lp_offset += n_levels
    end

    # Initialise per-individual and per-level caches
    ind_ll, re_lp, logp = _amh_compute_full_ll(dm, info, θ_re, b_current,
                                                const_cache, cache, blocks,
                                                dists_builder, model_funs, helpers)
    return _AdaptiveMHState(b_current, ind_ll, re_lp, logp, blocks, 0, 0)
end

# ---------------------------------------------------------------------------
# Compute full ind_ll / re_lp vectors (used at init and warm-start)
# ---------------------------------------------------------------------------

function _amh_compute_full_ll(dm::DataModel, info::_LaplaceBatchInfo,
                               θ_re::ComponentArray, b::Vector{Float64},
                               const_cache::LaplaceConstantsCache, cache::_LLCache,
                               blocks::Vector{_REAdaptBlock},
                               dists_builder, model_funs, helpers)
    n_inds   = length(info.inds)
    n_lp     = isempty(blocks) ? 0 : blocks[end].lp_offset + blocks[end].n_levels
    ind_ll   = Vector{Float64}(undef, n_inds)
    re_lp    = Vector{Float64}(undef, n_lp)

    # Per-individual likelihoods
    for (j, ind_global) in enumerate(info.inds)
        η_ind     = _build_eta_ind(dm, ind_global, info, b, const_cache, θ_re)
        ind_ll[j] = _loglikelihood_individual(dm, ind_global, θ_re, η_ind, cache)
    end

    # Per-level RE log-priors
    for block in blocks
        re_info = info.re_info[block.ri]
        for li in 1:block.n_levels
            lp_idx    = block.lp_offset + li
            level_id  = re_info.map.levels[li]
            const_cov = dm.individuals[re_info.reps[li]].const_cov
            dists     = dists_builder(θ_re, const_cov, model_funs, helpers)
            dist      = getproperty(dists, block.re_name)
            v         = _re_value_from_b(re_info, level_id, b)
            re_lp[lp_idx] = (v === nothing) ? 0.0 : Float64(logpdf(dist, v))
        end
    end

    logp = sum(ind_ll) + sum(re_lp)
    return ind_ll, re_lp, logp
end

# Recompute caches in-place after an M-step (θ changed, b_current unchanged).
function _amh_recompute_ll_cache!(state::_AdaptiveMHState,
                                   dm::DataModel, info::_LaplaceBatchInfo,
                                   θ::ComponentArray,
                                   const_cache::LaplaceConstantsCache,
                                   cache::_LLCache)
    θ_re          = _symmetrize_psd_params(θ, dm.model.fixed.fixed)
    dists_builder = get_create_random_effect_distribution(dm.model.random.random)
    model_funs    = cache.model_funs
    helpers       = cache.helpers
    b             = state.b_current

    # Per-individual likelihoods
    for (j, ind_global) in enumerate(info.inds)
        η_ind          = _build_eta_ind(dm, ind_global, info, b, const_cache, θ_re)
        state.ind_ll[j] = _loglikelihood_individual(dm, ind_global, θ_re, η_ind, cache)
    end

    # Per-level RE log-priors
    for block in state.blocks
        re_info = info.re_info[block.ri]
        for li in 1:block.n_levels
            lp_idx    = block.lp_offset + li
            level_id  = re_info.map.levels[li]
            const_cov = dm.individuals[re_info.reps[li]].const_cov
            dists     = dists_builder(θ_re, const_cov, model_funs, helpers)
            dist      = getproperty(dists, block.re_name)
            v         = _re_value_from_b(re_info, level_id, b)
            state.re_lp[block.lp_offset + li] = (v === nothing) ? 0.0 :
                                                  Float64(logpdf(dist, v))
        end
    end

    state.logp = sum(state.ind_ll) + sum(state.re_lp)
end

# ---------------------------------------------------------------------------
# Single MH step: per-level Gibbs-within-MH with incremental log-joint
# ---------------------------------------------------------------------------

function _amh_step!(state::_AdaptiveMHState, dm::DataModel,
                    info::_LaplaceBatchInfo, θ::ComponentArray,
                    const_cache::LaplaceConstantsCache, cache::_LLCache,
                    sampler::AdaptiveNoLimitsMH, rng::AbstractRNG;
                    anneal_sds::NamedTuple=NamedTuple())
    b             = state.b_current
    θ_re          = _symmetrize_psd_params(θ, dm.model.fixed.fixed)
    dists_builder = get_create_random_effect_distribution(dm.model.random.random)
    model_funs    = cache.model_funs
    helpers       = cache.helpers
    has_anneal    = !isempty(anneal_sds)

    for block in state.blocks
        re_info  = info.re_info[block.ri]
        n_levels = block.n_levels
        n_levels == 0 && continue

        for li in 1:n_levels
            r      = re_info.ranges[li]
            lp_idx = block.lp_offset + li

            # ------------------------------------------------------------------
            # 1. Save current value, propose in z-space
            # ------------------------------------------------------------------
            if re_info.is_scalar
                η_curr  = b[first(r)]
                old_val = η_curr          # scalar copy
            else
                old_val = copy(b[r])      # vector copy (heap; rare for scalars)
                η_curr  = old_val
            end

            z_curr = _amh_bij_forward(block.re_type, η_curr)
            z_prop = _amh_propose(rng, z_curr, block.C_chol_L)
            η_prop = _amh_bij_inverse(block.re_type, z_prop)

            # Write proposal into b in-place
            if re_info.is_scalar
                b[first(r)] = Float64(η_prop isa AbstractVector ? η_prop[1] : η_prop)
            else
                b[r] .= Float64.(η_prop)
            end

            # ------------------------------------------------------------------
            # 2. RE log-prior at proposal (cheap: single logpdf)
            # ------------------------------------------------------------------
            const_cov = dm.individuals[re_info.reps[li]].const_cov
            dists     = dists_builder(θ_re, const_cov, model_funs, helpers)
            dist      = getproperty(dists, block.re_name)
            if has_anneal && haskey(anneal_sds, block.re_name)
                dist = Normal(mean(dist), getfield(anneal_sds, block.re_name))
            end
            η_for_lp  = re_info.is_scalar ? b[first(r)] : b[r]
            new_re_lp = Float64(logpdf(dist, η_for_lp))
            old_re_lp = state.re_lp[lp_idx]
            delta_re  = new_re_lp - old_re_lp

            # ------------------------------------------------------------------
            # 3. Individual log-likelihoods for affected individuals only
            #    Fast path: the typical case is exactly 1 individual per level.
            # ------------------------------------------------------------------
            affected  = block.level_inds[li]
            log_jac   = _amh_bij_log_jac(block.re_type, z_prop, z_curr)
            accepted  = false

            if isfinite(new_re_lp)
                if length(affected) == 1
                    # ---- Fast path (no allocation) ----
                    j          = affected[1]
                    old_ll_j   = state.ind_ll[j]
                    ind_global = info.inds[j]
                    η_ind      = _build_eta_ind(dm, ind_global, info, b, const_cache, θ_re)
                    new_ll_j   = _loglikelihood_individual(dm, ind_global, θ_re, η_ind, cache)
                    if isfinite(new_ll_j)
                        log_alpha = delta_re + (new_ll_j - old_ll_j) + log_jac
                        state.n_total += 1
                        if log(rand(rng)) < log_alpha
                            state.re_lp[lp_idx] = new_re_lp
                            state.ind_ll[j]      = new_ll_j
                            state.logp           += delta_re + (new_ll_j - old_ll_j)
                            state.n_accept       += 1
                            accepted = true
                        end
                    else
                        state.n_total += 1
                    end
                elseif length(affected) > 1
                    # ---- General path (allocates; rare) ----
                    n_aff    = length(affected)
                    new_lls  = Vector{Float64}(undef, n_aff)
                    delta_ind = 0.0
                    all_fin  = true
                    for (k, j) in enumerate(affected)
                        ind_global  = info.inds[j]
                        η_ind       = _build_eta_ind(dm, ind_global, info, b,
                                                     const_cache, θ_re)
                        new_lls[k]  = _loglikelihood_individual(dm, ind_global,
                                                                 θ_re, η_ind, cache)
                        if !isfinite(new_lls[k])
                            all_fin = false
                            break
                        end
                        delta_ind += new_lls[k] - state.ind_ll[j]
                    end
                    state.n_total += 1
                    if all_fin
                        log_alpha = delta_re + delta_ind + log_jac
                        if log(rand(rng)) < log_alpha
                            state.re_lp[lp_idx] = new_re_lp
                            for (k, j) in enumerate(affected)
                                state.ind_ll[j] = new_lls[k]
                            end
                            state.logp   += delta_re + delta_ind
                            state.n_accept += 1
                            accepted = true
                        end
                    end
                else
                    # affected is empty — should not happen, but handle gracefully
                    state.n_total += 1
                end
            else
                state.n_total += 1
            end

            # Restore b on reject
            if !accepted
                if re_info.is_scalar
                    b[first(r)] = Float64(old_val)
                else
                    b[r] .= old_val
                end
            end

            # ------------------------------------------------------------------
            # 4. Haario update with the current (accepted or unchanged) z value
            # ------------------------------------------------------------------
            z_now = if re_info.is_scalar
                _amh_bij_forward(block.re_type, b[first(r)])
            else
                _amh_bij_forward(block.re_type, Vector{Float64}(b[r]))
            end
            _amh_haario_update!(block, z_now, sampler.adapt_start, sampler.eps_reg)
        end
    end
end

# ---------------------------------------------------------------------------
# _mcem_sample_batch dispatch for AdaptiveNoLimitsMH
# ---------------------------------------------------------------------------

function _mcem_sample_batch(dm::DataModel, info::_LaplaceBatchInfo,
                             θ::ComponentArray,
                             const_cache::LaplaceConstantsCache, cache::_LLCache,
                             sampler::AdaptiveNoLimitsMH, turing_kwargs::NamedTuple,
                             rng::AbstractRNG, re_names::Vector{Symbol},
                             warm_start, last_params;
                             anneal_sds::NamedTuple=NamedTuple())
    nb = info.n_b
    if nb == 0
        return (zeros(eltype(θ), 0, 0), nothing, eltype(θ)[])
    end
    n_samples = get(turing_kwargs, :n_samples, 100)

    # Restore adaptation state or initialise from prior
    state = if warm_start && last_params isa _AdaptiveMHState
        # θ has changed since last iteration — recompute ind_ll and re_lp in-place.
        _amh_recompute_ll_cache!(last_params, dm, info, θ, const_cache, cache)
        last_params
    else
        _amh_init_state(dm, info, θ, re_names, const_cache, cache, sampler, rng)
    end

    # Run the chain
    samples = Matrix{Float64}(undef, nb, n_samples)
    for i in 1:n_samples
        _amh_step!(state, dm, info, θ, const_cache, cache, sampler, rng;
                   anneal_sds=anneal_sds)
        samples[:, i] .= state.b_current
    end

    # Remove samples with zero prior density (same filter as the Turing path)
    samples = _filter_b_samples_by_prior(dm, info, θ, const_cache, cache, samples)
    if size(samples, 2) == 0
        return (zeros(eltype(θ), nb, 0), state, zeros(eltype(θ), nb))
    end

    lastb = vec(samples[:, end])
    return (samples, state, lastb)
end
