export FOCEI

using Optimization
using OptimizationOptimJL
using OptimizationNLopt
using SciMLBase
using ComponentArrays
using Random
using LineSearches
using ForwardDiff
using LinearAlgebra
using Distributions
using SpecialFunctions: trigamma

# =====================================================================================
# FOCEI / FOCE estimation.
#
# FOCEI is the Laplace approximation with the inner negative Hessian of the log-joint
# replaced by the Gauss-Newton / expected-information form
#
#     𝓗_i  =  Σ_j Jᵢⱼᵀ ℐ(φᵢⱼ) Jᵢⱼ  −  ∇²_η log π_i(η̂_i)
#
# where
#   * φᵢⱼ           are the parameters of observation j's outcome distribution,
#   * Jᵢⱼ = ∂φᵢⱼ/∂η is obtained by one (first-order) ForwardDiff Jacobian through the
#                    formula evaluation (and the ODE solve, if any),
#   * ℐ(φ)          is the closed-form Fisher information of the outcome family
#                    (the "expected information"), evaluated in Distributions.jl's
#                    native parameterization, and
#   * −∇²_η log π   is the exact curvature of the random-effects prior (= Ω⁻¹ for
#                    Normal/MvNormal random effects).
#
# Everything else — the empirical-Bayes inner solve, batching, the (q/2)·log(2π)
# constant, the Cholesky/log-det, caching and the result accessors — is shared with the
# Laplace implementation.  Only the Hessian builder differs, which drops the per-subject
# Hessian from second-order to first-order AD.
#
# FOCE (no interaction) freezes the dispersion-type parameters at the random-effects
# prior mean mᵢ = mean(πᵢ) (= 0 for centered Normal random effects, the classical case)
# and drops their η-sensitivity, while the location/structural parameters and their
# Jacobian stay at the conditional mode η̂.
# =====================================================================================

# -------------------------------------------------------------------------------------
# 1. Fisher-information registry
#
# For each supported outcome family we provide
#   _focei_params(d)                -> Vector of the distribution parameters (flat reals)
#   _focei_expected_information(d)  -> Matrix, the Fisher information ℐ(φ) in params order
#   _focei_dispersion_indices(d)    -> indices of _focei_params that FOCE freezes
#   _focei_is_supported(d)          -> Bool
#
# The Jacobian carries every link / transform / ODE dependence, so ℐ is stored purely in
# the native parameterization.  Distributions without a registered ℐ are rejected up
# front (no numeric fallback by design).
# -------------------------------------------------------------------------------------

_focei_is_supported(::Any) = false
_focei_dispersion_indices(::Any) = Int[]
function _focei_params(d)
    error("FOCEI: distribution $(typeof(d)) is not supported (no closed-form Fisher information). Use Laplace instead, or remove this outcome.")
end
function _focei_expected_information(d)
    error("FOCEI: no closed-form Fisher information for $(typeof(d)). Use Laplace instead.")
end
_focei_paramcount(d) = length(_focei_params(d))

# --- location-scale families with a frozen dispersion parameter (the "I" in FOCEI) ---

# Normal(μ, σ): ℐ = diag(1/σ², 2/σ²)
_focei_is_supported(::Normal) = true
_focei_params(d::Normal) = [d.μ, d.σ]
_focei_paramcount(::Normal) = 2
_focei_dispersion_indices(::Normal) = [2]
function _focei_expected_information(d::Normal)
    iv = inv(d.σ * d.σ)
    z = zero(iv)
    return [iv z; z 2iv]
end

# LogNormal(μ, σ): same information as Normal in (μ, σ) — invariant to the y↦exp data map.
_focei_is_supported(::LogNormal) = true
_focei_params(d::LogNormal) = [d.μ, d.σ]
_focei_paramcount(::LogNormal) = 2
_focei_dispersion_indices(::LogNormal) = [2]
function _focei_expected_information(d::LogNormal)
    iv = inv(d.σ * d.σ)
    z = zero(iv)
    return [iv z; z 2iv]
end

# Laplace(μ, b): ℐ = diag(1/b², 1/b²)   (Distributions stores the scale as field θ)
_focei_is_supported(::Distributions.Laplace) = true
_focei_params(d::Distributions.Laplace) = [d.μ, d.θ]
_focei_paramcount(::Distributions.Laplace) = 2
_focei_dispersion_indices(::Distributions.Laplace) = [2]
function _focei_expected_information(d::Distributions.Laplace)
    ib = inv(d.θ * d.θ)
    z = zero(ib)
    return [ib z; z ib]
end

# Cauchy(μ, σ): ℐ = diag(1/2σ², 1/2σ²)
_focei_is_supported(::Cauchy) = true
_focei_params(d::Cauchy) = [d.μ, d.σ]
_focei_paramcount(::Cauchy) = 2
_focei_dispersion_indices(::Cauchy) = [2]
function _focei_expected_information(d::Cauchy)
    ih = inv(2 * d.σ * d.σ)
    z = zero(ih)
    return [ih z; z ih]
end

# --- single-parameter families (FOCE ≡ FOCEI; no dispersion to freeze) ---

# Exponential(θ) [scale]: ℐ = 1/θ²
_focei_is_supported(::Exponential) = true
_focei_params(d::Exponential) = [d.θ]
_focei_paramcount(::Exponential) = 1
_focei_expected_information(d::Exponential) = fill(inv(d.θ * d.θ), 1, 1)

# Poisson(λ): ℐ = 1/λ
_focei_is_supported(::Poisson) = true
_focei_params(d::Poisson) = [d.λ]
_focei_paramcount(::Poisson) = 1
_focei_expected_information(d::Poisson) = fill(inv(d.λ), 1, 1)

# Bernoulli(p): ℐ = 1/(p(1-p))
_focei_is_supported(::Bernoulli) = true
_focei_params(d::Bernoulli) = [d.p]
_focei_paramcount(::Bernoulli) = 1
_focei_expected_information(d::Bernoulli) = fill(inv(d.p * (one(d.p) - d.p)), 1, 1)

# Binomial(n, p): n fixed; ℐ_pp = n/(p(1-p)).  Only p is a differentiated parameter.
_focei_is_supported(::Binomial) = true
_focei_params(d::Binomial) = [d.p]
_focei_paramcount(::Binomial) = 1
_focei_expected_information(d::Binomial) = fill(d.n * inv(d.p * (one(d.p) - d.p)), 1, 1)

# Geometric(p): ℐ = 1/(p²(1-p))
_focei_is_supported(::Geometric) = true
_focei_params(d::Geometric) = [d.p]
_focei_paramcount(::Geometric) = 1
_focei_expected_information(d::Geometric) = fill(inv(d.p * d.p * (one(d.p) - d.p)), 1, 1)

# --- two-parameter families without a clean location/dispersion split (FOCE ≡ FOCEI) ---

# Gamma(α, θ) [shape, scale]: ℐ = [[ψ'(α), 1/θ], [1/θ, α/θ²]]
_focei_is_supported(::Gamma) = true
_focei_params(d::Gamma) = [d.α, d.θ]
_focei_paramcount(::Gamma) = 2
function _focei_expected_information(d::Gamma)
    a = d.α
    θ = d.θ
    iθ = inv(θ)
    return [trigamma(a) iθ; iθ a*iθ*iθ]
end

# Beta(α, β): ℐ = [[ψ'(α)-ψ'(α+β), -ψ'(α+β)], [-ψ'(α+β), ψ'(β)-ψ'(α+β)]]
_focei_is_supported(::Beta) = true
_focei_params(d::Beta) = [d.α, d.β]
_focei_paramcount(::Beta) = 2
function _focei_expected_information(d::Beta)
    tab = trigamma(d.α + d.β)
    iaa = trigamma(d.α) - tab
    ibb = trigamma(d.β) - tab
    return [iaa -tab; -tab ibb]
end

# --- multivariate normal outcome ---

# MvNormal(μ, Σ): mean block Σ⁻¹, covariance block ½·tr(Σ⁻¹ Eₐ Σ⁻¹ E_b) in vech(Σ)
# coordinates, zero cross block.  The vech ordering matches `_vech` (column-major upper
# triangle).
_focei_is_supported(::MvNormal) = true
function _focei_params(d::MvNormal)
    # `cov` already materialises a dense matrix and `_vech` only reads it — a
    # converting `Matrix(...)` copy here doubled the allocation per observation.
    return vcat(collect(mean(d)), _vech(cov(d)))
end
function _focei_paramcount(d::MvNormal)
    k = length(d)
    return k + k * (k + 1) ÷ 2
end
# Symmetric single-entry basis matrices in vech (upper-triangle, column-major) order.
# Retained as the reference definition of the vech convention used by the closed-form
# `_focei_expected_information(::MvNormal)` below (and by its test); not on a hot path.
function _focei_vech_basis(k::Int)
    bases = Matrix{Float64}[]
    for j in 1:k
        for i in 1:j
            E = zeros(Float64, k, k)
            E[i, j] += 1.0
            E[j, i] += 1.0
            i == j && (E[i, j] = 1.0)  # diagonal entry appears once
            push!(bases, E)
        end
    end
    return bases
end
function _focei_expected_information(d::MvNormal)
    # Covariance block in closed form. With B_a = E_ij + E_ji (i<j) or E_ii (i==j),
    # tr(P E_rs P E_uv) = P[v,r]·P[s,u], so
    #   Fcov[a,b] = ½ Σ P-products over the 1–4 (r,s)×(u,v) combinations.
    # This replaces the previous per-call basis-matrix construction + nv² matrix
    # triple products (measured 18.7 KB / 3.2 μs per call at k=3, invoked once per
    # observation inside the Hessian assembly).
    # `Matrix(...)` is load-bearing: `cov(::MvNormal)` returns the AbstractPDMat
    # itself, and `inv(::PDMat)` is a Cholesky-based inverse — a different
    # algorithm (ulp-level different values) from the dense-LU `inv(::Matrix)`
    # this closed form was verified against.
    Σ = Matrix(cov(d))
    P = inv(Σ)
    k = size(Σ, 1)
    nv = k * (k + 1) ÷ 2
    T = eltype(P)
    F = zeros(T, k + nv, k + nv)
    @inbounds for j in 1:k, i in 1:k
        F[i, j] = P[i, j]
    end
    half = T(0.5)
    a = 0
    @inbounds for j1 in 1:k, i1 in 1:j1
        a += 1
        b = 0
        for j2 in 1:k, i2 in 1:j2
            b += 1
            s = P[j2, i1] * P[j1, i2]
            if i2 != j2
                s += P[i2, i1] * P[j1, j2]
            end
            if i1 != j1
                s += P[j2, j1] * P[i1, i2]
                if i2 != j2
                    s += P[i2, j1] * P[i1, j2]
                end
            end
            F[k + a, k + b] = half * s
        end
    end
    return F
end

# -------------------------------------------------------------------------------------
# 2. Per-observation distribution collection (mirrors `_loglikelihood_individual`,
#    returning the distribution objects instead of summing log-densities).
# -------------------------------------------------------------------------------------

# Shared per-individual scaffolding for the FOCEI observation collectors:
# solve the DE (if any) reusing the precomputed `pre`, hoist the row-constant
# formula context, and hand the row loop to `rows_f!` behind a function barrier
# (mirrors `_loglikelihood_individual`). `rows_f!` is called as
# `rows_f!(out, obs_f, ctx, sol_acc, const_cov, obs_series, vrows, obs_cols)`.
function _focei_collect_obs!(rows_f!::RF, out::Vector, dm::DataModel, idx::Int,
        θ, η_ind, cache::_LLCache) where {RF}
    model = dm.model
    ind = dm.individuals[idx]
    obs_rows = dm.row_groups.obs_rows[idx]
    const_cov = ind.const_cov
    obs_series = ind.series.obs
    vary_cache = cache.vary_cache === nothing ? nothing : cache.vary_cache[idx]
    if η_ind isa NamedTuple
        η_ind = ComponentArray(η_ind)
    end

    sol_accessors = nothing
    pre = nothing
    if model.de.de !== nothing
        pre = calculate_prede(model, θ, η_ind, const_cov)
        sol_accessors = _ll_solve_de(dm, idx, θ, η_ind, cache, pre)
        sol_accessors === nothing && return false
    end

    obs_cols = dm.config.obs_cols
    if _needs_rowwise_random_effects(dm, idx; obs_only = true)
        # Per-row η selection (rare; non-DE only).
        t_obs = vary_cache === nothing ? _get_col(dm.df, dm.config.time_col)[obs_rows] :
                nothing
        isempty(obs_rows) && return true
        row_tmpl = _row_re_template(dm, idx, 1, η_ind; obs_only = true)
        for i in eachindex(obs_rows)
            vary = vary_cache === nothing ? _varying_at(dm, ind, i, t_obs) : vary_cache[i]
            η_row = _row_random_effects_fill(dm, idx, i, η_ind, row_tmpl; obs_only = true)
            obs = sol_accessors === nothing ?
                  calculate_formulas_obs(model, θ, η_row, const_cov, vary) :
                  calculate_formulas_obs(model, θ, η_row, const_cov, vary, sol_accessors)
            for col in obs_cols
                y = getfield(obs_series, col)[i]
                y === missing && continue
                _focei_collect_push!(out, getproperty(obs, col))
            end
        end
        return true
    end

    pre === nothing && (pre = calculate_prede(model, θ, η_ind, const_cov))
    ctx = (; fixed_effects = θ, random_effects = η_ind, prede = pre,
        helpers = cache.helpers, model_funs = cache.model_funs)
    vrows = vary_cache !== nothing ? vary_cache :
            _build_vary_cache_individual(ind.series.vary, ind.series.dyn,
        _get_col(dm.df, dm.config.time_col)[obs_rows],
        length(obs_rows))
    sol_acc = sol_accessors === nothing ? NamedTuple() : sol_accessors
    rows_f!(out, model.formulas.obs, ctx, sol_acc, const_cov, obs_series, vrows, obs_cols)
    return true
end

# Element push dispatch: distribution objects for the dists collector, flattened
# native parameters for the typed Jacobian target.
@inline _focei_collect_push!(out::Vector{Any}, dist) = push!(out, dist)
@inline _focei_collect_push!(out::Vector{T}, dist) where {T <: Real} = append!(
    out, _focei_params(dist))

function _focei_rows_push!(out::Vector, obs_f::F, ctx::C, sol_acc::SA, const_cov::CC,
        obs_series::OS, vrows::Vector{V}, obs_cols) where {F, C, SA, CC, OS, V}
    for i in eachindex(vrows)
        obs = obs_f(ctx, sol_acc, const_cov, vrows[i])
        for col in obs_cols
            y = getfield(obs_series, col)[i]
            y === missing && continue
            _focei_collect_push!(out, getproperty(obs, col))
        end
    end
    return nothing
end

function _focei_collect_obs_dists!(
        out::Vector, dm::DataModel, idx::Int, θ, η_ind, cache::_LLCache)
    _focei_collect_obs!(_focei_rows_push!, out, dm, idx, θ, η_ind, cache)
end

# Returns a Vector of the outcome distributions for every (non-missing) observation in
# the batch, in a fixed order. Collected into `Vector{Any}`, then narrowed via
# `map(identity, ...)`: for the (typical) single-family model this yields a concretely
# typed vector, so every downstream `_focei_paramcount` / `_focei_diag_information` /
# `_focei_expected_information` call in the Hessian assembly dispatches statically —
# measured ~50× on `_focei_data_term` vs iterating the boxed `Vector{Any}`.
function _focei_obs_dists_batch(
        dm::DataModel, batch_info::_LaplaceBatchInfo, θ_re::ComponentArray,
        b, const_cache::LaplaceConstantsCache, cache::_LLCache)
    out = Vector{Any}()
    for i in batch_info.inds
        η_ind = _build_eta_ind(dm, i, batch_info, b, const_cache, θ_re)
        ok = _focei_collect_obs_dists!(out, dm, i, θ_re, η_ind, cache)
        ok || return Vector{Any}()
    end
    return map(identity, out)
end

# Stacked outcome-distribution parameters Φ(b) — the Jacobian target. Collected
# straight into a concretely-typed vector (this function runs under ForwardDiff
# duals once per Jacobian chunk; the old Vector{Any}-of-dists + reduce(vcat)
# route boxed every distribution and re-allocated the stack repeatedly).
function _focei_obs_params_batch(
        dm::DataModel, batch_info::_LaplaceBatchInfo, θ_re::ComponentArray,
        b, const_cache::LaplaceConstantsCache, cache::_LLCache)
    T = promote_type(eltype(θ_re), eltype(b))
    out = Vector{T}()
    for i in batch_info.inds
        η_ind = _build_eta_ind(dm, i, batch_info, b, const_cache, θ_re)
        ok = _focei_collect_obs!(_focei_rows_push!, out, dm, i, θ_re, η_ind, cache)
        ok || return Vector{T}()
    end
    return out
end

struct _FOCEIPhi{DM, INFO, TH, CC, CA}
    dm::DM
    info::INFO
    θ::TH
    const_cache::CC
    cache::CA
end
@inline (f::_FOCEIPhi)(b) = _focei_obs_params_batch(
    f.dm, f.info, f.θ, b, f.const_cache, f.cache)

struct _FOCEIPriorLogf{DM, INFO, TH, CC, CA}
    dm::DM
    info::INFO
    θ::TH
    const_cache::CC
    cache::CA
end
@inline (f::_FOCEIPriorLogf)(b) = _re_prior_logf_batch(
    f.dm, f.info, f.θ, b, f.const_cache, f.cache)

# -------------------------------------------------------------------------------------
# 3. Prior mean mᵢ in b-space (FOCE freezes dispersion parameters here).
# -------------------------------------------------------------------------------------

# Robustness wrapper: distribution construction can throw on a numerically
# degenerate θ the outer optimizer steps into (e.g. a singular Ω) — fall back to
# the zero vector so the subsequent evaluations return -Inf and the optimizer
# backtracks instead of crashing the fit (mirrors `_laplace_logf_batch`).
function _focei_prior_mean_b(
        dm::DataModel, batch_info::_LaplaceBatchInfo, θ_re::ComponentArray,
        const_cache::LaplaceConstantsCache, cache::_LLCache)
    try
        return _focei_prior_mean_b_impl(dm, batch_info, θ_re, const_cache, cache)
    catch err
        if _is_numeric_error(err)
            return zeros(eltype(θ_re), batch_info.n_b)
        end
        rethrow(err)
    end
end

function _focei_prior_mean_b_impl(
        dm::DataModel, batch_info::_LaplaceBatchInfo, θ_re::ComponentArray,
        const_cache::LaplaceConstantsCache, cache::_LLCache)
    nb = batch_info.n_b
    T = eltype(θ_re)
    m = zeros(T, nb)
    model_funs = cache.model_funs
    helpers = cache.helpers
    dists_builder = get_create_random_effect_distribution(dm.model.random.random)
    re_cache = dm.re_group_info.laplace_cache
    for (ri, re) in enumerate(re_cache.re_names)
        info = batch_info.re_info[ri]
        isempty(info.map.levels) && continue
        for (li, level_id) in enumerate(info.map.levels)
            idx = info.map.level_to_index[level_id]
            idx == 0 && continue
            rep_idx = info.reps[li]
            const_cov = dm.individuals[rep_idx].const_cov
            dists = dists_builder(θ_re, const_cov, model_funs, helpers)
            dist = getproperty(dists, re)
            r = info.ranges[idx]
            mv = mean(dist)
            if info.is_scalar
                m[first(r)] = mv isa AbstractVector ? mv[1] : mv
            else
                for (k, rr) in enumerate(r)
                    m[rr] = mv[k]
                end
            end
        end
    end
    return m
end

# -------------------------------------------------------------------------------------
# 4. The FOCEI / FOCE negative Hessian of the log-joint at (θ, b).
#    Returns the positive-(semi)definite precision 𝓗 = Σ Jⱼᵀ ℐ(φⱼ) Jⱼ − ∇²_b log π.
# -------------------------------------------------------------------------------------

# Diagonal Fisher information for the families where ℐ(φ) is diagonal — all the
# registered scalar families except Gamma and Beta (and the matrix-valued
# MvNormal). For these the data term Σⱼ Jⱼᵀ ℐⱼ Jⱼ collapses to Jᵀ(w .* J) with
# one weight per stacked parameter row: a single broadcast + GEMM instead of
# per-observation slices and matrix temporaries.
_focei_diag_information(d::Normal) = (iv = inv(d.σ * d.σ); (iv, 2iv))
_focei_diag_information(d::LogNormal) = (iv = inv(d.σ * d.σ); (iv, 2iv))
_focei_diag_information(d::Distributions.Laplace) = (ib = inv(d.θ * d.θ); (ib, ib))
_focei_diag_information(d::Cauchy) = (ih = inv(2 * d.σ * d.σ); (ih, ih))
_focei_diag_information(d::Exponential) = (inv(d.θ * d.θ),)
_focei_diag_information(d::Poisson) = (inv(d.λ),)
_focei_diag_information(d::Bernoulli) = (inv(d.p * (one(d.p) - d.p)),)
_focei_diag_information(d::Binomial) = (d.n * inv(d.p * (one(d.p) - d.p)),)
_focei_diag_information(d::Geometric) = (inv(d.p * d.p * (one(d.p) - d.p)),)
_focei_diag_information(::Any) = nothing

function _focei_data_term(J, dists_b::Vector, dists_m, interaction::Bool, T::Type)
    nb = size(J, 2)
    isempty(dists_b) && return zeros(T, nb, nb)
    # Diagonal-ℐ fast path. FOCE freezes dispersion parameters by zeroing their
    # rows of Jⱼ and evaluating ℐ at the prior-mean distribution; with diagonal ℐ
    # both effects reduce to the weight vector (Mᵀ ℐ M = diag of kept weights).
    w = Vector{T}(undef, size(J, 1))
    diag_ok = true
    offset = 0
    for (j, d) in enumerate(dists_b)
        if interaction
            # FOCEI: ℐ evaluated at the conditional-mode distribution, no
            # dispersion freeze — skip the (allocating) dispersion-index lookup.
            di = _focei_diag_information(d)
            if di === nothing
                diag_ok = false
                break
            end
            for k in eachindex(di)
                w[offset + k] = T(di[k])
            end
            offset += length(di)
        else
            disp = _focei_dispersion_indices(d)
            use_d = !isempty(disp) ? dists_m[j] : d
            di = _focei_diag_information(use_d)
            if di === nothing
                diag_ok = false
                break
            end
            for k in eachindex(di)
                w[offset + k] = k in disp ? zero(T) : T(di[k])
            end
            offset += length(di)
        end
    end
    if diag_ok && offset == size(J, 1)
        return transpose(J) * (w .* J)
    end
    # Non-diagonal fallback (Gamma/Beta/MvNormal, or FOCE-frozen dispersion):
    # accumulate Σⱼ Jⱼᵀ ℐⱼ Jⱼ into one preallocated buffer via 5-arg mul! over views —
    # no per-observation slice copies, no `push!`-and-`sum` of matrix temporaries.
    # The per-observation work lives in `_focei_data_term_obs!` so that mixed-family
    # batches (whose `dists_b` is abstractly typed) pay one runtime dispatch per
    # observation instead of one per field access / information call / matmul.
    H = zeros(T, nb, nb)
    tmp = Matrix{T}(undef, size(J, 1), nb)
    offset = 0
    for (j, d) in enumerate(dists_b)
        dmj = dists_m === nothing ? nothing : dists_m[j]
        offset = _focei_data_term_obs!(H, tmp, J, offset, d, dmj, interaction)
    end
    return H
end

# Containers cross the dynamic boundary as concrete Matrix/Int arguments — views
# are built inside, where every type is known (boxing a SubArray through dynamic
# dispatch costs more than the dispatch itself).
function _focei_data_term_obs!(
        H::Matrix, tmp::Matrix, J::Matrix, offset::Int, d, dmj, interaction::Bool)
    dj = _focei_paramcount(d)
    Jj = view(J, (offset + 1):(offset + dj), :)
    Tj = view(tmp, 1:dj, :)
    disp = _focei_dispersion_indices(d)
    if !interaction && !isempty(disp) && dmj !== nothing
        # FOCE: dispersion parameters frozen at the prior mean; their η-sensitivity
        # is dropped.  ℐ of the supported location-scale families depends only on the
        # dispersion parameters, so evaluating ℐ at the prior-mean distribution gives
        # the correct (location-independent) weight.  Zeroing the dispersion rows AND
        # columns of ℐ is algebraically identical to the historical row-masking of Jⱼ
        # ((M J)ᵀ ℐ (M J) = Jᵀ (M ℐ M) J with an exact 0/1 diagonal mask).
        # Every registered `_focei_expected_information` returns a fresh matrix, so
        # the dispersion mask is applied in place — no converting copy.
        Im = _focei_expected_information(dmj)
        z = zero(eltype(Im))
        for r in disp
            for c in 1:dj
                Im[r, c] = z
                Im[c, r] = z
            end
        end
        mul!(Tj, Im, Jj)
    else
        mul!(Tj, _focei_expected_information(d), Jj)
    end
    mul!(H, transpose(Jj), Tj, true, true)
    return offset + dj
end

# Robustness wrapper (mirrors `_laplace_logf_batch`): an ODE solve or distribution
# construction that throws on an unstable parameter region the optimizer steps into
# (e.g. a negative state driving a fractional power, or a negative scale) is turned into
# a NaN Hessian. That makes the Cholesky/log-det fail and the marginal -Inf, so the outer
# optimizer backtracks instead of crashing the whole fit — matching Laplace's behavior.
function _focei_negH_batch(dm::DataModel, batch_info::_LaplaceBatchInfo, θ, b,
        const_cache::LaplaceConstantsCache, cache::_LLCache;
        interaction::Bool, tctx = nothing)
    try
        return _focei_negH_batch_impl(dm, batch_info, θ, b, const_cache, cache;
            interaction = interaction, tctx = tctx)
    catch err
        if _is_numeric_error(err)
            nb = batch_info.n_b
            return fill(convert(promote_type(eltype(θ), eltype(b)), NaN), nb, nb)
        end
        rethrow(err)
    end
end

function _focei_negH_batch_impl(dm::DataModel, batch_info::_LaplaceBatchInfo, θ, b,
        const_cache::LaplaceConstantsCache, cache::_LLCache;
        interaction::Bool, tctx = nothing)
    θ_re = tctx === nothing ? _symmetrize_psd_params(θ, dm.model.fixed.fixed) : tctx.θ_re
    nb = batch_info.n_b
    T = promote_type(eltype(θ_re), eltype(b))
    nb == 0 && return zeros(T, 0, 0)

    dists_b = _focei_obs_dists_batch(dm, batch_info, θ_re, b, const_cache, cache)
    if isempty(dists_b)
        Hdata = zeros(T, nb, nb)
    else
        Φ = _FOCEIPhi(dm, batch_info, θ_re, const_cache, cache)
        J = ForwardDiff.jacobian(Φ, b)
        # Guard against an ODE solve that succeeds at the base point but reports failure
        # under the AD seed (length mismatch): signal a non-finite Hessian so the marginal
        # becomes -Inf and the outer optimizer backtracks out of the unstable region.
        D_expected = sum(_focei_paramcount, dists_b; init = 0)
        size(J, 1) == D_expected || return fill(convert(T, NaN), nb, nb)
        dists_m = nothing
        if !interaction && any(d -> !isempty(_focei_dispersion_indices(d)), dists_b)
            m_b = _focei_prior_mean_b(dm, batch_info, θ_re, const_cache, cache)
            dists_m = _focei_obs_dists_batch(dm, batch_info, θ_re, m_b, const_cache, cache)
        end
        Hdata = _focei_data_term(J, dists_b, dists_m, interaction, T)
    end

    # Prior curvature: for all-Gaussian REs it is constant in b and arrives
    # precomputed in the θ-context (its ∂/∂b is exactly zero either way, so the
    # cached Float64 matrix reproduces the per-call dual result bit-for-bit on
    # the b-differentiated paths). θ-dual callers pass tctx without a cached Λ
    # (or none at all), keeping ∂Λ/∂θ exact.
    Λ = if tctx !== nothing && tctx.prior_hess !== nothing
        tctx.prior_hess
    else
        -ForwardDiff.hessian(_FOCEIPriorLogf(dm, batch_info, θ_re, const_cache, cache), b)
    end
    return Hdata .+ Λ
end

# -------------------------------------------------------------------------------------
# 5. Hessian-builder plug-in for the shared Laplace objective/gradient machinery.
#
# FOCEI reuses Laplace's optimized marginal objective and trace-estimator gradient
# (config/buffer caching, the implicit db*/dθ term, batching, threading) verbatim,
# swapping only the inner Hessian via `_build_hess_b`.  The builder returns H = ∇²_b log f
# so that the downstream `negH = -H`; FOCEI's negH IS 𝓗 = Σ Jᵀ ℐ(φ) J − ∇²log π, hence it
# returns -𝓗.  Differentiating this first-order-AD Hessian for the log-determinant
# gradient is one AD order cheaper than Laplace's exact second-order Hessian, so the
# FOCEI marginal and its gradient are computed through the same fast code path.
# -------------------------------------------------------------------------------------

struct _FOCEIHess <: _HessMode
    interaction::Bool
end

@inline function _build_hess_b(m::_FOCEIHess, dm::DataModel, batch_info::_LaplaceBatchInfo,
        θ, b, const_cache::LaplaceConstantsCache, cache::_LLCache,
        ad_cache::Union{Nothing, LaplaceADCache}, bi::Int;
        ctx::AbstractString = "", tctx = nothing)
    return -_focei_negH_batch(dm, batch_info, θ, b, const_cache, cache;
        interaction = m.interaction, tctx = tctx)
end

# -------------------------------------------------------------------------------------
# 6. Method types.
# -------------------------------------------------------------------------------------

"""
    FOCEI(; interaction=true, optimizer, optim_kwargs, adtype, inner_*, multistart_*,
            jitter, max_tries, jitter_growth, adaptive_jitter, jitter_scale, theta_tol,
            lb, ub, ignore_model_bounds) <: FittingMethod

First-Order Conditional Estimation with Interaction for random-effects models.

`FOCEI` is the Laplace approximation with the inner negative Hessian of the log-joint
replaced by the expected-information form `Σ Jᵀ ℐ(φ) J − ∇²log π`, where `J = ∂φ/∂η` is a
first-order Jacobian of the outcome-distribution parameters and `ℐ(φ)` is the closed-form
Fisher information of the outcome family.  This drops the per-subject Hessian from
second-order to first-order automatic differentiation and yields a positive-definite
curvature by construction.

Set `interaction=false` for FOCE, which freezes dispersion-type parameters (e.g. a
residual-error standard deviation) at the random-effects prior mean and ignores their
dependence on the random effects.

Supported outcome families: `Normal`, `LogNormal`, `Laplace`, `Cauchy`, `Exponential`,
`Poisson`, `Bernoulli`, `Binomial`, `Geometric`, `Gamma`, `Beta`, `MvNormal`.  Hidden
Markov / Markov outcome models and any family without a registered Fisher information are
not supported — use [`Laplace`](@ref) for those.

Keyword arguments mirror [`Laplace`](@ref); the only addition is `interaction::Bool=true`.
"""
struct FOCEI{O, K, A, IO, HO, CO, MS, L, U} <: FittingMethod
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
    interaction::Bool
end

function FOCEI(; interaction::Bool = true,
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
        theta_tol = 0.0,
        lb = nothing,
        ub = nothing,
        ignore_model_bounds = false)
    inner = inner_options === nothing ?
            LaplaceInnerOptions(
        inner_optimizer, inner_kwargs, inner_adtype, inner_grad_tol) : inner_options
    hess = hessian_options === nothing ?
           LaplaceHessianOptions(
        jitter, max_tries, jitter_growth, adaptive_jitter, jitter_scale, true, false, 8) :
           hessian_options
    cache = cache_options === nothing ? LaplaceCacheOptions(theta_tol) : cache_options
    ms = multistart_options === nothing ?
         LaplaceMultistartOptions(multistart_n, multistart_k, multistart_grad_tol,
        multistart_max_rounds, multistart_sampling) : multistart_options
    FOCEI(optimizer, optim_kwargs, adtype, inner, hess, cache,
        ms, lb, ub, ignore_model_bounds, interaction)
end

# -------------------------------------------------------------------------------------
# 7. Distribution guard.
# -------------------------------------------------------------------------------------

function _focei_validate_distributions(
        dm::DataModel; ode_args::Tuple, ode_kwargs::NamedTuple)
    fe = dm.model.fixed.fixed
    θ0 = get_θ0_untransformed(fe)
    θ_re = _symmetrize_psd_params(θ0, fe)
    _, batch_infos, const_cache = _build_laplace_batch_infos(dm, NamedTuple())
    ll_cache = build_ll_cache(
        dm; ode_args = ode_args, ode_kwargs = ode_kwargs, force_saveat = true)
    for info in batch_infos
        info.n_b == 0 && continue
        b0 = zeros(Float64, info.n_b)
        dists = _focei_obs_dists_batch(dm, info, θ_re, b0, const_cache, ll_cache)
        isempty(dists) && continue
        for d in dists
            _focei_is_supported(d) ||
                error("FOCEI does not support outcome distribution $(typeof(d)). " *
                      "Supported families: Normal, LogNormal, Laplace, Cauchy, Exponential, " *
                      "Poisson, Bernoulli, Binomial, Geometric, Gamma, Beta, MvNormal. " *
                      "Use Laplace for hidden-Markov / unsupported outcomes.")
        end
        return nothing  # families are fixed by the formula; one observed batch suffices
    end
    return nothing
end

# -------------------------------------------------------------------------------------
# 8. Fit drivers.
# -------------------------------------------------------------------------------------

function _fit_model(dm::DataModel, method::FOCEI, args...;
        constants::NamedTuple = NamedTuple(),
        constants_re::NamedTuple = NamedTuple(),
        penalty::NamedTuple = NamedTuple(),
        ode_args::Tuple = (),
        ode_kwargs::NamedTuple = NamedTuple(),
        serialization::SciMLBase.EnsembleAlgorithm = EnsembleThreads(),
        rng::AbstractRNG = Xoshiro(0),
        theta_0_untransformed::Union{Nothing, ComponentArray} = nothing,
        store_data_model::Bool = true)
    fit_kwargs = (constants = constants, constants_re = constants_re, penalty = penalty,
        ode_args = ode_args, ode_kwargs = ode_kwargs, serialization = serialization,
        rng = rng, theta_0_untransformed = theta_0_untransformed, store_data_model = store_data_model)
    re_names = get_re_names(dm.model.random.random)
    isempty(re_names) &&
        error("FOCEI requires random effects. Use MLE/MAP for fixed-effects models.")
    fe = dm.model.fixed.fixed
    fixed_names = get_names(fe)
    isempty(fixed_names) && error("FOCEI requires at least one fixed effect.")
    fixed_set = Set(fixed_names)
    for name in keys(constants)
        name in fixed_set || error("Unknown constant parameter $(name).")
    end
    all(name in keys(constants) for name in fixed_names) &&
        error("FOCEI requires at least one free fixed effect. Remove constants or add a fixed/random effect.")

    # Shared driver (laplace.jl): FOCEI is Laplace with the Gauss-Newton /
    # Fisher-information inner Hessian, a hard NaN-backtrack policy, no
    # BlackBoxOptim support, and the closed-form-information outcome check.
    return _fit_laplace_family(dm, method, _FOCEIHess(method.interaction), args,
        fit_kwargs,
        d -> _focei_validate_distributions(d; ode_args = ode_args,
            ode_kwargs = ode_kwargs);
        nan_recovery = :backtrack, allow_bbo = false,
        constants = constants, constants_re = constants_re, penalty = penalty,
        ode_args = ode_args, ode_kwargs = ode_kwargs, serialization = serialization,
        rng = rng, theta_0_untransformed = theta_0_untransformed,
        store_data_model = store_data_model)
end
