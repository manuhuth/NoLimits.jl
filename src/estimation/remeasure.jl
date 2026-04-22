# remeasure.jl
# Random-effects measure abstraction for sparse-grid quadrature.
#
# Each subtype of AbstractREMeasure maps the standard-normal reference variable
# z ~ N(0, I) to the RE natural space η = transform(re, z), together with a
# log-correction factor log c(z) that accounts for any mismatch between the
# push-forward measure and the target RE prior.
#
# For GaussianRE (η = μ + Lz), the push-forward of N(0,I) IS the prior exactly,
# so logcorrection = 0 and the GH weights alone account for the prior measure.
#
# For NormalizingPlanarFlow RE (η = F(z), base N(0,I)):
#   p(η) * |det J_F(z)| = N(z;0,I) / |det J_F(z)| * |det J_F(z)| = N(z;0,I)
# The Jacobian cancels exactly, so logcorrection = 0 here too.

using LinearAlgebra
using Distributions
using Bijectors
using SpecialFunctions: loggamma

# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

abstract type AbstractREMeasure end

"""
    transform(re::AbstractREMeasure, z::AbstractVector) -> η::AbstractVector

Map standard-normal reference variable z → RE natural space η.
Must be ForwardDiff-compatible (non-mutating, no in-place operations on
argument arrays).
"""
function transform end

"""
    logcorrection(re::AbstractREMeasure, z::AbstractVector) -> scalar

Log correction term arising from the change of measure. For GaussianRE this
is zero; for transport maps it carries the log-Jacobian adjustment.
"""
function logcorrection end

# ---------------------------------------------------------------------------
# Phase 1: Gaussian RE measure  (η = μ + L z,  z ~ N(0, I))
# ---------------------------------------------------------------------------

"""
    GaussianRE{T, MT}

Batch-level Gaussian RE measure for Smolyak quadrature.

Fields:
- `μ`:   Concatenated prior means for all free RE levels in the batch, length n_b.
- `L`:   Block-diagonal lower-Cholesky factor (n_b × n_b) assembled from the
         individual per-level Cholesky factors. `LowerTriangular` wrapper.
- `n_b`: Total RE dimension of the batch.

The change of variables b = μ + L z with z ~ N(0,I) reproduces the joint
prior exactly, so the GH weights already account for the prior measure and
no correction factor is needed.
"""
struct GaussianRE{T<:Number, MT<:AbstractMatrix{T}} <: AbstractREMeasure
    μ::Vector{T}
    L::MT
    n_b::Int
end

transform(re::GaussianRE, z::AbstractVector) = re.μ + re.L * z
logcorrection(::GaussianRE, z::AbstractVector) = zero(eltype(z))
Base.eltype(::GaussianRE{T}) where T = T

# ---------------------------------------------------------------------------
# Build GaussianRE from a batch at current θ
# ---------------------------------------------------------------------------

"""
    build_gaussian_re_from_batch(batch_info, θ, const_cache, dm, ll_cache) -> GaussianRE

Construct the batch-level Gaussian RE measure for sparse-grid quadrature by
extracting prior means and Cholesky factors from the RE distributions implied
by the fixed-effects vector θ.

Only `Normal` and `MvNormal` RE distributions are supported in Phase 1.
Any other distribution type raises an informative error directing the user
to use `Laplace` instead or wait for Phase 2.

`θ` may carry ForwardDiff.Dual tags; all operations are non-mutating and
Dual-compatible for scalar Normal RE. MvNormal is Dual-compatible if and
only if the Cholesky decomposition in `PDMat` supports the element type
(holds for Float64; may require Phase 2 for Dual θ with MvNormal priors).
"""
function build_gaussian_re_from_batch(
    batch_info::_LaplaceBatchInfo,
    θ::ComponentArray,
    const_cache::LaplaceConstantsCache,
    dm::DataModel,
    ll_cache::_LLCache,
)
    n_b = batch_info.n_b
    n_b == 0 && error("build_gaussian_re_from_batch: batch has n_b == 0 (no free RE levels).")

    θ_re      = _symmetrize_psd_params(θ, dm.model.fixed.fixed)
    dists_b   = get_create_random_effect_distribution(dm.model.random.random)
    model_funs = ll_cache.model_funs
    helpers    = ll_cache.helpers
    re_cache   = dm.re_group_info.laplace_cache
    re_names   = re_cache.re_names

    # Collect (range, μ_seg, L_seg) for each free RE level
    μ_segs    = Vector{Any}()
    L_diags   = Vector{Any}()
    all_ranges = UnitRange{Int}[]

    for (ri, re) in enumerate(re_names)
        info = batch_info.re_info[ri]
        isempty(info.map.levels) && continue
        for (li, level_id) in enumerate(info.map.levels)
            rep_idx   = info.reps[li]
            const_cov = dm.individuals[rep_idx].const_cov
            dists     = dists_b(θ_re, const_cov, model_funs, helpers)
            dist      = getproperty(dists, re)
            range     = info.ranges[li]
            push!(all_ranges, range)

            if dist isa Distributions.Normal
                μ_k = [Distributions.mean(dist)]   # [T]  — 1-vector
                L_k = reshape([Distributions.std(dist)], 1, 1)  # [T;;]  — 1×1
            elseif dist isa Distributions.MvNormal
                μ_k = Vector(Distributions.mean(dist))
                # L is the lower-Cholesky factor of the covariance.
                # PDMat stores the pre-computed Cholesky; extract it to avoid
                # recomputing and for Dual compatibility when dist.Σ.chol exists.
                Σ = dist.Σ
                if Σ isa Distributions.PDMats.PDMat && isdefined(Σ, :chol)
                    L_k = Matrix(Σ.chol.L)
                else
                    # Fallback: compute Cholesky from the raw matrix.
                    # Note: this path fails when θ carries ForwardDiff.Dual tags
                    # and the covariance matrix was built via PDMat (uses LAPACK).
                    # Phase 2 will add a custom Cholesky rule to handle this.
                    Σ_mat = Σ isa AbstractMatrix ? Σ : Matrix(Σ)
                    L_k   = Matrix(cholesky(Symmetric(Σ_mat + 1e-12 * I)).L)
                end
            else
                error(
                    "GHQuadrature Phase 1 supports only Normal and MvNormal " *
                    "random effects. Found distribution of type " *
                    "$(typeof(dist)) for RE '$(re)'. " *
                    "Use Laplace for non-Gaussian RE distributions, or " *
                    "wait for GHQuadrature Phase 2 which will add transport " *
                    "maps for arbitrary distributions."
                )
            end
            push!(μ_segs, μ_k)
            push!(L_diags, L_k)
        end
    end

    # Infer element type from accumulated segments (handles Dual promotion)
    T = mapreduce(eltype, promote_type, μ_segs; init=Float64)
    T = mapreduce(eltype, promote_type, L_diags; init=T)

    # Assemble μ_full (concatenated means)
    μ_full = Vector{T}(undef, n_b)
    for (range, μ_k) in zip(all_ranges, μ_segs)
        μ_full[range] .= μ_k
    end

    # Assemble block-diagonal L_full
    L_full = zeros(T, n_b, n_b)
    for (range, L_k) in zip(all_ranges, L_diags)
        L_full[range, range] = L_k
    end

    return GaussianRE{T, LowerTriangular{T, Matrix{T}}}(
        μ_full, LowerTriangular(L_full), n_b
    )
end

# ---------------------------------------------------------------------------
# CompositeRE: general segment-by-segment transform for mixed RE batches
# ---------------------------------------------------------------------------

"""
    CompositeRE{T<:Number} <: AbstractREMeasure

General RE measure for batches that contain a mix of Gaussian and
NormalizingPlanarFlow (NPF) random effects, or pure-NPF batches.

Each free RE level in the batch is assigned a segment function
`segment_fns[k]` that maps `z[ranges[k]] → η_k`.  The full transform
concatenates the per-segment outputs:

    transform(re, z) = vcat(f_1(z[r_1]), f_2(z[r_2]), ..., f_K(z[r_K]))

`logcorrection = 0` for all supported segment types:
- Gaussian (`f_k(z_k) = μ_k + L_k z_k`): GH weights absorb `N(z_k; 0, I)` exactly.
- NPF (`f_k(z_k) = F_k(z_k)`, base `N(0,I)`): flow Jacobian cancels prior ratio.

The type parameter `T` tracks the element type of the accumulator (may be a
`ForwardDiff.Dual` when `θ` carries gradient tags).
"""
struct CompositeRE{T<:Number} <: AbstractREMeasure
    segment_fns::Vector{Any}          # f_k : ℝ^dk → ℝ^dk
    correction_fns::Vector{Any}       # c_k : ℝ^dk → scalar, or nothing for zero correction
    ranges::Vector{UnitRange{Int}}    # z-index range for each segment
    n_b::Int
    has_correction::Bool              # fast-path flag: false if all corrections are nothing
end

function transform(re::CompositeRE, z::AbstractVector)
    parts = [re.segment_fns[k](z[re.ranges[k]]) for k in eachindex(re.segment_fns)]
    return reduce(vcat, parts)
end

function logcorrection(re::CompositeRE, z::AbstractVector)
    re.has_correction || return zero(eltype(z))
    total = zero(eltype(z))
    for k in eachindex(re.correction_fns)
        re.correction_fns[k] === nothing && continue
        total += re.correction_fns[k](z[re.ranges[k]])
    end
    return total
end

Base.eltype(::CompositeRE{T}) where T = T

# ---------------------------------------------------------------------------
# build_re_measure_from_batch: handles Normal, MvNormal, and NormalizingPlanarFlow
# ---------------------------------------------------------------------------

"""
    build_re_measure_from_batch(batch_info, θ, const_cache, dm, ll_cache) -> AbstractREMeasure

Construct the batch-level RE measure for sparse-grid quadrature.

- Pure `Normal`/`MvNormal` batches → returns a [`GaussianRE`](@ref) (fast path).
- Any non-Gaussian RE (pure or mixed) → returns a [`CompositeRE`](@ref) where each
  segment uses the appropriate change-of-variables transport.

**Change-of-variables summary (by support):**

*Support ℝ — identity transport T(z) = z:*
- `Normal`/`MvNormal`: correction = 0 (GH weights absorb N(z;0,I) exactly).
- `TDist(ν)`: correction = logpdf(TDist(ν),z) + z²/2 + log(2π)/2.
- Any `ContinuousUnivariateDistribution` with ℝ support: same formula.

*Support (0,∞) — exp transport T(z) = exp(z):*
- `LogNormal(μ,σ)`: correction = 0 (push-forward of identity-scaled normal IS LogNormal).
- `Gamma(α,θ)`: correction = logpdf(Gamma,exp(z)) + z + z²/2 + log(2π)/2.
- `Exponential(θ)`: same as Gamma(1,θ), correction simplified accordingly.
- `Weibull(α,θ)`: correction = logpdf(Weibull,exp(z)) + z + z²/2 + log(2π)/2.
- Any `ContinuousUnivariateDistribution` with (0,∞) support: same formula.

*Support (a,b) finite — scaled-logistic transport T(z) = a + (b-a)σ(z):*
- `Beta(α,β)`: correction = logpdf(Beta,σ(z)) + log σ(z) + log(1-σ(z)) + z²/2 + log(2π)/2.
- Any `ContinuousUnivariateDistribution` with finite (a,b) support: same formula.

*Flows:*
- `NormalizingPlanarFlow`: correction = 0 (flow Jacobian cancels prior ratio).

**Generic fallback:** Any `ContinuousUnivariateDistribution` not matched by the explicit
branches above is handled via the generic transport, with `Distributions.logpdf` called
inside the correction closure. ForwardDiff compatibility depends on the distribution's
own logpdf implementation (works for all standard Distributions.jl types).

**ForwardDiff compatibility:** All segment types support Dual θ. `MvNormal` has the
same `PDMat` caveat as in `build_gaussian_re_from_batch`.
"""
function build_re_measure_from_batch(
    batch_info::_LaplaceBatchInfo,
    θ::ComponentArray,
    const_cache::LaplaceConstantsCache,
    dm::DataModel,
    ll_cache::_LLCache,
)
    n_b = batch_info.n_b
    n_b == 0 && error("build_re_measure_from_batch: batch has n_b == 0 (no free RE levels).")

    θ_re       = _symmetrize_psd_params(θ, dm.model.fixed.fixed)
    dists_b    = get_create_random_effect_distribution(dm.model.random.random)
    model_funs = ll_cache.model_funs
    helpers    = ll_cache.helpers
    re_cache   = dm.re_group_info.laplace_cache
    re_names   = re_cache.re_names

    # Accumulators shared with GaussianRE fast path
    μ_segs        = Vector{Any}()
    L_diags       = Vector{Any}()
    all_ranges    = UnitRange{Int}[]
    segment_fns   = Any[]
    correction_fns = Any[]
    has_npf        = false
    has_correction = false

    for (ri, re) in enumerate(re_names)
        info = batch_info.re_info[ri]
        isempty(info.map.levels) && continue
        for (li, _) in enumerate(info.map.levels)
            rep_idx   = info.reps[li]
            const_cov = dm.individuals[rep_idx].const_cov
            dists     = dists_b(θ_re, const_cov, model_funs, helpers)
            dist      = getproperty(dists, re)
            range     = info.ranges[li]
            push!(all_ranges, range)

            if dist isa Distributions.Normal
                μ_k = [Distributions.mean(dist)]
                L_k = reshape([Distributions.std(dist)], 1, 1)
                push!(μ_segs, μ_k); push!(L_diags, L_k)
                let μ = μ_k, L = L_k
                    push!(segment_fns, z_k -> μ .+ L * z_k)
                end
                push!(correction_fns, nothing)

            elseif dist isa Distributions.MvNormal
                μ_k = Vector(Distributions.mean(dist))
                Σ = dist.Σ
                if Σ isa Distributions.PDMats.PDMat && isdefined(Σ, :chol)
                    L_k = Matrix(Σ.chol.L)
                else
                    Σ_mat = Σ isa AbstractMatrix ? Σ : Matrix(Σ)
                    L_k   = Matrix(cholesky(Symmetric(Σ_mat + 1e-12 * I)).L)
                end
                push!(μ_segs, μ_k); push!(L_diags, L_k)
                let μ = μ_k, L = L_k
                    push!(segment_fns, z_k -> μ .+ L * z_k)
                end
                push!(correction_fns, nothing)

            elseif dist isa Distributions.LogNormal
                # η = exp(μ_log + σ_log * z), z ~ N(0,1)
                # Push-forward of N(0,1) under this map IS LogNormal(μ_log, σ_log),
                # so the GH weights integrate exactly and logcorrection = 0.
                μ_log, σ_log = Distributions.params(dist)
                push!(μ_segs, [μ_log]); push!(L_diags, reshape([σ_log], 1, 1))
                let μ = μ_log, σ = σ_log
                    push!(segment_fns, z_k -> [exp(μ + σ * z_k[1])])
                end
                push!(correction_fns, nothing)

            elseif dist isa Distributions.Beta
                # η = logistic(z), z ~ N(0,1) reference.
                # p(η) = Beta(α,β); Jacobian = η(1-η).
                # logcorrection = logpdf(Beta(α,β),η) + log η + log(1-η) + z²/2 + log(2π)/2
                has_npf = true   # forces CompositeRE path
                has_correction = true
                α_k, β_k = Distributions.params(dist)
                let a = α_k, b = β_k
                    push!(segment_fns, z_k -> begin
                        T_z = eltype(z_k)
                        [one(T_z) / (one(T_z) + exp(-z_k[1]))]
                    end)
                    push!(correction_fns, z_k -> begin
                        zz = z_k[1]
                        T_z = promote_type(eltype(z_k), typeof(a), typeof(b))
                        p   = one(T_z) / (one(T_z) + exp(-convert(T_z, zz)))
                        lp  = log(p)
                        lmp = log(one(T_z) - p)
                        (a * lp + b * lmp
                            - (loggamma(a) + loggamma(b) - loggamma(a + b))
                            + zz^2 / 2
                            + log(convert(T_z, 2π)) / 2)
                    end)
                end
                push!(μ_segs, nothing); push!(L_diags, nothing)

            elseif dist isa Distributions.Gamma
                # η = exp(z), z ~ N(0,1).  Transport: ℝ → (0,∞).
                # logcorrection = logpdf(Gamma(α,θ), exp(z)) + z + z²/2 + log(2π)/2
                #                = (α-1)z - exp(z)/θ - loggamma(α) - α log θ + z + z²/2 + log(2π)/2
                has_npf = true; has_correction = true
                α_k, θ_k = Distributions.params(dist)   # shape, scale
                let a = α_k, b = θ_k
                    push!(segment_fns, z_k -> [exp(z_k[1])])
                    push!(correction_fns, z_k -> begin
                        zz = z_k[1]
                        T_z = promote_type(eltype(z_k), typeof(a), typeof(b))
                        η   = exp(convert(T_z, zz))
                        (convert(T_z, a) - one(T_z)) * convert(T_z, zz) - η / convert(T_z, b) -
                            loggamma(convert(T_z, a)) - convert(T_z, a) * log(convert(T_z, b)) +
                            convert(T_z, zz) + convert(T_z, zz)^2 / 2 + log(convert(T_z, 2π)) / 2
                    end)
                end
                push!(μ_segs, nothing); push!(L_diags, nothing)

            elseif dist isa Distributions.Exponential
                # η = exp(z), z ~ N(0,1).  Exponential(θ) = Gamma(1, θ).
                # logcorrection = -exp(z)/θ - log θ + z + z²/2 + log(2π)/2
                has_npf = true; has_correction = true
                θ_k = Distributions.scale(dist)
                let b = θ_k
                    push!(segment_fns, z_k -> [exp(z_k[1])])
                    push!(correction_fns, z_k -> begin
                        zz = z_k[1]
                        T_z = promote_type(eltype(z_k), typeof(b))
                        η   = exp(convert(T_z, zz))
                        -η / convert(T_z, b) - log(convert(T_z, b)) +
                            convert(T_z, zz) + convert(T_z, zz)^2 / 2 + log(convert(T_z, 2π)) / 2
                    end)
                end
                push!(μ_segs, nothing); push!(L_diags, nothing)

            elseif dist isa Distributions.Weibull
                # η = exp(z), z ~ N(0,1).  Transport: ℝ → (0,∞).
                # logcorrection = logpdf(Weibull(α,θ), exp(z)) + z + z²/2 + log(2π)/2
                #                = log α - α log θ + (α-1)z - (exp(z)/θ)^α + z + z²/2 + log(2π)/2
                has_npf = true; has_correction = true
                α_k, θ_k = Distributions.params(dist)   # shape, scale
                let a = α_k, b = θ_k
                    push!(segment_fns, z_k -> [exp(z_k[1])])
                    push!(correction_fns, z_k -> begin
                        zz = z_k[1]
                        T_z = promote_type(eltype(z_k), typeof(a), typeof(b))
                        log(convert(T_z, a)) - convert(T_z, a) * log(convert(T_z, b)) +
                            (convert(T_z, a) - one(T_z)) * convert(T_z, zz) -
                            (exp(convert(T_z, zz)) / convert(T_z, b))^convert(T_z, a) +
                            convert(T_z, zz) + convert(T_z, zz)^2 / 2 + log(convert(T_z, 2π)) / 2
                    end)
                end
                push!(μ_segs, nothing); push!(L_diags, nothing)

            elseif dist isa Distributions.TDist
                # η = z (identity), z ~ N(0,1) reference.  Support: ℝ.
                # logcorrection = logpdf(TDist(ν), z) + z²/2 + log(2π)/2
                #                = loggamma((ν+1)/2) - loggamma(ν/2) - log(νπ)/2
                #                  - (ν+1)/2 * log(1 + z²/ν)  + z²/2 + log(2π)/2
                has_npf = true; has_correction = true
                ν_k = Distributions.dof(dist)
                let ν = ν_k
                    push!(segment_fns, z_k -> [z_k[1]])
                    push!(correction_fns, z_k -> begin
                        zz  = z_k[1]
                        T_z = promote_type(eltype(z_k), typeof(ν))
                        nν  = convert(T_z, ν)
                        loggamma((nν + one(T_z)) / 2) - loggamma(nν / 2) -
                            log(nν * convert(T_z, π)) / 2 -
                            (nν + one(T_z)) / 2 * log(one(T_z) + convert(T_z, zz)^2 / nν) +
                            convert(T_z, zz)^2 / 2 + log(convert(T_z, 2π)) / 2
                    end)
                end
                push!(μ_segs, nothing); push!(L_diags, nothing)

            elseif dist isa Distributions.ContinuousUnivariateDistribution
                # Generic fallback for any continuous univariate distribution.
                # Transport is chosen by support:
                #   ℝ         → identity   (T(z) = z,        log|T'| = 0)
                #   (0,∞)     → exp        (T(z) = exp(z),   log|T'| = z)
                #   (a,b) fin → scaled σ   (T(z) = a+(b-a)σ(z), log|T'| = log(b-a)+log σ+log(1-σ))
                # logcorrection = logpdf(dist, T(z)) + log|T'(z)| + z²/2 + log(2π)/2
                # Note: ForwardDiff compatibility depends on Distributions.jl's logpdf
                # for the specific distribution type (works for most built-in types).
                has_npf = true; has_correction = true
                lo, hi = Distributions.minimum(dist), Distributions.maximum(dist)
                if lo == -Inf && hi == Inf
                    # Identity transport
                    let d = dist
                        push!(segment_fns, z_k -> [z_k[1]])
                        push!(correction_fns, z_k -> begin
                            zz  = z_k[1]; T_z = eltype(z_k)
                            Distributions.logpdf(d, zz) +
                                convert(T_z, zz)^2 / 2 + log(convert(T_z, 2π)) / 2
                        end)
                    end
                elseif lo == 0 && hi == Inf
                    # Exp transport
                    let d = dist
                        push!(segment_fns, z_k -> [exp(z_k[1])])
                        push!(correction_fns, z_k -> begin
                            zz = z_k[1]; T_z = eltype(z_k)
                            η  = exp(convert(T_z, zz))
                            Distributions.logpdf(d, η) +
                                convert(T_z, zz) + convert(T_z, zz)^2 / 2 + log(convert(T_z, 2π)) / 2
                        end)
                    end
                elseif isfinite(lo) && isfinite(hi)
                    # Scaled logistic transport
                    let d = dist, a = Float64(lo), b_hi = Float64(hi)
                        push!(segment_fns, z_k -> begin
                            T_z = eltype(z_k)
                            [convert(T_z, a) + (convert(T_z, b_hi) - convert(T_z, a)) /
                                (one(T_z) + exp(-z_k[1]))]
                        end)
                        push!(correction_fns, z_k -> begin
                            zz  = z_k[1]; T_z = eltype(z_k)
                            σz  = one(T_z) / (one(T_z) + exp(-convert(T_z, zz)))
                            η   = convert(T_z, a) + (convert(T_z, b_hi) - convert(T_z, a)) * σz
                            Distributions.logpdf(d, η) +
                                log(convert(T_z, b_hi - a)) + log(σz) + log(one(T_z) - σz) +
                                convert(T_z, zz)^2 / 2 + log(convert(T_z, 2π)) / 2
                        end)
                    end
                else
                    error(
                        "build_re_measure_from_batch: unsupported support for distribution " *
                        "$(typeof(dist)) (lo=$lo, hi=$hi). " *
                        "GHQuadrature supports ℝ, (0,∞), and finite (a,b) supports."
                    )
                end
                push!(μ_segs, nothing); push!(L_diags, nothing)

            elseif dist isa AbstractNormalizingFlow
                has_npf = true
                bij = Bijectors.bijector(dist)   # flow transform F: ℝ^d → ℝ^d
                let b = bij
                    push!(segment_fns, z_k -> b(z_k))
                end
                push!(correction_fns, nothing)
                push!(μ_segs, nothing); push!(L_diags, nothing)

            else
                error(
                    "build_re_measure_from_batch: unsupported RE distribution type " *
                    "$(typeof(dist)) for RE '$(re)'. " *
                    "Supported: Normal, MvNormal, LogNormal, Beta, NormalizingPlanarFlow."
                )
            end
        end
    end

    if !has_npf
        # Fast path: all Gaussian → GaussianRE (single matrix multiply)
        T = mapreduce(eltype, promote_type, μ_segs; init=Float64)
        T = mapreduce(eltype, promote_type, L_diags; init=T)
        μ_full = Vector{T}(undef, n_b)
        L_full = zeros(T, n_b, n_b)
        for (range, μ_k, L_k) in zip(all_ranges, μ_segs, L_diags)
            μ_full[range] .= μ_k
            L_full[range, range] = L_k
        end
        return GaussianRE{T, LowerTriangular{T, Matrix{T}}}(
            μ_full, LowerTriangular(L_full), n_b)
    end

    # CompositeRE path: determine element type from non-nothing segments + θ_re
    non_nothing_μ = filter(!isnothing, μ_segs)
    non_nothing_L = filter(!isnothing, L_diags)
    T = isempty(non_nothing_μ) ? eltype(θ_re) :
        mapreduce(eltype, promote_type, non_nothing_μ; init=eltype(θ_re))
    T = isempty(non_nothing_L) ? T :
        mapreduce(eltype, promote_type, non_nothing_L; init=T)
    return CompositeRE{T}(segment_fns, correction_fns, all_ranges, n_b, has_correction)
end

# ---------------------------------------------------------------------------
# CenteredREMeasure: AGHQ measure centred at EBE mode b* with local curvature
# ---------------------------------------------------------------------------

"""
    CenteredREMeasure{T, LT, F} <: AbstractREMeasure

Adaptive Gauss-Hermite Quadrature (AGHQ) measure for a batch of random effects.

Unlike `GaussianRE` (which centres nodes at the prior mean μ), this measure
centres nodes at the empirical-Bayes mode **b***, scaled by
**S = chol(-H)^{-1}** where H is the Hessian of log p(b | y, θ) at b*.

The change-of-variables correction is

    logcorrection(z) = log p(b* + Sz | θ) + log|det(S)| + ½‖z‖² + ½d·log(2π)

which accounts for the RE prior density, the Jacobian of the mapping, and
the division by the N(0,I) quadrature measure.

For a Gaussian prior this reduces to `logcorrection = 0` as expected.
"""
struct CenteredREMeasure{T<:Number, LT<:LowerTriangular{T, Matrix{T}}, F} <: AbstractREMeasure
    b_star        :: Vector{T}    # EBE mode (free RE space)
    S             :: LT           # chol(-H)^{-1}: S*S' ≈ posterior covariance
    log_det_S     :: T            # log|det(S)|
    re_prior_logf :: F            # closure: b -> Σ_levels log p(b_level | θ)
    n_b           :: Int
end

transform(re::CenteredREMeasure, z::AbstractVector) = re.b_star + re.S * z
function logcorrection(re::CenteredREMeasure, z::AbstractVector)
    b = re.b_star + re.S * z
    re.re_prior_logf(b) + re.log_det_S + 0.5 * sum(abs2, z) + 0.5 * length(z) * log(2π)
end
Base.eltype(::CenteredREMeasure{T}) where T = T

"""
    build_centered_re_measure(b_star, batch_info, bi, θu, const_cache, dm, ll_cache; jitter, max_tries)

Build a `CenteredREMeasure` for AGHQ, centering the quadrature nodes at `b_star`
and scaling by the inverse Cholesky of the negative log-posterior Hessian at b*.

Returns `nothing` if the Cholesky factorization of `-H` fails (e.g. due to a
near-flat or indefinite Hessian). The caller is responsible for handling this case,
for example by falling back to a sampling-based marginal likelihood estimator.
"""
function build_centered_re_measure(
    b_star::AbstractVector,
    batch_info::_LaplaceBatchInfo,
    bi::Int,
    θu::ComponentArray,
    const_cache,
    dm::DataModel,
    ll_cache::_LLCache;
    jitter::Float64=1e-6,
    max_tries::Int=6,
)
    T = eltype(b_star)
    H = _laplace_hessian_b(dm, batch_info, θu, b_star, const_cache, ll_cache, nothing, bi)
    chol, _ = _laplace_cholesky_negH(H; jitter=jitter, max_tries=max_tries)
    (chol === nothing || chol.info != 0) && return nothing
    L = chol.L  # lower triangular, L*L' = -H
    S = LowerTriangular(inv(Matrix(L)))
    log_det_S = -sum(log, diag(L))
    re_prior_logf = b -> _re_prior_logf_batch(dm, batch_info, θu, b, const_cache, ll_cache)
    return CenteredREMeasure(Vector{T}(b_star), S, T(log_det_S), re_prior_logf, batch_info.n_b)
end

# ---------------------------------------------------------------------------
# Pre-validation: check all RE distributions are supported
# ---------------------------------------------------------------------------

"""
    _ghq_validate_re_distributions(dm::DataModel)

Throw an informative error if the model contains random effects whose
distribution type cannot be handled by GHQuadrature.

GHQuadrature supports all `ContinuousUnivariateDistribution` types (via explicit
or generic transport maps) plus `MvNormal` and `NormalizingPlanarFlow`.
Discrete distributions and unsupported multivariate types are rejected here.

Called once at the start of `_fit_model(::GHQuadrature, ...)` before any
expensive computation.
"""
function _ghq_validate_re_distributions(dm::DataModel)
    re   = dm.model.random.random
    re_t = get_re_types(re)
    isempty(re_t) && return

    # Distributions known to be unsupported (discrete or multivariate non-Gaussian).
    # Continuous univariate distributions not in this list are handled by the
    # generic fallback in build_re_measure_from_batch.
    explicitly_unsupported = (
        :Bernoulli, :Binomial, :Categorical, :DiscreteUniform, :Geometric,
        :Hypergeometric, :NegativeBinomial, :Poisson, :Skellam,
        :Dirichlet, :Multinomial, :MvLogNormal,
    )
    bad = Symbol[]
    for (name, dtype) in Base.pairs(re_t)
        if dtype in explicitly_unsupported
            push!(bad, name)
        end
    end
    if !isempty(bad)
        names_str = join(string.(bad), ", ")
        types_str = join([string(get_re_types(re)[n]) for n in bad], ", ")
        error(
            "GHQuadrature does not support discrete or unsupported multivariate " *
            "RE distributions.\n" *
            "Unsupported RE(s): $(names_str) (type(s): $(types_str)).\n" *
            "GHQuadrature supports all continuous univariate distributions, " *
            "MvNormal, and NormalizingPlanarFlow."
        )
    end
end
