# Internal helpers shared by MVDiscreteTimeHMM and MVContinuousTimeHMM.
# Not exported.

using Distributions, LinearAlgebra

# ---------------------------------------------------------------------------
# Number of outcomes
# ---------------------------------------------------------------------------

_mv_n_outcomes(dists::Tuple) = length(dists)
_mv_n_outcomes(dist::Distribution{Multivariate}) = length(dist)

# ---------------------------------------------------------------------------
# Emission logpdf
# ---------------------------------------------------------------------------

# Conditionally independent emissions: inner element is a Tuple of M scalar
# distributions. Missing entries contribute false (≡ 0, promotes to any
# numeric type, AD-safe).
#
# The sum is unrolled at compile time into a straight-line left fold,
#     acc = reduce_first(add_sum, term(dists[1], y[1]))
#     acc = add_sum(acc, term(dists[m], y[m]))      for m = 2, ..., M
# with a literal index m, so the element type of `dists[m]` and of its logpdf
# is static at every step. (The former `sum(ntuple(m -> ... dists[m] ..., Val(M)))`
# compiled the closure for a runtime `m`; on a heterogeneous tuple such as
# 9 Bernoulli + 2 Normal that is not inferable, so every element and every
# ForwardDiff.Dual result was boxed.) The `false` contributions, the left-to-right
# summation order and the `reduce_first` seed are exactly those of `sum` over the
# former tuple, so values and result types are unchanged (an all-missing row
# still gives `0::Int`). A recursive fold is deliberately avoided: its accumulator
# is a small Union (missing vs. observed), which gets union-split at every
# recursive call and makes the inlined code grow exponentially in M.
@inline _mv_emission_term(d, ym) = ismissing(ym) ? false : logpdf(d, ym)

@generated function _mv_emission_logpdf(dists::Tuple, y::AbstractVector)
    M = fieldcount(dists)
    M == 0 && return :(throw(ArgumentError("empty tuple of emission distributions")))
    body = Any[:(acc = Base.reduce_first(Base.add_sum, _mv_emission_term(dists[1], y[1])))]
    for m in 2:M
        push!(body, :(acc = Base.add_sum(acc, _mv_emission_term(dists[$m], y[$m]))))
    end
    push!(body, :(return acc))
    return Expr(:block, body...)
end

# Joint MvNormal emission: handles missings via analytic marginalization.
function _mv_emission_logpdf(dist::MvNormal, y::AbstractVector)
    obs_idx = findall(!ismissing, y)
    isempty(obs_idx) && return false
    y_obs = [y[m] for m in obs_idx]
    length(obs_idx) == length(y) && return logpdf(dist, y_obs)
    μ_obs = dist.μ[obs_idx]
    Σ_obs = Matrix(dist.Σ)[obs_idx, obs_idx]
    return logpdf(MvNormal(μ_obs, Σ_obs), y_obs)
end

# Other multivariate distributions: missings not supported.
function _mv_emission_logpdf(dist::Distribution{Multivariate}, y::AbstractVector)
    any(ismissing, y) && error(
        "Missing observations in multivariate HMM are only supported for " *
            "MvNormal emission distributions. Got $(typeof(dist))."
    )
    return logpdf(dist, collect(y))
end

# Row-prepared variants: obs_idx/y_obs are state-independent (functions of y
# only), so the HMM callers compute them once per row and pass them in, sparing
# the per-state findall + copy in the joint MvNormal path. Values are unchanged.
function _mv_emission_logpdf(dists::Tuple, y::AbstractVector, obs_idx, y_obs)
    return _mv_emission_logpdf(dists, y)
end

function _mv_emission_logpdf(dist::MvNormal, y::AbstractVector, obs_idx, y_obs)
    isempty(obs_idx) && return false
    length(obs_idx) == length(y) && return logpdf(dist, y_obs)
    μ_obs = dist.μ[obs_idx]
    Σ_obs = Matrix(dist.Σ)[obs_idx, obs_idx]
    return logpdf(MvNormal(μ_obs, Σ_obs), y_obs)
end

function _mv_emission_logpdf(
        dist::Distribution{Multivariate}, y::AbstractVector, obs_idx, y_obs
    )
    return _mv_emission_logpdf(dist, y)
end

# ---------------------------------------------------------------------------
# Emission mean
# ---------------------------------------------------------------------------

_mv_emission_mean(dists::Tuple) = [mean(dists[m]) for m in eachindex(dists)]
_mv_emission_mean(dist::Distribution{Multivariate}) = mean(dist)

# ---------------------------------------------------------------------------
# Emission covariance
# ---------------------------------------------------------------------------

# Independent case: diagonal matrix of per-outcome variances.
_mv_emission_cov(dists::Tuple) = Diagonal([var(dists[m]) for m in eachindex(dists)])
_mv_emission_cov(dist::Distribution{Multivariate}) = cov(dist)

# ---------------------------------------------------------------------------
# Emission rand
# ---------------------------------------------------------------------------

_mv_emission_rand(rng, dists::Tuple) = [rand(rng, dists[m]) for m in eachindex(dists)]
_mv_emission_rand(rng, dist::Distribution{Multivariate}) = rand(rng, dist)

# A wrong-length observation used to surface as a BoundsError from inside an emission
# distribution (#207).
@inline function _mv_check_obs_length(hmm, y::AbstractVector)
    length(y) == hmm.n_outcomes ||
        error("Multivariate HMM expects an observation vector of length $(hmm.n_outcomes); got length $(length(y)).")
    return nothing
end
