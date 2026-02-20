# Internal helpers shared by MVDiscreteTimeHMM and MVContinuousTimeHMM.
# Not exported.

using Distributions, LinearAlgebra

# ---------------------------------------------------------------------------
# Number of outcomes
# ---------------------------------------------------------------------------

_mv_n_outcomes(dists::Tuple)                     = length(dists)
_mv_n_outcomes(dist::Distribution{Multivariate}) = length(dist)

# ---------------------------------------------------------------------------
# Emission logpdf
# ---------------------------------------------------------------------------

# Conditionally independent emissions: inner element is a Tuple of M scalar
# distributions. Missing entries contribute false (≡ 0, promotes to any
# numeric type, AD-safe).
_mv_emission_logpdf(dists::Tuple, y::AbstractVector) =
    sum(ismissing(y[m]) ? false : logpdf(dists[m], y[m]) for m in eachindex(y))

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
        "MvNormal emission distributions. Got $(typeof(dist)).")
    return logpdf(dist, collect(y))
end

# ---------------------------------------------------------------------------
# Emission mean
# ---------------------------------------------------------------------------

_mv_emission_mean(dists::Tuple)                     = [mean(dists[m]) for m in eachindex(dists)]
_mv_emission_mean(dist::Distribution{Multivariate}) = mean(dist)

# ---------------------------------------------------------------------------
# Emission covariance
# ---------------------------------------------------------------------------

# Independent case: diagonal matrix of per-outcome variances.
_mv_emission_cov(dists::Tuple)                     = Diagonal([var(dists[m]) for m in eachindex(dists)])
_mv_emission_cov(dist::Distribution{Multivariate}) = cov(dist)

# ---------------------------------------------------------------------------
# Emission rand
# ---------------------------------------------------------------------------

_mv_emission_rand(rng, dists::Tuple)                     = [rand(rng, dists[m]) for m in eachindex(dists)]
_mv_emission_rand(rng, dist::Distribution{Multivariate}) = rand(rng, dist)
