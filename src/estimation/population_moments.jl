export population_moment_term
export ensemble_moments
export fixed_re_normals

using Random

# ============================================================================
# Population-moment likelihood terms (population-average / single-cell-snapshot).
#
# These build an objective term — a closure `θu -> Real` suitable for the
# `extra_objective` keyword of `fit_model` — that matches aggregate data (a
# time series of population means, optionally variances) to the moments of a
# model observable taken OVER the random-effect distribution. This is the
# nonlinear-mixed-effects treatment of bulk (population-average) and snapshot
# (single-cell-snapshot / FACS) data: the prediction is E[g(β + b)] (and, for
# snapshot data, Var[g(β + b)]) with b ~ N(0, D).
#
# The term depends only on the population parameters (β and the covariance D),
# so it composes with any point-estimate / EM estimator via `extra_objective`
# and shares the same θ as, e.g., a single-cell-time-lapse Laplace fit.
# ============================================================================

"""
    fixed_re_normals(n_re, n_samples; rng = Random.default_rng()) -> Matrix{Float64}

Draw an `n_re × n_samples` matrix of standard normals ONCE, to be reused across the
whole fit. Each objective evaluation forms the random-effect draws as `b = D^{1/2} r`
from the *current* covariance `D` (see [`ensemble_moments`](@ref)); keeping the
underlying `r` fixed makes the Monte-Carlo moment estimates a smooth, differentiable
function of the parameters (no resampling noise between optimizer steps).
"""
fixed_re_normals(n_re::Int, n_samples::Int; rng::AbstractRNG = Random.default_rng()) = randn(
    rng, n_re, n_samples)

"""
    ensemble_moments(simulate, θu, re_half, R) -> (μ, Σ)

Monte-Carlo estimate of the per-time mean `μ` and variance `Σ` of a model observable
over the random-effect distribution.

- `simulate(θu, b) -> AbstractVector`: the observable time series for a single cell
  whose random-effect realization is `b` (length `n_re`).
- `re_half(θu)`: maps the current parameters to `D^{1/2}` — either a length-`n_re`
  vector (interpreted as the diagonal of `D^{1/2}`, i.e. independent random effects)
  or an `n_re × n_re` matrix (a Cholesky/matrix square root for correlated effects).
- `R`: the fixed `n_re × n_samples` draw matrix from [`fixed_re_normals`](@ref).

Returns vectors `μ` and `Σ` (unbiased sample variance, `1/(n_s-1)`). Fully compatible
with ForwardDiff through `θu`.
"""
# draw b = D^{1/2} r for sample column l — dispatched (not branched) on the shape of
# D^{1/2}: a vector (diagonal, independent REs) or a matrix (Cholesky factor).
@inline _ens_draw(Dh::AbstractMatrix, R, l) = Dh * view(R, :, l)
@inline _ens_draw(Dh::AbstractVector, R, l) = Dh .* view(R, :, l)

function ensemble_moments(simulate, θu, re_half, R::AbstractMatrix)
    Dh = re_half(θu)
    n_s = size(R, 2)
    # single-pass Welford accumulation: O(n_time) memory (not O(n_samples·n_time)),
    # numerically stable, and ForwardDiff-compatible (in-place dual accumulators).
    y1 = simulate(θu, _ens_draw(Dh, R, 1))
    nt = length(y1)
    μ = collect(y1)
    M2 = zeros(eltype(μ), nt)
    @inbounds for l in 2:n_s
        y = simulate(θu, _ens_draw(Dh, R, l))
        for j in 1:nt
            d = y[j] - μ[j]
            μ[j] += d / l
            M2[j] += d * (y[j] - μ[j])
        end
    end
    Σ = n_s > 1 ? M2 ./ (n_s - 1) : zero(M2)
    return μ, Σ
end

@inline _pm_scalar_or(v, j) = v isa Number ? v : v[j]

# additive-Gaussian negative log-likelihood of an observed moment vector vs prediction,
# with per-entry or scalar noise standard deviation σ.
function _pm_gauss_nll(obs, pred, σ)
    acc = zero(eltype(pred))
    @inbounds for j in eachindex(obs)
        s = _pm_scalar_or(σ, j)
        r = (obs[j] - pred[j]) / s
        acc += log(2π * s^2) + r^2
    end
    return 0.5 * acc
end

"""
    population_moment_term(; simulate, re_half, samples,
                           mean = nothing, sd_mean = nothing,
                           var = nothing, sd_var = nothing,
                           meas_var = 0.0, postprocess = nothing) -> (θu -> Real)

Construct an objective term for population-average (PA) or single-cell-snapshot (SCSH)
data, to be passed to `fit_model(dm, method; extra_objective = term)` (optionally summed
with other terms). The returned closure maps the natural-scale parameters `θu` to a
negative log-likelihood.

Matches the population *mean* of the observable when `mean` is supplied (PA); matches the
population *variance* when `var` is supplied (SCSH), with the single-cell measurement
variance `meas_var` added to the model variance (the snapshot population variance is
biological variability plus measurement noise). Supply `mean`, `var`, or both — at least
one is required. A **variance-only** term (`mean = nothing`, `var`/`sd_var` given) is the
apoptosis SCSH case, whose data stores variances only (population means are unavailable),
and directly informs the random-effect SDs in `re_half`.

# Keyword arguments
- `simulate`, `re_half`, `samples`: passed to [`ensemble_moments`](@ref) (`samples` is
  the fixed draw matrix from [`fixed_re_normals`](@ref)).
- `mean`: observed population-mean time series (omit for a variance-only SCSH term);
  `sd_mean`: its noise SD (scalar or per-time vector; typically from experimental
  replicates). Required iff `mean` is supplied.
- `var`, `sd_var`: observed population-variance time series and its noise SD (SCSH).
- `meas_var`: single-cell measurement-noise variance added to the model variance (SCSH).
- `postprocess`: optional `μ -> μ'` transform applied to the population MEAN before it is
  compared to `mean` — e.g. the apoptosis population-average fraction normalization
  (species means divided by their totals, paper eqs 68–73). Applied to the mean only;
  the variance term (SCSH) is scored on the raw moments.

The term is differentiable through `θu` (ForwardDiff), matching how `extra_objective` is
differentiated by the gradient-based estimators.
"""
function population_moment_term(; simulate, re_half, samples::AbstractMatrix,
        mean = nothing, sd_mean = nothing,
        var = nothing, sd_var = nothing, meas_var = 0.0, postprocess = nothing)
    has_mean = mean !== nothing
    has_var = var !== nothing
    has_mean || has_var ||
        error("population_moment_term: supply `mean` (PA/SCSH) and/or `var` (SCSH).")
    has_mean && sd_mean === nothing &&
        error("population_moment_term: `sd_mean` is required when `mean` is supplied.")
    has_var && sd_var === nothing &&
        error("population_moment_term: `sd_var` is required when `var` is supplied (SCSH).")
    return function pop_moment_nll(θu)
        μ, Σ = ensemble_moments(simulate, θu, re_half, samples)
        nll = zero(eltype(μ))
        if has_mean
            μrep = postprocess === nothing ? μ : postprocess(μ)
            nll += _pm_gauss_nll(mean, μrep, sd_mean)
        end
        if has_var
            Σtot = [Σ[j] + meas_var for j in eachindex(Σ)]
            nll += _pm_gauss_nll(var, Σtot, sd_var)
        end
        return nll
    end
end
