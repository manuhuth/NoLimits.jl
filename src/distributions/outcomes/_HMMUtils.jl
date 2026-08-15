# Accepts tuples as well as vectors: the per-row HMM logpdf paths fuse their
# per-state terms into tuples (no intermediate vectors); index-order max scan
# and exp-sum are identical for both, so values are bit-identical.
@inline function _hmm_logsumexp(xs::Union{AbstractVector, Tuple})
    isempty(xs) && return -Inf
    m = xs[1]
    @inbounds for i in 2:length(xs)
        m = max(m, xs[i])
    end
    isfinite(m) || return m
    s = zero(m)
    # Ignore terms that are far below the max in value space. For Dual numbers,
    # these underflow-scale terms can carry non-finite sensitivities (e.g. -Inf)
    # that are numerically irrelevant to the value but can poison gradients.
    cutoff = -700.0
    @inbounds for x in xs
        δ = x - m
        δ > cutoff || continue
        s += exp(δ)
    end
    return m + log(s)
end

# Static-length view of a state-probability vector, sized by the (type-level)
# number of emission distributions: lets logpdf/posterior fuse their per-state
# work into tuple operations without allocating intermediate vectors.
@inline _hmm_probs_tuple(p::AbstractVector, ::NTuple{N, Any}) where {N} = ntuple(
    i -> @inbounds(p[i]), Val(N)
)

# Combined per-row HMM accessor: returns (logpdf(d, y), posterior_hidden_states(d, y)).
# The forward-filter loop in `_loglikelihood_individual` (and cv) needs BOTH every
# observed row. The continuous-time families recompute the state-probability
# propagation `exp(QΔt)` (the dominant per-row cost) once in `logpdf` and again in
# `posterior_hidden_states`; they specialise this accessor to propagate ONCE and
# reuse it for both, using the EXACT per-state ops of the two methods (bit-identical
# values). The generic fallback below just calls both — correct for every family;
# the discrete-time families inline `transpose(M)*p` and so share nothing, gaining
# nothing from a specialisation but losing nothing from the fallback. Specialisations
# live in the CT family files (ContinuousTimeHMM / MVContinuousTimeHMM /
# ContinuousTimeObservedStatesMarkovModel / CoarsedObservedStatesMarkoModel).
@inline _hmm_logpdf_and_posterior(d, y) = (logpdf(d, y), posterior_hidden_states(d, y))

# Constructor-time validation of HMM transition/generator matrices. A malformed matrix
# used to be accepted and then produce silent NaNs inside `probabilities_hidden_states`
# (#207). The tolerance is loose enough for stick-breaking round-off during fitting.
const _HMM_ROW_ATOL = 1.0e-6

function _hmm_check_transition_matrix(M::AbstractMatrix, label = "transition_matrix")
    n = size(M, 1)
    @inbounds for i in 1:n
        s = zero(eltype(M))
        for j in 1:n
            m = M[i, j]
            isfinite(m) ||
                error("$(label)[$(i), $(j)] is $(m); transition probabilities must be finite.")
            m >= 0 ||
                error("$(label)[$(i), $(j)] is $(m); transition probabilities must be nonnegative.")
            s += m
        end
        isapprox(s, one(s); atol = _HMM_ROW_ATOL) ||
            error("$(label) row $(i) sums to $(s); each row must sum to 1.")
    end
    return nothing
end

function _hmm_check_generator_matrix(Q::AbstractMatrix, label = "transition_matrix")
    n = size(Q, 1)
    @inbounds for i in 1:n
        s = zero(eltype(Q))
        for j in 1:n
            q = Q[i, j]
            isfinite(q) ||
                error("$(label)[$(i), $(j)] is $(q); generator entries must be finite.")
            i == j || q >= 0 ||
                error("$(label)[$(i), $(j)] is $(q); off-diagonal generator rates must be nonnegative.")
            s += q
        end
        isapprox(s, zero(s); atol = _HMM_ROW_ATOL) ||
            error("$(label) row $(i) sums to $(s); each row of a continuous-time generator must sum to 0.")
    end
    return nothing
end

# An observation impossible under every state leaves the posterior undefined (0/0). The
# filter falls back to the propagated prior; the likelihood contribution is -Inf either
# way, and a NaN posterior would poison every later row (#207).
@inline function _hmm_normalize_posterior(u, prior)
    s = sum(u)
    (isfinite(s) && s > 0) || return [convert(eltype(u), p) for p in prior]
    return [ui / s for ui in u]
end

# Quantile of the state-mixture: inf{y : cdf(y) >= p}. Averaging the per-state component
# quantiles (the old discrete-time implementation) is not the quantile of a mixture and
# could return values outside the support (#213).
function _hmm_mixture_quantile(hmm, p::Real)
    (p isa Real && 0 <= p <= 1) ||
        throw(DomainError(p, "quantile probability must be in [0, 1]."))
    dists = hmm.emission_dists
    p == 0 && return minimum(map(d -> quantile(d, 0.0), dists))
    p == 1 && return maximum(map(d -> quantile(d, 1.0), dists))
    lb = minimum(map(d -> quantile(d, 1.0e-9), dists))
    ub = maximum(map(d -> quantile(d, 1 - 1.0e-9), dists))
    if all(d -> d isa Distributions.DiscreteUnivariateDistribution, dists)
        lo, hi = floor(Int, lb), ceil(Int, ub)
        while lo < hi
            mid = fld(lo + hi, 2)
            cdf(hmm, mid) >= p ? (hi = mid) : (lo = mid + 1)
        end
        return lo
    end
    for _ in 1:200
        mid = (lb + ub) / 2
        cdf(hmm, mid) < p ? (lb = mid) : (ub = mid)
        abs(ub - lb) < 1.0e-10 && break
    end
    return (lb + ub) / 2
end
