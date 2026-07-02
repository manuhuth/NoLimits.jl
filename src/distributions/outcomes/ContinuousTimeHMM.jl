export ContinuousTimeDiscreteStatesHMM

using Distributions, ExponentialAction, Random, Lux
import Distributions: pdf, logpdf, rand, mean, var, median, quantile, mode, cdf, support,
                      params
import ForwardDiff

function _ct_hmm_validate_mode(mode::Symbol)
    mode in (:auto, :expv, :pathsum) ||
        error("Invalid CT-HMM propagation mode $(mode). Use one of :auto, :expv, :pathsum.")
end

function _ct_hmm_adjacency(transition_matrix::AbstractMatrix{<:Real}; atol::Real = 1e-12)
    n_states = size(transition_matrix, 1)
    adj = [Int[] for _ in 1:n_states]
    for i in 1:n_states
        for j in 1:n_states
            i == j && continue
            transition_matrix[i, j] > atol && push!(adj[i], j)
        end
    end
    return adj
end

function _ct_hmm_topological_order(adj::Vector{Vector{Int}})
    n = length(adj)
    indeg = zeros(Int, n)
    for i in 1:n
        for j in adj[i]
            indeg[j] += 1
        end
    end
    order = Vector{Int}(undef, n)
    queue = Int[]
    for i in 1:n
        indeg[i] == 0 && push!(queue, i)
    end
    head = 1
    k = 1
    while head <= length(queue)
        v = queue[head]
        head += 1
        order[k] = v
        k += 1
        for u in adj[v]
            indeg[u] -= 1
            indeg[u] == 0 && push!(queue, u)
        end
    end
    return (k == n + 1) ? order : nothing
end

function _ct_hmm_probabilities_pathsum_dag(
        transition_matrix::AbstractMatrix{<:Real},
        initial_p::AbstractVector{<:Real},
        Δt::Real;
        atol::Real = 1e-10
)
    n_states = size(transition_matrix, 1)
    adj = _ct_hmm_adjacency(transition_matrix; atol = atol)
    order = _ct_hmm_topological_order(adj)
    order === nothing && return nothing

    # parents[j] = all i with edge i -> j
    parents = [Int[] for _ in 1:n_states]
    for i in 1:n_states
        for j in adj[i]
            push!(parents[j], i)
        end
    end

    λ = [-transition_matrix[i, i] for i in 1:n_states]
    Tprob = promote_type(eltype(transition_matrix), eltype(initial_p), typeof(Δt))
    C = zeros(Tprob, n_states, n_states) # row j: coefficients for exp(-λ[r] t)
    eps_tol = Tprob(atol)

    # Solve lower-triangular-in-topological-order coefficient system:
    # p_j(t) = sum_r C[j,r] exp(-λ_r t)
    for j in order
        row_sum = zero(Tprob)
        for r in 1:n_states
            if r == j
                continue
            end
            rhs = zero(Tprob)
            for i in parents[j]
                rhs += transition_matrix[i, j] * C[i, r]
            end
            d = λ[j] - λ[r]
            if abs(d) <= eps_tol
                abs(rhs) <= eps_tol || return nothing
                C[j, r] = zero(Tprob)
            else
                C[j, r] = rhs / d
            end
            row_sum += C[j, r]
        end
        C[j, j] = initial_p[j] - row_sum
    end

    p = zeros(Tprob, n_states)
    for j in 1:n_states
        s = zero(Tprob)
        for r in 1:n_states
            s += C[j, r] * exp(-λ[r] * Δt)
        end
        p[j] = s
    end
    return p ./ sum(p)
end

# When spectral_gap(Q) * Δt exceeds this threshold, exp(Q Δt) v ≈ stationary distribution
# to within ~1e-12 for all ergodic Q.  The spectral gap (smallest |non-zero eigenvalue| of Q)
# governs convergence; using max_exit_rate was incorrect for Q matrices with disparate rates
# (e.g. funnel topologies) where the gap can be orders of magnitude smaller than max_exit.
# For ForwardDiff Dual-number inputs eigvals is unavailable; the shortcut is skipped and
# expv is used instead.
const _CT_HMM_EQUIL_THRESHOLD = 200.0

# Strip ForwardDiff Dual to its primal Float64 value (identity for plain floats).
@inline _ct_hmm_primal(x::AbstractFloat) = x
@inline _ct_hmm_primal(x::ForwardDiff.Dual) = ForwardDiff.value(x)

# Returns the spectral gap of Q: smallest |non-zero eigenvalue|.
# Operates on the primal Float64 values so it works for both plain floats and Dual arrays.
# The gap is a structural property of Q; the Dual (gradient) parts are irrelevant for
# determining whether the system has reached equilibrium.
function _ct_hmm_spectral_gap(Q::AbstractMatrix)
    Q_float = map(_ct_hmm_primal, Q)
    λs = eigvals(Q_float)
    return minimum(abs.(λs[abs.(λs) .> sqrt(eps(Float64))]))
end

# Stationary distribution of Q: null eigenvector of Qᵀ with ∑πᵢ = 1.
# Solved via the linear system [Qᵀ (last row replaced by 1…1)] π = e_n.
# Returns nothing for non-ergodic Q (multiple communicating classes or absorbing states),
# where the null space has dimension > 1 and the long-time limit of exp(Qt)v depends on v.
# Non-ergodicity is detected by any π component being negative after solving.
function _ct_hmm_stationary(Q::AbstractMatrix{T}) where {T}
    n = size(Q, 1)
    A = Matrix{T}(transpose(Q))
    for j in 1:n
        A[n, j] = one(T)
    end
    b = zeros(T, n)
    b[n] = one(T)
    # For non-ergodic Q (multiple communicating classes or absorbing states) the
    # system is rank-deficient.  `\` throws SingularException in the exactly
    # singular case; near-singular cases produce a solution with negative
    # components.  Both signal that the long-time limit depends on v, so we
    # return nothing and let the caller fall back to expv.
    local π
    try
        π = A \ b
    catch e
        e isa LinearAlgebra.SingularException && return nothing
        rethrow(e)
    end
    tol = sqrt(eps(float(real(one(T)))))
    any(x -> x < -tol, π) && return nothing
    s = sum(π)
    return π ./ s  # renormalize for floating-point safety
end

function _ct_hmm_probabilities_hidden_states(
        transition_matrix::AbstractMatrix{<:Real},
        initial_p::AbstractVector{<:Real},
        Δt::Real;
        mode::Symbol = :auto,
        atol::Real = 1e-10
)
    _ct_hmm_validate_mode(mode)

    if mode == :expv
        # Equilibrium shortcut: when spectral_gap(Q) × Δt >> 1, exp(Q Δt) v → stationary.
        # Gate on max_exit first (free) to avoid the eigvals call in the common case.
        # For ForwardDiff Dual inputs, _ct_hmm_spectral_gap strips to primal Float64 values;
        # _ct_hmm_stationary uses A\b which is generic and works for Dual.
        max_exit = maximum(-_ct_hmm_primal.(diag(transition_matrix)))
        if max_exit * _ct_hmm_primal(Δt) > _CT_HMM_EQUIL_THRESHOLD
            gap = _ct_hmm_spectral_gap(transition_matrix)
            if gap * _ct_hmm_primal(Δt) > _CT_HMM_EQUIL_THRESHOLD
                π = _ct_hmm_stationary(transition_matrix)
                π !== nothing && return π
                # Non-ergodic Q: fall through to expv
            end
        end
        return expv(Δt, transpose(transition_matrix), initial_p)
    end

    if mode == :pathsum || mode == :auto
        p = _ct_hmm_probabilities_pathsum_dag(transition_matrix, initial_p, Δt; atol = atol)
        if p !== nothing
            return p
        end
        if mode == :pathsum
            error("propagation_mode=:pathsum requested but path-sum is not applicable " *
                  "(cyclic graph or numerically degenerate exit rates).")
        end
    end

    # :auto fell through (cyclic graph) — apply equilibrium shortcut before expv
    max_exit = maximum(-_ct_hmm_primal.(diag(transition_matrix)))
    if max_exit * _ct_hmm_primal(Δt) > _CT_HMM_EQUIL_THRESHOLD
        gap = _ct_hmm_spectral_gap(transition_matrix)
        if gap * _ct_hmm_primal(Δt) > _CT_HMM_EQUIL_THRESHOLD
            π = _ct_hmm_stationary(transition_matrix)
            π !== nothing && return π
            # Non-ergodic Q: fall through to expv
        end
    end

    return expv(Δt, transpose(transition_matrix), initial_p)
end

"""
    ContinuousTimeDiscreteStatesHMM(transition_matrix, emission_dists, initial_dist, Δt)
    <: Distribution{Univariate, Continuous}

A continuous-time Hidden Markov Model (HMM) with a finite number of hidden states and
continuous or discrete emission distributions.

State propagation is performed via the matrix exponential `exp(Q·Δt)` where `Q` is the
rate matrix (`transition_matrix`). Implements the `Distributions.jl` interface.

# Arguments
- `transition_matrix::AbstractMatrix{<:Real}`: rate matrix (generator) of shape
  `(n_states, n_states)`. Off-diagonal entries must be non-negative; each row must sum
  to zero.
- `emission_dists::Tuple`: tuple of `n_states` emission distributions.
- `initial_dist::Distributions.Categorical`: prior over hidden states at the previous
  observation time.
- `Δt::Real`: time elapsed since the previous observation.
"""
struct ContinuousTimeDiscreteStatesHMM{
    M <: AbstractMatrix{<:Real},
    E <: Tuple,
    D <: Distributions.Categorical,
    T <: Real
} <: Distribution{Univariate, Continuous}
    n_states::Int
    transition_matrix::M
    emission_dists::E
    initial_dist::D
    Δt::T
    propagation_mode::Symbol
end

function ContinuousTimeDiscreteStatesHMM(
        transition_matrix::AbstractMatrix{<:Real},
        emission_dists::Tuple,
        initial_dist::Distributions.Categorical,
        Δt::Real;
        propagation_mode::Symbol = :auto
)
    _ct_hmm_validate_mode(propagation_mode)
    n_states = size(transition_matrix, 1)
    ContinuousTimeDiscreteStatesHMM(
        n_states, transition_matrix, emission_dists, initial_dist, Δt, propagation_mode)
end

# _ct_hmm_probabilities_hidden_states transposes the rate matrix internally, because
# expv expects a generator whose columns (not rows) sum to zero.
function probabilities_hidden_states(hmm::ContinuousTimeDiscreteStatesHMM)
    _ct_hmm_probabilities_hidden_states(
        hmm.transition_matrix, hmm.initial_dist.p, hmm.Δt; mode = hmm.propagation_mode)
end

"""
    posterior_hidden_states(hmm::ContinuousTimeDiscreteStatesHMM, y::Real)

Compute the posterior probability distribution of hidden states given observation `y`.

Returns a vector of probabilities `p` where `p[s]` is `P(State = s | Y = y)`.

Uses Bayes' rule: `P(S | Y) ∝ P(Y | S) * P(S)`
"""
function posterior_hidden_states(hmm::ContinuousTimeDiscreteStatesHMM, y::Real)
    # Per-state weighting/normalization fused into tuples (bit-identical op
    # order); the matrix-exponential propagation inside
    # `probabilities_hidden_states` remains the dominant cost.
    p_hidden = probabilities_hidden_states(hmm)
    dists = hmm.emission_dists
    pt = _hmm_probs_tuple(p_hidden, dists)
    u = map((pi, d) -> pi * pdf(d, y), pt, dists)
    su = sum(u)
    return [ui / su for ui in u]
end

function Distributions.pdf(hmm::ContinuousTimeDiscreteStatesHMM, y::Real)
    p_hidden = probabilities_hidden_states(hmm)
    p_obs = pdf.(hmm.emission_dists, Ref(y))
    return sum(p_hidden .* p_obs)
end

function Distributions.logpdf(hmm::ContinuousTimeDiscreteStatesHMM, y::Real)
    # Tuple-fused per-state terms (see posterior_hidden_states) — bit-identical.
    p_hidden = probabilities_hidden_states(hmm)
    dists = hmm.emission_dists
    pt = _hmm_probs_tuple(p_hidden, dists)
    xs = map((pi, d) -> log(pi) + logpdf(d, y), pt, dists)
    return _hmm_logsumexp(xs)
end

# Combined accessor sharing the single `exp(QΔt)` propagation (the dominant per-row
# cost) between the likelihood and the posterior. Reuses the EXACT per-state ops of
# `logpdf` and `posterior_hidden_states` above, so the returned pair is bit-identical
# to calling them separately.
function _hmm_logpdf_and_posterior(hmm::ContinuousTimeDiscreteStatesHMM, y::Real)
    p_hidden = probabilities_hidden_states(hmm)
    dists = hmm.emission_dists
    pt = _hmm_probs_tuple(p_hidden, dists)
    lp = _hmm_logsumexp(map((pi, d) -> log(pi) + logpdf(d, y), pt, dists))
    u = map((pi, d) -> pi * pdf(d, y), pt, dists)
    su = sum(u)
    return lp, [ui / su for ui in u]
end

function Distributions.rand(rng::AbstractRNG, hmm::ContinuousTimeDiscreteStatesHMM)
    p_hidden = probabilities_hidden_states(hmm)
    state = rand(rng, Categorical(p_hidden))
    return rand(rng, hmm.emission_dists[state])
end

function Distributions.mean(hmm::ContinuousTimeDiscreteStatesHMM)
    p_hidden = probabilities_hidden_states(hmm)
    emission_means = mean.(hmm.emission_dists)
    return sum(p_hidden .* emission_means)
end

function Distributions.var(hmm::ContinuousTimeDiscreteStatesHMM)
    p_hidden = probabilities_hidden_states(hmm)
    μ = mean(hmm)
    emission_means = mean.(hmm.emission_dists)
    emission_vars = var.(hmm.emission_dists)
    # Law of total variance: Var[Y] = E[Var[Y|S]] + Var[E[Y|S]]
    return sum(p_hidden .* emission_vars) + sum(p_hidden .* (emission_means .- μ) .^ 2)
end

function Distributions.cdf(hmm::ContinuousTimeDiscreteStatesHMM, y::Real)
    p_hidden = probabilities_hidden_states(hmm)
    return sum(p_hidden .* cdf.(hmm.emission_dists, Ref(y)))
end

function Distributions.quantile(hmm::ContinuousTimeDiscreteStatesHMM, p::Real)
    @assert 0<p<1 "p must be in (0, 1)"

    # Bound the search using component quantiles
    lower_bounds = quantile.(hmm.emission_dists, Ref(0.001))
    upper_bounds = quantile.(hmm.emission_dists, Ref(0.999))
    lb = minimum(lower_bounds)
    ub = maximum(upper_bounds)

    # Bisection to find y such that cdf(hmm, y) = p
    for _ in 1:100
        mid = (lb + ub) / 2
        if cdf(hmm, mid) < p
            lb = mid
        else
            ub = mid
        end
        abs(ub - lb) < 1e-10 && break
    end
    return (lb + ub) / 2
end

Distributions.median(hmm::ContinuousTimeDiscreteStatesHMM) = quantile(hmm, 0.5)

function Distributions.params(hmm::ContinuousTimeDiscreteStatesHMM)
    (hmm.transition_matrix, hmm.emission_dists, hmm.initial_dist, hmm.Δt)
end

Base.length(hmm::ContinuousTimeDiscreteStatesHMM) = 1
