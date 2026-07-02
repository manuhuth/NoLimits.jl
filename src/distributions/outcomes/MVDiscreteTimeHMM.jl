export MVDiscreteTimeDiscreteStatesHMM

using Distributions, LinearAlgebra, Random, Lux

"""
    MVDiscreteTimeDiscreteStatesHMM(transition_matrix, emission_dists, initial_dist)
    <: Distribution{Multivariate, Continuous}

A discrete-time Hidden Markov Model with shared latent states across M outcome
variables.

Two emission modes are supported:

- **Conditionally independent**: `emission_dists` is a `Tuple` of `n_states`
  inner `Tuple`s, where `emission_dists[k]` is a `Tuple` of M scalar
  distributions (one per outcome) for state `k`. Given the hidden state, all
  outcomes are treated as independent.

- **Joint**: `emission_dists` is a `Tuple` of `n_states` multivariate
  distributions (e.g. `MvNormal`), where `emission_dists[k]` is the joint
  emission distribution for state `k`.

Missing values in the observation vector are handled as follows:
- Independent mode: missing outcomes are skipped (contribute zero to the
  log-likelihood).
- Joint MvNormal mode: the marginal distribution over observed indices is
  computed analytically.
- Other joint distributions: an error is raised if any observation is missing.

# Arguments
- `transition_matrix`: row-stochastic matrix of shape `(n_states, n_states)`.
  Entry `[i, j]` is `P(State_t = j | State_{t-1} = i)`.
- `emission_dists`: `Tuple` of `n_states` emission elements (see above).
- `initial_dist`: `Distributions.Categorical` prior over hidden states at the
  previous time step. Propagated one step via `transition_matrix` before
  computing the emission likelihood.
"""
struct MVDiscreteTimeDiscreteStatesHMM{
    M <: AbstractMatrix{<:Real},
    E <: Tuple,
    D <: Distributions.Categorical
} <: Distribution{Multivariate, Continuous}
    n_states::Int
    n_outcomes::Int
    transition_matrix::M
    emission_dists::E
    initial_dist::D
end

function MVDiscreteTimeDiscreteStatesHMM(
        transition_matrix::AbstractMatrix{<:Real},
        emission_dists::Tuple,
        initial_dist::Distributions.Categorical
)
    n_states = size(transition_matrix, 1)
    size(transition_matrix, 2) == n_states ||
        error("transition_matrix must be square, got $(size(transition_matrix)).")
    length(emission_dists) == n_states ||
        error("length(emission_dists) must equal n_states ($n_states), " *
              "got $(length(emission_dists)).")
    length(initial_dist.p) == n_states ||
        error("length(initial_dist.p) must equal n_states ($n_states), " *
              "got $(length(initial_dist.p)).")
    n_outcomes = _mv_n_outcomes(emission_dists[1])
    for k in 2:n_states
        _mv_n_outcomes(emission_dists[k]) == n_outcomes ||
            error("All emission elements must have the same number of outcomes. " *
                  "Element 1 has $n_outcomes but element $k has " *
                  "$(_mv_n_outcomes(emission_dists[k])).")
    end
    return MVDiscreteTimeDiscreteStatesHMM(
        n_states, n_outcomes, transition_matrix, emission_dists, initial_dist)
end

# ---------------------------------------------------------------------------
# Hidden-state probabilities
# ---------------------------------------------------------------------------

"""
    probabilities_hidden_states(hmm::MVDiscreteTimeDiscreteStatesHMM) -> Vector

Marginal prior probabilities of the hidden states at the current observation
time, propagated one step from `hmm.initial_dist` via `hmm.transition_matrix`.
"""
function probabilities_hidden_states(hmm::MVDiscreteTimeDiscreteStatesHMM)
    p = transpose(hmm.transition_matrix) * hmm.initial_dist.p
    return p ./ sum(p)
end

"""
    posterior_hidden_states(hmm::MVDiscreteTimeDiscreteStatesHMM, y::AbstractVector)

Posterior probabilities of hidden states given the length-M observation vector
`y` (which may contain `missing` entries). Uses all non-missing outcomes jointly.
"""
function posterior_hidden_states(hmm::MVDiscreteTimeDiscreteStatesHMM, y::AbstractVector)
    # Tuple-fused twin of the former vector pipeline (same op order — values are
    # bit-identical). `map` over the emission tuple also replaces the runtime
    # `emission_dists[k]` indexing, which dispatched dynamically whenever the
    # per-state emissions have heterogeneous types.
    p = transpose(hmm.transition_matrix) * hmm.initial_dist.p
    s = sum(p)
    dists = hmm.emission_dists
    pt = _hmm_probs_tuple(p, dists)
    u = map((pi, d) -> (pi / s) * exp(_mv_emission_logpdf(d, y)), pt, dists)
    su = sum(u)
    return [ui / su for ui in u]
end

# ---------------------------------------------------------------------------
# Distributions.jl interface
# ---------------------------------------------------------------------------

function Distributions.logpdf(hmm::MVDiscreteTimeDiscreteStatesHMM, y::AbstractVector)
    # Tuple-fused per-state terms (see posterior_hidden_states) — bit-identical.
    p = transpose(hmm.transition_matrix) * hmm.initial_dist.p
    s = sum(p)
    dists = hmm.emission_dists
    pt = _hmm_probs_tuple(p, dists)
    xs = map((pi, d) -> log(pi / s) + _mv_emission_logpdf(d, y), pt, dists)
    return _hmm_logsumexp(xs)
end

function Distributions.pdf(hmm::MVDiscreteTimeDiscreteStatesHMM, y::AbstractVector)
    exp(logpdf(hmm, y))
end

# Combined accessor sharing the one-step propagation AND the per-state emission
# log-densities between the likelihood and posterior; reuses the EXACT per-state
# ops of `logpdf`/`posterior_hidden_states` above (bit-identical).
function _hmm_logpdf_and_posterior(hmm::MVDiscreteTimeDiscreteStatesHMM,
        y::AbstractVector)
    p = transpose(hmm.transition_matrix) * hmm.initial_dist.p
    s = sum(p)
    dists = hmm.emission_dists
    pt = _hmm_probs_tuple(p, dists)
    ls = map(d -> _mv_emission_logpdf(d, y), dists)
    lp = _hmm_logsumexp(map((pi, l) -> log(pi / s) + l, pt, ls))
    u = map((pi, l) -> (pi / s) * exp(l), pt, ls)
    su = sum(u)
    return lp, [ui / su for ui in u]
end

function Distributions.rand(rng::AbstractRNG, hmm::MVDiscreteTimeDiscreteStatesHMM)
    state = rand(rng, Categorical(probabilities_hidden_states(hmm)))
    return _mv_emission_rand(rng, hmm.emission_dists[state])
end

function Distributions.mean(hmm::MVDiscreteTimeDiscreteStatesHMM)
    p = probabilities_hidden_states(hmm)
    return sum(p[k] * _mv_emission_mean(hmm.emission_dists[k]) for k in 1:(hmm.n_states))
end

"""
    cov(hmm::MVDiscreteTimeDiscreteStatesHMM) -> Matrix

Full variance-covariance matrix of the mixture, computed via the law of total
covariance:

    Cov[Y] = E[Cov[Y|S]] + Cov[E[Y|S]]
"""
function Distributions.cov(hmm::MVDiscreteTimeDiscreteStatesHMM)
    p = probabilities_hidden_states(hmm)
    μ_k = [_mv_emission_mean(hmm.emission_dists[k]) for k in 1:(hmm.n_states)]
    μ = sum(p[k] * μ_k[k] for k in 1:(hmm.n_states))
    within = sum(p[k] * Matrix(_mv_emission_cov(hmm.emission_dists[k]))
    for k in 1:(hmm.n_states))
    between = sum(p[k] * (μ_k[k] - μ) * (μ_k[k] - μ)' for k in 1:(hmm.n_states))
    return within + between
end

Distributions.var(hmm::MVDiscreteTimeDiscreteStatesHMM) = diag(cov(hmm))

Base.length(hmm::MVDiscreteTimeDiscreteStatesHMM) = hmm.n_outcomes

function Distributions.params(hmm::MVDiscreteTimeDiscreteStatesHMM)
    (hmm.transition_matrix, hmm.emission_dists, hmm.initial_dist)
end
