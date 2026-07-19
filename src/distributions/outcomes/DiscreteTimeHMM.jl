export DiscreteTimeDiscreteStatesHMM
export probabilities_hidden_states
export posterior_hidden_states

using Distributions

"""
    DiscreteTimeDiscreteStatesHMM(transition_matrix, emission_dists, initial_dist)
    <: Distribution{Univariate, Continuous}

A discrete-time Hidden Markov Model (HMM) with a finite number of hidden states and
continuous or discrete emission distributions.

Implements the `Distributions.jl` interface (`pdf`, `logpdf`, `rand`, `mean`, `var`).
Used as an observation distribution in `@formulas` blocks to model outcomes with latent
state dynamics.

# Arguments
- `transition_matrix::AbstractMatrix{<:Real}`: row-stochastic transition matrix of shape
  `(n_states, n_states)`. Entry `[i, j]` is `P(State_t = j | State_{t-1} = i)`.
- `emission_dists::Tuple`: tuple of `n_states` emission distributions, one per state.
- `initial_dist::Distributions.Categorical`: prior over hidden states at the current time
  step. Propagated one step via `transition_matrix` before computing the emission
  likelihood.
"""
struct DiscreteTimeDiscreteStatesHMM{
    M <: AbstractMatrix{<:Real}, E <: Tuple, D <: Distributions.Categorical} <:
       Distribution{Univariate, Continuous}
    n_states::Int
    transition_matrix::M
    emission_dists::E
    initial_dist::D
end

function DiscreteTimeDiscreteStatesHMM(transition_matrix::AbstractMatrix{<:Real},
        emission_dists::Tuple,
        initial_dist::Distributions.Categorical)
    n_states = size(transition_matrix, 1)
    size(transition_matrix, 2) == n_states || error("transition_matrix must be square.")
    length(emission_dists) == n_states ||
        error("Number of emission distributions must match number of states.")
    length(initial_dist.p) == n_states ||
        error("Initial distribution size must match number of states.")
    return DiscreteTimeDiscreteStatesHMM(
        n_states, transition_matrix, emission_dists, initial_dist)
end

"""
    probabilities_hidden_states(hmm::DiscreteTimeDiscreteStatesHMM) -> Vector{Float64}
    probabilities_hidden_states(hmm::ContinuousTimeDiscreteStatesHMM) -> Vector{Float64}

Compute the marginal prior probabilities of the hidden states at the current observation
time, propagated from `hmm.initial_dist` through the transition dynamics.

Returns a normalized probability vector of length `n_states`.
"""
function probabilities_hidden_states(hmm::DiscreteTimeDiscreteStatesHMM)
    # Propagate one discrete step from the provided prior state probabilities.
    p = transpose(hmm.transition_matrix) * hmm.initial_dist.p
    return p ./ sum(p)
end

"""
    posterior_hidden_states(hmm::DiscreteTimeDiscreteStatesHMM, y::Real)

Compute posterior probabilities of hidden states given observation `y`.
"""
function posterior_hidden_states(hmm::DiscreteTimeDiscreteStatesHMM, y::Real)
    # Same op order as the former vector pipeline (propagate, normalize by the
    # same sum, weight by the emission pdf, normalize) with the per-state steps
    # fused into tuples — bit-identical values, one matvec + the output vector
    # instead of four intermediate vectors per call (this runs once per row in
    # the HMM forward filter).
    p = transpose(hmm.transition_matrix) * hmm.initial_dist.p
    s = sum(p)
    dists = hmm.emission_dists
    pt = _hmm_probs_tuple(p, dists)
    u = map((pi, d) -> (pi / s) * pdf(d, y), pt, dists)
    su = sum(u)
    return [ui / su for ui in u]
end

function Distributions.pdf(hmm::DiscreteTimeDiscreteStatesHMM, y::Real)
    p_hidden = probabilities_hidden_states(hmm)
    p_obs = pdf.(hmm.emission_dists, Ref(y))
    return sum(p_hidden .* p_obs)
end

function Distributions.logpdf(hmm::DiscreteTimeDiscreteStatesHMM, y::Real)
    # Tuple-fused twin of the former vector pipeline (see posterior_hidden_states):
    # the propagation matvec is the only remaining allocation per call.
    p = transpose(hmm.transition_matrix) * hmm.initial_dist.p
    s = sum(p)
    dists = hmm.emission_dists
    pt = _hmm_probs_tuple(p, dists)
    xs = map((pi, d) -> log(pi / s) + logpdf(d, y), pt, dists)
    return _hmm_logsumexp(xs)
end

# Combined accessor sharing the one-step propagation (p, s, pt) between the
# likelihood and the posterior; the per-state pdf/logpdf calls stay separate, so
# this is bit-identical to calling `logpdf`/`posterior_hidden_states` above. The
# generic fallback in `_HMMUtils.jl` recomputes the propagation matvec twice.
function _hmm_logpdf_and_posterior(hmm::DiscreteTimeDiscreteStatesHMM, y::Real)
    p = transpose(hmm.transition_matrix) * hmm.initial_dist.p
    s = sum(p)
    dists = hmm.emission_dists
    pt = _hmm_probs_tuple(p, dists)
    lp = _hmm_logsumexp(map((pi, d) -> log(pi / s) + logpdf(d, y), pt, dists))
    u = map((pi, d) -> (pi / s) * pdf(d, y), pt, dists)
    su = sum(u)
    return lp, [ui / su for ui in u]
end

function Distributions.rand(rng::AbstractRNG, hmm::DiscreteTimeDiscreteStatesHMM)
    state = rand(rng, Categorical(probabilities_hidden_states(hmm)))
    return rand(rng, hmm.emission_dists[state])
end

function Distributions.mean(hmm::DiscreteTimeDiscreteStatesHMM)
    p_hidden = probabilities_hidden_states(hmm)
    μ = mean.(hmm.emission_dists)
    return sum(p_hidden .* μ)
end

function Distributions.var(hmm::DiscreteTimeDiscreteStatesHMM)
    p_hidden = probabilities_hidden_states(hmm)
    μ = mean.(hmm.emission_dists)
    v = var.(hmm.emission_dists)
    μ_mix = sum(p_hidden .* μ)
    return sum(p_hidden .* (v .+ (μ .- μ_mix) .^ 2))
end

function Distributions.cdf(hmm::DiscreteTimeDiscreteStatesHMM, y::Real)
    p_hidden = probabilities_hidden_states(hmm)
    c = cdf.(hmm.emission_dists, Ref(y))
    return sum(p_hidden .* c)
end

function Distributions.quantile(hmm::DiscreteTimeDiscreteStatesHMM, p::Real)
    p_hidden = probabilities_hidden_states(hmm)
    q = quantile.(hmm.emission_dists, Ref(p))
    return sum(p_hidden .* q)
end

Distributions.median(hmm::DiscreteTimeDiscreteStatesHMM) = quantile(hmm, 0.5)
function Distributions.params(hmm::DiscreteTimeDiscreteStatesHMM)
    (hmm.transition_matrix, hmm.emission_dists, hmm.initial_dist)
end
Base.length(hmm::DiscreteTimeDiscreteStatesHMM) = 1
