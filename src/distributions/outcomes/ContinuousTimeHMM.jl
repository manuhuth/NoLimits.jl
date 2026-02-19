export ContinuousTimeDiscreteStatesHMM

using Distributions, ExponentialAction, Random, Lux
import Distributions: pdf, logpdf, rand, mean, var, median, quantile, mode, cdf, support, params

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
struct ContinuousTimeDiscreteStatesHMM{M<:AbstractMatrix{<:Real}, E<:Tuple, D<:Distributions.Categorical, T<:Real} <: Distribution{Univariate, Continuous}
    n_states::Int
    transition_matrix::M
    emission_dists::E
    initial_dist::D
    Δt::T
end

function ContinuousTimeDiscreteStatesHMM(transition_matrix::AbstractMatrix{<:Real}, emission_dists::Tuple, initial_dist::Distributions.Categorical, Δt::Real)
    n_states = size(transition_matrix, 1)
    ContinuousTimeDiscreteStatesHMM(n_states, transition_matrix, emission_dists, initial_dist, Δt)
end

# Transition Matrix must be transposed here as expv uses column sums that equal zero
probabilities_hidden_states(hmm::ContinuousTimeDiscreteStatesHMM) = expv(hmm.Δt, transpose(hmm.transition_matrix), hmm.initial_dist.p)

"""
    posterior_hidden_states(hmm::ContinuousTimeDiscreteStatesHMM, y::Real)

Compute the posterior probability distribution of hidden states given observation `y`.

Returns a vector of probabilities `p` where `p[s]` is `P(State = s | Y = y)`.

Uses Bayes' rule: `P(S | Y) ∝ P(Y | S) * P(S)`
"""
function posterior_hidden_states(hmm::ContinuousTimeDiscreteStatesHMM, y::Real)
    p_hidden = probabilities_hidden_states(hmm)
    p_obs_given_state = pdf.(hmm.emission_dists, Ref(y))
    unnormalized = p_hidden .* p_obs_given_state
    return unnormalized ./ sum(unnormalized)
end

function Distributions.pdf(hmm::ContinuousTimeDiscreteStatesHMM, y::Real)
    p_hidden = probabilities_hidden_states(hmm)
    p_obs = pdf.(hmm.emission_dists, Ref(y))
    return sum(p_hidden .* p_obs)
end

function Distributions.logpdf(hmm::ContinuousTimeDiscreteStatesHMM, y::Real)
    log_p_hidden = log.(probabilities_hidden_states(hmm))
    log_p_obs = logpdf.(hmm.emission_dists, Ref(y))
    return Lux.logsumexp(log_p_hidden .+ log_p_obs)
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
    return sum(p_hidden .* emission_vars) + sum(p_hidden .* (emission_means .- μ).^2)
end

function Distributions.cdf(hmm::ContinuousTimeDiscreteStatesHMM, y::Real)
    p_hidden = probabilities_hidden_states(hmm)
    return sum(p_hidden .* cdf.(hmm.emission_dists, Ref(y)))
end

function Distributions.quantile(hmm::ContinuousTimeDiscreteStatesHMM, p::Real)
    @assert 0 < p < 1 "p must be in (0, 1)"

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

Distributions.params(hmm::ContinuousTimeDiscreteStatesHMM) = (hmm.transition_matrix, hmm.emission_dists, hmm.initial_dist, hmm.Δt)

Base.length(hmm::ContinuousTimeDiscreteStatesHMM) = 1

