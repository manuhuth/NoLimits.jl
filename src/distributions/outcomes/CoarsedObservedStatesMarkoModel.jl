export CoarsedObservedStatesMarkovModel, coarsed

using Distributions

"""
    CoarsedObservedStatesMarkovModel(base_dist)

Observed-state Markov-model outcome wrapper for coarsed (set-valued) observations.

This distribution is created via [`coarsed`](@ref). It preserves the same hidden-state
probabilities as `base_dist`, but evaluates observation likelihoods by summing over the
compatible observed labels supplied in each set-valued observation.
"""
struct CoarsedObservedStatesMarkovModel{D <: Distribution{Univariate, Discrete}} <:
       Distribution{Univariate, Discrete}
    base_dist::D
end

"""
    coarsed(dist::Distribution{Univariate, Discrete}) -> CoarsedObservedStatesMarkovModel

Wrap an observed-state Markov-model distribution so that observations are interpreted as
coarsed/set-valued state labels.

Accepted base distributions:
- [`DiscreteTimeObservedStatesMarkovModel`](@ref)
- [`ContinuousTimeObservedStatesMarkovModel`](@ref)

For a set-valued observation `y = [s1, s2, ...]`, the likelihood contribution is
`sum(P(state = sk))` over all compatible labels in `y`.

Use this wrapper in `@formulas` when the observation column contains `AbstractVector`
entries, e.g. `[2,3,4]`.

# Example
```julia
dist = ContinuousTimeObservedStatesMarkovModel(Q, π, labels, Δt)
@formulas begin
    y ~ coarsed(dist)
end
```
"""
function coarsed(dist::D) where {D <: Distribution{Univariate, Discrete}}
    _omm_is_observed_markov_dist(dist) || error(
        "coarsed(...) is only defined for observed Markov-model distributions " *
        "(DiscreteTimeObservedStatesMarkovModel or ContinuousTimeObservedStatesMarkovModel)."
    )
    return CoarsedObservedStatesMarkovModel(dist)
end

probabilities_hidden_states(dist::CoarsedObservedStatesMarkovModel) =
    probabilities_hidden_states(dist.base_dist)

function posterior_hidden_states(dist::CoarsedObservedStatesMarkovModel, y::AbstractVector)
    idxs = _omm_coarsed_observation_indices(dist.base_dist.state_labels, y)
    p = probabilities_hidden_states(dist.base_dist)
    T = eltype(p)
    post = zeros(T, dist.base_dist.n_states)
    isempty(idxs) && return post
    mass = zero(T)
    for idx in idxs
        mass += p[idx]
    end
    (isfinite(mass) && mass > zero(T)) || return post
    inv_mass = inv(mass)
    for idx in idxs
        post[idx] = p[idx] * inv_mass
    end
    return post
end

function posterior_hidden_states(dist::CoarsedObservedStatesMarkovModel, y)
    _omm_coarsed_observation_indices(dist.base_dist.state_labels, y)
    return zeros(eltype(probabilities_hidden_states(dist.base_dist)), dist.base_dist.n_states)
end

function Distributions.logpdf(dist::CoarsedObservedStatesMarkovModel, y::AbstractVector)
    idxs = _omm_coarsed_observation_indices(dist.base_dist.state_labels, y)
    isempty(idxs) && return -Inf
    p = probabilities_hidden_states(dist.base_dist)
    mass = zero(eltype(p))
    for idx in idxs
        mass += p[idx]
    end
    (isfinite(mass) && mass > zero(eltype(p))) || return -Inf
    return log(mass)
end

function Distributions.logpdf(dist::CoarsedObservedStatesMarkovModel, y)
    _omm_coarsed_observation_indices(dist.base_dist.state_labels, y)
    return -Inf
end

Distributions.pdf(dist::CoarsedObservedStatesMarkovModel, y) = exp(logpdf(dist, y))

Distributions.rand(rng::AbstractRNG, dist::CoarsedObservedStatesMarkovModel) =
    rand(rng, dist.base_dist)

Distributions.mean(dist::CoarsedObservedStatesMarkovModel) = mean(dist.base_dist)
Distributions.var(dist::CoarsedObservedStatesMarkovModel) = var(dist.base_dist)
Distributions.cdf(dist::CoarsedObservedStatesMarkovModel, y::Real) = cdf(dist.base_dist, y)
Distributions.params(dist::CoarsedObservedStatesMarkovModel) = params(dist.base_dist)
Base.length(dist::CoarsedObservedStatesMarkovModel) = length(dist.base_dist)
