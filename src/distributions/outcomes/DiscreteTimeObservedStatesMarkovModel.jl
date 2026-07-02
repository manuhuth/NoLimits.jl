export DiscreteTimeObservedStatesMarkovModel
export probabilities_hidden_states, posterior_hidden_states

using Distributions, Random

"""
    DiscreteTimeObservedStatesMarkovModel(transition_matrix, initial_dist)
    DiscreteTimeObservedStatesMarkovModel(transition_matrix, initial_dist, state_labels)
    <: Distribution{Univariate, Discrete}

A discrete-time Markov model with observed states. Unlike the HMM variants, the latent state is
directly observable when a single state label is provided.

Implements the `Distributions.jl` interface (`logpdf`, `pdf`, `rand`, `mean`, `var`, `cdf`).
Used as an observation distribution in `@formulas` blocks.

# Arguments
- `transition_matrix::AbstractMatrix{<:Real}`: row-stochastic matrix of shape `(n_states, n_states)`.
  Entry `[i, j]` is `P(State_t = j | State_{t-1} = i)`.
- `initial_dist::Distributions.Categorical`: prior over states at the **previous** observation
  time. Propagated one step via `transition_matrix` before computing the likelihood.
  (Same convention as HMM variants.)
- `state_labels::Vector{T}`: maps state index → label. Defaults to `[1, 2, ..., n_states]`.

# Missing data
When the observation is `missing`, the predicted state distribution (after one transition step)
is propagated forward without a likelihood contribution — identical to the HMM missing-data
behavior.
"""
struct DiscreteTimeObservedStatesMarkovModel{
    M <: AbstractMatrix{<:Real},
    D <: Distributions.Categorical,
    T
} <: Distribution{Univariate, Discrete}
    n_states::Int
    transition_matrix::M
    initial_dist::D
    state_labels::Vector{T}
end

# --- Constructors ---

function DiscreteTimeObservedStatesMarkovModel(
        transition_matrix::AbstractMatrix{<:Real},
        initial_dist::Distributions.Categorical,
        state_labels::Vector{T}
) where {T}
    n_states = size(transition_matrix, 1)
    size(transition_matrix, 2) == n_states ||
        error("transition_matrix must be square, got $(size(transition_matrix)).")
    length(initial_dist.p) == n_states ||
        error("length(initial_dist.p) must equal n_states ($n_states), " *
              "got $(length(initial_dist.p)).")
    length(state_labels) == n_states ||
        error("length(state_labels) must equal n_states ($n_states), " *
              "got $(length(state_labels)).")
    return DiscreteTimeObservedStatesMarkovModel(
        n_states, transition_matrix, initial_dist, state_labels)
end

# Default constructor: integer labels 1..n_states
function DiscreteTimeObservedStatesMarkovModel(
        transition_matrix::AbstractMatrix{<:Real},
        initial_dist::Distributions.Categorical
)
    n_states = size(transition_matrix, 1)
    return DiscreteTimeObservedStatesMarkovModel(
        transition_matrix, initial_dist, collect(1:n_states))
end

@inline _omm_is_observed_markov_dist(::DiscreteTimeObservedStatesMarkovModel) = true

# --- Hidden state probabilities (shared interface with HMM variants) ---

"""
    probabilities_hidden_states(dist::DiscreteTimeObservedStatesMarkovModel) -> Vector

Marginal prior probabilities of the state at the current observation time, propagated one
step from `dist.initial_dist` via `dist.transition_matrix`.
"""
function probabilities_hidden_states(dist::DiscreteTimeObservedStatesMarkovModel)
    p = transpose(dist.transition_matrix) * dist.initial_dist.p
    return p ./ sum(p)
end

# The posterior/logpdf/pdf/rand/moments interface is shared with the
# continuous-time observed-states model and lives in
# ContinuousTimeObservedStatesMarkovModel.jl (defined once against the
# `_ObservedStatesMarkovModel` Union after both types exist).

function Distributions.params(dist::DiscreteTimeObservedStatesMarkovModel)
    (dist.transition_matrix, dist.initial_dist, dist.state_labels)
end
