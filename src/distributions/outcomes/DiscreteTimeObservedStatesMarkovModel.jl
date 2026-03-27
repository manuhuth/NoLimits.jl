export DiscreteTimeObservedStatesMarkovModel
export probabilities_hidden_states, posterior_hidden_states

using Distributions, Random

"""
    DiscreteTimeObservedStatesMarkovModel(transition_matrix, emission_dists, initial_dist)
    DiscreteTimeObservedStatesMarkovModel(transition_matrix, emission_dists, initial_dist, state_labels)
    <: Distribution{Univariate, Discrete}

A discrete-time Markov model with **fully observed** states. Unlike the HMM variants, the latent
state is directly observable — the observation IS the state label. The posterior after observing
a state is therefore a one-hot vector (certainty), making forward filtering exact and efficient.

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
behaviour.
"""
struct DiscreteTimeObservedStatesMarkovModel{
    M <: AbstractMatrix{<:Real},
    D <: Distributions.Categorical,
    T,
} <: Distribution{Univariate, Discrete}
    n_states          :: Int
    transition_matrix :: M
    initial_dist      :: D
    state_labels      :: Vector{T}
end

# --- Constructors ---

function DiscreteTimeObservedStatesMarkovModel(
    transition_matrix :: AbstractMatrix{<:Real},
    initial_dist      :: Distributions.Categorical,
    state_labels      :: Vector{T},
) where T
    n_states = size(transition_matrix, 1)
    size(transition_matrix, 2) == n_states ||
        error("transition_matrix must be square, got $(size(transition_matrix)).")
    length(initial_dist.p) == n_states ||
        error("length(initial_dist.p) must equal n_states ($n_states), " *
              "got $(length(initial_dist.p)).")
    length(state_labels) == n_states ||
        error("length(state_labels) must equal n_states ($n_states), " *
              "got $(length(state_labels)).")
    return DiscreteTimeObservedStatesMarkovModel(n_states, transition_matrix, initial_dist, state_labels)
end

# Default constructor: integer labels 1..n_states
function DiscreteTimeObservedStatesMarkovModel(
    transition_matrix :: AbstractMatrix{<:Real},
    initial_dist      :: Distributions.Categorical,
)
    n_states = size(transition_matrix, 1)
    return DiscreteTimeObservedStatesMarkovModel(
        transition_matrix, initial_dist, collect(1:n_states))
end

# --- State label lookup ---

function _find_state_index(dist::DiscreteTimeObservedStatesMarkovModel, y)
    return findfirst(==(y), dist.state_labels)
end

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

"""
    posterior_hidden_states(dist::DiscreteTimeObservedStatesMarkovModel, y)

Returns the one-hot posterior after observing state `y`. Since the state is fully observed,
the posterior is degenerate: probability 1 at the index of `y`, 0 elsewhere.

Returns a zero vector if `y` is not found in `state_labels`.
"""
function posterior_hidden_states(dist::DiscreteTimeObservedStatesMarkovModel, y)
    idx = _find_state_index(dist, y)
    post = zeros(Float64, dist.n_states)
    idx !== nothing && (post[idx] = 1.0)
    return post
end

# --- Distributions.jl interface ---

function Distributions.logpdf(dist::DiscreteTimeObservedStatesMarkovModel, y)
    idx = _find_state_index(dist, y)
    idx === nothing && return -Inf
    p = probabilities_hidden_states(dist)
    return log(p[idx])
end

Distributions.pdf(dist::DiscreteTimeObservedStatesMarkovModel, y) = exp(logpdf(dist, y))

function Distributions.rand(rng::AbstractRNG, dist::DiscreteTimeObservedStatesMarkovModel)
    p = probabilities_hidden_states(dist)
    idx = rand(rng, Categorical(p))
    return dist.state_labels[idx]
end

# mean/var/cdf only defined for numeric (Real) label types
function Distributions.mean(dist::DiscreteTimeObservedStatesMarkovModel{M, D, T}) where {M, D, T}
    T <: Real || throw(ArgumentError(
        "mean is not defined for DiscreteTimeObservedStatesMarkovModel with label type $T. " *
        "Only Real-valued labels are supported."))
    p = probabilities_hidden_states(dist)
    return sum(p[k] * dist.state_labels[k] for k in 1:dist.n_states)
end

function Distributions.var(dist::DiscreteTimeObservedStatesMarkovModel{M, D, T}) where {M, D, T}
    T <: Real || throw(ArgumentError(
        "var is not defined for DiscreteTimeObservedStatesMarkovModel with label type $T. " *
        "Only Real-valued labels are supported."))
    p  = probabilities_hidden_states(dist)
    μ  = sum(p[k] * dist.state_labels[k] for k in 1:dist.n_states)
    return sum(p[k] * (dist.state_labels[k] - μ)^2 for k in 1:dist.n_states)
end

function Distributions.cdf(dist::DiscreteTimeObservedStatesMarkovModel{M, D, T}, y::Real) where {M, D, T}
    T <: Real || throw(ArgumentError(
        "cdf is not defined for DiscreteTimeObservedStatesMarkovModel with label type $T. " *
        "Only Real-valued labels are supported."))
    p = probabilities_hidden_states(dist)
    return sum((p[k] for k in 1:dist.n_states if dist.state_labels[k] <= y); init=zero(eltype(p)))
end

Distributions.params(dist::DiscreteTimeObservedStatesMarkovModel) =
    (dist.transition_matrix, dist.initial_dist, dist.state_labels)

Base.length(dist::DiscreteTimeObservedStatesMarkovModel) = 1
