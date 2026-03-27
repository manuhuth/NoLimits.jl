export ContinuousTimeObservedStatesMarkovModel

using Distributions, ExponentialAction, Random

"""
    ContinuousTimeObservedStatesMarkovModel(transition_matrix, initial_dist, Δt)
    ContinuousTimeObservedStatesMarkovModel(transition_matrix, initial_dist, Δt, state_labels)
    <: Distribution{Univariate, Discrete}

A continuous-time Markov model with **fully observed** states. State propagation is performed
via the matrix exponential `exp(Q · Δt)` where `Q` is the rate matrix (generator). Unlike the
CT-HMM variants, the latent state is directly observable — the observation IS the state label.
The posterior after observing a state is therefore a one-hot vector (certainty).

Implements the `Distributions.jl` interface (`logpdf`, `pdf`, `rand`, `mean`, `var`, `cdf`).
Used as an observation distribution in `@formulas` blocks.

# Arguments
- `transition_matrix::AbstractMatrix{<:Real}`: rate matrix (generator) of shape
  `(n_states, n_states)`. Off-diagonal entries must be non-negative; each row must sum to zero.
- `initial_dist::Distributions.Categorical`: prior over states at the **previous** observation
  time. (Same convention as CT-HMM variants.)
- `Δt::Real`: time elapsed since the previous observation.
- `state_labels::Vector{T}`: maps state index → label. Defaults to `[1, 2, ..., n_states]`.
- `propagation_mode::Symbol`: `:auto` (default), `:expv`, or `:pathsum`. Controls how
  `exp(Q · Δt)` is computed. See `ContinuousTimeDiscreteStatesHMM` for details.

# Missing data
When the observation is `missing`, the predicted state distribution (via `exp(Q·Δt)`) is
propagated forward without a likelihood contribution.
"""
struct ContinuousTimeObservedStatesMarkovModel{
    M <: AbstractMatrix{<:Real},
    D <: Distributions.Categorical,
    T,
    Δ <: Real,
} <: Distribution{Univariate, Discrete}
    n_states          :: Int
    transition_matrix :: M
    initial_dist      :: D
    Δt                :: Δ
    state_labels      :: Vector{T}
    propagation_mode  :: Symbol
end

# --- Constructors ---

function ContinuousTimeObservedStatesMarkovModel(
    transition_matrix :: AbstractMatrix{<:Real},
    initial_dist      :: Distributions.Categorical,
    Δt                :: Real,
    state_labels      :: Vector{T};
    propagation_mode  :: Symbol = :auto,
) where T
    _ct_hmm_validate_mode(propagation_mode)
    n_states = size(transition_matrix, 1)
    size(transition_matrix, 2) == n_states ||
        error("transition_matrix must be square, got $(size(transition_matrix)).")
    length(initial_dist.p) == n_states ||
        error("length(initial_dist.p) must equal n_states ($n_states), " *
              "got $(length(initial_dist.p)).")
    length(state_labels) == n_states ||
        error("length(state_labels) must equal n_states ($n_states), " *
              "got $(length(state_labels)).")
    return ContinuousTimeObservedStatesMarkovModel(
        n_states, transition_matrix, initial_dist, Δt, state_labels, propagation_mode)
end

# Default constructor: integer labels 1..n_states
function ContinuousTimeObservedStatesMarkovModel(
    transition_matrix :: AbstractMatrix{<:Real},
    initial_dist      :: Distributions.Categorical,
    Δt                :: Real;
    propagation_mode  :: Symbol = :auto,
)
    n_states = size(transition_matrix, 1)
    return ContinuousTimeObservedStatesMarkovModel(
        transition_matrix, initial_dist, Δt, collect(1:n_states);
        propagation_mode=propagation_mode)
end

# --- State label lookup ---

function _find_state_index(dist::ContinuousTimeObservedStatesMarkovModel, y)
    return findfirst(==(y), dist.state_labels)
end

# --- Hidden state probabilities (shared interface with HMM variants) ---

"""
    probabilities_hidden_states(dist::ContinuousTimeObservedStatesMarkovModel) -> Vector

Marginal prior probabilities of the state at the current observation time, propagated from
`dist.initial_dist` via `exp(Q · Δt)`. Reuses the CT-HMM propagation kernel.
"""
function probabilities_hidden_states(dist::ContinuousTimeObservedStatesMarkovModel)
    return _ct_hmm_probabilities_hidden_states(
        dist.transition_matrix, dist.initial_dist.p, dist.Δt;
        mode=dist.propagation_mode)
end

"""
    posterior_hidden_states(dist::ContinuousTimeObservedStatesMarkovModel, y)

Returns the one-hot posterior after observing state `y`. Since the state is fully observed,
the posterior is degenerate: probability 1 at the index of `y`, 0 elsewhere.

Returns a zero vector if `y` is not found in `state_labels`.
"""
function posterior_hidden_states(dist::ContinuousTimeObservedStatesMarkovModel, y)
    idx = _find_state_index(dist, y)
    post = zeros(Float64, dist.n_states)
    idx !== nothing && (post[idx] = 1.0)
    return post
end

# --- Distributions.jl interface ---

function Distributions.logpdf(dist::ContinuousTimeObservedStatesMarkovModel, y)
    idx = _find_state_index(dist, y)
    idx === nothing && return -Inf
    p = probabilities_hidden_states(dist)
    return log(p[idx])
end

Distributions.pdf(dist::ContinuousTimeObservedStatesMarkovModel, y) = exp(logpdf(dist, y))

function Distributions.rand(rng::AbstractRNG, dist::ContinuousTimeObservedStatesMarkovModel)
    p   = probabilities_hidden_states(dist)
    idx = rand(rng, Categorical(p))
    return dist.state_labels[idx]
end

# mean/var/cdf only defined for numeric (Real) label types
function Distributions.mean(dist::ContinuousTimeObservedStatesMarkovModel{M, D, T}) where {M, D, T}
    T <: Real || throw(ArgumentError(
        "mean is not defined for ContinuousTimeObservedStatesMarkovModel with label type $T. " *
        "Only Real-valued labels are supported."))
    p = probabilities_hidden_states(dist)
    return sum(p[k] * dist.state_labels[k] for k in 1:dist.n_states)
end

function Distributions.var(dist::ContinuousTimeObservedStatesMarkovModel{M, D, T}) where {M, D, T}
    T <: Real || throw(ArgumentError(
        "var is not defined for ContinuousTimeObservedStatesMarkovModel with label type $T. " *
        "Only Real-valued labels are supported."))
    p = probabilities_hidden_states(dist)
    μ = sum(p[k] * dist.state_labels[k] for k in 1:dist.n_states)
    return sum(p[k] * (dist.state_labels[k] - μ)^2 for k in 1:dist.n_states)
end

function Distributions.cdf(dist::ContinuousTimeObservedStatesMarkovModel{M, D, T}, y::Real) where {M, D, T}
    T <: Real || throw(ArgumentError(
        "cdf is not defined for ContinuousTimeObservedStatesMarkovModel with label type $T. " *
        "Only Real-valued labels are supported."))
    p = probabilities_hidden_states(dist)
    return sum((p[k] for k in 1:dist.n_states if dist.state_labels[k] <= y); init=zero(eltype(p)))
end

Distributions.params(dist::ContinuousTimeObservedStatesMarkovModel) =
    (dist.transition_matrix, dist.initial_dist, dist.Δt, dist.state_labels)

Base.length(dist::ContinuousTimeObservedStatesMarkovModel) = 1
