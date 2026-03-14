export ContinuousTimeDiscreteStatesHMM

using Distributions, ExponentialAction, Random, Lux
import Distributions: pdf, logpdf, rand, mean, var, median, quantile, mode, cdf, support, params

_ct_hmm_validate_mode(mode::Symbol) =
    mode in (:auto, :expv, :pathsum) ||
    error("Invalid CT-HMM propagation mode $(mode). Use one of :auto, :expv, :pathsum.")

function _ct_hmm_adjacency(transition_matrix::AbstractMatrix{<:Real}; atol::Real=1e-12)
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

function _ct_hmm_is_acyclic(transition_matrix::AbstractMatrix{<:Real}; atol::Real=1e-12)
    adj = _ct_hmm_adjacency(transition_matrix; atol=atol)
    return _ct_hmm_topological_order(adj) !== nothing
end

function _ct_hmm_path_kernel(lambdas::AbstractVector{<:Real}, Δt::Real; atol::Real=1e-10)
    m = length(lambdas)
    m == 1 && return exp(-lambdas[1] * Δt)

    acc = zero(promote_type(eltype(lambdas), typeof(Δt)))
    for r in 1:m
        denom = one(acc)
        λr = lambdas[r]
        for l in 1:m
            l == r && continue
            d = lambdas[l] - λr
            abs(d) <= atol && return nothing
            denom *= d
        end
        acc += exp(-λr * Δt) / denom
    end
    return acc
end

function _ct_hmm_collect_paths(adj::Vector{Vector{Int}}, src::Int)
    paths = Dict{Int, Vector{Vector{Int}}}()
    cur = Int[src]

    function dfs(v::Int)
        for u in adj[v]
            push!(cur, u)
            if !haskey(paths, u)
                paths[u] = Vector{Vector{Int}}()
            end
            push!(paths[u], copy(cur))
            dfs(u)
            pop!(cur)
        end
    end

    dfs(src)
    return paths
end

function _ct_hmm_transition_matrix_pathsum(transition_matrix::AbstractMatrix{<:Real}, Δt::Real; atol::Real=1e-10)
    n_states = size(transition_matrix, 1)
    Tprob = zeros(promote_type(eltype(transition_matrix), typeof(Δt)), n_states, n_states)
    adj = _ct_hmm_adjacency(transition_matrix; atol=atol)
    exit_rates = [-transition_matrix[s, s] for s in 1:n_states]

    for src in 1:n_states
        Tprob[src, src] = exp(-exit_rates[src] * Δt)
        paths_to = _ct_hmm_collect_paths(adj, src)
        for (dst, paths) in paths_to
            total = zero(eltype(Tprob))
            for path in paths
                weight = one(eltype(Tprob))
                lambdas = similar(exit_rates, length(path))
                for k in eachindex(path)
                    lambdas[k] = exit_rates[path[k]]
                end
                for k in 1:(length(path)-1)
                    weight *= transition_matrix[path[k], path[k+1]]
                end
                kernel = _ct_hmm_path_kernel(lambdas, Δt; atol=atol)
                kernel === nothing && return nothing
                total += weight * kernel
            end
            Tprob[src, dst] = total
        end
    end

    return Tprob
end

function _ct_hmm_probabilities_pathsum_dag(
    transition_matrix::AbstractMatrix{<:Real},
    initial_p::AbstractVector{<:Real},
    Δt::Real;
    atol::Real=1e-10
)
    n_states = size(transition_matrix, 1)
    adj = _ct_hmm_adjacency(transition_matrix; atol=atol)
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

function _ct_hmm_probabilities_hidden_states(
    transition_matrix::AbstractMatrix{<:Real},
    initial_p::AbstractVector{<:Real},
    Δt::Real;
    mode::Symbol=:auto,
    atol::Real=1e-10
)
    _ct_hmm_validate_mode(mode)

    if mode == :expv
        return expv(Δt, transpose(transition_matrix), initial_p)
    end

    if mode == :pathsum || mode == :auto
        p = _ct_hmm_probabilities_pathsum_dag(transition_matrix, initial_p, Δt; atol=atol)
        if p !== nothing
            return p
        end
        if mode == :pathsum
            error("propagation_mode=:pathsum requested but path-sum is not applicable " *
                  "(cyclic graph or numerically degenerate exit rates).")
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
    M<:AbstractMatrix{<:Real},
    E<:Tuple,
    D<:Distributions.Categorical,
    T<:Real,
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
    propagation_mode::Symbol=:auto
)
    _ct_hmm_validate_mode(propagation_mode)
    n_states = size(transition_matrix, 1)
    ContinuousTimeDiscreteStatesHMM(
        n_states, transition_matrix, emission_dists, initial_dist, Δt, propagation_mode)
end

# Transition Matrix must be transposed here as expv uses column sums that equal zero
probabilities_hidden_states(hmm::ContinuousTimeDiscreteStatesHMM) =
    _ct_hmm_probabilities_hidden_states(
        hmm.transition_matrix, hmm.initial_dist.p, hmm.Δt; mode=hmm.propagation_mode)

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
    return _hmm_logsumexp(log_p_hidden .+ log_p_obs)
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
