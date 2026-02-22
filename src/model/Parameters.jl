using LinearAlgebra
using Parameters
using Distributions
using Lux
using Optimisers
using Random
using NormalizingFlows
using FunctionChains


export AbstractParameterBlock
export RealNumber, RealVector, RealPSDMatrix, RealDiagonalMatrix, NNParameters, NPFParameter, SoftTreeParameters, SplineParameters
export ProbabilityVector, DiscreteTransitionMatrix, ContinuousTransitionMatrix
export Priorless

"""
    Priorless()

Sentinel type indicating that no prior distribution is assigned to a parameter.
Used as the default `prior` value in all parameter block constructors.
"""
struct Priorless end

"""
    AbstractParameterBlock

Abstract base type for all parameter block types used in `@fixedEffects`.

Concrete subtypes: [`RealNumber`](@ref), [`RealVector`](@ref),
[`RealPSDMatrix`](@ref), [`RealDiagonalMatrix`](@ref),
[`NNParameters`](@ref), [`SoftTreeParameters`](@ref),
[`SplineParameters`](@ref), [`NPFParameter`](@ref).
"""
abstract type AbstractParameterBlock end

"""
    RealNumber(value; name, scale, lower, upper, prior, calculate_se) -> RealNumber

A scalar real-valued fixed-effect parameter block.

# Arguments
- `value::Real`: initial value on the natural (untransformed) scale.

# Keyword Arguments
- `name::Symbol = :unnamed`: parameter name (injected automatically by `@fixedEffects`).
- `scale::Symbol = :identity`: reparameterisation applied during optimisation.
  Must be one of `REAL_SCALES` (`:identity`, `:log`).
- `lower::Real = -Inf`: lower bound on the natural scale (defaults to `EPSILON` when `scale=:log`).
- `upper::Real = Inf`: upper bound on the natural scale.
- `prior = Priorless()`: prior distribution (`Distributions.Distribution`) or `Priorless()`.
- `calculate_se::Bool = false`: whether to include this parameter in standard-error calculations.
"""
@with_kw struct RealNumber{T<:Real} <: AbstractParameterBlock
    name::Symbol = :unnamed
    value::T
    scale::Symbol = :identity
    lower::T = -Inf
    upper::T = Inf
    prior = Priorless()
    calculate_se::Bool = false
end

function RealNumber(value::Real; name::Symbol = :unnamed, scale::Symbol = :identity,
    lower::Real = -Inf, upper::Real = Inf, prior = Priorless(), calculate_se::Bool = false)
    _check_prior(prior, name)
    scale in REAL_SCALES || error("Invalid scale for parameter $(name). Expected one of $(REAL_SCALES); got $(scale).")
    scale == :log && lower == -Inf && (lower = EPSILON)
    lower < upper || error("Invalid bounds for parameter $(name). Expected lower < upper; got lower=$(lower), upper=$(upper).")
    if scale == :log
        lower >= 0 || error("Invalid lower bound for parameter $(name). Expected lower > 0 for scale :log; got lower=$(lower).")
    end
    if scale == :logit
        (value > 0 && value < 1) || error("Invalid initial value for parameter $(name). Expected value ∈ (0, 1) for scale :logit; got value=$(value).")
    end
    T = value isa AbstractFloat ? typeof(value) : Float64
    v = T(value)
    l = T(lower)
    u = T(upper)
    v >= l && v <= u || error("Initial value out of bounds for parameter $(name). Expected $(l) ≤ value ≤ $(u); got value=$(v).")
    return RealNumber{T}(name, v, scale, l, u, prior, calculate_se)
end

"""
    RealVector(value; name, scale, lower, upper, prior, calculate_se) -> RealVector

A vector of real-valued fixed-effect parameters with per-element scale options.

# Arguments
- `value::AbstractVector{<:Real}`: initial values on the natural scale.

# Keyword Arguments
- `name::Symbol = :unnamed`: parameter name (injected automatically by `@fixedEffects`).
- `scale`: per-element scale symbols. A single `Symbol` or a `Vector{Symbol}` of
  the same length as `value`. Each element must be in `REAL_SCALES` (`:identity`, `:log`).
  Defaults to all `:identity`.
- `lower`: lower bounds per element. Defaults to `-Inf` (or `EPSILON` for `:log` elements).
- `upper`: upper bounds per element. Defaults to `Inf`.
- `prior = Priorless()`: a `Distributions.Distribution`, a `Vector{Distribution}` of
  matching length, or `Priorless()`.
- `calculate_se::Bool = false`: whether to include this parameter in standard-error calculations.
"""
@with_kw struct RealVector{T<:Real, VT<:AbstractVector{T}} <: AbstractParameterBlock
    name::Symbol = :unnamed
    value::VT
    scale::Vector{Symbol} = fill(:identity, length(value))
    lower::VT = fill(-Inf, length(value))
    upper::VT = fill(Inf, length(value))
    prior = Priorless()
    calculate_se::Bool = false
end

function RealVector(value::AbstractVector{<:Real};
    name::Symbol = :unnamed,
    scale = fill(:identity, length(value)),
    lower = fill(-Inf, length(value)),
    upper = fill(Inf, length(value)),
    prior = Priorless(),
    calculate_se::Bool = false)
    _check_prior(prior, name)
    all(s -> s in REAL_SCALES, scale) || error("Invalid scale for parameter $(name). Expected each scale in $(REAL_SCALES); got $(scale).")
    s = collect(scale)
    if any(s .== :log)
        lower = map((l, sc) -> (sc == :log && l == -Inf) ? EPSILON : l, lower, s)
    end
    for (idx, (sc, vi)) in enumerate(zip(s, value))
        if sc == :logit && !(vi > 0 && vi < 1)
            error("Invalid initial value for parameter $(name) at index $(idx). Expected value ∈ (0, 1) for scale :logit; got value=$(vi).")
        end
    end

    T = eltype(value) <: AbstractFloat ? eltype(value) : Float64
    v = T.(value)
    l = T.(lower)
    u = T.(upper)

    length(v) == length(s) || error("Scale length mismatch for parameter $(name). Expected length $(length(v)); got $(length(s)).")
    length(v) == length(l) || error("Lower bound length mismatch for parameter $(name). Expected length $(length(v)); got $(length(l)).")
    length(v) == length(u) || error("Upper bound length mismatch for parameter $(name). Expected length $(length(v)); got $(length(u)).")
    all(l .< u) || error("Invalid bounds for parameter $(name). Expected all lower < upper; got lower=$(l), upper=$(u).")
    for (idx, (sc, lb)) in enumerate(zip(s, l))
        if sc == :log && lb < 0
            error("Invalid lower bound for parameter $(name) at index $(idx). Expected lower > 0 for scale :log; got lower=$(lb).")
        end
    end
    if !all((v .>= l) .& (v .<= u))
        bad = findall(.!((v .>= l) .& (v .<= u)))
        error("Initial values out of bounds for parameter $(name). Expected l ≤ value ≤ u. Violations at indices $(bad); value=$(v), lower=$(l), upper=$(u).")
    end
    return RealVector{T, typeof(v)}(name, v, Vector{Symbol}(s), l, u, prior, calculate_se)
end

"""
    RealPSDMatrix(value; name, scale, prior, calculate_se) -> RealPSDMatrix

A symmetric positive semi-definite (PSD) matrix parameter block, typically used
to parameterise covariance matrices of random-effect distributions.

The matrix is reparameterised during optimisation to ensure PSD constraints are
automatically satisfied.

# Arguments
- `value::AbstractMatrix{<:Real}`: initial symmetric PSD matrix.

# Keyword Arguments
- `name::Symbol = :unnamed`: parameter name (injected automatically by `@fixedEffects`).
- `scale::Symbol = :cholesky`: reparameterisation. Must be one of `PSD_SCALES`
  (`:cholesky`, `:expm`).
- `prior = Priorless()`: a `Distributions.Distribution` (e.g. `Wishart`) or `Priorless()`.
- `calculate_se::Bool = false`: whether to include this parameter in standard-error calculations.
"""
@with_kw struct RealPSDMatrix{T<:Real, MT<:AbstractMatrix{T}} <: AbstractParameterBlock
    name::Symbol = :unnamed
    value::MT
    scale::Symbol = :cholesky
    prior = Priorless()
    calculate_se::Bool = false
end

function _is_psd(mat::AbstractMatrix{<:Real}; atol::Real = EPSILON)
    issymmetric(mat) || return false
    vals = eigen(Symmetric(mat)).values
    return minimum(vals) >= -atol
end

function RealPSDMatrix(value::AbstractMatrix{<:Real}; name::Symbol = :unnamed,
    scale::Symbol = :cholesky, prior = Priorless(), calculate_se::Bool = false)
    _check_prior(prior, name)
    scale in PSD_SCALES || error("Invalid scale for parameter $(name). Expected one of $(PSD_SCALES); got $(scale).")
    T = eltype(value) <: AbstractFloat ? eltype(value) : Float64
    v = T.(value)
    _is_psd(v) || error("Invalid initial value for parameter $(name). Expected symmetric positive semi-definite matrix; got matrix with min eigenvalue $(minimum(eigen(Symmetric(v)).values)).")
    return RealPSDMatrix{T, typeof(v)}(name, v, scale, prior, calculate_se)
end

"""
    RealDiagonalMatrix(value; name, scale, prior, calculate_se) -> RealDiagonalMatrix

A diagonal positive-definite matrix parameter block, stored as a vector of the
diagonal entries. Useful for diagonal covariance matrices.

All diagonal entries must be strictly positive. They are stored and optimised on the
log scale.

# Arguments
- `value`: initial diagonal entries as an `AbstractVector{<:Real}` or a diagonal
  `AbstractMatrix`. If a matrix is provided, off-diagonal entries are ignored with a warning.

# Keyword Arguments
- `name::Symbol = :unnamed`: parameter name (injected automatically by `@fixedEffects`).
- `scale::Symbol = :log`: reparameterisation. Must be in `DIAGONAL_SCALES` (`:log`).
- `prior = Priorless()`: a `Distributions.Distribution` or `Priorless()`.
- `calculate_se::Bool = false`: whether to include this parameter in standard-error calculations.
"""
@with_kw struct RealDiagonalMatrix{T<:Real, VT<:AbstractVector{T}} <: AbstractParameterBlock
    name::Symbol = :unnamed
    value::VT
    scale::Symbol = :identity
    prior = Priorless()
    calculate_se::Bool = false
end

function RealDiagonalMatrix(value::AbstractVector{<:Real}; name::Symbol = :unnamed,
    scale::Symbol = :log, prior = Priorless(), calculate_se::Bool = false)
    _check_prior(prior, name)
    scale in DIAGONAL_SCALES || error("Invalid scale for parameter $(name). Expected one of $(DIAGONAL_SCALES); got $(scale).")
    T = eltype(value) <: AbstractFloat ? eltype(value) : Float64
    v = T.(value)
    all(v .> 0) || error("Invalid diagonal values for parameter $(name). Expected all entries > 0 for scale :log; got values=$(v).")
    return RealDiagonalMatrix{T, typeof(v)}(name, v, scale, prior, calculate_se)
end

function RealDiagonalMatrix(value::AbstractMatrix{<:Real}; name::Symbol = :unnamed,
    scale::Symbol = :log, prior = Priorless(), calculate_se::Bool = false)
    diag_only = Diagonal(value)
    if !isapprox(Matrix(diag_only), Matrix(value); atol=zero(eltype(value)), rtol=zero(eltype(value)))
        @warn "RealDiagonalMatrix received a matrix with non-zero off-diagonals for parameter $(name). Using diagonal entries only."
    end
    return RealDiagonalMatrix(diag(value); name=name, scale=scale, prior=prior, calculate_se=calculate_se)
end

"""
    NNParameters(chain; name, function_name, seed, prior, calculate_se) -> NNParameters

A parameter block that wraps the flattened parameters of a Lux.jl neural-network chain.

The resulting parameter is optimised as a flat real vector. Inside model blocks
(`@randomEffects`, `@preDifferentialEquation`, `@formulas`) the network is called
as `function_name(input, θ_slice)`, where `θ_slice` is the corresponding slice of
the fixed-effects `ComponentArray`.

# Arguments
- `chain`: a `Lux.Chain` defining the neural-network architecture.

# Keyword Arguments
- `name::Symbol = :unnamed`: parameter name (injected automatically by `@fixedEffects`).
- `function_name::Symbol`: the name used to call the network in model blocks.
- `seed::Integer = 0`: random seed for initialising the Lux parameters.
- `prior = Priorless()`: `Priorless()`, a `Vector{Distribution}` of length equal to
  the number of parameters, or a multivariate `Distribution` with matching `length`.
- `calculate_se::Bool = false`: whether to include this parameter in standard-error calculations.
"""
@with_kw struct NNParameters{T<:Real, VT<:AbstractVector{T}, C, R} <: AbstractParameterBlock
    name::Symbol = :unnamed
    function_name::Symbol
    chain::C
    value::VT
    reconstructor::R
    lower::VT = fill(-Inf, length(value))
    upper::VT = fill(Inf, length(value))
    prior = Priorless()
    calculate_se::Bool = false
end

function NNParameters(chain; name::Symbol = :unnamed, function_name::Symbol, seed::Integer = 0,
    prior = Priorless(), calculate_se::Bool = false)
    if !isa(chain, Lux.Chain) 
        error("Invalid chain for parameter $(name). Expected a Lux chain; got $(typeof(chain)).")
    end
    rng = Xoshiro(seed)
    init_params = Lux.initialparameters(rng, chain)
    flat, reconstructor = Optimisers.destructure(init_params)
    T = eltype(flat) <: AbstractFloat ? eltype(flat) : Float64
    v = T.(flat)
    l = fill(T(-Inf), length(v))
    u = fill(T(Inf), length(v))
    _check_nn_prior(prior, name, length(v))
    return NNParameters{T, typeof(v), typeof(chain), typeof(reconstructor)}(name, function_name, chain, v, reconstructor, l, u, prior, calculate_se)
end

"""
    NPFParameter(n_input, n_layers; name, seed, init, prior, calculate_se) -> NPFParameter

A parameter block for a Normalizing Planar Flow (NPF), enabling flexible non-Gaussian
distributions in `@randomEffects` via `RandomEffect(NormalizingPlanarFlow(ψ); column=:ID)`.

The flow is composed of `n_layers` planar transformations on an `n_input`-dimensional
base Gaussian. Parameters are stored as a flat real vector.

# Arguments
- `n_input::Integer`: dimensionality of the latent space (typically 1 for scalar random effects).
- `n_layers::Integer`: number of planar flow layers.

# Keyword Arguments
- `name::Symbol = :unnamed`: parameter name (injected automatically by `@fixedEffects`).
- `seed::Integer = 0`: random seed for initialisation.
- `init::Function`: weight initialisation function; defaults to `x -> sqrt(1/n_input) .* x`.
- `prior = Priorless()`: `Priorless()`, a `Vector{Distribution}`, or a multivariate `Distribution`.
- `calculate_se::Bool = false`: whether to include this parameter in standard-error calculations.
"""
@with_kw struct NPFParameter{T<:Real, VT<:AbstractVector{T}, R} <: AbstractParameterBlock
    name::Symbol = :unnamed
    n_input::Int
    n_layers::Int
    seed::Int = 0
    init::Function = x -> sqrt((1 / n_input)) .* x
    value::VT
    reconstructor::R
    lower::VT = fill(-Inf, length(value))
    upper::VT = fill(Inf, length(value))
    prior = Priorless()
    calculate_se::Bool = false
end

function NPFParameter(n_input::Integer, n_layers::Integer; name::Symbol = :unnamed, seed::Integer = 0,
    init::Function = x -> sqrt((1 / n_input)) .* x, prior = Priorless(), calculate_se::Bool = false)
    n_input > 0 || error("Invalid n_input for parameter $(name). Expected n_input > 0; got $(n_input).")
    n_layers > 0 || error("Invalid n_layers for parameter $(name). Expected n_layers > 0; got $(n_layers).")
    rng = Xoshiro(seed)
    d = Int(n_input)
    Ls = [PlanarLayer(d, init) for _ in 1:Int(n_layers)]
    ts = fchain(Ls)
    flat, reconstructor = Optimisers.destructure(ts)
    T = eltype(flat) <: AbstractFloat ? eltype(flat) : Float64
    v = T.(flat)
    l = fill(T(-Inf), length(v))
    u = fill(T(Inf), length(v))
    _check_nn_prior(prior, name, length(v))
    return NPFParameter{T, typeof(v), typeof(reconstructor)}(name, d, Int(n_layers), Int(seed), init, v, reconstructor, l, u, prior, calculate_se)
end

"""
    SplineParameters(knots; name, function_name, degree, prior, calculate_se) -> SplineParameters

A parameter block for a B-spline function whose coefficients are optimised as fixed effects.

The number of coefficients is determined by `length(knots) - degree - 1`. All coefficients
are initialised to zero. Inside model blocks the spline is evaluated as
`function_name(x, θ_slice)`.

# Arguments
- `knots::AbstractVector{<:Real}`: B-spline knot vector (including boundary knots).

# Keyword Arguments
- `name::Symbol = :unnamed`: parameter name (injected automatically by `@fixedEffects`).
- `function_name::Symbol`: the name used to call the spline in model blocks.
- `degree::Integer = 3`: polynomial degree of the B-spline (e.g. `2` for quadratic, `3` for cubic).
- `prior = Priorless()`: `Priorless()`, a `Vector{Distribution}`, or a multivariate `Distribution`.
- `calculate_se::Bool = false`: whether to include this parameter in standard-error calculations.
"""
@with_kw struct SplineParameters{T<:Real, VT<:AbstractVector{T}} <: AbstractParameterBlock
    name::Symbol = :unnamed
    function_name::Symbol
    knots::Vector{T}
    degree::Int
    value::VT
    lower::VT = fill(-Inf, length(value))
    upper::VT = fill(Inf, length(value))
    prior = Priorless()
    calculate_se::Bool = false
end

function SplineParameters(knots::AbstractVector{<:Real}; name::Symbol = :unnamed,
    function_name::Symbol, degree::Integer = 3, prior = Priorless(), calculate_se::Bool = false)
    _check_prior(prior, name)
    degree >= 0 || error("Invalid degree for parameter $(name). Expected degree >= 0; got $(degree).")
    n = length(knots) - Int(degree) - 1
    n > 0 || error("Invalid knots/degree for parameter $(name). Expected length(knots) > degree+1; got length(knots)=$(length(knots)), degree=$(degree).")
    T = eltype(knots) <: AbstractFloat ? eltype(knots) : Float64
    k = T.(knots)
    v = zeros(T, n)
    l = fill(T(-Inf), n)
    u = fill(T(Inf), n)
    _check_nn_prior(prior, name, n)
    return SplineParameters{T, typeof(v)}(name, function_name, collect(k), Int(degree), v, l, u, prior, calculate_se)
end
"""
    SoftTreeParameters(input_dim, depth; name, function_name, n_output, seed, prior, calculate_se) -> SoftTreeParameters

A parameter block for a soft decision tree whose parameters are optimised as fixed effects.

The tree takes a real-valued vector of length `input_dim` and produces a vector of
length `n_output`. Parameters are stored as a flat real vector. Inside model blocks
the tree is called as `function_name(x, θ_slice)`.

# Arguments
- `input_dim::Integer`: number of input features.
- `depth::Integer`: depth of the tree (number of internal split levels).

# Keyword Arguments
- `name::Symbol = :unnamed`: parameter name (injected automatically by `@fixedEffects`).
- `function_name::Symbol`: the name used to call the tree in model blocks.
- `n_output::Integer = 1`: number of output values.
- `seed::Integer = 0`: random seed for initialisation.
- `prior = Priorless()`: `Priorless()`, a `Vector{Distribution}`, or a multivariate `Distribution`.
- `calculate_se::Bool = false`: whether to include this parameter in standard-error calculations.
"""
@with_kw struct SoftTreeParameters{T<:Real, VT<:AbstractVector{T}, R} <: AbstractParameterBlock
    name::Symbol = :unnamed
    function_name::Symbol
    input_dim::Int
    depth::Int
    n_output::Int = 1
    seed::Int = 0
    value::VT
    reconstructor::R
    lower::VT = fill(-Inf, length(value))
    upper::VT = fill(Inf, length(value))
    prior = Priorless()
    calculate_se::Bool = false
end

function SoftTreeParameters(input_dim::Integer, depth::Integer; name::Symbol = :unnamed,
    function_name::Symbol, n_output::Integer = 1, seed::Integer = 0,
    prior = Priorless(), calculate_se::Bool = false)
    input_dim > 0 || error("Invalid input_dim for parameter $(name). Expected input_dim > 0; got $(input_dim).")
    depth > 0 || error("Invalid depth for parameter $(name). Expected depth > 0; got $(depth).")
    n_output > 0 || error("Invalid n_output for parameter $(name). Expected n_output > 0; got $(n_output).")

    tree = SoftTree(Int(input_dim), Int(depth), Int(n_output))
    params = init_params(tree, Xoshiro(seed))
    flat, recon = destructure_params(params)
    T = eltype(flat) <: AbstractFloat ? eltype(flat) : Float64
    v = T.(flat)
    l = fill(T(-Inf), length(v))
    u = fill(T(Inf), length(v))
    _check_nn_prior(prior, name, length(v))
    return SoftTreeParameters{T, typeof(v), typeof(recon)}(name, function_name, Int(input_dim), Int(depth), Int(n_output), Int(seed), v, recon, l, u, prior, calculate_se)
end

"""
    ProbabilityVector(value; name, scale, prior, calculate_se) -> ProbabilityVector

A probability vector parameter block: a vector of `k ≥ 2` non-negative entries
summing to 1. Optimised via the logistic stick-breaking reparameterisation, which
maps the simplex to `k-1` unconstrained reals.

# Arguments
- `value::AbstractVector{<:Real}`: initial probability vector. All entries must be
  non-negative and sum to 1 (within tolerance); if the sum differs by less than
  `atol=1e-6`, the vector is silently normalised.

# Keyword Arguments
- `name::Symbol = :unnamed`: parameter name (injected automatically by `@fixedEffects`).
- `scale::Symbol = :stickbreak`: reparameterisation. Must be in `PROBABILITY_SCALES`.
- `prior = Priorless()`: a `Distributions.Distribution` or `Priorless()`.
- `calculate_se::Bool = false`: whether to include this parameter in standard-error calculations.
"""
@with_kw struct ProbabilityVector{T<:Real, VT<:AbstractVector{T}} <: AbstractParameterBlock
    name::Symbol = :unnamed
    value::VT
    scale::Symbol = :stickbreak
    prior = Priorless()
    calculate_se::Bool = false
end

function ProbabilityVector(value::AbstractVector{<:Real};
    name::Symbol = :unnamed,
    scale::Symbol = :stickbreak,
    prior = Priorless(),
    calculate_se::Bool = false)
    _check_prior(prior, name)
    scale in PROBABILITY_SCALES || error("Invalid scale for parameter $(name). Expected one of $(PROBABILITY_SCALES); got $(scale).")
    length(value) >= 2 || error("ProbabilityVector for parameter $(name) requires at least 2 elements; got $(length(value)).")
    T = eltype(value) <: AbstractFloat ? eltype(value) : Float64
    v = T.(value)
    all(v .>= 0) || error("All entries of ProbabilityVector for parameter $(name) must be non-negative.")
    s = sum(v)
    atol = T(1e-6)
    abs(s - one(T)) <= atol || error("ProbabilityVector for parameter $(name) must sum to 1 (within 1e-6); got sum=$(s).")
    v = v ./ s   # silent normalisation
    return ProbabilityVector{T, typeof(v)}(name, v, scale, prior, calculate_se)
end

"""
    DiscreteTransitionMatrix(value; name, scale, prior, calculate_se) -> DiscreteTransitionMatrix

A square row-stochastic matrix parameter block of size `n×n` (`n ≥ 2`). Each row
is a probability vector and is independently reparameterised via the logistic
stick-breaking transform, yielding `n*(n-1)` unconstrained reals.

# Arguments
- `value::AbstractMatrix{<:Real}`: initial row-stochastic matrix. Each row must be
  non-negative and sum to 1 (within tolerance); rows are silently normalised if needed.

# Keyword Arguments
- `name::Symbol = :unnamed`: parameter name (injected automatically by `@fixedEffects`).
- `scale::Symbol = :stickbreakrows`: reparameterisation. Must be in `TRANSITION_SCALES`.
- `prior = Priorless()`: a `Distributions.Distribution` or `Priorless()`.
- `calculate_se::Bool = false`: whether to include this parameter in standard-error calculations.
"""
@with_kw struct DiscreteTransitionMatrix{T<:Real, MT<:AbstractMatrix{T}} <: AbstractParameterBlock
    name::Symbol = :unnamed
    value::MT
    scale::Symbol = :stickbreakrows
    prior = Priorless()
    calculate_se::Bool = false
end

function DiscreteTransitionMatrix(value::AbstractMatrix{<:Real};
    name::Symbol = :unnamed,
    scale::Symbol = :stickbreakrows,
    prior = Priorless(),
    calculate_se::Bool = false)
    _check_prior(prior, name)
    scale in TRANSITION_SCALES || error("Invalid scale for parameter $(name). Expected one of $(TRANSITION_SCALES); got $(scale).")
    n, m = size(value)
    n == m || error("DiscreteTransitionMatrix for parameter $(name) must be square; got $(n)×$(m).")
    n >= 2 || error("DiscreteTransitionMatrix for parameter $(name) requires at least 2 states; got $(n).")
    T = eltype(value) <: AbstractFloat ? eltype(value) : Float64
    v = T.(value)
    all(v .>= 0) || error("All entries of DiscreteTransitionMatrix for parameter $(name) must be non-negative.")
    row_sums = sum(v; dims=2)
    atol = T(1e-6)
    all(abs.(row_sums .- one(T)) .<= atol) || error("Each row of DiscreteTransitionMatrix for parameter $(name) must sum to 1 (within 1e-6); got row sums=$(vec(row_sums)).")
    v = v ./ row_sums   # silent row-wise normalisation
    return DiscreteTransitionMatrix{T, typeof(v)}(name, v, scale, prior, calculate_se)
end

"""
    ContinuousTransitionMatrix(value; name, scale, prior, calculate_se) -> ContinuousTransitionMatrix

An `n×n` rate matrix (Q-matrix) parameter block for continuous-time Markov chains (`n ≥ 2`).

The Q-matrix has:
- Off-diagonal entries `Q[i,j] ≥ 0` (transition rates from state `i` to state `j`, `i ≠ j`).
- Diagonal entries `Q[i,i] = -∑_{j≠i} Q[i,j]` (rows sum to zero).

The `n*(n-1)` off-diagonal rates are optimised on the log scale (`:lograterows`), mapping
each rate to an unconstrained real via `log`. The diagonal is recomputed from the off-diagonals
and is not an independent free parameter.

# Arguments
- `value::AbstractMatrix{<:Real}`: initial `n×n` Q-matrix. Off-diagonal entries must be
  non-negative. The diagonal is always silently recomputed as `-rowsum` of the off-diagonals,
  so any diagonal values provided in `value` are ignored.

# Keyword Arguments
- `name::Symbol = :unnamed`: parameter name (injected automatically by `@fixedEffects`).
- `scale::Symbol = :lograterows`: reparameterisation. Must be in `RATE_MATRIX_SCALES`.
- `prior = Priorless()`: a `Distributions.Distribution` or `Priorless()`.
- `calculate_se::Bool = false`: whether to include this parameter in standard-error calculations.
"""
@with_kw struct ContinuousTransitionMatrix{T<:Real, MT<:AbstractMatrix{T}} <: AbstractParameterBlock
    name::Symbol = :unnamed
    value::MT
    scale::Symbol = :lograterows
    prior = Priorless()
    calculate_se::Bool = false
end

function ContinuousTransitionMatrix(value::AbstractMatrix{<:Real};
    name::Symbol = :unnamed,
    scale::Symbol = :lograterows,
    prior = Priorless(),
    calculate_se::Bool = false)
    _check_prior(prior, name)
    scale in RATE_MATRIX_SCALES || error("Invalid scale for parameter $(name). Expected one of $(RATE_MATRIX_SCALES); got $(scale).")
    n, m = size(value)
    n == m || error("ContinuousTransitionMatrix for parameter $(name) must be square; got $(n)×$(m).")
    n >= 2 || error("ContinuousTransitionMatrix for parameter $(name) requires at least 2 states; got $(n).")
    T = eltype(value) <: AbstractFloat ? eltype(value) : Float64
    v = T.(value)
    # Validate off-diagonals: must be non-negative.
    for i in 1:n
        for j in 1:n
            i == j && continue
            v[i, j] >= zero(T) || error("ContinuousTransitionMatrix for parameter $(name): off-diagonal entry Q[$(i),$(j)] must be non-negative; got $(v[i,j]).")
        end
    end
    # Always recompute diagonal from off-diagonals (diagonal is a derived quantity).
    for i in 1:n
        v[i, i] = -sum(v[i, j] for j in 1:n if j != i)
    end
    return ContinuousTransitionMatrix{T, typeof(v)}(name, v, scale, prior, calculate_se)
end

function _check_prior(prior, name::Symbol)
    if !(prior isa Priorless) && !(prior isa Distribution)
        error("Invalid prior for parameter $(name). Expected Priorless() or Distributions.Distribution; got $(typeof(prior)).")
    end
    return nothing
end

function _check_nn_prior(prior, name::Symbol, n::Integer)
    prior isa Priorless && return nothing
    if prior isa AbstractVector{<:Distribution}
        length(prior) == n || error("Invalid prior for parameter $(name). Expected length $(n); got $(length(prior)).")
        return nothing
    end
    if prior isa Distribution
        if hasmethod(length, Tuple{typeof(prior)})
            d = length(prior)
            d == n || error("Invalid prior for parameter $(name). Expected distribution length $(n); got $(d).")
            return nothing
        end
        error("Invalid prior for parameter $(name). Expected a vector of Distributions with length $(n) or a Distribution with length $(n); got $(typeof(prior)).")
    end
    error("Invalid prior for parameter $(name). Expected Priorless() or a Distribution; got $(typeof(prior)).")
end
