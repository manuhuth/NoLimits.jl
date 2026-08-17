using Random
using StatsFuns: logistic, log1pexp, softmax

export FFNNParameters

# Plain feed-forward network living in core NoLimits: it plugs into the existing
# `NNParameters` plumbing through the same `_nn_model_fun` dispatch slot the Lux and
# SimpleChains extensions use, so a multilayer perceptron needs no weak dependency.

_ffnn_relu(x) = max(zero(x), x)
_ffnn_swish(x) = x * logistic(x)
# tanh approximation of GELU: keeps SpecialFunctions (erf) out of the dependency set.
const _GELU_C = sqrt(2 / pi)
function _ffnn_gelu(x)
    return x / 2 * (one(x) + tanh(oftype(x, _GELU_C) * (x + oftype(x, 0.044715) * x^3)))
end

# Marker for the one output transform that maps a vector to a vector rather than acting
# element-wise; dispatch on it in `_ffnn_apply_out` instead of branching per call.
struct _FFNNSoftmax end

_ffnn_apply_out(::_FFNNSoftmax, z) = softmax(z)
_ffnn_apply_out(f, z) = f.(z)

const _FFNN_ACTIVATIONS = (
    tanh = tanh, relu = _ffnn_relu, sigmoid = logistic, logistic = logistic,
    softplus = log1pexp, identity = identity, gelu = _ffnn_gelu, swish = _ffnn_swish,
)

# Output-only: `softmax` is a vector transform, so it does not belong between hidden
# layers. `logit` is deliberately absent: it needs inputs in (0, 1), while the output
# layer emits an unbounded affine value, so it would just throw a DomainError.
const _FFNN_OUTPUT_ACTIVATIONS = (softmax = _FFNNSoftmax(),)

const _FFNN_ACTIVATION_HINT = "For a (0, 1)-bounded output use :logistic, for probabilities use :softmax."

function _ffnn_activation(a; output::Bool)
    a isa Union{Symbol, AbstractString} || return a   # any callable passes through
    s = _as_symbol(a)
    haskey(_FFNN_ACTIVATIONS, s) && return getfield(_FFNN_ACTIVATIONS, s)
    if haskey(_FFNN_OUTPUT_ACTIVATIONS, s)
        output || error("Invalid FFNN activation :$(s). It is only available as an `output_activation` (:softmax is a vector transform). $(_FFNN_ACTIVATION_HINT)")
        return getfield(_FFNN_OUTPUT_ACTIVATIONS, s)
    end
    valid = join(string.(keys(_FFNN_ACTIVATIONS)), ", ")
    out_only = join(string.(keys(_FFNN_OUTPUT_ACTIVATIONS)), ", ")
    return error("Unknown FFNN activation :$(s). Valid names are $(valid) (hidden or output) and $(out_only) (output only); alternatively pass any callable. $(_FFNN_ACTIVATION_HINT)")
end

"""
    FFNN(sizes, activation = :tanh, output_activation = :identity)

A plain feed-forward network (multilayer perceptron) described by its layer sizes and
activations. `sizes` is `(n_input, hidden..., n_output)` with at least two entries.
Activations are `Symbol`s or `String`s from the registry (see [`FFNNParameters`](@ref))
or any callable.

The network is callable as `net(x, θ)` with `θ` the flat parameter vector holding, per
layer, `vec(W)` (an `n_out × n_in` matrix) followed by the length-`n_out` bias. Use
[`FFNNParameters`](@ref) to declare one as a fixed effect.
"""
struct FFNN{A, O}
    sizes::Vector{Int}
    activation::A
    output_activation::O

    function FFNN(sizes, activation = :tanh, output_activation = :identity)
        all(n -> n isa Integer, sizes) ||
            error("Invalid FFNN sizes $(sizes). Expected integers; got element types $(unique(typeof.(collect(sizes)))).")
        s = collect(Int, sizes)
        length(s) >= 2 ||
            error("Invalid FFNN sizes $(sizes). Expected at least 2 entries (input and output dimension); got $(length(s)).")
        all(>(0), s) ||
            error("Invalid FFNN sizes $(sizes). All layer sizes must be positive.")
        af = _ffnn_activation(activation; output = false)
        of = _ffnn_activation(output_activation; output = true)
        return new{typeof(af), typeof(of)}(s, af, of)
    end
end

function _ffnn_nparams(net::FFNN)
    s = net.sizes
    return sum(s[i + 1] * (s[i] + 1) for i in 1:(length(s) - 1))
end

# `convert` (not `T(x)`) so nested `Dual` inputs promote instead of erroring, and always
# to a concrete `Vector{T}` so the layer loop below keeps one element type throughout.
_ffnn_promote(::Type{T}, x::AbstractVector) where {T} = convert(Vector{T}, x)

# Non-mutating forward pass: every intermediate is a fresh array, so nothing is shared
# across nested (or concurrent) calls.
function (net::FFNN)(x::AbstractVector, θ::AbstractVector)
    s = net.sizes
    length(x) == s[1] ||
        error("FFNN input has length $(length(x)); the network expects $(s[1]).")
    length(θ) == _ffnn_nparams(net) ||
        error("FFNN parameter vector has length $(length(θ)); the network expects $(_ffnn_nparams(net)).")
    T = promote_type(eltype(θ), eltype(x))
    y = _ffnn_promote(T, x)
    offset = 0
    n_layers = length(s) - 1
    for i in 1:n_layers
        n_in, n_out = s[i], s[i + 1]
        W = reshape(view(θ, (offset + 1):(offset + n_in * n_out)), n_out, n_in)
        offset += n_in * n_out
        b = view(θ, (offset + 1):(offset + n_out))
        offset += n_out
        z = W * y .+ b
        y = i == n_layers ? _ffnn_apply_out(net.output_activation, z) : net.activation.(z)
    end
    return y
end

# Glorot-uniform weights, zero biases, in the flat layout the forward pass reads.
function _ffnn_init_params(net::FFNN, ::Type{T}, rng::Random.AbstractRNG) where {T}
    v = Vector{T}(undef, _ffnn_nparams(net))
    s = net.sizes
    offset = 0
    for i in 1:(length(s) - 1)
        n_in, n_out = s[i], s[i + 1]
        bound = sqrt(T(6) / T(n_in + n_out))
        for j in 1:(n_in * n_out)
            v[offset + j] = (2 * rand(rng, T) - 1) * bound
        end
        offset += n_in * n_out
        v[(offset + 1):(offset + n_out)] .= zero(T)
        offset += n_out
    end
    return v
end

function NNParameters(
        net::FFNN; name::Symbol = :unnamed, function_name::Symbol, seed::Integer = 0,
        prior = Priorless(), calculate_se::Bool = false
    )
    T = Float64
    v = _ffnn_init_params(net, T, Xoshiro(seed))
    l = fill(T(-Inf), length(v))
    u = fill(T(Inf), length(v))
    _check_nn_prior(prior, name, length(v))
    # Parameters are already flat, so `reconstructor` is unused (as for SimpleChains).
    return NNParameters{T, typeof(v), typeof(net), typeof(identity)}(
        name, function_name, net, v, identity, l, u, prior, calculate_se
    )
end

# The FFNN is itself the forward pass, so it needs no wrapper closure; promotion happens
# inside the call from both argument eltypes.
_nn_model_fun(net::FFNN, ::NNParameters, ::Type{T}) where {T} = net

"""
    FFNNParameters(sizes; activation, output_activation, name, function_name, seed, prior, calculate_se)

A parameter block for a plain feed-forward neural network (multilayer perceptron) whose
weights are estimated as fixed effects. Unlike [`NNParameters`](@ref) it needs no
optional dependency: the architecture is a specification (layer sizes plus activations)
and the forward pass lives in NoLimits itself.

```julia
ζ = FFNNParameters((1, 8, 8, 1); activation = :tanh, function_name = :NN1)
# used in any model block as: NN1([t], ζ)[1]
```

Returns an [`NNParameters`](@ref) block, so priors, `calculate_se` masks, random effects
on the weights, and every estimator work exactly as for a Lux or SimpleChains network.
Use [`NNParameters`](@ref) for architectures beyond a plain MLP.

# Arguments
- `sizes`: layer sizes as a `Tuple` or `AbstractVector` of positive integers,
  `(n_input, hidden..., n_output)`, with at least two entries.

# Keyword Arguments
- `activation = :tanh`: hidden-layer activation. A `Symbol`, a `String`, or any callable.
  Registry: `:tanh`, `:relu`, `:sigmoid` (alias `:logistic`), `:softplus`, `:identity`,
  `:gelu` (tanh approximation), `:swish`.
- `output_activation = :identity`: transform of the output layer. Same registry, plus the
  output-only `:softmax` (numerically stable, sums to 1).
- `name::Symbol = :unnamed`: parameter name (injected automatically by `@fixedEffects`).
- `function_name::Symbol`: name the network is called under inside model blocks.
- `seed::Integer = 0`: seed for the Glorot-uniform weight initialization (biases start at
  zero); the initial values are deterministic given `seed`.
- `prior = Priorless()`: `Priorless()`, a `Vector{Distribution}` of length equal to the
  number of weights, or a multivariate `Distribution` of that dimension.
- `calculate_se::Bool = false`: whether standard errors are computed for the weights.

# Examples
```julia
@fixedEffects begin
    σ = RealNumber(0.5, scale = :log)
    ζ = FFNNParameters((2, 8, 1); activation = :relu, function_name = :NN1)
    # Softmax output for a 3-category probability vector:
    ω = FFNNParameters((1, 4, 3); output_activation = :softmax, function_name = :NN2)
end
```

See also [`NNParameters`](@ref), [`SoftTreeParameters`](@ref), [`SplineParameters`](@ref).
"""
function FFNNParameters(
        sizes::Union{Tuple, AbstractVector};
        activation = :tanh, output_activation = :identity, kwargs...
    )
    return NNParameters(FFNN(sizes, activation, output_activation); kwargs...)
end
