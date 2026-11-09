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

# Allocating reference forward pass (`W * y .+ b` per layer). Used for callables whose
# output type differs from the promoted element type, where the scalar loops below would
# convert results.
function _ffnn_forward_generic(net::FFNN, x::AbstractVector, θ::AbstractVector, ::Type{T}) where {T}
    s = net.sizes
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

# Pre-activation of unit `j` of a dense layer: Σ_k W[j, k] y[k] + b[j], with W stored
# column-major at `θ[off+1 : off+n_out*n_in]` and b right after it. The sum starts at b
# and accumulates with `muladd` (fused multiply-add where available), so results differ
# from `W * y .+ b` by rounding only (about 1e-16 absolute). `y` is read unpromoted: a
# plain-number input times a Dual weight skips the zero partials that promotion would
# multiply through.
@inline function _ffnn_preact(::Type{T}, θ, y, off::Int, n_in::Int, n_out::Int, j::Int) where {T}
    acc = zero(T) + @inbounds(θ[off + n_in * n_out + j])
    @inbounds for k in 1:n_in
        acc = muladd(θ[off + (k - 1) * n_out + j], y[k], acc)
    end
    return acc
end

# Hidden layer `y -> act.(W y .+ b)` into a fresh vector (no buffer shared across calls).
function _ffnn_hidden(act, ::Type{T}, θ, y, off::Int, n_in::Int, n_out::Int) where {T}
    h = Vector{T}(undef, n_out)
    @inbounds for j in 1:n_out
        h[j] = act(_ffnn_preact(T, θ, y, off, n_in, n_out, j))
    end
    return h
end

# Last hidden layer fused with the output layer: each hidden unit is computed once and
# immediately accumulated into the outputs, so the last hidden vector is never stored.
# Each output starts at its bias and accumulates over hidden units with `muladd`.
function _ffnn_last(net::FFNN, ::Type{T}, θ, y, off::Int, n_in::Int, n_hid::Int, n_out::Int) where {T}
    off2 = off + n_hid * (n_in + 1)
    act = net.activation
    if n_out == 1
        # Scalar output (the common case in model formulas): the sum stays in a register.
        acc = zero(T) + @inbounds(θ[off2 + n_hid + 1])
        @inbounds for j in 1:n_hid
            acc = muladd(θ[off2 + j], act(_ffnn_preact(T, θ, y, off, n_in, n_hid, j)), acc)
        end
        return _ffnn_apply_out!(net.output_activation, T[acc])
    end
    z = Vector{T}(undef, n_out)
    @inbounds for o in 1:n_out
        z[o] = θ[off2 + n_hid * n_out + o]
    end
    @inbounds for j in 1:n_hid
        h = act(_ffnn_preact(T, θ, y, off, n_in, n_hid, j))
        for o in 1:n_out
            z[o] = muladd(θ[off2 + (j - 1) * n_out + o], h, z[o])
        end
    end
    return _ffnn_apply_out!(net.output_activation, z)
end

# Output layer without a hidden layer in front (sizes `(n_in, n_out)`).
function _ffnn_linear_out(net::FFNN, ::Type{T}, θ, y, n_in::Int, n_out::Int) where {T}
    z = Vector{T}(undef, n_out)
    @inbounds for o in 1:n_out
        z[o] = _ffnn_preact(T, θ, y, 0, n_in, n_out, o)
    end
    return _ffnn_apply_out!(net.output_activation, z)
end

_ffnn_apply_out!(::_FFNNSoftmax, z) = softmax(z)
function _ffnn_apply_out!(f, z)
    @inbounds for o in eachindex(z)
        z[o] = f(z[o])
    end
    return z
end

# Forward pass. Scalar loops over the flat parameter vector: no promoted input copy, no
# `W * y` / broadcast temporaries; the only allocations are one vector per hidden layer
# beyond the last and the returned output. Nothing is shared across nested (or concurrent)
# calls.
function (net::FFNN)(x::AbstractVector, θ::AbstractVector)
    s = net.sizes
    length(x) == s[1] ||
        error("FFNN input has length $(length(x)); the network expects $(s[1]).")
    length(θ) == _ffnn_nparams(net) ||
        error("FFNN parameter vector has length $(length(θ)); the network expects $(_ffnn_nparams(net)).")
    T = promote_type(eltype(θ), eltype(x))
    _ffnn_scalar_ok(net, T) || return _ffnn_forward_generic(net, x, θ, T)
    _ffnn_use_blas(net, T) && return _ffnn_forward_generic(net, x, θ, T)
    n_layers = length(s) - 1
    n_layers == 1 && return _ffnn_linear_out(net, T, θ, x, s[1], s[2])
    y = x
    offset = 0
    for i in 1:(n_layers - 2)
        y = _ffnn_hidden(net.activation, T, θ, y, offset, s[i], s[i + 1])
        offset += s[i + 1] * (s[i] + 1)
    end
    return _ffnn_last(net, T, θ, y, offset, s[n_layers - 1], s[n_layers], s[n_layers + 1])
end

# Plain floating-point evaluations of wide networks go through BLAS: from a hidden width
# of 16 a BLAS matrix-vector product plus a separate (vectorizable) activation loop is as
# fast as or faster than the fused scalar loop (one thread, 2026-11 measurements: width 32
# BLAS 1.9x faster, width 8 scalar 1.6x faster). Dual numbers never reach BLAS, and for
# them the scalar loop wins at every width measured (up to 64), so they always take it.
const _FFNN_BLAS_MIN_WIDTH = 16
function _ffnn_use_blas(net::FFNN, ::Type{T}) where {T}
    T <: LinearAlgebra.BlasFloat || return false
    s = net.sizes
    for i in 2:(length(s) - 1)
        s[i] >= _FFNN_BLAS_MIN_WIDTH && return true
    end
    return false
end

# The scalar loops store activations in `Vector{T}`; callables that change the element
# type (e.g. user functions returning Float32) take the broadcast path instead.
_ffnn_scalar_ok(net::FFNN, ::Type{T}) where {T} =
    Base.promote_op(net.activation, T) === T && _ffnn_out_ok(net.output_activation, T)
_ffnn_out_ok(::_FFNNSoftmax, ::Type) = true
_ffnn_out_ok(f, ::Type{T}) where {T} = Base.promote_op(f, T) === T

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
