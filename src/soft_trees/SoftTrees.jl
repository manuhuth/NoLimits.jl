using LinearAlgebra
using Random
using Zygote
using Functors
using Optimisers

export SoftTree
export SoftTreeParams
export init_params
export destructure_params

"""
    SoftTree(input_dim::Int, depth::Int, n_output::Int)

A differentiable soft decision tree with `input_dim` input features, `depth` levels,
and `n_output` outputs.

The tree has `2^depth - 1` internal nodes and `2^depth` leaves. Each internal node
applies a soft sigmoid split to route inputs; each leaf stores a learnable output value.
The forward pass returns the weighted sum of leaf values, differentiable with respect
to both inputs and parameters.

# Arguments
- `input_dim::Int`: number of input features (must be > 0).
- `depth::Int`: number of tree levels (must be > 0).
- `n_output::Int`: number of output values per evaluation (must be > 0).
"""
struct SoftTree
    input_dim::Int
    depth::Int
    n_output::Int
    function SoftTree(input_dim::Int, depth::Int, n_output::Int)
        input_dim > 0 || error("Invalid input_dim. Expected input_dim > 0; got $(input_dim).")
        depth > 0 || error("Invalid depth. Expected depth > 0; got $(depth).")
        n_output > 0 || error("Invalid n_output. Expected n_output > 0; got $(n_output).")
        return new(input_dim, depth, n_output)
    end
end

"""
    SoftTreeParams{WM, BV, LM}

Parameters for a [`SoftTree`](@ref). Created via [`init_params`](@ref).

Fields:
- `node_weights::WM`: weight matrix of shape `(n_internal, input_dim)`.
- `node_biases::BV`: bias vector of length `n_internal`.
- `leaf_values::LM`: leaf value matrix of shape `(n_output, n_leaves)`.
"""
struct SoftTreeParams{WM<:AbstractMatrix, BV<:AbstractVector, LM<:AbstractMatrix}
    node_weights::WM
    node_biases::BV
    leaf_values::LM
end

@functor SoftTreeParams

function SoftTree(input_dim::Integer, depth::Integer, n_output::Integer)
    return SoftTree(Int(input_dim), Int(depth), Int(n_output))
end

"""
    init_params(tree::SoftTree; init_weight=0.0, init_bias=0.0, init_leaf=0.0)
    -> SoftTreeParams

    init_params(tree::SoftTree, rng::AbstractRNG; init_weight_std=0.1,
                init_bias_std=0.0, init_leaf_std=0.1) -> SoftTreeParams

Initialise parameters for a [`SoftTree`](@ref).

The no-`rng` overload fills all parameters with the given constant values.
The `rng` overload draws parameters from zero-mean Normal distributions with the
specified standard deviations.

# Arguments
- `tree::SoftTree`: the soft tree architecture.
- `rng::AbstractRNG`: random-number generator (second overload only).

# Keyword Arguments (constant initialisation)
- `init_weight::Real = 0.0`: node weight initial value.
- `init_bias::Real = 0.0`: node bias initial value.
- `init_leaf::Real = 0.0`: leaf value initial value.

# Keyword Arguments (random initialisation)
- `init_weight_std::Real = 0.1`: standard deviation for node weights.
- `init_bias_std::Real = 0.0`: standard deviation for node biases.
- `init_leaf_std::Real = 0.1`: standard deviation for leaf values.
"""
function init_params(tree::SoftTree; init_weight::Real = 0.0, init_bias::Real = 0.0, init_leaf::Real = 0.0)
    n_internal = 2^tree.depth - 1
    n_leaves = 2^tree.depth

    W = fill(float(init_weight), n_internal, tree.input_dim)
    b = fill(float(init_bias), n_internal)
    V = fill(float(init_leaf), tree.n_output, n_leaves)
    return SoftTreeParams(W, b, V)
end

function init_params(tree::SoftTree, rng::AbstractRNG;
    init_weight_std::Real = 0.1, init_bias_std::Real = 0.0, init_leaf_std::Real = 0.1)
    n_internal = 2^tree.depth - 1
    n_leaves = 2^tree.depth

    W = randn(rng, n_internal, tree.input_dim) .* float(init_weight_std)
    b = randn(rng, n_internal) .* float(init_bias_std)
    V = randn(rng, tree.n_output, n_leaves) .* float(init_leaf_std)
    return SoftTreeParams(W, b, V)
end

function SoftTreeParams(tree::SoftTree, node_weights::AbstractMatrix{<:Number},
    node_biases::AbstractVector{<:Number}, leaf_values::AbstractMatrix{<:Number})
    n_internal = 2^tree.depth - 1
    n_leaves = 2^tree.depth

    size(node_weights, 1) == n_internal || error("Invalid node_weights rows. Expected $(n_internal); got $(size(node_weights, 1)).")
    size(node_weights, 2) == tree.input_dim || error("Invalid node_weights cols. Expected $(tree.input_dim); got $(size(node_weights, 2)).")
    length(node_biases) == n_internal || error("Invalid node_biases length. Expected $(n_internal); got $(length(node_biases)).")
    size(leaf_values, 1) == tree.n_output || error("Invalid leaf_values rows. Expected $(tree.n_output); got $(size(leaf_values, 1)).")
    size(leaf_values, 2) == n_leaves || error("Invalid leaf_values cols. Expected $(n_leaves); got $(size(leaf_values, 2)).")

    T = promote_type(eltype(node_weights), eltype(node_biases), eltype(leaf_values))
    W = T.(node_weights)
    b = T.(node_biases)
    V = T.(leaf_values)
    return SoftTreeParams(W, b, V)
end

"""
    destructure_params(params::SoftTreeParams) -> (Vector, Restructure)

Flatten a [`SoftTreeParams`](@ref) to a parameter vector and return the vector
together with a reconstruction function (using `Optimisers.destructure`).

The reconstruction function can be called with a new flat vector to reconstruct
a `SoftTreeParams` with the same structure.
"""
function destructure_params(params::SoftTreeParams)
    return Optimisers.destructure(params)
end

@inline _sigmoid(x) = Base.inv(one(x) + exp(-x))

function (tree::SoftTree)(x::AbstractVector{<:Real}, params::SoftTreeParams)
    # Pure implementation for AD friendliness (Zygote-compatible).
    length(x) == tree.input_dim || error("Invalid input length. Expected $(tree.input_dim); got $(length(x)).")
    T = promote_type(eltype(params.node_weights), eltype(x))
    probs = [one(T)]
    for level in 0:(tree.depth - 1)
        start_idx = 2^level
        end_idx = 2^(level + 1) - 1
        pvec = [_sigmoid(dot(view(params.node_weights, i, :), x) + params.node_biases[i]) for i in start_idx:end_idx]
        probs = vcat(probs .* pvec, probs .* (one(T) .- pvec))
    end
    return params.leaf_values * probs
end

function (tree::SoftTree)(x::AbstractVector{<:Real}, params::SoftTreeParams, ::Val{:inplace})
    length(x) == tree.input_dim || error("Invalid input length. Expected $(tree.input_dim); got $(length(x)).")
    n_internal = 2^tree.depth - 1
    n_leaves = 2^tree.depth
    T = promote_type(eltype(params.node_weights), eltype(x))

    probs = Zygote.Buffer(zeros(T, n_internal + n_leaves))
    probs[1] = one(T)
    for i in 1:n_internal
        w = view(params.node_weights, i, :)
        b = params.node_biases[i]
        p = _sigmoid(dot(w, x) + b)
        left = 2i
        right = 2i + 1
        probs[left] = probs[i] * p
        probs[right] = probs[i] * (one(T) - p)
    end

    y = Zygote.Buffer(zeros(T, tree.n_output))
    for j in 1:n_leaves
        prob = probs[n_internal + j]
        y .= y .+ prob .* view(params.leaf_values, :, j)
    end

    return copy(y)
end

function (tree::SoftTree)(x::AbstractVector{<:Real}, params::SoftTreeParams, ::Val{:fast})
    length(x) == tree.input_dim || error("Invalid input length. Expected $(tree.input_dim); got $(length(x)).")
    n_internal = 2^tree.depth - 1
    n_leaves = 2^tree.depth
    T = promote_type(eltype(params.node_weights), eltype(x))

    probs = zeros(T, n_internal + n_leaves)
    probs[1] = one(T)
    for i in 1:n_internal
        w = view(params.node_weights, i, :)
        b = params.node_biases[i]
        p = _sigmoid(dot(w, x) + b)
        left = 2i
        right = 2i + 1
        probs[left] = probs[i] * p
        probs[right] = probs[i] * (one(T) - p)
    end

    y = zeros(T, tree.n_output)
    for j in 1:n_leaves
        prob = probs[n_internal + j]
        y .+= prob .* view(params.leaf_values, :, j)
    end

    return y
end
