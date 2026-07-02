using LinearAlgebra
using Random
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
        input_dim > 0 ||
            error("Invalid input_dim. Expected input_dim > 0; got $(input_dim).")
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
struct SoftTreeParams{WM <: AbstractMatrix, BV <: AbstractVector, LM <: AbstractMatrix}
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

Initialize parameters for a [`SoftTree`](@ref).

The no-`rng` overload fills all parameters with the given constant values.
The `rng` overload draws parameters from zero-mean Normal distributions with the
specified standard deviations.

# Arguments
- `tree::SoftTree`: the soft tree architecture.
- `rng::AbstractRNG`: random-number generator (second overload only).

# Keyword Arguments (constant initialization)
- `init_weight::Real = 0.0`: node weight initial value.
- `init_bias::Real = 0.0`: node bias initial value.
- `init_leaf::Real = 0.0`: leaf value initial value.

# Keyword Arguments (random initialization)
- `init_weight_std::Real = 0.1`: standard deviation for node weights.
- `init_bias_std::Real = 0.0`: standard deviation for node biases.
- `init_leaf_std::Real = 0.1`: standard deviation for leaf values.
"""
function init_params(tree::SoftTree; init_weight::Real = 0.0,
        init_bias::Real = 0.0, init_leaf::Real = 0.0)
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

    size(node_weights, 1) == n_internal ||
        error("Invalid node_weights rows. Expected $(n_internal); got $(size(node_weights, 1)).")
    size(node_weights, 2) == tree.input_dim ||
        error("Invalid node_weights cols. Expected $(tree.input_dim); got $(size(node_weights, 2)).")
    length(node_biases) == n_internal ||
        error("Invalid node_biases length. Expected $(n_internal); got $(length(node_biases)).")
    size(leaf_values, 1) == tree.n_output ||
        error("Invalid leaf_values rows. Expected $(tree.n_output); got $(size(leaf_values, 1)).")
    size(leaf_values, 2) == n_leaves ||
        error("Invalid leaf_values cols. Expected $(n_leaves); got $(size(leaf_values, 2)).")

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

"""
    softtree_params_from_flat(θ::AbstractVector, tree::SoftTree) -> SoftTreeParams

Rebuild a [`SoftTreeParams`](@ref) from a flat parameter vector laid out as
`[vec(node_weights); node_biases; vec(leaf_values)]` — the order produced by
`Optimisers.destructure`. Unlike the `Restructure` closure, this is a plain
positional reconstruction without `Functors.fmap`/`IdDict` machinery, keeping it
type-stable and Enzyme-compatible (Enzyme forward mode has no derivative rule for
the `jl_eqtable_get` IdDict lookups inside `fmap`).
"""
function softtree_params_from_flat(θ::AbstractVector, tree::SoftTree)
    n_internal = 2^tree.depth - 1
    n_leaves = 2^tree.depth
    nw_len = n_internal * tree.input_dim
    # Views, not copying getindex: this runs on every model-fun evaluation and the
    # downstream eval only ever indexes elementwise (no BLAS), so reshaped views are
    # free and AD-transparent.
    node_weights = reshape(view(θ, 1:nw_len), n_internal, tree.input_dim)
    node_biases = view(θ, nw_len .+ (1:n_internal))
    leaf_values = reshape(
        view(θ, (nw_len + n_internal) .+ (1:(tree.n_output * n_leaves))),
        tree.n_output, n_leaves)
    return SoftTreeParams(node_weights, node_biases, leaf_values)
end

@inline _sigmoid(x) = Base.inv(one(x) + exp(-x))

# Scalar-accumulation row/column products. Deliberately BLAS-free: Enzyme forward
# mode has no rule for BLAS calls under runtime activity ("Runtime Activity not yet
# implemented for Forward-Mode BLAS calls" for `dot`/`gemv`), and at SoftTree sizes
# plain loops are at least as fast as BLAS dispatch anyway.
@inline function _st_rowdot(W::AbstractMatrix, i::Integer, x::AbstractVector)
    s = zero(promote_type(eltype(W), eltype(x)))
    for j in eachindex(x)
        s += W[i, j] * x[j]
    end
    return s
end

@inline function _st_leafdot(V::AbstractMatrix, o::Integer, p::AbstractVector)
    s = zero(promote_type(eltype(V), eltype(p)))
    for j in eachindex(p)
        s += V[o, j] * p[j]
    end
    return s
end

function (tree::SoftTree)(x::AbstractVector{<:Real}, params::SoftTreeParams)
    # BLAS-free, single-buffer implementation (ForwardDiff/Enzyme-compatible: plain
    # index assignment into a locally allocated, promoted-eltype vector).
    #
    # Leaf probabilities are produced in LEVEL-CONCATENATION order — at each level
    # the new vector is [old .* p ; old .* (1 .- p)] — exactly the order of the
    # historical `vcat`-doubling implementation (bit-identical results). This order
    # is the semantic anchor: fitted `SoftTreeParameters` pair leaf columns with it.
    # Note it differs from binary-heap order by a bit-reversal permutation of leaves.
    length(x) == tree.input_dim ||
        error("Invalid input length. Expected $(tree.input_dim); got $(length(x)).")
    n_leaves = 2^tree.depth
    T = promote_type(
        eltype(params.node_weights), eltype(params.node_biases), eltype(x))
    probs = Vector{T}(undef, n_leaves)
    probs[1] = one(T)
    @inbounds for level in 0:(tree.depth - 1)
        n_prev = 2^level
        start_idx = 2^level
        for k in 1:n_prev
            old = probs[k]
            p = _sigmoid(_st_rowdot(params.node_weights, start_idx + k - 1, x) +
                         params.node_biases[start_idx + k - 1])
            probs[n_prev + k] = old * (one(T) - p)
            probs[k] = old * p
        end
    end
    return [_st_leafdot(params.leaf_values, o, probs)
            for o in 1:size(params.leaf_values, 1)]
end

# NOTE on leaf ordering: the historical `Val(:fast)`/`Val(:inplace)` variants walked
# the tree in binary-heap order (probs[2i], probs[2i+1]), which orders leaves by a
# bit-reversal permutation of the default's level-concatenation order. For asymmetric
# parameters the two therefore returned DIFFERENT values for the same `params` (the
# old equality tests only passed at all-zero init). Both variants now share the
# default's ordering so all three entry points agree for any parameters.
function (tree::SoftTree)(
        x::AbstractVector{<:Real}, params::SoftTreeParams, ::Val{:inplace})
    # Kept for API compatibility: the body had become identical to the default
    # method (its `zeros` vs `undef` buffer init is unobservable — the level loop
    # writes every leaf slot), so it simply delegates, like `Val(:fast)`.
    return tree(x, params)
end

function (tree::SoftTree)(x::AbstractVector{<:Real}, params::SoftTreeParams, ::Val{:fast})
    # Kept for API compatibility: the default method is now the single-buffer,
    # BLAS-free implementation, so `Val(:fast)` simply delegates to it.
    return tree(x, params)
end
