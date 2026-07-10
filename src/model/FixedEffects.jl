#Functions from NoLimits dont have to be imported

using ComponentArrays
using Distributions
using LinearAlgebra
using Random
using Functors

export @fixedEffects
export get_collect_names
export FixedEffects
export logprior
export get_transforms
export get_names
export get_flat_names
export get_θ0_untransformed
export get_θ0_transformed
export get_bounds_untransformed
export get_bounds_transformed
export get_transform
export get_inverse_transform
export get_priors
export get_se_mask
export get_se_names
export get_model_funs
export get_params

struct FixedEffectsMeta
    names::Vector{Symbol}
    flat_names::Vector{Symbol}
end

struct FixedEffectsValues
    θ0_untransformed::ComponentArray
    θ0_transformed::ComponentArray
end

struct FixedEffectsBounds
    untransformed::Tuple{ComponentArray, ComponentArray}
    transformed::Tuple{ComponentArray, ComponentArray}
end

struct FixedEffectsTransforms{F <: ForwardTransform, I <: InverseTransform}
    forward::F
    inverse::I
end

struct FixedEffectsExtras{P <: NamedTuple, MF <: NamedTuple, PA <: NamedTuple}
    priors::P
    se_mask_transformed::Vector{Bool}
    se_names_transformed::Vector{Symbol}
    model_funs::MF
    params::PA
end

"""
    FixedEffects

Compiled representation of a `@fixedEffects` block. Contains initial parameter values,
bounds, forward/inverse transforms, priors, SE masks, and model functions (NNs, splines,
soft trees, NPFs).

Use accessor functions rather than accessing fields directly.
"""
struct FixedEffects{E <: FixedEffectsExtras, TR <: FixedEffectsTransforms}
    meta::FixedEffectsMeta
    values::FixedEffectsValues
    bounds::FixedEffectsBounds
    transforms::TR
    extras::E
end

"""
    get_transforms(fe::FixedEffects) -> FixedEffectsTransforms

Return the `ForwardTransform` / `InverseTransform` pair for this parameter block.
"""
get_transforms(fe::FixedEffects) = fe.transforms

"""
    get_names(fe::FixedEffects) -> Vector{Symbol}

Return the top-level parameter names as declared in `@fixedEffects`.
"""
get_names(fe::FixedEffects) = fe.meta.names

"""
    get_flat_names(fe::FixedEffects) -> Vector{Symbol}

Return the flattened parameter names on the transformed scale (e.g. matrix entries
and vector elements are expanded element-wise).
"""
get_flat_names(fe::FixedEffects) = fe.meta.flat_names

"""
    get_θ0_untransformed(fe::FixedEffects) -> ComponentArray

Return the initial parameter vector on the natural (untransformed) scale.
"""
get_θ0_untransformed(fe::FixedEffects) = fe.values.θ0_untransformed

"""
    get_θ0_transformed(fe::FixedEffects) -> ComponentArray

Return the initial parameter vector on the transformed (optimization) scale.
"""
get_θ0_transformed(fe::FixedEffects) = fe.values.θ0_transformed

"""
    get_bounds_untransformed(fe::FixedEffects) -> Tuple{ComponentArray, ComponentArray}

Return `(lower, upper)` bounds on the natural scale.
"""
get_bounds_untransformed(fe::FixedEffects) = fe.bounds.untransformed

"""
    get_bounds_transformed(fe::FixedEffects) -> Tuple{ComponentArray, ComponentArray}

Return `(lower, upper)` bounds on the transformed scale.
"""
get_bounds_transformed(fe::FixedEffects) = fe.bounds.transformed

"""
    get_transform(fe::FixedEffects) -> ForwardTransform

Return the callable `ForwardTransform` that maps a natural-scale `ComponentArray`
to the optimization scale.
"""
get_transform(fe::FixedEffects) = fe.transforms.forward

"""
    get_inverse_transform(fe::FixedEffects) -> InverseTransform

Return the callable `InverseTransform` that maps an optimization-scale `ComponentArray`
back to the natural scale.
"""
get_inverse_transform(fe::FixedEffects) = fe.transforms.inverse

"""
    get_priors(fe::FixedEffects) -> NamedTuple

Return a `NamedTuple` mapping each parameter name to its prior distribution
(or `Priorless()` if none was specified).
"""
get_priors(fe::FixedEffects) = fe.extras.priors

"""
    get_se_mask(fe::FixedEffects) -> Vector{Bool}

Return a Boolean mask over the flattened transformed parameter vector indicating
which elements should be included in standard-error calculations.
"""
get_se_mask(fe::FixedEffects) = fe.extras.se_mask_transformed

"""
    get_se_names(fe::FixedEffects) -> Vector{Symbol}

Return the flat parameter names for which `calculate_se = true`.
"""
get_se_names(fe::FixedEffects) = fe.extras.se_names_transformed

"""
    get_model_funs(fe::FixedEffects) -> NamedTuple

Return a `NamedTuple` of callable model functions derived from `NNParameters`,
`SoftTreeParameters`, `SplineParameters`, and `NPFParameter` blocks.
Each function has the signature matching its block type's `function_name`.
"""
get_model_funs(fe::FixedEffects) = fe.extras.model_funs

"""
    get_params(fe::FixedEffects) -> NamedTuple

Return the raw parameter block structs as a `NamedTuple` keyed by parameter name.
"""
get_params(fe::FixedEffects) = fe.extras.params

"""
    get_collect_names(fe::FixedEffects) -> Vector{Symbol}

Return the names of fixed-effect parameters whose untransformed values must be
materialised (via `collect`) before use in formula functions.

`ComponentArray` returns `SubArray` views for multi-element parameters. For
`ProbabilityVector` and `DiscreteTransitionMatrix` parameters the probability
values must be a concrete `Vector` or `Matrix` (not a view) so that
`Distributions.Categorical(p)` and similar constructors work correctly under
both `Float64` and `ForwardDiff.Dual` element types.
"""
function get_collect_names(fe::FixedEffects)
    params = get_params(fe)
    names = Symbol[]
    for name in get_names(fe)
        p = getfield(params, name)
        if p isa ProbabilityVector || p isa DiscreteTransitionMatrix ||
           p isa ContinuousTransitionMatrix
            push!(names, name)
        end
    end
    return names
end

"""
    logprior(fe::FixedEffects, θ_untransformed::ComponentArray) -> Real

Compute the total log-prior of all parameters with assigned priors, evaluated at
the natural-scale parameter vector `θ_untransformed`. Returns `0.0` if all priors
are `Priorless()`.
"""
function logprior(fe::FixedEffects, θ_untransformed::ComponentArray)
    return logprior(fe.extras.priors, θ_untransformed)
end

# Subset a name-keyed ComponentArray (or any property container) to `names` in
# order — the standard free-parameter extraction used across the fit entry points.
@inline function _ca_subset(x, names)
    return ComponentArray(NamedTuple{Tuple(names)}(Tuple(getproperty(x, n)
    for n in names)))
end

function logprior(priors::NamedTuple, θ::ComponentArray)
    # getproperty (a view), not θ[name] (which copies array-valued components) —
    # logpdf is evaluated on identical values either way, and this runs per
    # MAP/profile/wald objective evaluation. Typed zero keeps the accumulator
    # concrete under Dual-θ gradient sweeps.
    total = zero(promote_type(eltype(θ), Float64))
    for name in keys(priors)
        prior = getfield(priors, name)
        prior isa Priorless && continue
        val = getproperty(θ, name)
        total += _logprior_eval(prior, val)
    end
    return total
end

function logprior(pd::Distribution, θ::ComponentArray)
    return logpdf(pd, θ)
end

function logprior(pd::Distribution, θ::AbstractVector)
    return logpdf(pd, θ)
end

function _with_name_kw(rhs::Expr, name::Symbol)
    if rhs.head != :call
        return rhs
    end
    has_params = !isempty(rhs.args) && rhs.args[2] isa Expr &&
                 rhs.args[2].head == :parameters
    if has_params
        params = rhs.args[2]
        for arg in params.args
            if arg isa Expr && arg.head == :(=) && arg.args[1] == :name
                return rhs
            end
        end
        push!(params.args, Expr(:(=), :name, QuoteNode(name)))
        return rhs
    else
        params = Expr(:parameters, Expr(:(=), :name, QuoteNode(name)))
        return Expr(:call, rhs.args[1], params, rhs.args[2:end]...)
    end
end

# Parameter-block constructors owned by NoLimits. The macro resolves these call heads
# to this module so models build even when another loaded package (e.g. Makie, which
# exports `RealVector`) puts a same-named binding into the caller's namespace.
const _PARAM_BLOCK_CTORS = (:RealNumber, :RealVector, :RealPSDMatrix, :RealLiePSDMatrix,
    :RealDiagonalMatrix, :ProbabilityVector, :DiscreteTransitionMatrix,
    :ContinuousTransitionMatrix, :NNParameters, :SoftTreeParameters, :SplineParameters,
    :NPFParameter)

function _qualify_ctor_head(ex::Expr)
    head = ex.args[1]
    if head isa Symbol && head in _PARAM_BLOCK_CTORS
        return Expr(:call, GlobalRef(@__MODULE__, head), ex.args[2:end]...)
    end
    return ex
end

function _parse_fixed_effects(block::Expr)
    block.head == :block || error("@fixedEffects expects a begin ... end block.")
    names = Symbol[]
    exprs = Expr[]
    for stmt in block.args
        stmt isa LineNumberNode && continue
        stmt isa Expr || error("Invalid statement in @fixedEffects block.")
        stmt.head == :(=) || error("Only assignments are allowed in @fixedEffects block.")
        lhs, rhs = stmt.args
        lhs isa Symbol || error("Left-hand side must be a symbol in @fixedEffects block.")
        rhs isa Expr && rhs.head == :call ||
            error("Right-hand side must be a constructor call in @fixedEffects block.")
        push!(names, lhs)
        push!(exprs, _qualify_ctor_head(_with_name_kw(rhs, lhs)))
    end
    return names, exprs
end

"""
    @fixedEffects begin
        name = ParameterBlockType(...)
        ...
    end

Compile a block of fixed-effect parameter declarations into a [`FixedEffects`](@ref) struct.

Each statement must be an assignment `name = constructor(...)` where the right-hand side
is one of the parameter block constructors: [`RealNumber`](@ref), [`RealVector`](@ref),
[`RealPSDMatrix`](@ref), [`RealDiagonalMatrix`](@ref), [`NNParameters`](@ref),
[`SoftTreeParameters`](@ref), [`SplineParameters`](@ref), or [`NPFParameter`](@ref).

The LHS symbol becomes the parameter name and is automatically injected as the `name`
keyword argument into each constructor.

The `@fixedEffects` block is typically used inside `@Model`. It can also be used
standalone to construct a `FixedEffects` object directly.
"""
macro fixedEffects(block)
    names, exprs = _parse_fixed_effects(block)
    isempty(names) && return :(build_fixed_effects(NamedTuple()))
    assigns = [:($(names[i]) = $(esc(exprs[i]))) for i in eachindex(names)]
    nt = Expr(:tuple, (Expr(:(=), names[i], names[i]) for i in eachindex(names))...)
    return quote
        $(assigns...)
        build_fixed_effects($nt)
    end
end

function _value_type(val)
    val isa Number && return typeof(val)
    val isa AbstractArray && return eltype(val)
    return Float64
end

function _infer_scalar_type(values)
    T = Float64
    for v in values
        T = promote_type(T, _value_type(v))
    end
    return T
end

function _to_type(::Type{T}, x) where {T}
    x isa Number && return T(x)
    x isa AbstractArray && return T.(x)
    return x
end

function build_fixed_effects(params::NamedTuple)
    names = Symbol[collect(keys(params))...]

    if isempty(names)
        empty_nt = NamedTuple()
        θ0 = ComponentArray(empty_nt)
        bounds = (ComponentArray(empty_nt), ComponentArray(empty_nt))
        specs = TransformSpec[]
        transform = ForwardTransform(Symbol[], specs, getaxes(θ0), 0)
        inverse_transform = InverseTransform(Symbol[], specs, getaxes(θ0), 0)
        meta = FixedEffectsMeta(Symbol[], Symbol[])
        values = FixedEffectsValues(θ0, θ0)
        bounds_obj = FixedEffectsBounds(bounds, bounds)
        transforms = FixedEffectsTransforms(transform, inverse_transform)
        extras = FixedEffectsExtras(NamedTuple(), Bool[], Symbol[], NamedTuple(), empty_nt)
        return FixedEffects(meta, values, bounds_obj, transforms, extras)
    end

    value_pairs = Pair{Symbol, Any}[]
    lower_pairs = Pair{Symbol, Any}[]
    upper_pairs = Pair{Symbol, Any}[]
    specs = TransformSpec[]
    priors = Pair{Symbol, Any}[]
    se_flags = Dict{Symbol, Bool}()
    model_fun_pairs = Pair{Symbol, Any}[]

    for name in names
        p = getfield(params, name)
        push!(value_pairs, name => _param_value(p))
        push!(lower_pairs, name => _param_lower(p))
        push!(upper_pairs, name => _param_upper(p))
        push!(specs, _param_spec(name, p))
        push!(priors, name => p.prior)
        se_flags[name] = p.calculate_se
    end

    scalar_T = _infer_scalar_type(last.(value_pairs))
    for name in names
        p = getfield(params, name)
        _collect_model_fun!(p, model_fun_pairs, scalar_T)
    end

    θ0_untransformed = ComponentArray(NamedTuple{Tuple(first.(value_pairs))}(Tuple(last.(value_pairs))))
    bounds_untransformed = (
        ComponentArray(NamedTuple{Tuple(first.(lower_pairs))}(Tuple(last.(lower_pairs)))),
        ComponentArray(NamedTuple{Tuple(first.(upper_pairs))}(Tuple(last.(upper_pairs))))
    )

    # Bootstrap with the legacy (dynamic) transform to learn the output layout, then
    # rebuild the transforms with precomputed axes so applications use the type-stable
    # assembly path (required for Enzyme; see ParameterTranformations.jl).
    transform0 = ForwardTransform(names, specs)
    θ0_transformed = transform0(θ0_untransformed)
    transform = ForwardTransform(
        names, specs, getaxes(θ0_transformed), length(θ0_transformed))
    inverse_transform = InverseTransform(
        names, specs, getaxes(θ0_untransformed), length(θ0_untransformed))
    bounds_transformed = _transform_bounds(
        bounds_untransformed, names, specs; params = params)

    flat_names, _ = _flatten_by_specs(θ0_transformed, names, specs)

    # Keep untransformed values/bounds in top-level (name -> value) shape.

    se_mask = _se_mask(θ0_transformed, names, specs, se_flags)
    se_names = [flat_names[i] for i in eachindex(se_mask) if se_mask[i]]

    model_funs = NamedTuple(model_fun_pairs)
    priors_nt = NamedTuple{Tuple(first.(priors))}(Tuple(last.(priors)))

    meta = FixedEffectsMeta(names, flat_names)
    values = FixedEffectsValues(θ0_untransformed, θ0_transformed)
    bounds = FixedEffectsBounds(bounds_untransformed, bounds_transformed)
    transforms = FixedEffectsTransforms(transform, inverse_transform)
    extras = FixedEffectsExtras(priors_nt, se_mask, se_names, model_funs, params)

    return FixedEffects(meta, values, bounds, transforms, extras)
end

# Every parameter block stores its (initial) value in `.value`.
_param_value(p::AbstractParameterBlock) = p.value

# Blocks with explicit user-settable bounds store them in `.lower`/`.upper`;
# the constrained blocks (PSD/diagonal/simplex/transition) derive theirs below.
const _ExplicitBoundsBlock = Union{RealNumber, RealVector, NNParameters, NPFParameter,
    SoftTreeParameters, SplineParameters}
_param_lower(p::_ExplicitBoundsBlock) = p.lower
_param_upper(p::_ExplicitBoundsBlock) = p.upper

function _param_lower(p::RealPSDMatrix)
    return fill(-Inf, size(p.value))
end

# Natural-scale bounds on the matrix entries are unbounded; the eigenvalue box bounds
# are applied on the transformed (log-eigenvalue) scale in `_transform_bounds`.
function _param_lower(p::RealLiePSDMatrix)
    return fill(-Inf, size(p.value))
end

function _param_lower(p::RealDiagonalMatrix)
    return fill(-Inf, length(p.value))
end

function _param_lower(p::ProbabilityVector)
    return fill(zero(eltype(p.value)), length(p.value))
end

function _param_lower(p::DiscreteTransitionMatrix)
    return fill(zero(eltype(p.value)), size(p.value))
end

function _param_lower(p::ContinuousTransitionMatrix)
    # Off-diagonal entries must be ≥ 0; diagonal has no lower bound (it's derived).
    n = size(p.value, 1)
    T = eltype(p.value)
    lb = fill(zero(T), n, n)
    for i in 1:n
        lb[i, i] = T(-Inf)
    end
    return lb
end

function _param_upper(p::RealPSDMatrix)
    return fill(Inf, size(p.value))
end

function _param_upper(p::RealLiePSDMatrix)
    return fill(Inf, size(p.value))
end

function _param_upper(p::RealDiagonalMatrix)
    return fill(Inf, length(p.value))
end

function _param_upper(p::ProbabilityVector)
    return fill(one(eltype(p.value)), length(p.value))
end

function _param_upper(p::DiscreteTransitionMatrix)
    return fill(one(eltype(p.value)), size(p.value))
end

function _param_upper(p::ContinuousTransitionMatrix)
    return fill(Inf, size(p.value))
end
function _param_spec(name::Symbol, p::RealNumber)
    return TransformSpec(name, p.scale, (1, 1), nothing)
end

function _param_spec(name::Symbol, p::RealVector)
    scales = p.scale  # Vector{Symbol}
    n = length(p.value)
    first_scale = scales[1]
    if all(s -> s === first_scale, scales)
        return TransformSpec(name, first_scale, (n, 1), nothing)
    else
        return TransformSpec(name, :elementwise, (n, 1), collect(scales))
    end
end

function _param_spec(name::Symbol, p::RealPSDMatrix)
    return TransformSpec(name, p.scale, size(p.value), nothing)
end

function _param_spec(name::Symbol, p::RealLiePSDMatrix)
    layout = _build_lie_layout(p.value, p.blocks, p.fixed_eigenvalues)
    return TransformSpec(name, :lie, size(p.value), nothing, layout)
end

function _param_spec(name::Symbol, p::RealDiagonalMatrix)
    return TransformSpec(name, :log, (length(p.value), 1), nothing)
end

function _param_spec(name::Symbol, p::NNParameters)
    return TransformSpec(name, :identity, (length(p.value), 1), nothing)
end

function _param_spec(name::Symbol, p::NPFParameter)
    return TransformSpec(name, :identity, (length(p.value), 1), nothing)
end

function _param_spec(name::Symbol, p::SoftTreeParameters)
    return TransformSpec(name, :identity, (length(p.value), 1), nothing)
end

function _param_spec(name::Symbol, p::SplineParameters)
    return TransformSpec(name, :identity, (length(p.value), 1), nothing)
end

function _param_spec(name::Symbol, p::ProbabilityVector)
    k = length(p.value)
    return TransformSpec(name, :stickbreak, (k, 1), nothing)
end

function _param_spec(name::Symbol, p::DiscreteTransitionMatrix)
    n = size(p.value, 1)
    return TransformSpec(name, :stickbreakrows, (n, n), nothing)
end

function _param_spec(name::Symbol, p::ContinuousTransitionMatrix)
    n = size(p.value, 1)
    return TransformSpec(name, :lograterows, (n, n), nothing)
end
# NN parameters are rebuilt from the flat vector via ComponentArray axes instead of
# the Optimisers.Restructure closure: Restructure goes through Functors.fmap, whose
# IdDict cache lookups (`jl_eqtable_get`) have no Enzyme forward-mode rule. The
# ComponentArray construction is also type-stable and skips the per-call fmap
# conversion the old path always performed (`eltype(::NamedTuple)` is never `=== T`).
# Layout equality with the destructure order used for `p.value` is asserted at
# model-build time via `_check_nn_flat_layout`.
function _nn_axes_template(p::NNParameters, ::Type{T}) where {T}
    ps_template = Functors.fmap(y -> _to_type(T, y), p.reconstructor(p.value))
    ps_ca = ComponentArray(ps_template)
    _check_nn_flat_layout(p, ps_ca, T)
    return getaxes(ps_ca)
end

function _check_nn_flat_layout(p::NNParameters, ps_ca::ComponentArray, ::Type{T}) where {T}
    collect(ps_ca) == _to_type(T, p.value) ||
        error("NN parameter $(p.name): ComponentArray flat layout does not match the Optimisers.destructure order of the stored value. Cannot build $(p.function_name).")
    return nothing
end

# SimpleChains backend (NNParameters built from a `SimpleChains.SimpleChain`): parameters are
# natively a flat vector, so the forward pass is `chain(x, θ)` directly — no Lux states and no
# ComponentArray-axes rebuild. `_to_type` materialises the input and the ComponentArray view of θ
# into dense `Vector{TT}` (what SimpleChains expects), promoting so both share an eltype. This
# path is ForwardDiff-compatible (SimpleChains' forwarddiff_matmul handles `Dual`s) but NOT
# Enzyme-differentiable (LoopVectorization `@turbo`) — use a Lux chain for AutoEnzyme. Output is an
# `SVector`, indexable exactly like the Lux `Vector` output (`NN1(x, θ)[1]`). The backend is chosen
# by branching on `p.chain isa SimpleChain` (resolved at compile time for each concrete `p`),
# because parametric dispatch on the NNParameters chain type does not reliably out-specialise the
# unconstrained Lux methods when a free `Type{T}` argument is present.
function _simplechain_model_fun(chain)
    return (x, θ) -> begin
        TT = promote_type(eltype(θ), _value_type(x))
        return chain(_to_type(TT, x), _to_type(TT, θ))
    end
end

function _collect_model_fun!(p::NNParameters, model_fun_pairs, ::Type{T}) where {T}
    if p.chain isa SimpleChain
        push!(model_fun_pairs, p.function_name => _simplechain_model_fun(p.chain))
        return nothing
    end
    st = Lux.initialstates(Xoshiro(0), p.chain)
    stT = Functors.fmap(y -> _to_type(T, y), st)
    ps_axes = _nn_axes_template(p, T)
    push!(model_fun_pairs,
        p.function_name => (x, θ) -> begin
            TT = promote_type(eltype(θ), _value_type(x))
            if TT === T
                xT = _to_type(T, x)
                psT = ComponentArray(_to_type(T, θ), ps_axes)
                return first(Lux.apply(p.chain, xT, psT, stT))
            end
            xTT = _to_type(TT, x)
            psTT = ComponentArray(_to_type(TT, θ), ps_axes)
            stTT = Functors.fmap(y -> _to_type(TT, y), stT)
            return first(Lux.apply(p.chain, xTT, psTT, stTT))
        end)
end

# SoftTree parameters are rebuilt positionally from the flat vector
# (`softtree_params_from_flat`) instead of via the Optimisers.Restructure closure —
# same Enzyme rationale as for NNParameters above. Layout equality with the
# destructure order is asserted at model-build time.
function _check_softtree_flat_layout(p::SoftTreeParameters, tree::SoftTree)
    p_pos = softtree_params_from_flat(p.value, tree)
    p_ref = p.reconstructor(p.value)
    (p_pos.node_weights == p_ref.node_weights &&
     p_pos.node_biases == p_ref.node_biases &&
     p_pos.leaf_values == p_ref.leaf_values) ||
        error("SoftTree parameter $(p.name): positional flat layout does not match the Optimisers.destructure order of the stored value. Cannot build $(p.function_name).")
    return nothing
end

function _collect_model_fun!(p::SoftTreeParameters, model_fun_pairs, ::Type{T}) where {T}
    tree = SoftTree(p.input_dim, p.depth, p.n_output)
    _check_softtree_flat_layout(p, tree)
    push!(model_fun_pairs,
        p.function_name => (x, θ) -> begin
            TT = promote_type(eltype(θ), _value_type(x))
            if TT === T
                return tree(_to_type(T, x), softtree_params_from_flat(_to_type(T, θ), tree))
            end
            return tree(_to_type(TT, x), softtree_params_from_flat(_to_type(TT, θ), tree))
        end)
end

function _collect_model_fun!(p::SplineParameters, model_fun_pairs, ::Type{T}) where {T}
    push!(model_fun_pairs,
        p.function_name => (x, θ) -> begin
            TT = promote_type(eltype(θ), _value_type(x))
            if TT === T
                return bspline_eval(_to_type(T, x), _to_type(T, θ), p.knots, p.degree)
            end
            return bspline_eval(_to_type(TT, x), _to_type(TT, θ), p.knots, p.degree)
        end)
end
# The NormalizingPlanarFlow flat-θ constructor rebuilds the planar chain
# positionally (`_planar_chain_from_flat`) instead of via the stored
# Optimisers.Restructure — same Enzyme rationale as for NNParameters/SoftTree
# above. Layout equality with the destructure order is asserted at model-build time.
function _check_npf_flat_layout(p::NPFParameter)
    bij = _planar_chain_from_flat(p.value, p.n_input)
    flat_ref, _ = Optimisers.destructure(bij)
    flat_ref == p.value ||
        error("NPF parameter $(p.name): positional flat layout does not match the Optimisers.destructure order of the stored value. Cannot build NPF_$(p.name).")
    return nothing
end

function _collect_model_fun!(p::NPFParameter, model_fun_pairs, ::Type{T}) where {T}
    key = Symbol("NPF_", p.name)
    q0T = _adapt_base_dist(p.base_dist, T)
    _check_npf_flat_layout(p)
    push!(model_fun_pairs,
        key => (θ) -> begin
            TT = eltype(θ)
            if TT === T
                return NormalizingPlanarFlow(_to_type(T, θ), p.reconstructor, q0T)
            end
            q0 = _adapt_base_dist(p.base_dist, TT)
            return NormalizingPlanarFlow(_to_type(TT, θ), p.reconstructor, q0)
        end)
end

# Adapt base distribution element type for ForwardDiff compatibility.
# MvNormal is re-parameterized with typed mean/cov; other distributions are used as-is.
# The covariance STRUCTURE is preserved: densifying a ScalMat/PDiagMat to a full
# PDMat would route logpdf through a LAPACK triangular solve (`trtrs`), which
# Enzyme forward mode cannot handle under runtime activity ("Runtime Activity not
# yet implemented for Forward-Mode BLAS calls"); the scalar/diagonal forms use
# plain dot/loops instead.
function _adapt_base_dist(
        d::MvNormal{<:Real, <:Distributions.PDMats.ScalMat}, ::Type{T}) where {T}
    MvNormal(T.(mean(d)), Distributions.PDMats.ScalMat(length(d), T(d.Σ.value)))
end
function _adapt_base_dist(
        d::MvNormal{<:Real, <:Distributions.PDMats.PDiagMat}, ::Type{T}) where {T}
    MvNormal(T.(mean(d)), Distributions.PDMats.PDiagMat(T.(d.Σ.diag)))
end
function _adapt_base_dist(d::MvNormal, ::Type{T}) where {T}
    MvNormal(T.(mean(d)), Matrix{T}(cov(d)))
end
_adapt_base_dist(d, ::Type{T}) where {T} = d
function _collect_model_fun!(p, model_fun_pairs, ::Type{T}) where {T}
    return nothing
end

function _logprior_eval(prior::Distribution, val)
    return logpdf(prior, val)
end

function _logprior_eval(prior::AbstractVector{<:Distribution}, val::AbstractVector)
    length(prior) == length(val) ||
        error("Prior length mismatch. Expected $(length(val)); got $(length(prior)).")
    pd = product_distribution(prior)
    return logpdf(pd, val)
end

function _flatten_by_specs(
        θ::ComponentArray, names::Vector{Symbol}, specs::Vector{TransformSpec})
    flat_names = Symbol[]
    flat_vals = Float64[]

    for (i, name) in enumerate(names)
        spec = specs[i]
        val = θ[name]
        if spec.kind == :cholesky
            mat = reshape(val, spec.size...)
            for r in 1:spec.size[1]
                for c in 1:spec.size[2]
                    push!(flat_names, Symbol(name, "_", r, "_", c))
                    push!(flat_vals, mat[r, c])
                end
            end
        elseif val isa AbstractMatrix
            for r in 1:size(val, 1)
                for c in 1:size(val, 2)
                    push!(flat_names, Symbol(name, "_", r, "_", c))
                    push!(flat_vals, val[r, c])
                end
            end
        elseif val isa AbstractVector
            for j in eachindex(val)
                push!(flat_names, Symbol(name, "_", j))
                push!(flat_vals, val[j])
            end
        else
            push!(flat_names, name)
            push!(flat_vals, val)
        end
    end

    return flat_names, flat_vals
end

function _transform_bounds(bounds::Tuple{ComponentArray, ComponentArray},
        names::Vector{Symbol}, specs::Vector{TransformSpec}; params = NamedTuple())
    lower, upper = bounds
    lower_pairs = Pair{Symbol, Any}[]
    upper_pairs = Pair{Symbol, Any}[]

    for (i, name) in enumerate(names)
        spec = specs[i]
        l = lower[name]
        u = upper[name]
        if spec.kind == :log
            if l isa AbstractArray
                l2 = map(x -> x == -Inf ? log(EPSILON) : log(x), l)
                u2 = map(x -> x == Inf ? Inf : log(x), u)
                push!(lower_pairs, name => l2)
                push!(upper_pairs, name => u2)
            else
                l2 = l == -Inf ? log(EPSILON) : log(l)
                u2 = u == Inf ? Inf : log(u)
                push!(lower_pairs, name => l2)
                push!(upper_pairs, name => u2)
            end
        elseif spec.kind == :logit
            if l isa AbstractArray
                l2 = map(x -> (isinf(x) || x <= 0) ? -Inf : log(x / (1 - x)), l)
                u2 = map(x -> (isinf(x) || x >= 1) ? Inf : log(x / (1 - x)), u)
                push!(lower_pairs, name => l2)
                push!(upper_pairs, name => u2)
            else
                l2 = (isinf(l) || l <= 0) ? -Inf : log(l / (1 - l))
                u2 = (isinf(u) || u >= 1) ? Inf : log(u / (1 - u))
                push!(lower_pairs, name => l2)
                push!(upper_pairs, name => u2)
            end
        elseif spec.kind == :elementwise
            mask = spec.mask
            l2 = copy(collect(l))
            u2 = copy(collect(u))
            for j in eachindex(mask)
                if mask[j] === :log
                    l2[j] = l2[j] == -Inf ? log(EPSILON) : log(l2[j])
                    u2[j] = u2[j] == Inf ? Inf : log(u2[j])
                elseif mask[j] === :logit
                    lj = l2[j]
                    uj = u2[j]
                    l2[j] = (isinf(lj) || lj <= 0) ? -Inf : log(lj / (1 - lj))
                    u2[j] = (isinf(uj) || uj >= 1) ? Inf : log(uj / (1 - uj))
                end
            end
            push!(lower_pairs, name => l2)
            push!(upper_pairs, name => u2)
        elseif spec.kind == :cholesky
            n1, n2 = spec.size
            push!(lower_pairs, name => fill(-Inf, n1 * n2))
            push!(upper_pairs, name => fill(Inf, n1 * n2))
        elseif spec.kind == :expm
            n1, n2 = spec.size
            n = n1 * (n1 + 1) ÷ 2
            push!(lower_pairs, name => fill(-Inf, n))
            push!(upper_pairs, name => fill(Inf, n))
        elseif spec.kind == :stickbreak
            k = spec.size[1]
            push!(lower_pairs, name => fill(-Inf, k - 1))
            push!(upper_pairs, name => fill(Inf, k - 1))
        elseif spec.kind == :stickbreakrows
            n = spec.size[1]
            push!(lower_pairs, name => fill(-Inf, n * (n - 1)))
            push!(upper_pairs, name => fill(Inf, n * (n - 1)))
        elseif spec.kind == :lograterows
            n = spec.size[1]
            push!(lower_pairs, name => fill(-Inf, n * (n - 1)))
            push!(upper_pairs, name => fill(Inf, n * (n - 1)))
        elseif spec.kind == :lie
            # Transformed layout is `[λ (n); α (nA)]` (or, when structured, only the free
            # entries). Eigenvalue box bounds map through the log to bounds on the λ slots
            # (0 → -Inf, Inf → Inf); rotation-coefficient (α) slots are unbounded.
            n = spec.size[1]
            p = hasproperty(params, name) ? getproperty(params, name) : nothing
            lo_eig = p isa RealLiePSDMatrix ? p.eigenvalue_lower : fill(0.0, n)
            up_eig = p isa RealLiePSDMatrix ? p.eigenvalue_upper : fill(Inf, n)
            _lie_lo(i) = i > n ? -Inf : (lo_eig[i] <= 0 ? -Inf : log(lo_eig[i]))
            _lie_up(i) = i > n ? Inf : (isinf(up_eig[i]) ? Inf : log(up_eig[i]))
            slots = spec.lie === nothing ? (1:(n * (n + 1) ÷ 2)) : spec.lie.free_idx
            l2 = [_lie_lo(i) for i in slots]
            u2 = [_lie_up(i) for i in slots]
            push!(lower_pairs, name => l2)
            push!(upper_pairs, name => u2)
        else
            push!(lower_pairs, name => l)
            push!(upper_pairs, name => u)
        end
    end

    return (
        ComponentArray(NamedTuple{Tuple(first.(lower_pairs))}(Tuple(last.(lower_pairs)))),
        ComponentArray(NamedTuple{Tuple(first.(upper_pairs))}(Tuple(last.(upper_pairs))))
    )
end

function _se_mask(θ::ComponentArray, names::Vector{Symbol},
        specs::Vector{TransformSpec}, flags::Dict{Symbol, Bool})
    mask = Bool[]
    for (i, name) in enumerate(names)
        spec = specs[i]
        val = θ[name]
        n = val isa AbstractMatrix ? length(val) :
            (val isa AbstractVector ? length(val) : 1)
        append!(mask, fill(flags[name], n))
    end
    return mask
end
