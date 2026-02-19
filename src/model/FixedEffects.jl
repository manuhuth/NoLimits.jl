#Functions from NoLimits dont have to be imported

using ComponentArrays
using Distributions
using LinearAlgebra
using Random
using Functors

export @fixedEffects
export FixedEffects
export logprior
export get_meta
export get_values
export get_bounds
export get_transforms
export get_extras
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

struct FixedEffectsTransforms
    forward::ForwardTransform
    inverse::InverseTransform
end

struct FixedEffectsExtras
    priors::NamedTuple
    se_mask_transformed::Vector{Bool}
    se_names_transformed::Vector{Symbol}
    model_funs::NamedTuple
    params::NamedTuple
end

"""
    FixedEffects

Compiled representation of a `@fixedEffects` block. Contains initial parameter values,
bounds, forward/inverse transforms, priors, SE masks, and model functions (NNs, splines,
soft trees, NPFs).

Use accessor functions rather than accessing fields directly.
"""
struct FixedEffects
    meta::FixedEffectsMeta
    values::FixedEffectsValues
    bounds::FixedEffectsBounds
    transforms::FixedEffectsTransforms
    extras::FixedEffectsExtras
end

"""    get_meta(fe::FixedEffects) -> FixedEffectsMeta

Return the metadata struct containing parameter names and flat names.
"""
get_meta(fe::FixedEffects) = fe.meta

"""
    get_values(fe::FixedEffects) -> FixedEffectsValues

Return the initial parameter values on both the natural and transformed scales.
"""
get_values(fe::FixedEffects) = fe.values

"""
    get_bounds(fe::FixedEffects) -> FixedEffectsBounds

Return the parameter bounds on both the natural and transformed scales.
"""
get_bounds(fe::FixedEffects) = fe.bounds

"""
    get_transforms(fe::FixedEffects) -> FixedEffectsTransforms

Return the `ForwardTransform` / `InverseTransform` pair for this parameter block.
"""
get_transforms(fe::FixedEffects) = fe.transforms

"""
    get_extras(fe::FixedEffects) -> FixedEffectsExtras

Return the extras struct (priors, SE mask, model functions, raw parameter blocks).
"""
get_extras(fe::FixedEffects) = fe.extras

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

Return the initial parameter vector on the transformed (optimisation) scale.
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
to the optimisation scale.
"""
get_transform(fe::FixedEffects) = fe.transforms.forward

"""
    get_inverse_transform(fe::FixedEffects) -> InverseTransform

Return the callable `InverseTransform` that maps an optimisation-scale `ComponentArray`
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
    logprior(fe::FixedEffects, θ_untransformed::ComponentArray) -> Real

Compute the total log-prior of all parameters with assigned priors, evaluated at
the natural-scale parameter vector `θ_untransformed`. Returns `0.0` if all priors
are `Priorless()`.
"""
function logprior(fe::FixedEffects, θ_untransformed::ComponentArray)
    return logprior(fe.extras.priors, θ_untransformed)
end

function logprior(priors::NamedTuple, θ::ComponentArray)
    total = 0.0
    for name in keys(priors)
        prior = getfield(priors, name)
        prior isa Priorless && continue
        val = θ[name]
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
    has_params = !isempty(rhs.args) && rhs.args[2] isa Expr && rhs.args[2].head == :parameters
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
        rhs isa Expr && rhs.head == :call || error("Right-hand side must be a constructor call in @fixedEffects block.")
        push!(names, lhs)
        push!(exprs, _with_name_kw(rhs, lhs))
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
    assigns = [:( $(names[i]) = $(esc(exprs[i])) ) for i in eachindex(names)]
    nt = Expr(:tuple, (Expr(:(=), names[i], names[i]) for i in eachindex(names))...)
    return quote
        $(assigns...)
        build_fixed_effects($nt)
    end
end

function _param_value_type(val)
    val isa Number && return typeof(val)
    val isa AbstractArray && return eltype(val)
    return Float64
end

function _value_type(val)
    val isa Number && return typeof(val)
    val isa AbstractArray && return eltype(val)
    return Float64
end

function _infer_scalar_type(values)
    T = Float64
    for v in values
        T = promote_type(T, _param_value_type(v))
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
        transform = ForwardTransform(Symbol[], specs)
        inverse_transform = InverseTransform(Symbol[], specs)
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

    transform = ForwardTransform(names, specs)
    inverse_transform = InverseTransform(names, specs)

    θ0_transformed = transform(θ0_untransformed)
    bounds_transformed = _transform_bounds(bounds_untransformed, names, specs)

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

function _param_value(p::RealNumber)
    return p.value
end

function _param_value(p::RealVector)
    return p.value
end

function _param_value(p::RealPSDMatrix)
    return p.value
end

function _param_value(p::RealDiagonalMatrix)
    return p.value
end

function _param_value(p::NNParameters)
    return p.value
end

function _param_value(p::NPFParameter)
    return p.value
end

function _param_value(p::SoftTreeParameters)
    return p.value
end

function _param_value(p::SplineParameters)
    return p.value
end
function _param_lower(p::RealNumber)
    return p.lower
end

function _param_lower(p::RealVector)
    return p.lower
end

function _param_lower(p::RealPSDMatrix)
    return fill(-Inf, size(p.value))
end

function _param_lower(p::RealDiagonalMatrix)
    return fill(-Inf, length(p.value))
end

function _param_lower(p::NNParameters)
    return p.lower
end

function _param_lower(p::NPFParameter)
    return p.lower
end

function _param_lower(p::SoftTreeParameters)
    return p.lower
end

function _param_lower(p::SplineParameters)
    return p.lower
end
function _param_upper(p::RealNumber)
    return p.upper
end

function _param_upper(p::RealVector)
    return p.upper
end

function _param_upper(p::RealPSDMatrix)
    return fill(Inf, size(p.value))
end

function _param_upper(p::RealDiagonalMatrix)
    return fill(Inf, length(p.value))
end

function _param_upper(p::NNParameters)
    return p.upper
end

function _param_upper(p::NPFParameter)
    return p.upper
end

function _param_upper(p::SoftTreeParameters)
    return p.upper
end

function _param_upper(p::SplineParameters)
    return p.upper
end
function _param_spec(name::Symbol, p::RealNumber)
    return TransformSpec(name, p.scale, (1, 1), nothing)
end

function _param_spec(name::Symbol, p::RealVector)
    mask = p.scale .== :log
    if any(mask)
        return TransformSpec(name, :log, (length(p.value), 1), collect(mask))
    end
    return TransformSpec(name, :identity, (length(p.value), 1), nothing)
end

function _param_spec(name::Symbol, p::RealPSDMatrix)
    return TransformSpec(name, p.scale, size(p.value), nothing)
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
function _collect_model_fun!(p::NNParameters, model_fun_pairs)
    st = Lux.initialstates(Xoshiro(0), p.chain)
    push!(model_fun_pairs, p.function_name => (x, θ) -> first(Lux.apply(p.chain, x, p.reconstructor(θ), st)))
end

function _collect_model_fun!(p::NNParameters, model_fun_pairs, ::Type{T}) where {T}
    st = Lux.initialstates(Xoshiro(0), p.chain)
    stT = Functors.fmap(y -> _to_type(T, y), st)
    push!(model_fun_pairs, p.function_name => (x, θ) -> begin
        TT = promote_type(eltype(θ), _value_type(x))
        if TT === T
            xT = _to_type(T, x)
            ps = p.reconstructor(_to_type(T, θ))
            psT = eltype(ps) === T ? ps : Functors.fmap(y -> _to_type(T, y), ps)
            return first(Lux.apply(p.chain, xT, psT, stT))
        end
        xTT = _to_type(TT, x)
        ps = p.reconstructor(_to_type(TT, θ))
        psTT = eltype(ps) === TT ? ps : Functors.fmap(y -> _to_type(TT, y), ps)
        stTT = Functors.fmap(y -> _to_type(TT, y), stT)
        return first(Lux.apply(p.chain, xTT, psTT, stTT))
    end)
end

function _collect_model_fun!(p::SoftTreeParameters, model_fun_pairs)
    tree = SoftTree(p.input_dim, p.depth, p.n_output)
    push!(model_fun_pairs, p.function_name => (x, θ) -> tree(x, p.reconstructor(θ)))
end

function _collect_model_fun!(p::SoftTreeParameters, model_fun_pairs, ::Type{T}) where {T}
    tree = SoftTree(p.input_dim, p.depth, p.n_output)
    push!(model_fun_pairs, p.function_name => (x, θ) -> begin
        TT = promote_type(eltype(θ), _value_type(x))
        if TT === T
            return tree(_to_type(T, x), _to_type(T, p.reconstructor(θ)))
        end
        return tree(_to_type(TT, x), _to_type(TT, p.reconstructor(θ)))
    end)
end

function _collect_model_fun!(p::SplineParameters, model_fun_pairs)
    push!(model_fun_pairs, p.function_name => (x, θ) -> bspline_eval(x, θ, p.knots, p.degree))
end

function _collect_model_fun!(p::SplineParameters, model_fun_pairs, ::Type{T}) where {T}
    push!(model_fun_pairs, p.function_name => (x, θ) -> begin
        TT = promote_type(eltype(θ), _value_type(x))
        if TT === T
            return bspline_eval(_to_type(T, x), _to_type(T, θ), p.knots, p.degree)
        end
        return bspline_eval(_to_type(TT, x), _to_type(TT, θ), p.knots, p.degree)
    end)
end
function _collect_model_fun!(p::NPFParameter, model_fun_pairs)
    key = Symbol("NPF_", p.name)
    q0 = MvNormal(zeros(p.n_input), I)
    push!(model_fun_pairs, key => (θ) -> NormalizingPlanarFlow(θ, p.reconstructor, q0))
end
function _collect_model_fun!(p::NPFParameter, model_fun_pairs, ::Type{T}) where {T}
    key = Symbol("NPF_", p.name)
    q0T = MvNormal(zeros(T, p.n_input), I)
    push!(model_fun_pairs, key => (θ) -> begin
        TT = eltype(θ)
        if TT === T
            return NormalizingPlanarFlow(_to_type(T, θ), p.reconstructor, q0T)
        end
        q0 = MvNormal(zeros(TT, p.n_input), I)
        return NormalizingPlanarFlow(_to_type(TT, θ), p.reconstructor, q0)
    end)
end
function _collect_model_fun!(p, model_fun_pairs)
    return nothing
end
function _collect_model_fun!(p, model_fun_pairs, ::Type{T}) where {T}
    return nothing
end

function _logprior_for_param(p, val)
    p.prior isa Priorless && return 0.0
    return _logprior_eval(p.prior, val)
end

function _logprior_eval(prior::Distribution, val)
    return logpdf(prior, val)
end

function _logprior_eval(prior::AbstractVector{<:Distribution}, val::AbstractVector)
    length(prior) == length(val) || error("Prior length mismatch. Expected $(length(val)); got $(length(prior)).")
    pd = product_distribution(prior)
    return logpdf(pd, val)
end

function _flatten_by_specs(θ::ComponentArray, names::Vector{Symbol}, specs::Vector{TransformSpec})
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

function _transform_bounds(bounds::Tuple{ComponentArray, ComponentArray}, names::Vector{Symbol}, specs::Vector{TransformSpec})
    lower, upper = bounds
    lower_pairs = Pair{Symbol, Any}[]
    upper_pairs = Pair{Symbol, Any}[]

    for (i, name) in enumerate(names)
        spec = specs[i]
        l = lower[name]
        u = upper[name]
        if spec.kind == :log
            if l isa AbstractArray
                l2 = copy(l)
                u2 = copy(u)
                if spec.mask === nothing
                    for j in eachindex(l2)
                        l2[j] = l2[j] == -Inf ? log(EPSILON) : log(l2[j])
                        u2[j] = u2[j] == Inf ? Inf : log(u2[j])
                    end
                else
                    for j in eachindex(spec.mask)
                        if spec.mask[j]
                            l2[j] = l2[j] == -Inf ? log(EPSILON) : log(l2[j])
                            u2[j] = u2[j] == Inf ? Inf : log(u2[j])
                        end
                    end
                end
                push!(lower_pairs, name => l2)
                push!(upper_pairs, name => u2)
            else
                l2 = l == -Inf ? log(EPSILON) : log(l)
                u2 = u == Inf ? Inf : log(u)
                push!(lower_pairs, name => l2)
                push!(upper_pairs, name => u2)
            end
        elseif spec.kind == :cholesky
            n1, n2 = spec.size
            push!(lower_pairs, name => fill(-Inf, n1 * n2))
            push!(upper_pairs, name => fill(Inf, n1 * n2))
        elseif spec.kind == :expm
            n1, n2 = spec.size
            n = n1 * (n1 + 1) ÷ 2
            push!(lower_pairs, name => fill(-Inf, n))
            push!(upper_pairs, name => fill(Inf, n))
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

function _se_mask(θ::ComponentArray, names::Vector{Symbol}, specs::Vector{TransformSpec}, flags::Dict{Symbol, Bool})
    mask = Bool[]
    for (i, name) in enumerate(names)
        spec = specs[i]
        val = θ[name]
        n = val isa AbstractMatrix ? length(val) : (val isa AbstractVector ? length(val) : 1)
        append!(mask, fill(flags[name], n))
    end
    return mask
end
