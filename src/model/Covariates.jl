using DataInterpolations

export @covariates
export Covariates
export ConstantCovariate, ConstantCovariateVector
export Covariate, CovariateVector
export DynamicCovariate, DynamicCovariateVector
export finalize_covariates

abstract type AbstractCovariate end

"""
    ConstantCovariate(; constant_on) -> ConstantCovariate

A scalar covariate that is constant within a grouping (e.g. a subject-level baseline).

In `@covariates`, the LHS name determines the data column:
```julia
@covariates begin
    Age = ConstantCovariate(; constant_on=:ID)
end
```

# Keyword Arguments
- `constant_on`: a `Symbol` or vector of `Symbol`s naming the grouping column(s) within
  which the covariate is constant. When only one random-effect group exists, this defaults
  to that group's column.
"""
struct ConstantCovariate <: AbstractCovariate
    column::Symbol
    constant_on::Vector{Symbol}
end

"""
    Covariate() -> Covariate

A time-varying scalar covariate read row-by-row from the data frame.

In `@covariates`, the LHS name determines the data column and must refer to a column
present in the data frame. This type is also used to declare the time column:
```julia
@covariates begin
    t = Covariate()
    z = Covariate()
end
```
"""
struct Covariate <: AbstractCovariate
    column::Symbol
end

"""
    ConstantCovariateVector(columns::Vector{Symbol}; constant_on) -> ConstantCovariateVector

A vector of scalar covariates, each constant within a grouping.

The LHS name in `@covariates` becomes the accessor name; `columns` specifies which
data-frame columns to read:
```julia
@covariates begin
    x = ConstantCovariateVector([:Age, :BMI]; constant_on=:ID)
end
```
Accessed in model blocks as `x.Age`, `x.BMI`.

# Keyword Arguments
- `constant_on`: a `Symbol` or vector of `Symbol`s naming the grouping column(s).
"""
struct ConstantCovariateVector <: AbstractCovariate
    columns::Vector{Symbol}
    constant_on::Vector{Symbol}
end

"""
    CovariateVector(columns::Vector{Symbol}) -> CovariateVector

A vector of time-varying scalar covariates read row-by-row.

```julia
@covariates begin
    z = CovariateVector([:z1, :z2])
end
```
Accessed in model blocks as `z.z1`, `z.z2`.

# Arguments
- `columns::Vector{Symbol}`: names of the data-frame columns to collect.
"""
struct CovariateVector <: AbstractCovariate
    columns::Vector{Symbol}
end

"""
    DynamicCovariate(; interpolation=LinearInterpolation) -> DynamicCovariate

A time-varying covariate represented as a DataInterpolations.jl interpolant, callable
as `w(t)` inside `@DifferentialEquation` and `@formulas`.

In `@covariates`, the LHS name provides both the accessor name and the data-frame column.

# Keyword Arguments
- `interpolation`: a DataInterpolations.jl interpolation type (not instance). Must be
  one of `ConstantInterpolation`, `SmoothedConstantInterpolation`, `LinearInterpolation`,
  `QuadraticInterpolation`, `LagrangeInterpolation`, `QuadraticSpline`, `CubicSpline`,
  or `AkimaInterpolation`. Defaults to `LinearInterpolation`.
"""
struct DynamicCovariate <: AbstractCovariate
    column::Symbol
    interpolation
end

"""
    DynamicCovariateVector(columns::Vector{Symbol}; interpolations) -> DynamicCovariateVector

A vector of time-varying covariates, each represented as a separate interpolant.

```julia
@covariates begin
    inputs = DynamicCovariateVector([:i1, :i2]; interpolations=[LinearInterpolation, CubicSpline])
end
```

# Arguments
- `columns::Vector{Symbol}`: data-frame column names.

# Keyword Arguments
- `interpolations`: a `Vector` of DataInterpolations.jl types, one per column.
  Defaults to `LinearInterpolation` for all columns.
"""
struct DynamicCovariateVector <: AbstractCovariate
    columns::Vector{Symbol}
    interpolations::Vector
end

const ALLOWED_INTERPOLATIONS = Set([
    ConstantInterpolation,
    SmoothedConstantInterpolation,
    LinearInterpolation,
    QuadraticInterpolation,
    LagrangeInterpolation,
    QuadraticSpline,
    CubicSpline,
    AkimaInterpolation,
])

function DynamicCovariate(column::Symbol; interpolation=LinearInterpolation)
    _check_interpolation(interpolation, :dynamic_covariate)
    return DynamicCovariate(column, interpolation)
end

function ConstantCovariate(column::Symbol; constant_on=Symbol[])
    constant_on = constant_on isa Symbol ? [constant_on] : collect(constant_on)
    return ConstantCovariate(column, constant_on)
end

function ConstantCovariateVector(columns::Vector{Symbol}; constant_on=Symbol[])
    constant_on = constant_on isa Symbol ? [constant_on] : collect(constant_on)
    return ConstantCovariateVector(columns, constant_on)
end

function DynamicCovariateVector(columns::Vector{Symbol}; interpolations=fill(LinearInterpolation, length(columns)))
    length(columns) == length(interpolations) || error("Interpolation length mismatch: expected $(length(columns)); got $(length(interpolations)).")
    for itp in interpolations
        _check_interpolation(itp, :dynamic_covariate_vector)
    end
    return DynamicCovariateVector(columns, interpolations)
end

function _normalize_constant_on(val)
    val isa Symbol && return [val]
    return collect(val)
end

function _set_constant_on(p::ConstantCovariate, constant_on::Vector{Symbol})
    return ConstantCovariate(p.column; constant_on=constant_on)
end

function _set_constant_on(p::ConstantCovariateVector, constant_on::Vector{Symbol})
    return ConstantCovariateVector(p.columns; constant_on=constant_on)
end

"""
    Covariates

Compiled representation of a `@covariates` block. Stores the covariate names
categorised as constant, varying, or dynamic, along with interpolation types
and the raw covariate parameter structs.

Use the `model.covariates.covariates` field (or the `CovariatesBundle`) to inspect
this struct. Typically accessed indirectly via `DataModel` construction.
"""
struct Covariates
    names::Vector{Symbol}
    flat_names::Vector{Symbol}
    constants::Vector{Symbol}
    varying::Vector{Symbol}
    dynamic::Vector{Symbol}
    interpolations::Dict{Symbol, Any}
    params::NamedTuple
end

function _parse_covariates(block::Expr)
    block.head == :block || error("@covariates expects a begin ... end block.")
    names = Symbol[]
    exprs = Expr[]
    for stmt in block.args
        stmt isa LineNumberNode && continue
        stmt isa Expr || error("Invalid statement in @covariates block.")
        stmt.head == :(=) || error("Only assignments are allowed in @covariates block.")
        lhs, rhs = stmt.args
        lhs isa Symbol || error("Left-hand side must be a symbol in @covariates block.")
        rhs isa Expr && rhs.head == :call || error("Right-hand side must be a constructor call in @covariates block.")
        _validate_covariate_ctor(rhs)
        rhs = _rewrite_univariate_covariate(lhs, rhs)
        push!(names, lhs)
        push!(exprs, rhs)
    end
    return names, exprs
end

function _covariate_ctor_name(fn)
    if fn === :Covariate || fn === :ConstantCovariate || fn === :DynamicCovariate
        return fn
    end
    if fn isa GlobalRef && (fn.name === :Covariate || fn.name === :ConstantCovariate || fn.name === :DynamicCovariate)
        return fn.name
    end
    if fn isa Expr && fn.head == :.
        last = fn.args[end]
        last isa QuoteNode && (last = last.value)
        if last === :Covariate || last === :ConstantCovariate || last === :DynamicCovariate
            return last
        end
    end
    return nothing
end

function _covariate_vector_ctor_name(fn)
    if fn === :CovariateVector || fn === :ConstantCovariateVector || fn === :DynamicCovariateVector
        return fn
    end
    if fn isa GlobalRef && (fn.name === :CovariateVector || fn.name === :ConstantCovariateVector || fn.name === :DynamicCovariateVector)
        return fn.name
    end
    if fn isa Expr && fn.head == :.
        last = fn.args[end]
        last isa QuoteNode && (last = last.value)
        if last === :CovariateVector || last === :ConstantCovariateVector || last === :DynamicCovariateVector
            return last
        end
    end
    return nothing
end

function _covariate_has_kwargs(rhs::Expr)
    return any(arg -> arg isa Expr && (arg.head == :kw || arg.head == :parameters), rhs.args)
end

function _covariate_collect_kwargs(rhs::Expr)
    kwargs = Expr[]
    for arg in rhs.args
        if arg isa Expr && arg.head == :kw
            push!(kwargs, arg)
        elseif arg isa Expr && arg.head == :parameters
            append!(kwargs, arg.args)
        end
    end
    return kwargs
end

function _validate_covariate_ctor(rhs::Expr)
    fn = rhs.args[1]
    ctor = _covariate_ctor_name(fn)
    if ctor === :Covariate
        has_kwargs = _covariate_has_kwargs(rhs)
        has_kwargs && error("Covariate does not accept keyword arguments; use DynamicCovariate(...; interpolation=...).")
    end
    vctor = _covariate_vector_ctor_name(fn)
    if vctor === :CovariateVector
        has_kwargs = _covariate_has_kwargs(rhs)
        has_kwargs && error("CovariateVector does not accept keyword arguments; use DynamicCovariateVector(...; interpolations=...).")
    end
    return nothing
end

function _rewrite_univariate_covariate(lhs::Symbol, rhs::Expr)
    fn = rhs.args[1]
    ctor = _covariate_ctor_name(fn)
    if ctor !== nothing
        # Disallow explicit column naming; LHS provides the column.
        has_kwargs = _covariate_has_kwargs(rhs)
        if ctor === :Covariate
            has_kwargs && error("Covariate does not accept keyword arguments; use DynamicCovariate(...; interpolation=...).")
        end
        # No positional args allowed (the macro injects the column).
        positional = [arg for arg in rhs.args[2:end] if !(arg isa Expr && (arg.head == :kw || arg.head == :parameters))]
        isempty(positional) || error("Do not pass a name to $(ctor); the LHS provides the name.")
        col = QuoteNode(lhs)
        if has_kwargs && ctor === :DynamicCovariate
            kwargs = _covariate_collect_kwargs(rhs)
            for kw in kwargs
                kw.args[1] == :column && error("Do not pass name=... to DynamicCovariate; the LHS provides the name.")
            end
            return Expr(:call, fn, col, kwargs...)
        end
        if has_kwargs && ctor === :ConstantCovariate
            kwargs = _covariate_collect_kwargs(rhs)
            for kw in kwargs
                kw.args[1] == :column && error("Do not pass name=... to ConstantCovariate; the LHS provides the name.")
            end
            return Expr(:call, fn, col, kwargs...)
        end
        return Expr(:call, fn, col)
    end
    return rhs
end

"""
    @covariates begin
        name = CovariateType(...)
        ...
    end

Compile covariate declarations into a [`Covariates`](@ref) struct.

Each statement must be an assignment `name = constructor(...)` where the constructor
is one of: [`Covariate`](@ref), [`CovariateVector`](@ref),
[`ConstantCovariate`](@ref), [`ConstantCovariateVector`](@ref),
[`DynamicCovariate`](@ref), or [`DynamicCovariateVector`](@ref).

For scalar types (`Covariate`, `ConstantCovariate`, `DynamicCovariate`), the LHS
symbol determines the data-frame column name — do not pass an explicit column argument.

The `@covariates` block is typically used inside `@Model`. It can also be used
standalone to construct a `Covariates` object directly.
"""
macro covariates(block)
    names, exprs = _parse_covariates(block)
    isempty(names) && return :(build_covariates(NamedTuple()))
    assigns = [:( $(names[i]) = $(esc(exprs[i])) ) for i in eachindex(names)]
    nt = Expr(:tuple, (Expr(:(=), names[i], names[i]) for i in eachindex(names))...)
    return quote
        $(assigns...)
        build_covariates($nt)
    end
end

function build_covariates(params::NamedTuple)
    names = Symbol[collect(keys(params))...]
    flat_names = Symbol[]
    constants = Symbol[]
    varying = Symbol[]
    dynamic = Symbol[]
    interpolations = Dict{Symbol, Any}()

    for name in names
        p = getfield(params, name)
        if p isa ConstantCovariate
            push!(constants, name)
            push!(flat_names, name)
            _ensure_no_interpolation(p, name)
        elseif p isa ConstantCovariateVector
            push!(constants, name)
            for (i, col) in enumerate(p.columns)
                push!(flat_names, Symbol(name, "_", i))
            end
            _ensure_no_interpolation(p, name)
        elseif p isa Covariate
            push!(varying, name)
            push!(flat_names, name)
        elseif p isa CovariateVector
            push!(varying, name)
            for (i, col) in enumerate(p.columns)
                push!(flat_names, Symbol(name, "_", i))
            end
        elseif p isa DynamicCovariate
            push!(varying, name)
            push!(dynamic, name)
            push!(flat_names, name)
            _ensure_interpolation(p, name)
            interpolations[name] = p.interpolation
        elseif p isa DynamicCovariateVector
            push!(varying, name)
            push!(dynamic, name)
            for (i, col) in enumerate(p.columns)
                push!(flat_names, Symbol(name, "_", i))
            end
            _ensure_interpolation(p, name)
            interpolations[name] = p.interpolations
        else
            error("Invalid covariate type for $(name).")
        end
    end

    return Covariates(names, flat_names, constants, varying, dynamic, interpolations, params)
end

function _ensure_no_interpolation(p::AbstractCovariate, name::Symbol)
    if p isa ConstantCovariate || p isa ConstantCovariateVector || p isa Covariate || p isa CovariateVector
        return nothing
    end
    return nothing
end

function _ensure_interpolation(p::AbstractCovariate, name::Symbol)
    if p isa DynamicCovariate && p.interpolation === nothing
        error("DynamicCovariate $(name) must declare interpolation.")
    end
    if p isa DynamicCovariateVector
        length(p.columns) == length(p.interpolations) || error("DynamicCovariateVector $(name) interpolation length mismatch.")
    end
    return nothing
end

function _check_interpolation(itp, ctx)
    itp isa Type || error("Interpolation must be a type from DataInterpolations.jl; got $(typeof(itp)).")
    if !(itp in ALLOWED_INTERPOLATIONS)
        allowed = join(string.(collect(ALLOWED_INTERPOLATIONS)), ", ")
        error("Invalid interpolation $(itp). Allowed: $(allowed).")
    end
    return nothing
end

"""
    finalize_covariates(covariates::Covariates, random_effects::RandomEffects) -> Covariates

Resolve the `constant_on` grouping column for each `ConstantCovariate` /
`ConstantCovariateVector` that did not specify one explicitly.

- When there is exactly one random-effect grouping column, `constant_on` defaults to it.
- When there are multiple grouping columns, `constant_on` must be explicit.
- Validates that covariates used inside random-effect distributions are declared
  `constant_on` for the correct grouping column.

This function is called automatically by `@Model` after `@covariates` and `@randomEffects`
are evaluated.
"""
function finalize_covariates(covariates::Covariates, random_effects)
    re_names = get_re_names(random_effects)
    re_groups = get_re_groups(random_effects)
    re_syms = get_re_syms(random_effects)
    re_cols = unique([getfield(re_groups, n) for n in re_names])

    used_const = Set{Symbol}()
    for re in re_names
        syms = getfield(re_syms, re)
        for s in syms
            s in covariates.constants && push!(used_const, s)
        end
    end

    params = covariates.params
    pairs = Pair{Symbol, Any}[]
    for name in covariates.names
        p = getfield(params, name)
        if p isa ConstantCovariate || p isa ConstantCovariateVector
            constant_on = _normalize_constant_on(p.constant_on)
            if isempty(constant_on)
                if length(re_cols) == 1
                    constant_on = [re_cols[1]]
                elseif length(re_cols) > 1
                    error("Constant covariate $(name) must declare constant_on when multiple random-effect grouping columns exist ($(re_cols)). Use $(name) = ConstantCovariate(...; constant_on=...) or ConstantCovariateVector(...; constant_on=...).")
                end
            end

            if name in used_const
                for re in re_names
                    group_col = getfield(re_groups, re)
                    if name in getfield(re_syms, re) && !(group_col in constant_on)
                        error("RandomEffect $(re) uses constant covariate $(name), but $(name) is not declared constant_on for group $(group_col). Add constant_on=$(group_col) (or include it in the constant_on vector).")
                    end
                end
            end

            p = _set_constant_on(p, constant_on)
        end
        push!(pairs, name => p)
    end

    return build_covariates(NamedTuple(pairs))
end
