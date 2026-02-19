export @DifferentialEquation
export DifferentialEquation
export get_de_meta
export get_de_builders
export get_de_states
export get_de_signals
export get_de_lines
export get_de_f
export get_de_f!
export get_de_compiler
export get_de_accessors_builder
export DEStateAccessor
export DESignalAccessor
export DEStaticContext
export DEParams
export build_de_params
export DETunable

using ComponentArrays
using RuntimeGeneratedFunctions
import SciMLStructures: isscimlstructure, ismutablescimlstructure, canonicalize, hasportion
using SciMLStructures
using Functors
import SciMLSensitivity: recursive_copyto!, recursive_add!, recursive_sub!, recursive_neg!, allocate_vjp
import StaticArrays

RuntimeGeneratedFunctions.init(@__MODULE__)

struct DifferentialEquationMeta
    state_names::Vector{Symbol}
    signal_names::Vector{Symbol}
    var_syms::Vector{Symbol}
    fun_syms::Vector{Symbol}
    lines::Vector{Expr}
end

DifferentialEquationMeta(state_names::Vector{Symbol},
                         signal_names::Vector{Symbol},
                         var_syms::Vector{Symbol},
                         fun_syms::Vector{Symbol}) = DifferentialEquationMeta(state_names, signal_names, var_syms, fun_syms, Expr[])

struct DifferentialEquationBuilders
    compile::Function
    f!::Function
    f::Function
end

"""
    DifferentialEquation

Compiled representation of a `@DifferentialEquation` block. Stores state and signal
names, in-place/out-of-place ODE functions, a parameter compiler, and an accessor
builder for retrieving state/signal values from a solution object.
"""
struct DifferentialEquation
    meta::DifferentialEquationMeta
    builders::DifferentialEquationBuilders
    accessors_builder::Function
end

"""
    get_de_meta(de::DifferentialEquation) -> DifferentialEquationMeta

Return the metadata struct (state names, signal names, variable and function symbols).
"""
get_de_meta(de::DifferentialEquation) = de.meta

"""
    get_de_builders(de::DifferentialEquation) -> DifferentialEquationBuilders

Return the builders struct (compile, f!, f functions).
"""
get_de_builders(de::DifferentialEquation) = de.builders

"""
    get_de_states(de::DifferentialEquation) -> Vector{Symbol}

Return the names of all ODE state variables (left-hand sides of `D(x) ~ ...`).
"""
get_de_states(de::DifferentialEquation) = de.meta.state_names

"""
    get_de_signals(de::DifferentialEquation) -> Vector{Symbol}

Return the names of all derived signals (left-hand sides of `s(t) = ...`).
"""
get_de_signals(de::DifferentialEquation) = de.meta.signal_names

"""
    get_de_lines(de::DifferentialEquation) -> Vector{Expr}

Return the raw equation expressions from the `@DifferentialEquation` block.
"""
get_de_lines(de::DifferentialEquation) = de.meta.lines

"""
    get_de_f!(de::DifferentialEquation) -> Function

Return the in-place ODE function with signature `f!(du, u, p, t)` compatible with
SciML/OrdinaryDiffEq.jl, where `p` is a [`DEParams`](@ref) struct.
"""
get_de_f!(de::DifferentialEquation) = de.builders.f!

"""
    get_de_f(de::DifferentialEquation) -> Function

Return the out-of-place ODE function with signature `f(u, p, t)` compatible with
SciML/OrdinaryDiffEq.jl, where `p` is a [`DEParams`](@ref) struct.
"""
get_de_f(de::DifferentialEquation) = de.builders.f

"""
    get_de_compiler(de::DifferentialEquation) -> Function

Return the parameter compiler function with signature `compile(p) -> (vars=..., funs=...)`.
It extracts the named variable and function bindings from a raw parameter `NamedTuple`
and caches them inside a `DEParams`.
"""
get_de_compiler(de::DifferentialEquation) = de.builders.compile

"""
    get_de_accessors_builder(de::DifferentialEquation) -> Function

Return the accessor builder with signature `build(sol, pc) -> NamedTuple`.
The returned `NamedTuple` maps each state name to a [`DEStateAccessor`](@ref) and
each signal name to a [`DESignalAccessor`](@ref), both callable as `accessor(t)`.
"""
get_de_accessors_builder(de::DifferentialEquation) = de.accessors_builder

"""
    DEStateAccessor{S, I}

Callable struct that evaluates an ODE state at a given time `t` by indexing into a
SciML solution object: `accessor(t)` returns the state value at time `t`.

Returned by [`get_de_accessors_builder`](@ref) and passed to formula evaluation.
"""
struct DEStateAccessor{S, I}
    sol::S
    idx::I
end

@inline (a::DEStateAccessor)(t) = a.sol(t; idxs=a.idx)

"""
    DESignalAccessor{S, P, F}

Callable struct that evaluates a derived ODE signal at a given time `t`:
`accessor(t)` recomputes the signal expression using the solution and pre-compiled
parameters.

Returned by [`get_de_accessors_builder`](@ref) and passed to formula evaluation.
"""
struct DESignalAccessor{S, P, F}
    sol::S
    pc::P
    f::F
end

@inline (a::DESignalAccessor)(t) = a.f(a.sol, a.pc, t)

"""
    DEStaticContext{B, I}

Immutable context struct holding all time-constant inputs needed to evaluate an ODE
(constant covariates, varying covariates, helpers, model functions, pre-DE builder,
inverse parameter transform, DE compiler, and tunable-mode flag).

Embedded inside [`DEParams`](@ref) and treated as a non-differentiable leaf by Functors.
"""
struct DEStaticContext{B, I}
    constant_covariates::NamedTuple
    varying_covariates::NamedTuple
    helpers::NamedTuple
    model_funs::NamedTuple
    prede_builder::B
    inverse_transform::I
    de_compiler::Function
    tunable::Symbol
end

"""
    DEParams{T, S} <: AbstractVector{T}

SciMLStructures-compatible parameter struct for an ODE problem. Holds the fixed-effects
`ComponentArray` `θ`, random-effects `ComponentArray` `η`, a [`DEStaticContext`](@ref),
and compiled variable/function bindings.

Behaves as a flat `AbstractVector` whose elements are the concatenation of `θ` and `η`,
allowing SciML adjoint sensitivity methods to differentiate through it.

Construct via [`build_de_params`](@ref).
"""
struct DEParams{T, S} <: AbstractVector{T}
    θ::ComponentArray
    η::ComponentArray
    static::S
    compiled::NamedTuple
end

"""
    DETunable{T, A, B} <: AbstractVector{T}

View into the tunable (differentiable) portion of a [`DEParams`](@ref) struct.
The `mode` field controls which parameters are exposed:
- `:θ` — only fixed effects,
- `:η` — only random effects,
- `:both` — concatenation of both.

Used internally by the SciMLStructures `canonicalize` interface.
"""
struct DETunable{T, A, B} <: AbstractVector{T}
    θ::A
    η::B
    mode::Symbol
end

DETunable(θ, η, mode::Symbol) = DETunable{eltype(θ), typeof(θ), typeof(η)}(θ, η, mode)

Functors.@leaf DEStaticContext
Functors.@leaf DEParams
Functors.functor(p::DEParams) = ((), _ -> p)
Functors.functor(p::DEStaticContext) = ((), _ -> p)

recursive_copyto!(y::AbstractArray, x::DEParams) = copyto!(y, vcat(collect(x.θ), collect(x.η)))
recursive_copyto!(y::DEParams, x::DEParams) = (copyto!(y.θ, x.θ); copyto!(y.η, x.η); y)
recursive_neg!(x::DEParams) = (x.θ .*= -1; x.η .*= -1; x)
recursive_add!(y::DEParams, x::DEParams) = (y.θ .+= x.θ; y.η .+= x.η; y)
recursive_sub!(y::DEParams, x::DEParams) = (y.θ .-= x.θ; y.η .-= x.η; y)
allocate_vjp(λ::AbstractArray, x::DEParams) = fill!(similar(λ, length(x)), zero(eltype(λ)))
allocate_vjp(x::DEParams) = zero(vcat(collect(x.θ), collect(x.η)))

function Base.getproperty(p::DEParams, s::Symbol)
    if s === :vars
        return getfield(p, :compiled).vars
    elseif s === :funs
        return getfield(p, :compiled).funs
    end
    return getfield(p, s)
end

Base.propertynames(::DEParams) = (:θ, :η, :static, :compiled, :vars, :funs)

Base.IndexStyle(::Type{<:DEParams}) = IndexLinear()
Base.eltype(p::DEParams) = eltype(p.θ)
Base.size(p::DEParams) = (length(p.θ) + length(p.η),)
Base.length(p::DEParams) = length(p.θ) + length(p.η)
Base.axes(p::DEParams) = (Base.OneTo(length(p)),)
Base.getindex(p::DEParams, i::Int) = i <= length(p.θ) ? p.θ[i] : p.η[i - length(p.θ)]
Base.iterate(p::DEParams, state...) = iterate(vcat(collect(p.θ), collect(p.η)), state...)

Base.size(t::DETunable) = t.mode == :θ ? (length(t.θ),) :
                          t.mode == :η ? (length(t.η),) :
                          (length(t.θ) + length(t.η),)
Base.length(t::DETunable) = t.mode == :θ ? length(t.θ) :
                            t.mode == :η ? length(t.η) :
                            length(t.θ) + length(t.η)
Base.axes(t::DETunable) = (Base.OneTo(length(t)),)
Base.getindex(t::DETunable, i::Int) = t.mode == :θ ? t.θ[i] :
                                      t.mode == :η ? t.η[i] :
                                      (i <= length(t.θ) ? t.θ[i] : t.η[i - length(t.θ)])
Base.iterate(t::DETunable, state...) = iterate(t.mode == :θ ? collect(t.θ) :
                                              t.mode == :η ? collect(t.η) :
                                              vcat(collect(t.θ), collect(t.η)), state...)

function _de_build_compiled(θ::ComponentArray, η::ComponentArray, static::DEStaticContext)
    fe_un = static.inverse_transform(θ)
    prede = static.prede_builder(fe_un, η, static.constant_covariates, static.model_funs, static.helpers)
    raw = (; fixed_effects = fe_un,
            random_effects = η,
            constant_covariates = static.constant_covariates,
            varying_covariates = static.varying_covariates,
            helpers = static.helpers,
            model_funs = static.model_funs,
            preDE = prede)
    return static.de_compiler(raw)
end

"""
    build_de_params(de::DifferentialEquation, θ::ComponentArray; kwargs...) -> DEParams

Construct a [`DEParams`](@ref) struct suitable for passing to an ODE solver.

# Arguments
- `de::DifferentialEquation`: the compiled DE block.
- `θ::ComponentArray`: fixed-effects parameter vector on the transformed scale.

# Keyword Arguments
- `random_effects::ComponentArray = ComponentArray(NamedTuple())`: random-effects vector.
- `constant_covariates::NamedTuple = NamedTuple()`: constant covariate values.
- `varying_covariates::NamedTuple = NamedTuple()`: dynamic covariate interpolants.
- `helpers::NamedTuple = NamedTuple()`: helper functions.
- `model_funs::NamedTuple = NamedTuple()`: model functions (NNs, splines, etc.).
- `prede_builder`: pre-DE builder function; defaults to a no-op returning `NamedTuple()`.
- `inverse_transform`: inverse parameter transform; defaults to `identity`.
- `tunable::Symbol = :both`: which parameters are tunable (`:θ`, `:η`, or `:both`).
"""
function build_de_params(de::DifferentialEquation,
                         θ::ComponentArray;
                         random_effects::ComponentArray=ComponentArray(NamedTuple()),
                         constant_covariates::NamedTuple=NamedTuple(),
                         varying_covariates::NamedTuple=NamedTuple(),
                         helpers::NamedTuple=NamedTuple(),
                         model_funs::NamedTuple=NamedTuple(),
                         prede_builder=(fe, re, consts, model_funs, helpers) -> NamedTuple(),
                         inverse_transform=identity,
                         tunable::Symbol=:both)
    static = DEStaticContext(constant_covariates, varying_covariates,
                             helpers, model_funs, prede_builder, inverse_transform,
                             get_de_compiler(de), tunable)
    tunable in (:θ, :η, :both) || error("tunable must be :θ, :η, or :both.")
    compiled = _de_build_compiled(θ, random_effects, static)
    return DEParams{eltype(θ), typeof(static)}(θ, random_effects, static, compiled)
end

isscimlstructure(::DEParams) = true
isscimlstructure(::Type{<:DEParams}) = true
isscimlstructure(::Type{DEParams{S}}) where {S} = true
ismutablescimlstructure(::DEParams) = false

hasportion(::SciMLStructures.Tunable, ::DEParams) = true
hasportion(::SciMLStructures.Constants, ::DEParams) = false
hasportion(::SciMLStructures.Caches, ::DEParams) = false
hasportion(::SciMLStructures.Discrete, ::DEParams) = false
hasportion(::SciMLStructures.Input, ::DEParams) = false
hasportion(::SciMLStructures.Initials, ::DEParams) = false

function canonicalize(::SciMLStructures.Tunable, p::DEParams)
    tunables = DETunable(p.θ, p.η, getfield(p.static, :tunable))
    vals = collect(tunables)
    repack = function (new_vals)
        mode = getfield(p.static, :tunable)
        if mode == :θ
            θnew = ComponentArray(new_vals, getaxes(p.θ))
            ηnew = p.η
        elseif mode == :η
            θnew = p.θ
            ηnew = ComponentArray(new_vals, getaxes(p.η))
        else
            nθ = length(p.θ)
            θnew = ComponentArray(new_vals[1:nθ], getaxes(p.θ))
            ηnew = ComponentArray(new_vals[(nθ + 1):end], getaxes(p.η))
        end
        compiled = _de_build_compiled(θnew, ηnew, p.static)
        return DEParams(θnew, ηnew, p.static, compiled)
    end
    return vals, repack, false
end

function canonicalize(::SciMLStructures.Constants, p::DEParams)
    empty = similar(collect(p.θ), 0)
    repack = _ -> p
    return empty, repack, false
end

function canonicalize(::SciMLStructures.Caches, p::DEParams)
    empty = similar(collect(p.θ), 0)
    repack = _ -> p
    return empty, repack, false
end

function canonicalize(::SciMLStructures.Discrete, p::DEParams)
    empty = similar(collect(p.θ), 0)
    repack = _ -> p
    return empty, repack, false
end

function canonicalize(::SciMLStructures.Input, p::DEParams)
    empty = similar(collect(p.θ), 0)
    repack = _ -> p
    return empty, repack, false
end

function canonicalize(::SciMLStructures.Initials, p::DEParams)
    empty = similar(collect(p.θ), 0)
    repack = _ -> p
    return empty, repack, false
end

function _de_is_identifier(sym::Symbol)
    return Base.isidentifier(sym)
end

function _de_call_name(f)
    if f isa Symbol
        return f
    elseif f isa GlobalRef
        return f.name
    elseif f isa Expr && f.head == :.
        last = f.args[end]
        last isa QuoteNode && (last = last.value)
        return last isa Symbol ? last : nothing
    end
    return nothing
end

function _de_collect_call_symbols(ex, out::Set{Symbol})
    ex isa Expr || return out
    if ex.head == :call
        f = ex.args[1]
        if f isa Symbol && _de_is_identifier(f)
            push!(out, f)
        end
        for arg in ex.args[2:end]
            _de_collect_call_symbols(arg, out)
        end
        return out
    elseif ex.head == :ref
        _de_collect_call_symbols(ex.args[1], out)
        return out
    else
        for arg in ex.args
            _de_collect_call_symbols(arg, out)
        end
        return out
    end
end

function _de_collect_var_symbols(ex, out::Set{Symbol})
    ex isa Symbol && return (push!(out, ex); out)
    ex isa Expr || return out
    if ex.head == :call
        for arg in ex.args[2:end]
            _de_collect_var_symbols(arg, out)
        end
        return out
    elseif ex.head == :ref
        _de_collect_var_symbols(ex.args[1], out)
        return out
    elseif ex.head == :.
        _de_collect_var_symbols(ex.args[1], out)
        return out
    else
        for arg in ex.args
            _de_collect_var_symbols(arg, out)
        end
        return out
    end
end

function _de_collect_property_bases(ex, out::Set{Symbol})
    ex isa Expr || return out
    if ex.head == :.
        base = ex.args[1]
        base isa Symbol && push!(out, base)
        _de_collect_property_bases(base, out)
        return out
    end
    for arg in ex.args
        _de_collect_property_bases(arg, out)
    end
    return out
end

function _de_collect_symbols(ex, out::Set{Symbol})
    ex isa Symbol && return (push!(out, ex); out)
    ex isa Expr || return out
    for arg in ex.args
        _de_collect_symbols(arg, out)
    end
    return out
end

function _de_signal_used_bare(ex, name::Symbol)
    ex isa Symbol && return ex == name
    ex isa Expr || return false
    if ex.head == :call && ex.args[1] == name
        # Allowed call: name(t) or name(ξ); anything else is treated as bare usage.
        return !(length(ex.args) == 2 && (ex.args[2] == :t || ex.args[2] == :ξ))
    end
    for arg in ex.args
        _de_signal_used_bare(arg, name) && return true
    end
    return false
end

function _de_collect_time_calls(ex, out::Set{Symbol})
    ex isa Expr || return out
    if ex.head == :call
        f = ex.args[1]
        if f isa Symbol && length(ex.args) == 2 && (ex.args[2] == :t || ex.args[2] == :ξ)
            push!(out, f)
        end
        for arg in ex.args[2:end]
            _de_collect_time_calls(arg, out)
        end
        return out
    else
        for arg in ex.args
            _de_collect_time_calls(arg, out)
        end
        return out
    end
end

function _de_is_signal_call(ex, name::Symbol)
    ex isa Expr || return false
    ex.head == :call || return false
    ex.args[1] == name || return false
    length(ex.args) == 2 || return false
    return ex.args[2] == :t || ex.args[2] == :ξ
end

function _de_rewrite_expr(ex, state_map::Dict{Symbol, Int}, var_syms::Set{Symbol}, fun_syms::Set{Symbol})
    ex isa Expr || return ex
    if ex.head == :call
        f = ex.args[1]
        new_args = [_de_rewrite_expr(arg, state_map, var_syms, fun_syms) for arg in ex.args[2:end]]
        if f isa Symbol && f in fun_syms
            return Expr(:call, Expr(:., :funs, QuoteNode(f)), new_args...)
        end
        return Expr(:call, f, new_args...)
    elseif ex.head == :.
        base = _de_rewrite_expr(ex.args[1], state_map, var_syms, fun_syms)
        return Expr(:., base, ex.args[2])
    else
        return Expr(ex.head, map(arg -> _de_rewrite_expr(arg, state_map, var_syms, fun_syms), ex.args)...)
    end
end

function _de_rewrite_symbol(ex, state_map::Dict{Symbol, Int}, var_syms::Set{Symbol})
    ex isa Symbol || return ex
    if haskey(state_map, ex)
        return ex
    elseif ex in var_syms
        return Expr(:., :vars, QuoteNode(ex))
    end
    return ex
end

function _de_rewrite_all(ex, state_map::Dict{Symbol, Int}, var_syms::Set{Symbol}, fun_syms::Set{Symbol})
    ex isa Symbol && return _de_rewrite_symbol(ex, state_map, var_syms)
    ex isa Expr || return ex
    if ex.head == :call
        f = ex.args[1]
        new_args = [_de_rewrite_all(arg, state_map, var_syms, fun_syms) for arg in ex.args[2:end]]
        if f isa Symbol && f in fun_syms
            return Expr(:call, Expr(:., :funs, QuoteNode(f)), new_args...)
        end
        return Expr(:call, f, new_args...)
    elseif ex.head == :.
        base = _de_rewrite_all(ex.args[1], state_map, var_syms, fun_syms)
        return Expr(:., base, ex.args[2])
    else
        return Expr(ex.head, map(arg -> _de_rewrite_all(arg, state_map, var_syms, fun_syms), ex.args)...)
    end
end

function _de_replace_signal_calls(ex, names::Set{Symbol})
    ex isa Expr || return ex
    if ex.head == :call
        f = ex.args[1]
        if f isa Symbol && f in names && length(ex.args) == 2 && (ex.args[2] == :t || ex.args[2] == :ξ)
            return f
        end
    end
    return Expr(ex.head, map(arg -> _de_replace_signal_calls(arg, names), ex.args)...)
end

function _de_rewrite_small_vectors(ex)
    ex isa Expr || return ex
    if ex.head == :vect
        if length(ex.args) <= 8
            new_args = map(_de_rewrite_small_vectors, ex.args)
            return Expr(:call, GlobalRef(StaticArrays, :SVector), new_args...)
        end
    end
    return Expr(ex.head, map(arg -> _de_rewrite_small_vectors(arg), ex.args)...)
end

function _parse_de(block::Expr)
    block.head == :block || error("@DifferentialEquation expects a begin ... end block.")
    state_names = Symbol[]
    rhs_exprs = Any[]
    signal_names = Symbol[]
    signal_exprs = Any[]
    line_exprs = Expr[]
    seen = Set{Symbol}()

    for stmt in block.args
        stmt isa LineNumberNode && continue
        stmt isa Expr || error("Invalid statement in @DifferentialEquation block.")

        if stmt.head == :(=)
            lhs, rhs = stmt.args
            lhs isa Expr && lhs.head == :call || error("Derived signals must be function-like: s(t)=... .")
            name = lhs.args[1]
            name isa Symbol || error("Derived signal name must be a Symbol.")
            length(lhs.args) == 2 || error("Derived signal must be of form s(t) = expr.")
            arg = lhs.args[2]
            (arg == :t || arg == :ξ) || error("Derived signal argument must be t or ξ.")
            name in seen && error("Duplicate DE name: $(name).")
            push!(seen, name)
            push!(signal_names, name)
            push!(signal_exprs, rhs)
            push!(line_exprs, Expr(:(=), lhs, rhs))
            continue
        end

        stmt.head == :call && stmt.args[1] == :~ || error("Only D(x) ~ expr or s(t)=expr are allowed in @DifferentialEquation.")
        lhs, rhs = stmt.args[2], stmt.args[3]
        lhs isa Expr && lhs.head == :call && lhs.args[1] == :D || error("Left-hand side must be D(state).")
        length(lhs.args) == 2 || error("Left-hand side must be D(state) with a single symbol.")
        state = lhs.args[2]
        state isa Symbol || error("State name must be a Symbol.")
        state in seen && error("Duplicate DE name: $(state).")
        push!(seen, state)
        push!(state_names, state)
        push!(rhs_exprs, rhs)
        push!(line_exprs, Expr(:call, :~, lhs, rhs))
    end

    return state_names, rhs_exprs, signal_names, signal_exprs, line_exprs
end

"""
    @DifferentialEquation begin
        D(state) ~ rhs_expr
        signal(t) = signal_expr
        ...
    end

Compile an ODE system into a [`DifferentialEquation`](@ref) struct.

Two statement forms are supported:
- `D(state) ~ rhs`: defines a state variable whose time derivative equals `rhs`.
- `signal(t) = expr`: defines a derived signal computed from states and parameters.

Symbols in right-hand sides are resolved from (in order): pre-DE variables, random
effects, fixed effects, constant covariates, dynamic covariates (called as `w(t)`),
model functions, and helper functions. Varying (non-dynamic) covariates are not allowed
inside the DE.

Small vector literals (up to length 8) are automatically replaced with `StaticArrays.SVector`
for allocation-free ODE evaluation.

Must be paired with `@initialDE` when used inside `@Model`.
"""
macro DifferentialEquation(block)
    RuntimeGeneratedFunctions.init(__module__)
    state_names, rhs_exprs, signal_names, signal_exprs, line_exprs = _parse_de(block)
    signal_set = Set(signal_names)

    # Disallow derived signal names used without (t) in RHS.
    for rhs in rhs_exprs
        for s in signal_set
            _de_signal_used_bare(rhs, s) && error("Derived signal $(s) must be called as $(s)(t) in @DifferentialEquation.")
        end
    end

    # Replace derived signal calls s(t) with local symbols s.
    rhs_rewritten = [_de_replace_signal_calls(rhs, signal_set) for rhs in rhs_exprs]
    signal_rewritten = [_de_replace_signal_calls(rhs, signal_set) for rhs in signal_exprs]

    # Rewrite small vector literals to StaticArrays.SVector for allocation-free hot loops.
    rhs_rewritten = [_de_rewrite_small_vectors(rhs) for rhs in rhs_rewritten]
    signal_rewritten = [_de_rewrite_small_vectors(rhs) for rhs in signal_rewritten]

    call_syms = Set{Symbol}()
    var_syms = Set{Symbol}()
    prop_syms = Set{Symbol}()
    time_call_syms = Set{Symbol}()
    for ex in vcat(rhs_rewritten, signal_rewritten)
        _de_collect_call_symbols(ex, call_syms)
        _de_collect_var_symbols(ex, var_syms)
        _de_collect_property_bases(ex, prop_syms)
        _de_collect_time_calls(ex, time_call_syms)
    end

    delete!(var_syms, :t)
    delete!(var_syms, :ξ)
    delete!(var_syms, :u)
    delete!(var_syms, :du)

    call_syms = Set([s for s in call_syms if !(isdefined(Base, s) || isdefined(@__MODULE__, s))])
    var_syms = Set([s for s in var_syms if Base.isidentifier(s)])
    skip_vars = Set([:Inf, :NaN, :nothing, :missing, :true, :false])
    var_syms = Set([s for s in var_syms if !(s in skip_vars)])

    var_syms_no_states = Set([s for s in var_syms if !(s in state_names) && !(s in signal_set)])
    fun_syms = Set([s for s in call_syms if !(s in signal_set)])
    state_map = Dict{Symbol, Int}((state_names[i] => i) for i in eachindex(state_names))
    rhs_fast = [_de_rewrite_all(ex, state_map, var_syms_no_states, fun_syms) for ex in rhs_rewritten]
    signal_fast = [_de_rewrite_all(ex, state_map, var_syms_no_states, fun_syms) for ex in signal_rewritten]

    compile_vars = [
        quote
            if hasproperty(preDE, $(QuoteNode(sym)))
                getproperty(preDE, $(QuoteNode(sym)))
            elseif hasproperty(random_effects, $(QuoteNode(sym)))
                getproperty(random_effects, $(QuoteNode(sym)))
            elseif hasproperty(fixed_effects, $(QuoteNode(sym)))
                getproperty(fixed_effects, $(QuoteNode(sym)))
            elseif hasproperty(constant_covariates, $(QuoteNode(sym)))
                getproperty(constant_covariates, $(QuoteNode(sym)))
            else
                error("Unknown symbol $(string($(QuoteNode(sym)))) in DifferentialEquation.")
            end
        end for sym in var_syms_no_states
    ]
    vars_nt = Expr(:call, Expr(:curly, :NamedTuple, Expr(:tuple, QuoteNode.(collect(var_syms_no_states))...)),
                   Expr(:tuple, compile_vars...))

    compile_funs = [
        quote
            if hasproperty(varying_covariates, $(QuoteNode(sym)))
                getproperty(varying_covariates, $(QuoteNode(sym)))
            elseif hasproperty(model_funs, $(QuoteNode(sym)))
                getproperty(model_funs, $(QuoteNode(sym)))
            elseif hasproperty(helper_functions, $(QuoteNode(sym)))
                getproperty(helper_functions, $(QuoteNode(sym)))
            else
                error("Unknown function $(string($(QuoteNode(sym)))) in DifferentialEquation.")
            end
        end for sym in fun_syms
    ]
    funs_nt = Expr(:call, Expr(:curly, :NamedTuple, Expr(:tuple, QuoteNode.(collect(fun_syms))...)),
                   Expr(:tuple, compile_funs...))

    compile_expr = :(function (p)
        fixed_effects = p.fixed_effects
        random_effects = p.random_effects
        constant_covariates = p.constant_covariates
        varying_covariates = p.varying_covariates
        helper_functions = p.helpers
        model_funs = p.model_funs
        preDE = p.preDE
        vars = $vars_nt
        funs = $funs_nt
        return (vars=vars, funs=funs)
    end)

    state_binds = [:( $(state_names[i]) = u[$i] ) for i in eachindex(state_names)]
    signal_assigns = [:( $(signal_names[i]) = $(signal_fast[i]) ) for i in eachindex(signal_names)]
    du_assigns = [:( du[$i] = $(rhs_fast[i]) ) for i in eachindex(state_names)]
    f_expr = Expr(:vect, rhs_fast...)

    f!_expr = :(function (du::AbstractVector,
                           u::AbstractVector,
                           p,
                           t)
        vars = p.vars
        funs = p.funs
        $(state_binds...)
        $(signal_assigns...)
        $(du_assigns...)
        return nothing
    end)

    f_expr = :(function (u::AbstractVector, p, t)
        vars = p.vars
        funs = p.funs
        $(state_binds...)
        $(signal_assigns...)
        return $f_expr
    end)

    state_sol_binds = [:( $(state_names[i]) = sol(t; idxs=$i) ) for i in eachindex(state_names)]
    signal_fn_exprs = [
        :(function (sol, pc, t)
                vars = pc.vars
                funs = pc.funs
                $(state_sol_binds...)
                $(signal_assigns[1:i]...)
                return $(signal_names[i])
            end) for i in eachindex(signal_names)
    ]
    accessor_names = vcat(state_names, signal_names)
    accessor_vals = vcat(
        [:(DEStateAccessor(sol, $i)) for i in eachindex(state_names)],
        [:(DESignalAccessor(sol, pc, signal_fns[$i])) for i in eachindex(signal_names)]
    )
    accessors_nt = Expr(:call,
        Expr(:curly, :NamedTuple, Expr(:tuple, QuoteNode.(accessor_names)...)),
        Expr(:tuple, accessor_vals...))
    accessors_fn_sym = gensym(:de_accessors_)
    accessors_expr = :(function $(accessors_fn_sym)(sol, pc)
        return $accessors_nt
    end)

    state_names_expr = Expr(:vect, QuoteNode.(state_names)...)
    signal_names_expr = Expr(:vect, QuoteNode.(signal_names)...)
    lines_expr = Expr(:vect, QuoteNode.(line_exprs)...)
    return quote
        compile_rgf = RuntimeGeneratedFunction(@__MODULE__, @__MODULE__, $(QuoteNode(compile_expr)))
        f!_rgf = RuntimeGeneratedFunction(@__MODULE__, @__MODULE__, $(QuoteNode(f!_expr)))
        f_rgf = RuntimeGeneratedFunction(@__MODULE__, @__MODULE__, $(QuoteNode(f_expr)))
        meta = DifferentialEquationMeta($state_names_expr, $signal_names_expr,
                                        $(Expr(:vect, map(QuoteNode, collect(var_syms_no_states))...)),
                                        $(Expr(:vect, map(QuoteNode, collect(fun_syms))...)),
                                        $lines_expr)
        builders = DifferentialEquationBuilders(compile_rgf, f!_rgf, f_rgf)
        signal_fns = ($( [:(RuntimeGeneratedFunction(@__MODULE__, @__MODULE__, $(QuoteNode(signal_fn_exprs[i])))) for i in eachindex(signal_fn_exprs)]... ),)
        $(accessors_expr)
        DifferentialEquation(meta, builders, $(accessors_fn_sym))
    end
end
