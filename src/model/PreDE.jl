export @preDifferentialEquation
export PreDifferentialEquation
export get_prede_meta
export get_prede_builder
export get_prede_names
export get_prede_syms
export get_prede_lines

using ComponentArrays
using Distributions
using RuntimeGeneratedFunctions

RuntimeGeneratedFunctions.init(@__MODULE__)

struct PreDEMeta
    names::Vector{Symbol}
    syms::Vector{Symbol}
    lines::Vector{Expr}
end

PreDEMeta(names::Vector{Symbol}, syms::Vector{Symbol}) = PreDEMeta(names, syms, Expr[])

struct PreDEBuilder
    build::Function
end

"""
    PreDifferentialEquation

Compiled representation of a `@preDifferentialEquation` block. Stores derived variable
names, symbol dependencies, raw expression lines, and a runtime builder function.

Pre-DE variables are time-constant quantities computed from fixed effects, random
effects, and constant covariates before the ODE is solved. They are available inside
`@DifferentialEquation` and `@initialDE`.
"""
struct PreDifferentialEquation
    meta::PreDEMeta
    builder::PreDEBuilder
end

"""
    get_prede_meta(p::PreDifferentialEquation) -> PreDEMeta

Return the metadata struct (names, symbol dependencies, expression lines).
"""
get_prede_meta(p::PreDifferentialEquation) = p.meta

"""
    get_prede_builder(p::PreDifferentialEquation) -> Function

Return the builder function with signature:

    (θ::ComponentArray, η::ComponentArray, const_cov::NamedTuple,
     model_funs::NamedTuple, helpers::NamedTuple) -> NamedTuple

Returns a `NamedTuple` of computed pre-DE variable values.
"""
get_prede_builder(p::PreDifferentialEquation) = p.builder.build

"""
    get_prede_names(p::PreDifferentialEquation) -> Vector{Symbol}

Return the names of all variables defined in the `@preDifferentialEquation` block.
"""
get_prede_names(p::PreDifferentialEquation) = p.meta.names

"""
    get_prede_syms(p::PreDifferentialEquation) -> Vector{Symbol}

Return the symbols from fixed effects, random effects, and covariates that appear
in the pre-DE expressions.
"""
get_prede_syms(p::PreDifferentialEquation) = p.meta.syms

"""
    get_prede_lines(p::PreDifferentialEquation) -> Vector{Expr}

Return the raw assignment expressions from the `@preDifferentialEquation` block.
"""
get_prede_lines(p::PreDifferentialEquation) = p.meta.lines

function _prede_is_identifier(sym::Symbol)
    return Base.isidentifier(sym)
end

function _prede_call_name(f)
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

function _prede_is_mutating_assign(ex::Expr)
    if ex.head == :(=)
        lhs = ex.args[1]
        return lhs isa Expr && (lhs.head == :ref || lhs.head == :.)
    end
    head_str = string(ex.head)
    return startswith(head_str, ".") && endswith(head_str, "=")
end

function _prede_contains_mutation(ex)
    ex isa Expr || return false
    _prede_is_mutating_assign(ex) && return true
    if ex.head == :call
        fname = _prede_call_name(ex.args[1])
        fname !== nothing && endswith(String(fname), "!") && return true
    end
    for arg in ex.args
        _prede_contains_mutation(arg) && return true
    end
    return false
end

function _warn_if_mutating_prede(name::Symbol, ex::Expr)
    _prede_contains_mutation(ex) || return nothing
    @warn "Possible mutation detected in @preDifferentialEquation for $(name). Zygote may still work depending on the exact code, but mutation often breaks it. ForwardDiff usually handles mutation but can increase compile time and runtime for large models. Consider a non-mutating form if you need Zygote."
    return nothing
end

function _prede_collect_call_symbols(ex, out::Set{Symbol})
    ex isa Expr || return out
    if ex.head == :call
        f = ex.args[1]
        if f isa Symbol && _prede_is_identifier(f)
            push!(out, f)
        end
        for arg in ex.args[2:end]
            _prede_collect_call_symbols(arg, out)
        end
        return out
    elseif ex.head == :ref
        _prede_collect_call_symbols(ex.args[1], out)
        return out
    else
        for arg in ex.args
            _prede_collect_call_symbols(arg, out)
        end
        return out
    end
end

function _prede_collect_var_symbols(ex, out::Set{Symbol})
    ex isa Symbol && return (push!(out, ex); out)
    ex isa Expr || return out
    if ex.head == :call
        for arg in ex.args[2:end]
            _prede_collect_var_symbols(arg, out)
        end
        return out
    elseif ex.head == :ref
        _prede_collect_var_symbols(ex.args[1], out)
        return out
    elseif ex.head == :.
        _prede_collect_var_symbols(ex.args[1], out)
        return out
    else
        for arg in ex.args
            _prede_collect_var_symbols(arg, out)
        end
        return out
    end
end

function _prede_collect_property_bases(ex, out::Set{Symbol})
    ex isa Expr || return out
    if ex.head == :.
        base = ex.args[1]
        base isa Symbol && push!(out, base)
        _prede_collect_property_bases(base, out)
        return out
    end
    for arg in ex.args
        _prede_collect_property_bases(arg, out)
    end
    return out
end

function _prede_forbidden_symbol(ex)
    ex isa Symbol && (ex == :t || ex == :ξ) && return ex
    ex isa Expr || return nothing
    for arg in ex.args
        found = _prede_forbidden_symbol(arg)
        found === nothing || return found
    end
    return nothing
end

function _parse_prede(block::Expr)
    block.head == :block || error("@preDifferentialEquation expects a begin ... end block.")
    names = Symbol[]
    exprs = Expr[]
    lines = Expr[]
    for stmt in block.args
        stmt isa LineNumberNode && continue
        stmt isa Expr || error("Invalid statement in @preDifferentialEquation block.")
        stmt.head == :(=) || error("Only assignments are allowed in @preDifferentialEquation block.")
        lhs, rhs = stmt.args
        lhs isa Symbol || error("Left-hand side must be a symbol in @preDifferentialEquation block.")
        rhs isa LineNumberNode && error("Invalid right-hand side in @preDifferentialEquation block.")
        forbidden = _prede_forbidden_symbol(rhs)
        forbidden === nothing || error("preDifferentialEquation uses forbidden symbol $(forbidden).")
        rhs isa Expr && _warn_if_mutating_prede(lhs, rhs)
        push!(names, lhs)
        push!(exprs, rhs isa Expr ? rhs : Expr(:call, :identity, rhs))
        push!(lines, Expr(:(=), lhs, rhs))
    end
    return names, exprs, lines
end

"""
    @preDifferentialEquation begin
        name = expr
        ...
    end

Compile time-constant derived quantities into a [`PreDifferentialEquation`](@ref) struct.

Each statement must be an assignment `name = expr`. The right-hand side may reference:
- fixed effects and random effects by name,
- constant covariates (including vector fields `x.field`),
- helper functions,
- model functions (NNs, splines, soft trees).

The symbols `t` and `ξ` are forbidden (pre-DE variables are time-constant).

Pre-DE variables are computed once per individual before the ODE is integrated and
are available inside `@DifferentialEquation` and `@initialDE`.

Mutating operations trigger a warning since they may break Zygote-based AD.
"""
macro preDifferentialEquation(block)
    RuntimeGeneratedFunctions.init(__module__)
    names, exprs, lines = _parse_prede(block)
    call_syms = Set{Symbol}()
    var_syms = Set{Symbol}()
    prop_syms = Set{Symbol}()
    for ex in exprs
        _prede_collect_call_symbols(ex, call_syms)
        _prede_collect_var_symbols(ex, var_syms)
        _prede_collect_property_bases(ex, prop_syms)
    end

    delete!(var_syms, :fixed_effects)
    delete!(var_syms, :random_effects)
    delete!(var_syms, :constant_features_i)
    delete!(var_syms, :model_funs)
    delete!(var_syms, :helper_functions)

    call_syms = Set([s for s in call_syms if !(isdefined(Base, s) || isdefined(Distributions, s) || isdefined(@__MODULE__, s))])
    var_syms = Set([s for s in var_syms if Base.isidentifier(s)])
    skip_vars = Set([:Inf, :NaN, :nothing, :missing, :true, :false])
    var_syms = Set([s for s in var_syms if !(s in skip_vars)])

    prop_syms_expr = Expr(:call, :Set, Expr(:vect, QuoteNode.(collect(prop_syms))...))
    binds_vars = [
        quote
            if $(QuoteNode(sym)) in $prop_syms_expr
                if hasproperty(constant_features_i, $(QuoteNode(sym)))
                    $(sym) = getproperty(constant_features_i, $(QuoteNode(sym)))
                end
            else
                if hasproperty(constant_features_i, $(QuoteNode(sym)))
                    $(sym) = getproperty(constant_features_i, $(QuoteNode(sym)))
                elseif hasproperty(random_effects, $(QuoteNode(sym)))
                    $(sym) = getproperty(random_effects, $(QuoteNode(sym)))
                elseif hasproperty(fixed_effects, $(QuoteNode(sym)))
                    $(sym) = getproperty(fixed_effects, $(QuoteNode(sym)))
                end
            end
        end for sym in var_syms
    ]

    binds_funs = [
        quote
            if hasproperty(model_funs, $(QuoteNode(sym)))
                $(sym) = getproperty(model_funs, $(QuoteNode(sym)))
            elseif hasproperty(helper_functions, $(QuoteNode(sym)))
                $(sym) = getproperty(helper_functions, $(QuoteNode(sym)))
            end
        end for sym in call_syms
    ]

    assigns = [:( $(names[i]) = $(exprs[i]) ) for i in eachindex(names)]
    ret_expr = Expr(:tuple, (Expr(:(=), names[i], names[i]) for i in eachindex(names))...)

    func_expr = :(function (fixed_effects::ComponentArray,
                            random_effects::ComponentArray,
                            constant_features_i::NamedTuple,
                            model_funs::NamedTuple,
                            helper_functions::NamedTuple)
        $(binds_vars...)
        $(binds_funs...)
        $(assigns...)
        $ret_expr
    end)

    names_expr = Expr(:vect, QuoteNode.(names)...)
    syms_expr = Expr(:vect, QuoteNode.(collect(var_syms))...)
    lines_expr = Expr(:vect, QuoteNode.(lines)...)
    return quote
        prede_fn = RuntimeGeneratedFunction(@__MODULE__, @__MODULE__, $(QuoteNode(func_expr)))
        function prede_wrapper(fixed_effects::ComponentArray,
                               random_effects::ComponentArray,
                               constant_features_i::NamedTuple,
                               model_funs::NamedTuple)
            return prede_fn(fixed_effects, random_effects, constant_features_i, model_funs, NamedTuple())
        end
        function prede_wrapper(fixed_effects::ComponentArray,
                               random_effects::ComponentArray,
                               constant_features_i::NamedTuple,
                               model_funs::NamedTuple,
                               helper_functions::NamedTuple)
            return prede_fn(fixed_effects, random_effects, constant_features_i, model_funs, helper_functions)
        end
        meta = PreDEMeta($names_expr, $syms_expr, $lines_expr)
        builder = PreDEBuilder(prede_wrapper)
        PreDifferentialEquation(meta, builder)
    end
end
