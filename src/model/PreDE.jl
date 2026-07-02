export @preDifferentialEquation
export get_prede_builder
export PreDifferentialEquation
export get_prede_meta
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

function _warn_if_mutating_prede(name::Symbol, ex::Expr)
    _macro_contains_mutation(ex) || return nothing
    @warn "Possible mutation detected in @preDifferentialEquation for $(name). ForwardDiff usually handles mutation but it can increase compile time and runtime for large models, and may break some reverse-mode AD backends. Consider a non-mutating form."
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
        stmt.head == :(=) ||
            error("Only assignments are allowed in @preDifferentialEquation block.")
        lhs, rhs = stmt.args
        lhs isa Symbol ||
            error("Left-hand side must be a symbol in @preDifferentialEquation block.")
        rhs isa LineNumberNode &&
            error("Invalid right-hand side in @preDifferentialEquation block.")
        forbidden = _macro_forbidden_symbol(rhs)
        forbidden === nothing ||
            error("preDifferentialEquation uses forbidden symbol $(forbidden).")
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

Mutating operations trigger a warning since they may break some reverse-mode AD backends.
"""
macro preDifferentialEquation(block)
    RuntimeGeneratedFunctions.init(__module__)
    names, exprs, lines = _parse_prede(block)
    call_syms = Set{Symbol}()
    var_syms = Set{Symbol}()
    prop_syms = Set{Symbol}()
    for ex in exprs
        _macro_collect_call_symbols(ex, call_syms)
        _macro_collect_var_symbols(ex, var_syms)
        _macro_collect_property_bases(ex, prop_syms)
    end

    delete!(var_syms, :fixed_effects)
    delete!(var_syms, :random_effects)
    delete!(var_syms, :constant_features_i)
    delete!(var_syms, :model_funs)
    delete!(var_syms, :helper_functions)

    call_syms = Set([s
                     for s in call_syms
                     if !(isdefined(Base, s) || isdefined(Distributions, s) ||
                          isdefined(@__MODULE__, s))])
    var_syms = Set([s for s in var_syms if Base.isidentifier(s)])
    skip_vars = Set([:Inf, :NaN, :nothing, :missing, :true, :false])
    var_syms = Set([s for s in var_syms if !(s in skip_vars)])

    # `sym in prop_syms` is decidable at macro-expansion time — emit the chosen
    # branch directly. (The old code carried the membership test into the generated
    # function as `sym in Set([...])`, allocating a fresh Set on every builder
    # invocation; `calculate_prede` runs per individual per objective evaluation.
    # Mirrors the identical fix in RandomEffects.jl.)
    binds_vars = [if sym in prop_syms
                      quote
                          if hasproperty(constant_features_i, $(QuoteNode(sym)))
                              $(sym) = getproperty(constant_features_i, $(QuoteNode(sym)))
                          end
                      end
                  else
                      quote
                          if hasproperty(constant_features_i, $(QuoteNode(sym)))
                              $(sym) = getproperty(constant_features_i, $(QuoteNode(sym)))
                          elseif hasproperty(random_effects, $(QuoteNode(sym)))
                              $(sym) = getproperty(random_effects, $(QuoteNode(sym)))
                          elseif hasproperty(fixed_effects, $(QuoteNode(sym)))
                              $(sym) = getproperty(fixed_effects, $(QuoteNode(sym)))
                          end
                      end
                  end
                  for sym in var_syms]

    binds_funs = [quote
                      if hasproperty(model_funs, $(QuoteNode(sym)))
                          $(sym) = getproperty(model_funs, $(QuoteNode(sym)))
                      elseif hasproperty(helper_functions, $(QuoteNode(sym)))
                          $(sym) = getproperty(helper_functions, $(QuoteNode(sym)))
                      end
                  end
                  for sym in call_syms]

    assigns = [:($(names[i]) = $(exprs[i])) for i in eachindex(names)]
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
        prede_fn = RuntimeGeneratedFunction(
            @__MODULE__, @__MODULE__, $(QuoteNode(func_expr)))
        function prede_wrapper(fixed_effects::ComponentArray,
                random_effects::ComponentArray,
                constant_features_i::NamedTuple,
                model_funs::NamedTuple)
            return prede_fn(fixed_effects, random_effects,
                constant_features_i, model_funs, NamedTuple())
        end
        function prede_wrapper(fixed_effects::ComponentArray,
                random_effects::ComponentArray,
                constant_features_i::NamedTuple,
                model_funs::NamedTuple,
                helper_functions::NamedTuple)
            return prede_fn(fixed_effects, random_effects,
                constant_features_i, model_funs, helper_functions)
        end
        meta = PreDEMeta($names_expr, $syms_expr, $lines_expr)
        builder = PreDEBuilder(prede_wrapper)
        PreDifferentialEquation(meta, builder)
    end
end
