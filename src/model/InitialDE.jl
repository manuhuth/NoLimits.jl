export @initialDE
export InitialDE
export get_initialde_meta
export get_initialde_names
export get_initialde_builder

using ComponentArrays
using RuntimeGeneratedFunctions
using StaticArrays

RuntimeGeneratedFunctions.init(@__MODULE__)

struct InitialDEMeta
    names::Vector{Symbol}
end

struct InitialDEIR
    names::Vector{Symbol}
    exprs::Vector{Expr}
    call_syms::Vector{Symbol}
    var_syms::Vector{Symbol}
    prop_syms::Vector{Symbol}
end

"""
    InitialDE

Compiled representation of an `@initialDE` block. Stores state names and the
intermediate representation (IR) of initial-condition expressions. A runtime builder
function is produced lazily via [`get_initialde_builder`](@ref).
"""
struct InitialDE
    meta::InitialDEMeta
    ir::InitialDEIR
end

"""
    get_initialde_meta(i::InitialDE) -> InitialDEMeta

Return the metadata struct (state names).
"""
get_initialde_meta(i::InitialDE) = i.meta

"""
    get_initialde_names(i::InitialDE) -> Vector{Symbol}

Return the names of the ODE states for which initial conditions are declared.
"""
get_initialde_names(i::InitialDE) = i.meta.names

function _initialde_is_identifier(sym::Symbol)
    return Base.isidentifier(sym)
end

function _initialde_call_name(f)
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

function _initialde_collect_call_symbols(ex, out::Set{Symbol})
    ex isa Expr || return out
    if ex.head == :call
        f = ex.args[1]
        if f isa Symbol && _initialde_is_identifier(f)
            push!(out, f)
        end
        for arg in ex.args[2:end]
            _initialde_collect_call_symbols(arg, out)
        end
        return out
    elseif ex.head == :ref
        _initialde_collect_call_symbols(ex.args[1], out)
        return out
    else
        for arg in ex.args
            _initialde_collect_call_symbols(arg, out)
        end
        return out
    end
end

function _initialde_collect_var_symbols(ex, out::Set{Symbol})
    ex isa Symbol && return (push!(out, ex); out)
    ex isa Expr || return out
    if ex.head == :call
        for arg in ex.args[2:end]
            _initialde_collect_var_symbols(arg, out)
        end
        return out
    elseif ex.head == :ref
        _initialde_collect_var_symbols(ex.args[1], out)
        return out
    elseif ex.head == :.
        _initialde_collect_var_symbols(ex.args[1], out)
        return out
    else
        for arg in ex.args
            _initialde_collect_var_symbols(arg, out)
        end
        return out
    end
end

function _initialde_collect_property_bases(ex, out::Set{Symbol})
    ex isa Expr || return out
    if ex.head == :.
        base = ex.args[1]
        base isa Symbol && push!(out, base)
        _initialde_collect_property_bases(base, out)
        return out
    end
    for arg in ex.args
        _initialde_collect_property_bases(arg, out)
    end
    return out
end

function _initialde_forbidden_symbol(ex)
    ex isa Symbol && (ex == :t || ex == :ξ) && return ex
    ex isa Expr || return nothing
    for arg in ex.args
        found = _initialde_forbidden_symbol(arg)
        found === nothing || return found
    end
    return nothing
end

function _parse_initialde(block::Expr)
    block.head == :block || error("@initialDE expects a begin ... end block.")
    names = Symbol[]
    exprs = Expr[]
    seen = Set{Symbol}()

    for stmt in block.args
        stmt isa LineNumberNode && continue
        stmt isa Expr || error("Invalid statement in @initialDE block.")
        stmt.head == :(=) || error("Only assignments are allowed in @initialDE block.")
        lhs, rhs = stmt.args
        lhs isa Symbol || error("Left-hand side must be a symbol in @initialDE block.")
        lhs in seen && error("Duplicate initial DE name: $(lhs).")
        rhs isa LineNumberNode && error("Invalid right-hand side in @initialDE block.")
        forbidden = _initialde_forbidden_symbol(rhs)
        forbidden === nothing || error("@initialDE uses forbidden symbol $(forbidden).")
        push!(seen, lhs)
        push!(names, lhs)
        push!(exprs, rhs isa Expr ? rhs : Expr(:call, :identity, rhs))
    end
    return names, exprs
end

"""
    @initialDE begin
        state = expr
        ...
    end

Compile initial-condition declarations for an ODE system into an [`InitialDE`](@ref) struct.

Each statement must assign a scalar expression to a state name matching one declared with
`D(state) ~ ...` in the paired `@DifferentialEquation` block. Every state must have exactly
one initial condition. The symbols `t` and `ξ` are forbidden.

Right-hand sides may reference fixed effects, random effects, constant covariates,
pre-DE variables, model functions, and helper functions.
"""
macro initialDE(block)
    names, exprs = _parse_initialde(block)
    call_syms = Set{Symbol}()
    var_syms = Set{Symbol}()
    prop_syms = Set{Symbol}()
    for ex in exprs
        _initialde_collect_call_symbols(ex, call_syms)
        _initialde_collect_var_symbols(ex, var_syms)
        _initialde_collect_property_bases(ex, prop_syms)
    end

    delete!(var_syms, :fixed_effects)
    delete!(var_syms, :random_effects)
    delete!(var_syms, :constant_covariates)
    delete!(var_syms, :model_funs)
    delete!(var_syms, :helpers)
    delete!(var_syms, :preDE)

    call_syms = Set([s for s in call_syms if !(isdefined(Base, s) || isdefined(@__MODULE__, s))])
    var_syms = Set([s for s in var_syms if Base.isidentifier(s)])
    skip_vars = Set([:Inf, :NaN, :nothing, :missing, :true, :false])
    var_syms = Set([s for s in var_syms if !(s in skip_vars)])

    meta = InitialDEMeta(collect(names))
    ir = InitialDEIR(
        collect(names),
        exprs,
        collect(call_syms),
        collect(var_syms),
        collect(prop_syms)
    )
    return :(InitialDE($meta, $ir))
end

"""
    get_initialde_builder(i::InitialDE, state_names::Vector{Symbol}; static=false) -> Function

Build and return the initial-condition function for the ODE system.

The returned function has signature:

    (θ::ComponentArray, η::ComponentArray, const_cov::NamedTuple,
     model_funs::NamedTuple, helpers::NamedTuple, preDE::NamedTuple) -> Vector

It returns the initial state vector ordered to match `state_names`.

# Arguments
- `i::InitialDE`: the compiled initial-condition block.
- `state_names::Vector{Symbol}`: ordered state names from the `@DifferentialEquation` block.

# Keyword Arguments
- `static::Bool = false`: if `true`, returns a `StaticArrays.SVector` for allocation-free
  ODE solving with small state dimensions.
"""
function get_initialde_builder(i::InitialDE,
                               state_names::Vector{Symbol};
                               static::Bool=false)
    init_names = i.ir.names
    name_set = Set(init_names)
    state_set = Set(state_names)
    missing = [s for s in state_names if !(s in name_set)]
    extra = [s for s in init_names if !(s in state_set)]
    isempty(missing) || error("@initialDE is missing initial values for states: $(missing).")
    isempty(extra) || error("@initialDE includes unknown state(s): $(extra).")

    expr_by_name = Dict{Symbol, Expr}((init_names[idx] => i.ir.exprs[idx]) for idx in eachindex(init_names))
    ordered_exprs = [expr_by_name[s] for s in state_names]

    call_syms = Set(i.ir.call_syms)
    var_syms = Set(i.ir.var_syms)
    prop_syms = Set(i.ir.prop_syms)

    fun_syms = Set([s for s in call_syms if !(isdefined(Base, s) || isdefined(@__MODULE__, s))])
    prop_syms_expr = Expr(:call, :Set, Expr(:vect, QuoteNode.(collect(prop_syms))...))

    binds_vars = [
        quote
            if $(QuoteNode(sym)) in $prop_syms_expr
                if hasproperty(constant_covariates, $(QuoteNode(sym)))
                    $(sym) = getproperty(constant_covariates, $(QuoteNode(sym)))
                else
                    error("Unknown symbol $(string($(QuoteNode(sym)))) in initialDE.")
                end
            else
                if hasproperty(preDE, $(QuoteNode(sym)))
                    $(sym) = getproperty(preDE, $(QuoteNode(sym)))
                elseif hasproperty(random_effects, $(QuoteNode(sym)))
                    $(sym) = getproperty(random_effects, $(QuoteNode(sym)))
                elseif hasproperty(fixed_effects, $(QuoteNode(sym)))
                    $(sym) = getproperty(fixed_effects, $(QuoteNode(sym)))
                elseif hasproperty(constant_covariates, $(QuoteNode(sym)))
                    $(sym) = getproperty(constant_covariates, $(QuoteNode(sym)))
                else
                    error("Unknown symbol $(string($(QuoteNode(sym)))) in initialDE.")
                end
            end
        end for sym in var_syms
    ]

    binds_funs = [
        quote
            if hasproperty(model_funs, $(QuoteNode(sym)))
                $(sym) = getproperty(model_funs, $(QuoteNode(sym)))
            elseif hasproperty(helpers, $(QuoteNode(sym)))
                $(sym) = getproperty(helpers, $(QuoteNode(sym)))
            else
                error("Unknown function $(string($(QuoteNode(sym)))) in initialDE.")
            end
        end for sym in fun_syms
    ]

    vec_expr = static ?
        Expr(:call, GlobalRef(StaticArrays, :SVector), ordered_exprs...) :
        Expr(:vect, ordered_exprs...)

    func_expr = :(function (fixed_effects::ComponentArray,
                            random_effects::ComponentArray,
                            constant_covariates::NamedTuple,
                            model_funs::NamedTuple,
                            helpers::NamedTuple,
                            preDE::NamedTuple)
        $(binds_vars...)
        $(binds_funs...)
        return $vec_expr
    end)

    init_rgf = RuntimeGeneratedFunction(@__MODULE__, @__MODULE__, func_expr)
    return init_rgf
end
