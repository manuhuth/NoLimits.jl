# Shared expression-walker utilities for the model-block macros.
#
# `@helpers`, `@randomEffects`, `@preDifferentialEquation`, `@initialDE`, and
# `@DifferentialEquation` all analyse user expressions the same way (collecting
# called/variable/property symbols, rejecting mutation and reserved time symbols).
# These walkers used to be copy-pasted per macro file with a per-file prefix; they
# live here once. (`@formulas` keeps its own `_formulas_*` variants on purpose —
# they differ semantically: no identifier filter and no call-head skip.)

function _macro_is_identifier(sym::Symbol)
    return Base.isidentifier(sym)
end

function _macro_call_name(f)
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

function _macro_is_mutating_assign(ex::Expr)
    if ex.head == :(=)
        lhs = ex.args[1]
        return lhs isa Expr && (lhs.head == :ref || lhs.head == :.)
    end
    head_str = string(ex.head)
    return startswith(head_str, ".") && endswith(head_str, "=")
end

function _macro_contains_mutation(ex)
    ex isa Expr || return false
    _macro_is_mutating_assign(ex) && return true
    if ex.head == :call
        fname = _macro_call_name(ex.args[1])
        fname !== nothing && endswith(String(fname), "!") && return true
    end
    for arg in ex.args
        _macro_contains_mutation(arg) && return true
    end
    return false
end

function _macro_collect_call_symbols(ex, out::Set{Symbol})
    ex isa Expr || return out
    if ex.head == :call
        f = ex.args[1]
        if f isa Symbol && _macro_is_identifier(f)
            push!(out, f)
        end
        for arg in ex.args[2:end]
            _macro_collect_call_symbols(arg, out)
        end
        return out
    elseif ex.head == :ref
        _macro_collect_call_symbols(ex.args[1], out)
        return out
    else
        for arg in ex.args
            _macro_collect_call_symbols(arg, out)
        end
        return out
    end
end

function _macro_collect_var_symbols(ex, out::Set{Symbol})
    ex isa Symbol && return (push!(out, ex); out)
    ex isa Expr || return out
    if ex.head == :call
        for arg in ex.args[2:end]
            _macro_collect_var_symbols(arg, out)
        end
        return out
    elseif ex.head == :ref
        _macro_collect_var_symbols(ex.args[1], out)
        return out
    elseif ex.head == :.
        _macro_collect_var_symbols(ex.args[1], out)
        return out
    else
        for arg in ex.args
            _macro_collect_var_symbols(arg, out)
        end
        return out
    end
end

function _macro_collect_property_bases(ex, out::Set{Symbol})
    ex isa Expr || return out
    if ex.head == :.
        base = ex.args[1]
        base isa Symbol && push!(out, base)
        _macro_collect_property_bases(base, out)
        return out
    end
    for arg in ex.args
        _macro_collect_property_bases(arg, out)
    end
    return out
end

function _macro_forbidden_symbol(ex)
    ex isa Symbol && (ex == :t || ex == :ξ) && return ex
    ex isa Expr || return nothing
    for arg in ex.args
        found = _macro_forbidden_symbol(arg)
        found === nothing || return found
    end
    return nothing
end
