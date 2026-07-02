export @helpers

struct HelperDef
    name::Symbol
    args::Vector{Symbol}
    body::Expr
    source::Expr
end

function _warn_if_mutating_helper(def::HelperDef)
    _macro_contains_mutation(def.body) || return nothing
    @warn "Possible mutation detected in @helpers for $(def.name). ForwardDiff usually handles mutation but it can increase compile time and runtime for large models, and may break some reverse-mode AD backends. Consider a non-mutating form."
    return nothing
end

function _helper_arg_name(arg::Any)::Symbol
    if arg isa Symbol
        return arg
    elseif arg isa Expr && arg.head == :(::) && arg.args[1] isa Symbol
        return arg.args[1]
    else
        error("Helper arguments must be simple symbols (optionally typed).")
    end
end

function _parse_helper(stmt::Expr)::HelperDef
    if stmt.head == :(=)
        lhs, rhs = stmt.args
        lhs isa Expr && lhs.head == :call ||
            error("Helper definitions must be function-like: f(x)=... or function f(x) ... end")
        name = lhs.args[1]
        name isa Symbol || error("Helper name must be a Symbol.")
        args = [_helper_arg_name(a) for a in lhs.args[2:end]]
        return HelperDef(name, args, rhs, stmt)
    elseif stmt.head == :function
        sig, body = stmt.args
        sig isa Expr && sig.head == :call ||
            error("Helper definitions must be function-like: f(x)=... or function f(x) ... end")
        name = sig.args[1]
        name isa Symbol || error("Helper name must be a Symbol.")
        args = [_helper_arg_name(a) for a in sig.args[2:end]]
        return HelperDef(name, args, body, stmt)
    else
        error("Only helper function definitions are allowed in @helpers blocks.")
    end
end

function parse_helpers(block::Expr)::Vector{HelperDef}
    block.head == :block || error("@helpers expects a begin ... end block.")
    defs = HelperDef[]
    seen = Set{Symbol}()
    for stmt in block.args
        stmt isa LineNumberNode && continue
        stmt isa Expr || error("Invalid statement in @helpers block.")
        def = _parse_helper(stmt)
        if def.name in seen
            error("Duplicate helper name: $(def.name). Helper names must be unique within @helpers.")
        end
        _warn_if_mutating_helper(def)
        push!(seen, def.name)
        push!(defs, def)
    end
    return defs
end

"""
    @helpers begin
        f(x) = ...
        g(x, y) = ...
    end

Define user-provided helper functions that are available inside `@randomEffects`,
`@preDifferentialEquation`, `@DifferentialEquation`, `@initialDE`, and `@formulas` blocks.

Each statement must be a function definition using short (`f(x) = expr`) or long
(`function f(x) ... end`) form. Helper names must be unique within the block.

Returns a `NamedTuple` mapping each function name to its compiled anonymous function.
The helpers `NamedTuple` is stored in the `Model` and passed automatically at
evaluation time via `get_helper_funs`.

Mutating operations (calls ending in `!`, indexed assignment) trigger a warning since
they may break some reverse-mode automatic differentiation backends.
"""
macro helpers(block)
    defs = parse_helpers(block)
    if isempty(defs)
        return esc(:(NamedTuple()))
    end
    func_vals = [Expr(:(=), def.name, Expr(:->, Expr(:tuple, def.args...), def.body))
                 for def in defs]
    nt_expr = Expr(:tuple, func_vals...)
    return esc(nt_expr)
end
