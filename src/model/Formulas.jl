export @formulas
export Formulas
export get_formulas_meta
export get_formulas_ir
export get_formulas_builders
export get_formulas_all
export get_formulas_obs
export get_formulas_time_offsets
export get_formulas_lines

using RuntimeGeneratedFunctions
using Distributions

RuntimeGeneratedFunctions.init(@__MODULE__)

struct FormulasMeta
    all_names::Vector{Symbol}
    obs_names::Vector{Symbol}
end

struct FormulasIR
    det_names::Vector{Symbol}
    det_exprs::Vector{Expr}
    obs_names::Vector{Symbol}
    obs_exprs::Vector{Expr}
    call_heads::Vector{Symbol}
    var_syms::Vector{Symbol}
    prop_syms::Vector{Symbol}
    time_call_syms::Vector{Symbol}
    lines::Vector{Expr}
end

FormulasIR(det_names::Vector{Symbol},
           det_exprs::Vector{Expr},
           obs_names::Vector{Symbol},
           obs_exprs::Vector{Expr},
           call_heads::Vector{Symbol},
           var_syms::Vector{Symbol},
           prop_syms::Vector{Symbol},
           time_call_syms::Vector{Symbol}) = FormulasIR(det_names, det_exprs, obs_names, obs_exprs, call_heads, var_syms, prop_syms, time_call_syms, Expr[])

"""
    Formulas

Compiled representation of a `@formulas` block. Stores deterministic-node and
observation-node names and expressions in an intermediate representation (`FormulasIR`).

Builder functions are produced via [`get_formulas_builders`](@ref).
"""
struct Formulas
    meta::FormulasMeta
    ir::FormulasIR
end

"""
    get_formulas_meta(f::Formulas) -> FormulasMeta

Return the metadata struct with `all_names` (deterministic + observation names) and
`obs_names` (observation names only).
"""
get_formulas_meta(f::Formulas) = f.meta

"""
    get_formulas_ir(f::Formulas) -> FormulasIR

Return the intermediate representation containing parsed expression lists and
collected symbol sets.
"""
get_formulas_ir(f::Formulas) = f.ir

"""
    get_formulas_lines(f::Formulas) -> Vector{Expr}

Return the raw statement expressions from the `@formulas` block (both `=` and `~` forms).
"""
get_formulas_lines(f::Formulas) = f.ir.lines

function _parse_numeric_literal(ex)
    ex isa Number && return Float64(ex)
    if ex isa Expr && ex.head == :call && ex.args[1] == :/ && length(ex.args) == 3
        a = ex.args[2]
        b = ex.args[3]
        if a isa Number && b isa Number
            return Float64(a / b)
        end
    end
    if ex isa Expr && ex.head == :call && ex.args[1] == :- && length(ex.args) == 2
        ex.args[2] isa Number || return nothing
        return -Float64(ex.args[2])
    end
    return nothing
end

function _extract_time_offset(arg)
    if arg == :t || arg == :ξ
        return 0.0
    end
    if arg isa Expr && arg.head == :call && length(arg.args) == 3
        op = arg.args[1]
        a = arg.args[2]
        b = arg.args[3]
        if op == :+
            if a == :t
                return _parse_numeric_literal(b)
            elseif b == :t
                return _parse_numeric_literal(a)
            end
        elseif op == :-
            if a == :t
                val = _parse_numeric_literal(b)
                val === nothing && return nothing
                return -val
            end
        end
    end
    return nothing
end

function _collect_time_offsets(ex, name_set::Set{Symbol}, offsets::Vector{Float64}, requires_dense::Base.RefValue{Bool})
    ex isa Expr || return
    if ex.head == :call
        f = ex.args[1]
        if f isa Symbol && f in name_set && length(ex.args) == 2
            off = _extract_time_offset(ex.args[2])
            if off === nothing
                requires_dense[] = true
            else
                push!(offsets, off)
            end
        end
    end
    for arg in ex.args
        _collect_time_offsets(arg, name_set, offsets, requires_dense)
    end
    return
end

"""
    get_formulas_time_offsets(f::Formulas, state_names, signal_names) -> (Vector{Float64}, Bool)

Scan the formula expressions for state/signal calls with constant time offsets
(e.g. `x1(t - 0.5)`) and return `(offsets, requires_dense)`.

- `offsets`: sorted unique numerical offsets found (including `0.0` for plain `x(t)` calls).
- `requires_dense`: `true` if any non-constant time argument is found, requiring a dense
  ODE solution (`saveat_mode=:dense`).

# Arguments
- `f::Formulas`: the compiled formulas block.
- `state_names::Vector{Symbol}`: ODE state names.
- `signal_names::Vector{Symbol}`: derived signal names.
"""
function get_formulas_time_offsets(f::Formulas, state_names::Vector{Symbol}, signal_names::Vector{Symbol})
    name_set = Set(vcat(state_names, signal_names))
    isempty(name_set) && return (Float64[], false)
    offsets = Float64[]
    requires_dense = Ref(false)
    for ex in vcat(f.ir.det_exprs, f.ir.obs_exprs)
        _collect_time_offsets(ex, name_set, offsets, requires_dense)
    end
    return (unique(offsets), requires_dense[])
end

function _formulas_collect_call_symbols(ex, out)
    ex isa Expr || return out
    if ex.head == :call
        f = ex.args[1]
        f isa Symbol && push!(out, f)
    end
    for arg in ex.args
        _formulas_collect_call_symbols(arg, out)
    end
    return out
end

function _formulas_collect_var_symbols(ex, out)
    ex isa Symbol && push!(out, ex)
    ex isa Expr || return out
    for arg in ex.args
        _formulas_collect_var_symbols(arg, out)
    end
    return out
end

function _formulas_collect_property_bases(ex, out)
    ex isa Expr || return out
    if ex.head == :.
        base = ex.args[1]
        base isa Symbol && push!(out, base)
        _formulas_collect_property_bases(base, out)
        return out
    end
    for arg in ex.args
        _formulas_collect_property_bases(arg, out)
    end
    return out
end

function _formulas_collect_time_calls(ex, out)
    ex isa Expr || return out
    if ex.head == :call
        f = ex.args[1]
        if f isa Symbol && length(ex.args) == 2 && _formulas_arg_uses_time(ex.args[2])
            push!(out, f)
        end
        for arg in ex.args
            _formulas_collect_time_calls(arg, out)
        end
        return out
    else
        for arg in ex.args
            _formulas_collect_time_calls(arg, out)
        end
        return out
    end
end

function _formulas_arg_uses_time(ex)
    ex == :t && return true
    ex == :ξ && return true
    ex isa Expr || return false
    for arg in ex.args
        _formulas_arg_uses_time(arg) && return true
    end
    return false
end

function _formulas_is_state_call(ex, name::Symbol)
    ex isa Expr || return false
    ex.head == :call || return false
    ex.args[1] == name || return false
    length(ex.args) == 2 || return false
    return ex.args[2] == :t || ex.args[2] == :ξ
end

function _formulas_state_used_bare(ex, state::Symbol)
    ex isa Symbol && ex == state && return true
    ex isa Expr || return false
    if ex.head == :call && ex.args[1] == state
        return false
    end
    for arg in ex.args
        _formulas_state_used_bare(arg, state) && return true
    end
    return false
end

function _formulas_replace_state_calls(ex, state_syms::Set{Symbol})
    ex isa Expr || return ex
    if ex.head == :call
        f = ex.args[1]
        if f isa Symbol && f in state_syms && length(ex.args) == 2 && _formulas_arg_uses_time(ex.args[2])
            return Expr(:call, Expr(:., :sol_accessors, QuoteNode(f)), ex.args[2])
        end
    end
    return Expr(ex.head, map(arg -> _formulas_replace_state_calls(arg, state_syms), ex.args)...)
end

function _formulas_rewrite_all(ex, fun_syms::Set{Symbol})
    ex isa Expr || return ex
    if ex.head == :call
        f = ex.args[1]
        new_args = [_formulas_rewrite_all(arg, fun_syms) for arg in ex.args[2:end]]
        if f isa Symbol && f in fun_syms
            return Expr(:call, Expr(:., :funs, QuoteNode(f)), new_args...)
        end
        return Expr(:call, f, new_args...)
    elseif ex.head == :.
        base = _formulas_rewrite_all(ex.args[1], fun_syms)
        return Expr(:., base, ex.args[2])
    else
        return Expr(ex.head, map(arg -> _formulas_rewrite_all(arg, fun_syms), ex.args)...)
    end
end

function _formulas_rewrite_calls(ex, helper_syms::Set{Symbol}, model_fun_syms::Set{Symbol})
    ex isa Expr || return ex
    if ex.head == :call
        f = ex.args[1]
        if f isa Symbol && f in helper_syms
            return Expr(:call, Expr(:., :helpers, QuoteNode(f)),
                        map(arg -> _formulas_rewrite_calls(arg, helper_syms, model_fun_syms), ex.args[2:end])...)
        elseif f isa Symbol && f in model_fun_syms
            return Expr(:call, Expr(:., :model_funs, QuoteNode(f)),
                        map(arg -> _formulas_rewrite_calls(arg, helper_syms, model_fun_syms), ex.args[2:end])...)
        end
        return Expr(:call, map(arg -> _formulas_rewrite_calls(arg, helper_syms, model_fun_syms), ex.args)...)
    end
    return Expr(ex.head, map(arg -> _formulas_rewrite_calls(arg, helper_syms, model_fun_syms), ex.args)...)
end

function _parse_formulas(block::Expr)
    block.head == :block || error("@formulas expects a begin ... end block.")
    det_names = Symbol[]
    det_exprs = Expr[]
    obs_names = Symbol[]
    obs_exprs = Expr[]
    lines = Expr[]

    det_set = Set{Symbol}()
    obs_set = Set{Symbol}()

    for stmt in block.args
        stmt isa LineNumberNode && continue
        stmt isa Expr || error("Invalid statement in @formulas block.")
        if stmt.head == :(=)
            lhs, rhs = stmt.args
            lhs isa Symbol || error("Left-hand side must be a symbol in @formulas block.")
            lhs == :t && error("Left-hand side cannot be t in @formulas block.")
            lhs == :ξ && error("Left-hand side cannot be ξ in @formulas block.")
            lhs in det_set && error("Duplicate deterministic name $(lhs) in @formulas block.")
            lhs in obs_set && error("Name $(lhs) is already used for an observation in @formulas block.")
            push!(det_names, lhs)
            push!(det_exprs, rhs)
            push!(det_set, lhs)
            push!(lines, Expr(:(=), lhs, rhs))
        elseif stmt.head == :call && stmt.args[1] == :~
            lhs, rhs = stmt.args[2], stmt.args[3]
            lhs isa Symbol || error("Left-hand side must be a symbol in @formulas block.")
            lhs == :t && error("Left-hand side cannot be t in @formulas block.")
            lhs == :ξ && error("Left-hand side cannot be ξ in @formulas block.")
            lhs in obs_set && error("Duplicate observation name $(lhs) in @formulas block.")
            lhs in det_set && error("Name $(lhs) is already used for a deterministic in @formulas block.")
            push!(obs_names, lhs)
            push!(obs_exprs, rhs)
            push!(obs_set, lhs)
            push!(lines, Expr(:call, :~, lhs, rhs))
        else
            error("Only assignments and ~ observations are allowed in @formulas block.")
        end
    end

    return det_names, det_exprs, obs_names, obs_exprs, lines
end

function _formulas_build_formulas_expr(ir::FormulasIR,
                                       fixed_names::Vector{Symbol},
                                       re_names::Vector{Symbol},
                                       prede_names::Vector{Symbol},
                                       const_cov_names::Vector{Symbol},
                                       varying_cov_names::Vector{Symbol},
                                       helper_names::Vector{Symbol},
                                       model_fun_names::Vector{Symbol},
                                       state_names::Vector{Symbol},
                                       signal_names::Vector{Symbol},
                                       index_sym::Symbol,
                                       collect_fixed_names::Vector{Symbol} = Symbol[])
    det_exprs = copy(ir.det_exprs)
    obs_exprs = copy(ir.obs_exprs)
    all_exprs = vcat(det_exprs, obs_exprs)

    var_syms = Set(ir.var_syms)
    prop_syms = Set(ir.prop_syms)
    time_call_syms = Set(ir.time_call_syms)

    for s in vcat(state_names, signal_names)
        if _formulas_state_used_bare.(all_exprs, s) |> any
            error("State or derived signal $(s) must be called as $(s)(t) in @formulas.")
        end
    end

    required_states = [s for s in state_names if s in time_call_syms]
    required_signals = [s for s in signal_names if s in time_call_syms]

    state_call_syms = Set(vcat(required_states, required_signals))
    all_exprs = [_formulas_replace_state_calls(ex, state_call_syms) for ex in all_exprs]
    det_exprs = all_exprs[1:length(det_exprs)]
    obs_exprs = all_exprs[length(det_exprs) + 1:end]

    helper_syms = Set(helper_names)
    model_fun_syms = Set(model_fun_names)
    all_exprs = [_formulas_rewrite_calls(ex, helper_syms, model_fun_syms) for ex in all_exprs]
    det_exprs = all_exprs[1:length(det_exprs)]
    obs_exprs = all_exprs[length(det_exprs) + 1:end]

    fixed_set = Set(fixed_names)
    re_set = Set(re_names)
    prede_set = Set(prede_names)
    const_cov_set = Set(const_cov_names)
    varying_cov_set = Set(varying_cov_names)

    for sym in var_syms
        in_fixed = sym in fixed_set
        in_re = sym in re_set
        in_pre = sym in prede_set
        in_const = sym in const_cov_set
        in_var = sym in varying_cov_set
        count = (in_fixed ? 1 : 0) + (in_re ? 1 : 0) + (in_pre ? 1 : 0) +
                (in_const ? 1 : 0) + (in_var ? 1 : 0)
        count > 1 && error("Symbol $(sym) is ambiguous in @formulas (appears in multiple namespaces).")
    end

    fixed_used = [s for s in fixed_names if s in var_syms]
    re_used = [s for s in re_names if s in var_syms]
    prede_used = [s for s in prede_names if s in var_syms]

    prop_const = [s for s in prop_syms if s in const_cov_set]
    prop_var = [s for s in prop_syms if s in varying_cov_set]

    bare_const = [s for s in var_syms if s in const_cov_set && !(s in prop_syms)]
    bare_var = [s for s in var_syms if s in varying_cov_set && !(s in prop_syms)]

    explicit_var = [s for s in bare_var if s in time_call_syms]
    implicit_var = [s for s in bare_var if !(s in time_call_syms)]

    collect_fixed_set = Set(collect_fixed_names)
    binds = Expr[]
    push!(binds, :(t = getproperty(varying_covariates, $(QuoteNode(index_sym)))))

    for s in fixed_used
        if s in collect_fixed_set
            push!(binds, :( $(s) = collect(getproperty(fixed_effects, $(QuoteNode(s)))) ))
        else
            push!(binds, :( $(s) = getproperty(fixed_effects, $(QuoteNode(s))) ))
        end
    end
    for s in re_used
        push!(binds, :( $(s) = getproperty(random_effects, $(QuoteNode(s))) ))
    end
    for s in prede_used
        push!(binds, :( $(s) = getproperty(prede, $(QuoteNode(s))) ))
    end
    for s in prop_const
        push!(binds, :( $(s) = getproperty(constant_covariates_i, $(QuoteNode(s))) ))
    end
    for s in prop_var
        push!(binds, :( $(s) = getproperty(varying_covariates, $(QuoteNode(s))) ))
    end
    for s in bare_const
        push!(binds, :( $(s) = getproperty(constant_covariates_i, $(QuoteNode(s))) ))
    end
    for s in explicit_var
        push!(binds, :( $(s) = getproperty(varying_covariates, $(QuoteNode(s))) ))
    end
    for s in implicit_var
        push!(binds, quote
            local _v = getproperty(varying_covariates, $(QuoteNode(s)))
            if _v isa Function
                $(s) = _v(t)
            else
                $(s) = _v
            end
        end)
    end

    det_assigns = [:( $(ir.det_names[i]) = $(det_exprs[i]) ) for i in eachindex(ir.det_names)]
    all_vals = vcat([:( $(name) ) for name in ir.det_names], obs_exprs)

    all_nt = Expr(:call,
        Expr(:curly, :NamedTuple, Expr(:tuple, QuoteNode.(vcat(ir.det_names, ir.obs_names))...)),
        Expr(:tuple, all_vals...))

    obs_nt = Expr(:call,
        Expr(:curly, :NamedTuple, Expr(:tuple, QuoteNode.(ir.obs_names)...)),
        Expr(:tuple, obs_exprs...))

    all_expr = :(function (ctx, sol_accessors, constant_covariates_i, varying_covariates)
        fixed_effects = ctx.fixed_effects
        random_effects = ctx.random_effects
        prede = ctx.prede
        helpers = ctx.helpers
        model_funs = ctx.model_funs
        $(binds...)
        $(det_assigns...)
        return $all_nt
    end)

    obs_expr = :(function (ctx, sol_accessors, constant_covariates_i, varying_covariates)
        fixed_effects = ctx.fixed_effects
        random_effects = ctx.random_effects
        prede = ctx.prede
        helpers = ctx.helpers
        model_funs = ctx.model_funs
        $(binds...)
        $(det_assigns...)
        return $obs_nt
    end)

    return (all_expr, obs_expr, required_states, required_signals)
end

"""
    get_formulas_builders(f::Formulas; fixed_names, random_names, prede_names,
                          const_cov_names, varying_cov_names, helper_names,
                          model_fun_names, state_names, signal_names,
                          index_sym) -> (all_fn, obs_fn, req_states, req_signals)

Compile the formula expressions into two runtime-generated functions and return them
together with lists of required DE states and signals.

# Returns
- `all_fn`: function `(ctx, sol_accessors, const_cov_i, vary_cov) -> NamedTuple`
  evaluating all deterministic and observation nodes.
- `obs_fn`: function `(ctx, sol_accessors, const_cov_i, vary_cov) -> NamedTuple`
  evaluating observation nodes only.
- `req_states::Vector{Symbol}`: DE state names that are accessed in the formulas.
- `req_signals::Vector{Symbol}`: derived signal names that are accessed in the formulas.

# Keyword Arguments
- `fixed_names`, `random_names`, `prede_names`, `const_cov_names`, `varying_cov_names`:
  symbol lists from each model namespace.
- `helper_names`, `model_fun_names`: callable symbol lists.
- `state_names`, `signal_names`: DE state and signal names for time-call rewriting.
- `index_sym::Symbol = :t`: the varying-covariates key used to extract the current time.
"""
function get_formulas_builders(f::Formulas;
                               fixed_names::Vector{Symbol} = Symbol[],
                               collect_fixed_names::Vector{Symbol} = Symbol[],
                               random_names::Vector{Symbol} = Symbol[],
                               prede_names::Vector{Symbol} = Symbol[],
                               const_cov_names::Vector{Symbol} = Symbol[],
                               varying_cov_names::Vector{Symbol} = Symbol[],
                               helper_names::Vector{Symbol} = Symbol[],
                               model_fun_names::Vector{Symbol} = Symbol[],
                               state_names::Vector{Symbol} = Symbol[],
                               signal_names::Vector{Symbol} = Symbol[],
                               index_sym::Symbol = :t)
    (form_all_expr, form_obs_expr, req_states, req_signals) = _formulas_build_formulas_expr(
        f.ir,
        fixed_names, random_names, prede_names,
        const_cov_names, varying_cov_names,
        helper_names, model_fun_names,
        state_names, signal_names,
        index_sym,
        collect_fixed_names
    )

    form_all_rgf = RuntimeGeneratedFunction(@__MODULE__, @__MODULE__, form_all_expr)
    form_obs_rgf = RuntimeGeneratedFunction(@__MODULE__, @__MODULE__, form_obs_expr)

    return (form_all_rgf, form_obs_rgf, req_states, req_signals)
end

"""
    get_formulas_all(f::Formulas, ctx, sol_accessors, const_cov_i, vary_cov; kwargs...) -> NamedTuple

Evaluate all formula nodes (deterministic and observation) and return a `NamedTuple`
mapping every defined name to its value.

`kwargs` are forwarded to [`get_formulas_builders`](@ref) to supply namespace context.
"""
function get_formulas_all(f::Formulas, ctx, sol_accessors, constant_covariates_i, varying_covariates; kwargs...)
    (form_all_rgf, _, _, _) = get_formulas_builders(f; kwargs...)
    return form_all_rgf(ctx, sol_accessors, constant_covariates_i, varying_covariates)
end

"""
    get_formulas_obs(f::Formulas, ctx, sol_accessors, const_cov_i, vary_cov; kwargs...) -> NamedTuple

Evaluate only the observation nodes (`y ~ dist`) and return a `NamedTuple` mapping
each observation name to its distribution.

`kwargs` are forwarded to [`get_formulas_builders`](@ref) to supply namespace context.
"""
function get_formulas_obs(f::Formulas, ctx, sol_accessors, constant_covariates_i, varying_covariates; kwargs...)
    (_, form_obs_rgf, _, _) = get_formulas_builders(f; kwargs...)
    return form_obs_rgf(ctx, sol_accessors, constant_covariates_i, varying_covariates)
end

"""
    @formulas begin
        name = expr          # deterministic node
        outcome ~ dist(...)  # observation node
        ...
    end

Compile the observation model into a [`Formulas`](@ref) struct.

Two statement forms are supported:
- `name = expr` — a deterministic intermediate variable. May reference any previously
  defined deterministic name or any model symbol.
- `outcome ~ dist(...)` — an observation distribution. The right-hand side must be a
  `Distributions.Distribution` constructor.

Symbols are resolved from (in order): fixed effects, random effects, pre-DE variables,
constant covariates, varying covariates, helper functions, model functions, and DE
state/signal accessors. State and signal names must be called with a time argument:
`x1(t)` or `x1(t - offset)`.

Dynamic covariates used without an explicit `(t)` call are evaluated implicitly at the
current time `t`.

The `@formulas` block is required in every `@Model`.
"""
macro formulas(block)
    det_names, det_exprs, obs_names, obs_exprs, line_exprs = _parse_formulas(block)
    all_names = vcat(det_names, obs_names)
    all_exprs = vcat(det_exprs, obs_exprs)
    det_set = Set(det_names)

    call_heads = Set{Symbol}()
    var_syms = Set{Symbol}()
    prop_syms = Set{Symbol}()
    time_call_syms = Set{Symbol}()
    for ex in all_exprs
        _formulas_collect_call_symbols(ex, call_heads)
        _formulas_collect_var_symbols(ex, var_syms)
        _formulas_collect_property_bases(ex, prop_syms)
        _formulas_collect_time_calls(ex, time_call_syms)
    end

    delete!(var_syms, :t)
    delete!(var_syms, :ξ)
    for name in det_names
        delete!(var_syms, name)
    end

    fun_syms = Set([s for s in call_heads if !(isdefined(Base, s) || isdefined(Distributions, s) || isdefined(@__MODULE__, s))])
    var_syms = Set([s for s in var_syms if Base.isidentifier(s)])
    skip_vars = Set([:Inf, :NaN, :nothing, :missing, :true, :false])
    var_syms = Set([s for s in var_syms if !(s in skip_vars)])

    # Enforce state usage with (t) if state names are provided.
    # The model macro will inject state_syms at expansion time if needed.
    state_syms = Set{Symbol}()
    for s in state_syms
        for ex in all_exprs
            _formulas_state_used_bare(ex, s) && error("State $(s) must be called as $(s)(t) in @formulas.")
        end
    end

    return quote
        meta = FormulasMeta($(Expr(:vect, QuoteNode.(all_names)...)), $(Expr(:vect, QuoteNode.(obs_names)...)))
        ir = FormulasIR(
            $(Expr(:vect, QuoteNode.(det_names)...)),
            $(Expr(:vect, QuoteNode.(det_exprs)...)),
            $(Expr(:vect, QuoteNode.(obs_names)...)),
            $(Expr(:vect, QuoteNode.(obs_exprs)...)),
            $(Expr(:vect, QuoteNode.(collect(call_heads))...)),
            $(Expr(:vect, QuoteNode.(collect(var_syms))...)),
            $(Expr(:vect, QuoteNode.(collect(prop_syms))...)),
            $(Expr(:vect, QuoteNode.(collect(time_call_syms))...)),
            $(Expr(:vect, QuoteNode.(line_exprs)...))
        )
        Formulas(meta, ir)
    end
end
