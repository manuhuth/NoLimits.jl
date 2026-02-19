export @randomEffects
export RandomEffects
export RandomEffect
export get_re_meta
export get_re_builders
export get_re_names
export get_re_groups
export get_re_types
export get_re_syms
export get_re_dist_exprs
export get_create_random_effect_distribution
export get_re_logpdf

using ComponentArrays
using Distributions
using LinearAlgebra
using RuntimeGeneratedFunctions

RuntimeGeneratedFunctions.init(@__MODULE__)

struct RandomEffectsMeta
    re_names::Vector{Symbol}
    re_groups::NamedTuple
    re_types::NamedTuple
    re_syms::NamedTuple
    re_dist_exprs::NamedTuple
end

struct RandomEffectsBuilders
    create_random_effect_distribution::Function
    logpdf::Function
end

"""
    RandomEffects

Compiled representation of a `@randomEffects` block. Stores metadata (names, grouping
columns, distribution types, symbol dependencies) and runtime builder functions for
constructing distributions and evaluating log-densities.

Use accessor functions rather than accessing fields directly.
"""
struct RandomEffects
    meta::RandomEffectsMeta
    builders::RandomEffectsBuilders
end

"""
    get_re_meta(re::RandomEffects) -> RandomEffectsMeta

Return the metadata struct for the random effects.
"""
get_re_meta(re::RandomEffects) = re.meta

"""
    get_re_builders(re::RandomEffects) -> RandomEffectsBuilders

Return the builders struct containing the distribution-constructor and logpdf functions.
"""
get_re_builders(re::RandomEffects) = re.builders

"""
    get_re_names(re::RandomEffects) -> Vector{Symbol}

Return the names of all declared random effects (in declaration order).
"""
get_re_names(re::RandomEffects) = re.meta.re_names

"""
    get_re_groups(re::RandomEffects) -> NamedTuple

Return a `NamedTuple` mapping each random-effect name to its grouping column `Symbol`.
"""
get_re_groups(re::RandomEffects) = re.meta.re_groups

"""
    get_re_types(re::RandomEffects) -> NamedTuple

Return a `NamedTuple` mapping each random-effect name to the `Symbol` of its
distribution constructor (e.g. `:Normal`, `:MvNormal`).
"""
get_re_types(re::RandomEffects) = re.meta.re_types

"""
    get_re_syms(re::RandomEffects) -> NamedTuple

Return a `NamedTuple` mapping each random-effect name to the `Vector{Symbol}` of
symbols that appear in its distribution expression.
"""
get_re_syms(re::RandomEffects) = re.meta.re_syms

"""
    get_re_dist_exprs(re::RandomEffects) -> NamedTuple

Return a `NamedTuple` mapping each random-effect name to the quoted distribution
expression as parsed from the `@randomEffects` block.
"""
get_re_dist_exprs(re::RandomEffects) = re.meta.re_dist_exprs

"""
    get_create_random_effect_distribution(re::RandomEffects) -> Function

Return the distribution-builder function with signature:

    (θ::ComponentArray, const_cov::NamedTuple, model_funs::NamedTuple,
     helpers::NamedTuple) -> NamedTuple

The returned `NamedTuple` maps each random-effect name to its instantiated distribution.
"""
get_create_random_effect_distribution(re::RandomEffects) = re.builders.create_random_effect_distribution

"""
    get_re_logpdf(re::RandomEffects) -> Function

Return the log-density function with signature:

    (dists::NamedTuple, re_values::NamedTuple) -> Real

Evaluates the sum of `logpdf(dist, value)` over all random effects.
"""
get_re_logpdf(re::RandomEffects) = re.builders.logpdf

function _is_component_axes(axs)
    axs isa Tuple || return false
    for ax in axs
        ax isa ComponentArrays.AbstractAxis || return false
    end
    return true
end

function _component_axes_from(x)
    x isa ComponentArray && return axes(x)
    if hasfield(typeof(x), :parent)
        axs = _component_axes_from(getfield(x, :parent))
        axs === nothing || return axs
    end
    if hasfield(typeof(x), :value)
        axs = _component_axes_from(getfield(x, :value))
        axs === nothing || return axs
    end
    return nothing
end

function _re_componentize(fixed_effects)
    fixed_effects isa ComponentArray && return fixed_effects
    axs = _component_axes_from(fixed_effects)
    if axs !== nothing
        _is_component_axes(axs) || error("RandomEffects expects ComponentArray-compatible fixed effects.")
        return ComponentArray(fixed_effects, axs...)
    end
    if fixed_effects isa AbstractArray
        axs = axes(fixed_effects)
        if _is_component_axes(axs)
            return ComponentArray(fixed_effects, axs...)
        elseif hasproperty(fixed_effects, :parent)
            parent = getproperty(fixed_effects, :parent)
            axs = axes(parent)
            _is_component_axes(axs) || error("RandomEffects expects ComponentArray-compatible fixed effects.")
            return ComponentArray(fixed_effects, axs...)
        end
    end
    error("RandomEffects expects ComponentArray-compatible fixed effects.")
end

struct RandomEffectDecl
    dist
    column::Symbol
end

"""
    RandomEffect(dist; column::Symbol) -> RandomEffectDecl

Declare a random effect with the given distribution and grouping column.

Used exclusively inside `@randomEffects`:
```julia
@randomEffects begin
    η = RandomEffect(Normal(0.0, σ); column=:ID)
end
```

# Arguments
- `dist`: a distribution expression (evaluated at model construction time). May reference
  fixed effects, constant covariates, helper functions, and model functions.
  The symbols `t` and `ξ` are forbidden (random effects are time-constant).

# Keyword Arguments
- `column::Symbol`: the data-frame column that defines the grouping for this random effect.
"""
RandomEffect(dist; column::Symbol) = RandomEffectDecl(dist, column)

function _re_is_identifier(sym::Symbol)
    return Base.isidentifier(sym)
end

function _re_collect_call_symbols(ex, out::Set{Symbol})
    ex isa Expr || return out
    if ex.head == :call
        f = ex.args[1]
        if f isa Symbol && _re_is_identifier(f)
            push!(out, f)
        end
        for arg in ex.args[2:end]
            _re_collect_call_symbols(arg, out)
        end
        return out
    elseif ex.head == :ref
        _re_collect_call_symbols(ex.args[1], out)
        return out
    else
        for arg in ex.args
            _re_collect_call_symbols(arg, out)
        end
        return out
    end
end

function _re_collect_var_symbols(ex, out::Set{Symbol})
    ex isa Symbol && return (push!(out, ex); out)
    ex isa Expr || return out
    if ex.head == :call
        for arg in ex.args[2:end]
            _re_collect_var_symbols(arg, out)
        end
        return out
    elseif ex.head == :ref
        _re_collect_var_symbols(ex.args[1], out)
        return out
    elseif ex.head == :.
        _re_collect_var_symbols(ex.args[1], out)
        return out
    else
        for arg in ex.args
            _re_collect_var_symbols(arg, out)
        end
        return out
    end
end

function _re_collect_property_bases(ex, out::Set{Symbol})
    ex isa Expr || return out
    if ex.head == :.
        base = ex.args[1]
        base isa Symbol && push!(out, base)
        _re_collect_property_bases(base, out)
        return out
    end
    for arg in ex.args
        _re_collect_property_bases(arg, out)
    end
    return out
end

function _re_forbidden_symbol(ex)
    ex isa Symbol && (ex == :t || ex == :ξ) && return ex
    ex isa Expr || return nothing
    for arg in ex.args
        found = _re_forbidden_symbol(arg)
        found === nothing || return found
    end
    return nothing
end

function _re_ctor_name(fn)
    if fn === :RandomEffect
        return :RandomEffect
    end
    if fn isa GlobalRef && fn.name === :RandomEffect
        return :RandomEffect
    end
    if fn isa Expr && fn.head == :.
        last = fn.args[end]
        last isa QuoteNode && (last = last.value)
        last === :RandomEffect && return :RandomEffect
    end
    return nothing
end

function _re_parse_call_args(rhs::Expr, lhs::Symbol)
    positional = Any[]
    kwargs = Dict{Symbol, Any}()
    for arg in rhs.args[2:end]
        if arg isa Expr && arg.head == :parameters
            for kw in arg.args
                if !(kw isa Expr && (kw.head == :(kw) || kw.head == :(=)) && kw.args[1] isa Symbol)
                    error("Invalid keyword argument in RandomEffect for $(lhs). Use RandomEffect(dist; column=:group).")
                end
                key = kw.args[1]
                haskey(kwargs, key) && error("Duplicate keyword $(key) in RandomEffect for $(lhs).")
                kwargs[key] = kw.args[2]
            end
        elseif arg isa Expr && arg.head == :kw
            key = arg.args[1]
            key isa Symbol || error("Invalid keyword argument in RandomEffect for $(lhs).")
            haskey(kwargs, key) && error("Duplicate keyword $(key) in RandomEffect for $(lhs).")
            kwargs[key] = arg.args[2]
        else
            push!(positional, arg)
        end
    end

    length(positional) == 1 ||
        error("RandomEffect for $(lhs) expects exactly one positional argument (the distribution). Use RandomEffect(dist; column=:group).")
    haskey(kwargs, :column) || error("RandomEffect for $(lhs) requires keyword column=:group.")

    bad_keys = sort([k for k in keys(kwargs) if k != :column])
    isempty(bad_keys) || error("Unsupported keyword(s) $(bad_keys) in RandomEffect for $(lhs). Allowed keywords: column.")

    column = kwargs[:column]
    if column isa QuoteNode
        column = column.value
    elseif column isa Expr && column.head == :quote
        column = column.args[1]
    end
    column isa Symbol || error("RandomEffect column for $(lhs) must be a Symbol.")

    return positional[1], column
end

function _parse_random_effects(block::Expr)
    block.head == :block || error("@randomEffects expects a begin ... end block.")
    re_names = Symbol[]
    dist_exprs = Expr[]
    columns = Symbol[]
    for stmt in block.args
        stmt isa LineNumberNode && continue
        stmt isa Expr || error("Invalid statement in @randomEffects block.")
        stmt.head == :(=) || error("Only assignments are allowed in @randomEffects block.")
        lhs, rhs = stmt.args
        lhs isa Symbol || error("Left-hand side must be a symbol in @randomEffects block.")
        rhs isa Expr && rhs.head == :call || error("Right-hand side must be a RandomEffect(...) call.")
        _re_ctor_name(rhs.args[1]) === :RandomEffect || error("Right-hand side must be a RandomEffect(...) call.")

        dist, column = _re_parse_call_args(rhs, lhs)

        forbidden = _re_forbidden_symbol(dist)
        forbidden === nothing || error("RandomEffect $(lhs) uses forbidden symbol $(forbidden).")

        push!(re_names, lhs)
        push!(dist_exprs, dist)
        push!(columns, column)
    end
    return re_names, dist_exprs, columns
end

function _dist_type_symbol(dist_expr)
    dist_expr isa Expr && dist_expr.head == :call && dist_expr.args[1] isa Symbol && return dist_expr.args[1]
    return :unknown
end

function _rewrite_npf_calls(ex)
    ex isa Expr || return ex
    if ex.head == :call && ex.args[1] == :NormalizingPlanarFlow && length(ex.args) == 2
        arg = ex.args[2]
        if arg isa Symbol
            return Expr(:call, Symbol("NPF_", arg), arg)
        end
    end
    return Expr(ex.head, map(_rewrite_npf_calls, ex.args)...)
end

"""
    @randomEffects begin
        name = RandomEffect(dist; column=:GroupCol)
        ...
    end

Compile random-effect declarations into a [`RandomEffects`](@ref) struct.

Each statement must be an assignment `name = RandomEffect(dist; column=:Col)`. The
distribution expression `dist` may reference fixed effects, constant covariates,
helper functions, and model functions (NNs, splines, soft trees). The symbols `t`
and `ξ` are forbidden.

When using `NormalizingPlanarFlow(ψ)` in a distribution, it is automatically rewritten
to call `model_funs.NPF_ψ(ψ)`, where the NPF callable is registered automatically
from the corresponding `NPFParameter` in `@fixedEffects`.

The `@randomEffects` block is typically used inside `@Model`.
"""
macro randomEffects(block)
    RuntimeGeneratedFunctions.init(__module__)
    re_names, dist_exprs, columns = _parse_random_effects(block)
    dist_exprs_rewritten = _rewrite_npf_calls.(dist_exprs)
    call_syms = Set{Symbol}()
    var_syms = Set{Symbol}()
    prop_syms = Set{Symbol}()
    re_syms = Vector{Vector{Symbol}}(undef, length(dist_exprs_rewritten))
    for (i, ex) in enumerate(dist_exprs_rewritten)
        local_call = Set{Symbol}()
        local_var = Set{Symbol}()
        local_prop = Set{Symbol}()
        _re_collect_call_symbols(ex, local_call)
        _re_collect_var_symbols(ex, local_var)
        _re_collect_property_bases(ex, local_prop)

        _re_collect_call_symbols(ex, call_syms)
        _re_collect_var_symbols(ex, var_syms)
        _re_collect_property_bases(ex, prop_syms)

        delete!(local_var, :fixed_effects)
        delete!(local_var, :constant_features_i)
        delete!(local_var, :model_funs)
        delete!(local_var, :helper_functions)

        local_var = Set([s for s in local_var if Base.isidentifier(s)])
        skip_vars = Set([:Inf, :NaN, :nothing, :missing, :true, :false])
        local_var = Set([s for s in local_var if !(s in skip_vars)])
        local_var = Set([s for s in local_var if !(isdefined(@__MODULE__, s) && getfield(@__MODULE__, s) isa Module)])
        local_var = Set([s for s in local_var if !(isdefined(Base, s) && getfield(Base, s) isa Module)])

        re_syms[i] = sort(collect(union(local_var, local_prop)))
    end

    delete!(var_syms, :fixed_effects)
    delete!(var_syms, :constant_features_i)
    delete!(var_syms, :model_funs)
    delete!(var_syms, :helper_functions)

    call_syms = Set([s for s in call_syms if !(isdefined(Base, s) || isdefined(Distributions, s) || isdefined(@__MODULE__, s))])
    var_syms = Set([s for s in var_syms if Base.isidentifier(s)])
    # Keep symbols even if Base defines them (e.g., μ, σ), but skip a few literals.
    skip_vars = Set([:Inf, :NaN, :nothing, :missing, :true, :false])
    var_syms = Set([s for s in var_syms if !(s in skip_vars)])
    var_syms = Set([s for s in var_syms if !(isdefined(@__MODULE__, s) && getfield(@__MODULE__, s) isa Module)])
    var_syms = Set([s for s in var_syms if !(isdefined(Base, s) && getfield(Base, s) isa Module)])

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

    dist_assigns = [:( $(re_names[i]) = $(dist_exprs_rewritten[i]) ) for i in eachindex(re_names)]
    ret_expr = Expr(:tuple, (Expr(:(=), re_names[i], re_names[i]) for i in eachindex(re_names))...)

    func_expr = :(function (fixed_effects::ComponentArray,
                            constant_features_i::NamedTuple,
                            model_funs::NamedTuple,
                            helper_functions::NamedTuple)
        $(binds_vars...)
        $(binds_funs...)
        $(dist_assigns...)
        $ret_expr
    end)

    re_names_expr = Expr(:vect, QuoteNode.(re_names)...)
    groups_values = Expr(:tuple, (QuoteNode.(columns))...)
    type_values = Expr(:tuple, (QuoteNode.(_dist_type_symbol.(dist_exprs)))...)
    re_syms_expr = Expr(:call, Expr(:curly, :NamedTuple, Expr(:tuple, QuoteNode.(re_names)...)),
                        Expr(:tuple, [Expr(:vect, QuoteNode.(syms)...) for syms in re_syms]...))
    dist_expr_values = Expr(:tuple, (QuoteNode.(dist_exprs))...)

    return quote
        create_dist = RuntimeGeneratedFunction(@__MODULE__, @__MODULE__, $(QuoteNode(func_expr)))
        function create_wrapper(fixed_effects::ComponentArray,
                                constant_features_i::NamedTuple,
                                model_funs::NamedTuple)
            return create_dist(fixed_effects, constant_features_i, model_funs, NamedTuple())
        end
        function create_wrapper(fixed_effects::AbstractArray,
                                constant_features_i::NamedTuple,
                                model_funs::NamedTuple)
            return create_dist(_re_componentize(fixed_effects), constant_features_i, model_funs, NamedTuple())
        end
        function create_wrapper(fixed_effects::ComponentArray,
                                constant_features_i::NamedTuple,
                                model_funs::NamedTuple,
                                helper_functions::NamedTuple)
            return create_dist(fixed_effects, constant_features_i, model_funs, helper_functions)
        end
        function create_wrapper(fixed_effects::AbstractArray,
                                constant_features_i::NamedTuple,
                                model_funs::NamedTuple,
                                helper_functions::NamedTuple)
            return create_dist(_re_componentize(fixed_effects), constant_features_i, model_funs, helper_functions)
        end
        meta = RandomEffectsMeta($re_names_expr,
                                 NamedTuple{($(QuoteNode.(re_names)...),)}($groups_values),
                                 NamedTuple{($(QuoteNode.(re_names)...),)}($type_values),
                                 $re_syms_expr,
                                 NamedTuple{($(QuoteNode.(re_names)...),)}($dist_expr_values))
        logpdf_fn = function (dists, re_values)
            total = 0.0
            $(Expr(:block, [:(total += logpdf(getproperty(dists, $(QuoteNode(n))), getproperty(re_values, $(QuoteNode(n))))) for n in re_names]...))
            total
        end
        builders = RandomEffectsBuilders(create_wrapper, logpdf_fn)
        RandomEffects(meta, builders)
    end
end
