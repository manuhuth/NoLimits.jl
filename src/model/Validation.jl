# Model-build validation shared by @Model: catches undefined/typo'd symbols, reserved
# names and structurally empty observation models at construction time instead of
# letting them surface as UndefVarError deep inside a fitted objective.

# Time symbols that may not be shadowed by an effect name. `ξ` is only reserved *inside*
# a random-effect distribution (checked by `_macro_forbidden_symbol`); it is a legal
# effect name, so only `t` -- which collides with the `x1(t)` call syntax -- is listed.
const _NL_RESERVED_EFFECT_NAMES = (:t,)
# Symbols that are always bound at evaluation time and are never "undefined".
const _NL_TIME_SYMBOLS = (:t, :ξ)

# Symbols that resolve to an actual binding at formula-evaluation time (Base/Distributions/
# NoLimits names, plus anything defined in the module where @Model was written).
function _nl_symbol_resolvable(s::Symbol, mod::Module)
    return isdefined(Base, s) || isdefined(Distributions, s) ||
           isdefined(@__MODULE__, s) || isdefined(mod, s)
end

# Names an expression binds locally: anonymous-function arguments, `do` arguments, and
# comprehension/generator iteration variables. `ntuple(s -> ntuple(j -> ..., 2), 2)` inside
# @formulas binds `s` and `j`, which are not undefined symbols.
function _nl_collect_bound_syms!(out::Set{Symbol}, ex)
    ex isa Symbol && (push!(out, ex); return out)
    ex isa Expr || return out
    if ex.head == :tuple || ex.head == :parameters
        for a in ex.args
            _nl_collect_bound_syms!(out, a)
        end
    elseif ex.head == :(::) || ex.head == :kw
        _nl_collect_bound_syms!(out, ex.args[1])
    elseif ex.head == :(=) && length(ex.args) == 2   # `for i in ...` inside a generator
        _nl_collect_bound_syms!(out, ex.args[1])
    end
    return out
end

# Free symbols of a user expression: call heads included; property names, keyword-argument
# names and locally bound names excluded (`x.Age` contributes `x`, `Normal(; σ = s)`
# contributes `s`, `j -> f(j)` contributes neither).
function _nl_collect_free_syms!(out::Set{Symbol}, ex, bound::Set{Symbol} = Set{Symbol}())
    ex isa Symbol && (ex in bound || push!(out, ex); return out)
    ex isa Expr || return out
    if ex.head == :.
        return _nl_collect_free_syms!(out, ex.args[1], bound)
    elseif ex.head == :kw && length(ex.args) == 2
        return _nl_collect_free_syms!(out, ex.args[2], bound)
    elseif ex.head == :-> || ex.head == :function || ex.head == :do
        inner = union(bound, _nl_collect_bound_syms!(Set{Symbol}(), ex.args[1]))
        return _nl_collect_free_syms!(out, ex.args[2], inner)
    elseif ex.head == :generator || ex.head == :comprehension ||
           ex.head == :typed_comprehension
        inner = copy(bound)
        for a in ex.args[2:end]
            _nl_collect_bound_syms!(inner, a)
        end
        for a in ex.args
            _nl_collect_free_syms!(out, a, inner)
        end
        return out
    end
    for a in ex.args
        _nl_collect_free_syms!(out, a, bound)
    end
    return out
end

function _nl_unknown_syms(exprs, known::Set{Symbol}, mod::Module)
    syms = Set{Symbol}()
    for ex in exprs
        _nl_collect_free_syms!(syms, ex)
    end
    return sort([s
                 for s in syms
                 if Base.isidentifier(s) && !(s in known) && !(s in _NL_TIME_SYMBOLS) &&
                        !_nl_symbol_resolvable(s, mod)])
end

function _nl_check_reserved_names(names, what::AbstractString)
    bad = [s for s in names if s in _NL_RESERVED_EFFECT_NAMES]
    isempty(bad) && return nothing
    error("$(what) may not be named $(join(string.(bad), ", ")): `t` is reserved for the time variable used in @DifferentialEquation and @formulas.")
end

"""
    _validate_model_symbols(...)

Run at the end of `@Model`. Rejects reserved names, an observation-free `@formulas`
block, constant covariates called as functions, and symbols in `@formulas` /
`@randomEffects` that resolve against no model namespace and no visible binding.
"""
function _nl_collect_field_accesses!(out::Vector{Pair{Symbol, Symbol}}, ex)
    ex isa Expr || return out
    if ex.head == :. && ex.args[1] isa Symbol && ex.args[2] isa QuoteNode &&
       ex.args[2].value isa Symbol
        push!(out, ex.args[1] => ex.args[2].value)
    end
    for a in ex.args
        _nl_collect_field_accesses!(out, a)
    end
    return out
end

# `x.age` when the covariate declares `[:Age]` used to surface as a ComponentArrays
# FieldError from inside the objective (#206).
function _nl_check_covariate_fields(exprs, covariates)
    accesses = Pair{Symbol, Symbol}[]
    for ex in exprs
        _nl_collect_field_accesses!(accesses, ex)
    end
    for (base, field) in accesses
        hasproperty(covariates.params, base) || continue
        p = getproperty(covariates.params, base)
        hasproperty(p, :columns) || continue
        field in p.columns ||
            error("Covariate `$(base)` has no column `$(field)`. Declared columns: $(join(string.(p.columns), ", ")).")
    end
    return nothing
end

# Declared input width / output width of a model-function parameter block, or `nothing`
# when it cannot be determined statically (duck-typed so no Lux/SimpleChains dependency).
_nl_fun_in_dim(p) = nothing
_nl_fun_in_dim(p::SoftTreeParameters) = p.input_dim
_nl_fun_in_dim(p::SplineParameters) = 1
function _nl_fun_in_dim(p::NNParameters)
    hasproperty(p.chain, :layers) || return nothing
    layers = p.chain.layers
    isempty(layers) && return nothing
    l = first(layers)
    return hasproperty(l, :in_dims) ? l.in_dims : nothing
end

_nl_fun_out_dim(p) = nothing
_nl_fun_out_dim(p::SoftTreeParameters) = p.n_output

# Collect `f(arg, θ)` and `f(arg, θ)[k]` for model-function names `f`.
function _nl_collect_fun_calls!(out::Vector{Tuple{Symbol, Any, Union{Nothing, Int}}},
        ex, fun_syms::Set{Symbol}, idx::Union{Nothing, Int} = nothing)
    ex isa Expr || return out
    if ex.head == :ref && length(ex.args) == 2 && ex.args[2] isa Integer
        _nl_collect_fun_calls!(out, ex.args[1], fun_syms, Int(ex.args[2]))
        return out
    end
    if ex.head == :call && ex.args[1] isa Symbol && ex.args[1] in fun_syms &&
       length(ex.args) == 3
        push!(out, (ex.args[1], ex.args[2], idx))
    end
    for a in ex.args
        _nl_collect_fun_calls!(out, a, fun_syms, nothing)
    end
    return out
end

# Call-shape mistakes on NN / soft-tree / spline model functions used to surface as a raw
# DimensionMismatch/MethodError/BoundsError from inside the AD-typed objective (#206,
# #209, #214, #215). The check is static, so it costs nothing at fit time.
function _nl_check_model_fun_calls(exprs, fixed)
    params = get_params(fixed)
    by_fun = Dict{Symbol, Any}()
    for name in keys(params)
        p = getproperty(params, name)
        hasproperty(p, :function_name) && (by_fun[p.function_name] = p)
    end
    isempty(by_fun) && return nothing

    calls = Tuple{Symbol, Any, Union{Nothing, Int}}[]
    for ex in exprs
        _nl_collect_fun_calls!(calls, ex, Set(keys(by_fun)))
    end
    for (fun, arg, idx) in calls
        p = by_fun[fun]
        want_in = _nl_fun_in_dim(p)
        if want_in !== nothing
            got = arg isa Expr && arg.head == :vect ? length(arg.args) :
                  (arg isa Expr && arg.head == :call && arg.args[1] === :vcat ?
                   length(arg.args) - 1 : nothing)
            if got === nothing
                # A bare scalar argument to a multi-input function is always wrong.
                (arg isa Symbol || arg isa Number) && want_in > 1 &&
                    error("$(fun) (parameter $(p.name)) expects a length-$(want_in) input vector; got the scalar `$(arg)`. Write $(fun)([a, b, ...], $(p.name)).")
            elseif got != want_in
                error("$(fun) (parameter $(p.name)) expects a length-$(want_in) input vector; got length $(got).")
            end
            want_in == 1 && p isa SplineParameters && arg isa Expr &&
                arg.head == :vect &&
                error("$(fun) (parameter $(p.name)) takes a scalar input; got a length-$(length(arg.args)) vector.")
        end
        want_out = _nl_fun_out_dim(p)
        want_out === nothing || idx === nothing || idx <= want_out ||
            error("$(fun) (parameter $(p.name)) returns $(want_out) output(s); index [$(idx)] is out of range.")
    end
    return nothing
end

function _validate_model_symbols(formulas, random, covariates, fixed;
        fixed_names, re_names, prede_names, const_cov_names, varying_cov_names,
        helper_names, model_fun_names, state_names, signal_names, context_module)
    _nl_check_reserved_names(fixed_names, "Fixed effects")
    _nl_check_reserved_names(re_names, "Random effects")
    # Covariates are exempt: `t = Covariate()` is how the time column is declared.

    ir = get_formulas_ir(formulas)
    isempty(ir.obs_names) &&
        error("@formulas defines no observation: add at least one `outcome ~ Distribution(...)` line. (A line written with `=` defines a deterministic node, not an observation.)")

    for i in eachindex(ir.det_names)
        rhs = ir.det_exprs[i]
        rhs isa Expr && rhs.head == :call && rhs.args[1] isa Symbol &&
            isdefined(Distributions, rhs.args[1]) &&
            getfield(Distributions, rhs.args[1]) isa Type &&
            getfield(Distributions, rhs.args[1]) <: Distribution &&
            error("@formulas node `$(ir.det_names[i])` assigns a distribution with `=`. Use `~` to declare an observation: `$(ir.det_names[i]) ~ $(rhs.args[1])(...)`.")
    end

    const_cov_set = Set(const_cov_names)
    called_consts = sort([s for s in ir.call_heads if s in const_cov_set])
    isempty(called_consts) ||
        error("Constant covariate(s) $(join(string.(called_consts), ", ")) are called like functions in @formulas. Use them as variables (e.g. `x` or `x.field`), or declare them as DynamicCovariate and call `w(t)`.")

    known = Set{Symbol}(vcat(fixed_names, re_names, prede_names, const_cov_names,
        varying_cov_names, helper_names, model_fun_names, state_names, signal_names,
        ir.det_names, ir.obs_names))
    _nl_check_covariate_fields(vcat(ir.det_exprs, ir.obs_exprs), covariates)
    _nl_check_model_fun_calls(vcat(ir.det_exprs, ir.obs_exprs), fixed)

    unknown = _nl_unknown_syms(
        vcat(ir.det_exprs, ir.obs_exprs), known, context_module)
    isempty(unknown) ||
        error("@formulas references undefined symbol(s) $(join(string.(unknown), ", ")). They are not a fixed effect, random effect, pre-DE variable, covariate, helper, model function or DE state/signal.")

    # Varying covariates in an RE distribution are rejected later, by DataModel, with a
    # message about `constant_on` -- do not pre-empt it here.
    re_known = Set{Symbol}(vcat(fixed_names, const_cov_names, varying_cov_names,
        helper_names, model_fun_names))
    re_syms = get_re_syms(random)
    for name in re_names
        bad = sort([s
                    for s in getproperty(re_syms, name)
                    if !(s in re_known) && !_nl_symbol_resolvable(s, context_module)])
        isempty(bad) ||
            error("RandomEffect `$(name)` references undefined symbol(s) $(join(string.(bad), ", ")). They are not a fixed effect, constant covariate, helper or model function.")
    end
    return nothing
end
