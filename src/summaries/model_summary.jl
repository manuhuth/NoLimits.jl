export ModelSummary
export summarize

"""
    ModelSummary

Structured summary of a [`Model`](@ref). Created by `summarize(model)`.

Provides counts and lists of all model components: fixed effects, random effects,
covariates, deterministic formulas, outcome distributions, and differential equation
states/signals. Displayed via `Base.show`.
"""
struct ModelSummary
    model_type::Symbol
    has_helpers::Bool
    has_fixed_effects::Bool
    has_random_effects::Bool
    has_covariates::Bool
    has_prede::Bool
    has_de::Bool
    has_initialde::Bool
    n_fixed_effect_blocks::Int
    n_fixed_effect_values::Int
    n_random_effects::Int
    n_random_effect_group_columns::Int
    n_covariates::Int
    n_covariates_varying::Int
    n_covariates_constant::Int
    n_covariates_dynamic::Int
    n_deterministic_formulas::Int
    n_outcomes::Int
    requires_de_accessors::Bool
    fixed_effect_summaries::Vector{NamedTuple}
    random_effect_summaries::Vector{NamedTuple}
    covariate_summaries::Vector{NamedTuple}
    deterministic_formula_names::Vector{Symbol}
    outcome_names::Vector{Symbol}
    outcome_distribution_types::NamedTuple
    required_states::Vector{Symbol}
    required_signals::Vector{Symbol}
    de_states::Vector{Symbol}
    de_signals::Vector{Symbol}
    helper_names::Vector{Symbol}
end

function _ms_call_head_symbol(f)
    if f isa Symbol
        return f
    elseif f isa GlobalRef
        return f.name
    elseif f isa Expr && f.head == :.
        last = f.args[end]
        last isa QuoteNode && (last = last.value)
        return last isa Symbol ? last : :unknown
    end
    return :unknown
end

function _ms_distribution_type_from_expr(ex)
    ex isa Expr || return :unknown
    ex.head == :call || return :unknown
    return _ms_call_head_symbol(ex.args[1])
end

function _ms_namedtuple_from_symbols(names::Vector{Symbol}, vals::Vector)
    isempty(names) && return NamedTuple()
    return NamedTuple{Tuple(names)}(Tuple(vals))
end

function _ms_resolve_re_distribution_types(re_model)
    re_names = get_re_names(re_model)
    raw_types = get_re_types(re_model)
    dist_exprs = get_re_dist_exprs(re_model)
    vals = Symbol[]
    for re in re_names
        t = getfield(raw_types, re)
        if t === :unknown
            ex = getfield(dist_exprs, re)
            t = _ms_distribution_type_from_expr(ex)
        end
        push!(vals, t)
    end
    return _ms_namedtuple_from_symbols(re_names, vals)
end

function _ms_prior_type(prior)
    prior isa Priorless && return "Priorless"
    if prior isa AbstractVector
        isempty(prior) && return "Vector"
        names = unique(string(nameof(typeof(p))) for p in prior)
        return length(names) == 1 ? "Vector{$(only(names))}" : "Vector{mixed}"
    end
    return string(nameof(typeof(prior)))
end

function _ms_scale_summary(p)
    hasproperty(p, :scale) || return "n/a"
    s = getfield(p, :scale)
    if s isa Symbol
        return String(s)
    elseif s isa AbstractVector
        counts = Dict{Symbol, Int}()
        for si in s
            si isa Symbol || continue
            counts[si] = get(counts, si, 0) + 1
        end
        isempty(counts) && return "n/a"
        parts = String[]
        for k in sort(collect(keys(counts)))
            push!(parts, "$(k)x$(counts[k])")
        end
        return join(parts, ", ")
    end
    return "n/a"
end

function _ms_finite_count(xs)
    c = 0
    n = 0
    for x in xs
        x isa Real || continue
        n += 1
        isfinite(Float64(x)) && (c += 1)
    end
    return c, n
end

function _ms_bounds_summary(p)
    if !(hasproperty(p, :lower) && hasproperty(p, :upper))
        return "n/a"
    end
    lower = getfield(p, :lower)
    upper = getfield(p, :upper)
    lvals = lower isa Number ? (lower,) : Tuple(lower)
    uvals = upper isa Number ? (upper,) : Tuple(upper)
    cl, nl = _ms_finite_count(lvals)
    cu, nu = _ms_finite_count(uvals)
    n = max(nl, nu)
    n == 0 && return "n/a"
    return "finite lower $(cl)/$(n), finite upper $(cu)/$(n)"
end

function _ms_size_summary(p)
    val = hasproperty(p, :value) ? getfield(p, :value) : nothing
    if val === nothing
        return "n/a"
    elseif val isa Number
        return "1"
    elseif val isa AbstractVector
        return string(length(val))
    elseif val isa AbstractMatrix
        return string(size(val, 1), "x", size(val, 2))
    end
    return "n/a"
end

function _ms_details(p)
    if p isa NNParameters
        return "function=$(p.function_name), weights=$(length(p.value))"
    elseif p isa SoftTreeParameters
        return "function=$(p.function_name), input_dim=$(p.input_dim), depth=$(p.depth), outputs=$(p.n_output), weights=$(length(p.value))"
    elseif p isa SplineParameters
        return "function=$(p.function_name), degree=$(p.degree), knots=$(length(p.knots)), coeffs=$(length(p.value))"
    elseif p isa NPFParameter
        return "input_dim=$(p.n_input), n_layers=$(p.n_layers), weights=$(length(p.value))"
    end
    return "-"
end

function _ms_interp_name(itp)
    s = string(itp)
    i = findfirst('{', s)
    return i === nothing ? s : s[1:(i - 1)]
end

function _ms_cov_summary(p)
    if p isa Covariate
        return (kind=:Covariate, columns=Symbol[p.column], constant_on=Symbol[], interpolation="-")
    elseif p isa CovariateVector
        return (kind=:CovariateVector, columns=collect(p.columns), constant_on=Symbol[], interpolation="-")
    elseif p isa ConstantCovariate
        return (kind=:ConstantCovariate, columns=Symbol[p.column], constant_on=collect(p.constant_on), interpolation="-")
    elseif p isa ConstantCovariateVector
        return (kind=:ConstantCovariateVector, columns=collect(p.columns), constant_on=collect(p.constant_on), interpolation="-")
    elseif p isa DynamicCovariate
        interp = _ms_interp_name(p.interpolation)
        return (kind=:DynamicCovariate, columns=Symbol[p.column], constant_on=Symbol[], interpolation=interp)
    elseif p isa DynamicCovariateVector
        names = String[]
        for itp in p.interpolations
            push!(names, _ms_interp_name(itp))
        end
        return (kind=:DynamicCovariateVector, columns=collect(p.columns), constant_on=Symbol[], interpolation=join(unique(names), ", "))
    end
    return (kind=Symbol(nameof(typeof(p))), columns=Symbol[], constant_on=Symbol[], interpolation="-")
end

function _ms_print_key_values(io::IO, title::String, rows::AbstractVector{<:Pair})
    println(io, title)
    isempty(rows) && (println(io, "  (none)"); return)
    keys_str = [string(first(r)) for r in rows]
    w = maximum(length, keys_str)
    for (i, r) in enumerate(rows)
        println(io, "  ", rpad(keys_str[i], w), " : ", last(r))
    end
end

function _ms_print_distribution_types(io::IO, title::String, nt::NamedTuple)
    println(io, title)
    ks = collect(keys(nt))
    if isempty(ks)
        println(io, "  (none)")
        return
    end
    w = maximum(length(string(k)) for k in ks)
    for k in ks
        println(io, "  ", rpad(string(k), w), " => ", getfield(nt, k))
    end
end

function _ms_join_syms(xs::Vector{Symbol})
    isempty(xs) && return "(none)"
    return join(string.(xs), ", ")
end

"""
    summarize(m::Model) -> ModelSummary
    summarize(dm::DataModel) -> DataModelSummary
    summarize(res::FitResult; scale, include_non_se, constants_re) -> FitResultSummary
    summarize(uq::UQResult; scale) -> UQResultSummary
    summarize(res::FitResult, uq::UQResult; scale, include_non_se, constants_re) -> FitResultSummary

Compute a structured summary of a model, data model, fit result, or UQ result.

Each overload returns a specialised summary struct that has a pretty-printed `show`
method for interactive inspection.

# Keyword Arguments (for fit/UQ overloads)
- `scale::Symbol = :natural`: parameter scale to report (`:natural` or `:transformed`).
- `include_non_se::Bool = false`: include parameters marked `calculate_se=false`.
- `constants_re::NamedTuple = NamedTuple()`: constants for random-effects reporting.
"""
function summarize(m::Model)
    has_prede = m.de.prede !== nothing
    has_de = m.de.de !== nothing
    has_initialde = m.de.initial !== nothing
    model_type = has_de ? :ode : :non_ode

    fe = m.fixed.fixed
    fe_names = get_names(fe)
    fe_params = get_params(fe)
    n_fixed_effect_blocks = length(fe_names)
    n_fixed_effect_values = length(get_θ0_untransformed(fe))

    fixed_effect_summaries = NamedTuple[]
    for name in fe_names
        p = getfield(fe_params, name)
        push!(fixed_effect_summaries, (;
            name=name,
            block_type=Symbol(nameof(typeof(p))),
            size=_ms_size_summary(p),
            calculate_se=hasproperty(p, :calculate_se) ? getfield(p, :calculate_se) : false,
            prior_type=_ms_prior_type(hasproperty(p, :prior) ? getfield(p, :prior) : Priorless()),
            scale=_ms_scale_summary(p),
            bounds=_ms_bounds_summary(p),
            details=_ms_details(p),
        ))
    end

    re_model = m.random.random
    re_names = get_re_names(re_model)
    re_groups = get_re_groups(re_model)
    re_types = _ms_resolve_re_distribution_types(re_model)
    random_effect_summaries = NamedTuple[]
    for re in re_names
        push!(random_effect_summaries, (; name=re, group=getfield(re_groups, re), dist_type=getfield(re_types, re)))
    end
    n_random_effects = length(re_names)
    n_random_effect_group_columns = length(unique([getfield(re_groups, n) for n in re_names]))

    cov = m.covariates.covariates
    covariate_summaries = NamedTuple[]
    for cname in cov.names
        p = getfield(cov.params, cname)
        cv = _ms_cov_summary(p)
        push!(covariate_summaries, (;
            name=cname,
            kind=cv.kind,
            columns=cv.columns,
            constant_on=cv.constant_on,
            interpolation=cv.interpolation,
        ))
    end

    formulas = m.formulas.formulas
    ir = get_formulas_ir(formulas)
    outcome_dist_types = _ms_namedtuple_from_symbols(ir.obs_names, [_ms_distribution_type_from_expr(ex) for ex in ir.obs_exprs])

    helper_names = Symbol[collect(keys(m.helpers.funcs))...]
    required_states = copy(m.formulas.required_states)
    required_signals = copy(m.formulas.required_signals)
    de_states = has_de ? copy(get_de_states(m.de.de)) : Symbol[]
    de_signals = has_de ? copy(get_de_signals(m.de.de)) : Symbol[]

    return ModelSummary(
        model_type,
        !isempty(helper_names),
        n_fixed_effect_blocks > 0,
        n_random_effects > 0,
        !isempty(cov.names),
        has_prede,
        has_de,
        has_initialde,
        n_fixed_effect_blocks,
        n_fixed_effect_values,
        n_random_effects,
        n_random_effect_group_columns,
        length(cov.names),
        length(cov.varying),
        length(cov.constants),
        length(cov.dynamic),
        length(ir.det_names),
        length(ir.obs_names),
        !isempty(required_states) || !isempty(required_signals),
        fixed_effect_summaries,
        random_effect_summaries,
        covariate_summaries,
        copy(ir.det_names),
        copy(ir.obs_names),
        outcome_dist_types,
        required_states,
        required_signals,
        de_states,
        de_signals,
        helper_names,
    )
end

Base.show(io::IO, s::ModelSummary) = show(io, MIME"text/plain"(), s)

function Base.show(io::IO, ::MIME"text/plain", s::ModelSummary)
    println(io, "ModelSummary")
    println(io, repeat("═", 96))

    _ms_print_key_values(io, "Overview", [
        "model type" => (s.model_type == :ode ? "ODE" : "non-ODE"),
        "fixed-effect blocks" => s.n_fixed_effect_blocks,
        "fixed-effect scalar values" => s.n_fixed_effect_values,
        "random effects" => s.n_random_effects,
        "random-effect grouping columns" => s.n_random_effect_group_columns,
        "covariates (declared)" => s.n_covariates,
        "formulas (deterministic / outcomes)" => "$(s.n_deterministic_formulas) / $(s.n_outcomes)",
        "requires DE accessors" => s.requires_de_accessors,
    ])
    println(io)

    _ms_print_key_values(io, "Structure blocks", [
        "helpers" => s.has_helpers,
        "fixed effects" => s.has_fixed_effects,
        "random effects" => s.has_random_effects,
        "covariates" => s.has_covariates,
        "preDE" => s.has_prede,
        "DifferentialEquation" => s.has_de,
        "initialDE" => s.has_initialde,
    ])
    println(io)

    _ms_print_key_values(io, "Covariate classes", [
        "varying" => s.n_covariates_varying,
        "constant" => s.n_covariates_constant,
        "dynamic" => s.n_covariates_dynamic,
    ])
    println(io)

    println(io, "Fixed-effects declarations")
    if isempty(s.fixed_effect_summaries)
        println(io, "  (none)")
    else
        name_w = max(length("name"), maximum(length(string(r.name)) for r in s.fixed_effect_summaries))
        type_w = max(length("type"), maximum(length(string(r.block_type)) for r in s.fixed_effect_summaries))
        size_w = max(length("size"), maximum(length(string(r.size)) for r in s.fixed_effect_summaries))
        se_w = length("se")
        prior_w = max(length("prior"), maximum(length(string(r.prior_type)) for r in s.fixed_effect_summaries))
        scale_w = max(length("scale"), maximum(length(string(r.scale)) for r in s.fixed_effect_summaries))
        bounds_w = max(length("bounds"), maximum(length(string(r.bounds)) for r in s.fixed_effect_summaries))
        println(io, "  ",
                rpad("name", name_w), "  ",
                rpad("type", type_w), "  ",
                lpad("size", size_w), "  ",
                rpad("se", se_w), "  ",
                rpad("prior", prior_w), "  ",
                rpad("scale", scale_w), "  ",
                rpad("bounds", bounds_w), "  details")
        println(io, "  ", repeat("-", name_w + type_w + size_w + se_w + prior_w + scale_w + bounds_w + 34))
        for r in s.fixed_effect_summaries
            println(io, "  ",
                    rpad(string(r.name), name_w), "  ",
                    rpad(string(r.block_type), type_w), "  ",
                    lpad(string(r.size), size_w), "  ",
                    rpad(r.calculate_se ? "yes" : "no", se_w), "  ",
                    rpad(string(r.prior_type), prior_w), "  ",
                    rpad(string(r.scale), scale_w), "  ",
                    rpad(string(r.bounds), bounds_w), "  ",
                    r.details)
        end
    end
    println(io)

    println(io, "Random-effects declarations")
    if isempty(s.random_effect_summaries)
        println(io, "  (none)")
    else
        name_w = max(length("name"), maximum(length(string(r.name)) for r in s.random_effect_summaries))
        grp_w = max(length("group"), maximum(length(string(r.group)) for r in s.random_effect_summaries))
        dst_w = max(length("dist"), maximum(length(string(r.dist_type)) for r in s.random_effect_summaries))
        println(io, "  ",
                rpad("name", name_w), "  ",
                rpad("group", grp_w), "  ",
                rpad("dist", dst_w))
        println(io, "  ", repeat("-", name_w + grp_w + dst_w + 6))
        for r in s.random_effect_summaries
            println(io, "  ",
                    rpad(string(r.name), name_w), "  ",
                    rpad(string(r.group), grp_w), "  ",
                    rpad(string(r.dist_type), dst_w))
        end
    end
    println(io)

    println(io, "Covariate declarations")
    if isempty(s.covariate_summaries)
        println(io, "  (none)")
    else
        name_w = max(length("name"), maximum(length(string(r.name)) for r in s.covariate_summaries))
        kind_w = max(length("kind"), maximum(length(string(r.kind)) for r in s.covariate_summaries))
        println(io, "  ",
                rpad("name", name_w), "  ",
                rpad("kind", kind_w), "  ",
                rpad("columns", 24), "  ",
                rpad("constant_on", 20), "  interpolation")
        println(io, "  ", repeat("-", name_w + kind_w + 74))
        for r in s.covariate_summaries
            cols = isempty(r.columns) ? "-" : join(string.(r.columns), ", ")
            const_on = isempty(r.constant_on) ? "-" : join(string.(r.constant_on), ", ")
            println(io, "  ",
                    rpad(string(r.name), name_w), "  ",
                    rpad(string(r.kind), kind_w), "  ",
                    rpad(cols, 24), "  ",
                    rpad(const_on, 20), "  ",
                    r.interpolation)
        end
    end
    println(io)

    _ms_print_key_values(io, "Formulas", [
        "deterministic names" => _ms_join_syms(s.deterministic_formula_names),
        "outcome names" => _ms_join_syms(s.outcome_names),
        "required DE states" => _ms_join_syms(s.required_states),
        "required DE signals" => _ms_join_syms(s.required_signals),
        "declared DE states" => _ms_join_syms(s.de_states),
        "declared DE signals" => _ms_join_syms(s.de_signals),
    ])
    _ms_print_distribution_types(io, "Outcome distribution types", s.outcome_distribution_types)
    println(io)

    _ms_print_key_values(io, "Helper functions", [
        "names" => _ms_join_syms(s.helper_names),
    ])
end
