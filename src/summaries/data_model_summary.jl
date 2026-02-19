export DescriptiveStats
export DataModelSummary
export summarize

using Statistics

"""
    DescriptiveStats

Descriptive statistics for a numeric variable. Contains `n`, `mean`, `sd`, `min`,
`q25`, `median`, `q75`, and `max`.
"""
struct DescriptiveStats
    n::Int
    mean::Float64
    sd::Float64
    min::Float64
    q25::Float64
    median::Float64
    q75::Float64
    max::Float64
end

"""
    DataModelSummary

Structured summary of a [`DataModel`](@ref). Created by `summarize(dm)`.

Provides individual-level, covariate, outcome, and random-effects statistics, as
well as data-quality checks (duplicate times, non-monotonic time, missing values).
Displayed via `Base.show`.
"""
struct DataModelSummary
    model_type::Symbol
    has_events::Bool
    n_individuals::Int
    n_rows_total::Int
    n_obs_rows::Int
    n_event_rows::Int
    n_fixed_effects::Int
    n_outcomes::Int
    n_covariates::Int
    n_covariates_varying::Int
    n_covariates_constant::Int
    n_covariates_dynamic::Int
    n_random_effects::Int
    n_single_obs_individuals::Int
    obs_per_individual::DescriptiveStats
    time_span_per_individual::DescriptiveStats
    median_dt_per_individual::DescriptiveStats
    global_time_min::Float64
    global_time_max::Float64
    n_unique_obs_times::Int
    n_duplicate_id_time_obs::Int
    n_monotonic_time_violations::Int
    outcome_distribution_types::NamedTuple
    random_effect_distribution_types::NamedTuple
    outcome_stats::Vector{NamedTuple}
    covariate_declarations::Vector{NamedTuple}
    covariate_stats::Vector{NamedTuple}
    nonnumeric_covariate_columns::Vector{String}
    random_effect_summaries::Vector{NamedTuple}
end

@inline _nan_stats() = DescriptiveStats(0, NaN, NaN, NaN, NaN, NaN, NaN, NaN)

function _descriptive_stats(values)
    vals = Float64[]
    for v in values
        if v === missing
            continue
        elseif v isa Real
            push!(vals, Float64(v))
        end
    end
    vals = [x for x in vals if isfinite(x)]
    isempty(vals) && return _nan_stats()
    q25 = quantile(vals, 0.25)
    q50 = quantile(vals, 0.50)
    q75 = quantile(vals, 0.75)
    return DescriptiveStats(
        length(vals),
        mean(vals),
        std(vals; corrected=false),
        minimum(vals),
        q25,
        q50,
        q75,
        maximum(vals),
    )
end

function _collect_obs_rows(dm::DataModel)
    n = sum(length, dm.row_groups.obs_rows)
    out = Vector{Int}(undef, n)
    k = 1
    for rows in dm.row_groups.obs_rows
        for r in rows
            out[k] = r
            k += 1
        end
    end
    return out
end

function _call_head_symbol(f)
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

function _distribution_type_from_expr(ex)
    ex isa Expr || return :unknown
    ex.head == :call || return :unknown
    return _call_head_symbol(ex.args[1])
end

function _resolve_re_distribution_types(re_model)
    re_names = get_re_names(re_model)
    raw_types = get_re_types(re_model)
    dist_exprs = get_re_dist_exprs(re_model)
    vals = Symbol[]
    for re in re_names
        t = getfield(raw_types, re)
        if t === :unknown
            ex = getfield(dist_exprs, re)
            t = _distribution_type_from_expr(ex)
        end
        push!(vals, t)
    end
    return _namedtuple_from_symbols(re_names, vals)
end

function _namedtuple_from_symbols(names::Vector{Symbol}, vals::Vector)
    isempty(names) && return NamedTuple()
    return NamedTuple{Tuple(names)}(Tuple(vals))
end

function _covariate_kind_and_columns(p)
    if p isa Covariate
        return :Covariate, Symbol[p.column], Symbol[]
    elseif p isa CovariateVector
        return :CovariateVector, collect(p.columns), Symbol[]
    elseif p isa ConstantCovariate
        return :ConstantCovariate, Symbol[p.column], collect(p.constant_on)
    elseif p isa ConstantCovariateVector
        return :ConstantCovariateVector, collect(p.columns), collect(p.constant_on)
    elseif p isa DynamicCovariate
        return :DynamicCovariate, Symbol[p.column], Symbol[]
    elseif p isa DynamicCovariateVector
        return :DynamicCovariateVector, collect(p.columns), Symbol[]
    end
    return Symbol(nameof(typeof(p))), Symbol[], Symbol[]
end

function _obs_time_values(dm::DataModel, obs_rows::Vector{Int})
    tcol = getproperty(dm.df, dm.config.time_col)
    vals = Float64[]
    for r in obs_rows
        v = tcol[r]
        (v === missing || !(v isa Real)) && continue
        push!(vals, Float64(v))
    end
    return vals
end

function _format_float(x::Float64)
    if isnan(x)
        return "NaN"
    end
    ax = abs(x)
    if ax >= 1e4 || (ax > 0 && ax < 1e-3)
        return string(round(x; sigdigits=4))
    end
    return string(round(x; digits=4))
end

function _print_key_values(io::IO, title::String, rows::AbstractVector{<:Pair})
    println(io, title)
    if isempty(rows)
        println(io, "  (none)")
        return
    end
    keys_str = [string(first(r)) for r in rows]
    w = maximum(length, keys_str)
    for (i, r) in enumerate(rows)
        v = last(r)
        println(io, "  ", rpad(keys_str[i], w), " : ", v)
    end
end

function _print_distribution_types(io::IO, title::String, nt::NamedTuple)
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

function _print_descriptive_table(io::IO, title::String, rows::AbstractVector; name_header::String="Variable")
    println(io, title)
    if isempty(rows)
        println(io, "  (none)")
        return
    end
    name_w = max(length(name_header), maximum(length(string(row.name)) for row in rows))
    println(io,
            "  ",
            rpad(name_header, name_w), "  ",
            lpad("n", 6), "  ",
            lpad("mean", 12), "  ",
            lpad("sd", 12), "  ",
            lpad("min", 12), "  ",
            lpad("q25", 12), "  ",
            lpad("median", 12), "  ",
            lpad("q75", 12), "  ",
            lpad("max", 12))
    println(io, "  ", repeat("-", name_w + 7 * 14 + 8))
    for row in rows
        s = row.stats
        println(io,
                "  ",
                rpad(string(row.name), name_w), "  ",
                lpad(string(s.n), 6), "  ",
                lpad(_format_float(s.mean), 12), "  ",
                lpad(_format_float(s.sd), 12), "  ",
                lpad(_format_float(s.min), 12), "  ",
                lpad(_format_float(s.q25), 12), "  ",
                lpad(_format_float(s.median), 12), "  ",
                lpad(_format_float(s.q75), 12), "  ",
                lpad(_format_float(s.max), 12))
    end
end

function _print_covariate_declarations(io::IO, decls::AbstractVector)
    println(io, "Declared covariates")
    if isempty(decls)
        println(io, "  (none)")
        return
    end
    name_w = max(length("name"), maximum(length(string(d.name)) for d in decls))
    kind_w = max(length("kind"), maximum(length(string(d.kind)) for d in decls))
    println(io, "  ", rpad("name", name_w), "  ", rpad("kind", kind_w), "  columns")
    println(io, "  ", repeat("-", name_w + kind_w + 24))
    for d in decls
        cols = isempty(d.columns) ? "(none)" : join(string.(d.columns), ", ")
        println(io, "  ", rpad(string(d.name), name_w), "  ", rpad(string(d.kind), kind_w), "  ", cols)
    end
end

function summarize(dm::DataModel)
    obs_rows = _collect_obs_rows(dm)
    n_obs_rows = length(obs_rows)
    n_individuals = length(dm.individuals)
    n_rows_total = length(getproperty(dm.df, dm.config.primary_id))
    has_events = dm.config.evid_col !== nothing
    n_event_rows = has_events ? max(n_rows_total - n_obs_rows, 0) : 0
    model_type = dm.model.de.de === nothing ? :non_ode : :ode

    fe = dm.model.fixed.fixed
    n_fixed_effects = length(get_names(fe))

    cov = dm.model.covariates.covariates
    n_covariates = length(cov.names)
    n_covariates_varying = length(cov.varying)
    n_covariates_constant = length(cov.constants)
    n_covariates_dynamic = length(cov.dynamic)

    formulas = dm.model.formulas.formulas
    ir = get_formulas_ir(formulas)
    obs_names = ir.obs_names
    n_outcomes = length(obs_names)
    outcome_dist_types = _namedtuple_from_symbols(obs_names, [_distribution_type_from_expr(ex) for ex in ir.obs_exprs])

    re_model = dm.model.random.random
    re_names = get_re_names(re_model)
    n_random_effects = length(re_names)
    re_types = _resolve_re_distribution_types(re_model)
    re_groups = get_re_groups(re_model)

    # Individual-level design stats
    obs_per_ind = Float64[]
    time_spans = Float64[]
    median_dts = Float64[]
    tcol = getproperty(dm.df, dm.config.time_col)
    for rows in dm.row_groups.obs_rows
        push!(obs_per_ind, Float64(length(rows)))
        if isempty(rows)
            push!(time_spans, NaN)
            push!(median_dts, NaN)
            continue
        end
        ts = Float64[tcol[r] for r in rows]
        push!(time_spans, maximum(ts) - minimum(ts))
        if length(ts) < 2
            push!(median_dts, NaN)
        else
            u = sort(unique(ts))
            if length(u) < 2
                push!(median_dts, NaN)
            else
                push!(median_dts, median(diff(u)))
            end
        end
    end

    obs_per_ind_stats = _descriptive_stats(obs_per_ind)
    time_span_stats = _descriptive_stats(time_spans)
    median_dt_stats = _descriptive_stats(median_dts)
    n_single_obs = count(==(1), Int.(round.(obs_per_ind)))

    # Time diagnostics on observation rows
    obs_t = _obs_time_values(dm, obs_rows)
    global_time_min = isempty(obs_t) ? NaN : minimum(obs_t)
    global_time_max = isempty(obs_t) ? NaN : maximum(obs_t)
    n_unique_obs_times = length(unique(obs_t))

    idcol = getproperty(dm.df, dm.config.primary_id)
    n_duplicate_id_time_obs = 0
    seen = Set{Tuple{Any, Any}}()
    for r in obs_rows
        k = (idcol[r], tcol[r])
        if k in seen
            n_duplicate_id_time_obs += 1
        else
            push!(seen, k)
        end
    end

    n_monotonic_time_violations = 0
    for rows in dm.row_groups.obs_rows
        for j in 2:length(rows)
            if tcol[rows[j]] < tcol[rows[j - 1]]
                n_monotonic_time_violations += 1
            end
        end
    end

    # Outcome descriptive stats on observation rows only
    outcome_stats = NamedTuple[]
    for obs in dm.config.obs_cols
        col = getproperty(dm.df, obs)
        vals = col[obs_rows]
        push!(outcome_stats, (; name=obs, stats=_descriptive_stats(vals)))
    end

    # Declared covariates + per-column stats (observation rows only)
    covariate_declarations = NamedTuple[]
    covariate_stats = NamedTuple[]
    nonnumeric_covariate_columns = String[]
    for cname in cov.names
        p = getfield(cov.params, cname)
        kind, cols, constant_on = _covariate_kind_and_columns(p)
        push!(covariate_declarations, (; name=cname, kind=kind, columns=cols, constant_on=constant_on))
        for colname in cols
            col = getproperty(dm.df, colname)
            vals = col[obs_rows]
            st = _descriptive_stats(vals)
            if st.n == 0
                push!(nonnumeric_covariate_columns, string(cname, ".", colname))
            else
                push!(covariate_stats, (; name=Symbol(string(cname, ".", colname)), stats=st))
            end
        end
    end

    # Per-random-effect summary
    random_effect_summaries = NamedTuple[]
    if n_random_effects > 0
        values_nt = dm.re_group_info.values
        index_nt = dm.re_group_info.index_by_row
        for re in re_names
            levels = getfield(values_nt, re)
            n_levels = length(levels)
            counts = zeros(Float64, n_levels)
            idx_by_row = getfield(index_nt, re)
            for r in obs_rows
                idx = idx_by_row[r]
                1 <= idx <= n_levels || continue
                counts[idx] += 1
            end
            push!(random_effect_summaries, (;
                name=re,
                group=getfield(re_groups, re),
                dist_type=getfield(re_types, re),
                n_levels=n_levels,
                rows_per_level=_descriptive_stats(counts),
            ))
        end
    end

    return DataModelSummary(
        model_type,
        has_events,
        n_individuals,
        n_rows_total,
        n_obs_rows,
        n_event_rows,
        n_fixed_effects,
        n_outcomes,
        n_covariates,
        n_covariates_varying,
        n_covariates_constant,
        n_covariates_dynamic,
        n_random_effects,
        n_single_obs,
        obs_per_ind_stats,
        time_span_stats,
        median_dt_stats,
        global_time_min,
        global_time_max,
        n_unique_obs_times,
        n_duplicate_id_time_obs,
        n_monotonic_time_violations,
        outcome_dist_types,
        re_types,
        outcome_stats,
        covariate_declarations,
        covariate_stats,
        nonnumeric_covariate_columns,
        random_effect_summaries,
    )
end

Base.show(io::IO, s::DataModelSummary) = show(io, MIME"text/plain"(), s)

function Base.show(io::IO, ::MIME"text/plain", s::DataModelSummary)
    println(io, "DataModelSummary")
    println(io, repeat("═", 96))

    _print_key_values(io, "Overview", [
        "model type" => (s.model_type == :ode ? "ODE" : "non-ODE"),
        "event-aware" => s.has_events,
        "individuals" => s.n_individuals,
        "rows (total / obs / event)" => "$(s.n_rows_total) / $(s.n_obs_rows) / $(s.n_event_rows)",
        "fixed effects (top-level)" => s.n_fixed_effects,
        "outcomes" => s.n_outcomes,
        "covariates (declared)" => s.n_covariates,
        "random effects" => s.n_random_effects,
    ])
    println(io)

    _print_key_values(io, "Covariate classes", [
        "varying" => s.n_covariates_varying,
        "constant" => s.n_covariates_constant,
        "dynamic" => s.n_covariates_dynamic,
    ])
    println(io)

    _print_distribution_types(io, "Outcome distribution types", s.outcome_distribution_types)
    println(io)
    _print_distribution_types(io, "Random-effect distribution types", s.random_effect_distribution_types)
    println(io)

    _print_key_values(io, "Individual design diagnostics", [
        "individuals with one observation" => s.n_single_obs_individuals,
        "global observed time range" => "$( _format_float(s.global_time_min) ) to $( _format_float(s.global_time_max) )",
        "unique observed time points" => s.n_unique_obs_times,
        "duplicate (ID, time) observation rows" => s.n_duplicate_id_time_obs,
        "monotonic-time violations (observation order)" => s.n_monotonic_time_violations,
    ])
    println(io)

    _print_descriptive_table(io, "Observations per individual", [ (; name=:count, stats=s.obs_per_individual) ]; name_header="metric")
    println(io)
    _print_descriptive_table(io, "Time span per individual", [ (; name=:span, stats=s.time_span_per_individual) ]; name_header="metric")
    println(io)
    _print_descriptive_table(io, "Median sampling interval per individual", [ (; name=:median_dt, stats=s.median_dt_per_individual) ]; name_header="metric")
    println(io)

    _print_descriptive_table(io, "Outcome descriptive statistics (observation rows)", s.outcome_stats)
    println(io)
    _print_covariate_declarations(io, s.covariate_declarations)
    println(io)
    _print_descriptive_table(io, "Covariate descriptive statistics (observation rows)", s.covariate_stats)
    if !isempty(s.nonnumeric_covariate_columns)
        println(io)
        println(io, "Non-numeric/unsupported covariate columns (observation rows):")
        for lbl in s.nonnumeric_covariate_columns
            println(io, "  - ", lbl)
        end
    end

    if !isempty(s.random_effect_summaries)
        println(io)
        println(io, "Per-random-effect summary")
        name_w = max(length("random effect"), maximum(length(string(r.name)) for r in s.random_effect_summaries))
        grp_w = max(length("group"), maximum(length(string(r.group)) for r in s.random_effect_summaries))
        dst_w = max(length("dist"), maximum(length(string(r.dist_type)) for r in s.random_effect_summaries))
        println(io,
                "  ",
                rpad("random effect", name_w), "  ",
                rpad("group", grp_w), "  ",
                rpad("dist", dst_w), "  ",
                lpad("levels", 8), "  ",
                lpad("rows/level min", 14), "  ",
                lpad("median", 12), "  ",
                lpad("max", 12))
        println(io, "  ", repeat("-", name_w + grp_w + dst_w + 56))
        for r in s.random_effect_summaries
            st = r.rows_per_level
            println(io,
                    "  ",
                    rpad(string(r.name), name_w), "  ",
                    rpad(string(r.group), grp_w), "  ",
                    rpad(string(r.dist_type), dst_w), "  ",
                    lpad(string(r.n_levels), 8), "  ",
                    lpad(_format_float(st.min), 14), "  ",
                    lpad(_format_float(st.median), 12), "  ",
                    lpad(_format_float(st.max), 12))
        end
    end
end
