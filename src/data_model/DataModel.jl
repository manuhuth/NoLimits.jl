export DataModel
export DataModelConfig
export PairingInfo
export get_model
export get_df
export get_individuals
export get_individual
export get_batches
export get_batch_ids
export get_primary_id
export get_row_groups
export get_re_group_info
export get_re_indices

using DataInterpolations
using DiffEqCallbacks
using SciMLBase
using Distributions

"""
    DataModelConfig{S}

Configuration struct for a [`DataModel`](@ref), storing column names, serialisation
algorithm, and save-time mode.

# Fields
- `primary_id::Symbol`: primary individual-grouping column.
- `time_col::Symbol`: time column name.
- `evid_col::Union{Nothing, Symbol}`: EVID column name, or `nothing` for non-PKPD data.
- `amt_col::Symbol`: AMT column name (used when `evid_col` is set).
- `rate_col::Symbol`: RATE column name (used when `evid_col` is set).
- `cmt_col::Symbol`: CMT column name (used when `evid_col` is set).
- `obs_cols::Vector{Symbol}`: observation outcome column names.
- `serialization::S`: SciML ensemble algorithm (e.g. `EnsembleSerial()`, `EnsembleThreads()`).
- `saveat_mode::Symbol`: `:dense` or `:saveat` (resolved from `:auto` at construction time).
"""
struct DataModelConfig{S}
    primary_id::Symbol
    time_col::Symbol
    evid_col::Union{Nothing, Symbol}
    amt_col::Symbol
    rate_col::Symbol
    cmt_col::Symbol
    obs_cols::Vector{Symbol}
    serialization::S
    saveat_mode::Symbol
end

struct RowGroups{R, O}
    rows::R
    obs_rows::O
end

struct LaplaceREIndex{L, M}
    levels::L
    level_to_index::M
end

struct LaplaceRECache{N, D, S, I, X}
    re_names::N
    dims::D
    is_scalar::S
    ind_level_ids::I
    re_index::X
end

struct REGroupInfo{V, I, L}
    values::V
    index_by_row::I
    laplace_cache::L
end

"""
    PairingInfo

Stores the batch grouping of individuals derived by transitive union-find across
all random-effect grouping columns. Individuals that share any random-effect level
are placed in the same batch.

# Fields
- `batch_ids::Vector{Int}`: batch index for each individual (length = number of individuals).
- `batches::Vector{Vector{Int}}`: list of individual-index vectors, one per batch.
"""
struct PairingInfo
    batch_ids::Vector{Int}
    batches::Vector{Vector{Int}}
end

struct IndividualSeries{O, V, D}
    obs::O
    vary::V
    dyn::D
end

struct Individual{S, C, CB, TS, RG, SA}
    series::S
    const_cov::C
    callbacks::CB
    tspan::TS
    re_groups::RG
    saveat::SA
end

struct EventCallbacks{C, R, RS, B}
    callback::C
    infusion_rates::R
    init_resets::RS
    init_bolus::B
end

"""
    DataModel{M, D, I, P, C, K, G, R}

Top-level struct pairing a [`Model`](@ref) with a dataset. Produced by the
[`DataModel`](@ref) constructor and passed to [`fit_model`](@ref) and plotting functions.

Use accessor functions rather than accessing fields directly:
[`get_model`](@ref), [`get_df`](@ref), [`get_individuals`](@ref),
[`get_individual`](@ref), [`get_batches`](@ref), [`get_batch_ids`](@ref),
[`get_primary_id`](@ref), [`get_row_groups`](@ref), [`get_re_group_info`](@ref),
[`get_re_indices`](@ref).
"""
struct DataModel{M, D, I, P, C, K, G, R}
    model::M
    df::D
    individuals::I
    pairing::P
    config::C
    id_index::K
    row_groups::G
    re_group_info::R
end

function _nl_datamodel_show_line(dm::DataModel)
    n_individuals = length(dm.individuals)
    n_rows = size(dm.df, 1)
    model_type = dm.model.de.de === nothing ? :non_ode : :ode
    re_names = get_re_names(dm.model.random.random)
    outcomes = dm.config.obs_cols
    return "DataModel(n_individuals=$(n_individuals), n_rows=$(n_rows), model_type=$(model_type), primary_id=$(repr(dm.config.primary_id)), time_col=$(repr(dm.config.time_col)), n_random_effects=$(length(re_names)), outcomes=$(_nl_short_join_symbols(outcomes)))"
end

Base.show(io::IO, dm::DataModel) = print(io, _nl_datamodel_show_line(dm))
Base.show(io::IO, ::MIME"text/plain", dm::DataModel) = print(io, _nl_datamodel_show_line(dm))

mutable struct UnionFind
    parent::Vector{Int}
    size::Vector{Int}
end

UnionFind(n::Int) = UnionFind(collect(1:n), ones(Int, n))

function _uf_find(uf::UnionFind, x::Int)
    p = uf.parent[x]
    if p != x
        uf.parent[x] = _uf_find(uf, p)
    end
    return uf.parent[x]
end

function _uf_union!(uf::UnionFind, a::Int, b::Int)
    ra = _uf_find(uf, a)
    rb = _uf_find(uf, b)
    ra == rb && return
    if uf.size[ra] < uf.size[rb]
        ra, rb = rb, ra
    end
    uf.parent[rb] = ra
    uf.size[ra] += uf.size[rb]
end

function _get_col(df, name::Symbol)
    return getproperty(df, name)
end

function _require_cols(df, cols::Vector{Symbol})
    for c in cols
        hasproperty(df, c) || error("DataModel expects column $(c). Add it or change constructor keyword arguments.")
    end
end

function _require_col(df, col::Symbol, label::AbstractString)
    hasproperty(df, col) || error("DataModel requires $(label) column $(col) in the DataFrame. Provide it or override the corresponding keyword.")
end

function _check_missing(col, name::Symbol)
    any(ismissing, col) && error("Column $(name) contains missing values. Remove/replace missings before constructing DataModel.")
end

function _validate_schema(model, df, config::DataModelConfig)
    _require_col(df, config.primary_id, "primary id")
    _require_col(df, config.time_col, "time")
    _require_cols(df, config.obs_cols)

    _check_missing(_get_col(df, config.time_col), config.time_col)
    _check_missing(_get_col(df, config.primary_id), config.primary_id)

    if config.evid_col !== nothing
        _require_col(df, config.evid_col, "EVID")
        _require_col(df, config.amt_col, "AMT")
        _require_col(df, config.rate_col, "RATE")
        _require_col(df, config.cmt_col, "CMT")
        _check_missing(_get_col(df, config.evid_col), config.evid_col)
        evid = _get_col(df, config.evid_col)
        evt_idx = findall(!=(0), evid)
        if !isempty(evt_idx)
            _check_missing(_get_col(df, config.amt_col)[evt_idx], config.amt_col)
            _check_missing(_get_col(df, config.rate_col)[evt_idx], config.rate_col)
            _check_missing(_get_col(df, config.cmt_col)[evt_idx], config.cmt_col)
        end
    end

    # Observation columns should not be missing on EVID=0 rows
    if config.evid_col !== nothing
        evid = _get_col(df, config.evid_col)
        obs_idx = findall(==(0), evid)
        for c in config.obs_cols
            _check_missing(_get_col(df, c)[obs_idx], c)
        end
    else
        for c in config.obs_cols
            _check_missing(_get_col(df, c), c)
        end
    end

    # Check that obs_cols from @formulas exist in the data.
    formula_obs = get_formulas_meta(model.formulas.formulas).obs_names
    if !isempty(formula_obs)
        missing_cols = [s for s in formula_obs if !(s in config.obs_cols)]
        !isempty(missing_cols) && error("Missing observation columns $(missing_cols) required by @formulas. Add them to the data.")
    end
    _validate_formula_covariates_missing(model, df, config)
    return nothing
end

function _formula_used_covariates(model)
    covariates = model.covariates.covariates
    isempty(covariates.names) && return Symbol[]
    ir = get_formulas_ir(model.formulas.formulas)
    used = Set{Symbol}(vcat(ir.var_syms, ir.prop_syms))
    return [name for name in covariates.names if name in used]
end

function _check_covariate_missing(df, col::Symbol, idx, cov_name::Symbol, scope::AbstractString)
    data = idx === nothing ? _get_col(df, col) : _get_col(df, col)[idx]
    if any(ismissing, data)
        error("Covariate $(cov_name) uses column $(col) in @formulas, but $(col) contains missing values on $(scope). Remove/replace missings before constructing DataModel.")
    end
    return nothing
end

function _validate_formula_covariates_missing(model, df, config::DataModelConfig)
    used_covariates = _formula_used_covariates(model)
    isempty(used_covariates) && return nothing
    covariates = model.covariates.covariates
    params = covariates.params
    obs_idx = if config.evid_col === nothing
        nothing
    else
        evid = _get_col(df, config.evid_col)
        findall(==(0), evid)
    end
    obs_scope = config.evid_col === nothing ? "all rows" : "observation rows"
    for name in used_covariates
        p = getfield(params, name)
        if p isa Covariate
            _check_covariate_missing(df, p.column, obs_idx, name, obs_scope)
        elseif p isa CovariateVector
            for c in p.columns
                _check_covariate_missing(df, c, obs_idx, name, obs_scope)
            end
        elseif p isa ConstantCovariate
            _check_covariate_missing(df, p.column, nothing, name, "all rows")
        elseif p isa ConstantCovariateVector
            for c in p.columns
                _check_covariate_missing(df, c, nothing, name, "all rows")
            end
        elseif p isa DynamicCovariate
            _check_covariate_missing(df, p.column, nothing, name, "all rows")
        elseif p isa DynamicCovariateVector
            for c in p.columns
                _check_covariate_missing(df, c, nothing, name, "all rows")
            end
        end
    end
    return nothing
end

function _validate_time_col_covariate(model, config::DataModelConfig)
    time_col = config.time_col
    cov = model.covariates.covariates
    params = cov.params
    found = false
    bad = Symbol[]
    for name in cov.names
        p = getfield(params, name)
        if p isa Covariate || p isa DynamicCovariate
            if p.column == time_col || name == time_col
                found = true
            end
        elseif p isa CovariateVector || p isa DynamicCovariateVector
            if time_col in p.columns || name == time_col
                push!(bad, name)
            end
        elseif p isa ConstantCovariate
            if p.column == time_col || name == time_col
                push!(bad, name)
            end
        elseif p isa ConstantCovariateVector
            if time_col in p.columns || name == time_col
                push!(bad, name)
            end
        end
    end
    if !isempty(bad)
        error("time_col $(time_col) must be declared as Covariate() or DynamicCovariate(); it is declared as $(bad).")
    end
    found || error("time_col $(time_col) must be declared as Covariate() or DynamicCovariate() in @covariates.")
end

function _const_cov_columns(covariates, name::Symbol)
    p = getfield(covariates.params, name)
    if p isa ConstantCovariate
        return [p.column]
    elseif p isa ConstantCovariateVector
        return p.columns
    end
    return Symbol[]
end

function _check_constant_within_group(df, group_col::Symbol, cov_cols::Vector{Symbol})
    refs = Dict{Any, Int}()
    bad = Dict{Any, Vector{Symbol}}()
    gcol = _get_col(df, group_col)
    for i in eachindex(gcol)
        g = gcol[i]
        if !haskey(refs, g)
            refs[g] = i
            continue
        end
        ref = refs[g]
        for c in cov_cols
            if !isequal(_get_col(df, c)[i], _get_col(df, c)[ref])
                push!(get!(bad, g, Symbol[]), c)
            end
        end
    end
    return bad
end

function _validate_re_group_constants(model, df, covariates)
    re_names = get_re_names(model.random.random)
    isempty(re_names) && return nothing
    re_groups = get_re_groups(model.random.random)
    re_syms = get_re_syms(model.random.random)
    const_names = covariates.constants

    for re in re_names
        deps = [s for s in getfield(re_syms, re) if s in const_names]
        isempty(deps) && continue
        group_col = getfield(re_groups, re)
        cov_cols = Symbol[]
        for dep in deps
            p = getfield(covariates.params, dep)
            constant_on = p isa ConstantCovariate ? p.constant_on : (p isa ConstantCovariateVector ? p.constant_on : Symbol[])
            if !(group_col in constant_on)
                error("RandomEffect $(re) uses constant covariate $(dep), but $(dep) is not declared constant_on for group $(group_col).")
            end
            append!(cov_cols, _const_cov_columns(covariates, dep))
        end
        isempty(cov_cols) && continue
        bad = _check_constant_within_group(df, group_col, cov_cols)
        if !isempty(bad)
            details = join(["$(k) => $(unique(bad[k]))" for k in keys(bad)], ", ")
            cols_str = join(cov_cols, ", ")
            deps_str = join(deps, ", ")
            error("RandomEffect $(re) uses constant covariates [$(deps_str)] (columns: $(cols_str)) in its distribution, but these vary within grouping column $(group_col). Offending group values: $(details). Ensure those covariates are constant within each $(group_col) group, or remove them from the RE distribution.")
        end
    end
    return nothing
end

function _validate_constant_covariates_primary(model, df, primary_id::Symbol, covariates)
    const_names = covariates.constants
    isempty(const_names) && return nothing
    for name in const_names
        p = getfield(covariates.params, name)
        cov_cols = _const_cov_columns(covariates, name)
        isempty(cov_cols) && continue

        # Must be constant within primary_id
        bad_primary = _check_constant_within_group(df, primary_id, cov_cols)
        if !isempty(bad_primary)
            details = join(["$(k) => $(unique(bad_primary[k]))" for k in keys(bad_primary)], ", ")
            error("Constant covariate $(name) (columns: $(cov_cols)) varies within primary_id $(primary_id). Offending ids: $(details). Constant covariates must be constant within primary_id.")
        end

        # Must be constant within each constant_on group, if specified
        constant_on = p isa ConstantCovariate ? p.constant_on : (p isa ConstantCovariateVector ? p.constant_on : Symbol[])
        for grp in constant_on
            bad = _check_constant_within_group(df, grp, cov_cols)
            if !isempty(bad)
                details = join(["$(k) => $(unique(bad[k]))" for k in keys(bad)], ", ")
                error("Constant covariate $(name) (columns: $(cov_cols)) varies within constant_on group $(grp). Offending group values: $(details). Constant covariates must be constant within all constant_on groups and primary_id.")
            end
        end
    end
    return nothing
end

function _validate_prede_random_effects(model, primary_id::Symbol)
    prede = model.de.prede
    prede === nothing && return nothing
    re_names = get_re_names(model.random.random)
    isempty(re_names) && return nothing
    used = Set(get_prede_syms(prede))
    re_groups = get_re_groups(model.random.random)
    for re in re_names
        if re in used
            group_col = getfield(re_groups, re)
            if group_col != primary_id
                error("@preDifferentialEquation uses random effect $(re) grouped by $(group_col), which can vary within individuals. Only random effects grouped by the primary id $(primary_id) are allowed in preDE.")
            end
        end
    end
    return nothing
end

function _validate_re_dist_covariates(model, covariates)
    re = model.random.random
    re_syms = get_re_syms(re)
    cov_syms = Set(vcat(covariates.names, covariates.flat_names, covariates.constants, covariates.varying, covariates.dynamic))
    const_syms = Set(covariates.constants)
    for r in get_re_names(re)
        syms = getfield(re_syms, r)
        used = sort([s for s in syms if s in cov_syms])
        bad = [s for s in used if !(s in const_syms)]
        if !isempty(bad)
            error("Random effect $(r) uses non-constant covariates $(bad) in its distribution. Only ConstantCovariate is allowed in random-effect distributions.")
        end
    end
end

function get_re_covariate_usage(dm::DataModel)
    re = dm.model.random.random
    cov = dm.model.covariates.covariates
    cov_syms = Set(vcat(cov.names, cov.flat_names, cov.constants, cov.varying, cov.dynamic))
    re_syms = get_re_syms(re)
    pairs = Pair{Symbol, Vector{Symbol}}[]
    for r in get_re_names(re)
        syms = getfield(re_syms, r)
        used = sort([s for s in syms if s in cov_syms])
        push!(pairs, r => used)
    end
    return NamedTuple(pairs)
end

function _validate_re_group_within_primary(model, df, primary_id::Symbol)
    re_names = get_re_names(model.random.random)
    isempty(re_names) && return nothing
    re_groups = get_re_groups(model.random.random)
    for re in re_names
        group_col = getfield(re_groups, re)
        group_col == primary_id && continue
        bad = _check_constant_within_group(df, primary_id, [group_col])
        if !isempty(bad)
            details = join(["$(k) => $(unique(bad[k]))" for k in keys(bad)], ", ")
            error("RandomEffect $(re) is grouped by $(group_col), which varies within primary_id $(primary_id). Offending ids: $(details). Time-varying random effects are not supported yet; ensure $(group_col) is constant within each $(primary_id) group or split the data.")
        end
    end
    return nothing
end

function _validate_re_group_missing(model, df)
    re_names = get_re_names(model.random.random)
    isempty(re_names) && return nothing
    re_groups = get_re_groups(model.random.random)
    seen = Set{Symbol}()
    for re in re_names
        group_col = getfield(re_groups, re)
        group_col in seen && continue
        push!(seen, group_col)
        col = _get_col(df, group_col)
        if any(ismissing, col)
            error("Random-effect grouping column $(group_col) contains missing values. Random-effect levels must be explicit and non-missing. Either drop rows with missing $(group_col), or replace missings with an explicit custom level before constructing DataModel.")
        end
    end
    return nothing
end

function _validate_re_group_identifiability(model, df, config::DataModelConfig)
    re_names = get_re_names(model.random.random)
    isempty(re_names) && return
    re_groups = get_re_groups(model.random.random)
    obs_rows = if config.evid_col === nothing
        eachindex(_get_col(df, config.primary_id))
    else
        evid = _get_col(df, config.evid_col)
        findall(==(0), evid)
    end
    for re in re_names
        group_col = getfield(re_groups, re)
        vals = _get_col(df, group_col)[obs_rows]
        length(unique(vals)) == length(vals) || continue
        @warn "RandomEffect $(re) grouped by $(group_col) has unique values for every observation. This can be weakly identified (e.g., confounding with residual noise) and may lead to unstable estimation. Use repeated observations per level when possible, and validate with identifiability_report(...)."
    end
    return nothing
end

function _info_numeric_re_group_levels(model, df)
    re_names = get_re_names(model.random.random)
    isempty(re_names) && return nothing
    re_groups = get_re_groups(model.random.random)
    group_cols = Symbol[]
    for re in re_names
        col = getfield(re_groups, re)
        col in group_cols || push!(group_cols, col)
    end
    numeric_cols = Symbol[]
    for col in group_cols
        vals = _get_col(df, col)
        any(v -> v isa Number, vals) || continue
        push!(numeric_cols, col)
    end
    isempty(numeric_cols) && return nothing
    cols_str = join(string.(numeric_cols), ", ")
    @info "DataModel detected numeric random-effect grouping levels in column(s) $(cols_str). You wwill not be able to use constant random-effects. If you want to use constantr andom effects, consider ralabeling your random efefcts to strings or symbols."
    return nothing
end

function _group_indices(df, primary_id::Symbol)
    n = length(_get_col(df, primary_id))
    groups = Dict{Any, Vector{Int}}()
    for i in 1:n
        key = _get_col(df, primary_id)[i]
        push!(get!(groups, key, Int[]), i)
    end
    keys_sorted = collect(keys(groups))
    try
        keys_sorted = sort(keys_sorted)
    catch err
        if !(err isa MethodError)
            rethrow(err)
        end
    end
    return keys_sorted, [groups[k] for k in keys_sorted]
end

function _extract_constants(params, rows)
    cons = NamedTuple()
    if isempty(rows)
        return cons
    end
    return params
end

function _build_const_cov(covariates, df, rows)
    params = covariates.params
    out = Pair{Symbol, Any}[]
    for name in covariates.constants
        p = getfield(params, name)
        if p isa ConstantCovariate
            col = p.column
            val = _get_col(df, col)[rows[1]]
            push!(out, name => val)
        elseif p isa ConstantCovariateVector
            vals = NamedTuple{Tuple(p.columns)}(Tuple(_get_col(df, c)[rows[1]] for c in p.columns))
            push!(out, name => vals)
        end
    end
    return NamedTuple(out)
end

function _build_vary_cov(covariates, df, rows, obs_rows, time_col, include_t::Bool)
    params = covariates.params
    out = Pair{Symbol, Any}[]
    if include_t
        push!(out, :t => _get_col(df, time_col)[obs_rows])
    end
    for name in covariates.varying
        p = getfield(params, name)
        if p isa Covariate
            col = p.column
            push!(out, name => _get_col(df, col)[obs_rows])
        elseif p isa CovariateVector
            vals = NamedTuple{Tuple(p.columns)}(Tuple(_get_col(df, c)[obs_rows] for c in p.columns))
            push!(out, name => vals)
        elseif p isa DynamicCovariate || p isa DynamicCovariateVector
            # dynamic covariates are handled separately
        end
    end
    return NamedTuple(out)
end

function _build_saveat(df, rows, obs_rows, time_col, config::DataModelConfig, time_offsets::Vector{Float64})
    config.saveat_mode == :dense && return nothing
    tvals = _get_col(df, time_col)[obs_rows]
    if !isempty(time_offsets)
        expanded = Float64[]
        for t in tvals
            for off in time_offsets
                push!(expanded, t + off)
            end
        end
        tvals = expanded
    end
    if config.evid_col !== nothing
        evid = _get_col(df, config.evid_col)[rows]
        evt_idx = findall(!=(0), evid)
        if !isempty(evt_idx)
            tvals = vcat(tvals, _get_col(df, time_col)[rows][evt_idx])
        end
    end
    return sort(unique(tvals))
end

function _build_dyn_cov(covariates, df, rows, time_col)
    params = covariates.params
    t = _get_col(df, time_col)[rows]
    out = Pair{Symbol, Any}[]
    for name in covariates.dynamic
        p = getfield(params, name)
        if p isa DynamicCovariate
            col = p.column
            y = _get_col(df, col)[rows]
            interp = p.interpolation(y, t)
            push!(out, name => interp)
        elseif p isa DynamicCovariateVector
            vals = NamedTuple{Tuple(p.columns)}(Tuple(p.interpolations[i](_get_col(df, p.columns[i])[rows], t) for i in eachindex(p.columns)))
            push!(out, name => vals)
        end
    end
    return NamedTuple(out)
end

function _validate_dynamic_covariates(covariates, rows, t, id_val)
    isempty(covariates.dynamic) && return nothing
    issorted(t) || error("Dynamic covariates require time to be sorted for each individual. Individual $(id_val) has unsorted time values.")
    n = length(rows)
    min_obs = Dict(
        ConstantInterpolation => 1,
        SmoothedConstantInterpolation => 2,
        LinearInterpolation => 2,
        QuadraticInterpolation => 3,
        LagrangeInterpolation => 2,
        QuadraticSpline => 3,
        CubicSpline => 3,
        AkimaInterpolation => 2,
    )
    alt_by_req = Dict{Int, Vector{Symbol}}()
    for (itp, req) in min_obs
        push!(get!(alt_by_req, req, Symbol[]), Symbol(itp))
    end
    function _alternatives(req, n)
        opts = Symbol[]
        for (r, names) in alt_by_req
            r < req && r <= n && append!(opts, names)
        end
        return sort!(unique!(opts))
    end
    params = covariates.params
    offenders = String[]
    for name in covariates.dynamic
        p = getfield(params, name)
        if p isa DynamicCovariate
            req = get(min_obs, p.interpolation, 1)
            if n < req
                itp_name = Symbol(p.interpolation)
                alts = _alternatives(req, n)
                alt_str = isempty(alts) ? "none" : join(alts, ", ")
                push!(offenders,
                      "Dynamic covariate $(name) uses $(itp_name) as interpolation method, this requires at least $(req) observations per individual. This is violated by individual $(id_val). It has only $(n) observations. Alternatives requiring at most $(n) observations are: $(alt_str).")
            end
        elseif p isa DynamicCovariateVector
            for (i, itp) in enumerate(p.interpolations)
                req = get(min_obs, itp, 1)
                if n < req
                    itp_name = Symbol(itp)
                    alts = _alternatives(req, n)
                    alt_str = isempty(alts) ? "none" : join(alts, ", ")
                    push!(offenders,
                          "Dynamic covariate $(name).$(p.columns[i]) uses $(itp_name) as interpolation method, this requires at least $(req) observations per individual. This is violated by individual $(id_val). It has only $(n) observations. Alternatives requiring at most $(n) observations are: $(alt_str).")
                end
            end
        end
    end
    if !isempty(offenders)
        error(join(offenders, " "))
    end
    return nothing
end

function _build_obs(df, rows, obs_rows, obs_cols)
    pairs = Pair{Symbol, Any}[]
    for col in obs_cols
        push!(pairs, col => _get_col(df, col)[obs_rows])
    end
    return NamedTuple(pairs)
end

function _build_callbacks(model, df, rows, config::DataModelConfig)
    config.evid_col === nothing && return nothing
    model.de.de === nothing && return nothing
    evid = _get_col(df, config.evid_col)[rows]
    evt_idx = findall(!=(0), evid)
    isempty(evt_idx) && return nothing

    tvals = _get_col(df, config.time_col)[rows][evt_idx]
    amt = _get_col(df, config.amt_col)[rows][evt_idx]
    rate = _get_col(df, config.rate_col)[rows][evt_idx]
    cmt = _get_col(df, config.cmt_col)[rows][evt_idx]
    evid = evid[evt_idx]

    state_names = get_de_states(model.de.de)
    n_states = length(state_names)
    state_map = Dict{Symbol, Int}((state_names[i] => i) for i in eachindex(state_names))
    Tt = eltype(tvals)
    bolus_by_time = Dict{Tt, Vector{Float64}}()
    reset_by_time = Dict{Tt, Vector{Tuple{Int, Float64}}}()
    rate_delta_by_time = Dict{Tt, Vector{Float64}}()
    t0 = minimum(_get_col(df, config.time_col)[rows])
    init_bolus = zeros(Float64, n_states)
    init_resets = Tuple{Int, Float64}[]
    init_rate_starts = zeros(Float64, n_states)

    function _get_or_init!(dict, tkey, n)
        return get!(dict, tkey) do
            zeros(Float64, n)
        end
    end

    cmt_types = Set{DataType}()
    for v in cmt
        v === missing && continue
        push!(cmt_types, typeof(v))
    end
    if any(t -> t <: Integer, cmt_types) && any(t -> (t <: AbstractString || t <: Symbol), cmt_types)
        error("CMT values must use a single style per DataFrame (all indices or all names). Mixed types were found.")
    end

    function _levenshtein(a::AbstractString, b::AbstractString)
        la = lastindex(a)
        lb = lastindex(b)
        da = collect(a)
        db = collect(b)
        m = length(da)
        n = length(db)
        dp = Vector{Int}(undef, n + 1)
        for j in 0:n
            dp[j + 1] = j
        end
        for i in 1:m
            prev = dp[1]
            dp[1] = i
            for j in 1:n
                temp = dp[j + 1]
                cost = da[i] == db[j] ? 0 : 1
                dp[j + 1] = min(dp[j + 1] + 1, dp[j] + 1, prev + cost)
                prev = temp
            end
        end
        return dp[n + 1]
    end

    function _suggest_states(name::Symbol)
        target = String(name)
        scores = [(s, _levenshtein(target, String(s))) for s in state_names]
        sort!(scores, by = x -> x[2])
        top = first(scores, min(3, length(scores)))
        return join([String(s[1]) for s in top], ", ")
    end

    function _resolve_cmt(v)
        if v isa Integer
            return Int(v)
        elseif v isa Symbol
            haskey(state_map, v) || error("CMT $(v) does not match any state. Closest matches: $(_suggest_states(v)).")
            return state_map[v]
        elseif v isa AbstractString
            sym = Symbol(v)
            haskey(state_map, sym) || error("CMT $(v) does not match any state. Closest matches: $(_suggest_states(sym)).")
            return state_map[sym]
        else
            error("CMT values must be Int, Symbol, or String; got $(typeof(v)).")
        end
    end

    for i in eachindex(tvals)
        t = tvals[i]
        ev = evid[i]
        amt_i = Float64(amt[i])
        rate_i = Float64(rate[i])
        cmt_i = _resolve_cmt(cmt[i])
        (1 <= cmt_i <= n_states) || error("CMT $(cmt_i) is out of bounds for $(n_states) ODE states.")

        if ev == 1
            if t == t0
                if rate_i == 0.0
                    init_bolus[cmt_i] += amt_i
                else
                    init_rate_starts[cmt_i] += rate_i
                    duration = abs(amt_i / rate_i)
                    stop_t = t + duration
                    stop_delta = _get_or_init!(rate_delta_by_time, stop_t, n_states)
                    stop_delta[cmt_i] -= rate_i
                end
            else
                if rate_i == 0.0
                    delta = _get_or_init!(bolus_by_time, t, n_states)
                    delta[cmt_i] += amt_i
                else
                    duration = abs(amt_i / rate_i)
                    start_delta = _get_or_init!(rate_delta_by_time, t, n_states)
                    start_delta[cmt_i] += rate_i
                    stop_t = t + duration
                    stop_delta = _get_or_init!(rate_delta_by_time, stop_t, n_states)
                    stop_delta[cmt_i] -= rate_i
                end
            end
        elseif ev == 2
            if t == t0
                push!(init_resets, (cmt_i, amt_i))
            else
                reset_list = get!(reset_by_time, t) do
                    Tuple{Int, Float64}[]
                end
                push!(reset_list, (cmt_i, amt_i))
            end
        else
            error("Unsupported EVID $(ev). Supported values are 1 (dose/infusion) and 2 (reset).")
        end
    end

    infusion_rates = zeros(Float64, n_states)
    infusion_rates .+= init_rate_starts

    affect! = function (integrator)
        t = integrator.t
        if haskey(reset_by_time, t)
            for (idx, val) in reset_by_time[t]
                integrator.u[idx] = val
            end
        end
        if haskey(bolus_by_time, t)
            delta = bolus_by_time[t]
            for j in eachindex(delta)
                dj = delta[j]
                dj == 0.0 && continue
                integrator.u[j] += dj
            end
        end
        if haskey(rate_delta_by_time, t)
            delta = rate_delta_by_time[t]
            for j in eachindex(delta)
                dj = delta[j]
                dj == 0.0 && continue
                infusion_rates[j] += dj
            end
        end
        return nothing
    end

    times = unique!(vcat(collect(keys(bolus_by_time)),
                         collect(keys(reset_by_time)),
                         collect(keys(rate_delta_by_time))))
    sort!(times)
    cb = isempty(times) ? nothing : PresetTimeCallback(times, affect!)

    init_bolus_pairs = Tuple{Int, Float64}[]
    for i in eachindex(init_bolus)
        init_bolus[i] == 0.0 && continue
        push!(init_bolus_pairs, (i, init_bolus[i]))
    end
    resets_out = isempty(init_resets) ? nothing : init_resets
    bolus_out = isempty(init_bolus_pairs) ? nothing : init_bolus_pairs
    (cb === nothing && resets_out === nothing && bolus_out === nothing) && return nothing
    return EventCallbacks(cb, infusion_rates, resets_out, bolus_out)
end

@inline function _apply_initial_events!(u0, callbacks::EventCallbacks)
    callbacks.init_resets !== nothing && @inbounds for (idx, val) in callbacks.init_resets
        u0[idx] = val
    end
    callbacks.init_bolus !== nothing && @inbounds for (idx, val) in callbacks.init_bolus
        u0[idx] += val
    end
    return u0
end

function _build_re_groups(model, df, rows)
    groups = get_re_groups(model.random.random)
    re_names = get_re_names(model.random.random)
    pairs = Pair{Symbol, Any}[]
    for name in re_names
        col = getfield(groups, name)
        vals = unique(_get_col(df, col)[rows])
        push!(pairs, name => vals)
    end
    return NamedTuple(pairs)
end

function _build_re_group_info(model, df, individuals)
    re_names = get_re_names(model.random.random)
    isempty(re_names) && return REGroupInfo(NamedTuple(), NamedTuple(), nothing)
    re_groups = get_re_groups(model.random.random)
    values_pairs = Pair{Symbol, Any}[]
    index_pairs = Pair{Symbol, Any}[]
    for re in re_names
        col = getfield(re_groups, re)
        gcol = _get_col(df, col)
        values = Vector{eltype(gcol)}()
        index_by_row = Vector{Int}(undef, length(gcol))
        idx_map = Dict{eltype(gcol), Int}()
        for i in eachindex(gcol)
            g = gcol[i]
            idx = get(idx_map, g, 0)
            if idx == 0
                push!(values, g)
                idx = length(values)
                idx_map[g] = idx
            end
            index_by_row[i] = idx
        end
        push!(values_pairs, re => values)
        push!(index_pairs, re => index_by_row)
    end
    values_nt = NamedTuple(values_pairs)
    index_nt = NamedTuple(index_pairs)
    cache = _build_laplace_re_cache(model, individuals, values_nt)
    return REGroupInfo(values_nt, index_nt, cache)
end

function _build_pairing(individuals, model)
    n = length(individuals)
    n == 0 && return PairingInfo(Int[], Vector{Vector{Int}}())
    uf = UnionFind(n)
    re_names = get_re_names(model.random.random)
    for re in re_names
        seen = Dict{Any, Int}()
        for i in 1:n
            g = getfield(individuals[i].re_groups, re)
            if g isa AbstractVector
                for gv in g
                    if haskey(seen, gv)
                        _uf_union!(uf, i, seen[gv])
                    else
                        seen[gv] = i
                    end
                end
            else
                if haskey(seen, g)
                    _uf_union!(uf, i, seen[g])
                else
                    seen[g] = i
                end
            end
        end
    end
    batch_ids = [_uf_find(uf, i) for i in 1:n]
    batches_dict = Dict{Int, Vector{Int}}()
    for i in 1:n
        push!(get!(batches_dict, batch_ids[i], Int[]), i)
    end
    return PairingInfo(batch_ids, collect(values(batches_dict)))
end

function _build_laplace_re_cache(model, individuals, values_nt)
    re_names = get_re_names(model.random.random)
    isempty(re_names) && return nothing
    nre = length(re_names)
    re_index = Vector{LaplaceREIndex}(undef, nre)
    for (ri, re) in enumerate(re_names)
        levels = getfield(values_nt, re)
        idx_map = Dict{eltype(levels), Int}()
        for (i, v) in enumerate(levels)
            idx_map[v] = i
        end
        re_index[ri] = LaplaceREIndex(levels, idx_map)
    end
    ind_level_ids = Vector{Vector{Vector{Int}}}(undef, length(individuals))
    for i in eachindex(individuals)
        ind = individuals[i]
        ids_by_re = Vector{Vector{Int}}(undef, nre)
        for (ri, re) in enumerate(re_names)
            idx_map = re_index[ri].level_to_index
            g = getfield(ind.re_groups, re)
            if g isa AbstractVector
                ids = Vector{Int}(undef, length(g))
                @inbounds for k in eachindex(g)
                    ids[k] = idx_map[g[k]]
                end
            else
                ids = Int[idx_map[g]]
            end
            ids_by_re[ri] = ids
        end
        ind_level_ids[i] = ids_by_re
    end
    dims = Vector{Int}(undef, nre)
    is_scalar = BitVector(undef, nre)
    if !isempty(individuals)
        fe = model.fixed.fixed
        θ0 = get_θ0_untransformed(fe)
        model_funs = get_model_funs(model)
        helpers = get_helper_funs(model)
        dists_builder = get_create_random_effect_distribution(model.random.random)
        const_cov = individuals[1].const_cov
        dists = dists_builder(θ0, const_cov, model_funs, helpers)
        for (ri, re) in enumerate(re_names)
            dist = getproperty(dists, re)
            is_scalar[ri] = dist isa Distributions.UnivariateDistribution
            dims[ri] = is_scalar[ri] ? 1 : length(dist)
            dims[ri] == 0 && error("Random effect $(re) has zero dimension.")
        end
    else
        fill!(dims, 0)
        fill!(is_scalar, true)
    end
    return LaplaceRECache(re_names, dims, is_scalar, ind_level_ids, re_index)
end

"""
    DataModel(model, df; primary_id, time_col, evid_col, amt_col, rate_col, cmt_col,
              serialization) -> DataModel

Construct a [`DataModel`](@ref) by pairing a [`Model`](@ref) with a `DataFrame`.

Performs comprehensive validation: schema checks, time-column covariate declaration,
constant-covariate constancy, random-effect group uniqueness, dynamic-covariate
observation counts, and saveat-mode resolution.

# Arguments
- `model::Model`: the compiled model (from `@Model`).
- `df`: a `DataFrame` containing all required columns.

# Keyword Arguments
- `primary_id::Union{Nothing, Symbol} = nothing`: primary individual-grouping column.
  Defaults to the single random-effect grouping column when only one exists.
- `time_col::Symbol = :TIME`: name of the time column.
- `evid_col::Union{Nothing, Symbol} = nothing`: EVID column name. When set, enables
  PKPD event handling (boluses, infusions, resets). Rows with EVID=0 are observations;
  EVID=1 are dose events; EVID=2 are reset events.
- `amt_col::Symbol = :AMT`: AMT column (dose amounts, used when `evid_col` is set).
- `rate_col::Symbol = :RATE`: RATE column (infusion rates, used when `evid_col` is set).
- `cmt_col::Symbol = :CMT`: CMT column (compartment index or name, used when `evid_col` is set).
- `serialization::SciMLBase.EnsembleAlgorithm = EnsembleSerial()`: parallelisation
  strategy for ODE solving. Use `EnsembleThreads()` for multi-threaded evaluation.
"""
function DataModel(model,
                   df;
                   primary_id::Union{Nothing, Symbol}=nothing,
                   time_col::Symbol=:TIME,
                   evid_col::Union{Nothing, Symbol}=nothing,
                   amt_col::Symbol=:AMT,
                   rate_col::Symbol=:RATE,
                   cmt_col::Symbol=:CMT,
                   serialization::SciMLBase.EnsembleAlgorithm=EnsembleSerial())

    if primary_id === nothing
        re_groups = get_re_groups(model.random.random)
        re_cols = unique([getfield(re_groups, n) for n in get_re_names(model.random.random)])
        if length(re_cols) == 1
            primary_id = re_cols[1]
        else
            error("DataModel needs a primary_id when multiple random-effect grouping columns exist ($(re_cols)). Choose one of these columns as primary_id=... to select the individual grouping.")
        end
    end

    obs_cols = collect(get_formulas_meta(model.formulas.formulas).obs_names)
    solver_cfg = get_solver_config(model)
    saveat_mode = solver_cfg.saveat_mode
    saveat_mode in (:dense, :saveat, :auto) || error("Unknown saveat_mode $(saveat_mode). Use :dense, :saveat, or :auto.")
    saveat_mode = saveat_mode == :auto ? :saveat : saveat_mode
    config = DataModelConfig(primary_id, time_col, evid_col, amt_col, rate_col, cmt_col, obs_cols, serialization, saveat_mode)
    _validate_schema(model, df, config)

    cov = model.covariates.covariates
    _validate_time_col_covariate(model, config)
    _validate_re_group_missing(model, df)
    _info_numeric_re_group_levels(model, df)
    _validate_re_group_constants(model, df, cov)
    _validate_constant_covariates_primary(model, df, primary_id, cov)
    _validate_re_dist_covariates(model, cov)
    _validate_re_group_within_primary(model, df, primary_id)
    _validate_re_group_identifiability(model, df, config)
    _validate_prede_random_effects(model, primary_id)
    include_t = model.de.de !== nothing
    state_names = model.de.de === nothing ? Symbol[] : get_de_states(model.de.de)
    signal_names = model.de.de === nothing ? Symbol[] : get_de_signals(model.de.de)
    (time_offsets, requires_dense) = get_formulas_time_offsets(model.formulas.formulas, state_names, signal_names)
    if saveat_mode != :dense && requires_dense
        error("Formulas include non-constant time offsets for DE states/signals. Use saveat_mode=:dense or rewrite formulas to use constant offsets.")
    end
    keys_sorted, groups = _group_indices(df, primary_id)
    individuals = Vector{Individual}(undef, length(groups))
    obs_groups = Vector{Vector{Int}}(undef, length(groups))

    bad_ids = Any[]
    bad_tmin = Dict{Any, Float64}()
    for (i, rows) in enumerate(groups)
        tvals = _get_col(df, time_col)[rows]
        _validate_dynamic_covariates(cov, rows, tvals, keys_sorted[i])
        if evid_col === nothing
            obs_rows = rows
        else
            evid = _get_col(df, evid_col)[rows]
            obs_rows = rows[findall(==(0), evid)]
        end
        obs_groups[i] = obs_rows
        obs = _build_obs(df, rows, obs_rows, obs_cols)
        vary = _build_vary_cov(cov, df, rows, obs_rows, time_col, include_t)
        dyn = _build_dyn_cov(cov, df, rows, time_col)
        series = IndividualSeries(obs, vary, dyn)
        const_cov = _build_const_cov(cov, df, rows)
        callbacks = _build_callbacks(model, df, rows, config)
        if isempty(time_offsets)
            tspan = (minimum(tvals), maximum(tvals))
        else
            extra_min = minimum(time_offsets)
            extra_max = maximum(time_offsets)
            tmin = minimum(tvals) + extra_min
            tmax = maximum(tvals) + extra_max
            if tmin < 0
                id_val = keys_sorted[i]
                push!(bad_ids, id_val)
                bad_tmin[id_val] = tmin
            end
            tspan = (tmin, tmax)
        end
        re_groups = _build_re_groups(model, df, rows)
        saveat = _build_saveat(df, rows, obs_rows, time_col, config, time_offsets)
        individuals[i] = Individual(series, const_cov, callbacks, tspan, re_groups, saveat)
    end

    if !isempty(bad_ids)
        details = join(["$(id) => $(bad_tmin[id])" for id in bad_ids], ", ")
        error("Formulas request times earlier than the first observation for individual ids: $(details). This would require to calculate the value of the ODE solution prior to the initial conditions (at t=0). Change the offset in your model or remove the corresponding data points that are observed at t_k for which t_k - offset < 0.")
    end

    pairing = _build_pairing(individuals, model)
    id_index = Dict{Any, Int}((keys_sorted[i] => i) for i in eachindex(keys_sorted))
    row_groups = RowGroups(groups, obs_groups)
    re_group_info = _build_re_group_info(model, df, individuals)
    return DataModel(model, df, individuals, pairing, config, id_index, row_groups, re_group_info)
end

"""
    get_model(dm::DataModel) -> Model

Return the [`Model`](@ref) stored in the `DataModel`.
"""
get_model(dm::DataModel) = dm.model

"""
    get_df(dm::DataModel) -> DataFrame

Return the original `DataFrame` used to construct the `DataModel`.
"""
get_df(dm::DataModel) = dm.df

"""
    get_individuals(dm::DataModel) -> Vector{Individual}

Return the vector of `Individual` structs (one per unique primary-id value).
"""
get_individuals(dm::DataModel) = dm.individuals

"""
    get_batches(dm::DataModel) -> Vector{Vector{Int}}

Return the list of batches, where each batch is a vector of individual indices.
Individuals in the same batch share at least one random-effect level and must be
estimated jointly.
"""
get_batches(dm::DataModel) = dm.pairing.batches

"""
    get_batch_ids(dm::DataModel) -> Vector{Int}

Return the batch index for each individual (length equals number of individuals).
"""
get_batch_ids(dm::DataModel) = dm.pairing.batch_ids

"""
    get_primary_id(dm::DataModel) -> Symbol

Return the primary individual-grouping column name.
"""
get_primary_id(dm::DataModel) = dm.config.primary_id

"""
    get_row_groups(dm::DataModel) -> RowGroups

Return the `RowGroups` struct mapping each individual index to its data-frame row
indices (all rows and observation-only rows).
"""
get_row_groups(dm::DataModel) = dm.row_groups

"""
    get_re_group_info(dm::DataModel) -> REGroupInfo

Return the `REGroupInfo` struct containing random-effect level values and per-row
level indices.
"""
get_re_group_info(dm::DataModel) = dm.re_group_info

function _find_individual_index(dm::DataModel, ind::Individual)
    for i in eachindex(dm.individuals)
        dm.individuals[i] === ind && return i
    end
    return nothing
end

"""
    get_re_indices(dm::DataModel, id_or_ind_or_idx; obs_only=true) -> NamedTuple

Return a `NamedTuple` mapping each random-effect name to a vector of level indices
for the specified individual.

The individual can be identified by its primary-id value, an `Individual` object,
or its integer position in `get_individuals(dm)`.

# Keyword Arguments
- `obs_only::Bool = true`: if `true`, only return indices for observation rows;
  if `false`, include all rows (including event rows).
"""
function get_re_indices(dm::DataModel, ind::Individual; obs_only::Bool=true)
    idx = _find_individual_index(dm, ind)
    idx === nothing && error("Individual not found in DataModel.")
    return get_re_indices(dm, idx; obs_only=obs_only)
end

function get_re_indices(dm::DataModel, id; obs_only::Bool=true)
    idx = get(dm.id_index, id, nothing)
    idx === nothing && error("No individual found for id $(id). Check primary_id or data values.")
    return get_re_indices(dm, idx; obs_only=obs_only)
end

function get_re_indices(dm::DataModel, idx::Int; obs_only::Bool=true)
    rows = dm.row_groups.rows[idx]
    obs_rows = dm.row_groups.obs_rows[idx]
    use_rows = obs_only ? obs_rows : rows
    re_names = get_re_names(dm.model.random.random)
    info = dm.re_group_info.index_by_row
    pairs = Pair{Symbol, Any}[]
    for re in re_names
        idxs = getfield(info, re)[use_rows]
        push!(pairs, re => idxs)
    end
    return NamedTuple(pairs)
end

"""
    get_individual(dm::DataModel, id) -> Individual

Return the `Individual` struct for the given primary-id value.
Raises an error if the id is not found.
"""
function get_individual(dm::DataModel, id)
    idx = get(dm.id_index, id, nothing)
    idx === nothing && error("No individual found for id $(id). Check primary_id or data values.")
    return dm.individuals[idx]
end
