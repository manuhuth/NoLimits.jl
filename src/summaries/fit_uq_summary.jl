export FitResultSummary
export UQResultSummary
export summarize

using Statistics
using MCMCChains
using Random

"""
    FitResultSummary

Structured summary of a [`FitResult`](@ref). Created by `summarize(res)` or
`summarize(res, uq)`. Contains per-parameter rows with estimates and optional
standard errors, outcome coverage statistics, and random-effects summaries.
Displayed via `Base.show`.
"""
struct FitResultSummary
    method::Symbol
    inference::Symbol
    scale::Symbol
    objective::Any
    converged::Any
    iterations::Any
    loglikelihood::Union{Nothing, Float64}
    include_non_se::Bool
    n_parameters_total::Int
    n_parameters_reported::Int
    n_parameters_uq_eligible::Int
    parameter_rows::Vector{NamedTuple}
    coverage_rows::Vector{NamedTuple}
    n_obs_total::Union{Nothing, Int}
    n_missing_total::Union{Nothing, Int}
    random_effect_label::String
    random_effect_rows::Vector{NamedTuple}
    notes::Vector{String}
end

"""
    UQResultSummary

Structured summary of a [`UQResult`](@ref). Created by `summarize(uq)`. Contains
per-parameter rows with point estimates, standard errors, and confidence/credible
intervals. Displayed via `Base.show`.
"""
struct UQResultSummary
    backend::Symbol
    source_method::Symbol
    inference::Symbol
    scale::Symbol
    objective::Any
    level::Union{Nothing, Float64}
    interval_label::String
    include_non_se::Bool
    n_parameters_total::Int
    n_parameters_reported::Int
    n_parameters_uq_eligible::Int
    parameter_rows::Vector{NamedTuple}
    coverage_rows::Vector{NamedTuple}
    n_obs_total::Union{Nothing, Int}
    n_missing_total::Union{Nothing, Int}
    random_effect_label::String
    random_effect_rows::Vector{NamedTuple}
    notes::Vector{String}
end

@inline _fq_fmt_missing(x) = x === nothing ? "-" : x

function _fq_fmt_num(x)
    x === nothing && return "-"
    x isa Missing && return "-"
    x isa Real || return string(x)
    xv = Float64(x)
    if !isfinite(xv)
        return string(xv)
    end
    ax = abs(xv)
    if ax >= 1e4 || (ax > 0 && ax < 1e-3)
        return string(round(xv; sigdigits=4))
    end
    return string(round(xv; digits=4))
end

function _fq_fmt_objective(x)
    x === nothing && return "-"
    x isa Missing && return "-"
    if x isa Real
        xv = Float64(x)
        isfinite(xv) || return "-"
    end
    return _fq_fmt_num(x)
end

function _fq_print_key_values(io::IO, title::String, rows::AbstractVector{<:Pair})
    println(io, title)
    isempty(rows) && (println(io, "  (none)"); return)
    keys_str = [string(first(r)) for r in rows]
    w = maximum(length, keys_str)
    for (i, r) in enumerate(rows)
        v = last(r)
        println(io, "  ", rpad(keys_str[i], w), " : ", v)
    end
end

function _fq_scale_symbol(scale::Symbol)
    scale in (:natural, :transformed) || error("scale must be :natural or :transformed.")
    return scale
end

function _fq_method_symbol(res::FitResult)
    return _method_symbol(get_method(res))
end

@inline _fq_inference_from_method(method::FittingMethod) = (method isa MCMC || method isa VI) ? :bayesian : :frequentist
@inline _fq_inference_from_uq(uq::UQResult) = (uq.backend == :chain || uq.backend == :mcmc_refit) ? :bayesian : :frequentist

function _fq_try_loglikelihood(res::FitResult)
    try
        ll = get_loglikelihood(res)
        return isfinite(ll) ? Float64(ll) : nothing
    catch
        return nothing
    end
end

function _fq_try_iterations(res::FitResult)
    try
        return get_iterations(res)
    catch
        return missing
    end
end

function _fq_try_outcome_coverage(res::FitResult)
    dm = get_data_model(res)
    dm === nothing && return (NamedTuple[], nothing, nothing, ["Data coverage unavailable: FitResult does not store DataModel."])
    rows = NamedTuple[]
    n_obs_total = 0
    n_missing_total = 0
    for col in dm.config.obs_cols
        v = getproperty(dm.df, col)
        miss = count(ismissing, v)
        obs = length(v) - miss
        n_obs_total += obs
        n_missing_total += miss
        push!(rows, (; outcome=col, n_obs=obs, n_missing=miss))
    end
    return (rows, n_obs_total, n_missing_total, String[])
end

function _fq_role_by_parent(model::Model)
    fe = model.fixed.fixed
    fixed_names = get_names(fe)
    fixed_set = Set(fixed_names)

    re_parents = Set{Symbol}()
    re_model = model.random.random
    re_names = get_re_names(re_model)
    re_syms = get_re_syms(re_model)
    for re in re_names
        for s in getfield(re_syms, re)
            s in fixed_set && push!(re_parents, s)
        end
    end

    ir = get_formulas_ir(model.formulas.formulas)
    formula_syms = Set{Symbol}(vcat(ir.var_syms, ir.prop_syms))
    formula_parents = Set([s for s in fixed_names if s in formula_syms])

    roles = Dict{Symbol, Symbol}()
    for n in fixed_names
        in_re = n in re_parents
        in_formula = n in formula_parents
        roles[n] = (in_re && in_formula) ? :Both :
                   in_re ? :RE_distribution : :General_Outcome
    end
    return roles
end

@inline function _fq_role_label(role::Symbol)
    role == :Both && return "Both"
    role == :RE_distribution && return "RE distribution"
    return "General/Outcome"
end

function _fq_mcmc_warmup(res::FitResult)
    diag = get_diagnostics(res)
    conv = hasproperty(diag, :convergence) ? getproperty(diag, :convergence) : NamedTuple()
    return hasproperty(conv, :n_adapt) ? Int(getproperty(conv, :n_adapt)) : 0
end

function _fq_chain_array(chain)
    arr = Array(chain)
    if ndims(arr) == 2
        arr = reshape(arr, size(arr, 1), size(arr, 2), 1)
    end
    ndims(arr) == 3 || error("Unexpected chain layout with ndims=$(ndims(arr)).")
    return arr
end

function _fq_chain_idx_map(chain)
    names = MCMCChains.names(chain, :parameters)
    idx_map = Dict{String, Int}()
    for (i, n) in enumerate(names)
        idx_map[string(n)] = i
    end
    return idx_map
end

function _fq_chain_values(arr, var_idx::Int, warmup::Int)
    n_iter = size(arr, 1)
    first_keep = clamp(warmup + 1, 1, n_iter)
    vals = vec(arr[first_keep:end, var_idx, :])
    isempty(vals) && error("No post-warmup MCMC draws available.")
    return Float64.(vals)
end

function _fq_spec_map(fe::FixedEffects)
    names = get_names(fe)
    specs = get_transforms(fe).forward.specs
    out = Dict{Symbol, TransformSpec}()
    for i in eachindex(names)
        out[names[i]] = specs[i]
    end
    return out
end

function _fq_value_from_lookup(v0, name::Symbol, spec::TransformSpec, lookup::Function)
    if v0 isa Number
        val = lookup(string(name))
        return val === nothing ? v0 : val
    end
    if spec.kind == :expm && v0 isa AbstractMatrix
        n = size(v0, 1)
        vv = Matrix{Float64}(undef, size(v0)...)
        for j in 1:n
            for i in 1:j
                key = string(name, "[", i, ",", j, "]")
                val = lookup(key)
                x = val === nothing ? Float64(v0[i, j]) : Float64(val)
                vv[i, j] = x
                vv[j, i] = x
            end
        end
        return vv
    end
    vv = similar(v0, Float64)
    for ci in CartesianIndices(v0)
        key = string(name, "[", join(Tuple(ci), ","), "]")
        val = lookup(key)
        vv[ci] = val === nothing ? Float64(v0[ci]) : Float64(val)
    end
    return vv
end

function _fq_fixed_point_estimate_from_lookup(fe::FixedEffects,
                                              constants::NamedTuple,
                                              lookup::Function)
    θu = deepcopy(get_θ0_untransformed(fe))
    spec_map = _fq_spec_map(fe)
    for name in get_names(fe)
        if haskey(constants, name)
            setproperty!(θu, name, getfield(constants, name))
            continue
        end
        v0 = getproperty(θu, name)
        spec = spec_map[name]
        setproperty!(θu, name, _fq_value_from_lookup(v0, name, spec, lookup))
    end
    return _as_component_array(θu)
end

function _fq_mcmc_fixed_point_estimate(res::FitResult, dm::DataModel)
    fe = dm.model.fixed.fixed
    constants = _fit_kw(res, :constants, NamedTuple())

    chain = get_chain(res)
    arr = _fq_chain_array(chain)
    warmup = _fq_mcmc_warmup(res)
    idx_map = _fq_chain_idx_map(chain)
    med_cache = Dict{Int, Float64}()
    lookup = key -> begin
        idx = _lookup_chain_index(idx_map, key)
        idx == 0 && return nothing
        if !haskey(med_cache, idx)
            med_cache[idx] = Float64(median(_fq_chain_values(arr, idx, warmup)))
        end
        return med_cache[idx]
    end
    return _fq_fixed_point_estimate_from_lookup(fe, constants, lookup)
end

function _fq_vi_fixed_point_estimate(res::FitResult, dm::DataModel; n_draws::Int=1000)
    fe = dm.model.fixed.fixed
    constants = _fit_kw(res, :constants, NamedTuple())
    n_draws >= 1 || error("n_draws must be >= 1.")
    draw_pack = sample_posterior(res; n_draws=n_draws, rng=Random.Xoshiro(0x4f13), return_names=true)
    draws = draw_pack.draws
    names = draw_pack.names
    idx_map = Dict{String, Int}()
    for (i, n) in enumerate(names)
        idx_map[string(n)] = i
    end
    med_cache = Dict{Int, Float64}()
    lookup = key -> begin
        idx = _lookup_chain_index(idx_map, key)
        idx == 0 && return nothing
        if !haskey(med_cache, idx)
            med_cache[idx] = Float64(median(@view draws[:, idx]))
        end
        return med_cache[idx]
    end
    return _fq_fixed_point_estimate_from_lookup(fe, constants, lookup)
end

function _fq_fit_component_estimates(res::FitResult, dm::DataModel, scale::Symbol)
    fe = dm.model.fixed.fixed
    method = get_method(res)
    if method isa MCMC
        θu = _fq_mcmc_fixed_point_estimate(res, dm)
        if scale == :natural
            return _coords_on_transformed_layout(fe, θu, get_names(fe); natural=true)
        else
            θt = get_transform(fe)(θu)
            return _coords_on_transformed_layout(fe, θt, get_names(fe); natural=false)
        end
    elseif method isa VI
        θu = _fq_vi_fixed_point_estimate(res, dm)
        if scale == :natural
            return _coords_on_transformed_layout(fe, θu, get_names(fe); natural=true)
        else
            θt = get_transform(fe)(θu)
            return _coords_on_transformed_layout(fe, θt, get_names(fe); natural=false)
        end
    else
        θ = scale == :natural ? get_params(res; scale=:untransformed) : get_params(res; scale=:transformed)
        return _coords_on_transformed_layout(fe, θ, get_names(fe); natural=(scale == :natural))
    end
end

function _fq_fit_parameter_rows(res::FitResult;
                                scale::Symbol=:natural,
                                include_non_se::Bool=false)
    dm = get_data_model(res)
    dm === nothing && return (NamedTuple[], 0, 0, ["Parameter table unavailable: FitResult does not store DataModel."])
    model = dm.model
    fe = model.fixed.fixed

    flat_names = get_flat_names(fe)
    parent_names = _flat_parent_names(fe)
    se_mask = get_se_mask(fe)
    roles_by_parent = _fq_role_by_parent(model)
    estimates = _fq_fit_component_estimates(res, dm, scale)
    length(estimates) == length(flat_names) || error("Internal summary error: estimate layout mismatch.")

    rows = NamedTuple[]
    for i in eachindex(flat_names)
        if !include_non_se && !se_mask[i]
            continue
        end
        p = parent_names[i]
        role = get(roles_by_parent, p, :General_Outcome)
        push!(rows, (;
            parameter=flat_names[i],
            role=_fq_role_label(role),
            estimate=estimates[i],
            calculate_se=se_mask[i],
        ))
    end
    return (rows, length(flat_names), count(identity, se_mask), String[])
end

function _fq_re_stats_rows_from_df_nt(re_nt::NamedTuple)
    rows = NamedTuple[]
    for re in keys(re_nt)
        df = getfield(re_nt, re)
        cols = names(df)
        length(cols) <= 1 && continue
        n_comp = length(cols) - 1
        for c in cols[2:end]
            vals = Float64[]
            for v in df[!, c]
                if v === missing
                    continue
                elseif v isa Real
                    push!(vals, Float64(v))
                end
            end
            st = _descriptive_stats(vals)
            push!(rows, (;
                random_effect=string(re),
                component=(n_comp == 1 ? "-" : string(c)),
                n=st.n,
                mean=st.mean,
                sd=st.sd,
                min=st.min,
                q25=st.q25,
                median=st.median,
                q75=st.q75,
                max=st.max,
            ))
        end
    end
    return rows
end

function _fq_parse_re_chain_name(s::String, re_set::Set{Symbol})
    for rx in (
        r"^(.+)_vals\[(\d+),\s*(\d+)\]$",
        r"^(.+)_vals\[(\d+)\]\[(\d+)\]$",
        r"^(.+)\[(\d+),\s*(\d+)\]$",
        r"^(.+)\[(\d+)\]\[(\d+)\]$",
    )
        m = match(rx, s)
        m === nothing && continue
        re = Symbol(m.captures[1])
        re in re_set || return nothing
        lvl = tryparse(Int, m.captures[2])
        dim = tryparse(Int, m.captures[3])
        (lvl === nothing || dim === nothing) && return nothing
        return (re, lvl, dim)
    end
    for rx in (
        r"^(.+)_vals\[(\d+)\]$",
        r"^(.+)\[(\d+)\]$",
    )
        m = match(rx, s)
        m === nothing && continue
        re = Symbol(m.captures[1])
        re in re_set || return nothing
        lvl = tryparse(Int, m.captures[2])
        lvl === nothing && return nothing
        return (re, lvl, 1)
    end
    return nothing
end

function _fq_re_rows_from_component_medians(by_key::Dict{Tuple{Symbol, Int}, Vector{Float64}})
    rows = NamedTuple[]
    re_dims = Dict{Symbol, Set{Int}}()
    for (k, _) in by_key
        re, dim = k
        push!(get!(re_dims, re, Set{Int}()), dim)
    end
    for (k, vals) in sort(collect(by_key); by=x -> (string(first(x[1])), x[1][2]))
        re, dim = k
        st = _descriptive_stats(vals)
        has_single_component = length(get(re_dims, re, Set([dim]))) == 1
        push!(rows, (;
            random_effect=string(re),
            component=(has_single_component ? "-" : "dim$(dim)"),
            n=st.n,
            mean=st.mean,
            sd=st.sd,
            min=st.min,
            q25=st.q25,
            median=st.median,
            q75=st.q75,
            max=st.max,
        ))
    end
    return rows
end

function _fq_mcmc_random_effect_rows(res::FitResult, dm::DataModel)
    re_names = get_re_names(dm.model.random.random)
    isempty(re_names) && return NamedTuple[]
    re_set = Set(re_names)

    chain = get_chain(res)
    arr = _fq_chain_array(chain)
    warmup = _fq_mcmc_warmup(res)
    idx_map = _fq_chain_idx_map(chain)

    by_key = Dict{Tuple{Symbol, Int}, Vector{Float64}}()
    for (k, idx) in pairs(idx_map)
        parsed = _fq_parse_re_chain_name(k, re_set)
        parsed === nothing && continue
        re, _, dim = parsed
        med = median(_fq_chain_values(arr, idx, warmup))
        push!(get!(by_key, (re, dim), Float64[]), med)
    end

    return _fq_re_rows_from_component_medians(by_key)
end

function _fq_vi_random_effect_rows(res::FitResult, dm::DataModel; n_draws::Int=1000)
    re_names = get_re_names(dm.model.random.random)
    isempty(re_names) && return NamedTuple[]
    re_set = Set(re_names)
    n_draws >= 1 || error("n_draws must be >= 1.")
    draw_pack = sample_posterior(res; n_draws=n_draws, rng=Random.Xoshiro(0x6a07), return_names=true)
    draws = draw_pack.draws
    names = draw_pack.names

    by_key = Dict{Tuple{Symbol, Int}, Vector{Float64}}()
    for (j, n) in enumerate(names)
        parsed = _fq_parse_re_chain_name(string(n), re_set)
        parsed === nothing && continue
        re, _, dim = parsed
        med = Float64(median(@view draws[:, j]))
        push!(get!(by_key, (re, dim), Float64[]), med)
    end

    return _fq_re_rows_from_component_medians(by_key)
end

function _fq_random_effect_block(res::FitResult; constants_re::NamedTuple=NamedTuple())
    dm = get_data_model(res)
    dm === nothing && return ("Random effects summary unavailable", NamedTuple[], ["Random-effects summary unavailable: FitResult does not store DataModel."])
    if isempty(get_re_names(dm.model.random.random))
        return ("Random effects summary", NamedTuple[], String[])
    end

    method = get_method(res)
    if method isa MCMC
        rows = _fq_mcmc_random_effect_rows(res, dm)
        label = "Posterior random effects summary (chain medians across draws)"
        return (label, rows, isempty(rows) ? ["No random-effects chain coordinates detected."] : String[])
    elseif method isa VI
        rows = _fq_vi_random_effect_rows(res, dm)
        label = "Posterior random effects summary (VI posterior medians across draws)"
        return (label, rows, isempty(rows) ? ["No random-effects VI coordinates detected."] : String[])
    end

    try
        re_nt = get_random_effects(res; constants_re=constants_re, flatten=true, include_constants=false)
        rows = _fq_re_stats_rows_from_df_nt(re_nt)
        return ("Empirical Bayes random effects summary (across RE levels)", rows, String[])
    catch err
        return ("Empirical Bayes random effects summary unavailable", NamedTuple[], ["Random-effects summary unavailable for method $(nameof(typeof(method))): $(sprint(showerror, err))"])
    end
end

function summarize(res::FitResult;
                   scale::Symbol=:natural,
                   include_non_se::Bool=false,
                   constants_re::NamedTuple=NamedTuple())
    scale = _fq_scale_symbol(scale)
    method = get_method(res)
    inference = _fq_inference_from_method(method)

    param_rows, n_total, n_eligible, notes1 = _fq_fit_parameter_rows(res; scale=scale, include_non_se=include_non_se)
    cov_rows, n_obs_total, n_missing_total, notes2 = _fq_try_outcome_coverage(res)
    re_label, re_rows, notes3 = _fq_random_effect_block(res; constants_re=constants_re)

    notes = String[]
    append!(notes, notes1)
    append!(notes, notes2)
    append!(notes, notes3)

    return FitResultSummary(
        _fq_method_symbol(res),
        inference,
        scale,
        get_objective(res),
        get_converged(res),
        _fq_try_iterations(res),
        _fq_try_loglikelihood(res),
        include_non_se,
        n_total,
        length(param_rows),
        n_eligible,
        param_rows,
        cov_rows,
        n_obs_total,
        n_missing_total,
        re_label,
        re_rows,
        notes,
    )
end

function _fq_uq_scale(scale::Symbol)
    scale in (:natural, :transformed) || error("scale must be :natural or :transformed.")
    return scale
end

function _fq_uq_base_vectors(uq::UQResult; scale::Symbol=:natural)
    est = get_uq_estimates(uq; scale=scale, as_component=false)
    ints = get_uq_intervals(uq; scale=scale, as_component=false)
    vcov = get_uq_vcov(uq; scale=scale)
    draws = get_uq_draws(uq; scale=scale)
    se = nothing
    if vcov !== nothing
        se = [sqrt(max(vcov[i, i], 0.0)) for i in 1:size(vcov, 1)]
    elseif draws !== nothing
        se = vec(std(draws; dims=1, corrected=true))
    end
    return (est=est, ints=ints, se=se)
end

function _fq_uq_interval_label(uq::UQResult)
    inf = _fq_inference_from_uq(uq)
    return inf == :bayesian ? "CrI" : "CI"
end

function summarize(uq::UQResult; scale::Symbol=:natural)
    scale = _fq_uq_scale(scale)
    inference = _fq_inference_from_uq(uq)
    base = _fq_uq_base_vectors(uq; scale=scale)
    level = base.ints === nothing ? nothing : base.ints.level
    interval_label = _fq_uq_interval_label(uq)

    rows = NamedTuple[]
    names = get_uq_parameter_names(uq)
    for i in eachindex(names)
        se_i = base.se === nothing ? nothing : base.se[i]
        lo = base.ints === nothing ? nothing : base.ints.lower[i]
        hi = base.ints === nothing ? nothing : base.ints.upper[i]
        push!(rows, (;
            parameter=names[i],
            role="Unknown",
            estimate=base.est[i],
            std_error=se_i,
            lower=lo,
            upper=hi,
            calculate_se=true,
        ))
    end

    return UQResultSummary(
        uq.backend,
        uq.source_method,
        inference,
        scale,
        nothing,
        level,
        interval_label,
        false,
        length(names),
        length(rows),
        length(rows),
        rows,
        NamedTuple[],
        nothing,
        nothing,
        "",
        NamedTuple[],
        String[],
    )
end

function summarize(res::FitResult, uq::UQResult;
                   scale::Symbol=:natural,
                   include_non_se::Bool=false,
                   constants_re::NamedTuple=NamedTuple())
    scale = _fq_uq_scale(scale)
    dm = get_data_model(res)
    dm === nothing && error("summarize(fit, uq) requires fit to store DataModel.")

    fe = dm.model.fixed.fixed
    model = dm.model
    flat_names = get_flat_names(fe)
    parent_names = _flat_parent_names(fe)
    se_mask = get_se_mask(fe)
    roles_by_parent = _fq_role_by_parent(model)
    fit_est = _fq_fit_component_estimates(res, dm, scale)
    length(fit_est) == length(flat_names) || error("Internal summary error: estimate layout mismatch.")

    base = _fq_uq_base_vectors(uq; scale=scale)
    uq_names = get_uq_parameter_names(uq)
    uq_idx = Dict{Symbol, Int}((uq_names[i] => i) for i in eachindex(uq_names))
    inference = _fq_inference_from_uq(uq)
    interval_label = _fq_uq_interval_label(uq)
    level = base.ints === nothing ? nothing : base.ints.level

    rows = NamedTuple[]
    for i in eachindex(flat_names)
        name = flat_names[i]
        if !include_non_se && !se_mask[i]
            continue
        end
        role = _fq_role_label(get(roles_by_parent, parent_names[i], :General_Outcome))
        if haskey(uq_idx, name)
            j = uq_idx[name]
            se_i = base.se === nothing ? nothing : base.se[j]
            lo = base.ints === nothing ? nothing : base.ints.lower[j]
            hi = base.ints === nothing ? nothing : base.ints.upper[j]
            push!(rows, (;
                parameter=name,
                role=role,
                estimate=base.est[j],
                std_error=se_i,
                lower=lo,
                upper=hi,
                calculate_se=se_mask[i],
            ))
        elseif include_non_se
            push!(rows, (;
                parameter=name,
                role=role,
                estimate=fit_est[i],
                std_error=nothing,
                lower=nothing,
                upper=nothing,
                calculate_se=se_mask[i],
            ))
        end
    end

    cov_rows, n_obs_total, n_missing_total, notes_cov = _fq_try_outcome_coverage(res)
    re_label, re_rows, notes_re = _fq_random_effect_block(res; constants_re=constants_re)
    notes = String[]
    append!(notes, notes_cov)
    append!(notes, notes_re)

    return UQResultSummary(
        uq.backend,
        uq.source_method,
        inference,
        scale,
        get_objective(res),
        level,
        interval_label,
        include_non_se,
        length(flat_names),
        length(rows),
        count(identity, se_mask),
        rows,
        cov_rows,
        n_obs_total,
        n_missing_total,
        re_label,
        re_rows,
        notes,
    )
end

function _fq_print_parameter_table_fit(io::IO, rows::Vector{NamedTuple})
    println(io, "Parameter estimates")
    isempty(rows) && (println(io, "  (none)"); return)
    name_w = max(length("parameter"), maximum(length(string(r.parameter)) for r in rows))
    println(io, "  ", rpad("parameter", name_w), "  ", lpad("Estimate", 12))
    println(io, "  ", repeat("-", name_w + 14))
    for r in rows
        println(io, "  ", rpad(string(r.parameter), name_w), "  ", lpad(_fq_fmt_num(r.estimate), 12))
    end
end

function _fq_print_parameter_table_uq(io::IO, rows::Vector{NamedTuple}, interval_label::String, show_se::Bool)
    println(io, "Parameter uncertainty summary")
    isempty(rows) && (println(io, "  (none)"); return)
    name_w = max(length("parameter"), maximum(length(string(r.parameter)) for r in rows))
    if show_se
        println(io, "  ", rpad("parameter", name_w), "  ",
                lpad("Estimate", 12), "  ", lpad("Std. Error", 12), "  ",
                lpad("$(interval_label) Lower", 12), "  ", lpad("$(interval_label) Upper", 12))
        println(io, "  ", repeat("-", name_w + 42))
        for r in rows
            println(io, "  ", rpad(string(r.parameter), name_w), "  ",
                    lpad(_fq_fmt_num(r.estimate), 12), "  ", lpad(_fq_fmt_num(r.std_error), 12), "  ",
                    lpad(_fq_fmt_num(r.lower), 12), "  ", lpad(_fq_fmt_num(r.upper), 12))
        end
    else
        println(io, "  ", rpad("parameter", name_w), "  ",
                lpad("Estimate", 12), "  ", lpad("$(interval_label) Lower", 12), "  ", lpad("$(interval_label) Upper", 12))
        println(io, "  ", repeat("-", name_w + 28))
        for r in rows
            println(io, "  ", rpad(string(r.parameter), name_w), "  ",
                    lpad(_fq_fmt_num(r.estimate), 12), "  ",
                    lpad(_fq_fmt_num(r.lower), 12), "  ", lpad(_fq_fmt_num(r.upper), 12))
        end
    end
end

function _fq_print_coverage_table(io::IO, rows::Vector{NamedTuple}, n_obs_total, n_missing_total)
    println(io, "Outcome data coverage")
    isempty(rows) && (println(io, "  (none)"); return)
    name_w = max(length("outcome"), maximum(length(string(r.outcome)) for r in rows))
    println(io, "  ", rpad("outcome", name_w), "  ", lpad("n_obs", 10), "  ", lpad("n_missing", 10))
    println(io, "  ", repeat("-", name_w + 24))
    for r in rows
        println(io, "  ", rpad(string(r.outcome), name_w), "  ", lpad(string(r.n_obs), 10), "  ", lpad(string(r.n_missing), 10))
    end
    println(io, "  ", rpad("TOTAL", name_w), "  ", lpad(string(n_obs_total), 10), "  ", lpad(string(n_missing_total), 10))
end

function _fq_print_re_table(io::IO, label::String, rows::Vector{NamedTuple})
    println(io, label)
    isempty(rows) && (println(io, "  (none)"); return)
    re_w = max(length("random effect"), maximum(length(string(r.random_effect)) for r in rows))
    show_component = any(r -> string(r.component) != "-", rows)
    if show_component
        cmp_w = max(length("component"), maximum(length(string(r.component)) for r in rows))
        println(io, "  ", rpad("random effect", re_w), "  ", rpad("component", cmp_w), "  ",
                lpad("n", 6), "  ", lpad("mean", 12), "  ", lpad("sd", 12), "  ",
                lpad("q25", 12), "  ", lpad("median", 12), "  ", lpad("q75", 12))
        println(io, "  ", repeat("-", re_w + cmp_w + 76))
        for r in rows
            println(io, "  ", rpad(string(r.random_effect), re_w), "  ", rpad(string(r.component), cmp_w), "  ",
                    lpad(string(r.n), 6), "  ",
                    lpad(_fq_fmt_num(r.mean), 12), "  ",
                    lpad(_fq_fmt_num(r.sd), 12), "  ",
                    lpad(_fq_fmt_num(r.q25), 12), "  ",
                    lpad(_fq_fmt_num(r.median), 12), "  ",
                    lpad(_fq_fmt_num(r.q75), 12))
        end
    else
        println(io, "  ", rpad("random effect", re_w), "  ",
                lpad("n", 6), "  ", lpad("mean", 12), "  ", lpad("sd", 12), "  ",
                lpad("q25", 12), "  ", lpad("median", 12), "  ", lpad("q75", 12))
        println(io, "  ", repeat("-", re_w + 62))
        for r in rows
            println(io, "  ", rpad(string(r.random_effect), re_w), "  ",
                    lpad(string(r.n), 6), "  ",
                    lpad(_fq_fmt_num(r.mean), 12), "  ",
                    lpad(_fq_fmt_num(r.sd), 12), "  ",
                    lpad(_fq_fmt_num(r.q25), 12), "  ",
                    lpad(_fq_fmt_num(r.median), 12), "  ",
                    lpad(_fq_fmt_num(r.q75), 12))
        end
    end
end

Base.show(io::IO, s::FitResultSummary) = show(io, MIME"text/plain"(), s)
Base.show(io::IO, s::UQResultSummary) = show(io, MIME"text/plain"(), s)

function Base.show(io::IO, ::MIME"text/plain", s::FitResultSummary)
    println(io, "FitResultSummary")
    println(io, repeat("═", 96))
    _fq_print_key_values(io, "Overview", [
        "method" => s.method,
        "inference" => s.inference,
        "scale" => s.scale,
        "objective" => _fq_fmt_objective(s.objective),
        "iterations" => s.iterations,
        "parameters shown (reported / total)" => "$(s.n_parameters_reported) / $(s.n_parameters_total)",
    ])
    println(io)
    _fq_print_parameter_table_fit(io, s.parameter_rows)
    println(io)
    _fq_print_coverage_table(io, s.coverage_rows, s.n_obs_total, s.n_missing_total)
    if !isempty(s.random_effect_rows)
        println(io)
        _fq_print_re_table(io, s.random_effect_label, s.random_effect_rows)
    end
    if !isempty(s.notes)
        println(io)
        _fq_print_key_values(io, "Notes", [string("note ", i) => s.notes[i] for i in eachindex(s.notes)])
    end
end

function Base.show(io::IO, ::MIME"text/plain", s::UQResultSummary)
    println(io, "UQResultSummary")
    println(io, repeat("═", 96))
    show_se = s.inference != :bayesian
    level_str = s.level === nothing ? "-" : _fq_fmt_num(s.level)
    _fq_print_key_values(io, "Overview", [
        "backend" => s.backend,
        "source_method" => s.source_method,
        "inference" => s.inference,
        "scale" => s.scale,
        "objective" => _fq_fmt_objective(s.objective),
        "interval level" => level_str,
        "parameters shown (reported / total)" => "$(s.n_parameters_reported) / $(s.n_parameters_total)",
    ])
    println(io)
    _fq_print_parameter_table_uq(io, s.parameter_rows, s.interval_label, show_se)
    show_cov = !isempty(s.coverage_rows) || s.n_obs_total !== nothing || s.n_missing_total !== nothing
    if show_cov
        println(io)
        _fq_print_coverage_table(io, s.coverage_rows, s.n_obs_total, s.n_missing_total)
    end
    if !isempty(s.random_effect_rows)
        println(io)
        _fq_print_re_table(io, s.random_effect_label, s.random_effect_rows)
    end
    if !isempty(s.notes)
        println(io)
        _fq_print_key_values(io, "Notes", [string("note ", i) => s.notes[i] for i in eachindex(s.notes)])
    end
end
