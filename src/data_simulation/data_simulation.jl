export simulate_data
export simulate_data_model

using Random
using DataFrames
using Distributions
using OrdinaryDiffEq
using SciMLBase
using ComponentArrays
using Distributed

function _build_const_cov_from_row(covariates, df, row_idx::Int)
    return _build_const_cov(covariates, df, [row_idx])
end

function _varying_at(dm::DataModel, ind::Individual, idx::Int, row::Int)
    pairs = Pair{Symbol, Any}[]
    vary = ind.series.vary
    if hasproperty(vary, :t)
        push!(pairs, :t => getproperty(vary, :t)[idx])
    else
        push!(pairs, :t => dm.df[row, dm.config.time_col])
    end
    for name in keys(vary)
        name == :t && continue
        v = getfield(vary, name)
        if v isa AbstractVector
            push!(pairs, name => v[idx])
        elseif v isa NamedTuple
            sub = NamedTuple{keys(v)}(Tuple(getfield(v, k)[idx] for k in keys(v)))
            push!(pairs, name => sub)
        end
    end
    return merge(NamedTuple(pairs), ind.series.dyn)
end

function _rngs_for_serialization(rng, serialization)
    if serialization isa SciMLBase.EnsembleThreads
        # Threads.@threads may use thread ids up to maxthreadid()
        return [Random.MersenneTwister(rand(rng, UInt)) for _ in 1:Threads.maxthreadid()]
    end
    return rng
end

function _simulate_individual!(df::DataFrame, dm::DataModel, idx::Int, θ, re_samples, rng, replace_missings::Bool)
    model = dm.model
    ind = dm.individuals[idx]
    rows = dm.row_groups.rows[idx]
    obs_rows = dm.row_groups.obs_rows[idx]
    const_cov = ind.const_cov

    re_names = get_re_names(model.random.random)
    re_info = dm.re_group_info.index_by_row
    re_pairs = Pair{Symbol, Any}[]
    for re in re_names
        group_idx = getfield(re_info, re)[rows[1]]
        push!(re_pairs, re => re_samples[re][group_idx])
    end
    η = ComponentArray(NamedTuple(re_pairs))

    sol_accessors = nothing
    if model.de.de !== nothing
        pre = calculate_prede(model, θ, η, const_cov)
        pc = (;
            fixed_effects = θ,
            random_effects = η,
            constant_covariates = const_cov,
            varying_covariates = merge((t = ind.series.vary.t[1],), ind.series.dyn),
            helpers = get_helper_funs(model),
            model_funs = get_model_funs(model),
            preDE = pre
        )
        compiled = get_de_compiler(model.de.de)(pc)
        u0 = calculate_initial_state(model, θ, η, const_cov)
        cb = nothing
        infusion_rates = nothing
        if ind.callbacks !== nothing
            _apply_initial_events!(u0, ind.callbacks)
            cb = ind.callbacks.callback
            infusion_rates = ind.callbacks.infusion_rates
        end
        f!_use = _with_infusion(get_de_f!(model.de.de), infusion_rates)
        prob = ODEProblem(f!_use, u0, ind.tspan, compiled)
        solver_cfg = get_solver_config(model)
        alg = solver_cfg.alg === nothing ? Tsit5() : solver_cfg.alg
        if ind.saveat === nothing
            solve_kwargs = _ode_solve_kwargs(solver_cfg.kwargs, NamedTuple(), (dense=true,))
            sol = cb === nothing ?
                  solve(prob, alg, solver_cfg.args...; solve_kwargs...) :
                  solve(prob, alg, solver_cfg.args...; solve_kwargs..., callback=cb)
        else
            solve_kwargs = _ode_solve_kwargs(solver_cfg.kwargs, NamedTuple(),
                                             (saveat=ind.saveat, save_everystep=false, dense=false))
            sol = cb === nothing ?
                  solve(prob, alg, solver_cfg.args...; solve_kwargs...) :
                  solve(prob, alg, solver_cfg.args...; solve_kwargs..., callback=cb)
        end
        sol_accessors = get_de_accessors_builder(model.de.de)(sol, compiled)
    end

    obs_cols = dm.config.obs_cols
    for (i, row) in enumerate(obs_rows)
        vary = _varying_at(dm, ind, i, row)
        obs = sol_accessors === nothing ?
              calculate_formulas_obs(model, θ, η, const_cov, vary) :
              calculate_formulas_obs(model, θ, η, const_cov, vary, sol_accessors)
        for col in obs_cols
            if !replace_missings && ismissing(df[row, col])
                continue
            end
            dist = getproperty(obs, col)
            val = rand(rng, dist)
            if val isa Number && (isnan(val) || isinf(val))
                _warn_bad_value(dm, row, col, dist, val)
            end
            df[row, col] = val
        end
    end
    return nothing
end

function _simulate_individual_values(dm::DataModel, idx::Int, θ, re_samples, rng, replace_missings::Bool)
    model = dm.model
    ind = dm.individuals[idx]
    rows = dm.row_groups.rows[idx]
    obs_rows = dm.row_groups.obs_rows[idx]
    const_cov = ind.const_cov

    re_names = get_re_names(model.random.random)
    re_info = dm.re_group_info.index_by_row
    re_pairs = Pair{Symbol, Any}[]
    for re in re_names
        group_idx = getfield(re_info, re)[rows[1]]
        push!(re_pairs, re => re_samples[re][group_idx])
    end
    η = ComponentArray(NamedTuple(re_pairs))

    sol_accessors = nothing
    if model.de.de !== nothing
        pre = calculate_prede(model, θ, η, const_cov)
        pc = (;
            fixed_effects = θ,
            random_effects = η,
            constant_covariates = const_cov,
            varying_covariates = merge((t = ind.series.vary.t[1],), ind.series.dyn),
            helpers = get_helper_funs(model),
            model_funs = get_model_funs(model),
            preDE = pre
        )
        compiled = get_de_compiler(model.de.de)(pc)
        u0 = calculate_initial_state(model, θ, η, const_cov)
        cb = nothing
        infusion_rates = nothing
        if ind.callbacks !== nothing
            _apply_initial_events!(u0, ind.callbacks)
            cb = ind.callbacks.callback
            infusion_rates = ind.callbacks.infusion_rates
        end
        f!_use = _with_infusion(get_de_f!(model.de.de), infusion_rates)
        prob = ODEProblem(f!_use, u0, ind.tspan, compiled)
        solver_cfg = get_solver_config(model)
        alg = solver_cfg.alg === nothing ? Tsit5() : solver_cfg.alg
        if ind.saveat === nothing
            solve_kwargs = _ode_solve_kwargs(solver_cfg.kwargs, NamedTuple(), (dense=true,))
            sol = cb === nothing ?
                  solve(prob, alg, solver_cfg.args...; solve_kwargs...) :
                  solve(prob, alg, solver_cfg.args...; solve_kwargs..., callback=cb)
        else
            solve_kwargs = _ode_solve_kwargs(solver_cfg.kwargs, NamedTuple(),
                                             (saveat=ind.saveat, save_everystep=false, dense=false))
            sol = cb === nothing ?
                  solve(prob, alg, solver_cfg.args...; solve_kwargs...) :
                  solve(prob, alg, solver_cfg.args...; solve_kwargs..., callback=cb)
        end
        sol_accessors = get_de_accessors_builder(model.de.de)(sol, compiled)
    end

    obs_cols = dm.config.obs_cols
    out = Dict{Symbol, Vector{Any}}()
    for col in obs_cols
        out[col] = Vector{Any}(undef, length(obs_rows))
    end

    for (i, row) in enumerate(obs_rows)
        vary = _varying_at(dm, ind, i, row)
        obs = sol_accessors === nothing ?
              calculate_formulas_obs(model, θ, η, const_cov, vary) :
              calculate_formulas_obs(model, θ, η, const_cov, vary, sol_accessors)
        for col in obs_cols
            if !replace_missings && ismissing(dm.df[row, col])
                out[col][i] = nothing
                continue
            end
            dist = getproperty(obs, col)
            val = rand(rng, dist)
            if val isa Number && (isnan(val) || isinf(val))
                _warn_bad_value(dm, row, col, dist, val)
            end
            out[col][i] = val
        end
    end
    return obs_rows, out
end

function _warn_bad_value(dm::DataModel, row::Int, col::Symbol, dist, val)
    id_val = dm.df[row, dm.config.primary_id]
    t_val = dm.df[row, dm.config.time_col]
    dist_str = string(typeof(dist))
    param_str = ""
    try
        param_str = string(Distributions.params(dist))
    catch
        param_str = "unavailable"
    end
    kind = isnan(val) ? "NaN" : "Inf"
    @warn "Invalid simulated value ($(kind))" individual=id_val row=row time=t_val outcome=col distribution=dist_str params=param_str value=val
    return nothing
end

function _sample_random_effects(dm::DataModel, rng)
    model = dm.model
    re_names = get_re_names(model.random.random)
    isempty(re_names) && return Dict{Symbol, Any}()
    re_groups = get_re_groups(model.random.random)
    re_values = dm.re_group_info.values
    re_index_by_row = dm.re_group_info.index_by_row
    cov = model.covariates.covariates
    θ = get_θ0_untransformed(model.fixed.fixed)
    model_funs = get_model_funs(model)
    helpers = get_helper_funs(model)
    create = get_create_random_effect_distribution(model.random.random)

    samples = Dict{Symbol, Vector{Any}}()
    for re in re_names
        group_vals = getfield(re_values, re)
        index_by_row = getfield(re_index_by_row, re)
        first_row = fill(0, length(group_vals))
        for i in eachindex(index_by_row)
            idx = index_by_row[i]
            first_row[idx] == 0 && (first_row[idx] = i)
        end
        vals = Vector{Any}(undef, length(group_vals))
        for gidx in eachindex(group_vals)
            row_idx = first_row[gidx]
            const_cov = _build_const_cov_from_row(cov, dm.df, row_idx)
            dists = create(θ, const_cov, model_funs, helpers)
            vals[gidx] = rand(rng, getproperty(dists, re))
        end
        samples[re] = vals
    end
    return samples
end

function _attach_random_effects!(df::DataFrame, dm::DataModel, re_samples)
    re_names = get_re_names(dm.model.random.random)
    isempty(re_names) && return nothing
    index_by_row = dm.re_group_info.index_by_row
    for re in re_names
        vals = re_samples[re]
        idxs = getfield(index_by_row, re)
        first_val = vals[1]
        flat_names = flatten_re_names(re, first_val)
        flat_vals = [Vector{Any}(undef, nrow(df)) for _ in flat_names]
        for row in 1:nrow(df)
            v = vals[idxs[row]]
            vflat = flatten_re_values(v)
            for j in eachindex(flat_vals)
                flat_vals[j][row] = vflat[j]
            end
        end
        for (j, name) in enumerate(flat_names)
            df[!, name] = flat_vals[j]
        end
    end
    return nothing
end

"""
    simulate_data(dm::DataModel; rng, replace_missings, serialization) -> DataFrame

Simulate observations from a `DataModel` using the model's initial parameter values.

Random effects are drawn from their prior distributions and observation columns are
replaced with draws from the model's observation distributions. Non-observation columns
are left unchanged.

# Keyword Arguments
- `rng::AbstractRNG = Random.default_rng()`: random-number generator.
- `replace_missings::Bool = false`: if `true`, fill `missing` observation entries with
  simulated values; otherwise leave them as `missing`.
- `serialization::SciMLBase.EnsembleAlgorithm = EnsembleSerial()`: parallelisation
  strategy (e.g. `EnsembleThreads()`).

# Returns
A copy of `dm.df` with simulated observation values.
"""
function simulate_data(dm::DataModel; rng=Random.default_rng(),
                       replace_missings::Bool=false,
                       serialization::SciMLBase.EnsembleAlgorithm=EnsembleSerial())
    df = deepcopy(dm.df)
    θ = get_θ0_untransformed(dm.model.fixed.fixed)

    re_samples = _sample_random_effects(dm, rng)
    _attach_random_effects!(df, dm, re_samples)

    rngs = _rngs_for_serialization(rng, serialization)
    if serialization isa SciMLBase.EnsembleThreads
        Threads.@threads for i in eachindex(dm.individuals)
            _simulate_individual!(df, dm, i, θ, re_samples, rngs[Threads.threadid()], replace_missings)
        end
    elseif serialization isa SciMLBase.EnsembleDistributed
        parts = pmap(i -> begin
                local_rng = Random.MersenneTwister(rand(rng, UInt))
                _simulate_individual_values(dm, i, θ, re_samples, local_rng, replace_missings)
            end, collect(eachindex(dm.individuals)))
        for (obs_rows, out) in parts
            for col in dm.config.obs_cols
                vals = out[col]
                for (j, row) in enumerate(obs_rows)
                    vals[j] === nothing && continue
                    df[row, col] = vals[j]
                end
            end
        end
    else
        for i in eachindex(dm.individuals)
            _simulate_individual!(df, dm, i, θ, re_samples, rng, replace_missings)
        end
    end

    return df
end

"""
    simulate_data_model(dm::DataModel; rng, replace_missings, serialization) -> DataModel

Simulate observations from a `DataModel` and return a new `DataModel` wrapping the
simulated data.

Calls [`simulate_data`](@ref) and constructs a fresh `DataModel` from the resulting
`DataFrame`, preserving the original model, id columns, and serialization settings.

# Keyword Arguments
- `rng::AbstractRNG = Random.default_rng()`: random-number generator.
- `replace_missings::Bool = false`: forwarded to [`simulate_data`](@ref).
- `serialization::SciMLBase.EnsembleAlgorithm`: parallelisation strategy; defaults to
  the strategy stored in `dm`.
"""
function simulate_data_model(dm::DataModel; rng=Random.default_rng(),
                             replace_missings::Bool=false,
                             serialization::SciMLBase.EnsembleAlgorithm=dm.config.serialization)
    df_sim = simulate_data(dm; rng=rng, replace_missings=replace_missings, serialization=serialization)
    cfg = dm.config
    return DataModel(dm.model, df_sim;
                     primary_id=cfg.primary_id,
                     time_col=cfg.time_col,
                     evid_col=cfg.evid_col,
                     amt_col=cfg.amt_col,
                     rate_col=cfg.rate_col,
                     cmt_col=cfg.cmt_col,
                     serialization=serialization)
end
