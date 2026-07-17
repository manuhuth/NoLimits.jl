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
    vary = get_vary(get_series(ind))
    if hasproperty(vary, :t)
        push!(pairs, :t => getproperty(vary, :t)[idx])
    else
        push!(pairs, :t => get_df(dm)[row, get_time_col(dm)])
    end
    for name in keys(vary)
        name == :t && continue
        v = getfield(vary, name)
        if v isa AbstractVector
            push!(pairs, name => v[idx])
        elseif v isa NamedTuple
            sub = _covariate_vector(NamedTuple{keys(v)}(Tuple(getfield(v, k)[idx]
            for k in keys(v))))
            push!(pairs, name => sub)
        end
    end
    return merge(NamedTuple(pairs), get_dyn(get_series(ind)))
end

function _rngs_for_serialization(rng, serialization)
    if serialization isa SciMLBase.EnsembleThreads
        # Threads.@threads may use thread ids up to maxthreadid()
        return [Random.MersenneTwister(rand(rng, UInt)) for _ in 1:Threads.maxthreadid()]
    end
    return rng
end

function _sampled_random_effects_for_individual(dm::DataModel, idx::Int, re_samples)
    re_names = get_re_names(get_random(get_model(dm)))
    isempty(re_names) && return ComponentArray(NamedTuple())
    ind = get_individuals(dm)[idx]
    info = get_re_group_info(dm).index_by_individual
    nt_pairs = Pair{Symbol, Any}[]
    for re in re_names
        re_info = getfield(info, re)
        nlevels = length(getfield(get_re_groups(ind), re))
        local_level_ids = fill(0, nlevels)
        level_ids_all = re_info.level_ids_all[idx]
        unique_pos_all = re_info.unique_pos_all[idx]
        @inbounds for k in eachindex(level_ids_all)
            pos = unique_pos_all[k]
            local_level_ids[pos] == 0 && (local_level_ids[pos] = level_ids_all[k])
        end
        sample = re_samples[re][local_level_ids[1]]
        if nlevels == 1
            push!(nt_pairs, re => (sample isa AbstractVector ? copy(sample) : sample))
            continue
        end
        if sample isa AbstractVector
            sample_copy = copy(sample)
            vals_re = Vector{typeof(sample_copy)}(undef, nlevels)
            vals_re[1] = sample_copy
            @inbounds for pos in 2:nlevels
                vals_re[pos] = copy(re_samples[re][local_level_ids[pos]])
            end
            push!(nt_pairs, re => vals_re)
        else
            vals_re = Vector{typeof(sample)}(undef, nlevels)
            vals_re[1] = sample
            @inbounds for pos in 2:nlevels
                vals_re[pos] = re_samples[re][local_level_ids[pos]]
            end
            push!(nt_pairs, re => vals_re)
        end
    end
    return ComponentArray(NamedTuple(nt_pairs))
end

# Shared DE setup for the two per-individual simulators: build the compiled
# context, solve the individual's ODE problem, and return the solution accessors
# (`nothing` for non-DE models). Draws no randomness, so factoring this out of
# the simulators leaves their RNG streams unchanged.
function _simulate_sol_accessors(dm::DataModel, idx::Int, θ, η)
    model = get_model(dm)
    get_de(model) === nothing && return nothing
    ind = get_individuals(dm)[idx]
    const_cov = get_const_cov(ind)
    pre = calculate_prede(model, θ, η, const_cov)
    pc = (;
        fixed_effects = θ,
        random_effects = η,
        constant_covariates = const_cov,
        varying_covariates = merge(
            (t = get_vary(get_series(ind)).t[1],), get_dyn(get_series(ind))),
        helpers = get_helper_funs(model),
        model_funs = get_model_funs(model),
        preDE = pre
    )
    compiled = get_de_compiler(get_de(model))(pc)
    u0 = calculate_initial_state(model, θ, η, const_cov)
    cb = nothing
    infusion_rates = nothing
    if get_callbacks(ind) !== nothing
        _apply_initial_events!(u0, get_callbacks(ind))
        cb = get_callback(get_callbacks(ind))
        infusion_rates = get_infusion_rates(get_callbacks(ind))
    end
    plan = get_closed_form_plan(dm)
    if is_cf_eligible(plan) && (_cf_is_whole(plan) || cb === nothing)
        solver_cfg = get_solver_config(model)
        cf_alg = solver_cfg.alg === nothing ? Tsit5() : solver_cfg.alg
        sol = _cf_dispatch_solve(
            model, compiled, u0, get_tspan(ind), get_saveat(ind), plan,
            get_callbacks(ind), cf_alg, solver_cfg.args,
            _ode_solve_kwargs(solver_cfg.kwargs, NamedTuple(), NamedTuple()))
        sol !== nothing && return get_de_accessors_builder(get_de(model))(sol, compiled)
    end
    f!_use = _with_infusion(get_de_f!(get_de(model)), infusion_rates)
    prob = ODEProblem(f!_use, u0, get_tspan(ind), compiled)
    solver_cfg = get_solver_config(model)
    alg = solver_cfg.alg === nothing ? Tsit5() : solver_cfg.alg
    if get_saveat(ind) === nothing
        solve_kwargs = _ode_solve_kwargs(
            solver_cfg.kwargs, NamedTuple(), (dense = true,))
        sol = cb === nothing ?
              solve(prob, alg, solver_cfg.args...; solve_kwargs...) :
              solve(prob, alg, solver_cfg.args...; solve_kwargs..., callback = cb)
    else
        solve_kwargs = _ode_solve_kwargs(solver_cfg.kwargs, NamedTuple(),
            (saveat = get_saveat(ind), save_everystep = false, dense = false))
        sol = cb === nothing ?
              solve(prob, alg, solver_cfg.args...; solve_kwargs...) :
              solve(prob, alg, solver_cfg.args...; solve_kwargs..., callback = cb)
    end
    return get_de_accessors_builder(get_de(model))(sol, compiled)
end

function _simulate_individual!(
        df::DataFrame, dm::DataModel, idx::Int, θ, re_samples, rng, replace_missings::Bool)
    model = get_model(dm)
    ind = get_individuals(dm)[idx]
    obs_rows = get_obs_rows(get_row_groups(dm))[idx]
    const_cov = get_const_cov(ind)

    η = _sampled_random_effects_for_individual(dm, idx, re_samples)
    sol_accessors = _simulate_sol_accessors(dm, idx, θ, η)

    obs_cols = get_obs_cols(dm)
    rowwise_re = _needs_rowwise_random_effects(dm, idx; obs_only = true)
    hmm_states = Dict{Symbol, Int}()
    for (i, row) in enumerate(obs_rows)
        vary = _varying_at(dm, ind, i, row)
        η_row = _row_random_effects_at(dm, idx, i, η, rowwise_re; obs_only = true)
        obs = sol_accessors === nothing ?
              calculate_formulas_obs(model, θ, η_row, const_cov, vary) :
              calculate_formulas_obs(model, θ, η_row, const_cov, vary, sol_accessors)
        for col in obs_cols
            dist = getproperty(obs, col)
            if _is_hmm_dist(dist)
                prev_state = get(hmm_states, col, 0)
                state = prev_state == 0 ?
                        _sample_hmm_hidden_state(rng, dist) :
                        _sample_hmm_hidden_state(rng, dist, prev_state)
                hmm_states[col] = state
                if !replace_missings && ismissing(df[row, col])
                    continue
                end
                val = _hmm_emission_rand(rng, dist, state)
            else
                if !replace_missings && ismissing(df[row, col])
                    continue
                end
                val = rand(rng, dist)
            end
            if val isa Number && (isnan(val) || isinf(val))
                _warn_bad_value(dm, row, col, dist, val)
            end
            df[row, col] = val
        end
    end
    return nothing
end

function _simulate_individual_values(
        dm::DataModel, idx::Int, θ, re_samples, rng, replace_missings::Bool)
    model = get_model(dm)
    ind = get_individuals(dm)[idx]
    obs_rows = get_obs_rows(get_row_groups(dm))[idx]
    const_cov = get_const_cov(ind)

    η = _sampled_random_effects_for_individual(dm, idx, re_samples)
    sol_accessors = _simulate_sol_accessors(dm, idx, θ, η)

    obs_cols = get_obs_cols(dm)
    out = Dict{Symbol, Vector{Any}}()
    for col in obs_cols
        out[col] = Vector{Any}(undef, length(obs_rows))
    end

    rowwise_re = _needs_rowwise_random_effects(dm, idx; obs_only = true)
    hmm_states = Dict{Symbol, Int}()
    for (i, row) in enumerate(obs_rows)
        vary = _varying_at(dm, ind, i, row)
        η_row = _row_random_effects_at(dm, idx, i, η, rowwise_re; obs_only = true)
        obs = sol_accessors === nothing ?
              calculate_formulas_obs(model, θ, η_row, const_cov, vary) :
              calculate_formulas_obs(model, θ, η_row, const_cov, vary, sol_accessors)
        for col in obs_cols
            dist = getproperty(obs, col)
            if _is_hmm_dist(dist)
                prev_state = get(hmm_states, col, 0)
                state = prev_state == 0 ?
                        _sample_hmm_hidden_state(rng, dist) :
                        _sample_hmm_hidden_state(rng, dist, prev_state)
                hmm_states[col] = state
                if !replace_missings && ismissing(get_df(dm)[row, col])
                    out[col][i] = nothing
                    continue
                end
                val = _hmm_emission_rand(rng, dist, state)
            else
                if !replace_missings && ismissing(get_df(dm)[row, col])
                    out[col][i] = nothing
                    continue
                end
                val = rand(rng, dist)
            end
            if val isa Number && (isnan(val) || isinf(val))
                _warn_bad_value(dm, row, col, dist, val)
            end
            out[col][i] = val
        end
    end
    return obs_rows, out
end

function _warn_bad_value(dm::DataModel, row::Int, col::Symbol, dist, val)
    id_val = get_df(dm)[row, get_primary_id(dm)]
    t_val = get_df(dm)[row, get_time_col(dm)]
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

function _sample_random_effects(dm::DataModel, rng, θ)
    model = get_model(dm)
    re_names = get_re_names(get_random(model))
    isempty(re_names) && return Dict{Symbol, Any}()
    re_groups = get_re_groups(get_random(model))
    re_values = get_re_group_info(dm).values
    re_index_by_row = get_re_group_info(dm).index_by_row
    cov = get_covariates(model)

    model_funs = get_model_funs(model)
    helpers = get_helper_funs(model)
    create = create_random_effect_distribution(get_random(model))

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
            const_cov = _build_const_cov_from_row(cov, get_df(dm), row_idx)
            dists = create(θ, const_cov, model_funs, helpers)
            vals[gidx] = rand(rng, getproperty(dists, re))
        end
        samples[re] = vals
    end
    return samples
end

function _attach_random_effects!(df::DataFrame, dm::DataModel, re_samples)
    re_names = get_re_names(get_random(get_model(dm)))
    isempty(re_names) && return nothing
    index_by_row = get_re_group_info(dm).index_by_row
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

function _resolve_sim_theta(dm::DataModel, theta_untransformed)
    fe = get_fixed(get_model(dm))
    if theta_untransformed === nothing
        return get_θ0_untransformed(fe)
    end
    θ = theta_untransformed isa ComponentArray ? theta_untransformed :
        ComponentArray(theta_untransformed)
    for name in get_names(fe)
        hasproperty(θ, name) || error("theta_untransformed is missing parameter $(name).")
    end
    return θ
end

"""
    simulate_data(dm::DataModel; rng, replace_missings, serialization, theta_untransformed) -> DataFrame

Simulate observations from a `DataModel` using the provided parameter values, or the model's initial parameter values if none are given.


Random effects are drawn from their prior distributions and observation columns are
replaced with draws from the model's observation distributions. Non-observation columns
are left unchanged.

# Keyword Arguments
- `rng::AbstractRNG = Random.default_rng()`: random-number generator.
- `replace_missings::Bool = false`: if `true`, fill `missing` observation entries with
  simulated values; otherwise leave them as `missing`.
- `serialization::SciMLBase.EnsembleAlgorithm = EnsembleSerial()`: parallelisation
  strategy (e.g. `EnsembleThreads()`).
- `theta_untransformed = nothing`: fixed-effect parameter vector used for simulation on
  the natural scale. If `nothing`, use the model's declared initial values (`θ0`).

# Returns
A copy of `dm.df` with simulated observation values.
"""
function simulate_data(dm::DataModel; rng = Random.default_rng(),
        replace_missings::Bool = false,
        serialization::SciMLBase.EnsembleAlgorithm = EnsembleSerial(),
        theta_untransformed = nothing)
    df = deepcopy(get_df(dm))
    θ = _resolve_sim_theta(dm, theta_untransformed)

    re_samples = _sample_random_effects(dm, rng, θ)
    _attach_random_effects!(df, dm, re_samples)

    rngs = _rngs_for_serialization(rng, serialization)
    if serialization isa SciMLBase.EnsembleThreads
        # Chunk-indexed RNG assignment: each task owns RNG slot `c` for its whole stride.
        # `rngs[Threads.threadid()]` is unsafe under task migration (two tasks could share
        # one RNG, racing on its state and making the simulation nondeterministic).
        n_chunks = length(rngs)
        Threads.@threads for c in 1:n_chunks
            rng_c = rngs[c]
            for i in c:n_chunks:length(get_individuals(dm))
                _simulate_individual!(df, dm, i, θ, re_samples, rng_c, replace_missings)
            end
        end
    elseif serialization isa SciMLBase.EnsembleDistributed
        parts = pmap(
            i -> begin
                local_rng = Random.MersenneTwister(rand(rng, UInt))
                _simulate_individual_values(
                    dm, i, θ, re_samples, local_rng, replace_missings)
            end,
            collect(eachindex(get_individuals(dm))))
        for (obs_rows, out) in parts
            for col in get_obs_cols(dm)
                vals = out[col]
                for (j, row) in enumerate(obs_rows)
                    vals[j] === nothing && continue
                    df[row, col] = vals[j]
                end
            end
        end
    else
        for i in eachindex(get_individuals(dm))
            _simulate_individual!(df, dm, i, θ, re_samples, rng, replace_missings)
        end
    end

    return df
end

"""
    simulate_data_model(dm::DataModel; rng, replace_missings, serialization, theta_untransformed) -> DataModel

Simulate observations from a `DataModel` and return a new `DataModel` wrapping the
simulated data.

Calls [`simulate_data`](@ref) and constructs a fresh `DataModel` from the resulting
`DataFrame`, preserving the original model, id columns, and serialization settings.

# Keyword Arguments
- `rng::AbstractRNG = Random.default_rng()`: random-number generator.
- `replace_missings::Bool = false`: forwarded to [`simulate_data`](@ref).
- `serialization::SciMLBase.EnsembleAlgorithm`: parallelisation strategy; defaults to
  the strategy stored in `dm`.
- `theta_untransformed = nothing`: fixed-effect parameter vector used for simulation on
  the natural scale. Forwarded to [`simulate_data`](@ref).
"""
function simulate_data_model(dm::DataModel; rng = Random.default_rng(),
        replace_missings::Bool = false,
        serialization::SciMLBase.EnsembleAlgorithm = dm.config.serialization,
        theta_untransformed = nothing)
    df_sim = simulate_data(dm;
        rng = rng,
        replace_missings = replace_missings,
        serialization = serialization,
        theta_untransformed = theta_untransformed)
    cfg = dm.config
    return DataModel(get_model(dm), df_sim;
        primary_id = cfg.primary_id,
        time_col = cfg.time_col,
        evid_col = cfg.evid_col,
        amt_col = cfg.amt_col,
        rate_col = cfg.rate_col,
        cmt_col = cfg.cmt_col,
        serialization = serialization)
end
