export plot_vpc

using Distributions
using Random
using StatsFuns

function _require_varying_covariate(dm::DataModel, x_axis_feature)
    cov = get_covariates(get_model(dm))
    if x_axis_feature === nothing
        return get_time_col(dm)
    end
    if x_axis_feature == get_time_col(dm)
        return x_axis_feature
    end
    x_axis_feature in cov.varying ||
        error("x_axis_feature must be a varying covariate. Got $(x_axis_feature).")
    return x_axis_feature
end

function _vpc_x_values(
        dm::DataModel, ind::Individual, obs_rows::Vector{Int}, x_axis_feature)
    return _get_x_values(dm, ind, obs_rows, x_axis_feature)
end

function _bin_edges_quantile(x::Vector{Float64}, n_bins::Int)
    x_min = minimum(x)
    x_max = maximum(x)
    if x_min == x_max
        return [x_min, x_max]
    end
    qs = range(0.0, 1.0; length = n_bins + 1)
    edges = [quantile(x, q) for q in qs]
    edges[1] = x_min
    edges[end] = x_max
    return edges
end

function _assign_bins(x::Vector{Float64}, edges::Vector{Float64})
    bins = Vector{Int}(undef, length(x))
    for (i, xi) in enumerate(x)
        idx = searchsortedlast(edges, xi)
        idx = clamp(idx, 1, length(edges) - 1)
        bins[i] = idx
    end
    return bins
end

function _weighted_quantile(values::Vector{Float64}, weights::Vector{Float64}, p::Float64)
    idx = sortperm(values)
    v = values[idx]
    w = weights[idx]
    s = sum(w)
    s == 0 && return NaN
    cdf = cumsum(w) ./ s
    i = searchsortedfirst(cdf, p)
    i = clamp(i, 1, length(v))
    return v[i]
end

function _collect_observed_xy(ind::Individual,
        dm::DataModel,
        obs_rows::Vector{Int},
        obs_name::Symbol,
        x_axis_feature)
    x_raw = _vpc_x_values(dm, ind, obs_rows, x_axis_feature)
    y_raw = getfield(get_obs(get_series(ind)), obs_name)
    x_all = Float64[]
    x_obs = Float64[]
    y_obs = Float64[]
    for (xv, yv) in zip(x_raw, y_raw)
        xv === missing && continue
        xv isa Real || continue
        xf = Float64(xv)
        isfinite(xf) || continue
        push!(x_all, xf)
        yv === missing && continue
        yv isa Real || continue
        yf = Float64(yv)
        isfinite(yf) || continue
        push!(x_obs, xf)
        push!(y_obs, yf)
    end
    return x_all, x_obs, y_obs
end

function _kernel_quantiles(x::Vector{Float64},
        y::Vector{Float64},
        xgrid::Vector{Float64},
        bandwidth::Float64,
        percentiles::Vector{Float64})
    out = Dict{Float64, Vector{Float64}}((p => Vector{Float64}(undef, length(xgrid)))
    for p in percentiles)
    for (i, xg) in enumerate(xgrid)
        w = exp.(-0.5 .* ((x .- xg) ./ bandwidth) .^ 2)
        for p in percentiles
            out[p][i] = _weighted_quantile(y, w, p / 100)
        end
    end
    return out
end

function _resolve_n_bins(x::Vector{Float64}, n_bins::Union{Nothing, Int})
    if n_bins !== nothing
        n_bins >= 1 || error("n_bins must be >= 1.")
        n_unique = length(unique(x))
        if n_unique < 1
            return 1
        end
        n_bins > n_unique &&
            @warn "n_bins exceeds unique x values; reducing bins." requested=n_bins used=n_unique
        return min(n_bins, n_unique)
    end
    n_unique = length(unique(x))
    return max(1, min(10, n_unique))
end

function _extend_bin_series(
        x_centers::Vector{Float64}, y::Vector{Float64}, edges::Vector{Float64})
    length(x_centers) == length(y) || error("Bin series length mismatch.")
    x = [edges[1]; x_centers; edges[end]]
    y_ext = [y[1]; y; y[end]]
    return x, y_ext
end

function _re_level_reps(dm::DataModel, re::Symbol)
    reps = Dict{Any, Int}()
    for (i, ind) in enumerate(get_individuals(dm))
        g = getfield(get_re_groups(ind), re)
        if g isa AbstractVector
            for gv in g
                haskey(reps, gv) || (reps[gv] = i)
            end
        else
            haskey(reps, g) || (reps[g] = i)
        end
    end
    return reps
end

function _sample_random_effects_levels(dm::DataModel,
        θ::ComponentArray,
        constants_re::NamedTuple,
        rng::AbstractRNG)
    re_names = get_re_names(get_random(get_model(dm)))
    isempty(re_names) && return Dict{Symbol, Dict{Any, Any}}()
    fixed_maps = _normalize_constants_re(dm, constants_re)
    dists_builder = create_random_effect_distribution(get_random(get_model(dm)))
    model_funs = get_model_funs(get_model(dm))
    helpers = get_helper_funs(get_model(dm))
    values = get_re_values(get_re_group_info(dm))
    out = Dict{Symbol, Dict{Any, Any}}()
    for re in re_names
        reps = _re_level_reps(dm, re)
        fixed = haskey(fixed_maps, re) ? getfield(fixed_maps, re) : Dict{Any, Any}()
        level_vals = Dict{Any, Any}()
        for lvl in getfield(values, re)
            if haskey(fixed, lvl)
                level_vals[lvl] = fixed[lvl]
            else
                rep = reps[lvl]
                const_cov = get_const_cov(get_individuals(dm)[rep])
                dist = getproperty(dists_builder(θ, const_cov, model_funs, helpers), re)
                level_vals[lvl] = rand(rng, dist)
            end
        end
        out[re] = level_vals
    end
    return out
end

function _eta_vec_from_levels(dm::DataModel, level_vals::Dict{Symbol, Dict{Any, Any}})
    re_names = get_re_names(get_random(get_model(dm)))
    η_vec = Vector{ComponentArray}(undef, length(get_individuals(dm)))
    for (i, ind) in enumerate(get_individuals(dm))
        pairs = Pair{Symbol, Any}[]
        for re in re_names
            g = getfield(get_re_groups(ind), re)
            vals = level_vals[re]
            if g isa AbstractVector
                if length(g) == 1
                    push!(pairs, re => vals[g[1]])
                else
                    push!(pairs, re => [vals[gv] for gv in g])
                end
            else
                push!(pairs, re => vals[g])
            end
        end
        η_vec[i] = ComponentArray(NamedTuple(pairs))
    end
    return η_vec
end

function _simulate_obs(dm::DataModel,
        θ::ComponentArray,
        η_vec::Vector{ComponentArray},
        obs_name::Symbol,
        rng::AbstractRNG,
        x_axis_feature)
    sim_vals = Vector{Vector{Float64}}(undef, length(get_individuals(dm)))
    sim_x = Vector{Vector{Float64}}(undef, length(get_individuals(dm)))
    for (i, ind) in enumerate(get_individuals(dm))
        obs_rows = get_obs_rows(get_row_groups(dm))[i]
        x = _vpc_x_values(dm, ind, obs_rows, x_axis_feature)
        sim_x[i] = Float64.(x)
        η_ind = η_vec[i]
        rowwise_re = _needs_rowwise_random_effects(dm, i; obs_only = true)
        sol_accessors = nothing
        if get_de(get_model(dm)) !== nothing
            sol, compiled = _solve_dense_individual(dm, ind, θ, η_ind)
            sol_accessors = _sol_accessors_with_crossings(
                get_model(dm), sol, compiled, θ, η_ind, get_const_cov(ind))
        end
        vals = Vector{Float64}(undef, length(obs_rows))
        hmm_prev_state = 0
        for (j, row) in enumerate(obs_rows)
            vary = _varying_at(dm, ind, j, row)
            η_row = _row_random_effects_at(dm, i, j, η_ind, rowwise_re; obs_only = true)
            obs = sol_accessors === nothing ?
                  calculate_formulas_obs(
                get_model(dm), θ, η_row, get_const_cov(ind), vary) :
                  calculate_formulas_obs(
                get_model(dm), θ, η_row, get_const_cov(ind), vary, sol_accessors)
            dist = getproperty(obs, obs_name)
            if _is_hmm_dist(dist)
                state = hmm_prev_state == 0 ?
                        _sample_hmm_hidden_state(rng, dist) :
                        _sample_hmm_hidden_state(rng, dist, hmm_prev_state)
                hmm_prev_state = state
                vals[j] = _float_if_real(_hmm_emission_rand(rng, dist, state))
            else
                vals[j] = rand(rng, dist)
            end
        end
        sim_vals[i] = vals
    end
    return sim_x, sim_vals
end

function _representative_dist(dm::DataModel, obs_name::Symbol, x_axis_feature)
    θ = get_θ0_untransformed(get_fixed(get_model(dm)))
    η_vec = _default_random_effects_from_dm(dm, NamedTuple(), θ)
    ind = get_individuals(dm)[1]
    obs_rows = get_obs_rows(get_row_groups(dm))[1]
    η_ind = η_vec[1]
    rowwise_re = _needs_rowwise_random_effects(dm, 1; obs_only = true)
    sol_accessors = nothing
    if get_de(get_model(dm)) !== nothing
        sol, compiled = _solve_dense_individual(dm, ind, θ, η_ind)
        sol_accessors = _sol_accessors_with_crossings(
            get_model(dm), sol, compiled, θ, η_ind, get_const_cov(ind))
    end
    vary = _varying_at(dm, ind, 1, obs_rows[1])
    if get_de(get_model(dm)) === nothing && x_axis_feature !== nothing
        vary = merge(vary, (t = 0.0,))
    end
    η_row = _row_random_effects_at(dm, 1, 1, η_ind, rowwise_re; obs_only = true)
    obs = sol_accessors === nothing ?
          calculate_formulas_obs(get_model(dm), θ, η_row, get_const_cov(ind), vary) :
          calculate_formulas_obs(
        get_model(dm), θ, η_row, get_const_cov(ind), vary, sol_accessors)
    return getproperty(obs, obs_name)
end
