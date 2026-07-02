export plot_dv_pred
export plot_dv_ipred
export plot_wres_pred
export plot_shrinkage

using Distributions
using Random

function _get_dm(res, dm::Union{Nothing, DataModel})
    dm !== nothing && return dm
    dm = get_data_model(res)
    dm === nothing &&
        error("This fit result does not store a DataModel; pass dm=... explicitly.")
    return dm
end

function _get_observable(dm::DataModel, observable)
    obs = get_formulas_meta(dm.model.formulas.formulas).obs_names
    if observable === nothing
        length(obs) > 1 &&
            @warn "Multiple observables found; using the first." observable=obs[1]
        return obs[1]
    end
    observable in obs || error("Observable $(observable) not found. Available: $(obs).")
    return observable
end

function _time_values(dm::DataModel, ind::Individual, obs_rows::Vector{Int})
    vary = ind.series.vary
    if hasproperty(vary, dm.config.time_col)
        vals = getfield(vary, dm.config.time_col)
        vals isa AbstractVector && return vals
    end
    if hasproperty(vary, :t)
        vals = getfield(vary, :t)
        vals isa AbstractVector && return vals
    end
    return dm.df[obs_rows, dm.config.time_col]
end

function _get_x_values(
        dm::DataModel, ind::Individual, obs_rows::Vector{Int}, x_axis_feature)
    if dm.model.de.de !== nothing
        return _time_values(dm, ind, obs_rows)
    end
    if x_axis_feature === nothing
        return _time_values(dm, ind, obs_rows)
    end
    cov = dm.model.covariates.covariates
    if x_axis_feature == dm.config.time_col
        return _time_values(dm, ind, obs_rows)
    end
    x_axis_feature in cov.varying ||
        error("x_axis_feature must be a varying covariate. Got $(x_axis_feature).")
    v = getfield(ind.series.vary, x_axis_feature)
    if v isa AbstractVector
        return v
    elseif v isa NamedTuple
        if length(keys(v)) == 1
            return getfield(v, first(keys(v)))
        end
        error("x_axis_feature $(x_axis_feature) is a vector covariate; choose a single column.")
    end
    return v
end

function _dense_time_grid(ind::Individual; n::Int = 200)
    t0, t1 = ind.tspan
    base = collect(range(t0, t1; length = n))
    # Include all callback fire times (infusion starts, stops, bolus, resets) so that
    # short infusions whose stop time falls between uniform grid points are not missed.
    # Without this, a 1-day infusion in a 400-day tspan (grid step ~2 days) would be
    # invisible and V would appear to peak at t≈2 instead of t=1.
    if ind.callbacks !== nothing && !isempty(ind.callbacks.all_times)
        return sort!(unique!(vcat(base, ind.callbacks.all_times)))
    end
    return base
end

function _can_dense_plot(dm::DataModel)
    cov = dm.model.covariates.covariates
    return all(v -> v == dm.config.time_col, cov.varying)
end

function _stat_from_dist(dist, f)
    try
        return f(dist)
    catch
        if f === mode
            return Distributions.mode(dist)
        end
        rethrow()
    end
end

function _is_bernoulli(dist)
    return dist isa Bernoulli
end

function _is_discrete(dist)
    return dist isa DiscreteDistribution
end

function _dist_quantile_bounds(dists, coverage)
    qlo = (1 - coverage) / 2
    qhi = 1 - qlo
    lows = Float64[]
    highs = Float64[]
    for d in dists
        applicable(quantile, d, 0.5) || return nothing
        push!(lows, quantile(d, qlo))
        push!(highs, quantile(d, qhi))
    end
    return (minimum(lows), maximum(highs))
end

function _density_grid_continuous(dists, coverage, n_points; bounds = nothing)
    if bounds !== nothing
        y_min, y_max = bounds
    else
        b = _dist_quantile_bounds(dists, coverage)
        b === nothing && return nothing
        y_min, y_max = b
    end
    y = range(y_min, y_max; length = n_points)
    z = zeros(length(y), length(dists))
    for (j, d) in enumerate(dists)
        z[:, j] = pdf.(Ref(d), y)
    end
    return (y = y, z = z)
end

function _density_grid_discrete(dist, coverage)
    qlo = (1 - coverage) / 2
    qhi = 1 - qlo
    applicable(quantile, dist, 0.5) || return nothing
    lo = floor(Int, quantile(dist, qlo))
    hi = ceil(Int, quantile(dist, qhi))
    if lo > hi
        lo, hi = hi, lo
    end
    vals = collect(lo:hi)
    probs = pdf.(Ref(dist), vals)
    return (vals = vals, probs = probs)
end

function _pad_limits(lo, hi; frac = 0.05)
    lo == hi && return (lo - 1, hi + 1)
    pad = (hi - lo) * frac
    return (lo - pad, hi + pad)
end

function _merge_limits(lims, vals)
    vals === nothing && return lims
    iter = vals isa Number || vals === missing ? (vals,) : vals
    lo = Inf
    hi = -Inf
    has_finite = false
    for v in iter
        v === missing && continue
        v isa Real || continue
        x = float(v)
        isfinite(x) || continue
        has_finite = true
        lo = min(lo, x)
        hi = max(hi, x)
    end
    has_finite || return lims
    return lims === nothing ? (lo, hi) : (min(lims[1], lo), max(lims[2], hi))
end

function _float_if_real(x)
    return x isa Real ? float(x) : x
end

function _obs_multivariate_info(dm::DataModel, obs_name::Symbol)
    for ind in dm.individuals
        y = getfield(ind.series.obs, obs_name)
        for val in y
            val === missing && continue
            return val isa AbstractVector ? (true, length(val)) : (false, 1)
        end
    end
    return (false, 1)
end

function _marginal_label(obs_name::Symbol, idx::Int)
    return "$(obs_name)[$idx]"
end

function _marginal_colors(n::Int, style::PlotStyle)
    base = [style.color_secondary, style.color_primary, style.color_accent,
        style.color_dark, style.color_density, style.color_reference]
    return [base[mod1(i, length(base))] for i in 1:n]
end

function _collect_multivariate_series(
        x, y, n_marginals; marginal_idx::Union{Nothing, Int} = nothing)
    xs = [Vector{Any}() for _ in 1:n_marginals]
    ys = [Vector{Any}() for _ in 1:n_marginals]
    for (xi, yi) in zip(x, y)
        yi === missing && continue
        for m in 1:n_marginals
            val = yi[m]
            val === missing && continue
            push!(xs[m], xi)
            push!(ys[m], val)
        end
    end
    if marginal_idx === nothing
        return xs, ys
    end
    return xs[marginal_idx], ys[marginal_idx]
end

function _collect_scalar_series(x, y)
    xs = Vector{Any}()
    ys = Vector{Any}()
    for (xi, yi) in zip(x, y)
        (xi === missing || yi === missing) && continue
        push!(xs, xi)
        push!(ys, yi)
    end
    return xs, ys
end

"""
    plot_hidden_states(res::FitResult; dm, observable, individuals_idx, x_axis_feature,
                       shared_x_axis, shared_y_axis, ncols, style, kwargs_subplot,
                       kwargs_layout, save_path, plot_path, params, constants_re,
                       mcmc_draws, rng) -> Plots.Plot

    plot_hidden_states(dm::DataModel; observable, individuals_idx, x_axis_feature,
                       shared_x_axis, shared_y_axis, ncols, style, kwargs_subplot,
                       kwargs_layout, save_path, plot_path, params, constants_re,
                       rng) -> Plots.Plot

Plot the posterior hidden-state probabilities implied by a multivariate discrete-time
HMM observable. Each individual panel displays a stacked bar for each time point.

# Keyword Arguments
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
- `observable`: name of the multivariate outcome column.
- `individuals_idx`: indices or IDs of individuals to include, or `nothing` for all.
- `x_axis_feature::Union{Symbol, Nothing} = nothing`: covariate for the x-axis.
- `shared_x_axis::Bool = true`, `shared_y_axis::Bool = true`: share axis ranges.
- `ncols::Int = 3`: number of subplot columns.
- `figure_layout::Symbol = :single`: `:single` returns one combined figure with one subplot per individual, `:vector` produces a vector of figures (one per individual) while still arranging data by individuals.
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `kwargs_subplot`, `kwargs_layout`: extra keyword arguments for subplots and layout.
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
- `params::NamedTuple = NamedTuple()`, `constants_re::NamedTuple = NamedTuple()`: overrides.
- `mcmc_draws::Int = 1`: draws for estimating random effects (ignored for most fits).
- `rng::AbstractRNG = Random.default_rng()`: random-number generator.
"""

function _resolve_emission_row(dm::DataModel,
        obs_rows::Vector{Int},
        time_idx::Union{Nothing, Int},
        time_point,
        time_col::Symbol)
    length(obs_rows) == 0 && error("No observation rows found.")
    if time_idx !== nothing
        (1 <= time_idx <= length(obs_rows)) ||
            error("time_idx must be between 1 and $(length(obs_rows)).")
        return obs_rows[time_idx]
    end
    if time_point !== nothing
        vals = dm.df[obs_rows, time_col]
        numeric = [ismissing(v) ? missing : float(v) for v in vals]
        has_missing = any(ismissing, numeric)
        has_missing &&
            error("Time column $(time_col) contains missing values for individual observations.")
        distances = abs.(numeric .- float(time_point))
        idx = argmin(distances)
        return obs_rows[idx]
    end
    return obs_rows[1]
end

function _state_emission_marginals(emission)
    if emission isa Tuple
        return collect(emission)
    elseif emission isa Distribution{Multivariate}
        emission isa MvNormal ||
            error("Emission distributions must be MvNormal when joint.")
        μ = emission.μ
        Σ = Matrix(emission.Σ)
        marginals = Vector{Distribution}(undef, length(μ))
        for i in 1:length(μ)
            marginals[i] = Normal(μ[i], sqrt(max(Σ[i, i], zero(Σ[i, i]))))
        end
        return marginals
    else
        error("Unsupported emission element type: $(typeof(emission)).")
    end
end

@inline _as_fit_result_for_plotting(res::FitResult) = res
@inline _as_fit_result_for_plotting(res::MultistartFitResult) = get_multistart_best(res)

function _as_fit_result_for_plotting(res)
    error("plot_fits_comparison expects FitResult or MultistartFitResult entries. Got $(typeof(res)).")
end

function _same_data_model_for_fits(dm1::DataModel, dm2::DataModel)
    dm1 === dm2 && return true

    cfg1 = dm1.config
    cfg2 = dm2.config
    cfg1.primary_id == cfg2.primary_id || return false
    cfg1.time_col == cfg2.time_col || return false
    cfg1.evid_col == cfg2.evid_col || return false
    cfg1.amt_col == cfg2.amt_col || return false
    cfg1.rate_col == cfg2.rate_col || return false
    cfg1.cmt_col == cfg2.cmt_col || return false
    cfg1.obs_cols == cfg2.obs_cols || return false
    cfg1.serialization == cfg2.serialization || return false
    cfg1.saveat_mode == cfg2.saveat_mode || return false

    length(dm1.individuals) == length(dm2.individuals) || return false
    dm1.row_groups.obs_rows == dm2.row_groups.obs_rows || return false
    get_formulas_meta(dm1.model.formulas.formulas).obs_names ==
    get_formulas_meta(dm2.model.formulas.formulas).obs_names || return false

    propertynames(dm1.df) == propertynames(dm2.df) || return false
    return isequal(dm1.df, dm2.df)
end

function _validate_same_data_model_for_comparison(dms::AbstractVector{<:DataModel})
    dm_ref = dms[1]
    for j in 2:length(dms)
        _same_data_model_for_fits(dm_ref, dms[j]) ||
            error("All fit results passed to plot_fits_comparison must use the same DataModel.")
    end
    return dm_ref
end

function _comparison_line_colors(n::Int, style::PlotStyle)
    base = [
        style.color_secondary,
        style.color_primary,
        style.color_accent,
        COLOR_DARK,
        COLOR_ERROR,
        "#56B4E9",
        "#CC79A7",
        "#F0E442",
        "#009E73",
        "#E69F00"
    ]
    return [base[mod1(i, length(base))] for i in 1:n]
end

function _comparison_line_style(label::String, style::PlotStyle)
    return get(style.comparison_line_styles, label, style.comparison_default_linestyle)
end

function _fit_curve_from_cache(dm::DataModel,
        cache::PlotCache,
        ind_idx::Int,
        obs_name::Symbol,
        x_axis_feature::Union{Nothing, Symbol},
        plot_func)
    ind = dm.individuals[ind_idx]
    obs_rows = dm.row_groups.obs_rows[ind_idx]
    use_dense = dm.model.de.de !== nothing &&
                _can_dense_plot(dm) &&
                (x_axis_feature === nothing || x_axis_feature == dm.config.time_col)
    x_obs = _get_x_values(dm, ind, obs_rows, x_axis_feature)
    x_fit = use_dense ? _dense_time_grid(ind) : x_obs

    θ = cache.params
    η_ind = cache.random_effects[ind_idx]
    sol_accessors = nothing
    if dm.model.de.de !== nothing
        sol = cache.sols[ind_idx]
        compiled = get_de_compiler(dm.model.de.de)((;
            fixed_effects = θ,
            random_effects = η_ind,
            constant_covariates = ind.const_cov,
            varying_covariates = merge((t = ind.series.vary.t[1],), ind.series.dyn),
            helpers = get_helper_funs(dm.model),
            model_funs = get_model_funs(dm.model),
            preDE = calculate_prede(dm.model, θ, η_ind, ind.const_cov)
        ))
        sol_accessors = get_de_accessors_builder(dm.model.de.de)(sol, compiled)
    end

    preds = Vector{Float64}(undef, length(x_fit))
    rowwise_re = _needs_rowwise_random_effects(dm, ind_idx; obs_only = true)
    if use_dense
        for (j, t) in enumerate(x_fit)
            vary = (t = t,)
            obs = calculate_formulas_obs(
                dm.model, θ, η_ind, ind.const_cov, vary, sol_accessors)
            preds[j] = _stat_from_dist(getproperty(obs, obs_name), plot_func)
        end
    else
        y_obs_series_cmp = getfield(ind.series.obs, obs_name)
        hmm_priors_cmp = Dict{Symbol, Any}()
        for (j, row) in enumerate(obs_rows)
            vary = _varying_at(dm, ind, j, row)
            η_row = _row_random_effects_at(
                dm, ind_idx, j, η_ind, rowwise_re; obs_only = true)
            obs = sol_accessors === nothing ?
                  calculate_formulas_obs(dm.model, θ, η_row, ind.const_cov, vary) :
                  calculate_formulas_obs(
                dm.model, θ, η_row, ind.const_cov, vary, sol_accessors)
            dist = _apply_hmm_filter!(
                hmm_priors_cmp, obs_name, getproperty(obs, obs_name), y_obs_series_cmp[j])
            preds[j] = _stat_from_dist(dist, plot_func)
        end
    end
    return (x_obs = x_obs, x_fit = x_fit, preds = preds)
end

function _pred_re_per_individual(dm::DataModel, θ::ComponentArray)
    re_names = get_re_names(dm.model.random.random)
    isempty(re_names) && return fill(ComponentArray(NamedTuple()), length(dm.individuals))

    dists_builder = get_create_random_effect_distribution(dm.model.random.random)
    model_funs = get_model_funs(dm.model)
    helpers = get_helper_funs(dm.model)

    η_vec = Vector{ComponentArray}(undef, length(dm.individuals))
    for (i, ind) in enumerate(dm.individuals)
        nt_pairs = Pair{Symbol, Any}[]
        for re in re_names
            dist = getproperty(dists_builder(θ, ind.const_cov, model_funs, helpers), re)
            if dist isa Distributions.UnivariateDistribution
                v = try
                    Float64(Distributions.mean(dist))
                catch
                    0.0
                end
                push!(nt_pairs, re => v)
            else
                v = try
                    Vector{Float64}(Distributions.mean(dist))
                catch
                    zeros(Float64, length(dist))
                end
                push!(nt_pairs, re => v)
            end
        end
        η_vec[i] = ComponentArray(NamedTuple(nt_pairs))
    end
    return η_vec
end

# Collect (dv, pred, sigma_pred) across all non-missing observations using η_pop.
function _collect_pred_series(dm::DataModel, obs_name::Symbol,
        θ::ComponentArray, η_pop::Vector)
    dv_all = Float64[]
    pred_all = Float64[]
    sigma_all = Float64[]

    for i in eachindex(dm.individuals)
        ind = dm.individuals[i]
        obs_rows = dm.row_groups.obs_rows[i]
        y_series = getfield(ind.series.obs, obs_name)
        η_ind = η_pop[i]
        rowwise_re = _needs_rowwise_random_effects(dm, i; obs_only = true)

        sol_accessors = nothing
        if dm.model.de.de !== nothing
            sol, compiled = _solve_dense_individual(dm, ind, θ, η_ind)
            sol_accessors = get_de_accessors_builder(dm.model.de.de)(sol, compiled)
        end

        for (j, row) in enumerate(obs_rows)
            yj = y_series[j]
            (yj === missing || !(yj isa Real)) && continue
            y = Float64(yj)
            isfinite(y) || continue

            vary = _varying_at(dm, ind, j, row)
            η_row = _row_random_effects_at(dm, i, j, η_ind, rowwise_re; obs_only = true)
            obs_nt = sol_accessors === nothing ?
                     calculate_formulas_obs(dm.model, θ, η_row, ind.const_cov, vary) :
                     calculate_formulas_obs(
                dm.model, θ, η_row, ind.const_cov, vary, sol_accessors)
            dist = getproperty(obs_nt, obs_name)

            pred_val = try
                Float64(mean(dist))
            catch
                NaN
            end
            isfinite(pred_val) || continue
            sigma_val = try
                Float64(std(dist))
            catch
                NaN
            end
            (isfinite(sigma_val) && sigma_val > 0) || continue

            push!(dv_all, y)
            push!(pred_all, pred_val)
            push!(sigma_all, sigma_val)
        end
    end
    return dv_all, pred_all, sigma_all
end

# Collect (dv, ipred) across all non-missing observations using EBEs.
function _collect_ipred_series(dm::DataModel, obs_name::Symbol,
        θ::ComponentArray, η_ebe::Vector)
    dv_all = Float64[]
    ipred_all = Float64[]

    for i in eachindex(dm.individuals)
        ind = dm.individuals[i]
        obs_rows = dm.row_groups.obs_rows[i]
        y_series = getfield(ind.series.obs, obs_name)
        η_ind = η_ebe[i]
        rowwise_re = _needs_rowwise_random_effects(dm, i; obs_only = true)

        sol_accessors = nothing
        if dm.model.de.de !== nothing
            sol, compiled = _solve_dense_individual(dm, ind, θ, η_ind)
            sol_accessors = get_de_accessors_builder(dm.model.de.de)(sol, compiled)
        end

        for (j, row) in enumerate(obs_rows)
            yj = y_series[j]
            (yj === missing || !(yj isa Real)) && continue
            y = Float64(yj)
            isfinite(y) || continue

            vary = _varying_at(dm, ind, j, row)
            η_row = _row_random_effects_at(dm, i, j, η_ind, rowwise_re; obs_only = true)
            obs_nt = sol_accessors === nothing ?
                     calculate_formulas_obs(dm.model, θ, η_row, ind.const_cov, vary) :
                     calculate_formulas_obs(
                dm.model, θ, η_row, ind.const_cov, vary, sol_accessors)
            dist = getproperty(obs_nt, obs_name)

            ipred_val = try
                Float64(mean(dist))
            catch
                NaN
            end
            isfinite(ipred_val) || continue

            push!(dv_all, y)
            push!(ipred_all, ipred_val)
        end
    end
    return dv_all, ipred_all
end
