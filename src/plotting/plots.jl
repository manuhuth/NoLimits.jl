using Distributions
using Plots

function _get_dm(res, dm::Union{Nothing, DataModel})
    dm !== nothing && return dm
    dm = get_data_model(res)
    dm === nothing && error("This fit result does not store a DataModel; pass dm=... explicitly.")
    return dm
end

function _get_observable(dm::DataModel, observable)
    obs = get_formulas_meta(dm.model.formulas.formulas).obs_names
    if observable === nothing
        length(obs) > 1 && @warn "Multiple observables found; using the first." observable=obs[1]
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

function _get_x_values(dm::DataModel, ind::Individual, obs_rows::Vector{Int}, x_axis_feature)
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
    x_axis_feature in cov.varying || error("x_axis_feature must be a varying covariate. Got $(x_axis_feature).")
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

function _dense_time_grid(ind::Individual; n::Int=200)
    t0, t1 = ind.tspan
    return collect(range(t0, t1; length=n))
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

function _density_grid_continuous(dists, coverage, n_points)
    bounds = _dist_quantile_bounds(dists, coverage)
    bounds === nothing && return nothing
    y_min, y_max = bounds
    y = range(y_min, y_max; length=n_points)
    z = zeros(length(y), length(dists))
    for (j, d) in enumerate(dists)
        z[:, j] = pdf.(Ref(d), y)
    end
    return (y=y, z=z)
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
    return (vals=vals, probs=probs)
end

function _pad_limits(lo, hi; frac=0.05)
    lo == hi && return (lo - 1, hi + 1)
    pad = (hi - lo) * frac
    return (lo - pad, hi + pad)
end

function _plot_data_dm(dm::DataModel;
                       x_axis_feature::Union{Symbol, Nothing}=nothing,
                       individuals_idx=nothing,
                       shared_x_axis::Bool=true,
                       shared_y_axis::Bool=true,
                       ncols::Int=DEFAULT_PLOT_COLS,
                       style::PlotStyle=PlotStyle(),
                       kwargs_subplot=NamedTuple(),
                       kwargs_layout=NamedTuple(),
                       save_path::Union{Nothing, String}=nothing)
    obs_name = _get_observable(dm, nothing)
    inds = individuals_idx === nothing ? collect(eachindex(dm.individuals)) : collect(individuals_idx)

    plots = Vector{Any}(undef, length(inds))
    xlims = nothing
    ylims = nothing
    for (k, i) in enumerate(inds)
        ind = dm.individuals[i]
        obs_rows = dm.row_groups.obs_rows[i]
        x = _get_x_values(dm, ind, obs_rows, x_axis_feature)
        y = getfield(ind.series.obs, obs_name)
        title_id = string(dm.config.primary_id, ": ", dm.df[obs_rows[1], dm.config.primary_id])
        p = create_styled_plot(title=title_id,
                               xlabel=x_axis_feature === nothing ? "Time" : _axis_label(x_axis_feature),
                               ylabel=_axis_label(obs_name),
                               style=style,
                               kwargs_subplot...)
        create_styled_scatter!(p, x, y; label="", style=style)
        plots[k] = p
        xlims = xlims === nothing ? (minimum(x), maximum(x)) : (min(xlims[1], minimum(x)), max(xlims[2], maximum(x)))
        ylims = ylims === nothing ? (minimum(y), maximum(y)) : (min(ylims[1], minimum(y)), max(ylims[2], maximum(y)))
    end

    if shared_x_axis || shared_y_axis
        xlim_use = shared_x_axis ? _pad_limits(xlims[1], xlims[2]) : nothing
        ylim_use = shared_y_axis ? _pad_limits(ylims[1], ylims[2]) : nothing
        _apply_shared_axes!(plots, xlim_use, ylim_use)
    end
    p = combine_plots(plots; ncols=ncols, kwargs_layout...)
    return _save_plot!(p, save_path)
end

"""
    plot_data(res::FitResult; dm, x_axis_feature, individuals_idx, shared_x_axis,
              shared_y_axis, ncols, style, kwargs_subplot, kwargs_layout, save_path)
              -> Plots.Plot

    plot_data(dm::DataModel; x_axis_feature, individuals_idx, shared_x_axis,
              shared_y_axis, ncols, style, kwargs_subplot, kwargs_layout, save_path)
              -> Plots.Plot

Plot raw observed data for each individual as a multi-panel figure.

# Keyword Arguments
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
- `x_axis_feature::Union{Symbol, Nothing} = nothing`: covariate to use as the x-axis;
  defaults to the time column.
- `individuals_idx`: indices or IDs of individuals to include, or `nothing` for all.
- `shared_x_axis::Bool = true`: share the x-axis range across panels.
- `shared_y_axis::Bool = true`: share the y-axis range across panels.
- `ncols::Int = 3`: number of subplot columns.
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `kwargs_subplot`, `kwargs_layout`: extra keyword arguments for subplots and layout.
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
"""
function plot_data(res::FitResult;
                   dm::Union{Nothing, DataModel}=nothing,
                   x_axis_feature::Union{Symbol, Nothing}=nothing,
                   individuals_idx=nothing,
                   shared_x_axis::Bool=true,
                   shared_y_axis::Bool=true,
                   ncols::Int=DEFAULT_PLOT_COLS,
                   style::PlotStyle=PlotStyle(),
                   kwargs_subplot=NamedTuple(),
                   kwargs_layout=NamedTuple(),
                   save_path::Union{Nothing, String}=nothing,
                   plot_path::Union{Nothing, String}=nothing)
    dm = _get_dm(res, dm)
    save_path = _resolve_plot_path(save_path, plot_path)
    return _plot_data_dm(dm;
                         x_axis_feature=x_axis_feature,
                         individuals_idx=individuals_idx,
                         shared_x_axis=shared_x_axis,
                         shared_y_axis=shared_y_axis,
                         ncols=ncols,
                         style=style,
                         kwargs_subplot=kwargs_subplot,
                         kwargs_layout=kwargs_layout,
                         save_path=save_path)
end

function plot_data(dm::DataModel;
                   x_axis_feature::Union{Symbol, Nothing}=nothing,
                   individuals_idx=nothing,
                   shared_x_axis::Bool=true,
                   shared_y_axis::Bool=true,
                   ncols::Int=DEFAULT_PLOT_COLS,
                   style::PlotStyle=PlotStyle(),
                   kwargs_subplot=NamedTuple(),
                   kwargs_layout=NamedTuple(),
                   save_path::Union{Nothing, String}=nothing,
                   plot_path::Union{Nothing, String}=nothing)
    save_path = _resolve_plot_path(save_path, plot_path)
    return _plot_data_dm(dm;
                         x_axis_feature=x_axis_feature,
                         individuals_idx=individuals_idx,
                         shared_x_axis=shared_x_axis,
                         shared_y_axis=shared_y_axis,
                         ncols=ncols,
                         style=style,
                         kwargs_subplot=kwargs_subplot,
                         kwargs_layout=kwargs_layout,
                         save_path=save_path)
end

"""
    plot_fits(res::FitResult; dm, plot_density, plot_func, plot_data_points, observable,
              individuals_idx, x_axis_feature, shared_x_axis, shared_y_axis, ncols,
              style, kwargs_subplot, kwargs_layout, save_path, cache, params,
              constants_re, cache_obs_dists, plot_mcmc_quantiles, mcmc_quantiles,
              mcmc_quantiles_alpha, mcmc_draws, mcmc_warmup, rng) -> Plots.Plot

    plot_fits(dm::DataModel; params, constants_re, observable, individuals_idx,
              x_axis_feature, shared_x_axis, shared_y_axis, ncols, plot_data_points,
              style, kwargs_subplot, kwargs_layout, save_path, cache) -> Plots.Plot

Plot model predictions against observed data for each individual as a multi-panel figure.

# Keyword Arguments
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
- `plot_density::Bool = false`: overlay the predictive distribution density.
- `plot_func = mean`: function applied to the predictive distribution to obtain the
  prediction line (e.g. `mean`, `median`).
- `plot_data_points::Bool = true`: overlay the observed data points.
- `observable`: name of the outcome variable to plot, or `nothing` to use the first.
- `individuals_idx`: indices or IDs of individuals to include, or `nothing` for all.
- `x_axis_feature::Union{Nothing, Symbol} = nothing`: covariate for the x-axis.
- `shared_x_axis::Bool = true`, `shared_y_axis::Bool = true`: share axis ranges.
- `ncols::Int = 3`: number of subplot columns.
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `kwargs_subplot`, `kwargs_layout`: extra keyword arguments for subplots and layout.
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
- `cache::Union{Nothing, PlotCache} = nothing`: pre-computed plot cache.
- `params::NamedTuple = NamedTuple()`: fixed-effect overrides.
- `constants_re::NamedTuple = NamedTuple()`: random-effect constants.
- `cache_obs_dists::Bool = false`: pre-compute observation distributions when building cache.
- `plot_mcmc_quantiles::Bool = false`: plot posterior predictive quantile bands (MCMC).
- `mcmc_quantiles::Vector = [5, 95]`: quantile percentages for posterior bands.
- `mcmc_quantiles_alpha::Float64 = 0.8`: opacity of the quantile band.
- `mcmc_draws::Int = 1000`: number of MCMC draws for posterior predictive plotting.
- `mcmc_warmup::Union{Nothing, Int} = nothing`: warm-up count override for MCMC.
- `rng::AbstractRNG = Random.default_rng()`: random-number generator.
"""
function plot_fits(res::FitResult;
                   dm::Union{Nothing, DataModel}=nothing,
                   plot_density::Bool=false,
                   plot_func=mean,
                   plot_data_points::Bool=true,
                   observable=nothing,
                   individuals_idx=nothing,
                   x_axis_feature::Union{Nothing, Symbol}=nothing,
                   shared_x_axis::Bool=true,
                   shared_y_axis::Bool=true,
                   ncols::Int=DEFAULT_PLOT_COLS,
                   style::PlotStyle=PlotStyle(),
                   kwargs_subplot=NamedTuple(),
                   kwargs_layout=NamedTuple(),
                   save_path::Union{Nothing, String}=nothing,
                   plot_path::Union{Nothing, String}=nothing,
                   cache::Union{Nothing, PlotCache}=nothing,
                   params::NamedTuple=NamedTuple(),
                   constants_re::NamedTuple=NamedTuple(),
                   cache_obs_dists::Bool=false,
                   plot_mcmc_quantiles::Bool=false,
                   mcmc_quantiles::Vector{<:Real}=[5, 95],
                   mcmc_quantiles_alpha::Float64=0.8,
                   mcmc_draws::Int=1000,
                   mcmc_warmup::Union{Nothing, Int}=nothing,
                   rng::AbstractRNG=Random.default_rng())
    dm = _get_dm(res, dm)
    save_path = _resolve_plot_path(save_path, plot_path)
    constants_re_use = _res_constants_re(res, constants_re)
    obs_name = _get_observable(dm, observable)
    inds = individuals_idx === nothing ? collect(eachindex(dm.individuals)) : collect(individuals_idx)

    if cache === nothing
        cache = build_plot_cache(res; dm=dm, params=params, constants_re=constants_re_use,
                                 cache_obs_dists=cache_obs_dists, mcmc_draws=mcmc_draws,
                                 mcmc_warmup=mcmc_warmup, rng=rng)
    end

    plots = Vector{Any}(undef, length(inds))
    xlims = nothing
    ylims = nothing

    is_mcmc = _is_posterior_draw_fit(res)
    θ_draws = nothing
    η_draws = nothing
    if is_mcmc
        res = _with_posterior_warmup(res, mcmc_warmup)
        θ_draws, η_draws, _ = _posterior_drawn_params(res, dm, constants_re_use, params, mcmc_draws, rng)
    end

    if plot_density && plot_mcmc_quantiles
        @warn "plot_mcmc_quantiles ignored because plot_density=true."
        plot_mcmc_quantiles = false
    end
    if plot_mcmc_quantiles
        mcmc_quantiles = sort(Float64.(collect(mcmc_quantiles)))
        (length(mcmc_quantiles) >= 2 && all(0 .<= mcmc_quantiles .<= 100)) || error("mcmc_quantiles must be in [0,100] with length >= 2.")
    end

    for (k, i) in enumerate(inds)
        ind = dm.individuals[i]
        obs_rows = dm.row_groups.obs_rows[i]
        use_dense = dm.model.de.de !== nothing &&
                    _can_dense_plot(dm) &&
                    (x_axis_feature === nothing || x_axis_feature == dm.config.time_col)
        x_obs = _get_x_values(dm, ind, obs_rows, x_axis_feature)
        x_fit = use_dense ? _dense_time_grid(ind) : x_obs
        x_density = use_dense ? x_fit : x_obs
        y_obs = getfield(ind.series.obs, obs_name)
        title_id = string(dm.config.primary_id, ": ", dm.df[obs_rows[1], dm.config.primary_id])
        p = create_styled_plot(title=title_id,
                               xlabel=x_axis_feature === nothing ? "Time" : _axis_label(x_axis_feature),
                               ylabel=_axis_label(obs_name),
                               style=style,
                               kwargs_subplot...)
        if plot_data_points
            create_styled_scatter!(p, x_obs, y_obs; label="data", color=style.color_primary, style=style)
        end

        if is_mcmc
            n_draws = length(θ_draws)
            preds = zeros(Float64, n_draws, length(use_dense ? x_fit : obs_rows))
            dists_for_density = plot_density ? Vector{Vector{Distribution}}(undef, n_draws) : nothing
            for d in 1:n_draws
                θ = θ_draws[d]
                η_ind = η_draws[d][i]
                sol_accessors = nothing
                compiled = nothing
                if dm.model.de.de !== nothing
                    sol, compiled = _solve_dense_individual(dm, ind, θ, η_ind)
                    sol_accessors = get_de_accessors_builder(dm.model.de.de)(sol, compiled)
                end
                if use_dense
                    dists = plot_density ? Vector{Distribution}(undef, length(x_fit)) : Distribution[]
                    for (j, t) in enumerate(x_fit)
                        vary = (t = t,)
                        obs = calculate_formulas_obs(dm.model, θ, η_ind, ind.const_cov, vary, sol_accessors)
                        dist = getproperty(obs, obs_name)
                        preds[d, j] = _stat_from_dist(dist, plot_func)
                        if plot_density
                            dists[j] = dist
                        end
                    end
                else
                    dists = Vector{Distribution}(undef, length(obs_rows))
                    for (j, row) in enumerate(obs_rows)
                        vary = _varying_at_plot(dm, ind, j, row)
                        obs = sol_accessors === nothing ?
                              calculate_formulas_obs(dm.model, θ, η_ind, ind.const_cov, vary) :
                              calculate_formulas_obs(dm.model, θ, η_ind, ind.const_cov, vary, sol_accessors)
                        dist = getproperty(obs, obs_name)
                        dists[j] = dist
                        preds[d, j] = _stat_from_dist(dist, plot_func)
                    end
                end
                if plot_density
                    dists_for_density[d] = dists
                end
            end
            mean_curve = vec(mean(preds, dims=1))
            create_styled_line!(p, x_fit, mean_curve; label="fit", color=style.color_secondary, style=style)
            ylims = ylims === nothing ? (minimum(mean_curve), maximum(mean_curve)) :
                    (min(ylims[1], minimum(mean_curve)), max(ylims[2], maximum(mean_curve)))

            if plot_mcmc_quantiles
                for q in mcmc_quantiles
                    qvals = mapslices(x -> quantile(vec(x), q / 100), preds; dims=1)
                    qvals = vec(qvals)
                    plot!(p, x_fit, qvals; color=style.color_secondary, alpha=mcmc_quantiles_alpha,
                          linestyle=:dash, label="$(q)%")
                    ylims = (min(ylims[1], minimum(qvals)), max(ylims[2], maximum(qvals)))
                end
            end

            if plot_density
                if _is_bernoulli(dists_for_density[1][1])
                    # Skip Bernoulli density overlay; the fit line already represents p(y=1).
                elseif _is_discrete(dists_for_density[1][1])
                    for j in eachindex(x_density)
                        dist0 = dists_for_density[1][j]
                        grid = _density_grid_discrete(dist0, 0.995)
                        grid === nothing && continue
                        probs = zeros(Float64, length(grid.vals))
                        for d in 1:n_draws
                            probs .+= pdf.(Ref(dists_for_density[d][j]), grid.vals)
                        end
                        probs ./= n_draws
                        scatter!(p, fill(x_density[j], length(grid.vals)), grid.vals;
                                 marker_z=probs, color=:viridis, marker=:x,
                                 markersize=style.marker_size_pmf, markerstrokewidth=style.marker_stroke_width_pmf,
                                 label="")
                    end
                else
                    dists = dists_for_density[1]
                    grid = _density_grid_continuous(dists, 0.995, 100)
                    if grid !== nothing
                        z = zeros(length(grid.y), length(x_density))
                        for d in 1:n_draws
                            for j in eachindex(x_density)
                                z[:, j] .+= pdf.(Ref(dists_for_density[d][j]), grid.y)
                            end
                        end
                        z ./= n_draws
                        heatmap!(p, x_density, grid.y, z; color=:viridis, alpha=0.5, label="")
                    end
                end
            end
        else
            θ = cache.params
            η_ind = cache.random_effects[i]
            sol_accessors = nothing
            if dm.model.de.de !== nothing
                sol = cache.sols[i]
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

            dists = Vector{Distribution}(undef, length(use_dense ? x_fit : obs_rows))
            preds = Vector{Float64}(undef, length(obs_rows))
            if use_dense
                preds_dense = Vector{Float64}(undef, length(x_fit))
                for (j, t) in enumerate(x_fit)
                    vary = (t = t,)
                    obs = calculate_formulas_obs(dm.model, θ, η_ind, ind.const_cov, vary, sol_accessors)
                    dist = getproperty(obs, obs_name)
                    preds_dense[j] = _stat_from_dist(dist, plot_func)
                    if plot_density
                        dists[j] = dist
                    end
                end
                create_styled_line!(p, x_fit, preds_dense; label="fit", color=style.color_secondary, style=style)
                ylims = ylims === nothing ? (minimum(preds_dense), maximum(preds_dense)) :
                        (min(ylims[1], minimum(preds_dense)), max(ylims[2], maximum(preds_dense)))
            else
                for (j, row) in enumerate(obs_rows)
                    vary = _varying_at_plot(dm, ind, j, row)
                    obs = sol_accessors === nothing ?
                          calculate_formulas_obs(dm.model, θ, η_ind, ind.const_cov, vary) :
                          calculate_formulas_obs(dm.model, θ, η_ind, ind.const_cov, vary, sol_accessors)
                    dist = getproperty(obs, obs_name)
                    dists[j] = dist
                    preds[j] = _stat_from_dist(dist, plot_func)
                end
                create_styled_line!(p, x_fit, preds; label="fit", color=style.color_secondary, style=style)
                ylims = ylims === nothing ? (minimum(preds), maximum(preds)) :
                        (min(ylims[1], minimum(preds)), max(ylims[2], maximum(preds)))
            end

            if plot_density
                if _is_bernoulli(dists[1])
                    # Skip Bernoulli density overlay; the fit line already represents p(y=1).
                elseif _is_discrete(dists[1])
                    for j in eachindex(dists)
                        grid = _density_grid_discrete(dists[j], 0.995)
                        grid === nothing && continue
                        scatter!(p, fill(x_density[j], length(grid.vals)), grid.vals;
                                 marker_z=grid.probs, color=:viridis, marker=:x,
                                 markersize=style.marker_size_pmf, markerstrokewidth=style.marker_stroke_width_pmf,
                                 label="")
                    end
                else
                    grid = _density_grid_continuous(dists, 0.995, 100)
                    if grid !== nothing
                        heatmap!(p, x_density, grid.y, grid.z; color=:viridis, alpha=0.5, label="")
                    end
                end
            end
        end

        plots[k] = p
        xlims = xlims === nothing ? (minimum(x_fit), maximum(x_fit)) : (min(xlims[1], minimum(x_fit)), max(xlims[2], maximum(x_fit)))
        ylims = ylims === nothing ? (minimum(y_obs), maximum(y_obs)) :
                (min(ylims[1], minimum(y_obs)), max(ylims[2], maximum(y_obs)))
    end

    if shared_x_axis || shared_y_axis
        xlim_use = shared_x_axis ? _pad_limits(xlims[1], xlims[2]) : nothing
        ylim_use = shared_y_axis ? _pad_limits(ylims[1], ylims[2]) : nothing
        _apply_shared_axes!(plots, xlim_use, ylim_use)
    end
    p = combine_plots(plots; ncols=ncols, kwargs_layout...)
    return _save_plot!(p, save_path)
end

function plot_fits(dm::DataModel;
                   plot_density::Bool=false,
                   plot_func=mean,
                   plot_data_points::Bool=true,
                   observable=nothing,
                   individuals_idx=nothing,
                   x_axis_feature::Union{Nothing, Symbol}=nothing,
                   shared_x_axis::Bool=true,
                   shared_y_axis::Bool=true,
                   ncols::Int=DEFAULT_PLOT_COLS,
                   style::PlotStyle=PlotStyle(),
                   kwargs_subplot=NamedTuple(),
                   kwargs_layout=NamedTuple(),
                   save_path::Union{Nothing, String}=nothing,
                   plot_path::Union{Nothing, String}=nothing,
                   params::NamedTuple=NamedTuple(),
                   constants_re::NamedTuple=NamedTuple(),
                   cache_obs_dists::Bool=false,
                   mcmc_draws::Int=1000,
                   mcmc_warmup::Union{Nothing, Int}=nothing,
                   rng::AbstractRNG=Random.default_rng())
    save_path = _resolve_plot_path(save_path, plot_path)
    cache = build_plot_cache(dm; params=params, constants_re=constants_re, cache_obs_dists=cache_obs_dists, rng=rng)
    res = FitResult(MLE(), MLEResult(NamedTuple(), 0.0, 0, NamedTuple(), NamedTuple()),
                    FitSummary(0.0, true, FitParameters(ComponentArray(), ComponentArray()), NamedTuple()),
                    FitDiagnostics((;), (;), (;), (;)), dm, (), NamedTuple())
    return plot_fits(res;
                     dm=dm,
                     plot_density=plot_density,
                     plot_func=plot_func,
                     plot_data_points=plot_data_points,
                     observable=observable,
                     individuals_idx=individuals_idx,
                     x_axis_feature=x_axis_feature,
                     shared_x_axis=shared_x_axis,
                     shared_y_axis=shared_y_axis,
                     ncols=ncols,
                     style=style,
                     kwargs_subplot=kwargs_subplot,
                     kwargs_layout=kwargs_layout,
                     save_path=save_path,
                     cache=cache,
                     params=params,
                     constants_re=constants_re,
                     cache_obs_dists=cache_obs_dists,
                     mcmc_draws=mcmc_draws,
                     mcmc_warmup=mcmc_warmup,
                     rng=rng)
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
    get_formulas_meta(dm1.model.formulas.formulas).obs_names == get_formulas_meta(dm2.model.formulas.formulas).obs_names || return false

    propertynames(dm1.df) == propertynames(dm2.df) || return false
    return isequal(dm1.df, dm2.df)
end

function _validate_same_data_model_for_comparison(dms::AbstractVector{<:DataModel})
    dm_ref = dms[1]
    for j in 2:length(dms)
        _same_data_model_for_fits(dm_ref, dms[j]) || error("All fit results passed to plot_fits_comparison must use the same DataModel.")
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
        "#E69F00",
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
    if use_dense
        for (j, t) in enumerate(x_fit)
            vary = (t = t,)
            obs = calculate_formulas_obs(dm.model, θ, η_ind, ind.const_cov, vary, sol_accessors)
            preds[j] = _stat_from_dist(getproperty(obs, obs_name), plot_func)
        end
    else
        for (j, row) in enumerate(obs_rows)
            vary = _varying_at_plot(dm, ind, j, row)
            obs = sol_accessors === nothing ?
                  calculate_formulas_obs(dm.model, θ, η_ind, ind.const_cov, vary) :
                  calculate_formulas_obs(dm.model, θ, η_ind, ind.const_cov, vary, sol_accessors)
            preds[j] = _stat_from_dist(getproperty(obs, obs_name), plot_func)
        end
    end
    return (x_obs=x_obs, x_fit=x_fit, preds=preds)
end

function _plot_fits_comparison_impl(fits::AbstractVector{<:FitResult},
                                    labels::Vector{String};
                                    dm::Union{Nothing, DataModel}=nothing,
                                    plot_func=mean,
                                    plot_data_points::Bool=true,
                                    observable=nothing,
                                    individuals_idx=nothing,
                                    x_axis_feature::Union{Nothing, Symbol}=nothing,
                                    shared_x_axis::Bool=true,
                                    shared_y_axis::Bool=true,
                                    ncols::Int=DEFAULT_PLOT_COLS,
                                    style::PlotStyle=PlotStyle(),
                                    kwargs_subplot=NamedTuple(),
                                    kwargs_layout=NamedTuple(),
                                    save_path::Union{Nothing, String}=nothing,
                                    plot_path::Union{Nothing, String}=nothing)
    length(fits) == length(labels) || error("Internal error: fits and labels length mismatch.")
    isempty(fits) && error("plot_fits_comparison requires at least one fit result.")

    save_path = _resolve_plot_path(save_path, plot_path)
    dms = dm === nothing ? [_get_dm(res, nothing) for res in fits] : fill(dm, length(fits))
    dm_ref = _validate_same_data_model_for_comparison(dms)

    obs_name = _get_observable(dm_ref, observable)
    inds = individuals_idx === nothing ? collect(eachindex(dm_ref.individuals)) : collect(individuals_idx)
    caches = [build_plot_cache(fits[j]; dm=dms[j], cache_obs_dists=false) for j in eachindex(fits)]
    line_colors = _comparison_line_colors(length(fits), style)

    plots = Vector{Any}(undef, length(inds))
    xlims = nothing
    ylims = nothing
    for (k, i) in enumerate(inds)
        ind = dm_ref.individuals[i]
        obs_rows = dm_ref.row_groups.obs_rows[i]
        x_obs = _get_x_values(dm_ref, ind, obs_rows, x_axis_feature)
        y_obs = getfield(ind.series.obs, obs_name)
        title_id = string(dm_ref.config.primary_id, ": ", dm_ref.df[obs_rows[1], dm_ref.config.primary_id])
        p = create_styled_plot(title=title_id,
                               xlabel=x_axis_feature === nothing ? "Time" : _axis_label(x_axis_feature),
                               ylabel=_axis_label(obs_name),
                               style=style,
                               kwargs_subplot...)
        if plot_data_points
            create_styled_scatter!(p, x_obs, y_obs; label="data", color=style.color_primary, style=style)
        end

        for j in eachindex(fits)
            curve = _fit_curve_from_cache(dms[j], caches[j], i, obs_name, x_axis_feature, plot_func)
            create_styled_line!(
                p,
                curve.x_fit,
                curve.preds;
                label=labels[j],
                color=line_colors[j],
                style=style,
                linestyle=_comparison_line_style(labels[j], style),
            )
            xlims = xlims === nothing ? (minimum(curve.x_fit), maximum(curve.x_fit)) :
                    (min(xlims[1], minimum(curve.x_fit)), max(xlims[2], maximum(curve.x_fit)))
            ylims = ylims === nothing ? (minimum(curve.preds), maximum(curve.preds)) :
                    (min(ylims[1], minimum(curve.preds)), max(ylims[2], maximum(curve.preds)))
        end

        ylims = ylims === nothing ? (minimum(y_obs), maximum(y_obs)) :
                (min(ylims[1], minimum(y_obs)), max(ylims[2], maximum(y_obs)))
        plots[k] = p
    end

    if shared_x_axis || shared_y_axis
        xlim_use = shared_x_axis ? _pad_limits(xlims[1], xlims[2]) : nothing
        ylim_use = shared_y_axis ? _pad_limits(ylims[1], ylims[2]) : nothing
        _apply_shared_axes!(plots, xlim_use, ylim_use)
    end
    p = combine_plots(plots; ncols=ncols, kwargs_layout...)
    return _save_plot!(p, save_path)
end

"""
    plot_fits_comparison(res::Union{FitResult, MultistartFitResult}; kwargs...)
                         -> Plots.Plot

    plot_fits_comparison(results::AbstractVector; kwargs...) -> Plots.Plot

    plot_fits_comparison(results::NamedTuple; kwargs...) -> Plots.Plot

    plot_fits_comparison(results::AbstractDict; kwargs...) -> Plots.Plot

Plot predictions from one or more fitted models side-by-side for visual comparison.

When called with a single `FitResult` or `MultistartFitResult`, behaves like
[`plot_fits`](@ref). When called with a collection, overlays predictions from each
model on the same panel, labelled by vector index, `NamedTuple` key, or `Dict` key.

All keyword arguments are forwarded to the underlying `plot_fits` implementation.
"""
function plot_fits_comparison(res::Union{FitResult, MultistartFitResult}; kwargs...)
    return plot_fits(_as_fit_result_for_plotting(res); kwargs...)
end

function plot_fits_comparison(results::AbstractVector; kwargs...)
    isempty(results) && error("plot_fits_comparison requires a non-empty vector of fit results.")
    labels = ["Model $(i)" for i in eachindex(results)]
    fits = [_as_fit_result_for_plotting(results[i]) for i in eachindex(results)]
    return _plot_fits_comparison_impl(fits, labels; kwargs...)
end

function plot_fits_comparison(results::NamedTuple; kwargs...)
    keys_nt = collect(keys(results))
    isempty(keys_nt) && error("plot_fits_comparison requires a non-empty NamedTuple of fit results.")
    labels = [string(k) for k in keys_nt]
    fits = [_as_fit_result_for_plotting(getfield(results, k)) for k in keys_nt]
    return _plot_fits_comparison_impl(fits, labels; kwargs...)
end

function plot_fits_comparison(results::AbstractDict; kwargs...)
    isempty(results) && error("plot_fits_comparison requires a non-empty Dict of fit results.")
    labels = String[]
    fits = FitResult[]
    for (k, v) in pairs(results)
        push!(labels, string(k))
        push!(fits, _as_fit_result_for_plotting(v))
    end
    return _plot_fits_comparison_impl(fits, labels; kwargs...)
end
