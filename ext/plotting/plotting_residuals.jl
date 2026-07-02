function _plot_residuals_df(df::DataFrame;
        metric::Symbol = :quantile,
        x_label::String = "x",
        shared_x_axis::Bool = true,
        shared_y_axis::Bool = true,
        ncols::Int = DEFAULT_PLOT_COLS,
        style::PlotStyle = PlotStyle(),
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple(),
        save_path::Union{Nothing, String} = nothing)
    metric = _validate_plot_metric(metric)
    col = _residual_metric_column(metric)
    plots = Vector{Any}()
    xlims = nothing
    ylims = nothing

    for g in groupby(df, [:observable, :individual_idx])
        obs_name = g.observable[1]
        id_val = g.id[1]
        mask = .!ismissing.(g.x) .& .!ismissing.(g[!, col])
        x = Float64.(g.x[mask])
        y = Float64.(g[!, col][mask])
        isempty(x) && continue

        p = create_styled_plot(title = string(obs_name, " | id=", id_val),
            xlabel = x_label,
            ylabel = _residual_metric_label(metric),
            style = style,
            kwargs_subplot...)
        create_styled_scatter!(
            p, x, y; label = "", color = style.color_secondary, style = style)
        if metric == :pit
            add_reference_line!(
                p, 0.5; orientation = :horizontal, color = style.color_dark, label = "")
        elseif metric != :logscore
            add_reference_line!(
                p, 0.0; orientation = :horizontal, color = style.color_dark, label = "")
        end
        push!(plots, p)
        xlims = xlims === nothing ? (minimum(x), maximum(x)) :
                (min(xlims[1], minimum(x)), max(xlims[2], maximum(x)))
        ylims = ylims === nothing ? (minimum(y), maximum(y)) :
                (min(ylims[1], minimum(y)), max(ylims[2], maximum(y)))
    end

    if isempty(plots)
        p = create_styled_plot(
            title = "No residual data to plot.", style = style, kwargs_subplot...)
        return _save_plot!(p, save_path)
    end
    if shared_x_axis || shared_y_axis
        xlim_use = shared_x_axis && xlims !== nothing ? _pad_limits(xlims[1], xlims[2]) :
                   nothing
        ylim_use = shared_y_axis && ylims !== nothing ? _pad_limits(ylims[1], ylims[2]) :
                   nothing
        _apply_shared_axes!(plots, xlim_use, ylim_use)
    end
    p = combine_plots(plots; ncols = ncols, kwargs_layout...)
    return _save_plot!(p, save_path)
end

"""
    plot_residuals(res::FitResult; dm, cache, residual, observables, individuals_idx,
                   obs_rows, x_axis_feature, shared_x_axis, shared_y_axis, ncols,
                   style, params, constants_re, cache_obs_dists, fitted_stat,
                   randomize_discrete, cdf_fallback_mc, ode_args, ode_kwargs,
                   mcmc_draws, mcmc_warmup, mcmc_quantiles, rng, save_path,
                   kwargs_subplot, kwargs_layout) -> Plots.Plot

    plot_residuals(dm::DataModel; ...) -> Plots.Plot

Plot residuals versus time (or another x-axis feature) for each individual.

# Keyword Arguments
- `residual::Symbol = :quantile`: residual metric to plot. One of `:pit`, `:quantile`,
  `:raw`, `:pearson`, `:logscore`.
- All other arguments are forwarded to [`get_residuals`](@ref); see that function for
  descriptions.
- `shared_x_axis::Bool = true`, `shared_y_axis::Bool = true`: share axis ranges.
- `ncols::Int = 3`: number of subplot columns.
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `kwargs_subplot`, `kwargs_layout`: extra keyword arguments for subplots and layout.
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
"""
function plot_residuals(res::FitResult;
        dm::Union{Nothing, DataModel} = nothing,
        cache::Union{Nothing, PlotCache} = nothing,
        residual::Symbol = :quantile,
        observables = nothing,
        individuals_idx = nothing,
        obs_rows = nothing,
        x_axis_feature::Union{Nothing, Symbol} = nothing,
        shared_x_axis::Bool = true,
        shared_y_axis::Bool = true,
        ncols::Int = DEFAULT_PLOT_COLS,
        style::PlotStyle = PlotStyle(),
        params::NamedTuple = NamedTuple(),
        constants_re::NamedTuple = NamedTuple(),
        cache_obs_dists::Bool = true,
        fitted_stat = mean,
        randomize_discrete::Bool = true,
        cdf_fallback_mc::Int = 0,
        ode_args::Tuple = (),
        ode_kwargs::NamedTuple = NamedTuple(),
        mcmc_draws::Int = 1000,
        mcmc_warmup::Union{Nothing, Int} = nothing,
        mcmc_quantiles::Vector{<:Real} = [5, 95],
        rng::AbstractRNG = Random.default_rng(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing,
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple())
    save_path = _resolve_plot_path(save_path, plot_path)
    metric = _validate_plot_metric(residual)
    df = get_residuals(res; dm = dm, cache = cache, observables = observables,
        individuals_idx = individuals_idx, obs_rows = obs_rows,
        x_axis_feature = x_axis_feature, params = params, constants_re = constants_re, cache_obs_dists = cache_obs_dists,
        residuals = [metric], fitted_stat = fitted_stat, randomize_discrete = randomize_discrete,
        cdf_fallback_mc = cdf_fallback_mc, ode_args = ode_args, ode_kwargs = ode_kwargs,
        mcmc_draws = mcmc_draws, mcmc_warmup = mcmc_warmup, mcmc_quantiles = mcmc_quantiles, rng = rng)
    xlabel = x_axis_feature === nothing ? "Time" : _axis_label(x_axis_feature)
    return _plot_residuals_df(
        df; metric = metric, x_label = xlabel, shared_x_axis = shared_x_axis,
        shared_y_axis = shared_y_axis, ncols = ncols, style = style,
        kwargs_subplot = kwargs_subplot, kwargs_layout = kwargs_layout, save_path = save_path)
end

function plot_residuals(dm::DataModel;
        cache::Union{Nothing, PlotCache} = nothing,
        residual::Symbol = :quantile,
        observables = nothing,
        individuals_idx = nothing,
        obs_rows = nothing,
        x_axis_feature::Union{Nothing, Symbol} = nothing,
        shared_x_axis::Bool = true,
        shared_y_axis::Bool = true,
        ncols::Int = DEFAULT_PLOT_COLS,
        style::PlotStyle = PlotStyle(),
        params::NamedTuple = NamedTuple(),
        constants_re::NamedTuple = NamedTuple(),
        cache_obs_dists::Bool = true,
        fitted_stat = mean,
        randomize_discrete::Bool = true,
        cdf_fallback_mc::Int = 0,
        ode_args::Tuple = (),
        ode_kwargs::NamedTuple = NamedTuple(),
        mcmc_draws::Int = 1000,
        mcmc_warmup::Union{Nothing, Int} = nothing,
        mcmc_quantiles::Vector{<:Real} = [5, 95],
        rng::AbstractRNG = Random.default_rng(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing,
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple())
    save_path = _resolve_plot_path(save_path, plot_path)
    metric = _validate_plot_metric(residual)
    df = get_residuals(dm; cache = cache, observables = observables,
        individuals_idx = individuals_idx, obs_rows = obs_rows,
        x_axis_feature = x_axis_feature, params = params, constants_re = constants_re, cache_obs_dists = cache_obs_dists,
        residuals = [metric], fitted_stat = fitted_stat, randomize_discrete = randomize_discrete,
        cdf_fallback_mc = cdf_fallback_mc, ode_args = ode_args, ode_kwargs = ode_kwargs,
        mcmc_draws = mcmc_draws, mcmc_warmup = mcmc_warmup, mcmc_quantiles = mcmc_quantiles, rng = rng)
    xlabel = x_axis_feature === nothing ? "Time" : _axis_label(x_axis_feature)
    return _plot_residuals_df(
        df; metric = metric, x_label = xlabel, shared_x_axis = shared_x_axis,
        shared_y_axis = shared_y_axis, ncols = ncols, style = style,
        kwargs_subplot = kwargs_subplot, kwargs_layout = kwargs_layout, save_path = save_path)
end

function _plot_residual_distribution_df(df::DataFrame;
        metric::Symbol = :quantile,
        shared_x_axis::Bool = true,
        shared_y_axis::Bool = true,
        ncols::Int = DEFAULT_PLOT_COLS,
        style::PlotStyle = PlotStyle(),
        bins::Int = 20,
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple(),
        save_path::Union{Nothing, String} = nothing)
    metric = _validate_plot_metric(metric)
    col = _residual_metric_column(metric)
    plots = Vector{Any}()
    xlims = nothing
    ylims = nothing
    for g in groupby(df, :observable)
        obs_name = g.observable[1]
        vals = Float64.(collect(skipmissing(g[!, col])))
        isempty(vals) && continue
        p = create_styled_plot(title = string(obs_name),
            xlabel = _residual_metric_label(metric),
            ylabel = "Probability",
            style = style, kwargs_subplot...)
        histogram!(p, vals; bins = bins, normalize = :probability,
            color = style.color_secondary, label = "data")
        if metric == :quantile
            xs = range(minimum(vals), maximum(vals); length = 200)
            plot!(p, xs, pdf.(Normal(), xs); color = style.color_dark,
                linestyle = :dash, label = "N(0,1)")
        elseif metric == :pit
            add_reference_line!(p, 1.0 / bins; orientation = :horizontal,
                color = style.color_dark, label = "")
        end
        push!(plots, p)
        xlims = xlims === nothing ? (minimum(vals), maximum(vals)) :
                (min(xlims[1], minimum(vals)), max(xlims[2], maximum(vals)))
        yv = p.series_list[end][:y]
        ymax = maximum(yv)
        ylims = ylims === nothing ? (0.0, ymax) : (0.0, max(ylims[2], ymax))
    end

    if isempty(plots)
        p = create_styled_plot(
            title = "No residual data to plot.", style = style, kwargs_subplot...)
        return _save_plot!(p, save_path)
    end
    if shared_x_axis || shared_y_axis
        xlim_use = shared_x_axis && xlims !== nothing ? _pad_limits(xlims[1], xlims[2]) :
                   nothing
        ylim_use = shared_y_axis && ylims !== nothing ? _pad_limits(ylims[1], ylims[2]) :
                   nothing
        _apply_shared_axes!(plots, xlim_use, ylim_use)
    end
    p = combine_plots(plots; ncols = ncols, kwargs_layout...)
    return _save_plot!(p, save_path)
end

"""
    plot_residual_distribution(res::FitResult; dm, cache, residual, observables,
                               individuals_idx, obs_rows, x_axis_feature,
                               shared_x_axis, shared_y_axis, ncols, style, bins,
                               params, constants_re, cache_obs_dists, fitted_stat,
                               randomize_discrete, cdf_fallback_mc, ode_args,
                               ode_kwargs, mcmc_draws, mcmc_warmup, mcmc_quantiles,
                               rng, save_path, kwargs_subplot, kwargs_layout)
                               -> Plots.Plot

    plot_residual_distribution(dm::DataModel; ...) -> Plots.Plot

Plot the marginal distribution of residuals as histograms with optional density overlays.

# Keyword Arguments
- `residual::Symbol = :quantile`: residual metric (`:pit`, `:quantile`, `:raw`,
  `:pearson`, `:logscore`).
- `bins::Int = 20`: number of histogram bins.
- All other arguments are forwarded to [`get_residuals`](@ref).
"""
function plot_residual_distribution(res::FitResult;
        dm::Union{Nothing, DataModel} = nothing,
        cache::Union{Nothing, PlotCache} = nothing,
        residual::Symbol = :quantile,
        observables = nothing,
        individuals_idx = nothing,
        obs_rows = nothing,
        x_axis_feature::Union{Nothing, Symbol} = nothing,
        shared_x_axis::Bool = true,
        shared_y_axis::Bool = true,
        ncols::Int = DEFAULT_PLOT_COLS,
        style::PlotStyle = PlotStyle(),
        bins::Int = 20,
        params::NamedTuple = NamedTuple(),
        constants_re::NamedTuple = NamedTuple(),
        cache_obs_dists::Bool = true,
        fitted_stat = mean,
        randomize_discrete::Bool = true,
        cdf_fallback_mc::Int = 0,
        ode_args::Tuple = (),
        ode_kwargs::NamedTuple = NamedTuple(),
        mcmc_draws::Int = 1000,
        mcmc_warmup::Union{Nothing, Int} = nothing,
        mcmc_quantiles::Vector{<:Real} = [5, 95],
        rng::AbstractRNG = Random.default_rng(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing,
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple())
    save_path = _resolve_plot_path(save_path, plot_path)
    metric = _validate_plot_metric(residual)
    df = get_residuals(res; dm = dm, cache = cache, observables = observables,
        individuals_idx = individuals_idx, obs_rows = obs_rows,
        x_axis_feature = x_axis_feature, params = params, constants_re = constants_re, cache_obs_dists = cache_obs_dists,
        residuals = [metric], fitted_stat = fitted_stat, randomize_discrete = randomize_discrete,
        cdf_fallback_mc = cdf_fallback_mc, ode_args = ode_args, ode_kwargs = ode_kwargs,
        mcmc_draws = mcmc_draws, mcmc_warmup = mcmc_warmup, mcmc_quantiles = mcmc_quantiles, rng = rng)
    return _plot_residual_distribution_df(
        df; metric = metric, shared_x_axis = shared_x_axis,
        shared_y_axis = shared_y_axis, ncols = ncols, style = style, bins = bins,
        kwargs_subplot = kwargs_subplot, kwargs_layout = kwargs_layout, save_path = save_path)
end

function plot_residual_distribution(dm::DataModel;
        cache::Union{Nothing, PlotCache} = nothing,
        residual::Symbol = :quantile,
        observables = nothing,
        individuals_idx = nothing,
        obs_rows = nothing,
        x_axis_feature::Union{Nothing, Symbol} = nothing,
        shared_x_axis::Bool = true,
        shared_y_axis::Bool = true,
        ncols::Int = DEFAULT_PLOT_COLS,
        style::PlotStyle = PlotStyle(),
        bins::Int = 20,
        params::NamedTuple = NamedTuple(),
        constants_re::NamedTuple = NamedTuple(),
        cache_obs_dists::Bool = true,
        fitted_stat = mean,
        randomize_discrete::Bool = true,
        cdf_fallback_mc::Int = 0,
        ode_args::Tuple = (),
        ode_kwargs::NamedTuple = NamedTuple(),
        mcmc_draws::Int = 1000,
        mcmc_warmup::Union{Nothing, Int} = nothing,
        mcmc_quantiles::Vector{<:Real} = [5, 95],
        rng::AbstractRNG = Random.default_rng(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing,
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple())
    save_path = _resolve_plot_path(save_path, plot_path)
    metric = _validate_plot_metric(residual)
    df = get_residuals(dm; cache = cache, observables = observables,
        individuals_idx = individuals_idx, obs_rows = obs_rows,
        x_axis_feature = x_axis_feature, params = params, constants_re = constants_re, cache_obs_dists = cache_obs_dists,
        residuals = [metric], fitted_stat = fitted_stat, randomize_discrete = randomize_discrete,
        cdf_fallback_mc = cdf_fallback_mc, ode_args = ode_args, ode_kwargs = ode_kwargs,
        mcmc_draws = mcmc_draws, mcmc_warmup = mcmc_warmup, mcmc_quantiles = mcmc_quantiles, rng = rng)
    return _plot_residual_distribution_df(
        df; metric = metric, shared_x_axis = shared_x_axis,
        shared_y_axis = shared_y_axis, ncols = ncols, style = style, bins = bins,
        kwargs_subplot = kwargs_subplot, kwargs_layout = kwargs_layout, save_path = save_path)
end

function _plot_residual_qq_df(df::DataFrame;
        metric::Symbol = :quantile,
        ncols::Int = DEFAULT_PLOT_COLS,
        style::PlotStyle = PlotStyle(),
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple(),
        save_path::Union{Nothing, String} = nothing)
    metric = _validate_plot_metric(metric)
    col = _residual_metric_column(metric)
    plots = Vector{Any}()
    for g in groupby(df, :observable)
        obs_name = g.observable[1]
        vals = sort(Float64.(collect(skipmissing(g[!, col]))))
        length(vals) < 2 && continue
        n = length(vals)
        probs = ((1:n) .- 0.5) ./ n
        theo = if metric == :pit
            probs
        else
            quantile.(Normal(), probs)
        end
        xlabel = metric == :pit ? "Theoretical Uniform Quantile" :
                 "Theoretical Normal Quantile"
        ylabel = string("Sample ", _residual_metric_label(metric))
        p = create_styled_plot(title = string(obs_name), xlabel = xlabel, ylabel = ylabel,
            style = style, kwargs_subplot...)
        scatter!(p, theo, vals; color = style.color_secondary,
            markersize = style.marker_size_small, label = "")
        lo = min(minimum(theo), minimum(vals))
        hi = max(maximum(theo), maximum(vals))
        plot!(
            p, [lo, hi], [lo, hi]; color = style.color_dark, linestyle = :dash, label = "")
        push!(plots, p)
    end
    if isempty(plots)
        p = create_styled_plot(
            title = "No residual data to plot.", style = style, kwargs_subplot...)
        return _save_plot!(p, save_path)
    end
    p = combine_plots(plots; ncols = ncols, kwargs_layout...)
    return _save_plot!(p, save_path)
end

"""
    plot_residual_qq(res::FitResult; dm, cache, residual, observables, individuals_idx,
                     obs_rows, x_axis_feature, ncols, style, params, constants_re,
                     cache_obs_dists, fitted_stat, randomize_discrete, cdf_fallback_mc,
                     ode_args, ode_kwargs, mcmc_draws, mcmc_warmup, mcmc_quantiles,
                     rng, save_path, kwargs_subplot, kwargs_layout) -> Plots.Plot

    plot_residual_qq(dm::DataModel; ...) -> Plots.Plot

Quantile-quantile plot of residuals against the theoretical distribution
(Uniform for `:pit`, Normal for other metrics).

# Keyword Arguments
- `residual::Symbol = :quantile`: residual metric (`:pit`, `:quantile`, `:raw`,
  `:pearson`, `:logscore`).
- All other arguments are forwarded to [`get_residuals`](@ref).
"""
function plot_residual_qq(res::FitResult;
        dm::Union{Nothing, DataModel} = nothing,
        cache::Union{Nothing, PlotCache} = nothing,
        residual::Symbol = :quantile,
        observables = nothing,
        individuals_idx = nothing,
        obs_rows = nothing,
        x_axis_feature::Union{Nothing, Symbol} = nothing,
        ncols::Int = DEFAULT_PLOT_COLS,
        style::PlotStyle = PlotStyle(),
        params::NamedTuple = NamedTuple(),
        constants_re::NamedTuple = NamedTuple(),
        cache_obs_dists::Bool = true,
        fitted_stat = mean,
        randomize_discrete::Bool = true,
        cdf_fallback_mc::Int = 0,
        ode_args::Tuple = (),
        ode_kwargs::NamedTuple = NamedTuple(),
        mcmc_draws::Int = 1000,
        mcmc_warmup::Union{Nothing, Int} = nothing,
        mcmc_quantiles::Vector{<:Real} = [5, 95],
        rng::AbstractRNG = Random.default_rng(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing,
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple())
    save_path = _resolve_plot_path(save_path, plot_path)
    metric = _validate_plot_metric(residual)
    df = get_residuals(res; dm = dm, cache = cache, observables = observables,
        individuals_idx = individuals_idx, obs_rows = obs_rows,
        x_axis_feature = x_axis_feature, params = params, constants_re = constants_re, cache_obs_dists = cache_obs_dists,
        residuals = [metric], fitted_stat = fitted_stat, randomize_discrete = randomize_discrete,
        cdf_fallback_mc = cdf_fallback_mc, ode_args = ode_args, ode_kwargs = ode_kwargs,
        mcmc_draws = mcmc_draws, mcmc_warmup = mcmc_warmup, mcmc_quantiles = mcmc_quantiles, rng = rng)
    return _plot_residual_qq_df(df; metric = metric, ncols = ncols, style = style,
        kwargs_subplot = kwargs_subplot, kwargs_layout = kwargs_layout, save_path = save_path)
end

function plot_residual_qq(dm::DataModel;
        cache::Union{Nothing, PlotCache} = nothing,
        residual::Symbol = :quantile,
        observables = nothing,
        individuals_idx = nothing,
        obs_rows = nothing,
        x_axis_feature::Union{Nothing, Symbol} = nothing,
        ncols::Int = DEFAULT_PLOT_COLS,
        style::PlotStyle = PlotStyle(),
        params::NamedTuple = NamedTuple(),
        constants_re::NamedTuple = NamedTuple(),
        cache_obs_dists::Bool = true,
        fitted_stat = mean,
        randomize_discrete::Bool = true,
        cdf_fallback_mc::Int = 0,
        ode_args::Tuple = (),
        ode_kwargs::NamedTuple = NamedTuple(),
        mcmc_draws::Int = 1000,
        mcmc_warmup::Union{Nothing, Int} = nothing,
        mcmc_quantiles::Vector{<:Real} = [5, 95],
        rng::AbstractRNG = Random.default_rng(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing,
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple())
    save_path = _resolve_plot_path(save_path, plot_path)
    metric = _validate_plot_metric(residual)
    df = get_residuals(dm; cache = cache, observables = observables,
        individuals_idx = individuals_idx, obs_rows = obs_rows,
        x_axis_feature = x_axis_feature, params = params, constants_re = constants_re, cache_obs_dists = cache_obs_dists,
        residuals = [metric], fitted_stat = fitted_stat, randomize_discrete = randomize_discrete,
        cdf_fallback_mc = cdf_fallback_mc, ode_args = ode_args, ode_kwargs = ode_kwargs,
        mcmc_draws = mcmc_draws, mcmc_warmup = mcmc_warmup, mcmc_quantiles = mcmc_quantiles, rng = rng)
    return _plot_residual_qq_df(df; metric = metric, ncols = ncols, style = style,
        kwargs_subplot = kwargs_subplot, kwargs_layout = kwargs_layout, save_path = save_path)
end

function _plot_residual_pit_df(df::DataFrame;
        show_hist::Bool = true,
        show_kde::Bool = false,
        show_qq::Bool = false,
        ncols::Int = DEFAULT_PLOT_COLS,
        style::PlotStyle = PlotStyle(),
        kde_bandwidth::Union{Nothing, Float64} = nothing,
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple(),
        save_path::Union{Nothing, String} = nothing)
    if (show_hist + show_kde + show_qq) > 1
        @warn "plot_residual_pit expects one plot type at a time; defaulting to histogram." show_hist=show_hist show_kde=show_kde show_qq=show_qq
        show_hist = true
        show_kde = false
        show_qq = false
    end
    plots = Vector{Any}()
    for g in groupby(df, :observable)
        obs_name = g.observable[1]
        pits = Float64.(collect(skipmissing(g.pit)))
        length(pits) < 2 && continue
        if show_hist
            p = create_styled_plot(title = string(obs_name, " | PIT Histogram"),
                xlabel = "PIT", ylabel = "Probability",
                style = style, kwargs_subplot...)
            histogram!(p, pits; bins = 20, normalize = :probability,
                color = style.color_secondary, label = "")
            push!(plots, p)
        elseif show_kde
            p = create_styled_plot(title = string(obs_name, " | PIT Density"),
                xlabel = "PIT", ylabel = "Density",
                style = style, kwargs_subplot...)
            xk, yk = _kde_xy(pits; bandwidth = kde_bandwidth)
            plot!(p, xk, yk; color = style.color_secondary, label = "")
            push!(plots, p)
        else
            p = create_styled_plot(title = string(obs_name, " | PIT QQ"),
                xlabel = "Theoretical Uniform Quantile",
                ylabel = "Empirical PIT Quantile",
                style = style, kwargs_subplot...)
            q = sort(pits)
            u = range(0.0, 1.0; length = length(q))
            scatter!(p, u, q; color = style.color_secondary, label = "")
            plot!(p, u, u; color = style.color_dark, linestyle = :dash, label = "")
            push!(plots, p)
        end
    end
    if isempty(plots)
        p = create_styled_plot(
            title = "No PIT data to plot.", style = style, kwargs_subplot...)
        return _save_plot!(p, save_path)
    end
    _apply_shared_axes!(plots, (0.0, 1.0), nothing)
    p = combine_plots(plots; ncols = ncols, kwargs_layout...)
    return _save_plot!(p, save_path)
end

"""
    plot_residual_pit(res::FitResult; dm, cache, observables, individuals_idx, obs_rows,
                      x_axis_feature, show_hist, show_kde, show_qq, ncols, style,
                      kde_bandwidth, params, constants_re, cache_obs_dists,
                      randomize_discrete, cdf_fallback_mc, ode_args, ode_kwargs,
                      mcmc_draws, mcmc_warmup, rng, save_path, kwargs_subplot,
                      kwargs_layout) -> Plots.Plot

    plot_residual_pit(dm::DataModel; ...) -> Plots.Plot

Plot the probability integral transform (PIT) values as histograms and/or KDE curves.
Uniform PIT values indicate a well-calibrated model.

# Keyword Arguments
- `show_hist::Bool = true`: show a histogram of PIT values.
- `show_kde::Bool = false`: overlay a kernel density estimate.
- `show_qq::Bool = false`: add a uniform QQ reference line.
- `kde_bandwidth::Union{Nothing, Float64} = nothing`: KDE bandwidth, or `nothing` for auto.
- All other arguments are forwarded to [`get_residuals`](@ref).
"""
function plot_residual_pit(res::FitResult;
        dm::Union{Nothing, DataModel} = nothing,
        cache::Union{Nothing, PlotCache} = nothing,
        observables = nothing,
        individuals_idx = nothing,
        obs_rows = nothing,
        x_axis_feature::Union{Nothing, Symbol} = nothing,
        show_hist::Bool = true,
        show_kde::Bool = false,
        show_qq::Bool = false,
        ncols::Int = DEFAULT_PLOT_COLS,
        style::PlotStyle = PlotStyle(),
        kde_bandwidth::Union{Nothing, Float64} = nothing,
        params::NamedTuple = NamedTuple(),
        constants_re::NamedTuple = NamedTuple(),
        cache_obs_dists::Bool = true,
        randomize_discrete::Bool = true,
        cdf_fallback_mc::Int = 0,
        ode_args::Tuple = (),
        ode_kwargs::NamedTuple = NamedTuple(),
        mcmc_draws::Int = 1000,
        mcmc_warmup::Union{Nothing, Int} = nothing,
        mcmc_quantiles::Vector{<:Real} = [5, 95],
        rng::AbstractRNG = Random.default_rng(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing,
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple())
    save_path = _resolve_plot_path(save_path, plot_path)
    df = get_residuals(res; dm = dm, cache = cache, observables = observables,
        individuals_idx = individuals_idx, obs_rows = obs_rows,
        x_axis_feature = x_axis_feature, params = params, constants_re = constants_re, cache_obs_dists = cache_obs_dists,
        residuals = [:pit], fitted_stat = mean, randomize_discrete = randomize_discrete,
        cdf_fallback_mc = cdf_fallback_mc, ode_args = ode_args, ode_kwargs = ode_kwargs,
        mcmc_draws = mcmc_draws, mcmc_warmup = mcmc_warmup, mcmc_quantiles = mcmc_quantiles, rng = rng)
    return _plot_residual_pit_df(
        df; show_hist = show_hist, show_kde = show_kde, show_qq = show_qq,
        ncols = ncols, style = style, kde_bandwidth = kde_bandwidth,
        kwargs_subplot = kwargs_subplot, kwargs_layout = kwargs_layout, save_path = save_path)
end

function plot_residual_pit(dm::DataModel;
        cache::Union{Nothing, PlotCache} = nothing,
        observables = nothing,
        individuals_idx = nothing,
        obs_rows = nothing,
        x_axis_feature::Union{Nothing, Symbol} = nothing,
        show_hist::Bool = true,
        show_kde::Bool = false,
        show_qq::Bool = false,
        ncols::Int = DEFAULT_PLOT_COLS,
        style::PlotStyle = PlotStyle(),
        kde_bandwidth::Union{Nothing, Float64} = nothing,
        params::NamedTuple = NamedTuple(),
        constants_re::NamedTuple = NamedTuple(),
        cache_obs_dists::Bool = true,
        randomize_discrete::Bool = true,
        cdf_fallback_mc::Int = 0,
        ode_args::Tuple = (),
        ode_kwargs::NamedTuple = NamedTuple(),
        mcmc_draws::Int = 1000,
        mcmc_warmup::Union{Nothing, Int} = nothing,
        mcmc_quantiles::Vector{<:Real} = [5, 95],
        rng::AbstractRNG = Random.default_rng(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing,
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple())
    save_path = _resolve_plot_path(save_path, plot_path)
    df = get_residuals(dm; cache = cache, observables = observables,
        individuals_idx = individuals_idx, obs_rows = obs_rows,
        x_axis_feature = x_axis_feature, params = params, constants_re = constants_re, cache_obs_dists = cache_obs_dists,
        residuals = [:pit], fitted_stat = mean, randomize_discrete = randomize_discrete,
        cdf_fallback_mc = cdf_fallback_mc, ode_args = ode_args, ode_kwargs = ode_kwargs,
        mcmc_draws = mcmc_draws, mcmc_warmup = mcmc_warmup, mcmc_quantiles = mcmc_quantiles, rng = rng)
    return _plot_residual_pit_df(
        df; show_hist = show_hist, show_kde = show_kde, show_qq = show_qq,
        ncols = ncols, style = style, kde_bandwidth = kde_bandwidth,
        kwargs_subplot = kwargs_subplot, kwargs_layout = kwargs_layout, save_path = save_path)
end

function _plot_residual_acf_df(df::DataFrame;
        metric::Symbol = :quantile,
        max_lag::Int = 5,
        ncols::Int = DEFAULT_PLOT_COLS,
        style::PlotStyle = PlotStyle(),
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple(),
        save_path::Union{Nothing, String} = nothing)
    max_lag >= 1 || error("max_lag must be >= 1.")
    metric = _validate_plot_metric(metric)
    col = _residual_metric_column(metric)
    plots = Vector{Any}()
    for gobs in groupby(df, :observable)
        obs_name = gobs.observable[1]
        by_ind = groupby(gobs, :individual_idx)
        acfs = Vector{Vector{Union{Missing, Float64}}}()
        for gi in by_ind
            order = sortperm(gi.obs_index)
            vals = gi[order, col]
            v = Float64.(collect(skipmissing(vals)))
            length(v) < 2 && continue
            push!(acfs, _acf_for_series(v, max_lag))
        end
        isempty(acfs) && continue
        acf_mean = Vector{Float64}(undef, max_lag)
        for lag in 1:max_lag
            vals_lag = Float64[]
            for a in acfs
                ismissing(a[lag]) || push!(vals_lag, a[lag])
            end
            acf_mean[lag] = isempty(vals_lag) ? NaN : mean(vals_lag)
        end
        lags = collect(1:max_lag)
        p = create_styled_plot(title = string(obs_name), xlabel = "Lag",
            ylabel = string(_residual_metric_label(metric), " Autocorrelation"),
            style = style, kwargs_subplot...)
        plot!(
            p, lags, acf_mean; color = style.color_secondary, marker = :circle, label = "")
        add_reference_line!(
            p, 0.0; orientation = :horizontal, color = style.color_dark, label = "")
        push!(plots, p)
    end
    if isempty(plots)
        p = create_styled_plot(
            title = "No residual data to plot.", style = style, kwargs_subplot...)
        return _save_plot!(p, save_path)
    end
    p = combine_plots(plots; ncols = ncols, kwargs_layout...)
    return _save_plot!(p, save_path)
end

"""
    plot_residual_acf(res::FitResult; dm, cache, residual, observables, individuals_idx,
                      obs_rows, x_axis_feature, max_lag, ncols, style, params,
                      constants_re, cache_obs_dists, fitted_stat, randomize_discrete,
                      cdf_fallback_mc, ode_args, ode_kwargs, mcmc_draws, mcmc_warmup,
                      mcmc_quantiles, rng, save_path, kwargs_subplot, kwargs_layout)
                      -> Plots.Plot

    plot_residual_acf(dm::DataModel; ...) -> Plots.Plot

Plot the autocorrelation function (ACF) of residuals across time lags for each outcome.

# Keyword Arguments
- `residual::Symbol = :quantile`: residual metric (`:pit`, `:quantile`, `:raw`,
  `:pearson`, `:logscore`).
- `max_lag::Int = 5`: maximum lag to compute and display.
- All other arguments are forwarded to [`get_residuals`](@ref).
"""
function plot_residual_acf(res::FitResult;
        dm::Union{Nothing, DataModel} = nothing,
        cache::Union{Nothing, PlotCache} = nothing,
        residual::Symbol = :quantile,
        observables = nothing,
        individuals_idx = nothing,
        obs_rows = nothing,
        x_axis_feature::Union{Nothing, Symbol} = nothing,
        max_lag::Int = 5,
        ncols::Int = DEFAULT_PLOT_COLS,
        style::PlotStyle = PlotStyle(),
        params::NamedTuple = NamedTuple(),
        constants_re::NamedTuple = NamedTuple(),
        cache_obs_dists::Bool = true,
        fitted_stat = mean,
        randomize_discrete::Bool = true,
        cdf_fallback_mc::Int = 0,
        ode_args::Tuple = (),
        ode_kwargs::NamedTuple = NamedTuple(),
        mcmc_draws::Int = 1000,
        mcmc_warmup::Union{Nothing, Int} = nothing,
        mcmc_quantiles::Vector{<:Real} = [5, 95],
        rng::AbstractRNG = Random.default_rng(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing,
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple())
    save_path = _resolve_plot_path(save_path, plot_path)
    metric = _validate_plot_metric(residual)
    df = get_residuals(res; dm = dm, cache = cache, observables = observables,
        individuals_idx = individuals_idx, obs_rows = obs_rows,
        x_axis_feature = x_axis_feature, params = params, constants_re = constants_re, cache_obs_dists = cache_obs_dists,
        residuals = [metric], fitted_stat = fitted_stat, randomize_discrete = randomize_discrete,
        cdf_fallback_mc = cdf_fallback_mc, ode_args = ode_args, ode_kwargs = ode_kwargs,
        mcmc_draws = mcmc_draws, mcmc_warmup = mcmc_warmup, mcmc_quantiles = mcmc_quantiles, rng = rng)
    return _plot_residual_acf_df(
        df; metric = metric, max_lag = max_lag, ncols = ncols, style = style,
        kwargs_subplot = kwargs_subplot, kwargs_layout = kwargs_layout, save_path = save_path)
end

function plot_residual_acf(dm::DataModel;
        cache::Union{Nothing, PlotCache} = nothing,
        residual::Symbol = :quantile,
        observables = nothing,
        individuals_idx = nothing,
        obs_rows = nothing,
        x_axis_feature::Union{Nothing, Symbol} = nothing,
        max_lag::Int = 5,
        ncols::Int = DEFAULT_PLOT_COLS,
        style::PlotStyle = PlotStyle(),
        params::NamedTuple = NamedTuple(),
        constants_re::NamedTuple = NamedTuple(),
        cache_obs_dists::Bool = true,
        fitted_stat = mean,
        randomize_discrete::Bool = true,
        cdf_fallback_mc::Int = 0,
        ode_args::Tuple = (),
        ode_kwargs::NamedTuple = NamedTuple(),
        mcmc_draws::Int = 1000,
        mcmc_warmup::Union{Nothing, Int} = nothing,
        mcmc_quantiles::Vector{<:Real} = [5, 95],
        rng::AbstractRNG = Random.default_rng(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing,
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple())
    save_path = _resolve_plot_path(save_path, plot_path)
    metric = _validate_plot_metric(residual)
    df = get_residuals(dm; cache = cache, observables = observables,
        individuals_idx = individuals_idx, obs_rows = obs_rows,
        x_axis_feature = x_axis_feature, params = params, constants_re = constants_re, cache_obs_dists = cache_obs_dists,
        residuals = [metric], fitted_stat = fitted_stat, randomize_discrete = randomize_discrete,
        cdf_fallback_mc = cdf_fallback_mc, ode_args = ode_args, ode_kwargs = ode_kwargs,
        mcmc_draws = mcmc_draws, mcmc_warmup = mcmc_warmup, mcmc_quantiles = mcmc_quantiles, rng = rng)
    return _plot_residual_acf_df(
        df; metric = metric, max_lag = max_lag, ncols = ncols, style = style,
        kwargs_subplot = kwargs_subplot, kwargs_layout = kwargs_layout, save_path = save_path)
end
