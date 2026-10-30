# =========================================================================================
# Diagnostics on AlgebraOfGraphics (#383, group 3): plot_fits_comparison,
# plot_observed_profiles, plot_dv_pred, plot_dv_ipred, plot_wres_pred, plot_shrinkage.
# Helpers of this section carry a `_diag_` prefix.
# =========================================================================================

# Logarithmic axis scales: Makie cannot draw non-positive values or `ABLines` on them.
_diag_is_log(scale) = scale === log10 || scale === log2 || scale === log

# (x is logarithmic, y is logarithmic) as requested through `kwargs_subplot`.
function _diag_log_axes(kwargs_subplot)
    kw = _nt(kwargs_subplot)
    return _diag_is_log(get(kw, :xscale, identity)), _diag_is_log(get(kw, :yscale, identity))
end

# Padded limits of the finite values (positive values only on a logarithmic axis, padded
# multiplicatively there); `nothing` when no value qualifies.
function _diag_lims(vals, logscale::Bool)
    lo = Inf
    hi = -Inf
    for v in vals
        x = _f64(v)
        (isfinite(x) && (!logscale || x > 0)) || continue
        lo = min(lo, x)
        hi = max(hi, x)
    end
    isfinite(lo) || return nothing
    if logscale
        f = lo == hi ? 2.0 : (hi / lo)^0.05
        return (lo / f, hi * f)
    end
    return _pad_limits(lo, hi)
end

# Drop the points a logarithmic axis cannot show (with one warning).
function _diag_keep_positive(xs, ys, xlog::Bool, ylog::Bool, fn::AbstractString)
    xs = collect(Float64, xs)
    ys = collect(Float64, ys)
    (xlog || ylog) || return xs, ys
    keep = [(!xlog || x > 0) && (!ylog || y > 0) for (x, y) in zip(xs, ys)]
    n_drop = count(!, keep)
    n_drop > 0 &&
        @warn "$(fn): dropping $(n_drop) non-positive value(s) that a logarithmic axis cannot show."
    return xs[keep], ys[keep]
end

# Dashed identity line y = x: `ablines!` on linear axes; on a logarithmic axis (where Makie
# rejects ABLines) a dense line over the x limits.
function _diag_identity_line!(ax, xlims, xlog::Bool, ylog::Bool, style::PlotStyle)
    attrs = (;
        color = style.color_reference, linestyle = :dash,
        linewidth = style.line_width_secondary,
    )
    if !(xlog || ylog)
        Makie.ablines!(ax, 0.0, 1.0; attrs...)
        return ax
    end
    xlims === nothing && return ax
    lo, hi = xlims
    xs = xlog ? exp10.(range(log10(lo), log10(hi); length = 200)) :
        collect(range(lo, hi; length = 200))
    ylog && (xs = filter(>(0), xs))
    isempty(xs) || lines!(ax, xs, xs; attrs...)
    return ax
end

"""
    _diag_gof_figure(specs, reference; xlabel, ylabel, style, kwargs_subplot, kwargs_layout,
                     save_path, return_panel, fn) -> Figure | AoGPanel

Shared drawing of the goodness-of-fit scatter plots (DV vs PRED, DV vs IPRED, WRES vs
PRED). `specs` holds one `(title, xs, ys)` per observable component; `reference` is
`:identity` (y = x, square limits) or `:zero` (horizontal line at 0). A scalar observable
gives one `AoGPanel` (returned as is with `return_panel = true`); a vector-valued
observable gives one facet per component with free axes and per-panel limits.
"""
function _diag_gof_figure(
        specs, reference::Symbol; xlabel, ylabel, style::PlotStyle, kwargs_subplot,
        kwargs_layout, save_path, return_panel::Bool, fn::AbstractString
    )
    xlog, ylog = _diag_log_axes(kwargs_subplot)
    user_limits = haskey(_nt(kwargs_subplot), :limits)
    df = _tidy()
    titles = String[]
    post = Dict{String, Vector{Any}}()
    for (title, xs_raw, ys_raw) in specs
        xs, ys = _diag_keep_positive(xs_raw, ys_raw, xlog, ylog, fn)
        isempty(xs) &&
            error("$(fn): no observations of :$(title) can be shown on a logarithmic axis.")
        _push_xy!(df, title, xs, ys, "data")
        push!(titles, title)
        if reference == :identity && xlog == ylog
            xl = _diag_lims(vcat(xs, ys), xlog)
            yl = xl
        else
            xl = _diag_lims(xs, xlog)
            yl = _diag_lims(ys, ylog)
        end
        hooks = Any[]
        if reference == :identity
            push!(hooks, ax -> _diag_identity_line!(ax, xl, xlog, ylog, style))
        elseif !ylog
            push!(
                hooks,
                ax -> hlines!(
                    ax, 0.0; color = style.color_reference, linestyle = :dash,
                    linewidth = style.line_width_secondary
                )
            )
        end
        user_limits || push!(hooks, ax -> Makie.limits!(ax, xl, yl))
        post[title] = hooks
    end
    scatter = _scatter_visual(style; color = style.color_primary)
    if length(specs) == 1
        title = only(specs)[1]
        panel = AoGPanel(
            AoG.data(df) * AoG.mapping(:x, :y) * scatter;
            axis = _aog_axis(
                style, kwargs_subplot;
                title = "", xlabel = xlabel, ylabel = ylabel
            ),
            post = post[title], legend = false
        )
        return_panel && return panel
        fig = combine_plots([panel]; ncols = 1, style = style, _nt(kwargs_layout)...)
        return _save_plot!(fig, save_path)
    end
    fig = _draw_facets(
        AoG.data(df) * AoG.mapping(:x, :y; layout = :panel) * scatter;
        panels = titles, ncols = min(length(titles), DEFAULT_PLOT_COLS), style = style,
        shared_x = false, shared_y = false,
        kwargs_subplot = kwargs_subplot, kwargs_layout = kwargs_layout,
        axis = (; xlabel = xlabel, ylabel = ylabel), post = post, legend = false
    )
    return _save_plot!(fig, save_path)
end

function _plot_fits_comparison_impl(
        fits::AbstractVector{<:FitResult},
        labels::Vector{String};
        dm::Union{Nothing, DataModel} = nothing,
        plot_func = mean,
        plot_data_points::Bool = true,
        observable = nothing,
        individuals_idx = nothing,
        x_axis_feature::Union{Nothing, Symbol} = nothing,
        shared_x_axis::Bool = true,
        shared_y_axis::Bool = true,
        ncols::Int = DEFAULT_PLOT_COLS,
        style::PlotStyle = PlotStyle(),
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing,
        rng::AbstractRNG = _default_rng()
    )
    length(fits) == length(labels) ||
        error("Internal error: fits and labels length mismatch.")
    isempty(fits) && error("plot_fits_comparison requires at least one fit result.")
    _check_positive_int(ncols, "ncols")

    save_path = _resolve_plot_path(save_path, plot_path)
    dms = dm === nothing ? [_get_dm(res, nothing) for res in fits] : fill(dm, length(fits))
    dm_ref = _validate_same_data_model_for_comparison(dms)

    obs_name = _get_observable(dm_ref, observable)
    inds = _resolve_individuals(dm_ref, individuals_idx; default_all = true)
    caches = [
        build_plot_cache(fits[j]; dm = dms[j], cache_obs_dists = false, rng = rng)
            for j in eachindex(fits)
    ]
    line_colors = _comparison_line_colors(length(fits), style)

    panels = String[]
    obs_df = _tidy()
    line_dfs = [_tidy() for _ in fits]
    xlims = nothing
    ylims = nothing
    for i in inds
        ind = get_individuals(dm_ref)[i]
        obs_rows = get_obs_rows(get_row_groups(dm_ref))[i]
        x_obs = _get_x_values(dm_ref, ind, obs_rows, x_axis_feature)
        y_obs = getfield(get_obs(get_series(ind)), obs_name)
        x_obs_plot, y_obs_plot = _collect_scalar_series(x_obs, y_obs)
        panel = _individual_panel(dm_ref, i)
        push!(panels, panel)
        plot_data_points && _push_xy!(obs_df, panel, x_obs_plot, y_obs_plot, "data")
        for j in eachindex(fits)
            curve = _fit_curve_from_cache(
                dms[j], caches[j], i, obs_name, x_axis_feature, plot_func
            )
            _push_xy!(line_dfs[j], panel, curve.x_fit, curve.preds, labels[j])
            xlims = _merge_limits(xlims, curve.x_fit)
            ylims = _merge_limits(ylims, curve.preds)
        end
        ylims = _merge_limits(ylims, y_obs_plot)
    end

    # One legend entry per model (line in the model's colour and line style) plus the data.
    series = Pair{String, Any}[]
    plot_data_points && push!(series, "data" => style.color_primary)
    for j in eachindex(fits)
        push!(series, labels[j] => line_colors[j])
    end
    xy = AoG.mapping(:x, :y; color = :series, layout = :panel)
    line_layers = [
        _layer(
                line_dfs[j], xy,
                _line_visual(style; linestyle = _comparison_line_style(labels[j], style))
            )
            for j in eachindex(fits)
    ]
    layers = _layers(line_layers..., _layer(obs_df, xy, _scatter_visual(style)))
    fig = _draw_facets(
        layers; panels = panels, ncols = ncols, style = style,
        shared_x = shared_x_axis, shared_y = shared_y_axis,
        kwargs_subplot = kwargs_subplot, kwargs_layout = kwargs_layout,
        axis = (;
            xlabel = x_axis_feature === nothing ? "Time" : _axis_label(x_axis_feature),
            ylabel = _axis_label(obs_name),
        ),
        series = series,
        xlim = _shared_lim(shared_x_axis, xlims), ylim = _shared_lim(shared_y_axis, ylims),
    )
    return _save_plot!(fig, save_path)
end

"""
    plot_fits_comparison(res::Union{FitResult, MultistartFitResult}; kwargs...)
                         -> Makie.Figure

    plot_fits_comparison(results::AbstractVector; kwargs...) -> Makie.Figure

    plot_fits_comparison(results::NamedTuple; kwargs...) -> Makie.Figure

    plot_fits_comparison(results::AbstractDict; kwargs...) -> Makie.Figure

Plot predictions from one or more fitted models side-by-side for visual comparison.

When called with a single `FitResult` or `MultistartFitResult`, behaves like
[`plot_fits`](@ref). When called with a collection, overlays predictions from each
model on the same panel, labeled by vector index (`"Model 1"`, ...), `NamedTuple` key, or
`Dict` key.

Panels are AlgebraOfGraphics facets titled by individual (linked axes when shared, inner
tick labels hidden), with one figure-level legend: `data` and one entry per model, drawn
in the model's colour and line style (`PlotStyle.comparison_line_styles`,
`PlotStyle.comparison_default_linestyle`).

# Keyword Arguments (collections)
- `dm`, `plot_func = mean`, `plot_data_points = true`, `observable`, `individuals_idx`,
  `x_axis_feature`, `shared_x_axis = true`, `shared_y_axis = true`, `ncols = 3`,
  `style`, `kwargs_subplot` (Makie `Axis` attributes, passed to AlgebraOfGraphics as
  `axis`), `kwargs_layout` (Makie `Figure` attributes), `save_path`, `plot_path`.
- `rng` (default: a fresh `Xoshiro(42)` per call) for the posterior-mean parameters of
  MCMC/VI fits.
"""
function plot_fits_comparison(res::Union{FitResult, MultistartFitResult}; kwargs...)
    return plot_fits(_as_fit_result_for_plotting(res); kwargs...)
end

function plot_fits_comparison(results::AbstractVector; kwargs...)
    isempty(results) &&
        error("plot_fits_comparison requires a non-empty vector of fit results.")
    labels = ["Model $(i)" for i in eachindex(results)]
    fits = [_as_fit_result_for_plotting(results[i]) for i in eachindex(results)]
    return _plot_fits_comparison_impl(fits, labels; kwargs...)
end

function plot_fits_comparison(results::NamedTuple; kwargs...)
    keys_nt = collect(keys(results))
    isempty(keys_nt) &&
        error("plot_fits_comparison requires a non-empty NamedTuple of fit results.")
    labels = [string(k) for k in keys_nt]
    fits = [_as_fit_result_for_plotting(getfield(results, k)) for k in keys_nt]
    return _plot_fits_comparison_impl(fits, labels; kwargs...)
end

function plot_fits_comparison(results::AbstractDict; kwargs...)
    isempty(results) &&
        error("plot_fits_comparison requires a non-empty Dict of fit results.")
    labels = String[]
    fits = FitResult[]
    for (k, v) in pairs(results)
        push!(labels, string(k))
        push!(fits, _as_fit_result_for_plotting(v))
    end
    return _plot_fits_comparison_impl(fits, labels; kwargs...)
end

"""
    plot_observed_profiles(dm::DataModel; observable, individuals_idx, x_axis_feature,
                           marginal_idx, style, kwargs_subplot, kwargs_layout, save_path,
                           plot_path) -> Makie.Figure

    plot_observed_profiles(res::FitResult; dm, observable, individuals_idx,
                           x_axis_feature, marginal_idx, style, kwargs_subplot,
                           kwargs_layout, save_path, plot_path) -> Makie.Figure

Plot observed trajectories for all (or selected) individuals overlaid on a single panel
(spaghetti plot). Each individual's time series is drawn as a connected line with
scatter points at the observed time points.

# Keyword Arguments
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
- `observable`: name of the outcome variable to plot, or `nothing` to use the first.
- `individuals_idx`: indices or IDs of individuals to include, or `nothing` for all.
- `x_axis_feature::Union{Symbol, Nothing} = nothing`: covariate for the x-axis;
  defaults to the time column.
- `marginal_idx::Union{Nothing, Int} = nothing`: for vector-valued observables, plot only
  this component; `nothing` overlays every component in its own color (with one
  figure-level legend entry per component).
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `kwargs_subplot`: additional Makie `Axis` attributes for the panel (user values win).
- `kwargs_layout`: additional Makie `Figure` attributes (user values win).
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
- `plot_path::Union{Nothing, String} = nothing`: alias for `save_path`.
"""
function plot_observed_profiles(
        dm::DataModel;
        observable = nothing,
        individuals_idx = nothing,
        x_axis_feature::Union{Symbol, Nothing} = nothing,
        marginal_idx::Union{Nothing, Int} = nothing,
        style::PlotStyle = PlotStyle(),
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing
    )
    save_path = _resolve_plot_path(save_path, plot_path)
    obs_name = _get_observable(dm, observable)
    inds = _resolve_individuals(dm, individuals_idx; default_all = true)
    (is_mv, n_marginals) = _obs_multivariate_info(dm, obs_name)
    marginal_idx !== nothing && !is_mv &&
        error("marginal_idx only valid when the observable is multivariate.")
    marginal_idx !== nothing && !(1 <= marginal_idx <= n_marginals) &&
        error("marginal_idx must be between 1 and $(n_marginals).")
    margins = is_mv ? (marginal_idx === nothing ? (1:n_marginals) : (marginal_idx:marginal_idx)) :
        (0:0)
    colors = _marginal_colors(max(n_marginals, 1), style)
    series_label(m) = m == 0 ? "data" : _marginal_label(obs_name, m)

    # One line per (individual, component): `grp` keeps the profiles apart.
    df = _tidy(; grp = String)
    for i in inds
        ind = get_individuals(dm)[i]
        obs_rows = get_obs_rows(get_row_groups(dm))[i]
        x = _get_x_values(dm, ind, obs_rows, x_axis_feature)
        y = getfield(get_obs(get_series(ind)), obs_name)
        for m in margins
            xs, ys = m == 0 ? _collect_scalar_series(x, y) :
                _collect_multivariate_series(x, y, n_marginals; marginal_idx = m)
            isempty(xs) && continue
            order = sortperm(xs)
            for k in order
                push!(
                    df, (
                        panel = "", x = _f64(xs[k]), y = _f64(ys[k]),
                        series = series_label(m), grp = string(i, "/", m),
                    )
                )
            end
        end
    end

    series = [series_label(m) => (m == 0 ? style.color_primary : colors[m]) for m in margins]
    layers = _layers(
        _layer(df, AoG.mapping(:x, :y; color = :series, group = :grp), _line_visual(style)),
        _layer(df, AoG.mapping(:x, :y; color = :series), _scatter_visual(style)),
    )
    panel = AoGPanel(
        layers;
        scales = (; Color = _series_palette(series)),
        axis = _aog_axis(
            style, kwargs_subplot;
            title = "Observed profiles",
            xlabel = x_axis_feature === nothing ? "Time" : _axis_label(x_axis_feature),
            ylabel = _axis_label(obs_name),
        ),
        legend = is_mv
    )
    fig = combine_plots([panel]; ncols = 1, style = style, _nt(kwargs_layout)...)
    return _save_plot!(fig, save_path)
end

function plot_observed_profiles(
        res::FitResult;
        dm::Union{Nothing, DataModel} = nothing,
        observable = nothing,
        individuals_idx = nothing,
        x_axis_feature::Union{Symbol, Nothing} = nothing,
        marginal_idx::Union{Nothing, Int} = nothing,
        style::PlotStyle = PlotStyle(),
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing
    )
    dm = _get_dm(res, dm)
    return plot_observed_profiles(
        dm;
        observable = observable,
        individuals_idx = individuals_idx,
        x_axis_feature = x_axis_feature,
        marginal_idx = marginal_idx,
        style = style,
        kwargs_subplot = kwargs_subplot,
        kwargs_layout = kwargs_layout,
        save_path = save_path,
        plot_path = plot_path
    )
end

"""
    plot_dv_pred(res::FitResult; dm, observable, style, kwargs_subplot,
                 kwargs_layout, save_path, plot_path, return_panel) -> Makie.Figure

Goodness-of-fit scatter plot of observed values (DV) against population
predictions (PRED). PRED is the model prediction evaluated at each
individual's population-mean random effects using that individual's own
constant covariates, preserving covariate-driven shifts such as allometric
weight scaling. A dashed identity line is overlaid for reference.

A vector-valued observable gets one panel per component (free axes). Logarithmic axes
(`kwargs_subplot = (; xscale = log10, yscale = log10)`) are supported; non-positive
values are dropped with a warning.

# Keyword Arguments
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
- `observable`: name of the outcome variable, or `nothing` for the first.
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `kwargs_subplot`: additional Makie `Axis` attributes (user values win, e.g. `limits`).
- `kwargs_layout`: additional Makie `Figure` attributes forwarded to the layout.
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
- `plot_path::Union{Nothing, String} = nothing`: alias for `save_path`.
- `return_panel::Bool = false`: return the panel instead of a `Figure`, for composing
  several diagnostics into one figure via `combine_plots` (scalar observables only).
"""
function plot_dv_pred(
        res::FitResult;
        dm::Union{Nothing, DataModel} = nothing,
        observable = nothing,
        style::PlotStyle = PlotStyle(),
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing,
        return_panel::Bool = false
    )
    dm = _get_dm(res, dm)
    save_path = _resolve_plot_path(save_path, plot_path)
    obs_name = _get_observable(dm, observable)
    θ = get_params(res; scale = :untransformed)
    η_pred = _pred_re_per_individual(dm, θ)

    panels_spec, n_marginals = _marginal_panels(dm, obs_name)
    return_panel && length(panels_spec) > 1 &&
        error("return_panel requires a scalar observable; :$(obs_name) has $(n_marginals) components.")
    specs = map(panels_spec) do (m, label)
        dv, pred, _ = _collect_pred_series(
            dm, obs_name, θ, η_pred;
            marginal = m, n_marginals = n_marginals, fn = "plot_dv_pred"
        )
        isempty(dv) &&
            error("No non-missing observations found for observable :$(label).")
        (label, pred, dv)
    end
    return _diag_gof_figure(
        specs, :identity;
        xlabel = "Population prediction (PRED)", ylabel = "Observed (DV)",
        style = style, kwargs_subplot = kwargs_subplot, kwargs_layout = kwargs_layout,
        save_path = save_path, return_panel = return_panel, fn = "plot_dv_pred"
    )
end

"""
    plot_dv_ipred(res::FitResult; dm, observable, style, rng, kwargs_subplot,
                  kwargs_layout, save_path, plot_path, return_panel) -> Makie.Figure

Goodness-of-fit scatter plot of observed values (DV) against individual
predictions (IPRED). IPRED is the model prediction evaluated at each
individual's empirical Bayes estimates of the random effects. A dashed identity
line is overlaid for reference.

A vector-valued observable gets one panel per component (free axes). Logarithmic axes
(`kwargs_subplot = (; xscale = log10, yscale = log10)`) are supported; non-positive
values are dropped with a warning.

# Keyword Arguments
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
- `observable`: name of the outcome variable, or `nothing` for the first.
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `rng::AbstractRNG = Xoshiro(42)`: random-number generator for the random effects of
  posterior-draw fits. A fresh generator with this fixed seed is created per call.
- `kwargs_subplot`: additional Makie `Axis` attributes (user values win, e.g. `limits`).
- `kwargs_layout`: additional Makie `Figure` attributes forwarded to the layout.
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
- `plot_path::Union{Nothing, String} = nothing`: alias for `save_path`.
- `return_panel::Bool = false`: return the panel instead of a `Figure`, for composing
  several diagnostics into one figure via `combine_plots` (scalar observables only).
"""
function plot_dv_ipred(
        res::FitResult;
        dm::Union{Nothing, DataModel} = nothing,
        observable = nothing,
        style::PlotStyle = PlotStyle(),
        rng::AbstractRNG = _default_rng(),
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing,
        return_panel::Bool = false
    )
    dm = _get_dm(res, dm)
    save_path = _resolve_plot_path(save_path, plot_path)
    obs_name = _get_observable(dm, observable)
    θ = get_params(res; scale = :untransformed)
    constants_re = _res_constants_re(res, NamedTuple())
    η_ebe = _default_random_effects(res, dm, constants_re, θ, rng, 1)

    panels_spec, n_marginals = _marginal_panels(dm, obs_name)
    return_panel && length(panels_spec) > 1 &&
        error("return_panel requires a scalar observable; :$(obs_name) has $(n_marginals) components.")
    specs = map(panels_spec) do (m, label)
        dv, ipred = _collect_ipred_series(
            dm, obs_name, θ, η_ebe;
            marginal = m, n_marginals = n_marginals, fn = "plot_dv_ipred"
        )
        isempty(dv) &&
            error("No non-missing observations found for observable :$(label).")
        (label, ipred, dv)
    end
    return _diag_gof_figure(
        specs, :identity;
        xlabel = "Individual prediction (IPRED)", ylabel = "Observed (DV)",
        style = style, kwargs_subplot = kwargs_subplot, kwargs_layout = kwargs_layout,
        save_path = save_path, return_panel = return_panel, fn = "plot_dv_ipred"
    )
end

"""
    plot_wres_pred(res::FitResult; dm, observable, style, kwargs_subplot,
                   kwargs_layout, save_path, plot_path, return_panel) -> Makie.Figure

Population residual diagnostic plot: weighted population residuals (WRES)
against population predictions (PRED). WRES is defined as
`(DV - PRED) / sigma`, where `sigma` is the standard deviation of the
observation distribution at the population-mean random effects. A horizontal
zero reference line is overlaid.

A vector-valued observable gets one panel per component (free axes). A logarithmic
x axis (`kwargs_subplot = (; xscale = log10)`) is supported; non-positive predictions are
dropped with a warning.

# Keyword Arguments
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
- `observable`: name of the outcome variable, or `nothing` for the first.
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `kwargs_subplot`: additional Makie `Axis` attributes (user values win, e.g. `limits`).
- `kwargs_layout`: additional Makie `Figure` attributes forwarded to the layout.
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
- `plot_path::Union{Nothing, String} = nothing`: alias for `save_path`.
- `return_panel::Bool = false`: return the panel instead of a `Figure`, for composing
  several diagnostics into one figure via `combine_plots` (scalar observables only).
"""
function plot_wres_pred(
        res::FitResult;
        dm::Union{Nothing, DataModel} = nothing,
        observable = nothing,
        style::PlotStyle = PlotStyle(),
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing,
        return_panel::Bool = false
    )
    dm = _get_dm(res, dm)
    save_path = _resolve_plot_path(save_path, plot_path)
    obs_name = _get_observable(dm, observable)
    θ = get_params(res; scale = :untransformed)
    η_pred = _pred_re_per_individual(dm, θ)

    panels_spec, n_marginals = _marginal_panels(dm, obs_name)
    return_panel && length(panels_spec) > 1 &&
        error("return_panel requires a scalar observable; :$(obs_name) has $(n_marginals) components.")
    specs = map(panels_spec) do (m, label)
        dv, pred, sigma = _collect_pred_series(
            dm, obs_name, θ, η_pred;
            marginal = m, n_marginals = n_marginals, fn = "plot_wres_pred"
        )
        isempty(dv) &&
            error("No non-missing observations found for observable :$(label).")
        (label, pred, (dv .- pred) ./ sigma)
    end
    return _diag_gof_figure(
        specs, :zero;
        xlabel = "Population prediction (PRED)", ylabel = "Weighted residual (WRES)",
        style = style, kwargs_subplot = kwargs_subplot, kwargs_layout = kwargs_layout,
        save_path = save_path, return_panel = return_panel, fn = "plot_wres_pred"
    )
end

"""
    plot_shrinkage(res::FitResult; dm, constants_re, threshold, bar_color, style,
                   kwargs_subplot, save_path, plot_path) -> Makie.Figure

Horizontal bar chart of eta shrinkage for all scalar random effects, with a dashed
vertical reference line at `threshold` (default 30 %, listed in the figure legend).

Bars are colored by severity: below `threshold` (green), between `threshold`
and 50 % (orange), above 50 % (red). Negative shrinkage (SD of the EBEs above
the fitted RE scale, common in small samples) is drawn as a bar extending left
of 0 and colored green. Shrinkage is computed via
[`compute_shrinkage`](@ref).

# Keyword Arguments
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
- `constants_re::NamedTuple = NamedTuple()`: random effects fixed at given values.
- `threshold::Real = 0.30`: reference line position (fraction, not percent).
- `bar_color::Union{Nothing, String} = nothing`: one colour for every bar instead of the
  severity colours.
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `kwargs_subplot`: additional Makie `Axis` attributes passed to the subplot.
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
- `plot_path::Union{Nothing, String} = nothing`: alias for `save_path`.
"""
function plot_shrinkage(
        res::FitResult;
        dm::Union{Nothing, DataModel} = nothing,
        constants_re::NamedTuple = NamedTuple(),
        threshold::Real = 0.3,
        bar_color::Union{Nothing, String} = nothing,
        style::PlotStyle = PlotStyle(),
        kwargs_subplot = NamedTuple(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing
    )
    save_path = _resolve_plot_path(save_path, plot_path)
    _check_unit_interval(threshold, "threshold")
    shrink_nt = compute_shrinkage(res; dm = dm, constants_re = constants_re)
    fig = _shrinkage_figure(shrink_nt; threshold = threshold, bar_color = bar_color, style = style, kwargs_subplot = kwargs_subplot)
    return _save_plot!(fig, save_path)
end

function _shrinkage_figure(shrink_nt::NamedTuple; threshold, bar_color, style, kwargs_subplot)
    isempty(shrink_nt) &&
        error("No shrinkage values computed; check that the model has scalar random effects.")

    re_labels = [string(k) for k in keys(shrink_nt)]
    values_pct = [Float64(getfield(v, :shrinkage)) * 100 for v in values(shrink_nt)]
    n = length(re_labels)

    bar_colors = if bar_color !== nothing
        fill(bar_color, n)
    else
        map(values_pct) do v
            v < threshold * 100 ? "#009E73" :
                v < 50.0 ? "#E69F00" : "#D55E00"
        end
    end
    pos = collect(1.0:n)
    bars = AoG.data(DataFrame(pos = pos, val = values_pct, col = Makie.to_color.(bar_colors))) *
        AoG.mapping(:pos, :val; color = :col => AoG.verbatim) *
        AoG.visual(Makie.BarPlot; direction = :x, width = 0.6)
    thr_label = "$(round(Int, threshold * 100))%"
    thr = AoG.data(DataFrame(x = [threshold * 100], series = [thr_label])) *
        AoG.mapping(:x; color = :series) *
        AoG.visual(Makie.VLines; linestyle = :dash, linewidth = style.line_width_secondary)

    xlim = (
        min(-2.0, minimum(values_pct) * 1.25 - 2.0),
        max(maximum(values_pct) * 1.25 + 5.0, threshold * 100 * 1.5),
    )
    post = Any[
        ax -> text!(
            ax, max.(values_pct, 0.0) .+ 1.5, pos;
            text = [string(round(v; digits = 1), "%") for v in values_pct],
            align = (:left, :center), fontsize = 9
        ),
    ]
    haskey(_nt(kwargs_subplot), :limits) ||
        push!(post, ax -> Makie.limits!(ax, xlim, (0.5, n + 0.5)))
    panel = AoGPanel(
        bars + thr;
        scales = (; Color = _series_palette([thr_label => style.color_reference])),
        axis = _aog_axis(
            style, kwargs_subplot;
            title = "ETA shrinkage", xlabel = "ETA shrinkage (%)", ylabel = "",
            yticks = (pos, re_labels),
        ),
        post = post
    )
    return combine_plots([panel]; ncols = 1, style = style)
end
