# Multivariate residual tables (Group 3 schema) store a `Vector{Float64}` cell per
# observation for `y`, `fitted` and every residual metric, whole-cell `missing` when
# undefined; `logscore` and `x` stay scalar. Expand every vector column in lockstep into
# one row per component so the scalar panels below plot one panel per component.
function _expand_residual_components(df::DataFrame, col::Symbol)
    # `logscore` stays a scalar joint score, so a logscore panel is left unexpanded.
    (hasproperty(df, col) && any(v -> v isa AbstractVector, df[!, col])) || return df
    vec_cols = [
        c for c in propertynames(df)
            if any(v -> v isa AbstractVector, df[!, c])
    ]
    isempty(vec_cols) && return df
    rows = NamedTuple[]
    for r in eachrow(df)
        # All vector cells of a row describe the same outcome components, so a length
        # mismatch means a `fitted_stat` returned the wrong shape; say so explicitly.
        n = 0
        ref = nothing
        for c in vec_cols
            r[c] isa AbstractVector || continue
            len = length(r[c])
            if ref === nothing
                n, ref = len, c
            elseif len != n
                throw(
                    ArgumentError(
                        "Residual column :$(c) has $(len) components for observable " *
                            "$(r.observable) (individual $(r.individual_idx)), but " *
                            ":$(ref) has $(n). All vector-valued residual columns of a " *
                            "row must agree in length."
                    )
                )
            end
        end
        base = NamedTuple(r)
        if n == 0
            push!(rows, base)
            continue
        end
        for m in 1:n
            vals = NamedTuple{Tuple(vec_cols)}(
                Tuple(r[c] isa AbstractVector ? r[c][m] : r[c] for c in vec_cols)
            )
            push!(
                rows,
                merge(
                    base,
                    (; observable = Symbol(_marginal_label(Symbol(r.observable), m))),
                    vals
                )
            )
        end
    end
    return DataFrame(rows)
end

# --- AlgebraOfGraphics helpers of the residual figures (prefix `_resid_`) --------------
#
# Every residual figure is a facet grid drawn with `_draw_facets` (one panel per
# observable, or per observable and individual for `plot_residuals`); reference lines are
# per-panel `post` hooks, histograms are exact uniform-bin `BarPlot` layers.

# Groups of the residual table in first-appearance order: individuals in data order,
# observables in model order (never AoG's lexicographic order).
_resid_groups(df::AbstractDataFrame, cols) = groupby(df, cols; sort = false)

# Residual rows reordered so individuals appear in the order of their first data row
# (`DataModel` sorts individuals by id, which is lexicographic for string ids: s1, s10,
# s11, ..., s2). Stable, so rows of one individual keep their order. Same policy as the
# core helper `_individuals_in_data_order`, applied to the residual table.
function _resid_in_data_order(df::DataFrame)
    (hasproperty(df, :row) && hasproperty(df, :individual_idx)) || return df
    first_row = Dict{Any, Int}()
    for (i, r) in zip(df.individual_idx, df.row)
        first_row[i] = min(get(first_row, i, typemax(Int)), Int(r))
    end
    perm = sortperm([first_row[i] for i in df.individual_idx]; alg = MergeSort)
    return df[perm, :]
end

# Finite `(x, y)` pairs of two residual columns. Non-finite values (e.g. the `Inf`
# logscore of an observation with zero probability) cannot be placed on an axis; they
# are dropped with a warning instead of producing infinite axis limits.
function _resid_xy(xcol, ycol, what::AbstractString)
    xs = Float64[]
    ys = Float64[]
    dropped = 0
    for (x, y) in zip(xcol, ycol)
        (ismissing(x) || ismissing(y)) && continue
        xf = Float64(x)
        yf = Float64(y)
        if isfinite(xf) && isfinite(yf)
            push!(xs, xf)
            push!(ys, yf)
        else
            dropped += 1
        end
    end
    dropped > 0 &&
        @warn "Dropped $(dropped) non-finite $(what) value(s) that cannot be drawn." maxlog = 1
    return xs, ys
end

_resid_vals(col, what::AbstractString) = last(_resid_xy(Iterators.repeated(0.0), col, what))

# Dashed horizontal reference line in one panel (`post` hook of `_draw_facets`).
_resid_hline(value::Real, style::PlotStyle) =
    ax -> hlines!(ax, Float64(value); color = style.color_dark, linestyle = :dash)

# Exact uniform-bin probability histogram of one panel as an AoG BarPlot layer (one layer
# per panel, because the bin width differs between panels). Returns the layer and the
# tallest bar.
function _resid_hist_layer(panel::AbstractString, vals, bins::Int, series::AbstractString, style::PlotStyle)
    h = _histogram_xy(vals; bins = bins, normalization = :probability)
    n = length(h.centers)
    t = DataFrame(
        panel = fill(String(panel), n), x = collect(Float64, h.centers),
        y = collect(Float64, h.heights), series = fill(String(series), n)
    )
    layer = AoG.data(t) * AoG.mapping(:x, :y; color = :series, layout = :panel) *
        AoG.visual(Makie.BarPlot; gap = 0, strokewidth = 0, width = h.width)
    return layer, maximum(h.heights)
end

# Figure returned when no panel has drawable residuals (one titled empty axis).
function _resid_empty(msg::AbstractString, style::PlotStyle, kwargs_subplot, kwargs_layout, save_path)
    panel = AoGPanel(nothing; axis = _aog_axis(style, kwargs_subplot; title = msg))
    fig = combine_plots([panel]; ncols = 1, style = style, _nt(kwargs_layout)...)
    return _save_plot!(fig, save_path)
end

function _plot_residuals_df(
        df::DataFrame;
        metric::Symbol = :quantile,
        x_label::String = "x",
        shared_x_axis::Bool = true,
        shared_y_axis::Bool = true,
        ncols::Int = DEFAULT_PLOT_COLS,
        style::PlotStyle = PlotStyle(),
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple(),
        save_path::Union{Nothing, String} = nothing,
        data_order::Bool = false
    )
    _check_positive_int(ncols, "ncols")
    metric = _validate_plot_metric(metric)
    col = _residual_metric_column(metric)
    df = _expand_residual_components(df, col)
    data_order && (df = _resid_in_data_order(df))
    label = _residual_metric_label(metric)
    ref = metric == :pit ? 0.5 : (metric == :logscore ? nothing : 0.0)

    pts = _tidy()
    panels = String[]
    post = Dict{String, Vector{Any}}()
    xlims = nothing
    ylims = nothing
    for g in _resid_groups(df, [:observable, :individual_idx])
        x, y = _resid_xy(g.x, g[!, col], label)
        isempty(x) && continue
        panel = string(g.observable[1], " | id=", g.id[1])
        push!(panels, panel)
        _push_xy!(pts, panel, x, y, "residual")
        ref === nothing || (post[panel] = Any[_resid_hline(ref, style)])
        xlims = _merge_limits(xlims, x)
        ylims = _merge_limits(ylims, y)
    end
    isempty(panels) &&
        return _resid_empty("No residual data to plot.", style, kwargs_subplot, kwargs_layout, save_path)

    layers = _layer(
        pts, AoG.mapping(:x, :y; layout = :panel),
        _scatter_visual(style; color = style.color_secondary)
    )
    fig = _draw_facets(
        layers; panels = panels, ncols = ncols, style = style,
        shared_x = shared_x_axis, shared_y = shared_y_axis,
        kwargs_subplot = kwargs_subplot, kwargs_layout = kwargs_layout,
        axis = (; xlabel = x_label, ylabel = label), legend = false, post = post,
        xlim = _shared_lim(shared_x_axis, xlims), ylim = _shared_lim(shared_y_axis, ylims),
    )
    return _save_plot!(fig, save_path)
end

"""
    plot_residuals(res::FitResult; dm, cache, residual, observables, individuals_idx,
                   obs_rows, x_axis_feature, shared_x_axis, shared_y_axis, ncols,
                   style, params, constants_re, cache_obs_dists, fitted_stat,
                   randomize_discrete, cdf_fallback_mc, ode_args, ode_kwargs,
                   mcmc_draws, mcmc_warmup, mcmc_quantiles, rng, save_path,
                   kwargs_subplot, kwargs_layout) -> Makie.Figure

    plot_residuals(dm::DataModel; ...) -> Makie.Figure

Plot residuals versus time (or another x-axis feature) for each individual.

One AlgebraOfGraphics facet per observable and individual (titled `"<observable> |
id=<id>"`; individuals in the order they first appear in the data, or in the order of
`individuals_idx` when given), with a dashed reference line at 0 (0.5 for `:pit`,
none for `:logscore`). Vector-valued observables get one facet per component.
Non-finite residuals (e.g. an `Inf` logscore) are dropped with a warning.

# Keyword Arguments
- `residual::Symbol = :quantile`: residual metric to plot. One of `:pit`, `:quantile`,
  `:raw`, `:pearson`, `:logscore`.
- `shared_x_axis::Bool = true`, `shared_y_axis::Bool = true`: share and link the axis
  across panels (padded common range; inner tick labels hidden).
- `ncols::Int = 3`: number of panel columns (panels wrap row-wise).
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `kwargs_subplot`: Makie `Axis` attributes for every panel (passed to
  AlgebraOfGraphics as `axis`; they win over the defaults).
- `kwargs_layout`: Makie `Figure` attributes (passed as `figure`; they win over the
  defaults).
- `save_path`/`plot_path`: file path to save the figure.
- All other arguments are forwarded to [`get_residuals`](@ref); see that function for
  descriptions.
"""
function plot_residuals(
        res::FitResult;
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
        rng::AbstractRNG = _default_rng(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing,
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple()
    )
    save_path = _resolve_plot_path(save_path, plot_path)
    metric = _validate_plot_metric(residual)
    df = get_residuals(
        res; dm = dm, cache = cache, observables = observables,
        individuals_idx = individuals_idx, obs_rows = obs_rows,
        x_axis_feature = x_axis_feature, params = params, constants_re = constants_re, cache_obs_dists = cache_obs_dists,
        residuals = [metric], fitted_stat = fitted_stat, randomize_discrete = randomize_discrete,
        cdf_fallback_mc = cdf_fallback_mc, ode_args = ode_args, ode_kwargs = ode_kwargs,
        mcmc_draws = mcmc_draws, mcmc_warmup = mcmc_warmup, mcmc_quantiles = mcmc_quantiles, rng = rng
    )
    xlabel = x_axis_feature === nothing ? "Time" : _axis_label(x_axis_feature)
    # all individuals: data order; an explicit `individuals_idx` keeps the caller's order
    return _plot_residuals_df(
        df; metric = metric, x_label = xlabel, shared_x_axis = shared_x_axis,
        shared_y_axis = shared_y_axis, ncols = ncols, style = style,
        kwargs_subplot = kwargs_subplot, kwargs_layout = kwargs_layout, save_path = save_path,
        data_order = individuals_idx === nothing
    )
end

function plot_residuals(
        dm::DataModel;
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
        rng::AbstractRNG = _default_rng(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing,
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple()
    )
    save_path = _resolve_plot_path(save_path, plot_path)
    metric = _validate_plot_metric(residual)
    df = get_residuals(
        dm; cache = cache, observables = observables,
        individuals_idx = individuals_idx, obs_rows = obs_rows,
        x_axis_feature = x_axis_feature, params = params, constants_re = constants_re, cache_obs_dists = cache_obs_dists,
        residuals = [metric], fitted_stat = fitted_stat, randomize_discrete = randomize_discrete,
        cdf_fallback_mc = cdf_fallback_mc, ode_args = ode_args, ode_kwargs = ode_kwargs,
        mcmc_draws = mcmc_draws, mcmc_warmup = mcmc_warmup, mcmc_quantiles = mcmc_quantiles, rng = rng
    )
    xlabel = x_axis_feature === nothing ? "Time" : _axis_label(x_axis_feature)
    # all individuals: data order; an explicit `individuals_idx` keeps the caller's order
    return _plot_residuals_df(
        df; metric = metric, x_label = xlabel, shared_x_axis = shared_x_axis,
        shared_y_axis = shared_y_axis, ncols = ncols, style = style,
        kwargs_subplot = kwargs_subplot, kwargs_layout = kwargs_layout, save_path = save_path,
        data_order = individuals_idx === nothing
    )
end

function _plot_residual_distribution_df(
        df::DataFrame;
        metric::Symbol = :quantile,
        shared_x_axis::Bool = true,
        shared_y_axis::Bool = true,
        ncols::Int = DEFAULT_PLOT_COLS,
        style::PlotStyle = PlotStyle(),
        bins::Int = 20,
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple(),
        save_path::Union{Nothing, String} = nothing
    )
    _check_positive_int(ncols, "ncols")
    _check_positive_int(bins, "bins")
    metric = _validate_plot_metric(metric)
    col = _residual_metric_column(metric)
    df = _expand_residual_components(df, col)
    label = _residual_metric_label(metric)

    hist_layers = Any[]
    curve = _tidy()
    panels = String[]
    post = Dict{String, Vector{Any}}()
    xlims = nothing
    ylims = nothing
    for g in _resid_groups(df, :observable)
        vals = _resid_vals(g[!, col], label)
        isempty(vals) && continue
        panel = string(g.observable[1])
        push!(panels, panel)
        layer, ymax = _resid_hist_layer(panel, vals, bins, "data", style)
        push!(hist_layers, layer)
        if metric == :quantile
            xs = range(minimum(vals), maximum(vals); length = 200)
            pdf_vals = pdf.(Normal(), xs)
            _push_xy!(curve, panel, xs, pdf_vals, "N(0,1)")
            ymax = max(ymax, maximum(pdf_vals))
        elseif metric == :pit
            post[panel] = Any[_resid_hline(1.0 / bins, style)]
        end
        xlims = _merge_limits(xlims, vals)
        # y range = histogram heights and the N(0,1) overlay, from 0.
        ylims = _merge_limits(ylims, (0.0, ymax))
    end
    isempty(panels) &&
        return _resid_empty("No residual data to plot.", style, kwargs_subplot, kwargs_layout, save_path)

    series = Pair{String, Any}["data" => style.color_secondary]
    metric == :quantile && push!(series, "N(0,1)" => style.color_dark)
    layers = _layers(
        reduce(+, hist_layers),
        _layer(
            curve, AoG.mapping(:x, :y; color = :series, layout = :panel),
            _line_visual(style; linestyle = :dash)
        ),
    )
    fig = _draw_facets(
        layers; panels = panels, ncols = ncols, style = style,
        shared_x = shared_x_axis, shared_y = shared_y_axis,
        kwargs_subplot = kwargs_subplot, kwargs_layout = kwargs_layout,
        axis = (; xlabel = label, ylabel = "Probability"),
        series = series, legend = metric == :quantile, post = post,
        xlim = _shared_lim(shared_x_axis, xlims), ylim = _shared_lim(shared_y_axis, ylims),
    )
    return _save_plot!(fig, save_path)
end

"""
    plot_residual_distribution(res::FitResult; dm, cache, residual, observables,
                               individuals_idx, obs_rows, x_axis_feature,
                               shared_x_axis, shared_y_axis, ncols, style, bins,
                               params, constants_re, cache_obs_dists, fitted_stat,
                               randomize_discrete, cdf_fallback_mc, ode_args,
                               ode_kwargs, mcmc_draws, mcmc_warmup, mcmc_quantiles,
                               rng, save_path, kwargs_subplot, kwargs_layout)
                               -> Makie.Figure

    plot_residual_distribution(dm::DataModel; ...) -> Makie.Figure

Plot the marginal distribution of residuals as probability histograms, one facet per
observable (component for vector-valued observables). `:quantile` adds the dashed N(0,1)
density and a figure legend; `:pit` adds a dashed reference line at the uniform bar height
`1 / bins`.

# Keyword Arguments
- `residual::Symbol = :quantile`: residual metric (`:pit`, `:quantile`, `:raw`,
  `:pearson`, `:logscore`).
- `bins::Int = 20`: number of equal-width histogram bins.
- `shared_x_axis::Bool = true`, `shared_y_axis::Bool = true`: share and link the axis
  across panels.
- `ncols::Int = 3`: number of panel columns (panels wrap row-wise).
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `kwargs_subplot`: Makie `Axis` attributes for every panel (passed to
  AlgebraOfGraphics as `axis`; they win over the defaults).
- `kwargs_layout`: Makie `Figure` attributes (passed as `figure`; they win over the
  defaults).
- `save_path`/`plot_path`: file path to save the figure.
- All other arguments are forwarded to [`get_residuals`](@ref).
"""
function plot_residual_distribution(
        res::FitResult;
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
        rng::AbstractRNG = _default_rng(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing,
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple()
    )
    save_path = _resolve_plot_path(save_path, plot_path)
    metric = _validate_plot_metric(residual)
    df = get_residuals(
        res; dm = dm, cache = cache, observables = observables,
        individuals_idx = individuals_idx, obs_rows = obs_rows,
        x_axis_feature = x_axis_feature, params = params, constants_re = constants_re, cache_obs_dists = cache_obs_dists,
        residuals = [metric], fitted_stat = fitted_stat, randomize_discrete = randomize_discrete,
        cdf_fallback_mc = cdf_fallback_mc, ode_args = ode_args, ode_kwargs = ode_kwargs,
        mcmc_draws = mcmc_draws, mcmc_warmup = mcmc_warmup, mcmc_quantiles = mcmc_quantiles, rng = rng
    )
    return _plot_residual_distribution_df(
        df; metric = metric, shared_x_axis = shared_x_axis,
        shared_y_axis = shared_y_axis, ncols = ncols, style = style, bins = bins,
        kwargs_subplot = kwargs_subplot, kwargs_layout = kwargs_layout, save_path = save_path
    )
end

function plot_residual_distribution(
        dm::DataModel;
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
        rng::AbstractRNG = _default_rng(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing,
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple()
    )
    save_path = _resolve_plot_path(save_path, plot_path)
    metric = _validate_plot_metric(residual)
    df = get_residuals(
        dm; cache = cache, observables = observables,
        individuals_idx = individuals_idx, obs_rows = obs_rows,
        x_axis_feature = x_axis_feature, params = params, constants_re = constants_re, cache_obs_dists = cache_obs_dists,
        residuals = [metric], fitted_stat = fitted_stat, randomize_discrete = randomize_discrete,
        cdf_fallback_mc = cdf_fallback_mc, ode_args = ode_args, ode_kwargs = ode_kwargs,
        mcmc_draws = mcmc_draws, mcmc_warmup = mcmc_warmup, mcmc_quantiles = mcmc_quantiles, rng = rng
    )
    return _plot_residual_distribution_df(
        df; metric = metric, shared_x_axis = shared_x_axis,
        shared_y_axis = shared_y_axis, ncols = ncols, style = style, bins = bins,
        kwargs_subplot = kwargs_subplot, kwargs_layout = kwargs_layout, save_path = save_path
    )
end

function _plot_residual_qq_df(
        df::DataFrame;
        metric::Symbol = :quantile,
        ncols::Int = DEFAULT_PLOT_COLS,
        style::PlotStyle = PlotStyle(),
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple(),
        save_path::Union{Nothing, String} = nothing
    )
    _check_positive_int(ncols, "ncols")
    metric = _validate_plot_metric(metric)
    col = _residual_metric_column(metric)
    df = _expand_residual_components(df, col)
    label = _residual_metric_label(metric)

    pts = _tidy()
    diag = _tidy()
    panels = String[]
    for g in _resid_groups(df, :observable)
        vals = sort(_resid_vals(g[!, col], label))
        length(vals) < 2 && continue
        n = length(vals)
        probs = ((1:n) .- 0.5) ./ n
        theo = metric == :pit ? collect(probs) : quantile.(Normal(), probs)
        panel = string(g.observable[1])
        push!(panels, panel)
        _push_xy!(pts, panel, theo, vals, "sample")
        lo = min(minimum(theo), minimum(vals))
        hi = max(maximum(theo), maximum(vals))
        _push_xy!(diag, panel, (lo, hi), (lo, hi), "reference")
    end
    isempty(panels) &&
        return _resid_empty("No residual data to plot.", style, kwargs_subplot, kwargs_layout, save_path)

    xy = AoG.mapping(:x, :y; layout = :panel)
    layers = _layers(
        _layer(diag, xy, _line_visual(style; color = style.color_dark, linestyle = :dash)),
        _layer(
            pts, xy,
            _scatter_visual(style; color = style.color_secondary, markersize = style.marker_size_small)
        ),
    )
    fig = _draw_facets(
        layers; panels = panels, ncols = ncols, style = style,
        shared_x = false, shared_y = false,
        kwargs_subplot = kwargs_subplot, kwargs_layout = kwargs_layout,
        axis = (;
            xlabel = metric == :pit ? "Theoretical Uniform Quantile" : "Theoretical Normal Quantile",
            ylabel = string("Sample ", label),
        ),
        legend = false,
    )
    return _save_plot!(fig, save_path)
end

"""
    plot_residual_qq(res::FitResult; dm, cache, residual, observables, individuals_idx,
                     obs_rows, x_axis_feature, ncols, style, params, constants_re,
                     cache_obs_dists, fitted_stat, randomize_discrete, cdf_fallback_mc,
                     ode_args, ode_kwargs, mcmc_draws, mcmc_warmup, mcmc_quantiles,
                     rng, save_path, kwargs_subplot, kwargs_layout) -> Makie.Figure

    plot_residual_qq(dm::DataModel; ...) -> Makie.Figure

Quantile-quantile plot of residuals against the theoretical distribution
(Uniform for `:pit`, Normal for other metrics), one facet per observable with a dashed
identity line. Axes are not linked (each panel has its own range).

# Keyword Arguments
- `residual::Symbol = :quantile`: residual metric (`:pit`, `:quantile`, `:raw`,
  `:pearson`, `:logscore`).
- `ncols::Int = 3`: number of panel columns (panels wrap row-wise).
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `kwargs_subplot`: Makie `Axis` attributes for every panel (passed to
  AlgebraOfGraphics as `axis`; they win over the defaults).
- `kwargs_layout`: Makie `Figure` attributes (passed as `figure`; they win over the
  defaults).
- `save_path`/`plot_path`: file path to save the figure.
- All other arguments are forwarded to [`get_residuals`](@ref).
"""
function plot_residual_qq(
        res::FitResult;
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
        rng::AbstractRNG = _default_rng(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing,
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple()
    )
    save_path = _resolve_plot_path(save_path, plot_path)
    metric = _validate_plot_metric(residual)
    df = get_residuals(
        res; dm = dm, cache = cache, observables = observables,
        individuals_idx = individuals_idx, obs_rows = obs_rows,
        x_axis_feature = x_axis_feature, params = params, constants_re = constants_re, cache_obs_dists = cache_obs_dists,
        residuals = [metric], fitted_stat = fitted_stat, randomize_discrete = randomize_discrete,
        cdf_fallback_mc = cdf_fallback_mc, ode_args = ode_args, ode_kwargs = ode_kwargs,
        mcmc_draws = mcmc_draws, mcmc_warmup = mcmc_warmup, mcmc_quantiles = mcmc_quantiles, rng = rng
    )
    return _plot_residual_qq_df(
        df; metric = metric, ncols = ncols, style = style,
        kwargs_subplot = kwargs_subplot, kwargs_layout = kwargs_layout, save_path = save_path
    )
end

function plot_residual_qq(
        dm::DataModel;
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
        rng::AbstractRNG = _default_rng(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing,
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple()
    )
    save_path = _resolve_plot_path(save_path, plot_path)
    metric = _validate_plot_metric(residual)
    df = get_residuals(
        dm; cache = cache, observables = observables,
        individuals_idx = individuals_idx, obs_rows = obs_rows,
        x_axis_feature = x_axis_feature, params = params, constants_re = constants_re, cache_obs_dists = cache_obs_dists,
        residuals = [metric], fitted_stat = fitted_stat, randomize_discrete = randomize_discrete,
        cdf_fallback_mc = cdf_fallback_mc, ode_args = ode_args, ode_kwargs = ode_kwargs,
        mcmc_draws = mcmc_draws, mcmc_warmup = mcmc_warmup, mcmc_quantiles = mcmc_quantiles, rng = rng
    )
    return _plot_residual_qq_df(
        df; metric = metric, ncols = ncols, style = style,
        kwargs_subplot = kwargs_subplot, kwargs_layout = kwargs_layout, save_path = save_path
    )
end

function _plot_residual_pit_df(
        df::DataFrame;
        show_hist::Bool = true,
        show_kde::Bool = false,
        show_qq::Bool = false,
        ncols::Int = DEFAULT_PLOT_COLS,
        style::PlotStyle = PlotStyle(),
        kde_bandwidth::Union{Nothing, Float64} = nothing,
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple(),
        save_path::Union{Nothing, String} = nothing
    )
    _check_positive_int(ncols, "ncols")
    (show_hist + show_kde + show_qq) == 0 &&
        error("plot_residual_pit has every display mode disabled; enable one of show_hist, show_kde or show_qq.")
    if (show_hist + show_kde + show_qq) > 1
        @warn "plot_residual_pit expects one plot type at a time; defaulting to histogram." show_hist = show_hist show_kde = show_kde show_qq = show_qq
        show_hist = true
        show_kde = false
        show_qq = false
    end
    kde_bandwidth === nothing || _check_positive(kde_bandwidth, "kde_bandwidth")
    df = _expand_residual_components(df, :pit)
    mode = show_hist ? "PIT Histogram" : (show_kde ? "PIT Density" : "PIT QQ")

    hist_layers = Any[]
    lines_df = _tidy()
    pts = _tidy()
    diag = _tidy()
    panels = String[]
    for g in _resid_groups(df, :observable)
        pits = _resid_vals(g.pit, "PIT")
        length(pits) < 2 && continue
        panel = string(g.observable[1], " | ", mode)
        push!(panels, panel)
        if show_hist
            layer, _ = _resid_hist_layer(panel, pits, 20, "data", style)
            push!(hist_layers, layer)
        elseif show_kde
            xk, yk = _kde_xy(pits; bandwidth = kde_bandwidth, what = "kde_bandwidth")
            _push_xy!(lines_df, panel, xk, yk, "data")
        else
            q = sort(pits)
            u = range(0.0, 1.0; length = length(q))
            _push_xy!(pts, panel, u, q, "data")
            _push_xy!(diag, panel, (0.0, 1.0), (0.0, 1.0), "reference")
        end
    end
    isempty(panels) &&
        return _resid_empty("No PIT data to plot.", style, kwargs_subplot, kwargs_layout, save_path)

    xy = AoG.mapping(:x, :y; layout = :panel)
    layers = _layers(
        isempty(hist_layers) ? nothing : reduce(+, hist_layers),
        _layer(lines_df, xy, _line_visual(style; color = style.color_secondary)),
        _layer(diag, xy, _line_visual(style; color = style.color_dark, linestyle = :dash)),
        _layer(pts, xy, _scatter_visual(style; color = style.color_secondary)),
    )
    ylabel = show_hist ? "Probability" : (show_kde ? "Density" : "Empirical PIT Quantile")
    xlabel = show_qq ? "Theoretical Uniform Quantile" : "PIT"
    # PIT values live on [0, 1]: one fixed, linked x range; y stays per panel.
    fig = _draw_facets(
        layers; panels = panels, ncols = ncols, style = style,
        shared_x = true, shared_y = false,
        kwargs_subplot = kwargs_subplot, kwargs_layout = kwargs_layout,
        axis = (; xlabel = xlabel, ylabel = ylabel),
        series = show_hist ? Pair{String, Any}["data" => style.color_secondary] : Pair{String, Any}[],
        legend = false, xlim = (0.0, 1.0),
    )
    return _save_plot!(fig, save_path)
end

"""
    plot_residual_pit(res::FitResult; dm, cache, observables, individuals_idx, obs_rows,
                      x_axis_feature, show_hist, show_kde, show_qq, ncols, style,
                      kde_bandwidth, params, constants_re, cache_obs_dists,
                      randomize_discrete, cdf_fallback_mc, ode_args, ode_kwargs,
                      mcmc_draws, mcmc_warmup, rng, save_path, kwargs_subplot,
                      kwargs_layout) -> Makie.Figure

    plot_residual_pit(dm::DataModel; ...) -> Makie.Figure

Plot the probability integral transform (PIT) values, one facet per observable, as a
histogram (20 bins), a kernel density estimate, or a uniform QQ plot. Uniform PIT values
indicate a well-calibrated model. The x axis is fixed to [0, 1] and linked.

Exactly one display mode is drawn: enabling none is an error, enabling several warns and
falls back to the histogram.

# Keyword Arguments
- `show_hist::Bool = true`: histogram of PIT values.
- `show_kde::Bool = false`: kernel density estimate of PIT values.
- `show_qq::Bool = false`: empirical PIT quantiles against uniform quantiles.
- `kde_bandwidth::Union{Nothing, Float64} = nothing`: positive KDE bandwidth, or
  `nothing` for automatic selection.
- `ncols::Int = 3`: number of panel columns (panels wrap row-wise).
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `kwargs_subplot`: Makie `Axis` attributes for every panel (passed to
  AlgebraOfGraphics as `axis`; they win over the defaults).
- `kwargs_layout`: Makie `Figure` attributes (passed as `figure`; they win over the
  defaults).
- `save_path`/`plot_path`: file path to save the figure.
- All other arguments are forwarded to [`get_residuals`](@ref).
"""
function plot_residual_pit(
        res::FitResult;
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
        rng::AbstractRNG = _default_rng(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing,
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple()
    )
    save_path = _resolve_plot_path(save_path, plot_path)
    df = get_residuals(
        res; dm = dm, cache = cache, observables = observables,
        individuals_idx = individuals_idx, obs_rows = obs_rows,
        x_axis_feature = x_axis_feature, params = params, constants_re = constants_re, cache_obs_dists = cache_obs_dists,
        residuals = [:pit], fitted_stat = mean, randomize_discrete = randomize_discrete,
        cdf_fallback_mc = cdf_fallback_mc, ode_args = ode_args, ode_kwargs = ode_kwargs,
        mcmc_draws = mcmc_draws, mcmc_warmup = mcmc_warmup, mcmc_quantiles = mcmc_quantiles, rng = rng
    )
    return _plot_residual_pit_df(
        df; show_hist = show_hist, show_kde = show_kde, show_qq = show_qq,
        ncols = ncols, style = style, kde_bandwidth = kde_bandwidth,
        kwargs_subplot = kwargs_subplot, kwargs_layout = kwargs_layout, save_path = save_path
    )
end

function plot_residual_pit(
        dm::DataModel;
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
        rng::AbstractRNG = _default_rng(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing,
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple()
    )
    save_path = _resolve_plot_path(save_path, plot_path)
    df = get_residuals(
        dm; cache = cache, observables = observables,
        individuals_idx = individuals_idx, obs_rows = obs_rows,
        x_axis_feature = x_axis_feature, params = params, constants_re = constants_re, cache_obs_dists = cache_obs_dists,
        residuals = [:pit], fitted_stat = mean, randomize_discrete = randomize_discrete,
        cdf_fallback_mc = cdf_fallback_mc, ode_args = ode_args, ode_kwargs = ode_kwargs,
        mcmc_draws = mcmc_draws, mcmc_warmup = mcmc_warmup, mcmc_quantiles = mcmc_quantiles, rng = rng
    )
    return _plot_residual_pit_df(
        df; show_hist = show_hist, show_kde = show_kde, show_qq = show_qq,
        ncols = ncols, style = style, kde_bandwidth = kde_bandwidth,
        kwargs_subplot = kwargs_subplot, kwargs_layout = kwargs_layout, save_path = save_path
    )
end

function _plot_residual_acf_df(
        df::DataFrame;
        metric::Symbol = :quantile,
        max_lag::Int = 5,
        ncols::Int = DEFAULT_PLOT_COLS,
        style::PlotStyle = PlotStyle(),
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple(),
        save_path::Union{Nothing, String} = nothing
    )
    max_lag >= 1 || error("max_lag must be >= 1.")
    _check_positive_int(ncols, "ncols")
    metric = _validate_plot_metric(metric)
    col = _residual_metric_column(metric)
    df = _expand_residual_components(df, col)
    label = _residual_metric_label(metric)

    acf_df = _tidy()
    panels = String[]
    post = Dict{String, Vector{Any}}()
    lags = collect(1:max_lag)
    for gobs in _resid_groups(df, :observable)
        acfs = Vector{Vector{Union{Missing, Float64}}}()
        for gi in _resid_groups(gobs, :individual_idx)
            order = sortperm(gi.obs_index)
            v = _resid_vals(gi[order, col], label)
            length(v) < 2 && continue
            push!(acfs, _acf_for_series(v, max_lag))
        end
        isempty(acfs) && continue
        # Mean ACF over individuals; a lag without any usable pair is a gap (NaN).
        acf_mean = map(lags) do lag
            vals_lag = Float64[a[lag] for a in acfs if !ismissing(a[lag])]
            isempty(vals_lag) ? NaN : mean(vals_lag)
        end
        panel = string(gobs.observable[1])
        push!(panels, panel)
        _push_xy!(acf_df, panel, lags, acf_mean, "acf")
        post[panel] = Any[_resid_hline(0.0, style)]
    end
    isempty(panels) &&
        return _resid_empty("No residual data to plot.", style, kwargs_subplot, kwargs_layout, save_path)

    layers = _layer(
        acf_df, AoG.mapping(:x, :y; layout = :panel),
        AoG.visual(
            Makie.ScatterLines; color = style.color_secondary,
            linewidth = style.line_width_primary, markersize = style.marker_size
        )
    )
    fig = _draw_facets(
        layers; panels = panels, ncols = ncols, style = style,
        shared_x = false, shared_y = false,
        kwargs_subplot = kwargs_subplot, kwargs_layout = kwargs_layout,
        axis = (; xlabel = "Lag", ylabel = string(label, " Autocorrelation")),
        legend = false, post = post,
    )
    return _save_plot!(fig, save_path)
end

"""
    plot_residual_acf(res::FitResult; dm, cache, residual, observables, individuals_idx,
                      obs_rows, x_axis_feature, max_lag, ncols, style, params,
                      constants_re, cache_obs_dists, fitted_stat, randomize_discrete,
                      cdf_fallback_mc, ode_args, ode_kwargs, mcmc_draws, mcmc_warmup,
                      mcmc_quantiles, rng, save_path, kwargs_subplot, kwargs_layout)
                      -> Makie.Figure

    plot_residual_acf(dm::DataModel; ...) -> Makie.Figure

Plot the autocorrelation function (ACF) of residuals across time lags, one facet per
observable: the per-individual ACF (observations in data order) averaged over
individuals, with a dashed reference line at 0. Lags without a usable pair are gaps.

# Keyword Arguments
- `residual::Symbol = :quantile`: residual metric (`:pit`, `:quantile`, `:raw`,
  `:pearson`, `:logscore`).
- `max_lag::Int = 5`: maximum lag to compute and display (`>= 1`).
- `ncols::Int = 3`: number of panel columns (panels wrap row-wise).
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `kwargs_subplot`: Makie `Axis` attributes for every panel (passed to
  AlgebraOfGraphics as `axis`; they win over the defaults).
- `kwargs_layout`: Makie `Figure` attributes (passed as `figure`; they win over the
  defaults).
- `save_path`/`plot_path`: file path to save the figure.
- All other arguments are forwarded to [`get_residuals`](@ref).
"""
function plot_residual_acf(
        res::FitResult;
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
        rng::AbstractRNG = _default_rng(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing,
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple()
    )
    save_path = _resolve_plot_path(save_path, plot_path)
    metric = _validate_plot_metric(residual)
    df = get_residuals(
        res; dm = dm, cache = cache, observables = observables,
        individuals_idx = individuals_idx, obs_rows = obs_rows,
        x_axis_feature = x_axis_feature, params = params, constants_re = constants_re, cache_obs_dists = cache_obs_dists,
        residuals = [metric], fitted_stat = fitted_stat, randomize_discrete = randomize_discrete,
        cdf_fallback_mc = cdf_fallback_mc, ode_args = ode_args, ode_kwargs = ode_kwargs,
        mcmc_draws = mcmc_draws, mcmc_warmup = mcmc_warmup, mcmc_quantiles = mcmc_quantiles, rng = rng
    )
    return _plot_residual_acf_df(
        df; metric = metric, max_lag = max_lag, ncols = ncols, style = style,
        kwargs_subplot = kwargs_subplot, kwargs_layout = kwargs_layout, save_path = save_path
    )
end

function plot_residual_acf(
        dm::DataModel;
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
        rng::AbstractRNG = _default_rng(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing,
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple()
    )
    save_path = _resolve_plot_path(save_path, plot_path)
    metric = _validate_plot_metric(residual)
    df = get_residuals(
        dm; cache = cache, observables = observables,
        individuals_idx = individuals_idx, obs_rows = obs_rows,
        x_axis_feature = x_axis_feature, params = params, constants_re = constants_re, cache_obs_dists = cache_obs_dists,
        residuals = [metric], fitted_stat = fitted_stat, randomize_discrete = randomize_discrete,
        cdf_fallback_mc = cdf_fallback_mc, ode_args = ode_args, ode_kwargs = ode_kwargs,
        mcmc_draws = mcmc_draws, mcmc_warmup = mcmc_warmup, mcmc_quantiles = mcmc_quantiles, rng = rng
    )
    return _plot_residual_acf_df(
        df; metric = metric, max_lag = max_lag, ncols = ncols, style = style,
        kwargs_subplot = kwargs_subplot, kwargs_layout = kwargs_layout, save_path = save_path
    )
end
