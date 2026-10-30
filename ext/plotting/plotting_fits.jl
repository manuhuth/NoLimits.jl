# plot_data / plot_fits: reference implementations of the AlgebraOfGraphics port.

# Fit/data/diagnostic plotting on the Makie panel layer (see plotting.jl). Ported 1:1
# from the Plots.jl extension; Plots draw calls translated to panel-layer helpers.

# Panel label of individual `i` ("ID: 3"): the facet title and the `:panel` value.
_individual_panel(dm::DataModel, i::Integer) = string(get_primary_id(dm), ": ", _individual_id(dm, i))

# Padded shared limits (`nothing` when the axis is not shared or nothing is finite).
_shared_lim(shared::Bool, lims) = shared && lims !== nothing ? _pad_limits(lims[1], lims[2]) : nothing

function _plot_data_dm(
        dm::DataModel;
        x_axis_feature::Union{Symbol, Nothing} = nothing,
        individuals_idx = nothing,
        shared_x_axis::Bool = true,
        shared_y_axis::Bool = true,
        ncols::Int = DEFAULT_PLOT_COLS,
        style::PlotStyle = PlotStyle(),
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple(),
        save_path::Union{Nothing, String} = nothing,
        marginal_idx::Union{Nothing, Int} = nothing
    )
    _check_positive_int(ncols, "ncols")
    obs_name = _get_observable(dm, nothing)
    inds = _resolve_individuals(dm, individuals_idx; default_all = true)
    (is_mv, n_marginals) = _obs_multivariate_info(dm, obs_name)
    marginal_colors = _marginal_colors(n_marginals, style)

    obs_df = _tidy()
    panels = String[]
    xlims = nothing
    ylims = nothing
    for i in inds
        ind = get_individuals(dm)[i]
        obs_rows = get_obs_rows(get_row_groups(dm))[i]
        x = _get_x_values(dm, ind, obs_rows, x_axis_feature)
        y = getfield(get_obs(get_series(ind)), obs_name)
        panel = _individual_panel(dm, i)
        push!(panels, panel)
        if is_mv
            ms = marginal_idx === nothing ? (1:n_marginals) : (marginal_idx:marginal_idx)
            xs, ys = _collect_multivariate_series(x, y, n_marginals)
            for m in ms
                _push_xy!(obs_df, panel, xs[m], ys[m], _marginal_label(obs_name, m))
                xlims = _merge_limits(xlims, xs[m])
                ylims = _merge_limits(ylims, ys[m])
            end
        else
            xs, ys = _collect_scalar_series(x, y)
            _push_xy!(obs_df, panel, xs, ys, "data")
            xlims = _merge_limits(xlims, xs)
            ylims = _merge_limits(ylims, ys)
        end
    end

    # Scalar data carry no legend (one series); vector observables get one entry per
    # plotted marginal.
    series = is_mv ?
        [
            _marginal_label(obs_name, m) => marginal_colors[m]
                for m in (marginal_idx === nothing ? (1:n_marginals) : (marginal_idx:marginal_idx))
        ] :
        ["data" => style.color_primary]
    layers = _layer(
        obs_df, AoG.mapping(:x, :y; color = :series, layout = :panel), _scatter_visual(style)
    )
    fig = _draw_facets(
        layers; panels = panels, ncols = ncols, style = style,
        shared_x = shared_x_axis, shared_y = shared_y_axis,
        kwargs_subplot = kwargs_subplot, kwargs_layout = kwargs_layout,
        axis = (;
            xlabel = x_axis_feature === nothing ? "Time" : _axis_label(x_axis_feature),
            ylabel = _axis_label(obs_name),
        ),
        series = series, legend = is_mv,
        xlim = _shared_lim(shared_x_axis, xlims), ylim = _shared_lim(shared_y_axis, ylims),
    )
    return _save_plot!(fig, save_path)
end

"""
    plot_data(res::FitResult; dm, x_axis_feature, individuals_idx, shared_x_axis,
              shared_y_axis, ncols, style, kwargs_subplot, kwargs_layout,
              save_path, plot_path, marginal_layout) -> Makie.Figure | Vector{Makie.Figure}

    plot_data(dm::DataModel; x_axis_feature, individuals_idx, shared_x_axis,
              shared_y_axis, ncols, style, kwargs_subplot, kwargs_layout,
              save_path, plot_path, marginal_layout) -> Makie.Figure | Vector{Makie.Figure}

Plot raw observed data for each individual as a multi-panel figure.

# Keyword Arguments
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
- `x_axis_feature::Union{Symbol, Nothing} = nothing`: covariate to use as the x-axis;
  defaults to the time column.
- `individuals_idx`: indices or IDs of individuals to include, or `nothing` for all.
- `shared_x_axis::Bool = true`: share and link the x axis across panels; inner tick labels
  are hidden and one x label spans the grid.
- `shared_y_axis::Bool = true`: share and link the y axis across panels (same treatment).
- `ncols::Int = 3`: number of subplot columns.
- `marginal_layout::Symbol = :single`: `:single` keeps one figure with every marginal overlaid per individual; `:vector` returns a figure per marginal (only valid for vector-valued observables and requires `save_path`/`plot_path` to be `nothing`).
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `kwargs_subplot`: additional Makie `Axis` attributes forwarded to each subplot (passed to
  AlgebraOfGraphics as `axis`; they win over the defaults).
- `kwargs_layout`: additional Makie `Figure` attributes forwarded to the combined layout
  (passed to AlgebraOfGraphics as `figure`; they win over the defaults).
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot (ignored for `:vector` mode).

Panels are AlgebraOfGraphics facets titled by individual. Vector-valued observables get one
figure-level legend entry per component.
"""
function plot_data(
        res::FitResult;
        dm::Union{Nothing, DataModel} = nothing,
        x_axis_feature::Union{Symbol, Nothing} = nothing,
        individuals_idx = nothing,
        shared_x_axis::Bool = true,
        shared_y_axis::Bool = true,
        ncols::Int = DEFAULT_PLOT_COLS,
        style::PlotStyle = PlotStyle(),
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing,
        marginal_layout::Symbol = :single
    )
    dm = _get_dm(res, dm)
    obs_name = _get_observable(dm, nothing)
    marginal_layout in (:single, :vector) ||
        error("marginal_layout must be :single or :vector.")
    (is_mv, n_marginals) = _obs_multivariate_info(dm, obs_name)
    if is_mv && marginal_layout == :vector
        (save_path === nothing && plot_path === nothing) ||
            error("save_path/plot_path are not supported when returning multiple marginal figures.")
        plots = Vector{Figure}(undef, n_marginals)
        for m in 1:n_marginals
            plots[m] = _plot_data_dm(
                dm;
                x_axis_feature = x_axis_feature,
                individuals_idx = individuals_idx,
                shared_x_axis = shared_x_axis,
                shared_y_axis = shared_y_axis,
                ncols = ncols,
                style = style,
                kwargs_subplot = kwargs_subplot,
                kwargs_layout = kwargs_layout,
                save_path = nothing,
                marginal_idx = m
            )
        end
        return plots
    end
    save_path = _resolve_plot_path(save_path, plot_path)
    return _plot_data_dm(
        dm;
        x_axis_feature = x_axis_feature,
        individuals_idx = individuals_idx,
        shared_x_axis = shared_x_axis,
        shared_y_axis = shared_y_axis,
        ncols = ncols,
        style = style,
        kwargs_subplot = kwargs_subplot,
        kwargs_layout = kwargs_layout,
        save_path = save_path
    )
end

function plot_data(
        dm::DataModel;
        x_axis_feature::Union{Symbol, Nothing} = nothing,
        individuals_idx = nothing,
        shared_x_axis::Bool = true,
        shared_y_axis::Bool = true,
        ncols::Int = DEFAULT_PLOT_COLS,
        style::PlotStyle = PlotStyle(),
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing,
        marginal_layout::Symbol = :single
    )
    obs_name = _get_observable(dm, nothing)
    marginal_layout in (:single, :vector) ||
        error("marginal_layout must be :single or :vector.")
    (is_mv, n_marginals) = _obs_multivariate_info(dm, obs_name)
    if is_mv && marginal_layout == :vector
        (save_path === nothing && plot_path === nothing) ||
            error("save_path/plot_path are not supported when returning multiple marginal figures.")
        plots = Vector{Figure}(undef, n_marginals)
        for m in 1:n_marginals
            plots[m] = _plot_data_dm(
                dm;
                x_axis_feature = x_axis_feature,
                individuals_idx = individuals_idx,
                shared_x_axis = shared_x_axis,
                shared_y_axis = shared_y_axis,
                ncols = ncols,
                style = style,
                kwargs_subplot = kwargs_subplot,
                kwargs_layout = kwargs_layout,
                save_path = nothing,
                marginal_idx = m
            )
        end
        return plots
    end
    save_path = _resolve_plot_path(save_path, plot_path)
    return _plot_data_dm(
        dm;
        x_axis_feature = x_axis_feature,
        individuals_idx = individuals_idx,
        shared_x_axis = shared_x_axis,
        shared_y_axis = shared_y_axis,
        ncols = ncols,
        style = style,
        kwargs_subplot = kwargs_subplot,
        kwargs_layout = kwargs_layout,
        save_path = save_path
    )
end

# Store `plot_func(dist)` into the (draw, point, component) prediction array.
function _store_stat!(preds, d, j, dist, plot_func, is_mv::Bool, n_marginals::Int)
    v = _stat_from_dist(dist, plot_func)
    if is_mv
        for m in 1:n_marginals
            preds[d, j, m] = _float_if_real(v[m])
        end
    else
        preds[d, j, 1] = _float_if_real(v)
    end
    return preds
end

# Component `m` of every stored distribution (`m === nothing` keeps them as they are).
function _component_dists(dists_by_draw, obs_name::Symbol, m, n_marginals::Int)
    m === nothing && return dists_by_draw
    return [
        [
                _marginal_obs_dist(d, obs_name, m, n_marginals; fn = "plot_fits")
                for d in draw
            ] for draw in dists_by_draw
    ]
end

# Transparent-to-component-color ramp so per-component density bands can overlay.
function _density_colormap(m, marginal_colors)
    m === nothing && return (:viridis, 0.5)
    c = Makie.to_color(marginal_colors[m])
    return Makie.cgrad([Makie.RGBAf(c.r, c.g, c.b, 0.0), Makie.RGBAf(c.r, c.g, c.b, 0.6)])
end

# Predictive-density overlay drawn into one facet axis `ax` (plain Makie, a `post` hook of
# `_draw_facets`): a heatmap of the draw-averaged density, or a PMF scatter for discrete
# outcomes. Drawn behind the AoG layers (z = -1). Each heatmap/scatter gets its own
# automatic colour range, i.e. the density is normalised per panel (and, for PMFs, per
# time point) exactly as before the AoG port; an AoG `Color` scale would instead share one
# range across the figure.
function _plot_density_band!(ax, x_density, dists_by_draw, colormap, style::PlotStyle)
    n_draws = length(dists_by_draw)
    d0 = dists_by_draw[1][1]
    # Bernoulli needs no overlay; the fit line already represents p(y=1).
    _is_bernoulli(d0) && return ax
    if _is_discrete(d0)
        for j in eachindex(x_density)
            grid = _density_grid_discrete(dists_by_draw[1][j], 0.995)
            grid === nothing && continue
            probs = zeros(Float64, length(grid.vals))
            for d in 1:n_draws
                probs .+= pdf.(Ref(dists_by_draw[d][j]), grid.vals)
            end
            probs ./= n_draws
            sc = scatter!(
                ax, fill(Float64(x_density[j]), length(grid.vals)), Float64.(grid.vals);
                color = probs, colormap = :viridis, marker = :xcross,
                markersize = style.marker_size_pmf,
                strokewidth = style.marker_stroke_width_pmf
            )
            Makie.translate!(sc, 0, 0, -1)
        end
        return ax
    end
    grid = _density_grid_continuous(dists_by_draw[1], 0.995, 100)
    grid === nothing && return ax
    z = zeros(length(grid.y), length(x_density))
    for d in 1:n_draws, j in eachindex(x_density)
        z[:, j] .+= pdf.(Ref(dists_by_draw[d][j]), grid.y)
    end
    z ./= n_draws
    # z is (ny, nx); Makie heatmap wants (nx, ny) -> permutedims.
    hm = heatmap!(ax, Float64.(x_density), grid.y, permutedims(z); colormap = colormap)
    Makie.translate!(hm, 0, 0, -1)
    return ax
end

# Quantile label without a trailing ".0" (5.0 -> "5", 2.5 -> "2.5").
_fmt_q(q) = isinteger(q) ? string(Int(q)) : string(q)

"""
    plot_fits(res::FitResult; dm, plot_density, plot_func, plot_data_points, observable,
              individuals_idx, x_axis_feature, shared_x_axis, shared_y_axis, ncols,
              style, kwargs_subplot, kwargs_layout, save_path, cache, params,
              constants_re, cache_obs_dists, plot_mcmc_quantiles, mcmc_quantiles,
              mcmc_quantiles_alpha, mcmc_draws, mcmc_warmup, rng) -> Makie.Figure

    plot_fits(dm::DataModel; params, constants_re, observable, individuals_idx,
              x_axis_feature, shared_x_axis, shared_y_axis, ncols, plot_data_points,
              style, kwargs_subplot, kwargs_layout, save_path, cache) -> Makie.Figure

Plot model predictions against observed data for each individual as a multi-panel figure.

# Keyword Arguments
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
- `plot_density::Bool = false`: overlay the predictive distribution density. For
  vector-valued observables one band per component marginal is drawn, in the component's
  color (restricted to `marginal_idx` when given).
- `plot_func = mean`: function applied to the predictive distribution to obtain the
  prediction line (e.g. `mean`, `median`).
- `plot_data_points::Bool = true`: overlay the observed data points.
- `observable`: name of the outcome variable to plot, or `nothing` to use the first.
- `individuals_idx`: indices or IDs of individuals to include, or `nothing` for all.
- `x_axis_feature::Union{Nothing, Symbol} = nothing`: covariate for the x-axis.
- `shared_x_axis::Bool = true`, `shared_y_axis::Bool = true`: share and link the axes
  across panels; inner tick labels are hidden and one axis label spans the grid.
- `ncols::Int = 3`: number of subplot columns.
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `kwargs_subplot`: additional Makie `Axis` attributes forwarded to each subplot (passed to
  AlgebraOfGraphics as `axis`; they win over the defaults).
- `kwargs_layout`: additional Makie `Figure` attributes forwarded to the combined layout
  (passed to AlgebraOfGraphics as `figure`; they win over the defaults).
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
- `rng::AbstractRNG = Xoshiro(42)`: random-number generator. A fresh generator with
  this fixed seed is created per call, so repeated calls draw the same values.

Panels are AlgebraOfGraphics facets titled by individual, with one figure-level legend
(`data`, `fit`, posterior quantiles, or one entry per component of a vector-valued
observable). Density overlays are normalised per panel.
"""
function plot_fits(
        res::FitResult;
        dm::Union{Nothing, DataModel} = nothing,
        plot_density::Bool = false,
        plot_func = mean,
        plot_data_points::Bool = true,
        observable = nothing,
        individuals_idx = nothing,
        x_axis_feature::Union{Nothing, Symbol} = nothing,
        shared_x_axis::Bool = true,
        shared_y_axis::Bool = true,
        ncols::Int = DEFAULT_PLOT_COLS,
        marginal_layout::Symbol = :single,
        style::PlotStyle = PlotStyle(),
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing,
        cache::Union{Nothing, PlotCache} = nothing,
        params::NamedTuple = NamedTuple(),
        constants_re::NamedTuple = NamedTuple(),
        cache_obs_dists::Bool = false,
        plot_mcmc_quantiles::Bool = false,
        mcmc_quantiles::Vector{<:Real} = [5, 95],
        mcmc_quantiles_alpha::Float64 = 0.8,
        mcmc_draws::Int = 1000,
        mcmc_warmup::Union{Nothing, Int} = nothing,
        marginal_idx::Union{Nothing, Int} = nothing,
        rng::AbstractRNG = _default_rng()
    )
    dm = _get_dm(res, dm)
    save_path = _resolve_plot_path(save_path, plot_path)
    _check_unit_interval(mcmc_quantiles_alpha, "mcmc_quantiles_alpha")
    constants_re_use = _res_constants_re(res, constants_re)
    obs_name = _get_observable(dm, observable)
    (is_mv, detected_n_marginals) = _obs_multivariate_info(dm, obs_name)
    n_marginals = is_mv ? detected_n_marginals : 1
    marginal_layout in (:single, :vector) ||
        error("marginal_layout must be :single or :vector.")
    marginal_idx !== nothing && (!is_mv) &&
        error("marginal_idx only valid when the observable is multivariate.")
    if marginal_idx !== nothing
        (1 <= marginal_idx <= n_marginals) ||
            error("marginal_idx must be between 1 and $(n_marginals).")
    end
    inds = _resolve_individuals(dm, individuals_idx; default_all = true)

    if cache === nothing
        cache = build_plot_cache(
            res; dm = dm, params = params, constants_re = constants_re_use,
            cache_obs_dists = cache_obs_dists, mcmc_draws = mcmc_draws,
            mcmc_warmup = mcmc_warmup, rng = rng
        )
    end

    if is_mv && marginal_layout == :vector && marginal_idx === nothing
        (save_path === nothing && plot_path === nothing) ||
            error("save_path/plot_path are not supported when returning multiple marginal figures.")
        plots_vector = Vector{Figure}(undef, n_marginals)
        for m in 1:n_marginals
            plots_vector[m] = plot_fits(
                res;
                dm = dm,
                plot_density = plot_density,
                plot_func = plot_func,
                plot_data_points = plot_data_points,
                observable = observable,
                individuals_idx = individuals_idx,
                x_axis_feature = x_axis_feature,
                shared_x_axis = shared_x_axis,
                shared_y_axis = shared_y_axis,
                ncols = ncols,
                marginal_layout = :single,
                style = style,
                kwargs_subplot = kwargs_subplot,
                kwargs_layout = kwargs_layout,
                save_path = nothing,
                plot_path = nothing,
                cache = cache,
                params = params,
                constants_re = constants_re,
                cache_obs_dists = cache_obs_dists,
                plot_mcmc_quantiles = plot_mcmc_quantiles,
                mcmc_quantiles = mcmc_quantiles,
                mcmc_quantiles_alpha = mcmc_quantiles_alpha,
                mcmc_draws = mcmc_draws,
                mcmc_warmup = mcmc_warmup,
                marginal_idx = m,
                rng = rng
            )
        end
        return plots_vector
    end

    _check_positive_int(ncols, "ncols")
    panels = String[]
    obs_df = _tidy()      # observed points
    fit_df = _tidy()      # fit / posterior-mean curves
    q_df = _tidy()        # posterior quantile curves (dashed)
    post = Dict{String, Vector{Any}}()   # per-panel density overlays (plain Makie)
    xlims = nothing
    ylims = nothing
    marginal_colors = _marginal_colors(max(n_marginals, 1), style)

    is_mcmc = _is_posterior_draw_fit(res)
    θ_draws = nothing
    η_draws = nothing
    if is_mcmc
        res = _with_posterior_warmup(res, mcmc_warmup)
        θ_draws, η_draws, _ = _posterior_drawn_params(
            res, dm, constants_re_use, params, mcmc_draws, rng
        )
    end

    if plot_density && plot_mcmc_quantiles
        @warn "plot_mcmc_quantiles ignored because plot_density=true."
        plot_mcmc_quantiles = false
    end
    if plot_mcmc_quantiles
        mcmc_quantiles = sort(Float64.(collect(mcmc_quantiles)))
        (length(mcmc_quantiles) >= 2 && all(0 .<= mcmc_quantiles .<= 100)) ||
            error("mcmc_quantiles must be in [0,100] with length >= 2.")
    end
    # Component indices to draw; `nothing` marks the scalar (single-curve) case.
    margin_range = is_mv ?
        (marginal_idx === nothing ? collect(1:n_marginals) : [marginal_idx]) :
        [nothing]
    _series_label(m) = m === nothing ? "fit" : _marginal_label(obs_name, m)
    _qlabel(m, q) = m === nothing ? "$(_fmt_q(q))%" : "$(_marginal_label(obs_name, m)) $(_fmt_q(q))%"

    for i in inds
        ind = get_individuals(dm)[i]
        obs_rows = get_obs_rows(get_row_groups(dm))[i]
        use_dense = get_de(get_model(dm)) !== nothing &&
            _can_dense_plot(dm) &&
            (x_axis_feature === nothing || x_axis_feature == get_time_col(dm))
        x_obs = _get_x_values(dm, ind, obs_rows, x_axis_feature)
        x_fit = use_dense ? _dense_time_grid(ind) : x_obs
        x_density = use_dense ? x_fit : x_obs
        y_obs = getfield(get_obs(get_series(ind)), obs_name)
        x_obs_plot, y_obs_plot = is_mv ? (nothing, nothing) :
            _collect_scalar_series(x_obs, y_obs)
        panel = _individual_panel(dm, i)
        push!(panels, panel)
        xs_per_margin = nothing
        ys_per_margin = nothing
        if is_mv
            xs_per_margin, ys_per_margin = _collect_multivariate_series(
                x_obs, y_obs, n_marginals
            )
        end
        if plot_data_points
            if is_mv
                for m in margin_range
                    _push_xy!(obs_df, panel, xs_per_margin[m], ys_per_margin[m], _series_label(m))
                end
            else
                _push_xy!(obs_df, panel, x_obs_plot, y_obs_plot, "data")
            end
        end

        if is_mcmc
            n_draws = length(θ_draws)
            n_points = length(use_dense ? x_fit : obs_rows)
            preds = zeros(Float64, n_draws, n_points, is_mv ? n_marginals : 1)
            dists_for_density = plot_density ?
                Vector{Vector{Distribution}}(undef, n_draws) : nothing
            rowwise_re = _needs_rowwise_random_effects(dm, i; obs_only = true)
            for d in 1:n_draws
                θ = θ_draws[d]
                η_ind = η_draws[d][i]
                sol_accessors = nothing
                compiled = nothing
                if get_de(get_model(dm)) !== nothing
                    sol, compiled = _solve_dense_individual(dm, ind, θ, η_ind)
                    sol_accessors = _sol_accessors_with_crossings(
                        get_model(dm), sol, compiled, θ, η_ind, get_const_cov(ind)
                    )
                end
                if use_dense
                    dists = plot_density ? Vector{Distribution}(undef, length(x_fit)) :
                        Distribution[]
                    for (j, t) in enumerate(x_fit)
                        vary = (t = t,)
                        obs = calculate_formulas_obs(
                            get_model(dm), θ, η_ind, get_const_cov(ind), vary, sol_accessors
                        )
                        dist = getproperty(obs, obs_name)
                        _store_stat!(preds, d, j, dist, plot_func, is_mv, n_marginals)
                        if plot_density
                            dists[j] = dist
                        end
                    end
                else
                    y_obs_series_mcmc = getfield(get_obs(get_series(ind)), obs_name)
                    hmm_priors_draw = Dict{Symbol, Any}()
                    dists = Vector{Distribution}(undef, length(obs_rows))
                    for (j, row) in enumerate(obs_rows)
                        vary = _varying_at(dm, ind, j, row)
                        η_row = _row_random_effects_at(
                            dm, i, j, η_ind, rowwise_re; obs_only = true
                        )
                        obs = sol_accessors === nothing ?
                            calculate_formulas_obs(
                                get_model(dm), θ, η_row, get_const_cov(ind), vary
                            ) :
                            calculate_formulas_obs(
                                get_model(dm), θ, η_row, get_const_cov(ind), vary, sol_accessors
                            )
                        dist = _apply_hmm_filter!(
                            hmm_priors_draw, obs_name,
                            getproperty(obs, obs_name), y_obs_series_mcmc[j]
                        )
                        dists[j] = dist
                        _store_stat!(preds, d, j, dist, plot_func, is_mv, n_marginals)
                    end
                end
                if plot_density
                    dists_for_density[d] = dists
                end
            end
            for m in margin_range
                slice = preds[:, :, m === nothing ? 1 : m]
                mean_curve = vec(mean(slice, dims = 1))
                _push_xy!(fit_df, panel, x_fit, mean_curve, _series_label(m))
                ylims = _merge_limits(ylims, mean_curve)

                if plot_mcmc_quantiles
                    for q in mcmc_quantiles
                        qvals = vec(
                            mapslices(x -> quantile(vec(x), q / 100), slice; dims = 1)
                        )
                        _push_xy!(q_df, panel, x_fit, qvals, _qlabel(m, q))
                        ylims = _merge_limits(ylims, qvals)
                    end
                end
            end

            if plot_density
                for m in margin_range
                    dd = _component_dists(dists_for_density, obs_name, m, n_marginals)
                    cm = _density_colormap(m, marginal_colors)
                    push!(get!(post, panel, Any[]), ax -> _plot_density_band!(ax, x_density, dd, cm, style))
                end
            end
        else
            θ = cache.params
            η_ind = cache.random_effects[i]
            rowwise_re = _needs_rowwise_random_effects(dm, i; obs_only = true)
            sol_accessors = nothing
            if get_de(get_model(dm)) !== nothing
                sol = cache.sols[i]
                compiled = get_de_compiler(get_de(get_model(dm)))(
                    (;
                        fixed_effects = θ,
                        random_effects = η_ind,
                        constant_covariates = get_const_cov(ind),
                        varying_covariates = merge(
                            (t = get_vary(get_series(ind)).t[1],), get_dyn(get_series(ind))
                        ),
                        helpers = get_helper_funs(get_model(dm)),
                        model_funs = get_model_funs(get_model(dm)),
                        preDE = calculate_prede(get_model(dm), θ, η_ind, get_const_cov(ind)),
                    )
                )
                sol_accessors = _sol_accessors_with_crossings(
                    get_model(dm), sol, compiled, θ, η_ind, get_const_cov(ind)
                )
            end

            n_points = length(use_dense ? x_fit : obs_rows)
            dists = plot_density ? Vector{Distribution}(undef, n_points) : nothing
            preds = is_mv ? Array{Float64}(undef, n_points, n_marginals) :
                Vector{Float64}(undef, n_points)
            y_obs_series = getfield(get_obs(get_series(ind)), obs_name)
            hmm_priors = Dict{Symbol, Any}()
            for j in 1:n_points
                if use_dense
                    vary = (t = x_fit[j],)
                    obs = calculate_formulas_obs(
                        get_model(dm), θ, η_ind, get_const_cov(ind), vary, sol_accessors
                    )
                    dist = getproperty(obs, obs_name)
                else
                    row = obs_rows[j]
                    vary = _varying_at(dm, ind, j, row)
                    η_row = _row_random_effects_at(
                        dm, i, j, η_ind, rowwise_re; obs_only = true
                    )
                    obs = sol_accessors === nothing ?
                        calculate_formulas_obs(
                            get_model(dm), θ, η_row, get_const_cov(ind), vary
                        ) :
                        calculate_formulas_obs(
                            get_model(dm), θ, η_row, get_const_cov(ind), vary, sol_accessors
                        )
                    dist = _apply_hmm_filter!(
                        hmm_priors, obs_name, getproperty(obs, obs_name), y_obs_series[j]
                    )
                end
                if is_mv
                    mean_vec = _stat_from_dist(dist, plot_func)
                    for m in 1:n_marginals
                        preds[j, m] = _float_if_real(mean_vec[m])
                    end
                else
                    preds[j] = _float_if_real(_stat_from_dist(dist, plot_func))
                end
                if plot_density
                    dists[j] = dist
                end
            end
            for m in margin_range
                curve = is_mv ? vec(preds[:, m]) : preds
                _push_xy!(fit_df, panel, x_fit, curve, _series_label(m))
                ylims = _merge_limits(ylims, curve)
            end

            if plot_density
                for m in margin_range
                    dd = _component_dists([dists], obs_name, m, n_marginals)
                    cm = _density_colormap(m, marginal_colors)
                    push!(get!(post, panel, Any[]), ax -> _plot_density_band!(ax, x_density, dd, cm, style))
                end
            end
        end

        # `_merge_limits` skips empty/non-finite series, so event-only individuals
        # contribute nothing instead of erroring in `minimum`.
        xlims = _merge_limits(xlims, x_fit)
        observed_values = is_mv ? [val for vec in ys_per_margin for val in vec] : y_obs_plot
        ylims = _merge_limits(ylims, observed_values)
    end

    # One legend entry per series: data + fit (scalar) or one per component (vector; data
    # points and curve share the component's color), plus dashed posterior quantiles.
    series = Pair{String, Any}[]
    if is_mv
        for m in margin_range
            push!(series, _series_label(m) => marginal_colors[m])
        end
    else
        plot_data_points && push!(series, "data" => style.color_primary)
        push!(series, "fit" => style.color_secondary)
    end
    if plot_mcmc_quantiles && is_mcmc
        for m in margin_range, q in mcmc_quantiles
            push!(series, _qlabel(m, q) => (m === nothing ? style.color_secondary : marginal_colors[m]))
        end
    end
    xy = AoG.mapping(:x, :y; color = :series, layout = :panel)
    layers = _layers(
        _layer(fit_df, xy, _line_visual(style)),
        _layer(q_df, xy, _line_visual(style; linestyle = :dash, alpha = mcmc_quantiles_alpha)),
        _layer(obs_df, xy, _scatter_visual(style)),
    )
    fig = _draw_facets(
        layers; panels = panels, ncols = ncols, style = style,
        shared_x = shared_x_axis, shared_y = shared_y_axis,
        kwargs_subplot = kwargs_subplot, kwargs_layout = kwargs_layout,
        axis = (;
            xlabel = x_axis_feature === nothing ? "Time" : _axis_label(x_axis_feature),
            ylabel = _axis_label(obs_name),
        ),
        series = series, post = post,
        xlim = _shared_lim(shared_x_axis, xlims), ylim = _shared_lim(shared_y_axis, ylims),
    )
    return _save_plot!(fig, save_path)
end

function plot_fits(
        dm::DataModel;
        plot_density::Bool = false,
        plot_func = mean,
        plot_data_points::Bool = true,
        observable = nothing,
        individuals_idx = nothing,
        x_axis_feature::Union{Nothing, Symbol} = nothing,
        shared_x_axis::Bool = true,
        shared_y_axis::Bool = true,
        ncols::Int = DEFAULT_PLOT_COLS,
        marginal_layout::Symbol = :single,
        style::PlotStyle = PlotStyle(),
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing,
        params::NamedTuple = NamedTuple(),
        constants_re::NamedTuple = NamedTuple(),
        cache_obs_dists::Bool = false,
        mcmc_draws::Int = 1000,
        mcmc_warmup::Union{Nothing, Int} = nothing,
        rng::AbstractRNG = _default_rng()
    )
    save_path = _resolve_plot_path(save_path, plot_path)
    cache = build_plot_cache(
        dm; params = params, constants_re = constants_re,
        cache_obs_dists = cache_obs_dists, rng = rng
    )
    res = FitResult(
        MLE(), FrequentistResult(NamedTuple(), 0.0, 0, NamedTuple(), NamedTuple()),
        FitSummary(
            0.0, true, FitParameters(ComponentArray(), ComponentArray()), NamedTuple()
        ),
        FitDiagnostics((;), (;), (;), (;)), dm, (), NamedTuple()
    )
    return plot_fits(
        res;
        dm = dm,
        plot_density = plot_density,
        plot_func = plot_func,
        plot_data_points = plot_data_points,
        observable = observable,
        individuals_idx = individuals_idx,
        x_axis_feature = x_axis_feature,
        shared_x_axis = shared_x_axis,
        shared_y_axis = shared_y_axis,
        ncols = ncols,
        marginal_layout = marginal_layout,
        style = style,
        kwargs_subplot = kwargs_subplot,
        kwargs_layout = kwargs_layout,
        save_path = save_path,
        cache = cache,
        params = params,
        constants_re = constants_re,
        cache_obs_dists = cache_obs_dists,
        mcmc_draws = mcmc_draws,
        mcmc_warmup = mcmc_warmup,
        rng = rng
    )
end

