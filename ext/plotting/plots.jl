# Fit/data/diagnostic plotting on the Makie panel layer (see plotting.jl). Ported 1:1
# from the Plots.jl extension; Plots draw calls translated to panel-layer helpers.

function _plot_data_dm(dm::DataModel;
        x_axis_feature::Union{Symbol, Nothing} = nothing,
        individuals_idx = nothing,
        shared_x_axis::Bool = true,
        shared_y_axis::Bool = true,
        ncols::Int = DEFAULT_PLOT_COLS,
        style::PlotStyle = PlotStyle(),
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple(),
        save_path::Union{Nothing, String} = nothing,
        marginal_idx::Union{Nothing, Int} = nothing)
    obs_name = _get_observable(dm, nothing)
    inds = individuals_idx === nothing ? collect(eachindex(dm.individuals)) :
           collect(individuals_idx)
    (is_mv, n_marginals) = _obs_multivariate_info(dm, obs_name)
    marginal_colors = _marginal_colors(n_marginals, style)

    plots = Vector{Any}(undef, length(inds))
    xlims = nothing
    ylims = nothing
    for (k, i) in enumerate(inds)
        ind = dm.individuals[i]
        obs_rows = dm.row_groups.obs_rows[i]
        x = _get_x_values(dm, ind, obs_rows, x_axis_feature)
        y = getfield(ind.series.obs, obs_name)
        title_id = string(
            dm.config.primary_id, ": ", dm.df[obs_rows[1], dm.config.primary_id])
        _kw249 = merge(
            (xlabel = x_axis_feature === nothing ? "Time" : _axis_label(x_axis_feature),
                ylabel = _axis_label(obs_name)),
            kwargs_subplot)
        p = create_styled_plot(; title = title_id, style = style, _kw249...)
        if is_mv
            if marginal_idx === nothing
                xs, ys = _collect_multivariate_series(x, y, n_marginals)
                for m in 1:n_marginals
                    create_styled_scatter!(p, xs[m], ys[m];
                        label = _marginal_label(obs_name, m),
                        color = marginal_colors[m],
                        style = style)
                    xlims = _merge_limits(xlims, xs[m])
                    ylims = _merge_limits(ylims, ys[m])
                end
            else
                xs, ys = _collect_multivariate_series(
                    x, y, n_marginals; marginal_idx = marginal_idx)
                create_styled_scatter!(p, xs, ys;
                    label = _marginal_label(obs_name, marginal_idx),
                    color = marginal_colors[marginal_idx],
                    style = style)
                xlims = _merge_limits(xlims, xs)
                ylims = _merge_limits(ylims, ys)
            end
        else
            xs, ys = _collect_scalar_series(x, y)
            create_styled_scatter!(p, xs, ys; label = "", style = style)
            xlims = _merge_limits(xlims, xs)
            ylims = _merge_limits(ylims, ys)
        end
        plots[k] = p
    end

    if shared_x_axis || shared_y_axis
        xlim_use = shared_x_axis && xlims !== nothing ? _pad_limits(xlims[1], xlims[2]) :
                   nothing
        ylim_use = shared_y_axis && ylims !== nothing ? _pad_limits(ylims[1], ylims[2]) :
                   nothing
        _apply_shared_axes!(plots, xlim_use, ylim_use)
    end
    p = combine_plots(plots; ncols = ncols, style = style, kwargs_layout...)
    return _save_plot!(p, save_path)
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
- `shared_x_axis::Bool = true`: share the x-axis range across panels.
- `shared_y_axis::Bool = true`: share the y-axis range across panels.
- `ncols::Int = 3`: number of subplot columns.
- `marginal_layout::Symbol = :single`: `:single` keeps one figure with every marginal overlaid per individual; `:vector` returns a figure per marginal (only valid for vector-valued observables and requires `save_path`/`plot_path` to be `nothing`).
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `kwargs_subplot`: additional Makie `Axis` attributes forwarded to each subplot.
- `kwargs_layout`: additional Makie `Figure` attributes forwarded to the combined layout.
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot (ignored for `:vector` mode).
"""
function plot_data(res::FitResult;
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
        marginal_layout::Symbol = :single)
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
            plots[m] = _plot_data_dm(dm;
                x_axis_feature = x_axis_feature,
                individuals_idx = individuals_idx,
                shared_x_axis = shared_x_axis,
                shared_y_axis = shared_y_axis,
                ncols = ncols,
                style = style,
                kwargs_subplot = kwargs_subplot,
                kwargs_layout = kwargs_layout,
                save_path = nothing,
                marginal_idx = m)
        end
        return plots
    end
    save_path = _resolve_plot_path(save_path, plot_path)
    return _plot_data_dm(dm;
        x_axis_feature = x_axis_feature,
        individuals_idx = individuals_idx,
        shared_x_axis = shared_x_axis,
        shared_y_axis = shared_y_axis,
        ncols = ncols,
        style = style,
        kwargs_subplot = kwargs_subplot,
        kwargs_layout = kwargs_layout,
        save_path = save_path)
end

function plot_data(dm::DataModel;
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
        marginal_layout::Symbol = :single)
    obs_name = _get_observable(dm, nothing)
    marginal_layout in (:single, :vector) ||
        error("marginal_layout must be :single or :vector.")
    (is_mv, n_marginals) = _obs_multivariate_info(dm, obs_name)
    if is_mv && marginal_layout == :vector
        (save_path === nothing && plot_path === nothing) ||
            error("save_path/plot_path are not supported when returning multiple marginal figures.")
        plots = Vector{Figure}(undef, n_marginals)
        for m in 1:n_marginals
            plots[m] = _plot_data_dm(dm;
                x_axis_feature = x_axis_feature,
                individuals_idx = individuals_idx,
                shared_x_axis = shared_x_axis,
                shared_y_axis = shared_y_axis,
                ncols = ncols,
                style = style,
                kwargs_subplot = kwargs_subplot,
                kwargs_layout = kwargs_layout,
                save_path = nothing,
                marginal_idx = m)
        end
        return plots
    end
    save_path = _resolve_plot_path(save_path, plot_path)
    return _plot_data_dm(dm;
        x_axis_feature = x_axis_feature,
        individuals_idx = individuals_idx,
        shared_x_axis = shared_x_axis,
        shared_y_axis = shared_y_axis,
        ncols = ncols,
        style = style,
        kwargs_subplot = kwargs_subplot,
        kwargs_layout = kwargs_layout,
        save_path = save_path)
end

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
- `kwargs_subplot`: additional Makie `Axis` attributes forwarded to each subplot.
- `kwargs_layout`: additional Makie `Figure` attributes forwarded to the combined layout.
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
        rng::AbstractRNG = Random.default_rng())
    dm = _get_dm(res, dm)
    save_path = _resolve_plot_path(save_path, plot_path)
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
    inds = individuals_idx === nothing ? collect(eachindex(dm.individuals)) :
           collect(individuals_idx)

    if cache === nothing
        cache = build_plot_cache(
            res; dm = dm, params = params, constants_re = constants_re_use,
            cache_obs_dists = cache_obs_dists, mcmc_draws = mcmc_draws,
            mcmc_warmup = mcmc_warmup, rng = rng)
    end

    if is_mv && marginal_layout == :vector && marginal_idx === nothing
        (save_path === nothing && plot_path === nothing) ||
            error("save_path/plot_path are not supported when returning multiple marginal figures.")
        plots_vector = Vector{Figure}(undef, n_marginals)
        for m in 1:n_marginals
            plots_vector[m] = plot_fits(res;
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
                rng = rng)
        end
        return plots_vector
    end

    plots = Vector{Any}(undef, length(inds))
    xlims = nothing
    ylims = nothing
    marginal_colors = _marginal_colors(max(n_marginals, 1), style)

    is_mcmc = _is_posterior_draw_fit(res)
    is_mv && is_mcmc &&
        error("plot_fits currently does not support posterior draws for multivariate HMM observables.")
    θ_draws = nothing
    η_draws = nothing
    if is_mcmc
        res = _with_posterior_warmup(res, mcmc_warmup)
        θ_draws, η_draws, _ = _posterior_drawn_params(
            res, dm, constants_re_use, params, mcmc_draws, rng)
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
    if is_mv && plot_density
        @warn "plot_density is ignored for multivariate HMM observables."
        plot_density = false
    end

    _default_xlabel = x_axis_feature === nothing ? "Time" : _axis_label(x_axis_feature)
    _default_ylabel = get(kwargs_subplot, :ylabel, _axis_label(obs_name))
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
        x_obs_plot, y_obs_plot = is_mv ? (nothing, nothing) :
                                 _collect_scalar_series(x_obs, y_obs)
        title_id = string(
            dm.config.primary_id, ": ", dm.df[obs_rows[1], dm.config.primary_id])
        is_leftmost = (k - 1) % ncols == 0
        _ylabel = is_leftmost ? _default_ylabel : ""
        # computed ylabel wins over kwargs_subplot (leftmost-column override); merge dedups.
        _kw574 = merge((xlabel = _default_xlabel,), kwargs_subplot, (ylabel = _ylabel,))
        p = create_styled_plot(; title = title_id, style = style, _kw574...)
        xs_per_margin = nothing
        ys_per_margin = nothing
        if is_mv
            xs_per_margin, ys_per_margin = _collect_multivariate_series(
                x_obs, y_obs, n_marginals)
        end
        if plot_data_points
            if is_mv
                margin_range = marginal_idx === nothing ? (1:n_marginals) : (marginal_idx,)
                for m in margin_range
                    create_styled_scatter!(p, xs_per_margin[m], ys_per_margin[m];
                        label = _marginal_label(obs_name, m),
                        color = marginal_colors[m],
                        style = style)
                end
            else
                create_styled_scatter!(p, x_obs_plot, y_obs_plot; label = "data",
                    color = style.color_primary, style = style)
            end
        end

        if is_mcmc
            n_draws = length(θ_draws)
            preds = zeros(Float64, n_draws, length(use_dense ? x_fit : obs_rows))
            dists_for_density = plot_density ?
                                Vector{Vector{Distribution}}(undef, n_draws) : nothing
            rowwise_re = _needs_rowwise_random_effects(dm, i; obs_only = true)
            for d in 1:n_draws
                θ = θ_draws[d]
                η_ind = η_draws[d][i]
                sol_accessors = nothing
                compiled = nothing
                if dm.model.de.de !== nothing
                    sol, compiled = _solve_dense_individual(dm, ind, θ, η_ind)
                    sol_accessors = _sol_accessors_with_crossings(
                        dm.model, sol, compiled, θ, η_ind, ind.const_cov)
                end
                if use_dense
                    dists = plot_density ? Vector{Distribution}(undef, length(x_fit)) :
                            Distribution[]
                    for (j, t) in enumerate(x_fit)
                        vary = (t = t,)
                        obs = calculate_formulas_obs(
                            dm.model, θ, η_ind, ind.const_cov, vary, sol_accessors)
                        dist = getproperty(obs, obs_name)
                        preds[d, j] = _stat_from_dist(dist, plot_func)
                        if plot_density
                            dists[j] = dist
                        end
                    end
                else
                    y_obs_series_mcmc = getfield(ind.series.obs, obs_name)
                    hmm_priors_draw = Dict{Symbol, Any}()
                    dists = Vector{Distribution}(undef, length(obs_rows))
                    for (j, row) in enumerate(obs_rows)
                        vary = _varying_at(dm, ind, j, row)
                        η_row = _row_random_effects_at(
                            dm, i, j, η_ind, rowwise_re; obs_only = true)
                        obs = sol_accessors === nothing ?
                              calculate_formulas_obs(
                            dm.model, θ, η_row, ind.const_cov, vary) :
                              calculate_formulas_obs(
                            dm.model, θ, η_row, ind.const_cov, vary, sol_accessors)
                        dist = _apply_hmm_filter!(hmm_priors_draw, obs_name,
                            getproperty(obs, obs_name), y_obs_series_mcmc[j])
                        dists[j] = dist
                        preds[d, j] = _stat_from_dist(dist, plot_func)
                    end
                end
                if plot_density
                    dists_for_density[d] = dists
                end
            end
            mean_curve = vec(mean(preds, dims = 1))
            create_styled_line!(p, x_fit, mean_curve; label = "fit",
                color = style.color_secondary, style = style)
            ylims = ylims === nothing ? (minimum(mean_curve), maximum(mean_curve)) :
                    (min(ylims[1], minimum(mean_curve)), max(ylims[2], maximum(mean_curve)))

            if plot_mcmc_quantiles
                for q in mcmc_quantiles
                    qvals = mapslices(x -> quantile(vec(x), q / 100), preds; dims = 1)
                    qvals = vec(qvals)
                    create_styled_line!(p, x_fit, qvals; color = style.color_secondary,
                        alpha = mcmc_quantiles_alpha,
                        linestyle = :dash, label = "$(q)%", style = style)
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
                        create_styled_scatter!(
                            p, fill(x_density[j], length(grid.vals)), grid.vals;
                            color = probs, colormap = :viridis, marker = :xcross,
                            markersize = style.marker_size_pmf,
                            strokewidth = style.marker_stroke_width_pmf, label = "")
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
                        # z is (ny, nx); Makie heatmap wants (nx, ny) -> permutedims.
                        _record!(p,
                            ax -> heatmap!(ax, x_density, grid.y, permutedims(z);
                                colormap = (:viridis, 0.5)))
                    end
                end
            end
        else
            θ = cache.params
            η_ind = cache.random_effects[i]
            rowwise_re = _needs_rowwise_random_effects(dm, i; obs_only = true)
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
                sol_accessors = _sol_accessors_with_crossings(
                    dm.model, sol, compiled, θ, η_ind, ind.const_cov)
            end

            n_points = length(use_dense ? x_fit : obs_rows)
            dists = plot_density ? Vector{Distribution}(undef, n_points) : nothing
            if is_mv
                preds = Array{Float64}(undef, n_points, n_marginals)
            else
                preds = Vector{Float64}(undef, n_points)
            end
            if use_dense
                if is_mv
                    preds_dense = zeros(Float64, length(x_fit), n_marginals)
                else
                    preds_dense = Vector{Float64}(undef, length(x_fit))
                end
                for (j, t) in enumerate(x_fit)
                    vary = (t = t,)
                    obs = calculate_formulas_obs(
                        dm.model, θ, η_ind, ind.const_cov, vary, sol_accessors)
                    dist = getproperty(obs, obs_name)
                    if is_mv
                        mean_vec = _stat_from_dist(dist, plot_func)
                        for m in 1:n_marginals
                            preds_dense[j, m] = _float_if_real(mean_vec[m])
                        end
                    else
                        preds_dense[j] = _float_if_real(_stat_from_dist(dist, plot_func))
                    end
                    if plot_density
                        dists[j] = dist
                    end
                end
                margin_range = marginal_idx === nothing ? (1:n_marginals) : (marginal_idx,)
                for m in margin_range
                    curve = is_mv ? vec(preds_dense[:, m]) : preds_dense
                    label = is_mv ? _marginal_label(obs_name, m) : "fit"
                    color = is_mv ? marginal_colors[m] : style.color_secondary
                    create_styled_line!(
                        p, x_fit, curve; label = label, color = color, style = style)
                    y_min = minimum(curve)
                    y_max = maximum(curve)
                    ylims = ylims === nothing ? (y_min, y_max) :
                            (min(ylims[1], y_min), max(ylims[2], y_max))
                end
            else
                y_obs_series = getfield(ind.series.obs, obs_name)
                hmm_priors = Dict{Symbol, Any}()
                for (j, row) in enumerate(obs_rows)
                    vary = _varying_at(dm, ind, j, row)
                    η_row = _row_random_effects_at(
                        dm, i, j, η_ind, rowwise_re; obs_only = true)
                    obs = sol_accessors === nothing ?
                          calculate_formulas_obs(dm.model, θ, η_row, ind.const_cov, vary) :
                          calculate_formulas_obs(
                        dm.model, θ, η_row, ind.const_cov, vary, sol_accessors)
                    dist = _apply_hmm_filter!(
                        hmm_priors, obs_name, getproperty(obs, obs_name), y_obs_series[j])
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
                margin_range = marginal_idx === nothing ? (1:n_marginals) : (marginal_idx,)
                for m in margin_range
                    curve = is_mv ? vec(preds[:, m]) : preds
                    label = is_mv ? _marginal_label(obs_name, m) : "fit"
                    color = is_mv ? marginal_colors[m] : style.color_secondary
                    create_styled_line!(
                        p, x_fit, curve; label = label, color = color, style = style)
                    y_min = minimum(curve)
                    y_max = maximum(curve)
                    ylims = ylims === nothing ? (y_min, y_max) :
                            (min(ylims[1], y_min), max(ylims[2], y_max))
                end
            end

            if plot_density && !is_mv
                if _is_bernoulli(dists[1])
                    # Skip Bernoulli density overlay; the fit line already represents p(y=1).
                elseif _is_discrete(dists[1])
                    for j in eachindex(dists)
                        grid = _density_grid_discrete(dists[j], 0.995)
                        grid === nothing && continue
                        create_styled_scatter!(
                            p, fill(x_density[j], length(grid.vals)), grid.vals;
                            color = grid.probs, colormap = :viridis, marker = :xcross,
                            markersize = style.marker_size_pmf,
                            strokewidth = style.marker_stroke_width_pmf, label = "")
                    end
                else
                    grid = _density_grid_continuous(dists, 0.995, 100)
                    if grid !== nothing
                        # grid.z is (ny, nx); Makie heatmap wants (nx, ny) -> permutedims.
                        _record!(p,
                            ax -> heatmap!(ax, x_density, grid.y, permutedims(grid.z);
                                colormap = (:viridis, 0.5)))
                    end
                end
            end
        end

        plots[k] = p
        xlims = xlims === nothing ? (minimum(x_fit), maximum(x_fit)) :
                (min(xlims[1], minimum(x_fit)), max(xlims[2], maximum(x_fit)))
        observed_values = is_mv ? [val for vec in ys_per_margin for val in vec] : y_obs_plot
        ylims = _merge_limits(ylims, observed_values)
    end

    if shared_x_axis || shared_y_axis
        xlim_use = shared_x_axis ? _pad_limits(xlims[1], xlims[2]) : nothing
        ylim_use = shared_y_axis ? _pad_limits(ylims[1], ylims[2]) : nothing
        _apply_shared_axes!(plots, xlim_use, ylim_use)
    end
    p = combine_plots(plots; ncols = ncols, style = style, kwargs_layout...)
    return _save_plot!(p, save_path)
end

function _plot_hidden_states_impl(dm::DataModel,
        obs_name::Symbol,
        θ,
        η_vec,
        x_axis_feature::Union{Nothing, Symbol},
        shared_x_axis::Bool,
        shared_y_axis::Bool,
        ncols::Int,
        style::PlotStyle,
        kwargs_subplot,
        kwargs_layout,
        save_path::Union{Nothing, String},
        individuals_idx = nothing)
    inds = individuals_idx === nothing ? collect(eachindex(dm.individuals)) :
           collect(individuals_idx)
    plots = Vector{Any}(undef, length(inds))
    xlims = nothing
    ylims = nothing
    for (k, i) in enumerate(inds)
        ind = dm.individuals[i]
        obs_rows = dm.row_groups.obs_rows[i]
        x_vals = _get_x_values(dm, ind, obs_rows, x_axis_feature)
        title_id = string(
            dm.config.primary_id, ": ", dm.df[obs_rows[1], dm.config.primary_id])
        _kw870 = merge(
            (xlabel = x_axis_feature === nothing ? "Time" : _axis_label(x_axis_feature),
                ylabel = "Hidden-state probability"),
            kwargs_subplot)
        p = create_styled_plot(; title = title_id, style = style, _kw870...)
        θ_ind = θ
        η_ind = η_vec[i]
        rowwise_re = _needs_rowwise_random_effects(dm, i; obs_only = true)
        sol_accessors = nothing
        if dm.model.de.de !== nothing
            sol = nothing
            compiled = nothing
            pre = calculate_prede(dm.model, θ_ind, η_ind, ind.const_cov)
            pc = (;
                fixed_effects = θ_ind,
                random_effects = η_ind,
                constant_covariates = ind.const_cov,
                varying_covariates = merge((t = ind.series.vary.t[1],), ind.series.dyn),
                helpers = get_helper_funs(dm.model),
                model_funs = get_model_funs(dm.model),
                preDE = pre
            )
            compiled = get_de_compiler(dm.model.de.de)(pc)
            sol = cache = nothing
            sol = dm.model.de.de !== nothing ?
                  _solve_dense_individual(dm, ind, θ_ind, η_ind)[1] : nothing
            sol_accessors = _sol_accessors_with_crossings(
                dm.model, sol, compiled, θ_ind, η_ind, ind.const_cov)
        end

        times = Float64[]
        posteriors = Vector{Vector{Float64}}()
        n_states = nothing
        hmm_priors_hs = Dict{Symbol, Any}()
        for (j, row) in enumerate(obs_rows)
            vary = _varying_at(dm, ind, j, row)
            η_row = _row_random_effects_at(dm, i, j, η_ind, rowwise_re; obs_only = true)
            obs = sol_accessors === nothing ?
                  calculate_formulas_obs(dm.model, θ_ind, η_row, ind.const_cov, vary) :
                  calculate_formulas_obs(
                dm.model, θ_ind, η_row, ind.const_cov, vary, sol_accessors)
            dist = getproperty(obs, obs_name)
            dist isa MVDiscreteTimeDiscreteStatesHMM ||
                error("Observable $(obs_name) must be MVDiscreteTimeDiscreteStatesHMM.")
            y_val = getfield(ind.series.obs, obs_name)[j]
            prior = get(hmm_priors_hs, obs_name, nothing)
            dist_filtered = _hmm_with_prior(dist, prior)
            if y_val === missing
                hmm_priors_hs[obs_name] = probabilities_hidden_states(dist_filtered)
                continue
            end
            post = posterior_hidden_states(dist_filtered, y_val)
            hmm_priors_hs[obs_name] = post
            if n_states === nothing
                n_states = dist.n_states
            end
            push!(times, x_vals[j])
            push!(posteriors, post)
        end

        isempty(times) &&
            @warn "No non-missing observations found for individual $(title_id)."
        state_labels = ["State $(m)" for m in 1:max(n_states === nothing ? 0 : n_states, 1)]
        state_colors = _marginal_colors(max(n_states === nothing ? 0 : n_states, 1), style)
        if !isempty(times)
            sorted_unique_times = sort(unique(times))
            spacings = diff(sorted_unique_times)
            positive_spacings = filter(x -> x > 0, spacings)
            base_width = isempty(positive_spacings) ? 1.0 : minimum(positive_spacings)
            bar_width = isempty(positive_spacings) ? 1.0 : min(base_width * 0.8, 1.0)
            half_width = bar_width / 2
            prob_mat = zeros(Float64, n_states, length(times))
            for (idx, post) in enumerate(posteriors)
                prob_mat[:, idx] = post
            end
            bottom = zeros(Float64, length(times))
            for m in 1:n_states
                # One stacked-bar rectangle per time with nonzero probability for state m.
                rects = Rect2f[]
                for (idx, t) in enumerate(times)
                    if prob_mat[m, idx] == 0
                        continue
                    end
                    x_left = t - half_width
                    y_bot = bottom[idx]
                    push!(rects, Rect2f(x_left, y_bot, bar_width, prob_mat[m, idx]))
                end
                isempty(rects) && continue
                lbl = _label(p, state_labels[m])
                _record!(p,
                    ax -> poly!(
                        ax, rects; color = state_colors[m], strokewidth = 0, label = lbl))
                bottom .+= prob_mat[m, :]
            end
            ylims = _merge_limits(ylims, [0.0, 1.0])
            time_min = minimum(times) - half_width
            time_max = maximum(times) + half_width
            xlims = xlims === nothing ? (time_min, time_max) :
                    (min(xlims[1], time_min), max(xlims[2], time_max))
        end
        plots[k] = p
        ylims = _merge_limits(ylims, [0.0, 1.0])
    end

    if shared_x_axis || shared_y_axis
        xlim_use = shared_x_axis && xlims !== nothing ? _pad_limits(xlims[1], xlims[2]) :
                   nothing
        ylim_use = shared_y_axis ? (0.0, 1.0) : nothing
        _apply_shared_axes!(plots, xlim_use, ylim_use)
    end
    p = combine_plots(plots; ncols = ncols, style = style, kwargs_layout...)
    return _save_plot!(p, save_path)
end

"""
    plot_hidden_states(res::FitResult; kwargs...)
    plot_hidden_states(dm::DataModel; kwargs...)

Plot the filtered posterior probability of each hidden state over time for a
hidden-Markov-model outcome, as a multi-panel figure (one panel per individual).

# Keyword Arguments
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
- `observable = nothing`: HMM observable to plot; defaults to the first one.
- `individuals_idx`: indices or IDs of individuals to include, or `nothing` for all.
- `x_axis_feature::Union{Nothing, Symbol} = nothing`: covariate for the x-axis; defaults to time.
- `ncols::Int = 3`, `figure_layout::Symbol = :single`: panel layout.
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
"""
function plot_hidden_states(res::FitResult;
        dm::Union{Nothing, DataModel} = nothing,
        observable = nothing,
        individuals_idx = nothing,
        x_axis_feature::Union{Nothing, Symbol} = nothing,
        shared_x_axis::Bool = true,
        shared_y_axis::Bool = true,
        ncols::Int = DEFAULT_PLOT_COLS,
        figure_layout::Symbol = :single,
        style::PlotStyle = PlotStyle(),
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing,
        params::NamedTuple = NamedTuple(),
        constants_re::NamedTuple = NamedTuple(),
        mcmc_draws::Int = 1,
        rng::AbstractRNG = Random.default_rng())
    dm = _get_dm(res, dm)
    figure_layout in (:single, :vector) ||
        error("figure_layout must be :single or :vector.")
    if figure_layout == :vector
        (save_path === nothing && plot_path === nothing) ||
            error("save_path/plot_path are not supported when returning multiple figures.")
    end
    save_path = _resolve_plot_path(save_path, plot_path)
    constants_re_use = _res_constants_re(res, constants_re)
    obs_name = _get_observable(dm, observable)
    (is_mv, _) = _obs_multivariate_info(dm, obs_name)
    is_mv || error("plot_hidden_states requires a multivariate observable.")
    _is_posterior_draw_fit(res) &&
        error("plot_hidden_states does not support posterior draws yet.")

    θ = get_params(res; scale = :untransformed)
    θ = _apply_param_overrides(θ, params)
    η_vec = _default_random_effects(res, dm, constants_re_use, θ, rng, mcmc_draws)
    inds = individuals_idx === nothing ? collect(eachindex(dm.individuals)) :
           collect(individuals_idx)

    if figure_layout == :vector
        plots = Vector{Figure}(undef, length(inds))
        for (k, idx) in enumerate(inds)
            plots[k] = _plot_hidden_states_impl(dm,
                obs_name,
                θ,
                η_vec,
                x_axis_feature,
                shared_x_axis,
                shared_y_axis,
                ncols,
                style,
                kwargs_subplot,
                kwargs_layout,
                nothing,
                [idx])
        end
        return plots
    end

    return _plot_hidden_states_impl(dm,
        obs_name,
        θ,
        η_vec,
        x_axis_feature,
        shared_x_axis,
        shared_y_axis,
        ncols,
        style,
        kwargs_subplot,
        kwargs_layout,
        save_path,
        inds)
end

function plot_hidden_states(dm::DataModel;
        observable = nothing,
        individuals_idx = nothing,
        x_axis_feature::Union{Nothing, Symbol} = nothing,
        shared_x_axis::Bool = true,
        shared_y_axis::Bool = true,
        ncols::Int = DEFAULT_PLOT_COLS,
        figure_layout::Symbol = :single,
        style::PlotStyle = PlotStyle(),
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing,
        params::NamedTuple = NamedTuple(),
        constants_re::NamedTuple = NamedTuple(),
        rng::AbstractRNG = Random.default_rng())
    figure_layout in (:single, :vector) ||
        error("figure_layout must be :single or :vector.")
    if figure_layout == :vector
        (save_path === nothing && plot_path === nothing) ||
            error("save_path/plot_path are not supported when returning multiple figures.")
    end
    save_path = _resolve_plot_path(save_path, plot_path)
    obs_name = _get_observable(dm, observable)
    (is_mv, _) = _obs_multivariate_info(dm, obs_name)
    is_mv || error("plot_hidden_states requires a multivariate observable.")

    θ = get_θ0_untransformed(dm.model.fixed.fixed)
    θ = _apply_param_overrides(θ, params)
    η_vec = _default_random_effects_from_dm(dm, constants_re, θ)
    inds = individuals_idx === nothing ? collect(eachindex(dm.individuals)) :
           collect(individuals_idx)

    if figure_layout == :vector
        plots = Vector{Figure}(undef, length(inds))
        for (k, idx) in enumerate(inds)
            plots[k] = _plot_hidden_states_impl(dm,
                obs_name,
                θ,
                η_vec,
                x_axis_feature,
                shared_x_axis,
                shared_y_axis,
                ncols,
                style,
                kwargs_subplot,
                kwargs_layout,
                nothing,
                [idx])
        end
        return plots
    end

    return _plot_hidden_states_impl(dm,
        obs_name,
        θ,
        η_vec,
        x_axis_feature,
        shared_x_axis,
        shared_y_axis,
        ncols,
        style,
        kwargs_subplot,
        kwargs_layout,
        save_path,
        inds)
end

function _plot_emission_for_individual(dm::DataModel,
        obs_name::Symbol,
        ind_idx::Int,
        θ,
        η_ind,
        row::Int,
        title_base::String,
        style::PlotStyle,
        kwargs_subplot,
        state_ncols::Int)
    ind = dm.individuals[ind_idx]
    obs_rows = dm.row_groups.obs_rows[ind_idx]
    row_pos = findfirst(==(row), obs_rows)
    row_pos === nothing &&
        error("Observation row $(row) not found for individual $(ind_idx).")

    sol_accessors = nothing
    if dm.model.de.de !== nothing
        compiled = nothing
        pre = calculate_prede(dm.model, θ, η_ind, ind.const_cov)
        pc = (;
            fixed_effects = θ,
            random_effects = η_ind,
            constant_covariates = ind.const_cov,
            varying_covariates = merge((t = ind.series.vary.t[1],), ind.series.dyn),
            helpers = get_helper_funs(dm.model),
            model_funs = get_model_funs(dm.model),
            preDE = pre
        )
        compiled = get_de_compiler(dm.model.de.de)(pc)
        sol = _solve_dense_individual(dm, ind, θ, η_ind)[1]
        sol_accessors = _sol_accessors_with_crossings(
            dm.model, sol, compiled, θ, η_ind, ind.const_cov)
    end

    vary = _varying_at(dm, ind, row_pos, row)
    rowwise_re = _needs_rowwise_random_effects(dm, ind_idx; obs_only = true)
    η_row = _row_random_effects_at(dm, ind_idx, row_pos, η_ind, rowwise_re; obs_only = true)
    obs = sol_accessors === nothing ?
          calculate_formulas_obs(dm.model, θ, η_row, ind.const_cov, vary) :
          calculate_formulas_obs(dm.model, θ, η_row, ind.const_cov, vary, sol_accessors)
    dist = getproperty(obs, obs_name)
    dist isa MVDiscreteTimeDiscreteStatesHMM ||
        error("Observable $(obs_name) must be MVDiscreteTimeDiscreteStatesHMM.")

    n_states = dist.n_states
    n_marginals = _mv_n_outcomes(dist.emission_dists[1])
    state_plots = Vector{MakiePanel}(undef, n_states)
    marginal_colors = _marginal_colors(n_marginals, style)
    overall_xlim = nothing
    overall_ylim = nothing

    for s in 1:n_states
        _kw1209 = merge((xlabel = "Outcome value", ylabel = "Density"), kwargs_subplot)
        p_state = create_styled_plot(;
            title = "$title_base • State $s", style = style, _kw1209...)
        emission = dist.emission_dists[s]
        marginals = _state_emission_marginals(emission)
        state_xlim = nothing
        state_ylim = nothing
        for (m, dist_m) in enumerate(marginals)
            dist_m === nothing && continue
            label = _marginal_label(obs_name, m)
            if _is_discrete(dist_m)
                grid = _density_grid_discrete(dist_m, 0.995)
                grid === nothing && continue
                create_styled_scatter!(p_state, grid.vals, grid.probs;
                    label = label,
                    color = marginal_colors[m],
                    style = style)
                state_xlim = _merge_limits(state_xlim, grid.vals)
                state_ylim = _merge_limits(state_ylim, grid.probs)
            else
                grid = _density_grid_continuous([dist_m], 0.995, 200)
                grid === nothing && continue
                densities = vec(grid.z[:, 1])
                create_styled_line!(p_state, grid.y, densities;
                    label = label,
                    color = marginal_colors[m],
                    style = style)
                state_xlim = _merge_limits(state_xlim, grid.y)
                state_ylim = _merge_limits(state_ylim, densities)
            end
        end
        if state_ylim !== nothing
            _apply_shared_axes!([p_state], nothing, state_ylim)
        end
        overall_xlim = _merge_limits(overall_xlim, state_xlim)
        overall_ylim = _merge_limits(overall_ylim, state_ylim)
        state_plots[s] = p_state
    end

    # Inner per-individual grid is a nested panel group; combined by the caller.
    group = MakiePanelGroup(state_plots, min(state_ncols, n_states))
    return (plot = group, xlims = overall_xlim, ylims = overall_ylim)
end

function _plot_emission_impl(dm::DataModel,
        obs_name::Symbol,
        θ,
        η_vec,
        individuals_idx,
        time_idx,
        time_point,
        time_col,
        shared_y_axis::Bool,
        ncols::Int,
        style::PlotStyle,
        kwargs_subplot,
        kwargs_layout,
        save_path::Union{Nothing, String},
        figure_layout::Symbol)
    inds = individuals_idx === nothing ? collect(eachindex(dm.individuals)) :
           collect(individuals_idx)
    groups = Vector{MakiePanelGroup}(undef, length(inds))
    xlims = nothing
    ylims = nothing
    time_col_use = time_col === nothing ? dm.config.time_col : time_col
    state_ncols = min(3, DEFAULT_PLOT_COLS)

    for (k, i) in enumerate(inds)
        ind = dm.individuals[i]
        obs_rows = dm.row_groups.obs_rows[i]
        row = _resolve_emission_row(dm, obs_rows, time_idx, time_point, time_col_use)
        title_id = string(dm.config.primary_id, ": ", dm.df[row, dm.config.primary_id])
        time_val = dm.df[row, time_col_use]
        time_label = string(time_val)
        record = _plot_emission_for_individual(dm,
            obs_name,
            i,
            θ,
            η_vec[i],
            row,
            "$title_id • t = $time_label",
            style,
            kwargs_subplot,
            state_ncols)
        groups[k] = record.plot
        xlims = _merge_limits(xlims, record.xlims === nothing ? () : record.xlims)
        ylims = _merge_limits(ylims, record.ylims === nothing ? () : record.ylims)
    end

    if shared_y_axis && ylims !== nothing
        _apply_shared_axes!(groups, nothing, _pad_limits(ylims[1], ylims[2]))
    end

    if figure_layout == :vector
        return [combine_plots(g.panels; ncols = g.ncols, style = style) for g in groups]
    end
    p = combine_plots(groups; ncols = ncols, style = style, kwargs_layout...)
    return _save_plot!(p, save_path)
end

"""
    plot_emission_distributions(res::FitResult; kwargs...)
    plot_emission_distributions(dm::DataModel; kwargs...)

Plot the per-state emission (observation) distributions of a hidden-Markov-model outcome
at a chosen time point, as a multi-panel figure.

# Keyword Arguments
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
- `observable = nothing`: HMM observable to plot; defaults to the first one.
- `individuals_idx`: indices or IDs of individuals to include, or `nothing` for all.
- `time_idx::Union{Nothing, Int} = nothing`, `time_point = nothing`: time at which to evaluate
  the emission distributions (by observation index or by value).
- `ncols::Int = 3`, `figure_layout::Symbol = :single`: panel layout.
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
"""
function plot_emission_distributions(res::FitResult;
        dm::Union{Nothing, DataModel} = nothing,
        observable = nothing,
        individuals_idx = nothing,
        time_idx::Union{Nothing, Int} = nothing,
        time_point = nothing,
        time_col::Union{Nothing, Symbol} = nothing,
        shared_y_axis::Bool = true,
        ncols::Int = DEFAULT_PLOT_COLS,
        figure_layout::Symbol = :single,
        style::PlotStyle = PlotStyle(),
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing,
        params::NamedTuple = NamedTuple(),
        constants_re::NamedTuple = NamedTuple(),
        mcmc_draws::Int = 1,
        rng::AbstractRNG = Random.default_rng())
    figure_layout in (:single, :vector) ||
        error("figure_layout must be :single or :vector.")
    (time_idx !== nothing) && (time_point !== nothing) &&
        error("Specify only one of time_idx or time_point.")
    if figure_layout == :vector
        (save_path === nothing && plot_path === nothing) ||
            error("save_path/plot_path are not supported when returning multiple figures.")
    end
    dm = _get_dm(res, dm)
    save_path = _resolve_plot_path(save_path, plot_path)
    constants_re_use = _res_constants_re(res, constants_re)
    obs_name = _get_observable(dm, observable)
    (is_mv, _) = _obs_multivariate_info(dm, obs_name)
    is_mv || error("plot_emission_distributions requires a multivariate observable.")
    _is_posterior_draw_fit(res) &&
        error("plot_emission_distributions does not support posterior draws.")

    θ = get_params(res; scale = :untransformed)
    θ = _apply_param_overrides(θ, params)
    η_vec = _default_random_effects(res, dm, constants_re_use, θ, rng, mcmc_draws)

    plots = _plot_emission_impl(dm,
        obs_name,
        θ,
        η_vec,
        individuals_idx,
        time_idx,
        time_point,
        time_col,
        shared_y_axis,
        ncols,
        style,
        kwargs_subplot,
        kwargs_layout,
        save_path,
        figure_layout)

    return plots === nothing ? [] : plots
end

function plot_emission_distributions(dm::DataModel;
        observable = nothing,
        individuals_idx = nothing,
        time_idx::Union{Nothing, Int} = nothing,
        time_point = nothing,
        time_col::Union{Nothing, Symbol} = nothing,
        shared_y_axis::Bool = true,
        ncols::Int = DEFAULT_PLOT_COLS,
        figure_layout::Symbol = :single,
        style::PlotStyle = PlotStyle(),
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing,
        params::NamedTuple = NamedTuple(),
        constants_re::NamedTuple = NamedTuple(),
        rng::AbstractRNG = Random.default_rng())
    figure_layout in (:single, :vector) ||
        error("figure_layout must be :single or :vector.")
    (time_idx !== nothing) && (time_point !== nothing) &&
        error("Specify only one of time_idx or time_point.")
    if figure_layout == :vector
        (save_path === nothing && plot_path === nothing) ||
            error("save_path/plot_path are not supported when returning multiple figures.")
    end
    save_path = _resolve_plot_path(save_path, plot_path)
    obs_name = _get_observable(dm, observable)
    (is_mv, _) = _obs_multivariate_info(dm, obs_name)
    is_mv || error("plot_emission_distributions requires a multivariate observable.")

    θ = get_θ0_untransformed(dm.model.fixed.fixed)
    θ = _apply_param_overrides(θ, params)
    η_vec = _default_random_effects_from_dm(dm, constants_re, θ)

    plots = _plot_emission_impl(dm,
        obs_name,
        θ,
        η_vec,
        individuals_idx,
        time_idx,
        time_point,
        time_col,
        shared_y_axis,
        ncols,
        style,
        kwargs_subplot,
        kwargs_layout,
        save_path,
        figure_layout)

    return plots === nothing ? [] : plots
end

function plot_fits(dm::DataModel;
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
        rng::AbstractRNG = Random.default_rng())
    save_path = _resolve_plot_path(save_path, plot_path)
    cache = build_plot_cache(dm; params = params, constants_re = constants_re,
        cache_obs_dists = cache_obs_dists, rng = rng)
    res = FitResult(MLE(), MLEResult(NamedTuple(), 0.0, 0, NamedTuple(), NamedTuple()),
        FitSummary(
            0.0, true, FitParameters(ComponentArray(), ComponentArray()), NamedTuple()),
        FitDiagnostics((;), (;), (;), (;)), dm, (), NamedTuple())
    return plot_fits(res;
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
        rng = rng)
end

function _plot_fits_comparison_impl(fits::AbstractVector{<:FitResult},
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
        plot_path::Union{Nothing, String} = nothing)
    length(fits) == length(labels) ||
        error("Internal error: fits and labels length mismatch.")
    isempty(fits) && error("plot_fits_comparison requires at least one fit result.")

    save_path = _resolve_plot_path(save_path, plot_path)
    dms = dm === nothing ? [_get_dm(res, nothing) for res in fits] : fill(dm, length(fits))
    dm_ref = _validate_same_data_model_for_comparison(dms)

    obs_name = _get_observable(dm_ref, observable)
    inds = individuals_idx === nothing ? collect(eachindex(dm_ref.individuals)) :
           collect(individuals_idx)
    caches = [build_plot_cache(fits[j]; dm = dms[j], cache_obs_dists = false)
              for j in eachindex(fits)]
    line_colors = _comparison_line_colors(length(fits), style)

    plots = Vector{Any}(undef, length(inds))
    xlims = nothing
    ylims = nothing
    for (k, i) in enumerate(inds)
        ind = dm_ref.individuals[i]
        obs_rows = dm_ref.row_groups.obs_rows[i]
        x_obs = _get_x_values(dm_ref, ind, obs_rows, x_axis_feature)
        y_obs = getfield(ind.series.obs, obs_name)
        x_obs_plot, y_obs_plot = _collect_scalar_series(x_obs, y_obs)
        title_id = string(dm_ref.config.primary_id, ": ",
            dm_ref.df[obs_rows[1], dm_ref.config.primary_id])
        _kw1619 = merge(
            (xlabel = x_axis_feature === nothing ? "Time" : _axis_label(x_axis_feature),
                ylabel = _axis_label(obs_name)),
            kwargs_subplot)
        p = create_styled_plot(; title = title_id, style = style, _kw1619...)
        if plot_data_points
            create_styled_scatter!(p, x_obs_plot, y_obs_plot; label = "data",
                color = style.color_primary, style = style)
        end

        for j in eachindex(fits)
            curve = _fit_curve_from_cache(
                dms[j], caches[j], i, obs_name, x_axis_feature, plot_func)
            create_styled_line!(
                p,
                curve.x_fit,
                curve.preds;
                label = labels[j],
                color = line_colors[j],
                style = style,
                linestyle = _comparison_line_style(labels[j], style)
            )
            xlims = xlims === nothing ? (minimum(curve.x_fit), maximum(curve.x_fit)) :
                    (
                min(xlims[1], minimum(curve.x_fit)), max(xlims[2], maximum(curve.x_fit)))
            ylims = ylims === nothing ? (minimum(curve.preds), maximum(curve.preds)) :
                    (
                min(ylims[1], minimum(curve.preds)), max(ylims[2], maximum(curve.preds)))
        end

        ylims = _merge_limits(ylims, y_obs_plot)
        plots[k] = p
    end

    if shared_x_axis || shared_y_axis
        xlim_use = shared_x_axis ? _pad_limits(xlims[1], xlims[2]) : nothing
        ylim_use = shared_y_axis ? _pad_limits(ylims[1], ylims[2]) : nothing
        _apply_shared_axes!(plots, xlim_use, ylim_use)
    end
    p = combine_plots(plots; ncols = ncols, style = style, kwargs_layout...)
    return _save_plot!(p, save_path)
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
model on the same panel, labeled by vector index, `NamedTuple` key, or `Dict` key.

All keyword arguments are forwarded to the underlying `plot_fits` implementation.
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
                           style, kwargs_subplot, kwargs_layout, save_path, plot_path)
                           -> Makie.Figure

    plot_observed_profiles(res::FitResult; dm, observable, individuals_idx,
                           x_axis_feature, style, kwargs_subplot, kwargs_layout,
                           save_path, plot_path) -> Makie.Figure

Plot observed trajectories for all (or selected) individuals overlaid on a single panel
(spaghetti plot). Each individual's time series is drawn as a connected line with
scatter points at the observed time points.

# Keyword Arguments
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
- `observable`: name of the outcome variable to plot, or `nothing` to use the first.
- `individuals_idx`: indices or IDs of individuals to include, or `nothing` for all.
- `x_axis_feature::Union{Symbol, Nothing} = nothing`: covariate for the x-axis;
  defaults to the time column.
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `kwargs_subplot`: additional Makie `Axis` attributes forwarded to the single panel.
- `kwargs_layout`: additional Makie `Figure` attributes forwarded to the layout call.
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
- `plot_path::Union{Nothing, String} = nothing`: alias for `save_path`.
"""
function plot_observed_profiles(dm::DataModel;
        observable = nothing,
        individuals_idx = nothing,
        x_axis_feature::Union{Symbol, Nothing} = nothing,
        style::PlotStyle = PlotStyle(),
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing)
    save_path = _resolve_plot_path(save_path, plot_path)
    obs_name = _get_observable(dm, observable)
    inds = individuals_idx === nothing ? collect(eachindex(dm.individuals)) :
           collect(individuals_idx)

    _kw_op = merge(
        (xlabel = x_axis_feature === nothing ? "Time" : _axis_label(x_axis_feature),
            ylabel = _axis_label(obs_name)),
        kwargs_subplot)
    p = create_styled_plot(; title = "Observed profiles", style = style, _kw_op...)
    p.legend_position = :none

    for i in inds
        ind = dm.individuals[i]
        obs_rows = dm.row_groups.obs_rows[i]
        x = _get_x_values(dm, ind, obs_rows, x_axis_feature)
        y = getfield(ind.series.obs, obs_name)
        xs, ys = _collect_scalar_series(x, y)
        isempty(xs) && continue
        order = sortperm(xs)
        xs_sorted = xs[order]
        ys_sorted = ys[order]
        create_styled_line!(
            p, xs_sorted, ys_sorted; label = "", color = style.color_primary, style = style)
        create_styled_scatter!(
            p, xs_sorted, ys_sorted; label = "", color = style.color_primary, style = style)
    end

    fig = combine_plots([p]; ncols = 1, style = style, kwargs_layout...)
    return _save_plot!(fig, save_path)
end

function plot_observed_profiles(res::FitResult;
        dm::Union{Nothing, DataModel} = nothing,
        observable = nothing,
        individuals_idx = nothing,
        x_axis_feature::Union{Symbol, Nothing} = nothing,
        style::PlotStyle = PlotStyle(),
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing)
    dm = _get_dm(res, dm)
    return plot_observed_profiles(dm;
        observable = observable,
        individuals_idx = individuals_idx,
        x_axis_feature = x_axis_feature,
        style = style,
        kwargs_subplot = kwargs_subplot,
        kwargs_layout = kwargs_layout,
        save_path = save_path,
        plot_path = plot_path)
end

# Per-individual covariate-adjusted population-mean random effects for PRED.
# For each individual, the prior distribution is evaluated at that individual's
# constant covariates, so weight-based covariate shifts are preserved.
"""
    plot_dv_pred(res::FitResult; dm, observable, style, kwargs_subplot,
                 kwargs_layout, save_path, plot_path) -> Makie.Figure

Goodness-of-fit scatter plot of observed values (DV) against population
predictions (PRED). PRED is the model prediction evaluated at each
individual's population-mean random effects using that individual's own
constant covariates, preserving covariate-driven shifts such as allometric
weight scaling. An identity line is overlaid for reference.

# Keyword Arguments
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
- `observable`: name of the outcome variable, or `nothing` for the first.
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `kwargs_subplot`: additional Makie `Axis` attributes forwarded to the subplot.
- `kwargs_layout`: additional Makie `Figure` attributes forwarded to the layout.
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
- `plot_path::Union{Nothing, String} = nothing`: alias for `save_path`.
"""
function plot_dv_pred(res::FitResult;
        dm::Union{Nothing, DataModel} = nothing,
        observable = nothing,
        style::PlotStyle = PlotStyle(),
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing)
    dm = _get_dm(res, dm)
    save_path = _resolve_plot_path(save_path, plot_path)
    obs_name = _get_observable(dm, observable)
    θ = get_params(res; scale = :untransformed)
    η_pred = _pred_re_per_individual(dm, θ)

    dv, pred, _ = _collect_pred_series(dm, obs_name, θ, η_pred)
    isempty(dv) && error("No non-missing observations found for observable :$(obs_name).")

    lo = min(minimum(dv), minimum(pred))
    hi = max(maximum(dv), maximum(pred))
    lims = _pad_limits(lo, hi)

    _kw_dvp = merge(
        (xlabel = "Population prediction (PRED)", ylabel = "Observed (DV)"), kwargs_subplot)
    p = create_styled_plot(; style = style, _kw_dvp...)
    create_styled_scatter!(
        p, pred, dv; label = "", color = style.color_primary, style = style)
    create_styled_line!(p, collect(lims), collect(lims);
        color = style.color_reference, linestyle = :dash,
        linewidth = style.line_width_secondary, label = "", style = style)
    _set_limits!(p; xlim = lims, ylim = lims)

    fig = combine_plots([p]; ncols = 1, style = style, kwargs_layout...)
    return _save_plot!(fig, save_path)
end

"""
    plot_dv_ipred(res::FitResult; dm, observable, style, kwargs_subplot,
                  kwargs_layout, save_path, plot_path) -> Makie.Figure

Goodness-of-fit scatter plot of observed values (DV) against individual
predictions (IPRED). IPRED is the model prediction evaluated at each
individual's empirical Bayes estimates of the random effects. An identity
line is overlaid for reference.

# Keyword Arguments
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
- `observable`: name of the outcome variable, or `nothing` for the first.
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `kwargs_subplot`: additional Makie `Axis` attributes forwarded to the subplot.
- `kwargs_layout`: additional Makie `Figure` attributes forwarded to the layout.
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
- `plot_path::Union{Nothing, String} = nothing`: alias for `save_path`.
"""
function plot_dv_ipred(res::FitResult;
        dm::Union{Nothing, DataModel} = nothing,
        observable = nothing,
        style::PlotStyle = PlotStyle(),
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing)
    dm = _get_dm(res, dm)
    save_path = _resolve_plot_path(save_path, plot_path)
    obs_name = _get_observable(dm, observable)
    θ = get_params(res; scale = :untransformed)
    constants_re = _res_constants_re(res, NamedTuple())
    η_ebe = _default_random_effects(res, dm, constants_re, θ, Random.default_rng(), 1)

    dv, ipred = _collect_ipred_series(dm, obs_name, θ, η_ebe)
    isempty(dv) && error("No non-missing observations found for observable :$(obs_name).")

    lo = min(minimum(dv), minimum(ipred))
    hi = max(maximum(dv), maximum(ipred))
    lims = _pad_limits(lo, hi)

    _kw_dvi = merge((xlabel = "Individual prediction (IPRED)", ylabel = "Observed (DV)"),
        kwargs_subplot)
    p = create_styled_plot(; style = style, _kw_dvi...)
    create_styled_scatter!(
        p, ipred, dv; label = "", color = style.color_primary, style = style)
    create_styled_line!(p, collect(lims), collect(lims);
        color = style.color_reference, linestyle = :dash,
        linewidth = style.line_width_secondary, label = "", style = style)
    _set_limits!(p; xlim = lims, ylim = lims)

    fig = combine_plots([p]; ncols = 1, style = style, kwargs_layout...)
    return _save_plot!(fig, save_path)
end

"""
    plot_wres_pred(res::FitResult; dm, observable, style, kwargs_subplot,
                   kwargs_layout, save_path, plot_path) -> Makie.Figure

Population residual diagnostic plot: weighted population residuals (WRES)
against population predictions (PRED). WRES is defined as
`(DV - PRED) / sigma`, where `sigma` is the standard deviation of the
observation distribution at the population-mean random effects. A horizontal
zero reference line is overlaid.

# Keyword Arguments
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
- `observable`: name of the outcome variable, or `nothing` for the first.
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `kwargs_subplot`: additional Makie `Axis` attributes forwarded to the subplot.
- `kwargs_layout`: additional Makie `Figure` attributes forwarded to the layout.
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
- `plot_path::Union{Nothing, String} = nothing`: alias for `save_path`.
"""
function plot_wres_pred(res::FitResult;
        dm::Union{Nothing, DataModel} = nothing,
        observable = nothing,
        style::PlotStyle = PlotStyle(),
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing)
    dm = _get_dm(res, dm)
    save_path = _resolve_plot_path(save_path, plot_path)
    obs_name = _get_observable(dm, observable)
    θ = get_params(res; scale = :untransformed)
    η_pred = _pred_re_per_individual(dm, θ)

    dv, pred, sigma = _collect_pred_series(dm, obs_name, θ, η_pred)
    isempty(dv) && error("No non-missing observations found for observable :$(obs_name).")

    wres = (dv .- pred) ./ sigma

    pred_lo, pred_hi = _pad_limits(minimum(pred), maximum(pred))
    wres_lo, wres_hi = _pad_limits(minimum(wres), maximum(wres))

    _kw_wres = merge(
        (xlabel = "Population prediction (PRED)", ylabel = "Weighted residual (WRES)"),
        kwargs_subplot)
    p = create_styled_plot(; style = style, _kw_wres...)
    create_styled_scatter!(
        p, pred, wres; label = "", color = style.color_primary, style = style)
    add_reference_line!(p, 0.0; orientation = :horizontal,
        color = style.color_reference, linewidth = style.line_width_secondary, label = "")
    _set_limits!(p; xlim = (pred_lo, pred_hi), ylim = (wres_lo, wres_hi))

    fig = combine_plots([p]; ncols = 1, style = style, kwargs_layout...)
    return _save_plot!(fig, save_path)
end

"""
    plot_shrinkage(res::FitResult; dm, constants_re, threshold, style,
                   kwargs_subplot, save_path, plot_path) -> Makie.Figure

Horizontal bar chart of eta shrinkage for all scalar random effects, with a
vertical reference line at `threshold` (default 30 %).

Bars are colored by severity: below `threshold` (green), between `threshold`
and 50 % (orange), above 50 % (red). Shrinkage is computed via
[`compute_shrinkage`](@ref).

# Keyword Arguments
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
- `constants_re::NamedTuple = NamedTuple()`: random effects fixed at given values.
- `threshold::Real = 0.30`: reference line position (fraction, not percent).
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `kwargs_subplot`: additional Makie `Axis` attributes passed to the subplot.
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
- `plot_path::Union{Nothing, String} = nothing`: alias for `save_path`.
"""
function plot_shrinkage(res::FitResult;
        dm::Union{Nothing, DataModel} = nothing,
        constants_re::NamedTuple = NamedTuple(),
        threshold::Real = 0.30,
        bar_color::Union{Nothing, String} = nothing,
        style::PlotStyle = PlotStyle(),
        kwargs_subplot = NamedTuple(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing)
    save_path = _resolve_plot_path(save_path, plot_path)
    shrink_nt = compute_shrinkage(res; dm = dm, constants_re = constants_re)
    isempty(shrink_nt) &&
        error("No shrinkage values computed; check that the model has scalar random effects.")

    re_labels = [string(k) for k in keys(shrink_nt)]
    values_pct = [getfield(v, :shrinkage) * 100 for v in values(shrink_nt)]
    n = length(re_labels)

    bar_colors = if bar_color !== nothing
        fill(bar_color, n)
    else
        map(values_pct) do v
            v < threshold * 100 ? "#009E73" :
            v < 50.0 ? "#E69F00" : "#D55E00"
        end
    end

    p = create_styled_plot(;
        xlabel = "ETA shrinkage (%)",
        ylabel = "",
        title = "ETA shrinkage",
        style = style,
        kwargs_subplot...)

    yticks_pos = collect(1:n)
    bar_vals = max.(values_pct, 0.0)
    _record!(p,
        ax -> barplot!(ax, yticks_pos, bar_vals;
            direction = :x, color = bar_colors, width = 0.6))
    for i in 1:n
        add_annotation!(p, bar_vals[i] + 1.5, yticks_pos[i],
            string(round(values_pct[i]; digits = 1), "%"); fontsize = 9, halign = :left)
    end

    add_reference_line!(p, threshold * 100; orientation = :vertical,
        color = style.color_reference, linewidth = style.line_width_secondary,
        label = "$(round(Int, threshold*100))%")

    _axis_attrs!(p; yticks = (yticks_pos, re_labels))
    _set_limits!(p;
        xlim = (-2.0, max(maximum(values_pct) * 1.25 + 5.0, threshold * 100 * 1.5)),
        ylim = (0.5, n + 0.5))
    p.legend_position = :rb

    fig = combine_plots([p]; ncols = 1, style = style)
    return _save_plot!(fig, save_path)
end
