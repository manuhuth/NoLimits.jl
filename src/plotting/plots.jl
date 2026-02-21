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

function _merge_limits(lims, vals)
    isempty(vals) && return lims
    lo = minimum(vals)
    hi = maximum(vals)
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

function _collect_multivariate_series(x, y, n_marginals; marginal_idx::Union{Nothing, Int}=nothing)
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


function _plot_data_dm(dm::DataModel;
                       x_axis_feature::Union{Symbol, Nothing}=nothing,
                       individuals_idx=nothing,
                       shared_x_axis::Bool=true,
                       shared_y_axis::Bool=true,
                       ncols::Int=DEFAULT_PLOT_COLS,
                       style::PlotStyle=PlotStyle(),
                       kwargs_subplot=NamedTuple(),
                       kwargs_layout=NamedTuple(),
                       save_path::Union{Nothing, String}=nothing,
                       marginal_idx::Union{Nothing, Int}=nothing)
    obs_name = _get_observable(dm, nothing)
    inds = individuals_idx === nothing ? collect(eachindex(dm.individuals)) : collect(individuals_idx)
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
        title_id = string(dm.config.primary_id, ": ", dm.df[obs_rows[1], dm.config.primary_id])
        p = create_styled_plot(title=title_id,
                               xlabel=x_axis_feature === nothing ? "Time" : _axis_label(x_axis_feature),
                               ylabel=_axis_label(obs_name),
                               style=style,
                               kwargs_subplot...)
        if is_mv
            if marginal_idx === nothing
                xs, ys = _collect_multivariate_series(x, y, n_marginals)
                for m in 1:n_marginals
                    create_styled_scatter!(p, xs[m], ys[m];
                                           label=_marginal_label(obs_name, m),
                                           color=marginal_colors[m],
                                           style=style)
                    xlims = _merge_limits(xlims, xs[m])
                    ylims = _merge_limits(ylims, ys[m])
                end
            else
                xs, ys = _collect_multivariate_series(x, y, n_marginals; marginal_idx=marginal_idx)
                create_styled_scatter!(p, xs, ys;
                                       label=_marginal_label(obs_name, marginal_idx),
                                       color=marginal_colors[marginal_idx],
                                       style=style)
                xlims = _merge_limits(xlims, xs)
                ylims = _merge_limits(ylims, ys)
            end
        else
            create_styled_scatter!(p, x, y; label="", style=style)
            xlims = _merge_limits(xlims, x)
            ylims = _merge_limits(ylims, y)
        end
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
    plot_data(res::FitResult; dm, x_axis_feature, individuals_idx, shared_x_axis,
              shared_y_axis, ncols, style, kwargs_subplot, kwargs_layout,
              save_path, plot_path, marginal_layout) -> Plots.Plot | Vector{Plots.Plot}

    plot_data(dm::DataModel; x_axis_feature, individuals_idx, shared_x_axis,
              shared_y_axis, ncols, style, kwargs_subplot, kwargs_layout,
              save_path, plot_path, marginal_layout) -> Plots.Plot | Vector{Plots.Plot}

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
- `marginal_layout::Symbol = :single`: `:single` produces one figure with subplots per individual showing every marginal; `:vector` returns a figure per marginal (only valid when the observable is vector-valued and requires `save_path` and `plot_path` to be `nothing`).
- `kwargs_subplot`, `kwargs_layout`: extra keyword arguments for subplots and layout.
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot (ignored for `:vector` mode).
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
                   plot_path::Union{Nothing, String}=nothing,
                   marginal_layout::Symbol=:single)
    dm = _get_dm(res, dm)
    obs_name = _get_observable(dm, nothing)
    marginal_layout in (:single, :vector) || error("marginal_layout must be :single or :vector.")
    (is_mv, n_marginals) = _obs_multivariate_info(dm, obs_name)
    if is_mv && marginal_layout == :vector
        (save_path === nothing && plot_path === nothing) ||
            error("save_path/plot_path are not supported when returning multiple marginal figures.")
        plots = Vector{Plots.Plot}(undef, n_marginals)
        for m in 1:n_marginals
            plots[m] = _plot_data_dm(dm;
                                     x_axis_feature=x_axis_feature,
                                     individuals_idx=individuals_idx,
                                     shared_x_axis=shared_x_axis,
                                     shared_y_axis=shared_y_axis,
                                     ncols=ncols,
                                     style=style,
                                     kwargs_subplot=kwargs_subplot,
                                     kwargs_layout=kwargs_layout,
                                     save_path=nothing,
                                     marginal_idx=m)
        end
        return plots
    end
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
                   plot_path::Union{Nothing, String}=nothing,
                   marginal_layout::Symbol=:single)
    obs_name = _get_observable(dm, nothing)
    marginal_layout in (:single, :vector) || error("marginal_layout must be :single or :vector.")
    (is_mv, n_marginals) = _obs_multivariate_info(dm, obs_name)
    if is_mv && marginal_layout == :vector
        (save_path === nothing && plot_path === nothing) ||
            error("save_path/plot_path are not supported when returning multiple marginal figures.")
        plots = Vector{Plots.Plot}(undef, n_marginals)
        for m in 1:n_marginals
            plots[m] = _plot_data_dm(dm;
                                     x_axis_feature=x_axis_feature,
                                     individuals_idx=individuals_idx,
                                     shared_x_axis=shared_x_axis,
                                     shared_y_axis=shared_y_axis,
                                     ncols=ncols,
                                     style=style,
                                     kwargs_subplot=kwargs_subplot,
                                     kwargs_layout=kwargs_layout,
                                     save_path=nothing,
                                     marginal_idx=m)
        end
        return plots
    end
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
                   marginal_layout::Symbol=:single,
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
                   marginal_idx::Union{Nothing, Int}=nothing,
                   rng::AbstractRNG=Random.default_rng())
    dm = _get_dm(res, dm)
    save_path = _resolve_plot_path(save_path, plot_path)
    constants_re_use = _res_constants_re(res, constants_re)
    obs_name = _get_observable(dm, observable)
    (is_mv, detected_n_marginals) = _obs_multivariate_info(dm, obs_name)
    n_marginals = is_mv ? detected_n_marginals : 1
    marginal_layout in (:single, :vector) || error("marginal_layout must be :single or :vector.")
    marginal_idx !== nothing && (!is_mv) && error("marginal_idx only valid when the observable is multivariate.")
    if marginal_idx !== nothing
        (1 <= marginal_idx <= n_marginals) || error("marginal_idx must be between 1 and $(n_marginals).")
    end
    inds = individuals_idx === nothing ? collect(eachindex(dm.individuals)) : collect(individuals_idx)

    if cache === nothing
        cache = build_plot_cache(res; dm=dm, params=params, constants_re=constants_re_use,
                                 cache_obs_dists=cache_obs_dists, mcmc_draws=mcmc_draws,
                                 mcmc_warmup=mcmc_warmup, rng=rng)
    end

    if is_mv && marginal_layout == :vector && marginal_idx === nothing
        (save_path === nothing && plot_path === nothing) ||
            error("save_path/plot_path are not supported when returning multiple marginal figures.")
        plots_vector = Vector{Plots.Plot}(undef, n_marginals)
        for m in 1:n_marginals
            plots_vector[m] = plot_fits(res;
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
                                         marginal_layout=:single,
                                         style=style,
                                         kwargs_subplot=kwargs_subplot,
                                         kwargs_layout=kwargs_layout,
                                         save_path=nothing,
                                         plot_path=nothing,
                                         cache=cache,
                                         params=params,
                                         constants_re=constants_re,
                                         cache_obs_dists=cache_obs_dists,
                                         plot_mcmc_quantiles=plot_mcmc_quantiles,
                                         mcmc_quantiles=mcmc_quantiles,
                                         mcmc_quantiles_alpha=mcmc_quantiles_alpha,
                                         mcmc_draws=mcmc_draws,
                                         mcmc_warmup=mcmc_warmup,
                                         marginal_idx=m,
                                         rng=rng)
        end
        return plots_vector
    end

    plots = Vector{Any}(undef, length(inds))
    xlims = nothing
    ylims = nothing
    marginal_colors = _marginal_colors(max(n_marginals, 1), style)

    is_mcmc = _is_posterior_draw_fit(res)
    is_mv && is_mcmc && error("plot_fits currently does not support posterior draws for multivariate HMM observables.")
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
    if is_mv && plot_density
        @warn "plot_density is ignored for multivariate HMM observables."
        plot_density = false
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
        xs_per_margin = nothing
        ys_per_margin = nothing
        if is_mv
            xs_per_margin, ys_per_margin = _collect_multivariate_series(x_obs, y_obs, n_marginals)
        end
        if plot_data_points
            if is_mv
                margin_range = marginal_idx === nothing ? (1:n_marginals) : (marginal_idx,)
                for m in margin_range
                    create_styled_scatter!(p, xs_per_margin[m], ys_per_margin[m];
                                           label=_marginal_label(obs_name, m),
                                           color=marginal_colors[m],
                                           style=style)
                end
            else
                create_styled_scatter!(p, x_obs, y_obs; label="data", color=style.color_primary, style=style)
            end
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
                    obs = calculate_formulas_obs(dm.model, θ, η_ind, ind.const_cov, vary, sol_accessors)
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
                    create_styled_line!(p, x_fit, curve; label=label, color=color, style=style)
                    y_min = minimum(curve)
                    y_max = maximum(curve)
                    ylims = ylims === nothing ? (y_min, y_max) : (min(ylims[1], y_min), max(ylims[2], y_max))
                end
            else
                for (j, row) in enumerate(obs_rows)
                    vary = _varying_at_plot(dm, ind, j, row)
                    obs = sol_accessors === nothing ?
                          calculate_formulas_obs(dm.model, θ, η_ind, ind.const_cov, vary) :
                          calculate_formulas_obs(dm.model, θ, η_ind, ind.const_cov, vary, sol_accessors)
                    dist = getproperty(obs, obs_name)
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
                    create_styled_line!(p, x_fit, curve; label=label, color=color, style=style)
                    y_min = minimum(curve)
                    y_max = maximum(curve)
                    ylims = ylims === nothing ? (y_min, y_max) : (min(ylims[1], y_min), max(ylims[2], y_max))
                end
            end

            if plot_density && !is_mv
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
        observed_values = is_mv ? [val for vec in ys_per_margin for val in vec] : y_obs
        ylims = _merge_limits(ylims, observed_values)
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
                                  individuals_idx=nothing)
    inds = individuals_idx === nothing ? collect(eachindex(dm.individuals)) : collect(individuals_idx)
    plots = Vector{Any}(undef, length(inds))
    xlims = nothing
    ylims = nothing
    for (k, i) in enumerate(inds)
        ind = dm.individuals[i]
        obs_rows = dm.row_groups.obs_rows[i]
        x_vals = _get_x_values(dm, ind, obs_rows, x_axis_feature)
        title_id = string(dm.config.primary_id, ": ", dm.df[obs_rows[1], dm.config.primary_id])
        p = create_styled_plot(title=title_id,
                               xlabel=x_axis_feature === nothing ? "Time" : _axis_label(x_axis_feature),
                               ylabel="Hidden-state probability",
                               style=style,
                               kwargs_subplot...)
        θ_ind = θ
        η_ind = η_vec[i]
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
            sol = dm.model.de.de !== nothing ? _solve_dense_individual(dm, ind, θ_ind, η_ind)[1] : nothing
            sol_accessors = get_de_accessors_builder(dm.model.de.de)(sol, compiled)
        end

        times = Float64[]
        posteriors = Vector{Vector{Float64}}()
        n_states = nothing
        for (j, row) in enumerate(obs_rows)
            vary = _varying_at_plot(dm, ind, j, row)
            obs = sol_accessors === nothing ?
                  calculate_formulas_obs(dm.model, θ_ind, η_ind, ind.const_cov, vary) :
                  calculate_formulas_obs(dm.model, θ_ind, η_ind, ind.const_cov, vary, sol_accessors)
            dist = getproperty(obs, obs_name)
            dist isa MVDiscreteTimeDiscreteStatesHMM || error("Observable $(obs_name) must be MVDiscreteTimeDiscreteStatesHMM.")
            y_val = getfield(ind.series.obs, obs_name)[j]
            y_val === missing && continue
            if n_states === nothing
                n_states = dist.n_states
            end
            push!(times, x_vals[j])
            post = posterior_hidden_states(dist, y_val)
            push!(posteriors, post)
        end

        isempty(times) && @warn "No non-missing observations found for individual $(title_id)."
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
                x_poly = Float64[]
                y_poly = Float64[]
                for (idx, t) in enumerate(times)
                    if prob_mat[m, idx] == 0
                        continue
                    end
                    x_left = t - half_width
                    x_right = t + half_width
                    y_bot = bottom[idx]
                    y_top = y_bot + prob_mat[m, idx]
                    append!(x_poly, (x_left, x_right, x_right, x_left, NaN))
                    append!(y_poly, (y_bot, y_bot, y_top, y_top, NaN))
                end
                isempty(x_poly) && continue
                plot!(p, x_poly, y_poly;
                      seriestype=:shape,
                      fillcolor=state_colors[m],
                      linecolor=:transparent,
                      label=state_labels[m])
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
        xlim_use = shared_x_axis && xlims !== nothing ? _pad_limits(xlims[1], xlims[2]) : nothing
        ylim_use = shared_y_axis ? (0.0, 1.0) : nothing
        _apply_shared_axes!(plots, xlim_use, ylim_use)
    end
    p = combine_plots(plots; ncols=ncols, kwargs_layout...)
    return _save_plot!(p, save_path)
end

function plot_hidden_states(res::FitResult;
                            dm::Union{Nothing, DataModel}=nothing,
                            observable=nothing,
                            individuals_idx=nothing,
                            x_axis_feature::Union{Nothing, Symbol}=nothing,
                            shared_x_axis::Bool=true,
                            shared_y_axis::Bool=true,
                            ncols::Int=DEFAULT_PLOT_COLS,
                            figure_layout::Symbol=:single,
                            style::PlotStyle=PlotStyle(),
                            kwargs_subplot=NamedTuple(),
                            kwargs_layout=NamedTuple(),
                            save_path::Union{Nothing, String}=nothing,
                            plot_path::Union{Nothing, String}=nothing,
                            params::NamedTuple=NamedTuple(),
                            constants_re::NamedTuple=NamedTuple(),
                            mcmc_draws::Int=1,
                            rng::AbstractRNG=Random.default_rng())
    dm = _get_dm(res, dm)
    figure_layout in (:single, :vector) || error("figure_layout must be :single or :vector.")
    if figure_layout == :vector
        (save_path === nothing && plot_path === nothing) ||
            error("save_path/plot_path are not supported when returning multiple figures.")
    end
    save_path = _resolve_plot_path(save_path, plot_path)
    constants_re_use = _res_constants_re(res, constants_re)
    obs_name = _get_observable(dm, observable)
    (is_mv, _) = _obs_multivariate_info(dm, obs_name)
    is_mv || error("plot_hidden_states requires a multivariate observable.")
    _is_posterior_draw_fit(res) && error("plot_hidden_states does not support posterior draws yet.")

    θ = get_params(res; scale=:untransformed)
    θ = _apply_param_overrides(θ, params)
    η_vec = _default_random_effects(res, dm, constants_re_use, θ, rng, mcmc_draws)
    inds = individuals_idx === nothing ? collect(eachindex(dm.individuals)) : collect(individuals_idx)

    if figure_layout == :vector
        plots = Vector{Plots.Plot}(undef, length(inds))
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
                            observable=nothing,
                            individuals_idx=nothing,
                            x_axis_feature::Union{Nothing, Symbol}=nothing,
                            shared_x_axis::Bool=true,
                            shared_y_axis::Bool=true,
                            ncols::Int=DEFAULT_PLOT_COLS,
                            figure_layout::Symbol=:single,
                            style::PlotStyle=PlotStyle(),
                            kwargs_subplot=NamedTuple(),
                            kwargs_layout=NamedTuple(),
                            save_path::Union{Nothing, String}=nothing,
                            plot_path::Union{Nothing, String}=nothing,
                            params::NamedTuple=NamedTuple(),
                            constants_re::NamedTuple=NamedTuple(),
                            rng::AbstractRNG=Random.default_rng())
    figure_layout in (:single, :vector) || error("figure_layout must be :single or :vector.")
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
    inds = individuals_idx === nothing ? collect(eachindex(dm.individuals)) : collect(individuals_idx)

    if figure_layout == :vector
        plots = Vector{Plots.Plot}(undef, length(inds))
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

function _resolve_emission_row(dm::DataModel,
                               obs_rows::Vector{Int},
                               time_idx::Union{Nothing, Int},
                               time_point,
                               time_col::Symbol)
    length(obs_rows) == 0 && error("No observation rows found.")
    if time_idx !== nothing
        (1 <= time_idx <= length(obs_rows)) || error("time_idx must be between 1 and $(length(obs_rows)).")
        return obs_rows[time_idx]
    end
    if time_point !== nothing
        vals = dm.df[obs_rows, time_col]
        numeric = [ismissing(v) ? missing : float(v) for v in vals]
        has_missing = any(ismissing, numeric)
        has_missing && error("Time column $(time_col) contains missing values for individual observations.")
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
        emission isa MvNormal || error("Emission distributions must be MvNormal when joint.")
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
    row_pos === nothing && error("Observation row $(row) not found for individual $(ind_idx).")

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
        sol_accessors = get_de_accessors_builder(dm.model.de.de)(sol, compiled)
    end

    vary = _varying_at_plot(dm, ind, row_pos, row)
    obs = sol_accessors === nothing ?
          calculate_formulas_obs(dm.model, θ, η_ind, ind.const_cov, vary) :
          calculate_formulas_obs(dm.model, θ, η_ind, ind.const_cov, vary, sol_accessors)
    dist = getproperty(obs, obs_name)
    dist isa MVDiscreteTimeDiscreteStatesHMM || error("Observable $(obs_name) must be MVDiscreteTimeDiscreteStatesHMM.")

    n_states = dist.n_states
    n_marginals = _mv_n_outcomes(dist.emission_dists[1])
    state_plots = Vector{Plots.Plot}(undef, n_states)
    marginal_colors = _marginal_colors(n_marginals, style)
    overall_xlim = nothing
    overall_ylim = nothing

    for s in 1:n_states
        p_state = create_styled_plot(title="$title_base • State $s",
                                     xlabel="Outcome value",
                                     ylabel="Density",
                                     style=style,
                                     kwargs_subplot...)
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
                                       label=label,
                                       color=marginal_colors[m],
                                       style=style)
                state_xlim = _merge_limits(state_xlim, grid.vals)
                state_ylim = _merge_limits(state_ylim, grid.probs)
            else
                grid = _density_grid_continuous([dist_m], 0.995, 200)
                grid === nothing && continue
                densities = vec(grid.z[:, 1])
                create_styled_line!(p_state, grid.y, densities;
                                    label=label,
                                    color=marginal_colors[m],
                                    style=style)
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

    combined = combine_plots(state_plots; ncols=min(state_ncols, n_states))
    return (plot=combined, xlims=overall_xlim, ylims=overall_ylim)
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
    inds = individuals_idx === nothing ? collect(eachindex(dm.individuals)) : collect(individuals_idx)
    plots = Vector{Plots.Plot}(undef, length(inds))
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
        plots[k] = record.plot
        xlims = _merge_limits(xlims, record.xlims === nothing ? () : record.xlims)
        ylims = _merge_limits(ylims, record.ylims === nothing ? () : record.ylims)
    end

    if shared_y_axis && ylims !== nothing
        _apply_shared_axes!(plots, nothing, _pad_limits(ylims[1], ylims[2]))
    end

    if figure_layout == :vector
        return plots
    end
    p = combine_plots(plots; ncols=ncols, kwargs_layout...)
    return _save_plot!(p, save_path)
end

function plot_emission_distributions(res::FitResult;
                                     dm::Union{Nothing, DataModel}=nothing,
                                     observable=nothing,
                                     individuals_idx=nothing,
                                     time_idx::Union{Nothing, Int}=nothing,
                                     time_point=nothing,
                                     time_col::Union{Nothing, Symbol}=nothing,
                                     shared_y_axis::Bool=true,
                                     ncols::Int=DEFAULT_PLOT_COLS,
                                     figure_layout::Symbol=:single,
                                     style::PlotStyle=PlotStyle(),
                                     kwargs_subplot=NamedTuple(),
                                     kwargs_layout=NamedTuple(),
                                     save_path::Union{Nothing, String}=nothing,
                                     plot_path::Union{Nothing, String}=nothing,
                                     params::NamedTuple=NamedTuple(),
                                     constants_re::NamedTuple=NamedTuple(),
                                     mcmc_draws::Int=1,
                                     rng::AbstractRNG=Random.default_rng())
    figure_layout in (:single, :vector) || error("figure_layout must be :single or :vector.")
    (time_idx !== nothing) && (time_point !== nothing) && error("Specify only one of time_idx or time_point.")
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
    _is_posterior_draw_fit(res) && error("plot_emission_distributions does not support posterior draws.")

    θ = get_params(res; scale=:untransformed)
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
                                     observable=nothing,
                                     individuals_idx=nothing,
                                     time_idx::Union{Nothing, Int}=nothing,
                                     time_point=nothing,
                                     time_col::Union{Nothing, Symbol}=nothing,
                                     shared_y_axis::Bool=true,
                                     ncols::Int=DEFAULT_PLOT_COLS,
                                     figure_layout::Symbol=:single,
                                     style::PlotStyle=PlotStyle(),
                                     kwargs_subplot=NamedTuple(),
                                     kwargs_layout=NamedTuple(),
                                     save_path::Union{Nothing, String}=nothing,
                                     plot_path::Union{Nothing, String}=nothing,
                                     params::NamedTuple=NamedTuple(),
                                     constants_re::NamedTuple=NamedTuple(),
                                     rng::AbstractRNG=Random.default_rng())
    figure_layout in (:single, :vector) || error("figure_layout must be :single or :vector.")
    (time_idx !== nothing) && (time_point !== nothing) && error("Specify only one of time_idx or time_point.")
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
                   plot_density::Bool=false,
                   plot_func=mean,
                   plot_data_points::Bool=true,
                   observable=nothing,
                   individuals_idx=nothing,
                   x_axis_feature::Union{Nothing, Symbol}=nothing,
                   shared_x_axis::Bool=true,
                   shared_y_axis::Bool=true,
                   ncols::Int=DEFAULT_PLOT_COLS,
                   marginal_layout::Symbol=:single,
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
                     marginal_layout=marginal_layout,
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
