"""
    plot_observation_distributions(res::FitResult; dm, individuals_idx, obs_rows,
                                   observables, x_axis_feature, shared_x_axis,
                                   shared_y_axis, ncols, style, cache, cache_obs_dists,
                                   constants_re, mcmc_quantiles, mcmc_quantiles_alpha,
                                   mcmc_draws, mcmc_warmup, rng, save_path,
                                   kwargs_subplot, kwargs_layout) -> Makie.Figure

Plot the predictive observation distributions at each time point as density or PMF
curves overlaid on the observed data, providing a detailed look at model calibration.

One panel per (individual, observation row, vector component), titled by ID and time,
with one legend below the panels. Shared axes are linked and their inner tick labels
hidden; panels keep their own y label when PMF and PDF panels are mixed, and x is only
linked when all panels show the same individual and outcome (x ranges are shared per
individual and outcome).

# Keyword Arguments
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
- `individuals_idx`: individuals to include (default: the first individual in data order).
- `obs_rows`: specific observation row indices, or `nothing` for all.
- `observables`: outcome name(s) to include, or `nothing` for first.
- `x_axis_feature::Union{Nothing, Symbol} = nothing`: covariate for the x-axis.
- `shared_x_axis::Bool = true`, `shared_y_axis::Bool = true`: share axes.
- `ncols::Int = 3`: number of subplot columns.
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `cache::Union{Nothing, PlotCache} = nothing`: pre-computed plot cache.
- `cache_obs_dists::Bool = false`: pre-compute observation distributions when building cache.
- `constants_re::NamedTuple = NamedTuple()`: random-effect constants.
- `mcmc_quantiles`, `mcmc_quantiles_alpha`, `mcmc_draws`, `mcmc_warmup`: MCMC settings.
- `rng::AbstractRNG = Xoshiro(42)`: random-number generator. A fresh generator with
  this fixed seed is created per call, so repeated calls draw the same values.
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
- `kwargs_subplot`: additional Makie `Axis` attributes forwarded to each subplot.
- `kwargs_layout`: additional Makie `Figure` attributes forwarded to the combined layout.
"""
function plot_observation_distributions(
        res::FitResult;
        dm::Union{Nothing, DataModel} = nothing,
        individuals_idx = nothing,
        obs_rows = nothing,
        observables = nothing,
        x_axis_feature::Union{Nothing, Symbol} = nothing,
        shared_x_axis::Bool = true,
        shared_y_axis::Bool = true,
        ncols::Int = DEFAULT_PLOT_COLS,
        style::PlotStyle = PlotStyle(),
        cache::Union{Nothing, PlotCache} = nothing,
        cache_obs_dists::Bool = false,
        constants_re::NamedTuple = NamedTuple(),
        mcmc_quantiles::Vector{<:Real} = [5, 95],
        mcmc_quantiles_alpha::Float64 = 0.8,
        mcmc_draws::Int = 1000,
        mcmc_warmup::Union{Nothing, Int} = nothing,
        rng::AbstractRNG = _default_rng(),
        save_path::Union{Nothing, String} = nothing,
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple()
    )
    dm = _get_dm(res, dm)
    _check_unit_interval(mcmc_quantiles_alpha, "mcmc_quantiles_alpha")
    constants_re_use = _res_constants_re(res, constants_re)
    obs_list = _resolve_observables(dm, observables)
    # default: the first individual in data order (DataModel sorts ids)
    inds = _plot_individuals(dm, individuals_idx; default_all = false)
    if get_de(get_model(dm)) === nothing
        x_axis_feature = _require_varying_covariate(dm, x_axis_feature)
    end

    is_mcmc = _is_posterior_draw_fit(res)
    if !is_mcmc && cache === nothing
        cache = build_plot_cache(
            res; dm = dm, constants_re = constants_re_use,
            cache_obs_dists = cache_obs_dists, rng = rng
        )
    end

    plots = Vector{Any}()
    plot_groups = Vector{Tuple{Int, Symbol, Int}}()
    xlims_by_group = Dict{Tuple{Int, Symbol, Int}, Tuple{Float64, Float64}}()
    ylims = nothing

    θ_draws = nothing
    η_draws = nothing
    if is_mcmc
        res = _with_posterior_warmup(res, mcmc_warmup)
        θ_draws, η_draws, _ = _posterior_drawn_params(
            res, dm, constants_re_use, NamedTuple(), mcmc_draws, rng
        )
        mcmc_quantiles = sort(Float64.(collect(mcmc_quantiles)))
        (length(mcmc_quantiles) >= 2 && all(0 .<= mcmc_quantiles .<= 100)) ||
            error("mcmc_quantiles must be in [0,100] with length >= 2.")
    end

    for i in inds
        ind = get_individuals(dm)[i]
        obs_rows_all = get_obs_rows(get_row_groups(dm))[i]
        obs_idx = _resolve_obs_rows(obs_rows, obs_rows_all)
        rowwise_re = _needs_rowwise_random_effects(dm, i; obs_only = true)

        for obs_name in obs_list
            (is_mv, n_marginals) = _obs_multivariate_info(dm, obs_name)
            # One marginal PDF/PMF panel per component; m == 0 is the scalar case.
            marg_range = is_mv ? (1:n_marginals) : (0:0)
            for j in obs_idx, m in marg_range
                marginal = m == 0 ? nothing : m
                row = obs_rows_all[j]
                y_obs = _obs_component(
                    getfield(get_obs(get_series(ind)), obs_name)[j], marginal
                )
                has_obs_val = y_obs isa Number && isfinite(float(y_obs))
                tval = get_df(dm)[row, get_time_col(dm)]
                p = _ObsDistPanel(
                    string(
                        get_primary_id(dm), ": ", get_df(dm)[row, get_primary_id(dm)], ", ",
                        get_time_col(dm), ": ", tval
                    ),
                    _axis_label(
                        marginal === nothing ? string(obs_name) :
                            _marginal_label(obs_name, m)
                    )
                )

                if is_mcmc
                    n_draws = length(θ_draws)
                    dists = Vector{Distribution}(undef, n_draws)
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
                        dists[d] = _marginal_obs_dist(
                            getproperty(obs, obs_name), obs_name, marginal, n_marginals
                        )
                    end

                    if dists[1] isa DiscreteDistribution
                        vals = _mean_pmf_support(dists, 0.995)
                        isempty(vals) &&
                            error("Unable to build PMF support for $(obs_name).")
                        probs = zeros(Float64, length(vals))
                        probs_all = zeros(Float64, n_draws, length(vals))
                        for d in 1:n_draws
                            pvec = pdf.(Ref(dists[d]), vals)
                            probs_all[d, :] .= pvec
                            probs .+= pvec
                        end
                        probs ./= n_draws
                        qlo = mapslices(
                            x -> quantile(vec(x), mcmc_quantiles[1] / 100),
                            probs_all; dims = 1
                        )
                        qhi = mapslices(
                            x -> quantile(vec(x), mcmc_quantiles[end] / 100),
                            probs_all; dims = 1
                        )
                        qlo = vec(qlo)
                        qhi = vec(qhi)
                        _push_xy!(p.bars, "", vals, probs, "posterior mean PMF")
                        _push_xy!(p.quantiles, "", vals, qlo, "$(mcmc_quantiles[1])%")
                        _push_xy!(p.quantiles, "", vals, qhi, "$(mcmc_quantiles[end])%")
                        p.ylabel = "Probability Mass"
                        has_obs_val && _push_xy!(p.observed, "", [y_obs], [NaN], "observed")
                        xlim = (minimum(vals), maximum(vals))
                        if has_obs_val
                            xlim = (min(xlim[1], float(y_obs)), max(xlim[2], float(y_obs)))
                        end
                        xlims_by_group[(i, obs_name, m)] = haskey(
                                xlims_by_group, (i, obs_name, m)
                            ) ?
                            (
                                min(xlims_by_group[(i, obs_name, m)][1], xlim[1]),
                                max(xlims_by_group[(i, obs_name, m)][2], xlim[2]),
                            ) :
                            xlim
                        ylims = ylims === nothing ? (minimum(qlo), maximum(qhi)) :
                            (min(ylims[1], minimum(qlo)), max(ylims[2], maximum(qhi)))
                    else
                        grid = _density_grid_continuous(dists, 0.99, 200)
                        grid === nothing &&
                            error("Unable to build PDF grid for $(obs_name).")
                        pdf_mean = vec(mean(grid.z, dims = 2))
                        qlo = mapslices(
                            x -> quantile(vec(x), mcmc_quantiles[1] / 100),
                            grid.z; dims = 2
                        )
                        qhi = mapslices(
                            x -> quantile(vec(x), mcmc_quantiles[end] / 100),
                            grid.z; dims = 2
                        )
                        qlo = vec(qlo)
                        qhi = vec(qhi)
                        _push_xy!(p.curves, "", grid.y, pdf_mean, "posterior mean PDF")
                        _push_xy!(p.quantiles, "", grid.y, qlo, "$(mcmc_quantiles[1])%")
                        _push_xy!(p.quantiles, "", grid.y, qhi, "$(mcmc_quantiles[end])%")
                        has_obs_val && _push_xy!(p.observed, "", [y_obs], [NaN], "observed")
                        xlim = (minimum(grid.y), maximum(grid.y))
                        if has_obs_val
                            xlim = (min(xlim[1], float(y_obs)), max(xlim[2], float(y_obs)))
                        end
                        xlims_by_group[(i, obs_name, m)] = haskey(
                                xlims_by_group, (i, obs_name, m)
                            ) ?
                            (
                                min(xlims_by_group[(i, obs_name, m)][1], xlim[1]),
                                max(xlims_by_group[(i, obs_name, m)][2], xlim[2]),
                            ) :
                            xlim
                        ylims = ylims === nothing ? (minimum(qlo), maximum(qhi)) :
                            (min(ylims[1], minimum(qlo)), max(ylims[2], maximum(qhi)))
                    end
                else
                    θ = cache.params
                    η_ind = cache.random_effects[i]
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
                                preDE = calculate_prede(
                                    get_model(dm), θ, η_ind, get_const_cov(ind)
                                ),
                            )
                        )
                        sol_accessors = _sol_accessors_with_crossings(
                            get_model(dm), sol, compiled, θ, η_ind, get_const_cov(ind)
                        )
                    end
                    dist_raw = if cache_obs_dists && cache.obs_dists !== nothing
                        getproperty(cache.obs_dists[i][j], obs_name)
                    else
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
                        getproperty(obs, obs_name)
                    end
                    dist = _marginal_obs_dist(dist_raw, obs_name, marginal, n_marginals)

                    if dist isa DiscreteDistribution
                        if dist isa Bernoulli
                            vals = [0, 1]
                        else
                            grid = _density_grid_discrete(dist, 0.995)
                            grid === nothing &&
                                error("Unable to build PMF grid for $(obs_name).")
                            vals = grid.vals
                        end
                        probs = pdf.(Ref(dist), vals)
                        _push_xy!(p.bars, "", vals, probs, "PMF")
                        p.ylabel = "Probability Mass"
                        has_obs_val && _push_xy!(p.observed, "", [y_obs], [NaN], "observed")
                        xlim = (minimum(vals), maximum(vals))
                        if has_obs_val
                            xlim = (min(xlim[1], float(y_obs)), max(xlim[2], float(y_obs)))
                        end
                        xlims_by_group[(i, obs_name, m)] = haskey(
                                xlims_by_group, (i, obs_name, m)
                            ) ?
                            (
                                min(xlims_by_group[(i, obs_name, m)][1], xlim[1]),
                                max(xlims_by_group[(i, obs_name, m)][2], xlim[2]),
                            ) :
                            xlim
                        ylims = ylims === nothing ? (minimum(probs), maximum(probs)) :
                            (
                                min(ylims[1], minimum(probs)), max(ylims[2], maximum(probs)),
                            )
                    else
                        grid = _density_grid_continuous([dist], 0.99, 200)
                        grid === nothing &&
                            error("Unable to build PDF grid for $(obs_name).")
                        pdf_vals = vec(grid.z[:, 1])
                        _push_xy!(p.curves, "", grid.y, pdf_vals, "PDF")
                        has_obs_val && _push_xy!(p.observed, "", [y_obs], [NaN], "observed")
                        xlim = (minimum(grid.y), maximum(grid.y))
                        if has_obs_val
                            xlim = (min(xlim[1], float(y_obs)), max(xlim[2], float(y_obs)))
                        end
                        xlims_by_group[(i, obs_name, m)] = haskey(
                                xlims_by_group, (i, obs_name, m)
                            ) ?
                            (
                                min(xlims_by_group[(i, obs_name, m)][1], xlim[1]),
                                max(xlims_by_group[(i, obs_name, m)][2], xlim[2]),
                            ) :
                            xlim
                        ylims = ylims === nothing ? (minimum(pdf_vals), maximum(pdf_vals)) :
                            (
                                min(ylims[1], minimum(pdf_vals)),
                                max(ylims[2], maximum(pdf_vals)),
                            )
                    end
                end

                push!(plots, p)
                push!(plot_groups, (i, obs_name, m))
            end
        end
    end

    isempty(plots) && error("Nothing to plot: no observation rows were selected.")
    xlims_panels = shared_x_axis ?
        [(l = xlims_by_group[key]; _pad_limits(l[1], l[2])) for key in plot_groups] :
        fill(nothing, length(plots))
    ylim = (shared_y_axis && ylims !== nothing) ? _pad_limits(ylims[1], ylims[2]) : nothing
    panels = [
        _obsdist_aog_panel(p, style, kwargs_subplot, mcmc_quantiles_alpha, xl, ylim)
            for (p, xl) in zip(plots, xlims_panels)
    ]
    # #383 layout defaults: y is shared across all panels and linked (inner labels hidden)
    # only when every panel has the same y label (PMF and PDF panels keep their own); x is
    # shared within one (individual, observable, component) group and linked only when all
    # panels belong to one group.
    fig = combine_plots(
        panels; ncols = ncols, style = style,
        link_x = shared_x_axis && length(panels) > 1 && allequal(plot_groups),
        link_y = shared_y_axis && ylims !== nothing && allequal(p.ylabel for p in plots),
        _nt(kwargs_layout)...
    )
    return _save_plot!(fig, save_path)
end

# Series of one observation-distribution panel, collected before drawing (tidy tables in
# the `_tidy()` layout; `observed` holds the observed value in `x`).
mutable struct _ObsDistPanel
    title::String
    xlabel::String
    ylabel::String
    curves::DataFrame
    quantiles::DataFrame
    bars::DataFrame
    observed::DataFrame
end

_ObsDistPanel(title, xlabel) = _ObsDistPanel(
    String(title), String(xlabel), "Probability Density", _tidy(), _tidy(), _tidy(), _tidy()
)

function _obsdist_aog_panel(p::_ObsDistPanel, style, kwargs_subplot, q_alpha, xlim, ylim)
    xy = AoG.mapping(:x, :y; color = :series)
    layers = _layers(
        _layer(p.bars, xy, AoG.visual(Makie.BarPlot)),
        _layer(p.curves, xy, _line_visual(style)),
        _layer(p.quantiles, xy, _line_visual(style; linestyle = :dash, alpha = q_alpha)),
        _layer(p.observed, AoG.mapping(:x; color = :series), AoG.visual(Makie.VLines; linestyle = :dash)),
    )
    palette = Pair{String, Any}[]
    for df in (p.bars, p.curves, p.quantiles), s in unique(df.series)
        push!(palette, s => style.color_secondary)
    end
    nrow(p.observed) > 0 && push!(palette, "observed" => style.color_primary)
    post = Any[]
    xlim === nothing || push!(post, ax -> xlims!(ax, xlim...))
    ylim === nothing || push!(post, ax -> ylims!(ax, ylim...))
    return AoGPanel(
        layers;
        scales = isempty(palette) ? NamedTuple() : (; Color = _series_palette(palette)),
        axis = _aog_axis(style, kwargs_subplot; title = p.title, xlabel = p.xlabel, ylabel = p.ylabel),
        post = post
    )
end

function plot_observation_distributions(dm::DataModel; kwargs...)
    constants_re = haskey(kwargs, :constants_re) ? kwargs[:constants_re] : NamedTuple()
    cache = build_plot_cache(dm; constants_re = constants_re, cache_obs_dists = false)
    res = FitResult(
        MLE(), FrequentistResult(NamedTuple(), 0.0, 0, NamedTuple(), NamedTuple()),
        FitSummary(
            0.0, true, FitParameters(ComponentArray(), ComponentArray()), NamedTuple()
        ),
        FitDiagnostics((;), (;), (;), (;)), dm, (), NamedTuple()
    )
    return plot_observation_distributions(res; dm = dm, cache = cache, kwargs...)
end
