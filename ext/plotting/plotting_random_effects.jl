# Random-effect diagnostics on AlgebraOfGraphics: plot_random_effects_pdf,
# plot_random_effects_scatter, plot_random_effect_pairplot,
# plot_random_effect_distributions, plot_random_effect_pit,
# plot_random_effect_standardized, plot_random_effect_standardized_scatter.
#
# Every function except the pairplot collects its panels into one tidy table
# (`_REFacets`) and draws them as AoG facets through `_draw_facets` (one figure legend,
# panels in creation order, linked axes when shared). The pairplot is a scatter matrix
# per random-effect grouping column, drawn cell by cell with `AoG.draw!`.

# --- Facet accumulator (group prefix `_re_`; candidates for the core layer) -------------

# Tidy table of all panels of one figure. `kind` selects the visual (see `_re_layers`):
# :bar (touching histogram bars), :pmf (PMF bars), :line, :qline (dashed MCMC quantile
# lines), :ref (dashed reference line), :scatter, :vline (dashed vertical line, x only).
struct _REFacets
    df::DataFrame
    panels::Vector{String}
    ylim::Dict{String, Tuple{Float64, Float64}}
    xlabel::Dict{String, String}
end

_REFacets() = _REFacets(
    _tidy(; grp = String, kind = Symbol), String[],
    Dict{String, Tuple{Float64, Float64}}(), Dict{String, String}()
)

# Append one series to `panel` (the panel is registered on first data, so panels that
# end up without data never appear).
function _re_push!(
        acc::_REFacets, panel::AbstractString, xs, ys, series::AbstractString, kind::Symbol;
        grp::AbstractString = series
    )
    panel in acc.panels || push!(acc.panels, String(panel))
    for (x, y) in zip(xs, ys)
        push!(acc.df, (String(panel), _f64(x), _f64(y), String(series), String(grp), kind); promote = false)
    end
    return acc
end

_re_vline!(acc::_REFacets, panel, x, series) = _re_push!(acc, panel, [x], [NaN], series, :vline)

# MCMC quantile band drawn as two dashed lines (separate line groups).
function _re_qlines!(acc::_REFacets, panel, xs, qlo, qhi, label)
    _re_push!(acc, panel, xs, qlo, label, :qline; grp = label * " lo")
    _re_push!(acc, panel, xs, qhi, label, :qline; grp = label * " hi")
    return acc
end

_re_qfmt(q) = isinteger(q) ? string(Int(q)) : string(q)
_re_qlabel(qs) = string(_re_qfmt(first(qs)), "-", _re_qfmt(last(qs)), "% quantiles")

# Per-draw curves (columns) -> mean and the outer quantiles across draws.
function _re_curve_summary(mat::AbstractMatrix, qs)
    m = vec(mean(mat; dims = 2))
    lo = vec(mapslices(x -> quantile(x, qs[1] / 100), mat; dims = 2))
    hi = vec(mapslices(x -> quantile(x, qs[end] / 100), mat; dims = 2))
    return m, lo, hi
end

# Probability histogram of each sample on fixed uniform edges (columns: draws).
function _re_fixed_hist(samples_by_draw, lo::Real, hi::Real, nbins::Int)
    out = Vector{Vector{Float64}}()
    for samp in samples_by_draw
        counts = zeros(Float64, nbins)
        for s in samp
            idx = s == hi ? nbins :
                clamp(floor(Int, (s - lo) / (hi - lo) * nbins) + 1, 1, nbins)
            counts[idx] += 1
        end
        ssum = sum(counts)
        ssum == 0 && continue
        push!(out, counts ./ ssum)
    end
    return out
end

_re_sub(df::DataFrame, k::Symbol) = df[df.kind .== k, :]

function _re_layers(acc::_REFacets, style::PlotStyle; qalpha::Real = 0.8, bar_kw = NamedTuple())
    df = acc.df
    m = AoG.mapping(:x, :y; color = :series, layout = :panel)
    mg = AoG.mapping(:x, :y; color = :series, group = :grp, layout = :panel)
    # Histogram bars: one layer per panel with the panel's bin width (a single layer would
    # size every bar by the smallest x spacing across all panels).
    bars = _re_sub(df, :bar)
    bar_layers = Any[]
    for panel in acc.panels
        sub = bars[bars.panel .== panel, :]
        nrow(sub) == 0 && continue
        xs = sort(unique(sub.x))
        w = length(xs) >= 2 ? minimum(diff(xs)) : 1.0
        push!(
            bar_layers, _layer(
                sub, m, AoG.visual(
                    Makie.BarPlot; merge((; gap = 0, strokewidth = 0, width = w), _nt(bar_kw))...
                )
            )
        )
    end
    return _layers(
        bar_layers...,
        _layer(_re_sub(df, :pmf), m, AoG.visual(Makie.BarPlot)),
        _layer(_re_sub(df, :line), mg, _line_visual(style)),
        _layer(_re_sub(df, :qline), mg, _line_visual(style; linestyle = :dash, alpha = qalpha)),
        _layer(_re_sub(df, :ref), mg, _line_visual(style; linestyle = :dash)),
        _layer(_re_sub(df, :scatter), m, _scatter_visual(style)),
        _layer(
            _re_sub(df, :vline), AoG.mapping(:x; color = :series, layout = :panel),
            AoG.visual(Makie.VLines; linestyle = :dash)
        ),
    )
end

# Placeholder figure (one titled empty axis) for "nothing to plot" results.
function _re_placeholder(msg::AbstractString, style::PlotStyle, kwargs_subplot, kwargs_layout)
    @warn msg
    panel = AoGPanel(nothing; axis = _aog_axis(style, kwargs_subplot; title = msg))
    return combine_plots([panel]; ncols = 1, style = style, _nt(kwargs_layout)...)
end

"""
    _re_draw(acc; style, ncols, series, axis, shared_x, shared_y, xlim, ylim, ...) -> Figure

Draw the panels of `acc` as AoG facets. `series` are `label => color` pairs (only labels
present in the data are used). Per-panel y limits stored in `acc.ylim` apply when no
common `ylim` is given; per-panel x labels in `acc.xlabel` apply when they differ between
panels (the x axis is then not linked) and the user did not set `xlabel`.
"""
function _re_draw(
        acc::_REFacets; style::PlotStyle, ncols::Integer, series, axis = NamedTuple(),
        shared_x::Bool = false, shared_y::Bool = false, xlim = nothing, ylim = nothing,
        kwargs_subplot = NamedTuple(), kwargs_layout = NamedTuple(), legend::Bool = true,
        qalpha::Real = 0.8, bar_kw = NamedTuple(), empty_msg::AbstractString
    )
    isempty(acc.panels) && return _re_placeholder(empty_msg, style, kwargs_subplot, kwargs_layout)
    present = Set(acc.df.series)
    series_use = [p for p in series if first(p) in present]
    user_ax = _nt(kwargs_subplot)
    axis = _nt(axis)
    xlabels = unique(values(acc.xlabel))
    per_panel_xlabel = length(xlabels) > 1
    if length(xlabels) == 1 && !haskey(axis, :xlabel)
        axis = merge(axis, (; xlabel = only(xlabels)))
    elseif per_panel_xlabel
        axis = merge(axis, (; xlabel = ""))
    end
    post = Dict{String, Vector{Any}}()
    for panel in acc.panels
        fs = Any[]
        if ylim === nothing && !shared_y && haskey(acc.ylim, panel)
            lo, hi = acc.ylim[panel]
            push!(fs, ax -> ylims!(ax, lo, hi))
        end
        if per_panel_xlabel && haskey(acc.xlabel, panel) && !haskey(user_ax, :xlabel)
            lbl = acc.xlabel[panel]
            push!(fs, ax -> (ax.xlabel = lbl; ax.xlabelvisible = true))
        end
        post[panel] = fs
    end
    fig = _draw_facets(
        _re_layers(acc, style; qalpha = qalpha, bar_kw = bar_kw);
        panels = acc.panels, ncols = ncols, style = style,
        shared_x = shared_x && !per_panel_xlabel, shared_y = shared_y,
        kwargs_subplot = kwargs_subplot, kwargs_layout = kwargs_layout,
        axis = axis, series = series_use, xlim = xlim, ylim = ylim,
        post = post, legend = legend,
    )
    return fig
end

_re_merge(lims, lo, hi) = lims === nothing ? (Float64(lo), Float64(hi)) :
    (min(lims[1], Float64(lo)), max(lims[2], Float64(hi)))

# Validated posterior settings shared by the pdf / distributions / PIT functions.
function _re_mcmc_setup(res, mcmc_warmup, mcmc_quantiles)
    res = _with_posterior_warmup(res, mcmc_warmup)
    qs = sort(Float64.(collect(mcmc_quantiles)))
    (length(qs) >= 2 && all(0 .<= qs .<= 100)) ||
        error("mcmc_quantiles must be in [0,100] with length >= 2.")
    return res, qs
end

# --- plot_random_effects_pdf -------------------------------------------------------------

"""
    plot_random_effects_pdf(res::FitResult; dm, re_names, levels, individuals_idx,
                            shared_x_axis, shared_y_axis, ncols, style, mcmc_draws,
                            mcmc_warmup, mcmc_quantiles, mcmc_quantiles_alpha,
                            flow_samples, flow_plot, flow_bins, flow_bandwidth, x_quantile,
                            xlims, rng, save_path, kwargs_subplot, kwargs_layout)
                            -> Makie.Figure

Plot the fitted population distribution (PDF, PMF for discrete random effects) of each
random effect, one panel per random effect (per component for multivariate random
effects). Random effects whose distribution depends on covariates are skipped. For
posterior-draw fits (MCMC, VI) the panel shows the mean density across `mcmc_draws`
posterior draws with dashed `mcmc_quantiles` lines.

# Keyword Arguments
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
- `re_names`: random-effect name(s) to include, or `nothing` for all.
- `levels`, `individuals_idx`: grouping levels / individuals. The population distribution
  does not depend on them; they are validated (unknown levels raise an error).
- `shared_x_axis::Bool = true`, `shared_y_axis::Bool = true`: share and link the axes
  across panels (inner tick labels hidden, padded common limits).
- `ncols::Int = 3`: number of subplot columns.
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `mcmc_draws`, `mcmc_warmup`, `mcmc_quantiles`, `mcmc_quantiles_alpha`: MCMC settings.
- `flow_samples::Int = 500`: number of samples for normalizing-flow distributions.
- `flow_plot::Symbol = :kde`: `:kde` or `:hist` for flow-based distributions.
- `flow_bins::Int = 20`, `flow_bandwidth`: histogram bins / KDE bandwidth for flows.
- `x_quantile::Float64 = 0.99`: coverage quantile used to set the x-axis range from the
  fitted distribution (e.g. `0.99` spans the 0.5th–99.5th percentiles).
- `xlims::Union{Nothing, Tuple{<:Real, <:Real}} = nothing`: explicit x-axis limits that
  override the quantile-based range; the PDF grid is also computed over this range.
- `rng::AbstractRNG = Xoshiro(42)`: random-number generator. A fresh generator with
  this fixed seed is created per call, so repeated calls draw the same values.
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
- `kwargs_subplot`: Makie `Axis` attributes for every panel (AlgebraOfGraphics `axis`;
  they win over the defaults).
- `kwargs_layout`: Makie `Figure` attributes (AlgebraOfGraphics `figure`; they win over
  the defaults).
"""
function plot_random_effects_pdf(
        res::FitResult;
        dm::Union{Nothing, DataModel} = nothing,
        re_names = nothing,
        levels = nothing,
        individuals_idx = nothing,
        shared_x_axis::Bool = true,
        shared_y_axis::Bool = true,
        ncols::Int = DEFAULT_PLOT_COLS,
        style::PlotStyle = PlotStyle(),
        mcmc_draws::Int = 1000,
        mcmc_warmup::Union{Nothing, Int} = nothing,
        mcmc_quantiles::Vector{<:Real} = [5, 95],
        mcmc_quantiles_alpha::Float64 = 0.8,
        flow_samples::Int = 500,
        flow_plot::Symbol = :kde,
        flow_bins::Int = 20,
        flow_bandwidth::Union{Nothing, Float64} = nothing,
        x_quantile::Float64 = 0.99,
        xlims::Union{Nothing, Tuple{<:Real, <:Real}} = nothing,
        rng::AbstractRNG = _default_rng(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing,
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple()
    )
    dm = _get_dm(res, dm)
    save_path = _resolve_plot_path(save_path, plot_path)
    _check_positive_int(ncols, "ncols")
    _check_unit_interval(mcmc_quantiles_alpha, "mcmc_quantiles_alpha")
    _check_positive_int(flow_bins, "flow_bins")
    flow_samples >= 2 ||
        error("flow_samples must be an integer >= 2; got $(flow_samples).")
    flow_bandwidth === nothing || _check_positive(flow_bandwidth, "flow_bandwidth")
    _require_re_supported(res)
    re_list = unique(_resolve_re_names(dm, re_names))
    # The population distribution does not depend on the levels, but invalid levels or
    # individuals are still an error.
    foreach(re -> _resolve_levels(dm, re, levels, individuals_idx), re_list)
    re_list = _filter_re_without_covariates(res, re_list)
    if isempty(re_list)
        fig = _re_placeholder(
            "No random-effect distributions without covariates to plot.",
            style, kwargs_subplot, kwargs_layout
        )
        return _save_plot!(fig, save_path)
    end
    is_mcmc = _is_posterior_draw_fit(res)
    qs = Float64[5, 95]
    if is_mcmc
        res, qs = _re_mcmc_setup(res, mcmc_warmup, mcmc_quantiles)
    end
    qlabel = _re_qlabel(qs)

    acc = _REFacets()
    xlims_acc = nothing
    ylims = nothing

    θ_base = is_mcmc ? _posterior_fixed_means(res, dm; rng = rng)[1] :
        get_params(res; scale = :untransformed)
    dists_builder = create_random_effect_distribution(get_random(get_model(dm)))
    model_funs = get_model_funs(get_model(dm))
    helpers = get_helper_funs(get_model(dm))

    θ_draws_cache = nothing
    for re in re_list
        const_cov = get_const_cov(get_individuals(dm)[1])
        dist0 = getproperty(dists_builder(θ_base, const_cov, model_funs, helpers), re)
        dim = dist0 isa Distributions.UnivariateDistribution ? 1 :
            length(vec(rand(rng, dist0)))
        value_cols = flatten_re_names(re, zeros(dim))
        show_comp = length(value_cols) > 1

        for (ci, comp_name) in enumerate(value_cols)
            panel = show_comp ? string(re, " | ", comp_name) : string(re)

            if is_mcmc
                if θ_draws_cache === nothing
                    constants_re = _fit_constants_re(res)
                    θ_draws_cache, _, _ = _posterior_drawn_params(
                        res, dm, constants_re, NamedTuple(), mcmc_draws, rng
                    )
                end
                samples_by_draw = Vector{Vector{Float64}}()
                grid_list = Vector{Tuple{Vector{Float64}, Vector{Float64}}}()
                min_v = Inf
                max_v = -Inf
                for θd in θ_draws_cache
                    dist_d = getproperty(
                        dists_builder(θd, const_cov, model_funs, helpers), re
                    )
                    dist_use = dist_d isa Distributions.MultivariateDistribution ?
                        _marginal_normal(dist_d, ci) : dist_d
                    if dist_use === nothing
                        if dist_d isa NormalizingPlanarFlow ||
                                dist_d isa Distributions.MultivariateDistribution
                            samp = if dist_d isa Distributions.UnivariateDistribution
                                vec(rand(rng, dist_d, flow_samples))
                            else
                                vec(rand(rng, dist_d, flow_samples)[ci, :])
                            end
                            length(samp) < 2 && continue
                            min_v = min(min_v, minimum(samp))
                            max_v = max(max_v, maximum(samp))
                            push!(samples_by_draw, samp)
                        end
                        continue
                    end
                    grid = _density_grid_continuous([dist_use], x_quantile, 200)
                    grid === nothing && continue
                    min_v = min(min_v, minimum(grid.y))
                    max_v = max(max_v, maximum(grid.y))
                    push!(grid_list, (grid.y, vec(grid.z[:, 1])))
                end
                if isempty(samples_by_draw) && isempty(grid_list)
                    @warn "Skipping RE pdf plot (insufficient draws)." re = re
                    continue
                end
                xlims_acc = _re_merge(xlims_acc, min_v, max_v)
                if flow_plot == :hist && !isempty(samples_by_draw)
                    edges = range(min_v, max_v; length = flow_bins + 1)
                    centers = (edges[1:(end - 1)] .+ edges[2:end]) ./ 2
                    curves = _re_fixed_hist(samples_by_draw, min_v, max_v, flow_bins)
                    isempty(curves) && continue
                    mean_c, qlo, qhi = _re_curve_summary(reduce(hcat, curves), qs)
                    _re_push!(acc, panel, centers, mean_c, "PDF", :bar)
                    _re_qlines!(acc, panel, centers, qlo, qhi, qlabel)
                else
                    xgrid = xlims !== nothing ? range(xlims[1], xlims[2]; length = 200) :
                        range(min_v, max_v; length = 200)
                    curves = Vector{Vector{Float64}}()
                    for samp in samples_by_draw
                        xk, yk = _kde_xy(samp; bandwidth = flow_bandwidth)
                        push!(curves, _interp_linear(xk, yk, xgrid))
                    end
                    for (xk, yk) in grid_list
                        push!(curves, _interp_linear(xk, yk, xgrid))
                    end
                    isempty(curves) && continue
                    mean_c, qlo, qhi = _re_curve_summary(reduce(hcat, curves), qs)
                    _re_push!(acc, panel, xgrid, mean_c, "PDF", :line)
                    _re_qlines!(acc, panel, xgrid, qlo, qhi, qlabel)
                end
                y_max = maximum(vcat(mean_c, qhi))
                acc.ylim[panel] = (0.0, y_max * 1.05)
                ylims = _re_merge(ylims, 0.0, y_max)
            else
                dist = dist0
                dist_use = dist isa Distributions.MultivariateDistribution ?
                    _marginal_normal(dist, ci) : dist
                if dist_use === nothing
                    if dist isa NormalizingPlanarFlow ||
                            dist isa Distributions.MultivariateDistribution
                        samp = if dist isa Distributions.UnivariateDistribution
                            vec(rand(rng, dist, flow_samples))
                        else
                            vec(rand(rng, dist, flow_samples)[ci, :])
                        end
                        length(samp) < 2 && continue
                        if flow_plot == :hist
                            h = _histogram_xy(
                                samp; bins = flow_bins, normalization = :probability
                            )
                            _re_push!(acc, panel, h.centers, h.heights, "PDF", :bar)
                            y_max = maximum(h.heights)
                            xlims_acc = _re_merge(xlims_acc, minimum(samp), maximum(samp))
                        else
                            xk, yk = _kde_xy(samp; bandwidth = flow_bandwidth)
                            _re_push!(acc, panel, xk, yk, "PDF", :line)
                            y_max = maximum(yk)
                            xlims_acc = _re_merge(xlims_acc, minimum(xk), maximum(xk))
                        end
                        acc.ylim[panel] = (0.0, y_max * 1.05)
                        ylims = _re_merge(ylims, 0.0, y_max)
                    else
                        @warn "Skipping RE pdf plot (missing mean/cov)." re = re
                        continue
                    end
                elseif dist_use isa DiscreteDistribution
                    grid = _density_grid_discrete(dist_use, 0.995)
                    grid === nothing && continue
                    _re_push!(acc, panel, grid.vals, grid.probs, "PMF", :pmf)
                    xlims_acc = _re_merge(xlims_acc, minimum(grid.vals), maximum(grid.vals))
                    ylims = _re_merge(ylims, minimum(grid.probs), maximum(grid.probs))
                else
                    grid = _density_grid_continuous(
                        [dist_use], x_quantile, 200; bounds = xlims
                    )
                    grid === nothing && continue
                    pdf_vals = vec(grid.z[:, 1])
                    _re_push!(acc, panel, grid.y, pdf_vals, "PDF", :line)
                    xlims_acc = _re_merge(xlims_acc, minimum(grid.y), maximum(grid.y))
                    ylims = _re_merge(ylims, minimum(pdf_vals), maximum(pdf_vals))
                end
            end
        end
    end

    xlim_use = if xlims !== nothing
        xlims
    elseif shared_x_axis && xlims_acc !== nothing
        _pad_limits(xlims_acc[1], xlims_acc[2])
    else
        nothing
    end
    ylim_use = shared_y_axis && ylims !== nothing ? _pad_limits(ylims[1], ylims[2]) : nothing
    fig = _re_draw(
        acc; style = style, ncols = ncols,
        series = ["PDF" => style.color_secondary, "PMF" => style.color_secondary, qlabel => style.color_secondary],
        axis = (; xlabel = "Random Effect", ylabel = "Probability Density"),
        shared_x = shared_x_axis, shared_y = shared_y_axis, xlim = xlim_use, ylim = ylim_use,
        kwargs_subplot = kwargs_subplot, kwargs_layout = kwargs_layout,
        qalpha = mcmc_quantiles_alpha, empty_msg = "No random-effect pdf plots to display."
    )
    return _save_plot!(fig, save_path)
end

# --- EBE scatters ------------------------------------------------------------------------

_re_level_xlabel(x_covariate, lvls_use) = x_covariate === nothing ?
    (any(lvl -> lvl isa Number, lvls_use) ? "Level" : "Index") : _axis_label(x_covariate)

function _re_check_x_covariate(dm::DataModel, x_covariate)
    x_covariate === nothing && return nothing
    cov = get_covariates(get_model(dm))
    x_covariate in cov.constants || error("x_covariate must be a constant covariate.")
    return nothing
end

# x value of level `lvl` (k-th selected level) for the EBE scatters.
function _re_level_x(lvl, k::Integer, const_cov, x_covariate)
    x_covariate === nothing && return lvl isa Number ? Float64(lvl) : Float64(k)
    xv_raw = getfield(const_cov, x_covariate)
    return ismissing(xv_raw) ? nothing : Float64(xv_raw)
end

function _re_scatter_figure(
        acc::_REFacets, xlims_val; style, ncols, ylabel, kwargs_subplot, kwargs_layout, empty_msg
    )
    return _re_draw(
        acc; style = style, ncols = ncols, series = ["EBE" => style.color_secondary],
        axis = (; ylabel = ylabel), shared_x = true, shared_y = false,
        xlim = xlims_val === nothing ? nothing : _pad_limits(xlims_val[1], xlims_val[2]),
        kwargs_subplot = kwargs_subplot, kwargs_layout = kwargs_layout,
        legend = false, empty_msg = empty_msg
    )
end

"""
    plot_random_effects_scatter(res::FitResult; dm, re_names, levels, individuals_idx,
                                x_covariate, mcmc_draws, rng, ncols, style, save_path,
                                kwargs_subplot, kwargs_layout) -> Makie.Figure

Scatter plot of empirical-Bayes estimates for each random effect against a constant
covariate or group level index, useful for detecting covariate relationships. One panel
per random effect (component); the x axis is shared and linked across panels.

# Keyword Arguments
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
- `re_names`: random-effect name(s) to include, or `nothing` for all.
- `levels`, `individuals_idx`: grouping level or individual filters.
- `x_covariate::Union{Nothing, Symbol} = nothing`: constant covariate for the x-axis;
  defaults to the group level index.
- `mcmc_draws::Int = 1000`: MCMC draws for posterior mean EBE.
- `rng::AbstractRNG = Xoshiro(42)`: random-number generator. A fresh generator with
  this fixed seed is created per call, so repeated calls draw the same values.
- `ncols::Int = 3`: number of subplot columns.
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
- `kwargs_subplot`, `kwargs_layout`: Makie `Axis` / `Figure` attributes (AlgebraOfGraphics
  `axis` / `figure`; they win over the defaults).
"""
function plot_random_effects_scatter(
        res::FitResult;
        dm::Union{Nothing, DataModel} = nothing,
        re_names = nothing,
        levels = nothing,
        individuals_idx = nothing,
        x_covariate::Union{Nothing, Symbol} = nothing,
        mcmc_draws::Int = 1000,
        rng::AbstractRNG = _default_rng(),
        ncols::Int = DEFAULT_PLOT_COLS,
        style::PlotStyle = PlotStyle(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing,
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple()
    )
    dm = _get_dm(res, dm)
    save_path = _resolve_plot_path(save_path, plot_path)
    _check_positive_int(ncols, "ncols")
    _require_re_supported(res)
    re_list = _filter_re_without_covariates(res, unique(_resolve_re_names(dm, re_names)))
    if isempty(re_list)
        fig = _re_placeholder(
            "No random-effect distributions without covariates to plot.",
            style, kwargs_subplot, kwargs_layout
        )
        return _save_plot!(fig, save_path)
    end
    _re_check_x_covariate(dm, x_covariate)
    is_post = _is_posterior_draw_fit(res)

    acc = _REFacets()
    xlims_val = nothing
    for re in re_list
        level_to_ind = _level_to_individual(dm, re)
        lvls = _resolve_levels(dm, re, levels, individuals_idx)
        ebe_map, value_cols = is_post ? _ebe_by_level_mcmc(dm, res, re, mcmc_draws, rng) :
            _ebe_by_level(dm, res, re)
        show_comp = length(value_cols) > 1
        lvls_use = [lvl for lvl in lvls if haskey(ebe_map, lvl)]
        isempty(lvls_use) && (lvls_use = collect(keys(ebe_map)))
        for (ci, comp_name) in enumerate(value_cols)
            xs = Float64[]
            ys = Float64[]
            for (k, lvl) in enumerate(lvls_use)
                haskey(ebe_map, lvl) || continue
                const_cov = get_const_cov(get_individuals(dm)[level_to_ind[lvl]])
                xv = _re_level_x(lvl, k, const_cov, x_covariate)
                xv === nothing && continue
                yv = Float64(ebe_map[lvl][ci])
                (isfinite(xv) && isfinite(yv)) || continue
                push!(xs, xv)
                push!(ys, yv)
            end
            isempty(xs) && continue
            panel = show_comp ? string(re, " | ", comp_name) : string(re)
            _re_push!(acc, panel, xs, ys, "EBE", :scatter)
            acc.xlabel[panel] = _re_level_xlabel(x_covariate, lvls_use)
            xlims_val = _re_merge(xlims_val, minimum(xs), maximum(xs))
        end
    end

    fig = _re_scatter_figure(
        acc, xlims_val; style = style, ncols = ncols,
        ylabel = is_post ? "Posterior Mean EBE" : "Empirical Bayes Estimate (EBE)",
        kwargs_subplot = kwargs_subplot, kwargs_layout = kwargs_layout,
        empty_msg = "No random-effect scatters to plot."
    )
    return _save_plot!(fig, save_path)
end

# --- plot_random_effect_pairplot -------------------------------------------------------
# Cell labels are the raw random-effect names: `_axis_label` would uppercase a leading
# Greek letter (η -> H).

"""
    plot_random_effect_pairplot(res::FitResult; dm, re_names, levels, individuals_idx,
                                ncols, style, kde_bandwidth, mcmc_draws, rng, save_path,
                                kwargs_subplot, kwargs_layout) -> Makie.Figure

Pairplot (scatter matrix) of empirical-Bayes estimates across all pairs of random
effects, useful for visualizing correlations and joint structure.

Random effects are grouped by their grouping column (only random effects of the same
column share levels). Each group is an `n × n` block of rows: the diagonal shows the
probability histogram (20 uniform bins, optional KDE) of one random effect (component),
the off-diagonal cells the scatter of row variable against column variable. Blocks of
several groups are stacked below each other; axes are not linked.

# Keyword Arguments
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
- `re_names`: random-effect names to include, or `nothing` for all.
- `levels`, `individuals_idx`: grouping level or individual filters.
- `ncols::Int = 3`: validated (`>= 1`); the matrix layout is fixed by the number of
  variables.
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `kde_bandwidth::Union{Nothing, Float64} = nothing`: when given, a KDE with this
  bandwidth is overlaid on the diagonal histograms.
- `mcmc_draws::Int = 1000`: MCMC draws for posterior mean EBE.
- `rng::AbstractRNG = Xoshiro(42)`: random-number generator. A fresh generator with
  this fixed seed is created per call, so repeated calls draw the same values.
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
- `kwargs_subplot`, `kwargs_layout`: Makie `Axis` / `Figure` attributes for every cell /
  the figure (they win over the defaults).
"""
function plot_random_effect_pairplot(
        res::FitResult;
        dm::Union{Nothing, DataModel} = nothing,
        re_names = nothing,
        levels = nothing,
        individuals_idx = nothing,
        ncols::Int = DEFAULT_PLOT_COLS,
        style::PlotStyle = PlotStyle(),
        kde_bandwidth::Union{Nothing, Float64} = nothing,
        mcmc_draws::Int = 1000,
        rng::AbstractRNG = _default_rng(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing,
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple()
    )
    dm = _get_dm(res, dm)
    save_path = _resolve_plot_path(save_path, plot_path)
    _check_positive_int(ncols, "ncols")
    kde_bandwidth === nothing || _check_positive(kde_bandwidth, "kde_bandwidth")
    _require_re_supported(res)
    re_list = _filter_re_without_covariates(res, unique(_resolve_re_names(dm, re_names)))
    if isempty(re_list)
        fig = _re_placeholder(
            "No random-effect distributions without covariates to plot.",
            style, kwargs_subplot, kwargs_layout
        )
        return _save_plot!(fig, save_path)
    end

    # Group by RE column (same levels), in the order of `re_list`.
    re_groups = get_re_groups(get_random(get_model(dm)))
    group_cols = Symbol[]
    group_map = Dict{Symbol, Vector{Symbol}}()
    for r in re_list
        col = getfield(re_groups, r)
        haskey(group_map, col) || (push!(group_cols, col); group_map[col] = Symbol[])
        push!(group_map[col], r)
    end

    blocks = Tuple{Vector{String}, Matrix{Float64}}[]
    for col in group_cols
        res_group = group_map[col]
        lvls = _resolve_levels(dm, res_group[1], levels, individuals_idx)
        ebe_maps = Dict{Symbol, Any}()
        comp_names = Dict{Symbol, Vector{String}}()
        for r in res_group
            ebe_map, value_cols = _is_posterior_draw_fit(res) ?
                _ebe_by_level_mcmc(dm, res, r, mcmc_draws, rng) :
                _ebe_by_level(dm, res, r)
            ebe_maps[r] = ebe_map
            comp_names[r] = string.(value_cols)
        end
        lvls_use = [lvl for lvl in lvls if all(haskey(ebe_maps[r], lvl) for r in res_group)]
        isempty(lvls_use) && (lvls_use = collect(keys(ebe_maps[res_group[1]])))
        # `c` already carries the RE-name prefix (flatten_re_names).
        labels = String[]
        for r in res_group, c in comp_names[r]
            push!(labels, length(comp_names[r]) == 1 ? string(r) : c)
        end
        nvars = length(labels)
        (nvars == 0 || isempty(lvls_use)) && continue
        vals = Matrix{Float64}(undef, length(lvls_use), nvars)
        for (i, lvl) in enumerate(lvls_use)
            k = 1
            for r in res_group, ci in eachindex(comp_names[r])
                vals[i, k] = Float64(ebe_maps[r][lvl][ci])
                k += 1
            end
        end
        push!(blocks, (labels, vals))
    end
    if isempty(blocks)
        fig = _re_placeholder("No random-effect pairplots to plot.", style, kwargs_subplot, kwargs_layout)
        return _save_plot!(fig, save_path)
    end

    max_nvars = maximum(length(first(b)) for b in blocks)
    nrows = sum(length(first(b)) for b in blocks)
    fig = Figure(; _aog_figure(style, nrows * max_nvars, max_nvars, kwargs_layout)...)
    row0 = 0
    for (labels, vals) in blocks
        nvars = length(labels)
        for i in 1:nvars, j in 1:nvars
            pos = fig[row0 + i, j]
            if i == j
                v = filter(isfinite, vals[:, i])
                axis = _aog_axis(style, kwargs_subplot; title = labels[i], xlabel = "", ylabel = "")
                if isempty(v)
                    Axis(pos; axis...)
                    continue
                end
                h = _histogram_xy(v; bins = 20, normalization = :probability)
                layers = AoG.data((; x = h.centers, y = h.heights)) * AoG.mapping(:x, :y) *
                    AoG.visual(Makie.BarPlot; gap = 0, strokewidth = 0, width = h.width, color = style.color_secondary)
                y_max = maximum(h.heights)
                if kde_bandwidth !== nothing && length(v) >= 2
                    xk, yk = _kde_xy(v; bandwidth = kde_bandwidth)
                    layers += AoG.data((; x = collect(xk), y = collect(yk))) * AoG.mapping(:x, :y) *
                        _line_visual(style; color = style.color_secondary)
                    y_max = max(y_max, maximum(yk))
                end
                ag = AoG.draw!(pos, layers; axis = axis)
                ylims!(first(ag).axis, 0.0, y_max * 1.05)
            else
                axis = _aog_axis(
                    style, kwargs_subplot; title = "",
                    xlabel = labels[j], ylabel = labels[i]
                )
                layer = AoG.data((; x = vals[:, j], y = vals[:, i])) * AoG.mapping(:x, :y) *
                    _scatter_visual(style; color = style.color_secondary)
                AoG.draw!(pos, layer; axis = axis)
            end
        end
        row0 += nvars
    end
    return _save_plot!(fig, save_path)
end

# --- plot_random_effect_distributions ----------------------------------------------------

"""
    plot_random_effect_distributions(res::FitResult; dm, re_names, levels,
                                     individuals_idx, shared_x_axis, shared_y_axis,
                                     ncols, style, mcmc_draws, mcmc_warmup,
                                     mcmc_quantiles, mcmc_quantiles_alpha, flow_samples,
                                     flow_plot, flow_bins, flow_bandwidth, x_quantile,
                                     xlims, rng, save_path, kwargs_subplot, kwargs_layout)
                                     -> Makie.Figure

Plot, for every selected level of each random effect, the fitted distribution of that
level (PDF, or PMF for discrete random effects) with its EBE (posterior mean for
posterior-draw fits) as a dashed vertical line. One panel per random effect, level and
component.

# Keyword Arguments
All arguments are identical to [`plot_random_effects_pdf`](@ref); `levels` and
`individuals_idx` select the levels shown.
"""
function plot_random_effect_distributions(
        res::FitResult;
        dm::Union{Nothing, DataModel} = nothing,
        re_names = nothing,
        levels = nothing,
        individuals_idx = nothing,
        shared_x_axis::Bool = true,
        shared_y_axis::Bool = true,
        ncols::Int = DEFAULT_PLOT_COLS,
        style::PlotStyle = PlotStyle(),
        mcmc_draws::Int = 1000,
        mcmc_warmup::Union{Nothing, Int} = nothing,
        mcmc_quantiles::Vector{<:Real} = [5, 95],
        mcmc_quantiles_alpha::Float64 = 0.8,
        flow_samples::Int = 500,
        flow_plot::Symbol = :kde,
        flow_bins::Int = 20,
        flow_bandwidth::Union{Nothing, Float64} = nothing,
        x_quantile::Float64 = 0.99,
        xlims::Union{Nothing, Tuple{<:Real, <:Real}} = nothing,
        rng::AbstractRNG = _default_rng(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing,
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple()
    )
    dm = _get_dm(res, dm)
    save_path = _resolve_plot_path(save_path, plot_path)
    _check_positive_int(ncols, "ncols")
    _check_unit_interval(mcmc_quantiles_alpha, "mcmc_quantiles_alpha")
    _check_positive_int(flow_bins, "flow_bins")
    flow_samples >= 2 ||
        error("flow_samples must be an integer >= 2; got $(flow_samples).")
    flow_bandwidth === nothing || _check_positive(flow_bandwidth, "flow_bandwidth")
    _require_re_supported(res)
    re_list = unique(_resolve_re_names(dm, re_names))
    is_mcmc = _is_posterior_draw_fit(res)
    qs = Float64[5, 95]
    if is_mcmc
        res, qs = _re_mcmc_setup(res, mcmc_warmup, mcmc_quantiles)
    end
    qlabel = _re_qlabel(qs)
    ebe_label = is_mcmc ? "posterior mean" : "EBE"

    acc = _REFacets()
    xlims_acc = nothing
    ylims = nothing
    θ_draws_cache = nothing

    θ_base = is_mcmc ? _posterior_fixed_means(res, dm; rng = rng)[1] :
        get_params(res; scale = :untransformed)
    dists_builder = create_random_effect_distribution(get_random(get_model(dm)))
    model_funs = get_model_funs(get_model(dm))
    helpers = get_helper_funs(get_model(dm))
    pid = get_primary_id(dm)

    for re in re_list
        level_to_ind = _level_to_individual(dm, re)
        lvls = _resolve_levels(dm, re, levels, individuals_idx)
        ebe_map, value_cols = is_mcmc ? _ebe_by_level_mcmc(dm, res, re, mcmc_draws, rng) :
            _ebe_by_level(dm, res, re)
        show_comp = length(value_cols) > 1
        lvls_use = [lvl for lvl in lvls if haskey(ebe_map, lvl)]
        isempty(lvls_use) && (lvls_use = collect(keys(ebe_map)))

        for lvl in lvls_use
            haskey(ebe_map, lvl) || continue
            const_cov = get_const_cov(get_individuals(dm)[level_to_ind[lvl]])
            dist = getproperty(dists_builder(θ_base, const_cov, model_funs, helpers), re)
            vals = ebe_map[lvl]
            for (ci, comp_name) in enumerate(value_cols)
                ebe_val = vals[ci]
                panel = show_comp ? string(re, " | ", pid, "=", lvl, " | ", comp_name) :
                    string(re, " | ", pid, "=", lvl)
                dist_use = dist isa Distributions.MultivariateDistribution ?
                    _marginal_normal(dist, ci) : dist
                if dist_use === nothing
                    if !(dist isa NormalizingPlanarFlow)
                        @warn "Skipping RE distribution plot (missing mean/cov)." re = re level = lvl
                        continue
                    end
                    # Flow: approximate marginal via sampling + KDE/hist.
                    if is_mcmc
                        @info "Flow marginal plotted via sampling (MCMC averaged)." re = re level = lvl samples = flow_samples
                        if θ_draws_cache === nothing
                            constants_re = _fit_constants_re(res)
                            θ_draws_cache, _, _ = _posterior_drawn_params(
                                res, dm, constants_re, NamedTuple(), mcmc_draws, rng
                            )
                        end
                        samples_by_draw = Vector{Vector{Float64}}()
                        min_v = Inf
                        max_v = -Inf
                        for θd in θ_draws_cache
                            dist_d = getproperty(
                                dists_builder(θd, const_cov, model_funs, helpers), re
                            )
                            samp = if dist_d isa Distributions.UnivariateDistribution
                                vec(rand(rng, dist_d, flow_samples))
                            else
                                vec(rand(rng, dist_d, flow_samples)[ci, :])
                            end
                            length(samp) < 2 && continue
                            min_v = min(min_v, minimum(samp))
                            max_v = max(max_v, maximum(samp))
                            push!(samples_by_draw, samp)
                        end
                        if isempty(samples_by_draw)
                            @warn "Skipping RE flow plot (insufficient samples)." re = re level = lvl
                            continue
                        end
                        if flow_plot == :hist
                            edges = range(min_v, max_v; length = flow_bins + 1)
                            centers = (edges[1:(end - 1)] .+ edges[2:end]) ./ 2
                            counts_list = _re_fixed_hist(samples_by_draw, min_v, max_v, flow_bins)
                            if isempty(counts_list)
                                @warn "Skipping RE flow histogram (insufficient samples)." re = re level = lvl
                                continue
                            end
                            mean_c, qlo, qhi = _re_curve_summary(reduce(hcat, counts_list), qs)
                            _re_push!(acc, panel, centers, mean_c, "flow", :bar)
                            _re_qlines!(acc, panel, centers, qlo, qhi, qlabel)
                        else
                            xgrid = range(min_v, max_v; length = 200)
                            dens_list = Vector{Vector{Float64}}()
                            for samp in samples_by_draw
                                xk, yk = _kde_xy(samp; bandwidth = flow_bandwidth)
                                push!(dens_list, _interp_linear(xk, yk, xgrid))
                            end
                            mean_c, qlo, qhi = _re_curve_summary(reduce(hcat, dens_list), qs)
                            _re_push!(acc, panel, xgrid, mean_c, "flow KDE", :line)
                            _re_qlines!(acc, panel, xgrid, qlo, qhi, qlabel)
                        end
                        y_max = maximum(vcat(mean_c, qhi))
                        xlims_acc = _re_merge(xlims_acc, min_v, max_v)
                    else
                        @info "Flow marginal plotted via sampling." re = re level = lvl samples = flow_samples
                        samples = if dist isa Distributions.UnivariateDistribution
                            vec(rand(rng, dist, flow_samples))
                        else
                            vec(rand(rng, dist, flow_samples)[ci, :])
                        end
                        if length(samples) < 2
                            @warn "Skipping RE flow plot (insufficient samples)." re = re level = lvl
                            continue
                        end
                        if flow_plot == :hist
                            h = _histogram_xy(samples; bins = flow_bins, normalization = :probability)
                            _re_push!(acc, panel, h.centers, h.heights, "flow", :bar)
                            y_max = maximum(h.heights)
                            xlims_acc = _re_merge(xlims_acc, minimum(samples), maximum(samples))
                        else
                            xk, yk = _kde_xy(samples; bandwidth = flow_bandwidth)
                            _re_push!(acc, panel, xk, yk, "flow KDE", :line)
                            y_max = maximum(yk)
                            xlims_acc = _re_merge(xlims_acc, minimum(xk), maximum(xk))
                        end
                    end
                    _re_vline!(acc, panel, ebe_val, ebe_label)
                    acc.ylim[panel] = (0.0, y_max * 1.05)
                    ylims = _re_merge(ylims, 0.0, y_max)
                    continue
                end
                if dist_use isa DiscreteDistribution
                    grid = _density_grid_discrete(dist_use, 0.995)
                    grid === nothing && error("Unable to build PMF grid for $(re).")
                    _re_push!(acc, panel, grid.vals, grid.probs, "PMF", :pmf)
                    xlims_acc = _re_merge(xlims_acc, minimum(grid.vals), maximum(grid.vals))
                    ylims = _re_merge(ylims, minimum(grid.probs), maximum(grid.probs))
                else
                    grid = _density_grid_continuous([dist_use], x_quantile, 200; bounds = xlims)
                    grid === nothing && error("Unable to build PDF grid for $(re).")
                    pdf_vals = vec(grid.z[:, 1])
                    _re_push!(acc, panel, grid.y, pdf_vals, "PDF", :line)
                    xlims_acc = _re_merge(xlims_acc, minimum(grid.y), maximum(grid.y))
                    ylims = _re_merge(ylims, minimum(pdf_vals), maximum(pdf_vals))
                end
                _re_vline!(acc, panel, ebe_val, ebe_label)
            end
        end
    end

    xlim_use = if xlims !== nothing
        xlims
    elseif shared_x_axis && xlims_acc !== nothing
        _pad_limits(xlims_acc[1], xlims_acc[2])
    else
        nothing
    end
    ylim_use = shared_y_axis && ylims !== nothing ? _pad_limits(ylims[1], ylims[2]) : nothing
    c2 = style.color_secondary
    fig = _re_draw(
        acc; style = style, ncols = ncols,
        series = [
            "PDF" => c2, "PMF" => c2, "flow" => c2, "flow KDE" => c2, qlabel => c2,
            ebe_label => style.color_primary,
        ],
        axis = (; xlabel = "Random Effect", ylabel = "Probability Density"),
        shared_x = shared_x_axis, shared_y = shared_y_axis, xlim = xlim_use, ylim = ylim_use,
        kwargs_subplot = kwargs_subplot, kwargs_layout = kwargs_layout,
        qalpha = mcmc_quantiles_alpha, empty_msg = "No random-effect distributions to plot."
    )
    return _save_plot!(fig, save_path)
end

# --- plot_random_effect_pit --------------------------------------------------------------

"""
    plot_random_effect_pit(res::FitResult; dm, re_names, levels, individuals_idx,
                           show_hist, show_kde, show_qq, shared_x_axis, shared_y_axis,
                           ncols, style, kde_bandwidth, mcmc_draws, mcmc_warmup,
                           mcmc_quantiles, mcmc_quantiles_alpha, flow_samples, rng,
                           save_path, kwargs_subplot, kwargs_layout) -> Makie.Figure

Plot the probability integral transform (PIT) of empirical-Bayes estimates under their
fitted prior distributions, providing a calibration check for the random-effects model.
One panel per random effect (component).

Only one of `show_hist`, `show_kde` and `show_qq` is drawn; enabling several warns and
falls back to the histogram. For posterior-draw fits (MCMC, VI) the panel shows the mean
across posterior draws (histogram bars on 20 bins in [0, 1], KDE or QQ line) with dashed
`mcmc_quantiles` lines. Random effects with fewer than two PIT values are skipped with a
warning.

# Keyword Arguments
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
- `re_names`: random-effect names to include, or `nothing` for all.
- `levels`, `individuals_idx`: grouping level or individual filters.
- `show_hist::Bool = true`: show a PIT histogram.
- `show_kde::Bool = false`: draw a KDE curve instead of a histogram.
- `show_qq::Bool = false`: draw a Uniform QQ plot instead of a histogram.
- `shared_x_axis::Bool = true`: common, linked x axis `[0, 1]`.
- `shared_y_axis::Bool = true`: common, linked y axis (padded union of the panels).
- `ncols::Int = 3`: number of subplot columns.
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `kde_bandwidth::Union{Nothing, Float64} = nothing`: KDE bandwidth.
- `mcmc_draws`, `mcmc_warmup`, `mcmc_quantiles`, `mcmc_quantiles_alpha`: MCMC settings.
- `flow_samples::Int = 500`: samples for normalizing-flow distributions.
- `rng::AbstractRNG = Xoshiro(42)`: random-number generator. A fresh generator with
  this fixed seed is created per call, so repeated calls draw the same values.
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
- `kwargs_subplot`, `kwargs_layout`: Makie `Axis` / `Figure` attributes (AlgebraOfGraphics
  `axis` / `figure`; they win over the defaults).
"""
function plot_random_effect_pit(
        res::FitResult;
        dm::Union{Nothing, DataModel} = nothing,
        re_names = nothing,
        levels = nothing,
        individuals_idx = nothing,
        x_covariate::Union{Nothing, Symbol} = nothing,
        show_hist::Bool = true,
        show_kde::Bool = false,
        show_qq::Bool = false,
        shared_x_axis::Bool = true,
        shared_y_axis::Bool = true,
        ncols::Int = DEFAULT_PLOT_COLS,
        style::PlotStyle = PlotStyle(),
        kde_bandwidth::Union{Nothing, Float64} = nothing,
        mcmc_draws::Int = 1000,
        mcmc_warmup::Union{Nothing, Int} = nothing,
        mcmc_quantiles::Vector{<:Real} = [5, 95],
        mcmc_quantiles_alpha::Float64 = 0.8,
        flow_samples::Int = 500,
        rng::AbstractRNG = _default_rng(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing,
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple()
    )
    dm = _get_dm(res, dm)
    save_path = _resolve_plot_path(save_path, plot_path)
    _check_positive_int(ncols, "ncols")
    _check_unit_interval(mcmc_quantiles_alpha, "mcmc_quantiles_alpha")
    kde_bandwidth === nothing || _check_positive(kde_bandwidth, "kde_bandwidth")
    _require_re_supported(res)
    re_list = unique(_resolve_re_names(dm, re_names))
    is_mcmc = _is_posterior_draw_fit(res)
    x_covariate === nothing ||
        throw(ArgumentError("`x_covariate` is not supported by `plot_random_effect_pit`. Use `plot_random_effects_scatter` or `plot_random_effect_standardized_scatter` instead."))

    if (show_hist + show_kde + show_qq) > 1
        @warn "plot_random_effect_pit expects one plot type at a time; defaulting to histogram." show_hist = show_hist show_kde = show_kde show_qq = show_qq
        show_hist = true
        show_kde = false
        show_qq = false
    end
    kind_title, xlabel, ylabel = if show_kde
        " | PIT KDE", "PIT", "Density"
    elseif show_qq
        " | PIT QQ", "Theoretical Uniform Quantile", "Empirical PIT Quantile"
    else
        " | PIT hist", "PIT", "Probability"
    end

    qs = Float64[5, 95]
    if is_mcmc
        res, qs = _re_mcmc_setup(res, mcmc_warmup, mcmc_quantiles)
    end
    qlabel = _re_qlabel(qs)

    acc = _REFacets()
    ylims = nothing

    θ_base = is_mcmc ? _posterior_fixed_means(res, dm; rng = rng)[1] :
        get_params(res; scale = :untransformed)
    dists_builder = create_random_effect_distribution(get_random(get_model(dm)))
    model_funs = get_model_funs(get_model(dm))
    helpers = get_helper_funs(get_model(dm))

    # PIT of value `v` (component `ci`) under `dist`; flows via the empirical CDF.
    function pit_of(dist, v, ci, re, lvl)
        dist_use = dist isa Distributions.MultivariateDistribution ?
            _marginal_normal(dist, ci) : dist
        if dist_use === nothing
            dist isa NormalizingPlanarFlow || return nothing
            @info "Flow PIT via empirical CDF (sampling)." re = re level = lvl samples = flow_samples
            samp = dist isa Distributions.UnivariateDistribution ?
                vec(rand(rng, dist, flow_samples)) : vec(rand(rng, dist, flow_samples)[ci, :])
            isempty(samp) && return nothing
            return count(<=(v), samp) / length(samp)
        end
        return _pit_value(dist_use, v)
    end

    for re in re_list
        level_to_ind = _level_to_individual(dm, re)
        lvls = _resolve_levels(dm, re, levels, individuals_idx)
        ebe_map, value_cols = is_mcmc ? _ebe_by_level_mcmc(dm, res, re, mcmc_draws, rng) :
            _ebe_by_level(dm, res, re)
        show_comp = length(value_cols) > 1
        lvls_use = [lvl for lvl in lvls if haskey(ebe_map, lvl)]
        isempty(lvls_use) && (lvls_use = collect(keys(ebe_map)))

        for (ci, comp_name) in enumerate(value_cols)
            panel = (show_comp ? string(re, " | ", comp_name) : string(re)) * kind_title
            if is_mcmc
                constants_re = _fit_constants_re(res)
                θ_draws, η_draws, _ = _posterior_drawn_params(
                    res, dm, constants_re, NamedTuple(), mcmc_draws, rng
                )
                pits_by_draw = Vector{Vector{Float64}}()
                for d in eachindex(θ_draws)
                    pits_d = Float64[]
                    for lvl in lvls_use
                        ind_idx = level_to_ind[lvl]
                        const_cov = get_const_cov(get_individuals(dm)[ind_idx])
                        dist_d = getproperty(
                            dists_builder(θ_draws[d], const_cov, model_funs, helpers), re
                        )
                        v = getproperty(η_draws[d][ind_idx], re)
                        v_use = v isa Number ? Float64(v) : Float64(v[ci])
                        pit = pit_of(dist_d, v_use, ci, re, lvl)
                        pit === nothing || push!(pits_d, pit)
                    end
                    isempty(pits_d) || push!(pits_by_draw, pits_d)
                end
                pits_by_draw = filter(p -> length(p) >= 2, pits_by_draw)
                if isempty(pits_by_draw)
                    @warn "PIT skipped (missing cdf or fewer than 2 PIT values)." re = re component = comp_name
                    continue
                end
                if show_hist
                    nbins = 20
                    edges = range(0.0, 1.0; length = nbins + 1)
                    xs = (edges[1:(end - 1)] .+ edges[2:end]) ./ 2
                    mat = reduce(hcat, _re_fixed_hist(pits_by_draw, 0.0, 1.0, nbins))
                    mean_c, qlo, qhi = _re_curve_summary(mat, qs)
                    _re_push!(acc, panel, xs, mean_c, "PIT", :bar)
                elseif show_kde
                    xs = range(0.0, 1.0; length = 200)
                    dens = [
                        _interp_linear(_kde_xy(p; bandwidth = kde_bandwidth)..., xs)
                            for p in pits_by_draw
                    ]
                    mean_c, qlo, qhi = _re_curve_summary(reduce(hcat, dens), qs)
                    _re_push!(acc, panel, xs, mean_c, "KDE", :line)
                else
                    xs = range(0.0, 1.0; length = 100)
                    qmat = reduce(hcat, [[quantile(p, u) for u in xs] for p in pits_by_draw])
                    mean_c, qlo, qhi = _re_curve_summary(qmat, qs)
                    _re_push!(acc, panel, xs, mean_c, "QQ", :line)
                    _re_push!(acc, panel, xs, xs, "Uniform", :ref)
                end
                _re_qlines!(acc, panel, xs, qlo, qhi, qlabel)
                y_max = maximum(vcat(mean_c, qhi))
            else
                pits = Float64[]
                for lvl in lvls
                    haskey(ebe_map, lvl) || continue
                    const_cov = get_const_cov(get_individuals(dm)[level_to_ind[lvl]])
                    dist = getproperty(
                        dists_builder(θ_base, const_cov, model_funs, helpers), re
                    )
                    pit = pit_of(dist, Float64(ebe_map[lvl][ci]), ci, re, lvl)
                    pit === nothing || push!(pits, pit)
                end
                if length(pits) < 2
                    @warn "PIT skipped (missing cdf or fewer than 2 PIT values)." re = re component = comp_name
                    continue
                end
                if show_hist
                    h = _histogram_xy(pits; bins = 20, normalization = :probability)
                    _re_push!(acc, panel, h.centers, h.heights, "PIT", :bar)
                    y_max = maximum(h.heights)
                elseif show_kde
                    xk, yk = _kde_xy(pits; bandwidth = kde_bandwidth)
                    _re_push!(acc, panel, xk, yk, "KDE", :line)
                    y_max = maximum(yk)
                else
                    q = sort(pits)
                    u = range(0, 1; length = length(q))
                    _re_push!(acc, panel, u, q, "QQ", :scatter)
                    _re_push!(acc, panel, u, u, "Uniform", :ref)
                    y_max = 1.0
                end
            end
            show_qq || (acc.ylim[panel] = (0.0, y_max * 1.05))
            ylims = show_qq ? _re_merge(ylims, 0.0, 1.0) : _re_merge(ylims, 0.0, y_max * 1.05)
        end
    end

    c2 = style.color_secondary
    fig = _re_draw(
        acc; style = style, ncols = ncols,
        series = [
            "PIT" => c2, "KDE" => c2, "QQ" => c2, qlabel => c2, "Uniform" => style.color_dark,
        ],
        axis = (; xlabel = xlabel, ylabel = ylabel),
        shared_x = shared_x_axis, shared_y = shared_y_axis,
        # slightly padded [0, 1] so the end tick labels of neighbouring facets do not touch
        xlim = shared_x_axis ? _pad_limits(0.0, 1.0; frac = 0.03) : nothing,
        ylim = shared_y_axis && ylims !== nothing && !show_qq ? ylims : nothing,
        kwargs_subplot = kwargs_subplot, kwargs_layout = kwargs_layout,
        qalpha = mcmc_quantiles_alpha,
        bar_kw = is_mcmc ? NamedTuple() : (; strokecolor = c2, strokewidth = 1),
        empty_msg = "No PIT plots to display."
    )
    return _save_plot!(fig, save_path)
end

# --- standardized EBEs -------------------------------------------------------------------

"""
    plot_random_effect_standardized(res::FitResult; dm, re_names, levels,
                                    individuals_idx, show_hist, show_kde, kde_bandwidth,
                                    mcmc_draws, flow_samples, rng, ncols, style, save_path,
                                    kwargs_subplot, kwargs_layout) -> Makie.Figure

Plot standardized (z-score) empirical-Bayes estimates for each random effect as a
probability histogram (20 uniform bins) and/or KDE, one panel per random effect
(component). Values far from zero indicate outliers or misspecification.

# Keyword Arguments
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
- `re_names`: random-effect names to include, or `nothing` for all.
- `levels`, `individuals_idx`: grouping level or individual filters.
- `show_hist::Bool = true`: show a histogram.
- `show_kde::Bool = false`: overlay a KDE curve (needs at least two values).
- `kde_bandwidth::Union{Nothing, Float64} = nothing`: KDE bandwidth.
- `mcmc_draws::Int = 1000`: MCMC draws for posterior mean EBE.
- `flow_samples::Int = 500`: samples for normalizing-flow distributions.
- `rng::AbstractRNG = Xoshiro(42)`: random-number generator. A fresh generator with
  this fixed seed is created per call, so repeated calls draw the same values.
- `ncols::Int = 3`: number of subplot columns.
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
- `kwargs_subplot`, `kwargs_layout`: Makie `Axis` / `Figure` attributes (AlgebraOfGraphics
  `axis` / `figure`; they win over the defaults).
"""
function plot_random_effect_standardized(
        res::FitResult;
        dm::Union{Nothing, DataModel} = nothing,
        re_names = nothing,
        levels = nothing,
        individuals_idx = nothing,
        show_hist::Bool = true,
        show_kde::Bool = false,
        kde_bandwidth::Union{Nothing, Float64} = nothing,
        mcmc_draws::Int = 1000,
        flow_samples::Int = 500,
        rng::AbstractRNG = _default_rng(),
        ncols::Int = DEFAULT_PLOT_COLS,
        style::PlotStyle = PlotStyle(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing,
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple()
    )
    dm = _get_dm(res, dm)
    save_path = _resolve_plot_path(save_path, plot_path)
    _check_positive_int(ncols, "ncols")
    kde_bandwidth === nothing || _check_positive(kde_bandwidth, "kde_bandwidth")
    _require_re_supported(res)
    re_list = unique(_resolve_re_names(dm, re_names))
    is_post = _is_posterior_draw_fit(res)
    θ_base = is_post ? _posterior_fixed_means(res, dm; rng = rng)[1] :
        get_params(res; scale = :untransformed)
    dists_builder = create_random_effect_distribution(get_random(get_model(dm)))
    model_funs = get_model_funs(get_model(dm))
    helpers = get_helper_funs(get_model(dm))

    acc = _REFacets()
    for re in re_list
        level_to_ind = _level_to_individual(dm, re)
        lvls = _resolve_levels(dm, re, levels, individuals_idx)
        ebe_map, value_cols = is_post ? _ebe_by_level_mcmc(dm, res, re, mcmc_draws, rng) :
            _ebe_by_level(dm, res, re)
        show_comp = length(value_cols) > 1
        lvls_use = [lvl for lvl in lvls if haskey(ebe_map, lvl)]
        isempty(lvls_use) && (lvls_use = collect(keys(ebe_map)))
        for (ci, comp_name) in enumerate(value_cols)
            zvals = Float64[]
            for lvl in lvls_use
                haskey(ebe_map, lvl) || continue
                const_cov = get_const_cov(get_individuals(dm)[level_to_ind[lvl]])
                dist = getproperty(dists_builder(θ_base, const_cov, model_funs, helpers), re)
                z = _standardize_re(
                    dist, Float64.(ebe_map[lvl]); flow_samples = flow_samples, rng = rng
                )
                z === nothing && continue
                isfinite(z[ci]) && push!(zvals, Float64(z[ci]))
            end
            if isempty(zvals)
                @warn "No standardized values to plot." re = re component = comp_name
                continue
            end
            panel = show_comp ? string(re, " | ", comp_name) : string(re)
            y_max = 0.0
            if show_hist
                h = _histogram_xy(zvals; bins = 20, normalization = :probability)
                _re_push!(acc, panel, h.centers, h.heights, "z", :bar)
                y_max = max(y_max, maximum(h.heights))
            end
            if show_kde
                if length(zvals) < 2
                    @warn "Skipping standardized KDE (fewer than 2 values)." re = re component = comp_name
                else
                    xk, yk = _kde_xy(zvals; bandwidth = kde_bandwidth)
                    _re_push!(acc, panel, xk, yk, "KDE", :line)
                    y_max = max(y_max, maximum(yk))
                end
            end
            y_max > 0 && (acc.ylim[panel] = (0.0, y_max * 1.05))
        end
    end
    y_label = show_hist && show_kde ? "Probability / Density" :
        (show_kde ? "Density" : "Probability")
    fig = _re_draw(
        acc; style = style, ncols = ncols,
        series = ["z" => style.color_secondary, "KDE" => style.color_secondary],
        axis = (; xlabel = "Standardized EBE", ylabel = y_label),
        kwargs_subplot = kwargs_subplot, kwargs_layout = kwargs_layout,
        empty_msg = "No standardized random effects to plot."
    )
    return _save_plot!(fig, save_path)
end

"""
    plot_random_effect_standardized_scatter(res::FitResult; dm, re_names, levels,
                                            individuals_idx, x_covariate, mcmc_draws,
                                            flow_samples, rng, ncols, style, save_path,
                                            kwargs_subplot, kwargs_layout) -> Makie.Figure

Scatter plot of standardized (z-score) empirical-Bayes estimates against a covariate
or group level index, one panel per random effect (component); the x axis is shared and
linked across panels. Useful for detecting systematic patterns in the residual structure
of the random-effects model.

# Keyword Arguments
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
- `re_names`: random-effect names to include, or `nothing` for all.
- `levels`, `individuals_idx`: grouping level or individual filters.
- `x_covariate::Union{Nothing, Symbol} = nothing`: constant covariate for the x-axis.
- `mcmc_draws::Int = 1000`: MCMC draws for posterior mean EBE.
- `flow_samples::Int = 500`: samples for normalizing-flow distributions.
- `rng::AbstractRNG = Xoshiro(42)`: random-number generator. A fresh generator with
  this fixed seed is created per call, so repeated calls draw the same values.
- `ncols::Int = 3`: number of subplot columns.
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
- `kwargs_subplot`, `kwargs_layout`: Makie `Axis` / `Figure` attributes (AlgebraOfGraphics
  `axis` / `figure`; they win over the defaults).
"""
function plot_random_effect_standardized_scatter(
        res::FitResult;
        dm::Union{Nothing, DataModel} = nothing,
        re_names = nothing,
        levels = nothing,
        individuals_idx = nothing,
        x_covariate::Union{Nothing, Symbol} = nothing,
        mcmc_draws::Int = 1000,
        flow_samples::Int = 500,
        rng::AbstractRNG = _default_rng(),
        ncols::Int = DEFAULT_PLOT_COLS,
        style::PlotStyle = PlotStyle(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing,
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple()
    )
    dm = _get_dm(res, dm)
    save_path = _resolve_plot_path(save_path, plot_path)
    _check_positive_int(ncols, "ncols")
    _require_re_supported(res)
    re_list = unique(_resolve_re_names(dm, re_names))
    _re_check_x_covariate(dm, x_covariate)
    is_post = _is_posterior_draw_fit(res)

    θ_base = is_post ? _posterior_fixed_means(res, dm; rng = rng)[1] :
        get_params(res; scale = :untransformed)
    dists_builder = create_random_effect_distribution(get_random(get_model(dm)))
    model_funs = get_model_funs(get_model(dm))
    helpers = get_helper_funs(get_model(dm))

    acc = _REFacets()
    xlims_val = nothing
    for re in re_list
        level_to_ind = _level_to_individual(dm, re)
        lvls = _resolve_levels(dm, re, levels, individuals_idx)
        ebe_map, value_cols = is_post ? _ebe_by_level_mcmc(dm, res, re, mcmc_draws, rng) :
            _ebe_by_level(dm, res, re)
        show_comp = length(value_cols) > 1
        lvls_use = [lvl for lvl in lvls if haskey(ebe_map, lvl)]
        isempty(lvls_use) && (lvls_use = collect(keys(ebe_map)))

        for (ci, comp_name) in enumerate(value_cols)
            xs = Float64[]
            ys = Float64[]
            for (k, lvl) in enumerate(lvls_use)
                haskey(ebe_map, lvl) || continue
                const_cov = get_const_cov(get_individuals(dm)[level_to_ind[lvl]])
                dist = getproperty(dists_builder(θ_base, const_cov, model_funs, helpers), re)
                z = _standardize_re(
                    dist, Float64.(ebe_map[lvl]); flow_samples = flow_samples, rng = rng
                )
                z === nothing && continue
                xv = _re_level_x(lvl, k, const_cov, x_covariate)
                xv === nothing && continue
                yv = Float64(z[ci])
                (isfinite(xv) && isfinite(yv)) || continue
                push!(xs, xv)
                push!(ys, yv)
            end
            isempty(xs) && continue
            panel = show_comp ? string(re, " | ", comp_name) : string(re)
            _re_push!(acc, panel, xs, ys, "EBE", :scatter)
            acc.xlabel[panel] = _re_level_xlabel(x_covariate, lvls_use)
            xlims_val = _re_merge(xlims_val, minimum(xs), maximum(xs))
        end
    end

    fig = _re_scatter_figure(
        acc, xlims_val; style = style, ncols = ncols, ylabel = "Standardized EBE (z-score)",
        kwargs_subplot = kwargs_subplot, kwargs_layout = kwargs_layout,
        empty_msg = "No standardized random effects to plot."
    )
    return _save_plot!(fig, save_path)
end
