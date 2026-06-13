function _plot_wald_closed_form!(p, kind::Symbol, μ::Float64, v::Float64,
        style::PlotStyle; show_legend::Bool, npts::Int = 300)
    xy = _wald_density_xy(kind, μ, v; npts = npts)
    if xy === nothing
        vline!(p, [μ];
            color = style.color_primary,
            linewidth = style.line_width_primary,
            label = show_legend ? "Approx. Density" : "")
        return nothing
    end
    x, y = xy
    plot!(p, x, y;
        color = style.color_primary,
        linewidth = style.line_width_primary,
        label = show_legend ? "Approx. Density" : "")
    return (x, y)
end

function _plot_density_interval_fill!(p,
        x::AbstractVector{<:Real},
        y::AbstractVector{<:Real},
        lo::Real,
        hi::Real,
        level::Real,
        interval_alpha::Float64;
        show_legend::Bool = false)
    sliced = _density_interval_slice(x, y, lo, hi)
    sliced === nothing && return false
    xs, ys = sliced
    plot!(p, xs, ys;
        fillrange = 0.0,
        fillcolor = COLOR_CI,
        fillalpha = interval_alpha,
        linealpha = 0.0,
        linewidth = 0.0,
        label = show_legend ? "$(round(Int, 100 * level))% Interval" : "")
    return true
end

"""
    plot_uq_distributions(uq::UQResult;
                          scale=:natural,
                          parameters=nothing,
                          interval_alpha=0.22,
                          histogram_alpha=0.45,
                          show_estimate=true,
                          show_interval=true,
                          show_legend=false,
                          bins=:auto,
                          plot_type=:density,
                          kde_bandwidth=nothing,
                          ncols=3,
                          style=PlotStyle(),
                          kwargs_subplot=NamedTuple(),
                          kwargs_layout=NamedTuple(),
                          save_path=nothing)

Plot marginal parameter distributions from a `UQResult`.

For `:chain` and `:mcmc_refit` backends, draws are shown as a KDE or histogram. For the
`:wald` backend, analytic Gaussian approximations are plotted where the parameter is on
a transformed scale; otherwise KDE is used. Point estimates and credible/confidence
intervals are overlaid as vertical lines and shaded regions.

# Arguments
- `uq::UQResult` - Uncertainty quantification result from `compute_uq`.

# Keyword Arguments
- `scale::Symbol` - Parameter scale for display: `:natural` (default) or `:transformed`.
- `parameters` - `Symbol`, vector of `Symbol`s, or `nothing` (all parameters, default).
- `interval_alpha::Float64` - Opacity of the shaded interval region (default: `0.22`).
- `histogram_alpha::Float64` - Opacity of histogram bars (default: `0.45`).
- `show_estimate::Bool` - Show point estimate as a vertical line (default: `true`).
- `show_interval::Bool` - Shade the credible/confidence interval (default: `true`).
- `show_legend::Bool` - Show plot legend (default: `false`).
- `bins` - Histogram bin count or `:auto` (default).
- `plot_type::Symbol` - `:density` (KDE, default) or `:histogram`.
- `kde_bandwidth::Union{Nothing, Float64}` - KDE bandwidth; `nothing` uses automatic
  selection (default).
- `ncols::Int` - Number of subplot columns (default: `3`).
- `style::PlotStyle` - Visual style configuration.
- `kwargs_subplot` - Extra keyword arguments forwarded to each subplot.
- `kwargs_layout` - Extra keyword arguments forwarded to the layout call.
- `save_path::Union{Nothing, String}` - File path to save the plot, or `nothing`.

# Returns
A `Plots.jl` plot object showing one panel per selected parameter.
"""
function plot_uq_distributions(uq::UQResult;
        scale::Symbol = :natural,
        parameters = nothing,
        interval_alpha::Float64 = 0.22,
        histogram_alpha::Float64 = 0.45,
        show_estimate::Bool = true,
        show_interval::Bool = true,
        show_legend::Bool = false,
        bins = :auto,
        plot_type::Symbol = :density,
        kde_bandwidth::Union{Nothing, Float64} = nothing,
        ncols::Int = DEFAULT_PLOT_COLS,
        style::PlotStyle = PlotStyle(),
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing)
    save_path = _resolve_plot_path(save_path, plot_path)
    plot_type in (:density, :histogram) ||
        error("plot_type must be :density or :histogram.")
    backend = get_uq_backend(uq)
    draws = get_uq_draws(uq; scale = scale)
    vcov_t = backend == :wald ? get_uq_vcov(uq; scale = :transformed) : nothing
    est_t = backend == :wald ?
            get_uq_estimates(uq; scale = :transformed, as_component = false) : nothing
    coord_transforms = backend == :wald ? _uq_wald_coord_transforms(uq) : nothing

    if plot_type == :histogram && draws === nothing
        error("Histogram plotting requires UQ draws for scale=$(scale). This backend/scale pair does not provide draws.")
    end

    est = get_uq_estimates(uq; scale = scale, as_component = false)
    ints = get_uq_intervals(uq; scale = scale, as_component = false)
    names = get_uq_parameter_names(uq; scale = scale)
    idx = _uq_param_indices(uq, parameters; scale = scale)
    pidx = length(idx)
    pidx >= 1 || error("No parameters selected for UQ plotting.")
    closed_form_kinds = [plot_type == :density ?
                         _wald_closed_form_kind(
                             backend, scale, j, vcov_t, coord_transforms) : :none
                         for j in idx]
    n_closed = count(!=(:none), closed_form_kinds)
    analytic_wald_all = backend == :wald && plot_type == :density && n_closed == pidx
    mixed_wald = backend == :wald && plot_type == :density && n_closed > 0 &&
                 n_closed < pidx
    if plot_type == :density && draws === nothing && n_closed < pidx
        error("UQ backend $(backend) does not store parameter draws for scale=$(scale). At least one selected parameter requires sampling + KDE.")
    end

    plots = Vector{Any}(undef, pidx)
    y_label = _uq_density_ylabel(backend; analytic_wald = analytic_wald_all,
        mixed_wald = mixed_wald, plot_type = plot_type)
    kde_fallback_params = Symbol[]
    for (k, j) in enumerate(idx)
        pname = _uq_param_label(names[j])
        p = create_styled_plot(title = pname,
            xlabel = pname,
            ylabel = y_label,
            style = style,
            legend = show_legend ? :best : false,
            kwargs_subplot...)
        xlims_param = nothing

        if plot_type == :histogram
            x = draws[:, j]
            xlims_param = _uq_merge_limits(xlims_param, minimum(x), maximum(x))
            if show_interval && ints !== nothing
                lo = ints.lower[j]
                hi = ints.upper[j]
                if isfinite(lo) && isfinite(hi)
                    vspan!(p, [lo, hi];
                        color = COLOR_CI,
                        alpha = interval_alpha,
                        label = show_legend ? "$(round(Int, 100 * ints.level))% Interval" :
                                "")
                    xlims_param = _uq_merge_limits(xlims_param, lo, hi)
                end
            end
            histogram!(p, x;
                normalize = :pdf,
                bins = bins,
                color = style.color_primary,
                alpha = histogram_alpha,
                linecolor = style.color_primary,
                linewidth = 0.5,
                label = show_legend ? "Histogram" : "")
        else
            kind = closed_form_kinds[k]
            if kind == :normal || kind == :lognormal || kind == :logitnormal
                xy = _wald_density_xy(kind, est_t[j], vcov_t[j, j])
                if show_interval && ints !== nothing && xy !== nothing
                    lo = ints.lower[j]
                    hi = ints.upper[j]
                    if isfinite(lo) && isfinite(hi)
                        _plot_density_interval_fill!(p, xy[1], xy[2], lo, hi, ints.level,
                            interval_alpha; show_legend = show_legend)
                        xlims_param = _uq_merge_limits(xlims_param, lo, hi)
                    end
                end
                if xy === nothing
                    vline!(p, [est[j]];
                        color = style.color_primary,
                        linewidth = style.line_width_primary,
                        label = show_legend ? "Approx. Density" : "")
                    xlims_param = _uq_merge_limits(xlims_param, est[j], est[j])
                else
                    plot!(p, xy[1], xy[2];
                        color = style.color_primary,
                        linewidth = style.line_width_primary,
                        label = show_legend ? "Approx. Density" : "")
                    xlims_param = _uq_merge_limits(
                        xlims_param, minimum(xy[1]), maximum(xy[1]))
                end
            else
                x = draws[:, j]
                xk, yk = _uq_kde_xy(x; bandwidth = kde_bandwidth)
                xlims_param = _uq_merge_limits(xlims_param, minimum(xk), maximum(xk))
                if show_interval && ints !== nothing
                    lo = ints.lower[j]
                    hi = ints.upper[j]
                    if isfinite(lo) && isfinite(hi)
                        _plot_density_interval_fill!(p, xk, yk, lo, hi, ints.level,
                            interval_alpha; show_legend = show_legend)
                        xlims_param = _uq_merge_limits(xlims_param, lo, hi)
                    end
                end
                plot!(p, xk, yk;
                    color = style.color_primary,
                    linewidth = style.line_width_primary,
                    alpha = 1.0,
                    label = show_legend ? "KDE" : "")
                push!(kde_fallback_params, names[j])
            end
        end

        if show_estimate
            vline!(p, [est[j]];
                color = style.color_dark,
                linewidth = style.line_width_secondary,
                linestyle = :dash,
                label = show_legend ? "Estimate" : "")
            xlims_param = _uq_merge_limits(xlims_param, est[j], est[j])
        end

        if xlims_param !== nothing
            plot!(p; xlims = _pad_limits(xlims_param[1], xlims_param[2]))
        end

        plots[k] = p
    end
    if plot_type == :density && !isempty(kde_fallback_params)
        @info "plot_uq_distributions used sampling + KDE because no closed-form density is available." backend=backend scale=scale parameters=kde_fallback_params
    end

    p = combine_plots(plots; ncols = ncols, kwargs_layout...)
    return _save_plot!(p, save_path)
end
