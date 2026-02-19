export plot_random_effect_distributions
export plot_random_effect_pit
export plot_random_effect_standardized
export plot_random_effect_standardized_scatter
export plot_random_effects_pdf
export plot_random_effect_pairplot
export plot_random_effects_scatter

using Distributions
using KernelDensity
using Plots
using Random
using Statistics

function _require_re_supported(res::FitResult)
    if res.result isa MLEResult || res.result isa MAPResult
        @warn "Random-effects diagnostics are not available for MLE/MAP."
        error("Random-effects diagnostics require Laplace/LaplaceMAP/FOCEI/FOCEIMAP/MCEM/SAEM/MCMC.")
    end
end

function _resolve_re_names(dm::DataModel, re_names)
    names = get_re_names(dm.model.random.random)
    isempty(names) && error("Model has no random effects.")
    re_list = re_names === nothing ? names : (re_names isa AbstractVector ? collect(re_names) : [re_names])
    for r in re_list
        r in names || error("Random effect $(r) not found. Available: $(names).")
    end
    return re_list
end

function _fit_constants_re(res::FitResult)
    if hasproperty(res, :fit_kwargs)
        kw = res.fit_kwargs
        return haskey(kw, :constants_re) ? getfield(kw, :constants_re) : NamedTuple()
    end
    return NamedTuple()
end

function _filter_re_without_covariates(res::FitResult, re_list)
    usage = get_re_covariate_usage(res)
    out = Symbol[]
    for r in re_list
        used = hasproperty(usage, r) ? getfield(usage, r) : Symbol[]
        isempty(used) && push!(out, r)
    end
    return out
end

function _resolve_levels(dm::DataModel, re::Symbol, levels, individuals_idx)
    levels_all = getfield(dm.re_group_info.values, re)
    if individuals_idx !== nothing
        inds = _resolve_individuals(dm, individuals_idx)
        re_groups = get_re_groups(dm.model.random.random)
        col = getfield(re_groups, re)
        selected = Set{Any}()
        for i in inds
            g = getfield(dm.individuals[i].re_groups, re)
            if g isa AbstractVector
                for lvl in g
                    push!(selected, lvl)
                end
            else
                push!(selected, g)
            end
        end
        levels_all = [lvl for lvl in levels_all if lvl in selected]
    end
    if levels === nothing
        return levels_all
    end
    lv = levels isa AbstractVector ? collect(levels) : [levels]
    for v in lv
        v in levels_all || error("Level $(v) not found for random effect $(re).")
    end
    return lv
end

"""
    plot_random_effects_pdf(res::FitResult; dm, re_names, levels, individuals_idx,
                            shared_x_axis, shared_y_axis, ncols, style, mcmc_draws,
                            mcmc_warmup, mcmc_quantiles, mcmc_quantiles_alpha,
                            flow_samples, flow_plot, flow_bins, flow_bandwidth, rng,
                            save_path, kwargs_subplot, kwargs_layout) -> Plots.Plot

Plot the fitted marginal PDF of each random effect alongside the posterior EBE histogram,
showing how well the parametric distribution fits the estimated random-effect values.

# Keyword Arguments
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
- `re_names`: random-effect name(s) to include, or `nothing` for all.
- `levels`, `individuals_idx`: grouping level or individual filters.
- `shared_x_axis::Bool = true`, `shared_y_axis::Bool = true`: share axes.
- `ncols::Int = 3`: number of subplot columns.
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `mcmc_draws`, `mcmc_warmup`, `mcmc_quantiles`, `mcmc_quantiles_alpha`: MCMC settings.
- `flow_samples::Int = 500`: number of samples for normalizing-flow distributions.
- `flow_plot::Symbol = :kde`: `:kde` or `:hist` for flow-based distributions.
- `flow_bins::Int = 20`, `flow_bandwidth`: histogram bins / KDE bandwidth for flows.
- `rng::AbstractRNG = Random.default_rng()`: random-number generator.
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
- `kwargs_subplot`, `kwargs_layout`: extra keyword arguments for subplots and layout.
"""
function plot_random_effects_pdf(res::FitResult;
                                 dm::Union{Nothing, DataModel}=nothing,
                                 re_names=nothing,
                                 levels=nothing,
                                 individuals_idx=nothing,
                                 shared_x_axis::Bool=true,
                                 shared_y_axis::Bool=true,
                                 ncols::Int=DEFAULT_PLOT_COLS,
                                 style::PlotStyle=PlotStyle(),
                                 mcmc_draws::Int=1000,
                                 mcmc_warmup::Union{Nothing, Int}=nothing,
                                 mcmc_quantiles::Vector{<:Real}=[5, 95],
                                 mcmc_quantiles_alpha::Float64=0.8,
                                 flow_samples::Int=500,
                                 flow_plot::Symbol=:kde,
                                 flow_bins::Int=20,
                                 flow_bandwidth::Union{Nothing, Float64}=nothing,
                                 rng::AbstractRNG=Random.default_rng(),
                                 save_path::Union{Nothing, String}=nothing,
                                 plot_path::Union{Nothing, String}=nothing,
                                 kwargs_subplot=NamedTuple(),
                                 kwargs_layout=NamedTuple())
    dm = _get_dm(res, dm)
    save_path = _resolve_plot_path(save_path, plot_path)
    _require_re_supported(res)
    re_list = _resolve_re_names(dm, re_names)
    re_list = _filter_re_without_covariates(res, re_list)
    if isempty(re_list)
        @warn "No random-effect distributions without covariates to plot."
        p = create_styled_plot(title="No random-effect distributions to plot.", style=style, kwargs_subplot...)
        return _save_plot!(p, save_path)
    end
    is_mcmc = res.result isa MCMCResult

    if is_mcmc
        if mcmc_warmup !== nothing
            conv = res.diagnostics.convergence
            conv = merge(conv, (n_adapt=mcmc_warmup,))
            res = FitResult(res.method, res.result, res.summary,
                            FitDiagnostics(res.diagnostics.timing, res.diagnostics.optimizer, conv, res.diagnostics.notes),
                            res.data_model, res.fit_args, res.fit_kwargs)
        end
        mcmc_quantiles = sort(Float64.(collect(mcmc_quantiles)))
        (length(mcmc_quantiles) >= 2 && all(0 .<= mcmc_quantiles .<= 100)) || error("mcmc_quantiles must be in [0,100] with length >= 2.")
    end

    plots = Vector{Any}()
    xlims = nothing
    ylims = nothing
    merge_limits = (lims, minv, maxv) -> lims === nothing ? (minv, maxv) :
        (min(lims[1], minv), max(lims[2], maxv))

    θ_base = res.result isa MCMCResult ? _mcmc_fixed_means(res, dm)[1] : get_params(res; scale=:untransformed)
    dists_builder = get_create_random_effect_distribution(dm.model.random.random)
    model_funs = get_model_funs(dm.model)
    helpers = get_helper_funs(dm.model)

    θ_draws_cache = nothing
    for re in re_list
        const_cov = dm.individuals[1].const_cov
        dist0 = getproperty(dists_builder(θ_base, const_cov, model_funs, helpers), re)
        dim = dist0 isa Distributions.UnivariateDistribution ? 1 : length(vec(rand(rng, dist0)))
        value_cols = flatten_re_names(re, zeros(dim))
        show_comp = length(value_cols) > 1

        for (ci, comp_name) in enumerate(value_cols)
            title = show_comp ? string(re, " | ", comp_name) : string(re)
            p = create_styled_plot(title=title, xlabel="Random Effect", ylabel="Probability Density", style=style, kwargs_subplot...)

            if is_mcmc
                if θ_draws_cache === nothing
                    constants_re = _fit_constants_re(res)
                    θ_draws_cache, _, _ = _mcmc_drawn_params(res, dm, constants_re, NamedTuple(), mcmc_draws, rng)
                end
                samples_by_draw = Vector{Vector{Float64}}()
                grid_list = Vector{Tuple{Vector{Float64}, Vector{Float64}}}()
                min_v = Inf
                max_v = -Inf
                for θd in θ_draws_cache
                    dist_d = getproperty(dists_builder(θd, const_cov, model_funs, helpers), re)
                    dist_use = dist_d isa Distributions.MultivariateDistribution ? _marginal_normal(dist_d, ci) : dist_d
                    if dist_use === nothing
                        if dist_d isa NormalizingPlanarFlow || dist_d isa Distributions.MultivariateDistribution
                            samp = if dist_d isa Distributions.UnivariateDistribution
                                vec(rand(rng, dist_d, flow_samples))
                            else
                                vals_s = rand(rng, dist_d, flow_samples)
                                vec(vals_s[ci, :])
                            end
                            length(samp) < 2 && continue
                            min_v = min(min_v, minimum(samp))
                            max_v = max(max_v, maximum(samp))
                            push!(samples_by_draw, samp)
                            continue
                        end
                        continue
                    end
                    grid = _density_grid_continuous([dist_use], 0.99, 200)
                    grid === nothing && continue
                    min_v = min(min_v, minimum(grid.y))
                    max_v = max(max_v, maximum(grid.y))
                    push!(grid_list, (grid.y, vec(grid.z[:, 1])))
                end
                if isempty(samples_by_draw) && isempty(grid_list)
                    @warn "Skipping RE pdf plot (insufficient draws)." re=re
                    continue
                end
                xgrid = range(min_v, max_v; length=200)
                xlims = merge_limits(xlims, min_v, max_v)
                curves = Vector{Vector{Float64}}()
                if flow_plot == :hist && !isempty(samples_by_draw)
                    edges = range(min_v, max_v; length=flow_bins + 1)
                    centers = (edges[1:end-1] .+ edges[2:end]) ./ 2
                    xgrid = centers
                    for samp in samples_by_draw
                        counts = zeros(Float64, flow_bins)
                        for s in samp
                            idx = s == max_v ? flow_bins : clamp(floor(Int, (s - min_v) / (max_v - min_v) * flow_bins) + 1, 1, flow_bins)
                            counts[idx] += 1
                        end
                        ssum = sum(counts)
                        ssum == 0 && continue
                        push!(curves, counts ./ ssum)
                    end
                else
                    for samp in samples_by_draw
                        xk, yk = _kde_xy(samp; bandwidth=flow_bandwidth)
                        ygrid = _interp_linear(xk, yk, xgrid)
                        push!(curves, ygrid)
                    end
                    for (xk, yk) in grid_list
                        ygrid = _interp_linear(xk, yk, xgrid)
                        push!(curves, ygrid)
                    end
                end
                isempty(curves) && continue
                curves_mat = reduce(hcat, curves)
                mean_dens = vec(mean(curves_mat; dims=2))
                qlo = vec(mapslices(x -> quantile(x, mcmc_quantiles[1] / 100), curves_mat; dims=2))
                qhi = vec(mapslices(x -> quantile(x, mcmc_quantiles[end] / 100), curves_mat; dims=2))
                plot!(p, xgrid, mean_dens; color=style.color_secondary, label="PDF")
                plot!(p, xgrid, qlo; color=style.color_secondary, alpha=mcmc_quantiles_alpha, linestyle=:dash, label="")
                plot!(p, xgrid, qhi; color=style.color_secondary, alpha=mcmc_quantiles_alpha, linestyle=:dash, label="")
                y_max = maximum(vcat(mean_dens, qhi))
                ylims!(p, (0.0, y_max * 1.05))
                ylims = merge_limits(ylims, 0.0, y_max)
            else
                dist = dist0
                dist_use = dist isa Distributions.MultivariateDistribution ? _marginal_normal(dist, ci) : dist
                if dist_use === nothing
                    if dist isa NormalizingPlanarFlow || dist isa Distributions.MultivariateDistribution
                        samp = if dist isa Distributions.UnivariateDistribution
                            vec(rand(rng, dist, flow_samples))
                        else
                            vals_s = rand(rng, dist, flow_samples)
                            vec(vals_s[ci, :])
                        end
                        length(samp) < 2 && continue
                        if flow_plot == :hist
                            histogram!(p, samp; bins=flow_bins, normalize=:probability,
                                       color=style.color_secondary, label="PDF")
                            ys = p.series_list[end][:y]
                            y_max = maximum(ys)
                            ylims!(p, (0.0, y_max * 1.05))
                            xlims = merge_limits(xlims, minimum(samp), maximum(samp))
                            ylims = merge_limits(ylims, 0.0, y_max)
                        else
                            xk, yk = _kde_xy(samp; bandwidth=flow_bandwidth)
                            plot!(p, xk, yk; color=style.color_secondary, label="PDF")
                            y_max = maximum(yk)
                            ylims!(p, (0.0, y_max * 1.05))
                            xlims = merge_limits(xlims, minimum(xk), maximum(xk))
                            ylims = merge_limits(ylims, 0.0, y_max)
                        end
                    else
                        @warn "Skipping RE pdf plot (missing mean/cov)." re=re
                        continue
                    end
                else
                    if dist_use isa DiscreteDistribution
                        grid = _density_grid_discrete(dist_use, 0.995)
                        grid === nothing && continue
                        bar!(p, grid.vals, grid.probs; color=style.color_secondary, label="PMF")
                        xlims = merge_limits(xlims, minimum(grid.vals), maximum(grid.vals))
                        ylims = merge_limits(ylims, minimum(grid.probs), maximum(grid.probs))
                    else
                        grid = _density_grid_continuous([dist_use], 0.99, 200)
                        grid === nothing && continue
                        pdf_vals = vec(grid.z[:, 1])
                        plot!(p, grid.y, pdf_vals; color=style.color_secondary, label="PDF")
                        xlims = merge_limits(xlims, minimum(grid.y), maximum(grid.y))
                        ylims = merge_limits(ylims, minimum(pdf_vals), maximum(pdf_vals))
                    end
                end
            end

            push!(plots, p)
        end
    end

    if shared_x_axis || shared_y_axis
        xlim_use = shared_x_axis && xlims !== nothing ? _pad_limits(xlims[1], xlims[2]) : nothing
        ylim_use = shared_y_axis && ylims !== nothing ? _pad_limits(ylims[1], ylims[2]) : nothing
        _apply_shared_axes!(plots, xlim_use, ylim_use)
    end
    if isempty(plots)
        @warn "No random-effect pdf plots to display."
        p = create_styled_plot(title="No random-effect pdf plots to display.", style=style, kwargs_subplot...)
        return _save_plot!(p, save_path)
    end
    p = combine_plots(plots; ncols=ncols, kwargs_layout...)
    return _save_plot!(p, save_path)
end

"""
    plot_random_effects_scatter(res::FitResult; dm, re_names, levels, individuals_idx,
                                x_covariate, mcmc_draws, ncols, style, save_path,
                                kwargs_subplot, kwargs_layout) -> Plots.Plot

Scatter plot of empirical-Bayes estimates for each random effect against a constant
covariate or group level index, useful for detecting covariate relationships.

# Keyword Arguments
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
- `re_names`: random-effect name(s) to include, or `nothing` for all.
- `levels`, `individuals_idx`: grouping level or individual filters.
- `x_covariate::Union{Nothing, Symbol} = nothing`: constant covariate for the x-axis;
  defaults to the group level index.
- `mcmc_draws::Int = 1000`: MCMC draws for posterior mean EBE.
- `ncols::Int = 3`: number of subplot columns.
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
- `kwargs_subplot`, `kwargs_layout`: extra keyword arguments.
"""
function plot_random_effects_scatter(res::FitResult;
                                     dm::Union{Nothing, DataModel}=nothing,
                                     re_names=nothing,
                                     levels=nothing,
                                     individuals_idx=nothing,
                                     x_covariate::Union{Nothing, Symbol}=nothing,
                                     mcmc_draws::Int=1000,
                                     ncols::Int=DEFAULT_PLOT_COLS,
                                     style::PlotStyle=PlotStyle(),
                                     save_path::Union{Nothing, String}=nothing,
                                     plot_path::Union{Nothing, String}=nothing,
                                     kwargs_subplot=NamedTuple(),
                                     kwargs_layout=NamedTuple())
    dm = _get_dm(res, dm)
    save_path = _resolve_plot_path(save_path, plot_path)
    _require_re_supported(res)
    re_list = _resolve_re_names(dm, re_names)
    re_list = _filter_re_without_covariates(res, re_list)
    if isempty(re_list)
        @warn "No random-effect distributions without covariates to plot."
        p = create_styled_plot(title="No random-effect scatters to plot.", style=style, kwargs_subplot...)
        return _save_plot!(p, save_path)
    end
    if x_covariate !== nothing
        cov = dm.model.covariates.covariates
        x_covariate in cov.constants || error("x_covariate must be a constant covariate.")
    end

    plots = Vector{Any}()
    xlims_val = nothing

    for re in re_list
        level_to_ind = _level_to_individual(dm, re)
        lvls = _resolve_levels(dm, re, levels, individuals_idx)
        ebe_map, value_cols = res.result isa MCMCResult ? _ebe_by_level_mcmc(dm, res, re, mcmc_draws, Random.default_rng()) : _ebe_by_level(dm, res, re)
        show_comp = length(value_cols) > 1
        lvls_use = [lvl for lvl in lvls if haskey(ebe_map, lvl)]
        isempty(lvls_use) && (lvls_use = collect(keys(ebe_map)))
        for (ci, comp_name) in enumerate(value_cols)
            xs = Float64[]
            ys = Float64[]
            for (k, lvl) in enumerate(lvls_use)
                haskey(ebe_map, lvl) || continue
                ind_idx = level_to_ind[lvl]
                const_cov = dm.individuals[ind_idx].const_cov
                xv = if x_covariate === nothing
                    lvl isa Number ? Float64(lvl) : Float64(k)
                else
                    xv_raw = getfield(const_cov, x_covariate)
                    ismissing(xv_raw) ? nothing : Float64(xv_raw)
                end
                xv === nothing && continue
                yv = Float64(ebe_map[lvl][ci])
                (isfinite(xv) && isfinite(yv)) || continue
                push!(xs, xv)
                push!(ys, yv)
            end
            title = show_comp ? string(re, " | ", comp_name) : string(re)
            xlabel = x_covariate === nothing ? (any(lvl -> lvl isa Number, lvls_use) ? "Level" : "Index") : _axis_label(x_covariate)
            ylabel = res.result isa MCMCResult ? "Posterior Mean EBE" : "Empirical Bayes Estimate (EBE)"
            p = create_styled_plot(title=title, xlabel=xlabel, ylabel=ylabel, style=style, kwargs_subplot...)
            create_styled_scatter!(p, xs, ys; label="", color=style.color_secondary, style=style)
            push!(plots, p)
            if !isempty(xs)
                xlims_val = xlims_val === nothing ? (minimum(xs), maximum(xs)) :
                            (min(xlims_val[1], minimum(xs)), max(xlims_val[2], maximum(xs)))
            end
        end
    end

    if isempty(plots)
        @warn "No random-effect scatters to plot."
        p = create_styled_plot(title="No random-effect scatters to plot.", style=style, kwargs_subplot...)
        return _save_plot!(p, save_path)
    end
    if xlims_val !== nothing
        _apply_shared_axes!(plots, _pad_limits(xlims_val[1], xlims_val[2]), nothing)
    end
    p = combine_plots(plots; ncols=ncols, kwargs_layout...)
    return _save_plot!(p, save_path)
end

"""
    plot_random_effect_pairplot(res::FitResult; dm, re_names, levels, individuals_idx,
                                ncols, style, kde_bandwidth, mcmc_draws, rng, save_path,
                                kwargs_subplot, kwargs_layout) -> Plots.Plot

Pairplot (scatter matrix) of empirical-Bayes estimates across all pairs of random
effects, useful for visualising correlations and joint structure.

# Keyword Arguments
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
- `re_names`: random-effect names to include, or `nothing` for all.
- `levels`, `individuals_idx`: grouping level or individual filters.
- `ncols::Int = 3`: number of subplot columns.
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `kde_bandwidth::Union{Nothing, Float64} = nothing`: KDE bandwidth for diagonal panels.
- `mcmc_draws::Int = 1000`: MCMC draws for posterior mean EBE.
- `rng::AbstractRNG = Random.default_rng()`: random-number generator.
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
- `kwargs_subplot`, `kwargs_layout`: extra keyword arguments.
"""
function plot_random_effect_pairplot(res::FitResult;
                                     dm::Union{Nothing, DataModel}=nothing,
                                     re_names=nothing,
                                     levels=nothing,
                                     individuals_idx=nothing,
                                     ncols::Int=DEFAULT_PLOT_COLS,
                                     style::PlotStyle=PlotStyle(),
                                     kde_bandwidth::Union{Nothing, Float64}=nothing,
                                     mcmc_draws::Int=1000,
                                     rng::AbstractRNG=Random.default_rng(),
                                     save_path::Union{Nothing, String}=nothing,
                                     plot_path::Union{Nothing, String}=nothing,
                                     kwargs_subplot=NamedTuple(),
                                     kwargs_layout=NamedTuple())
    dm = _get_dm(res, dm)
    save_path = _resolve_plot_path(save_path, plot_path)
    _require_re_supported(res)
    re_list = _resolve_re_names(dm, re_names)
    re_list = _filter_re_without_covariates(res, re_list)
    if isempty(re_list)
        @warn "No random-effect distributions without covariates to plot."
        p = create_styled_plot(title="No random-effect pairplots to plot.", style=style, kwargs_subplot...)
        return _save_plot!(p, save_path)
    end

    # Group by RE column (same IDs).
    re_groups = get_re_groups(dm.model.random.random)
    group_map = Dict{Symbol, Vector{Symbol}}()
    for r in re_list
        col = getfield(re_groups, r)
        haskey(group_map, col) || (group_map[col] = Symbol[])
        push!(group_map[col], r)
    end

    plots = Vector{Any}()
    max_nvars = 1
    for (_, res_group) in group_map
        level_to_ind = _level_to_individual(dm, res_group[1])
        lvls = _resolve_levels(dm, res_group[1], levels, individuals_idx)
        ebe_maps = Dict{Symbol, Any}()
        comp_names = Dict{Symbol, Vector{Symbol}}()
        for r in res_group
            ebe_map, value_cols = res.result isa MCMCResult ? _ebe_by_level_mcmc(dm, res, r, mcmc_draws, rng) : _ebe_by_level(dm, res, r)
            ebe_maps[r] = ebe_map
            comp_names[r] = Symbol.(value_cols)
        end

        # Build data matrix across levels.
        lvls_use = [lvl for lvl in lvls if all(haskey(ebe_maps[r], lvl) for r in res_group)]
        isempty(lvls_use) && (lvls_use = collect(keys(ebe_maps[res_group[1]])))
        labels = Symbol[]
        for r in res_group
            cols = comp_names[r]
            for c in cols
                if length(cols) == 1
                    push!(labels, Symbol(string(r)))
                else
                    push!(labels, Symbol(string(r), "_", c))
                end
            end
        end
        nvars = length(labels)
        max_nvars = max(max_nvars, nvars)
        nvars == 0 && continue
        vals = Matrix{Float64}(undef, length(lvls_use), nvars)
        for (i, lvl) in enumerate(lvls_use)
            k = 1
            for r in res_group
                cols = comp_names[r]
                for (ci, _) in enumerate(cols)
                    vals[i, k] = Float64(ebe_maps[r][lvl][ci])
                    k += 1
                end
            end
        end

        # Build pair plot.
        cell_plots = Vector{Any}(undef, nvars * nvars)
        for i in 1:nvars
            for j in 1:nvars
                idx = (i - 1) * nvars + j
                if i == j
                    p = create_styled_plot(title=string(labels[i]), xlabel="", ylabel="", style=style, kwargs_subplot...)
                    histogram!(p, vals[:, i]; bins=20, normalize=:probability, color=style.color_secondary, label="")
                    ys = p.series_list[end][:y]
                    y_max = maximum(ys)
                    ylims!(p, (0.0, y_max * 1.05))
                    if kde_bandwidth !== nothing
                        xk, yk = _kde_xy(vals[:, i]; bandwidth=kde_bandwidth)
                        plot!(p, xk, yk; color=style.color_secondary, label="")
                        y_max = max(y_max, maximum(yk))
                        ylims!(p, (0.0, y_max * 1.05))
                    end
                else
                    p = create_styled_plot(title="", xlabel=_axis_label(labels[j]), ylabel=_axis_label(labels[i]), style=style, kwargs_subplot...)
                    create_styled_scatter!(p, vals[:, j], vals[:, i]; label="", color=style.color_secondary, style=style)
                end
                cell_plots[idx] = p
            end
        end
        append!(plots, cell_plots)
    end

    if isempty(plots)
        @warn "No random-effect pairplots to plot."
        p = create_styled_plot(title="No random-effect pairplots to plot.", style=style, kwargs_subplot...)
        return _save_plot!(p, save_path)
    end
    ncols_use = isempty(group_map) ? ncols : max_nvars
    p = combine_plots(plots; ncols=ncols_use, kwargs_layout...)
    return _save_plot!(p, save_path)
end

function _level_to_individual(dm::DataModel, re::Symbol)
    map = Dict{Any, Int}()
    for (i, ind) in enumerate(dm.individuals)
        g = getfield(ind.re_groups, re)
        if g isa AbstractVector
            for lvl in g
                haskey(map, lvl) || (map[lvl] = i)
            end
        else
            haskey(map, g) || (map[g] = i)
        end
    end
    return map
end

function _ebe_by_level(dm::DataModel, res::FitResult, re::Symbol)
    # Skip fixed levels from constants_re to avoid empty bstars for those levels.
    constants_re = _fit_constants_re(res)
    re_df = get_random_effects(dm, res; constants_re=constants_re, flatten=true, include_constants=false)
    df = getproperty(re_df, re)
    re_groups = get_re_groups(dm.model.random.random)
    col = Symbol(getfield(re_groups, re))
    levels = df[!, col]
    value_cols = [c for c in names(df) if Symbol(c) != col]
    out = Dict{Any, Vector{Float64}}()
    for (i, lvl) in enumerate(levels)
        out[lvl] = Float64.(collect(df[i, value_cols]))
    end
    return out, value_cols
end

function _level_values_from_eta(dm::DataModel, re::Symbol, η_vec::Vector{ComponentArray})
    out = Dict{Any, Vector{Float64}}()
    for (i, ind) in enumerate(dm.individuals)
        g = getfield(ind.re_groups, re)
        v = getproperty(η_vec[i], re)
        if g isa AbstractVector
            if length(g) == 1
                lvl = g[1]
                out[lvl] = v isa Number ? [Float64(v)] : Float64.(collect(vec(v)))
            else
                for (gi, lvl) in pairs(g)
                    val = v isa Number ? Float64(v) : v[gi]
                    out[lvl] = val isa Number ? [Float64(val)] : Float64.(collect(vec(val)))
                end
            end
        else
            out[g] = v isa Number ? [Float64(v)] : Float64.(collect(vec(v)))
        end
    end
    return out
end

function _ebe_by_level_mcmc(dm::DataModel, res::FitResult, re::Symbol, mcmc_draws::Int, rng::AbstractRNG)
    constants_re = _fit_constants_re(res)
    θ_draws, η_draws, _ = _mcmc_drawn_params(res, dm, constants_re, NamedTuple(), mcmc_draws, rng)
    sums = Dict{Any, Vector{Float64}}()
    counts = Dict{Any, Int}()
    for d in eachindex(η_draws)
        levels = _level_values_from_eta(dm, re, η_draws[d])
        for (lvl, v) in levels
            if !haskey(sums, lvl)
                sums[lvl] = zeros(Float64, length(v))
                counts[lvl] = 0
            end
            sums[lvl] .+= v
            counts[lvl] += 1
        end
    end
    out = Dict{Any, Vector{Float64}}()
    for (lvl, s) in sums
        out[lvl] = s ./ counts[lvl]
    end
    dim = isempty(out) ? 1 : length(first(values(out)))
    value_cols = flatten_re_names(re, zeros(dim))
    return out, value_cols
end

function _standardize_re(dist, val::Vector{Float64}; flow_samples::Int=500)
    if dist isa NormalizingPlanarFlow
        n = flow_samples
        if dist isa Distributions.UnivariateDistribution
            samp = vec(rand(dist, n))
            μ = mean(samp)
            σ = std(samp)
            σ == 0 && return nothing
            return [(val[1] - μ) / σ]
        else
            samp = rand(dist, n)
            μv = vec(mean(samp; dims=2))
            Σm = cov(permutedims(samp))
            L = try
                cholesky(Σm).L
            catch
                return nothing
            end
            return L \ (val .- μv)
        end
    end
    if dist isa Distributions.MultivariateDistribution && !(dist isa MvNormal)
        @warn "Skipping multivariate RE standardization for non-Normal distribution." dist=typeof(dist)
        return nothing
    end
    μ = try
        Distributions.mean(dist)
    catch
        return nothing
    end
    Σ = try
        dist isa Distributions.MultivariateDistribution ? cov(dist) : var(dist)
    catch
        return nothing
    end
    μv = μ isa Number ? [Float64(μ)] : Float64.(collect(vec(μ)))

    Σm = Σ isa Number ? [Float64(Σ)] : Matrix{Float64}(reshape(collect(Σ), size(Σ)...))
    if length(μv) == 1
        σ = sqrt(Σm[1])
        σ == 0 && return nothing
        return [(val[1] - μv[1]) / σ]
    end
    L = try
        cholesky(Σm).L
    catch
        return nothing
    end
    return L \ (val .- μv)
end

function _marginal_normal(dist, i::Int)
    dist isa MvNormal || return nothing
    μ = try
        Distributions.mean(dist)
    catch
        return nothing
    end
    Σ = try
        cov(dist)
    catch
        return nothing
    end
    μv = μ isa Number ? [Float64(μ)] : Float64.(collect(vec(μ)))
    Σm = Σ isa Number ? [Float64(Σ)] : Matrix{Float64}(Σ)
    i <= length(μv) || return nothing
    σ2 = Σm[i, i]
    return Normal(μv[i], sqrt(σ2))
end

function _kde_xy(vals; bandwidth=nothing)
    kd = bandwidth === nothing ? kde(vals) : kde(vals; bandwidth=bandwidth)
    return kd.x, kd.density
end

function _interp_linear(x_src::AbstractVector, y_src::AbstractVector, x_tgt::AbstractVector)
    n = length(x_src)
    n == length(y_src) || error("Interpolation source x/y lengths differ.")
    n < 2 && return fill(y_src[1], length(x_tgt))
    y_tgt = similar(x_tgt, Float64)
    j = 1
    x1 = x_src[1]
    x2 = x_src[2]
    y1 = y_src[1]
    y2 = y_src[2]
    for (i, x) in enumerate(x_tgt)
        while x > x2 && j < n - 1
            j += 1
            x1 = x2
            y1 = y2
            x2 = x_src[j + 1]
            y2 = y_src[j + 1]
        end
        if x <= x1
            y_tgt[i] = y1
        elseif x >= x2
            y_tgt[i] = y2
        else
            t = (x - x1) / (x2 - x1)
            y_tgt[i] = y1 + t * (y2 - y1)
        end
    end
    return y_tgt
end

function _mahalanobis(dist, val::Vector{Float64})
    if dist isa Distributions.MultivariateDistribution && !(dist isa MvNormal)
        @warn "Skipping Mahalanobis distance for non-Normal multivariate RE." dist=typeof(dist)
        return nothing
    end
    μ = try
        Distributions.mean(dist)
    catch
        return nothing
    end
    Σ = try
        cov(dist)
    catch
        return nothing
    end
    μv = μ isa Number ? [Float64(μ)] : Float64.(collect(vec(μ)))
    Σm = Σ isa Number ? [Float64(Σ)] : Matrix{Float64}(Σ)
    d = val .- μv
    if length(d) == 1
        return (d[1]^2) / Σm[1]
    end
    invΣ = try
        inv(Σm)
    catch
        return nothing
    end
    return dot(d, invΣ * d)
end

function _pit_value(dist, val::Float64)
    applicable(cdf, dist, val) || return nothing
    return cdf(dist, val)
end

"""
    plot_random_effect_distributions(res::FitResult; dm, re_names, levels,
                                     individuals_idx, shared_x_axis, shared_y_axis,
                                     ncols, style, mcmc_draws, mcmc_warmup,
                                     mcmc_quantiles, mcmc_quantiles_alpha, flow_samples,
                                     flow_plot, flow_bins, flow_bandwidth, rng,
                                     save_path, kwargs_subplot, kwargs_layout)
                                     -> Plots.Plot

Plot empirical and fitted distributions for each random effect side-by-side,
combining the EBE histogram with the parametric prior PDF.

# Keyword Arguments
All arguments are identical to [`plot_random_effects_pdf`](@ref).
"""
function plot_random_effect_distributions(res::FitResult;
                                          dm::Union{Nothing, DataModel}=nothing,
                                          re_names=nothing,
                                          levels=nothing,
                                          individuals_idx=nothing,
                                          shared_x_axis::Bool=true,
                                          shared_y_axis::Bool=true,
                                          ncols::Int=DEFAULT_PLOT_COLS,
                                          style::PlotStyle=PlotStyle(),
                                          mcmc_draws::Int=1000,
                                          mcmc_warmup::Union{Nothing, Int}=nothing,
                                          mcmc_quantiles::Vector{<:Real}=[5, 95],
                                          mcmc_quantiles_alpha::Float64=0.8,
                                          flow_samples::Int=500,
                                          flow_plot::Symbol=:kde,
                                          flow_bins::Int=20,
                                          flow_bandwidth::Union{Nothing, Float64}=nothing,
                                          rng::AbstractRNG=Random.default_rng(),
                                          save_path::Union{Nothing, String}=nothing,
                                          plot_path::Union{Nothing, String}=nothing,
                                          kwargs_subplot=NamedTuple(),
                                          kwargs_layout=NamedTuple())
    dm = _get_dm(res, dm)
    save_path = _resolve_plot_path(save_path, plot_path)
    _require_re_supported(res)
    re_list = _resolve_re_names(dm, re_names)
    is_mcmc = res.result isa MCMCResult

    if is_mcmc
        if mcmc_warmup !== nothing
            conv = res.diagnostics.convergence
            conv = merge(conv, (n_adapt=mcmc_warmup,))
            res = FitResult(res.method, res.result, res.summary,
                            FitDiagnostics(res.diagnostics.timing, res.diagnostics.optimizer, conv, res.diagnostics.notes),
                            res.data_model, res.fit_args, res.fit_kwargs)
        end
        mcmc_quantiles = sort(Float64.(collect(mcmc_quantiles)))
        (length(mcmc_quantiles) >= 2 && all(0 .<= mcmc_quantiles .<= 100)) || error("mcmc_quantiles must be in [0,100] with length >= 2.")
    end

    plots = Vector{Any}()
    xlims_val = nothing
    xlims = nothing
    ylims = nothing
    θ_draws_cache = nothing

    θ_base = res.result isa MCMCResult ? _mcmc_fixed_means(res, dm)[1] : get_params(res; scale=:untransformed)
    dists_builder = get_create_random_effect_distribution(dm.model.random.random)
    model_funs = get_model_funs(dm.model)
    helpers = get_helper_funs(dm.model)

    for re in re_list
        level_to_ind = _level_to_individual(dm, re)
        lvls = _resolve_levels(dm, re, levels, individuals_idx)
        ebe_map, value_cols = is_mcmc ? _ebe_by_level_mcmc(dm, res, re, mcmc_draws, rng) : _ebe_by_level(dm, res, re)
        show_comp = length(value_cols) > 1
        lvls_use = [lvl for lvl in lvls if haskey(ebe_map, lvl)]
        isempty(lvls_use) && (lvls_use = collect(keys(ebe_map)))

        for lvl in lvls_use
            haskey(ebe_map, lvl) || continue
            ind_idx = level_to_ind[lvl]
            const_cov = dm.individuals[ind_idx].const_cov
            dist = getproperty(dists_builder(θ_base, const_cov, model_funs, helpers), re)
            vals = ebe_map[lvl]
            for (ci, comp_name) in enumerate(value_cols)
                val = [vals[ci]]
                dist_use = dist isa Distributions.MultivariateDistribution ? _marginal_normal(dist, ci) : dist
                if dist_use === nothing
                    if dist isa NormalizingPlanarFlow
                        # Flow: approximate marginal via sampling + KDE/hist.
                        title = show_comp ? string(re, " | ", dm.config.primary_id, "=", lvl, " | ", comp_name) :
                                string(re, " | ", dm.config.primary_id, "=", lvl)
                        p = create_styled_plot(title=title, xlabel="Random Effect", ylabel="Probability Density", style=style, kwargs_subplot...)
                        ebe_label = is_mcmc ? "posterior mean" : "EBE"
                        if is_mcmc
                            @info "Flow marginal plotted via sampling (MCMC averaged)." re=re level=lvl samples=flow_samples
                            if θ_draws_cache === nothing
                                constants_re = _fit_constants_re(res)
                                θ_draws_cache, _, _ = _mcmc_drawn_params(res, dm, constants_re, NamedTuple(), mcmc_draws, rng)
                            end
                            samples_by_draw = Vector{Vector{Float64}}()
                            min_v = Inf
                            max_v = -Inf
                            for θd in θ_draws_cache
                                dist_d = getproperty(dists_builder(θd, const_cov, model_funs, helpers), re)
                                samp = if dist_d isa Distributions.UnivariateDistribution
                                    vec(rand(dist_d, flow_samples))
                                else
                                    vals = rand(dist_d, flow_samples)
                                    vec(vals[ci, :])
                                end
                                length(samp) < 2 && continue
                                min_v = min(min_v, minimum(samp))
                                max_v = max(max_v, maximum(samp))
                                push!(samples_by_draw, samp)
                            end
                            if isempty(samples_by_draw)
                                @warn "Skipping RE flow plot (insufficient samples)." re=re level=lvl
                                continue
                            end
                            if flow_plot == :hist
                                edges = range(min_v, max_v; length=flow_bins + 1)
                                centers = (edges[1:end-1] .+ edges[2:end]) ./ 2
                                counts_list = Vector{Vector{Float64}}()
                                for samp in samples_by_draw
                                    counts = zeros(Float64, flow_bins)
                                    for s in samp
                                        idx = s == max_v ? flow_bins : clamp(floor(Int, (s - min_v) / (max_v - min_v) * flow_bins) + 1, 1, flow_bins)
                                        counts[idx] += 1
                                    end
                                    ssum = sum(counts)
                                    ssum == 0 && continue
                                    push!(counts_list, counts ./ ssum)
                                end
                                if isempty(counts_list)
                                    @warn "Skipping RE flow histogram (insufficient samples)." re=re level=lvl
                                    continue
                                end
                                counts_mat = reduce(hcat, counts_list)
                                mean_counts = vec(mean(counts_mat; dims=2))
                                qlo = vec(mapslices(x -> quantile(x, mcmc_quantiles[1] / 100), counts_mat; dims=2))
                                qhi = vec(mapslices(x -> quantile(x, mcmc_quantiles[end] / 100), counts_mat; dims=2))
                                bar!(p, centers, mean_counts; bar_width=edges[2] - edges[1],
                                     color=style.color_secondary, label="flow")
                                plot!(p, centers, qlo; color=style.color_secondary, alpha=mcmc_quantiles_alpha, linestyle=:dash, label="")
                                plot!(p, centers, qhi; color=style.color_secondary, alpha=mcmc_quantiles_alpha, linestyle=:dash, label="")
                                y_max = maximum(vcat(mean_counts, qhi))
                                ylims!(p, (0.0, y_max * 1.05))
                            else
                                xgrid = range(min_v, max_v; length=200)
                                dens_list = Vector{Vector{Float64}}()
                                for samp in samples_by_draw
                                    xk, yk = _kde_xy(samp; bandwidth=flow_bandwidth)
                                    ygrid = _interp_linear(xk, yk, xgrid)
                                    push!(dens_list, ygrid)
                                end
                                dens_mat = reduce(hcat, dens_list)
                                mean_dens = vec(mean(dens_mat; dims=2))
                                qlo = vec(mapslices(x -> quantile(x, mcmc_quantiles[1] / 100), dens_mat; dims=2))
                                qhi = vec(mapslices(x -> quantile(x, mcmc_quantiles[end] / 100), dens_mat; dims=2))
                                plot!(p, xgrid, mean_dens; color=style.color_secondary, label="flow KDE")
                                plot!(p, xgrid, qlo; color=style.color_secondary, alpha=mcmc_quantiles_alpha, linestyle=:dash, label="")
                                plot!(p, xgrid, qhi; color=style.color_secondary, alpha=mcmc_quantiles_alpha, linestyle=:dash, label="")
                                y_max = maximum(vcat(mean_dens, qhi))
                                ylims!(p, (0.0, y_max * 1.05))
                            end
                        else
                            @info "Flow marginal plotted via sampling." re=re level=lvl samples=flow_samples
                            samples = if dist isa Distributions.UnivariateDistribution
                                vec(rand(dist, flow_samples))
                            else
                                samp = rand(dist, flow_samples)
                                vec(samp[ci, :])
                            end
                            if length(samples) < 2
                                @warn "Skipping RE flow plot (insufficient samples)." re=re level=lvl
                                continue
                            end
                            if flow_plot == :hist
                                histogram!(p, samples; bins=flow_bins, normalize=:probability,
                                           color=style.color_secondary, label="flow")
                                ys = p.series_list[end][:y]
                                y_max = maximum(ys)
                                ylims!(p, (0.0, y_max * 1.05))
                            else
                                xk, yk = _kde_xy(samples; bandwidth=flow_bandwidth)
                                plot!(p, xk, yk; color=style.color_secondary, label="flow KDE")
                                y_max = maximum(yk)
                                ylims!(p, (0.0, y_max * 1.05))
                            end
                        end
                        vline!(p, [val[1]]; color=style.color_primary, label=ebe_label)
                        push!(plots, p)
                        continue
                    end
                    @warn "Skipping RE distribution plot (missing mean/cov)." re=re level=lvl
                    continue
                end
                title = show_comp ? string(re, " | ", dm.config.primary_id, "=", lvl, " | ", comp_name) :
                        string(re, " | ", dm.config.primary_id, "=", lvl)
                p = create_styled_plot(title=title, xlabel="Random Effect", ylabel="Probability Density", style=style, kwargs_subplot...)
                ebe_label = is_mcmc ? "posterior mean" : "EBE"
                if dist_use isa DiscreteDistribution
                    grid = _density_grid_discrete(dist_use, 0.995)
                    grid === nothing && error("Unable to build PMF grid for $(re).")
                    bar!(p, grid.vals, grid.probs; color=style.color_secondary, label="PMF")
                    vline!(p, [val[1]]; color=style.color_primary, label=ebe_label)
                    xlim = (minimum(grid.vals), maximum(grid.vals))
                    xlims = xlims === nothing ? xlim : (min(xlims[1], xlim[1]), max(xlims[2], xlim[2]))
                    ylims = ylims === nothing ? (minimum(grid.probs), maximum(grid.probs)) :
                            (min(ylims[1], minimum(grid.probs)), max(ylims[2], maximum(grid.probs)))
                else
                    grid = _density_grid_continuous([dist_use], 0.99, 200)
                    grid === nothing && error("Unable to build PDF grid for $(re).")
                    pdf_vals = vec(grid.z[:, 1])
                    plot!(p, grid.y, pdf_vals; color=style.color_secondary, label="PDF")
                    vline!(p, [val[1]]; color=style.color_primary, label=ebe_label)
                    xlim = (minimum(grid.y), maximum(grid.y))
                    xlims = xlims === nothing ? xlim : (min(xlims[1], xlim[1]), max(xlims[2], xlim[2]))
                    ylims = ylims === nothing ? (minimum(pdf_vals), maximum(pdf_vals)) :
                            (min(ylims[1], minimum(pdf_vals)), max(ylims[2], maximum(pdf_vals)))
                end
                push!(plots, p)
            end
        end
    end

    if shared_x_axis || shared_y_axis
        xlim_use = shared_x_axis && xlims !== nothing ? _pad_limits(xlims[1], xlims[2]) : nothing
        ylim_use = shared_y_axis && ylims !== nothing ? _pad_limits(ylims[1], ylims[2]) : nothing
        _apply_shared_axes!(plots, xlim_use, ylim_use)
    end

    if isempty(plots)
        @warn "No random-effect distributions to plot."
        p = create_styled_plot(title="No random-effect distributions to plot.", style=style, kwargs_subplot...)
        return _save_plot!(p, save_path)
    end
    p = combine_plots(plots; ncols=ncols, kwargs_layout...)
    return _save_plot!(p, save_path)
end

"""
    plot_random_effect_pit(res::FitResult; dm, re_names, levels, individuals_idx,
                           show_hist, show_kde, show_qq, shared_x_axis, shared_y_axis,
                           ncols, style, kde_bandwidth, mcmc_draws, mcmc_warmup,
                           mcmc_quantiles, mcmc_quantiles_alpha, flow_samples, rng,
                           save_path, kwargs_subplot, kwargs_layout) -> Plots.Plot

Plot the probability integral transform (PIT) of empirical-Bayes estimates under their
fitted prior distributions, providing a calibration check for the random-effects model.

# Keyword Arguments
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
- `re_names`: random-effect names to include, or `nothing` for all.
- `levels`, `individuals_idx`: grouping level or individual filters.
- `show_hist::Bool = true`: show a PIT histogram.
- `show_kde::Bool = false`: overlay a KDE curve.
- `show_qq::Bool = true`: add a Uniform QQ reference line.
- `shared_x_axis::Bool = true`, `shared_y_axis::Bool = true`: share axes.
- `ncols::Int = 3`: number of subplot columns.
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `kde_bandwidth::Union{Nothing, Float64} = nothing`: KDE bandwidth.
- `mcmc_draws`, `mcmc_warmup`, `mcmc_quantiles`, `mcmc_quantiles_alpha`: MCMC settings.
- `flow_samples::Int = 500`: samples for normalizing-flow distributions.
- `rng::AbstractRNG = Random.default_rng()`: random-number generator.
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
- `kwargs_subplot`, `kwargs_layout`: extra keyword arguments.
"""
function plot_random_effect_pit(res::FitResult;
                                dm::Union{Nothing, DataModel}=nothing,
                                re_names=nothing,
                                levels=nothing,
                                individuals_idx=nothing,
                                x_covariate::Union{Nothing, Symbol}=nothing,
                                show_hist::Bool=true,
                                show_kde::Bool=false,
                                show_qq::Bool=true,
                                shared_x_axis::Bool=true,
                                shared_y_axis::Bool=true,
                                ncols::Int=DEFAULT_PLOT_COLS,
                                style::PlotStyle=PlotStyle(),
                                kde_bandwidth::Union{Nothing, Float64}=nothing,
                                mcmc_draws::Int=1000,
                                mcmc_warmup::Union{Nothing, Int}=nothing,
                                mcmc_quantiles::Vector{<:Real}=[5, 95],
                                mcmc_quantiles_alpha::Float64=0.8,
                                flow_samples::Int=500,
                                rng::AbstractRNG=Random.default_rng(),
                                save_path::Union{Nothing, String}=nothing,
                                plot_path::Union{Nothing, String}=nothing,
                                kwargs_subplot=NamedTuple(),
                                kwargs_layout=NamedTuple())
    dm = _get_dm(res, dm)
    save_path = _resolve_plot_path(save_path, plot_path)
    _require_re_supported(res)
    re_list = _resolve_re_names(dm, re_names)
    is_mcmc = res.result isa MCMCResult
    x_covariate === nothing || throw(ArgumentError("`x_covariate` is not supported by `plot_random_effect_pit`. Use `plot_random_effects_scatter` or `plot_random_effect_standardized_scatter` instead."))

    if (show_hist + show_kde + show_qq) > 1
        @warn "plot_random_effect_pit expects one plot type at a time; defaulting to histogram." show_hist=show_hist show_kde=show_kde show_qq=show_qq
        show_hist = true
        show_kde = false
        show_qq = false
    end

    if is_mcmc
        if mcmc_warmup !== nothing
            conv = res.diagnostics.convergence
            conv = merge(conv, (n_adapt=mcmc_warmup,))
            res = FitResult(res.method, res.result, res.summary,
                            FitDiagnostics(res.diagnostics.timing, res.diagnostics.optimizer, conv, res.diagnostics.notes),
                            res.data_model, res.fit_args, res.fit_kwargs)
        end
        mcmc_quantiles = sort(Float64.(collect(mcmc_quantiles)))
        (length(mcmc_quantiles) >= 2 && all(0 .<= mcmc_quantiles .<= 100)) || error("mcmc_quantiles must be in [0,100] with length >= 2.")
    end

    plots = Vector{Any}()
    xlims = nothing
    ylims = nothing

    θ_base = res.result isa MCMCResult ? _mcmc_fixed_means(res, dm)[1] : get_params(res; scale=:untransformed)
    dists_builder = get_create_random_effect_distribution(dm.model.random.random)
    model_funs = get_model_funs(dm.model)
    helpers = get_helper_funs(dm.model)

    for re in re_list
        level_to_ind = _level_to_individual(dm, re)
        lvls = _resolve_levels(dm, re, levels, individuals_idx)
        ebe_map, value_cols = is_mcmc ? _ebe_by_level_mcmc(dm, res, re, mcmc_draws, rng) : _ebe_by_level(dm, res, re)
        show_comp = length(value_cols) > 1
        lvls_use = [lvl for lvl in lvls if haskey(ebe_map, lvl)]
        isempty(lvls_use) && (lvls_use = collect(keys(ebe_map)))

        for (ci, comp_name) in enumerate(value_cols)
            if is_mcmc
                constants_re = _fit_constants_re(res)
                θ_draws, η_draws, _ = _mcmc_drawn_params(res, dm, constants_re, NamedTuple(), mcmc_draws, rng)
                pits_by_draw = Vector{Vector{Float64}}()
                for d in eachindex(θ_draws)
                    pits_d = Float64[]
                    θ = θ_draws[d]
                    for lvl in lvls_use
                        ind_idx = level_to_ind[lvl]
                        const_cov = dm.individuals[ind_idx].const_cov
                        dist_d = getproperty(dists_builder(θ, const_cov, model_funs, helpers), re)
                        dist_use = dist_d isa Distributions.MultivariateDistribution ? _marginal_normal(dist_d, ci) : dist_d
                        if dist_use === nothing
                            if dist_d isa NormalizingPlanarFlow
                                @info "Flow PIT via empirical CDF (sampling)." re=re level=lvl samples=flow_samples
                                v = getproperty(η_draws[d][ind_idx], re)
                                v_use = v isa Number ? Float64(v) : Float64(v[ci])
                                samp = if dist_d isa Distributions.UnivariateDistribution
                                    vec(rand(dist_d, flow_samples))
                                else
                                    vals = rand(dist_d, flow_samples)
                                    vec(vals[ci, :])
                                end
                                length(samp) == 0 && continue
                                pit = count(<=(v_use), samp) / length(samp)
                                push!(pits_d, pit)
                            end
                            continue
                        end
                        v = getproperty(η_draws[d][ind_idx], re)
                        v_use = v isa Number ? Float64(v) : Float64(v[ci])
                        pit = _pit_value(dist_use, v_use)
                        pit === nothing && continue
                        push!(pits_d, pit)
                    end
                    isempty(pits_d) || push!(pits_by_draw, pits_d)
                end
                if isempty(pits_by_draw)
                    @warn "PIT skipped due to missing cdf." re=re component=comp_name
                    continue
                end

                title = show_comp ? string(re, " | ", comp_name) : string(re)
                if show_hist
                    p_hist = create_styled_plot(title=title * " | PIT hist", xlabel="PIT", ylabel="Probability",
                                                style=style, kwargs_subplot...)
                    nbins = 20
                    edges = range(0.0, 1.0; length=nbins + 1)
                    centers = (edges[1:end-1] .+ edges[2:end]) ./ 2
                    counts_list = Vector{Vector{Float64}}()
                    for pits_d in pits_by_draw
                        length(pits_d) < 2 && continue
                        counts = zeros(Float64, nbins)
                        for p in pits_d
                            idx = p == 1.0 ? nbins : clamp(floor(Int, p * nbins) + 1, 1, nbins)
                            counts[idx] += 1
                        end
                        s = sum(counts)
                        s == 0 && continue
                        push!(counts_list, counts ./ s)
                    end
                    if isempty(counts_list)
                        @warn "Skipping PIT histogram (insufficient PIT values)." re=re component=comp_name
                    else
                        counts_mat = reduce(hcat, counts_list)
                        mean_counts = vec(mean(counts_mat; dims=2))
                        qlo = vec(mapslices(x -> quantile(x, mcmc_quantiles[1] / 100), counts_mat; dims=2))
                        qhi = vec(mapslices(x -> quantile(x, mcmc_quantiles[end] / 100), counts_mat; dims=2))
                        bar!(p_hist, centers, mean_counts; bar_width=edges[2] - edges[1],
                             color=style.color_secondary, label="PIT")
                        plot!(p_hist, centers, qlo; color=style.color_secondary, alpha=mcmc_quantiles_alpha, linestyle=:dash, label="")
                        plot!(p_hist, centers, qhi; color=style.color_secondary, alpha=mcmc_quantiles_alpha, linestyle=:dash, label="")
                        y_max = maximum(vcat(mean_counts, qhi))
                        ylims!(p_hist, (0.0, y_max * 1.05))
                    end
                    push!(plots, p_hist)
                end
                if show_kde
                    p_kde = create_styled_plot(title=title * " | PIT KDE", xlabel="PIT", ylabel="Density",
                                               style=style, kwargs_subplot...)
                    xgrid = range(0.0, 1.0; length=200)
                    dens_list = Vector{Vector{Float64}}()
                    for pits_d in pits_by_draw
                        length(pits_d) < 2 && continue
                        xk, yk = _kde_xy(pits_d; bandwidth=kde_bandwidth)
                        ygrid = _interp_linear(xk, yk, xgrid)
                        push!(dens_list, ygrid)
                    end
                    if isempty(dens_list)
                        @warn "Skipping PIT KDE (insufficient PIT values)." re=re component=comp_name
                    else
                        dens_mat = reduce(hcat, dens_list)
                        mean_dens = vec(mean(dens_mat; dims=2))
                        qlo = vec(mapslices(x -> quantile(x, mcmc_quantiles[1] / 100), dens_mat; dims=2))
                        qhi = vec(mapslices(x -> quantile(x, mcmc_quantiles[end] / 100), dens_mat; dims=2))
                        plot!(p_kde, xgrid, mean_dens; color=style.color_secondary, label="KDE")
                        plot!(p_kde, xgrid, qlo; color=style.color_secondary, alpha=mcmc_quantiles_alpha, linestyle=:dash, label="")
                        plot!(p_kde, xgrid, qhi; color=style.color_secondary, alpha=mcmc_quantiles_alpha, linestyle=:dash, label="")
                        y_max = maximum(vcat(mean_dens, qhi))
                        ylims!(p_kde, (0.0, y_max * 1.05))
                    end
                    push!(plots, p_kde)
                end
                if show_qq
                    p_qq = create_styled_plot(title=title * " | PIT QQ", xlabel="Theoretical Uniform Quantile", ylabel="Empirical PIT Quantile",
                                              style=style, kwargs_subplot...)
                    ugrid = range(0.0, 1.0; length=100)
                    q_list = Vector{Vector{Float64}}()
                    for pits_d in pits_by_draw
                        length(pits_d) < 2 && continue
                        qvals = [quantile(pits_d, u) for u in ugrid]
                        push!(q_list, qvals)
                    end
                    if isempty(q_list)
                        @warn "Skipping PIT QQ (insufficient PIT values)." re=re component=comp_name
                    else
                        qmat = reduce(hcat, q_list)
                        mean_q = vec(mean(qmat; dims=2))
                        qlo = vec(mapslices(x -> quantile(x, mcmc_quantiles[1] / 100), qmat; dims=2))
                        qhi = vec(mapslices(x -> quantile(x, mcmc_quantiles[end] / 100), qmat; dims=2))
                        plot!(p_qq, ugrid, mean_q; color=style.color_secondary, label="QQ")
                        plot!(p_qq, ugrid, qlo; color=style.color_secondary, alpha=mcmc_quantiles_alpha, linestyle=:dash, label="")
                        plot!(p_qq, ugrid, qhi; color=style.color_secondary, alpha=mcmc_quantiles_alpha, linestyle=:dash, label="")
                        plot!(p_qq, ugrid, ugrid; color=style.color_dark, linestyle=:dash, label="Uniform")
                    end
                    push!(plots, p_qq)
                end
            else
                pits = Float64[]
                for lvl in lvls
                    haskey(ebe_map, lvl) || continue
                    ind_idx = level_to_ind[lvl]
                    const_cov = dm.individuals[ind_idx].const_cov
                    dist = getproperty(dists_builder(θ_base, const_cov, model_funs, helpers), re)
                    dist_use = dist isa Distributions.MultivariateDistribution ? _marginal_normal(dist, ci) : dist
                    if dist_use === nothing
                        if dist isa NormalizingPlanarFlow
                            @info "Flow PIT via empirical CDF (sampling)." re=re level=lvl samples=flow_samples
                            v_use = Float64(ebe_map[lvl][ci])
                            samp = if dist isa Distributions.UnivariateDistribution
                                vec(rand(dist, flow_samples))
                            else
                                vals = rand(dist, flow_samples)
                                vec(vals[ci, :])
                            end
                            length(samp) == 0 && continue
                            pit = count(<=(v_use), samp) / length(samp)
                            push!(pits, pit)
                        end
                        continue
                    end
                    pit = _pit_value(dist_use, Float64(ebe_map[lvl][ci]))
                    pit === nothing && continue
                    push!(pits, pit)
                end

                if isempty(pits)
                    @warn "PIT skipped due to missing cdf." re=re component=comp_name
                    continue
                end

                title = show_comp ? string(re, " | ", comp_name) : string(re)
                if show_hist
                    p_hist = create_styled_plot(title=title * " | PIT hist", xlabel="PIT", ylabel="Probability",
                                                style=style, kwargs_subplot...)
                    if length(pits) < 2
                        @warn "Skipping PIT histogram (insufficient PIT values)." re=re component=comp_name
                    else
                        histogram!(p_hist, pits; bins=20, normalize=:probability,
                                   color=style.color_secondary, fillcolor=style.color_secondary,
                                   linecolor=style.color_secondary, label="PIT")
                        ys = p_hist.series_list[end][:y]
                        y_max = maximum(ys)
                        ylims!(p_hist, (0.0, y_max * 1.05))
                    end
                    push!(plots, p_hist)
                end
                if show_kde
                    p_kde = create_styled_plot(title=title * " | PIT KDE", xlabel="PIT", ylabel="Density",
                                               style=style, kwargs_subplot...)
                    if length(pits) < 2
                        @warn "Skipping PIT KDE (insufficient PIT values)." re=re component=comp_name
                    else
                        xk, yk = _kde_xy(pits; bandwidth=kde_bandwidth)
                        plot!(p_kde, xk, yk; color=style.color_secondary, label="KDE")
                    end
                    push!(plots, p_kde)
                end
                if show_qq
                    p_qq = create_styled_plot(title=title * " | PIT QQ", xlabel="Theoretical Uniform Quantile", ylabel="Empirical PIT Quantile",
                                              style=style, kwargs_subplot...)
                    if length(pits) < 2
                        @warn "Skipping PIT QQ (insufficient PIT values)." re=re component=comp_name
                    else
                        q = sort(pits)
                        u = range(0, 1; length=length(q))
                        scatter!(p_qq, u, q; color=style.color_secondary, label="QQ")
                        plot!(p_qq, u, u; color=style.color_dark, linestyle=:dash, label="Uniform")
                    end
                    push!(plots, p_qq)
                end
            end
            xlims = xlims === nothing ? (0.0, 1.0) : xlims
        end
    end

    if shared_x_axis || shared_y_axis
        xlim_use = shared_x_axis ? (0.0, 1.0) : nothing
        ylim_use = shared_y_axis && ylims !== nothing ? _pad_limits(ylims[1], ylims[2]) : nothing
        _apply_shared_axes!(plots, xlim_use, ylim_use)
    end
    if isempty(plots)
        @warn "No PIT plots to display."
        p = create_styled_plot(title="No PIT plots to display.", style=style, kwargs_subplot...)
        return _save_plot!(p, save_path)
    end
    p = combine_plots(plots; ncols=ncols, kwargs_layout...)
    return _save_plot!(p, save_path)
end

"""
    plot_random_effect_standardized(res::FitResult; dm, re_names, levels,
                                    individuals_idx, show_hist, show_kde, kde_bandwidth,
                                    mcmc_draws, flow_samples, ncols, style, save_path,
                                    kwargs_subplot, kwargs_layout) -> Plots.Plot

Plot standardised (z-score) empirical-Bayes estimates for each random effect as a
histogram and/or KDE, with a standard Normal reference. Values far from zero indicate
outliers or misspecification.

# Keyword Arguments
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
- `re_names`: random-effect names to include, or `nothing` for all.
- `levels`, `individuals_idx`: grouping level or individual filters.
- `show_hist::Bool = true`: show a histogram.
- `show_kde::Bool = false`: overlay a KDE curve.
- `kde_bandwidth::Union{Nothing, Float64} = nothing`: KDE bandwidth.
- `mcmc_draws::Int = 1000`: MCMC draws for posterior mean EBE.
- `flow_samples::Int = 500`: samples for normalizing-flow distributions.
- `ncols::Int = 3`: number of subplot columns.
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
- `kwargs_subplot`, `kwargs_layout`: extra keyword arguments.
"""
function plot_random_effect_standardized(res::FitResult;
                                         dm::Union{Nothing, DataModel}=nothing,
                                         re_names=nothing,
                                         levels=nothing,
                                         individuals_idx=nothing,
                                         show_hist::Bool=true,
                                         show_kde::Bool=false,
                                         kde_bandwidth::Union{Nothing, Float64}=nothing,
                                         mcmc_draws::Int=1000,
                                         flow_samples::Int=500,
                                         ncols::Int=DEFAULT_PLOT_COLS,
                                         style::PlotStyle=PlotStyle(),
                                         save_path::Union{Nothing, String}=nothing,
                                         plot_path::Union{Nothing, String}=nothing,
                                         kwargs_subplot=NamedTuple(),
                                         kwargs_layout=NamedTuple())
    dm = _get_dm(res, dm)
    save_path = _resolve_plot_path(save_path, plot_path)
    _require_re_supported(res)
    re_list = _resolve_re_names(dm, re_names)
    θ_base = res.result isa MCMCResult ? _mcmc_fixed_means(res, dm)[1] : get_params(res; scale=:untransformed)
    dists_builder = get_create_random_effect_distribution(dm.model.random.random)
    model_funs = get_model_funs(dm.model)
    helpers = get_helper_funs(dm.model)

    plots = Vector{Any}()
    for re in re_list
        level_to_ind = _level_to_individual(dm, re)
        lvls = _resolve_levels(dm, re, levels, individuals_idx)
        ebe_map, value_cols = res.result isa MCMCResult ? _ebe_by_level_mcmc(dm, res, re, mcmc_draws, Random.default_rng()) : _ebe_by_level(dm, res, re)
        show_comp = length(value_cols) > 1
        lvls_use = [lvl for lvl in lvls if haskey(ebe_map, lvl)]
        isempty(lvls_use) && (lvls_use = collect(keys(ebe_map)))
        for (ci, comp_name) in enumerate(value_cols)
            zvals = Float64[]
            for lvl in lvls_use
                haskey(ebe_map, lvl) || continue
                ind_idx = level_to_ind[lvl]
                const_cov = dm.individuals[ind_idx].const_cov
                dist = getproperty(dists_builder(θ_base, const_cov, model_funs, helpers), re)
                z = _standardize_re(dist, Float64.(ebe_map[lvl]); flow_samples=flow_samples)
                z === nothing && continue
                push!(zvals, Float64(z[ci]))
            end
            title = show_comp ? string(re, " | ", comp_name) : string(re)
            y_label = show_hist && show_kde ? "Probability / Density" : (show_kde ? "Density" : "Probability")
            p = create_styled_plot(title=title, xlabel="Standardized EBE", ylabel=y_label, style=style, kwargs_subplot...)
            if isempty(zvals)
                @warn "No standardized values to plot." re=re component=comp_name
                continue
            end
            hist_max = nothing
            kde_max = nothing
            if show_hist
                histogram!(p, zvals; bins=20, normalize=:probability, color=style.color_secondary, label="z")
                ys = p.series_list[end][:y]
                hist_max = maximum(ys)
            end
            if show_kde
                xk, yk = _kde_xy(zvals; bandwidth=kde_bandwidth)
                plot!(p, xk, yk; color=style.color_secondary, label="KDE")
                kde_max = maximum(yk)
            end
            if hist_max !== nothing || kde_max !== nothing
                y_max = hist_max === nothing ? kde_max : (kde_max === nothing ? hist_max : max(hist_max, kde_max))
                ylims!(p, (0.0, y_max * 1.05))
            end
            push!(plots, p)
        end
    end
    if isempty(plots)
        @warn "No standardized random effects to plot."
        p = create_styled_plot(title="No standardized random effects to plot.", style=style, kwargs_subplot...)
        return _save_plot!(p, save_path)
    end
    p = combine_plots(plots; ncols=ncols, kwargs_layout...)
    return _save_plot!(p, save_path)
end

"""
    plot_random_effect_standardized_scatter(res::FitResult; dm, re_names, levels,
                                            individuals_idx, x_covariate, mcmc_draws,
                                            flow_samples, ncols, style, save_path,
                                            kwargs_subplot, kwargs_layout) -> Plots.Plot

Scatter plot of standardised (z-score) empirical-Bayes estimates against a covariate
or group level index. Useful for detecting systematic patterns in the residual structure
of the random-effects model.

# Keyword Arguments
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
- `re_names`: random-effect names to include, or `nothing` for all.
- `levels`, `individuals_idx`: grouping level or individual filters.
- `x_covariate::Union{Nothing, Symbol} = nothing`: constant covariate for the x-axis.
- `mcmc_draws::Int = 1000`: MCMC draws for posterior mean EBE.
- `flow_samples::Int = 500`: samples for normalizing-flow distributions.
- `ncols::Int = 3`: number of subplot columns.
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
- `kwargs_subplot`, `kwargs_layout`: extra keyword arguments.
"""
function plot_random_effect_standardized_scatter(res::FitResult;
                                                 dm::Union{Nothing, DataModel}=nothing,
                                                 re_names=nothing,
                                                 levels=nothing,
                                                 individuals_idx=nothing,
                                                 x_covariate::Union{Nothing, Symbol}=nothing,
                                                 mcmc_draws::Int=1000,
                                                 flow_samples::Int=500,
                                                 ncols::Int=DEFAULT_PLOT_COLS,
                                                 style::PlotStyle=PlotStyle(),
                                                 save_path::Union{Nothing, String}=nothing,
                                                 plot_path::Union{Nothing, String}=nothing,
                                                 kwargs_subplot=NamedTuple(),
                                                 kwargs_layout=NamedTuple())
    dm = _get_dm(res, dm)
    save_path = _resolve_plot_path(save_path, plot_path)
    _require_re_supported(res)
    re_list = _resolve_re_names(dm, re_names)
    if x_covariate !== nothing
        cov = dm.model.covariates.covariates
        x_covariate in cov.constants || error("x_covariate must be a constant covariate.")
    end

    θ_base = res.result isa MCMCResult ? _mcmc_fixed_means(res, dm)[1] : get_params(res; scale=:untransformed)
    dists_builder = get_create_random_effect_distribution(dm.model.random.random)
    model_funs = get_model_funs(dm.model)
    helpers = get_helper_funs(dm.model)

    plots = Vector{Any}()
    xlims_val = nothing
    for re in re_list
        level_to_ind = _level_to_individual(dm, re)
        lvls = _resolve_levels(dm, re, levels, individuals_idx)
        ebe_map, value_cols = res.result isa MCMCResult ? _ebe_by_level_mcmc(dm, res, re, mcmc_draws, Random.default_rng()) : _ebe_by_level(dm, res, re)
        show_comp = length(value_cols) > 1
        lvls_use = [lvl for lvl in lvls if haskey(ebe_map, lvl)]
        isempty(lvls_use) && (lvls_use = collect(keys(ebe_map)))

        for (ci, comp_name) in enumerate(value_cols)
            xs = Float64[]
            ys = Float64[]
            for (k, lvl) in enumerate(lvls_use)
                haskey(ebe_map, lvl) || continue
                ind_idx = level_to_ind[lvl]
                const_cov = dm.individuals[ind_idx].const_cov
                dist = getproperty(dists_builder(θ_base, const_cov, model_funs, helpers), re)
                z = _standardize_re(dist, Float64.(ebe_map[lvl]); flow_samples=flow_samples)
                z === nothing && continue
                xv = if x_covariate === nothing
                    lvl isa Number ? Float64(lvl) : Float64(k)
                else
                    xv_raw = getfield(const_cov, x_covariate)
                    ismissing(xv_raw) ? nothing : Float64(xv_raw)
                end
                xv === nothing && continue
                yv = Float64(z[ci])
                (isfinite(xv) && isfinite(yv)) || continue
                push!(xs, xv)
                push!(ys, yv)
            end

            title = show_comp ? string(re, " | ", comp_name) : string(re)
            if x_covariate === nothing
                uses_level = any(lvl -> lvl isa Number, lvls_use)
                xlabel = uses_level ? "Level" : "Index"
            else
                xlabel = _axis_label(x_covariate)
            end
            p = create_styled_plot(title=title, xlabel=xlabel, ylabel="Standardized EBE (z-score)", style=style, kwargs_subplot...)
            create_styled_scatter!(p, xs, ys; label="", color=style.color_secondary, style=style)
            push!(plots, p)
            if !isempty(xs)
                xlims_val = xlims_val === nothing ? (minimum(xs), maximum(xs)) :
                            (min(xlims_val[1], minimum(xs)), max(xlims_val[2], maximum(xs)))
            end
        end
    end

    if isempty(plots)
        @warn "No standardized random effects to plot."
        p = create_styled_plot(title="No standardized random effects to plot.", style=style, kwargs_subplot...)
        return _save_plot!(p, save_path)
    end
    if xlims_val !== nothing
        _apply_shared_axes!(plots, _pad_limits(xlims_val[1], xlims_val[2]), nothing)
    end
    p = combine_plots(plots; ncols=ncols, kwargs_layout...)
    return _save_plot!(p, save_path)
end
