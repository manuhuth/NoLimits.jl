export plot_observation_distributions

using Distributions
using Plots
using Random

function _resolve_individuals(dm::DataModel, individuals_idx)
    n = length(dm.individuals)
    if individuals_idx === nothing
        return [1]
    end
    ids = individuals_idx isa AbstractVector ? collect(individuals_idx) : [individuals_idx]
    if all(x -> x isa Integer && 1 <= x <= n, ids)
        return Int.(ids)
    end
    out = Int[]
    for id in ids
        haskey(dm.id_index, id) || error("Unknown individual id $(id).")
        push!(out, dm.id_index[id])
    end
    return out
end

function _resolve_obs_rows(obs_rows, obs_rows_all)
    if obs_rows === nothing
        return collect(1:length(obs_rows_all))
    end
    idxs = obs_rows isa AbstractVector ? collect(obs_rows) : [obs_rows]
    for idx in idxs
        1 <= idx <= length(obs_rows_all) || error("obs_rows index $(idx) out of bounds.")
    end
    return idxs
end

function _resolve_observables(dm::DataModel, observables)
    obs = get_formulas_meta(dm.model.formulas.formulas).obs_names
    if observables === nothing
        length(obs) > 1 && @warn "Multiple observables found; using the first." observable=obs[1]
        return [obs[1]]
    end
    obs_list = observables isa AbstractVector ? collect(observables) : [observables]
    for o in obs_list
        o in obs || error("Observable $(o) not found. Available: $(obs).")
    end
    return obs_list
end

function _mean_pmf_support(dists::Vector{Distribution}, coverage::Float64)
    vals_all = Int[]
    for d in dists
        grid = _density_grid_discrete(d, coverage)
        grid === nothing && continue
        append!(vals_all, grid.vals)
    end
    isempty(vals_all) && return Int[]
    return sort(unique(vals_all))
end

"""
    plot_observation_distributions(res::FitResult; dm, individuals_idx, obs_rows,
                                   observables, x_axis_feature, shared_x_axis,
                                   shared_y_axis, ncols, style, cache, cache_obs_dists,
                                   constants_re, mcmc_quantiles, mcmc_quantiles_alpha,
                                   mcmc_draws, mcmc_warmup, rng, save_path,
                                   kwargs_subplot, kwargs_layout) -> Plots.Plot

Plot the predictive observation distributions at each time point as density or PMF
curves overlaid on the observed data, providing a detailed look at model calibration.

# Keyword Arguments
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
- `individuals_idx`: individuals to include (default: first individual only).
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
- `rng::AbstractRNG = Random.default_rng()`: random-number generator.
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
- `kwargs_subplot`, `kwargs_layout`: extra keyword arguments.
"""
function plot_observation_distributions(res::FitResult;
                                        dm::Union{Nothing, DataModel}=nothing,
                                        individuals_idx=nothing,
                                        obs_rows=nothing,
                                        observables=nothing,
                                        x_axis_feature::Union{Nothing, Symbol}=nothing,
                                        shared_x_axis::Bool=true,
                                        shared_y_axis::Bool=true,
                                        ncols::Int=DEFAULT_PLOT_COLS,
                                        style::PlotStyle=PlotStyle(),
                                        cache::Union{Nothing, PlotCache}=nothing,
                                        cache_obs_dists::Bool=false,
                                        constants_re::NamedTuple=NamedTuple(),
                                        mcmc_quantiles::Vector{<:Real}=[5, 95],
                                        mcmc_quantiles_alpha::Float64=0.8,
                                        mcmc_draws::Int=1000,
                                        mcmc_warmup::Union{Nothing, Int}=nothing,
                                        rng::AbstractRNG=Random.default_rng(),
                                        save_path::Union{Nothing, String}=nothing,
                                        plot_path::Union{Nothing, String}=nothing,
                                        kwargs_subplot=NamedTuple(),
                                        kwargs_layout=NamedTuple())
    dm = _get_dm(res, dm)
    save_path = _resolve_plot_path(save_path, plot_path)
    constants_re_use = _res_constants_re(res, constants_re)
    obs_list = _resolve_observables(dm, observables)
    inds = _resolve_individuals(dm, individuals_idx)
    if dm.model.de.de === nothing
        x_axis_feature = _require_varying_covariate(dm, x_axis_feature)
    end

    is_mcmc = res.result isa MCMCResult
    if !is_mcmc && cache === nothing
        cache = build_plot_cache(res; dm=dm, constants_re=constants_re_use, cache_obs_dists=cache_obs_dists, rng=rng)
    end

    plots = Vector{Any}()
    plot_groups = Vector{Tuple{Int, Symbol}}()
    xlims_by_group = Dict{Tuple{Int, Symbol}, Tuple{Float64, Float64}}()
    ylims = nothing

    θ_draws = nothing
    η_draws = nothing
    if is_mcmc
        if mcmc_warmup !== nothing
            conv = res.diagnostics.convergence
            conv = merge(conv, (n_adapt=mcmc_warmup,))
            res = FitResult(res.method, res.result, res.summary,
                            FitDiagnostics(res.diagnostics.timing, res.diagnostics.optimizer, conv, res.diagnostics.notes),
                            res.data_model, res.fit_args, res.fit_kwargs)
        end
        θ_draws, η_draws, _ = _mcmc_drawn_params(res, dm, constants_re_use, NamedTuple(), mcmc_draws, rng)
        mcmc_quantiles = sort(Float64.(collect(mcmc_quantiles)))
        (length(mcmc_quantiles) >= 2 && all(0 .<= mcmc_quantiles .<= 100)) || error("mcmc_quantiles must be in [0,100] with length >= 2.")
    end

    for i in inds
        ind = dm.individuals[i]
        obs_rows_all = dm.row_groups.obs_rows[i]
        obs_idx = _resolve_obs_rows(obs_rows, obs_rows_all)

        for obs_name in obs_list
            for j in obs_idx
                row = obs_rows_all[j]
                y_obs = getfield(ind.series.obs, obs_name)[j]
                tval = dm.df[row, dm.config.time_col]
                p = create_styled_plot(title=string(dm.config.primary_id, ": ", dm.df[row, dm.config.primary_id], ", ",
                                                    dm.config.time_col, ": ", tval),
                                       xlabel=_axis_label(obs_name),
                                       ylabel="Probability Density",
                                       style=style,
                                       kwargs_subplot...)

                if is_mcmc
                    n_draws = length(θ_draws)
                    dists = Vector{Distribution}(undef, n_draws)
                    for d in 1:n_draws
                        θ = θ_draws[d]
                        η_ind = η_draws[d][i]
                        sol_accessors = nothing
                        compiled = nothing
                        if dm.model.de.de !== nothing
                            sol, compiled = _solve_dense_individual(dm, ind, θ, η_ind)
                            sol_accessors = get_de_accessors_builder(dm.model.de.de)(sol, compiled)
                        end
                        vary = _varying_at_plot(dm, ind, j, row)
                        obs = sol_accessors === nothing ?
                              calculate_formulas_obs(dm.model, θ, η_ind, ind.const_cov, vary) :
                              calculate_formulas_obs(dm.model, θ, η_ind, ind.const_cov, vary, sol_accessors)
                        dists[d] = getproperty(obs, obs_name)
                    end

                    if dists[1] isa DiscreteDistribution
                        vals = _mean_pmf_support(dists, 0.995)
                        isempty(vals) && error("Unable to build PMF support for $(obs_name).")
                        probs = zeros(Float64, length(vals))
                        probs_all = zeros(Float64, n_draws, length(vals))
                        for d in 1:n_draws
                            pvec = pdf.(Ref(dists[d]), vals)
                            probs_all[d, :] .= pvec
                            probs .+= pvec
                        end
                        probs ./= n_draws
                        qlo = mapslices(x -> quantile(vec(x), mcmc_quantiles[1] / 100), probs_all; dims=1)
                        qhi = mapslices(x -> quantile(vec(x), mcmc_quantiles[end] / 100), probs_all; dims=1)
                        qlo = vec(qlo); qhi = vec(qhi)
                        bar!(p, vals, probs; color=style.color_secondary, label="posterior mean PMF")
                        plot!(p, vals, qlo; color=style.color_secondary, alpha=mcmc_quantiles_alpha, linestyle=:dash, label="$(mcmc_quantiles[1])%")
                        plot!(p, vals, qhi; color=style.color_secondary, alpha=mcmc_quantiles_alpha, linestyle=:dash, label="$(mcmc_quantiles[end])%")
                        ylabel!(p, "Probability Mass")
                        vline!(p, [y_obs]; color=style.color_primary, label="observed")
                        xlim = (minimum(vals), maximum(vals))
                        if y_obs isa Number
                            xlim = (min(xlim[1], float(y_obs)), max(xlim[2], float(y_obs)))
                        end
                        xlims_by_group[(i, obs_name)] = haskey(xlims_by_group, (i, obs_name)) ?
                            (min(xlims_by_group[(i, obs_name)][1], xlim[1]), max(xlims_by_group[(i, obs_name)][2], xlim[2])) :
                            xlim
                        ylims = ylims === nothing ? (minimum(qlo), maximum(qhi)) :
                                (min(ylims[1], minimum(qlo)), max(ylims[2], maximum(qhi)))
                    else
                        grid = _density_grid_continuous(dists, 0.99, 200)
                        grid === nothing && error("Unable to build PDF grid for $(obs_name).")
                        pdf_mean = vec(mean(grid.z, dims=2))
                        qlo = mapslices(x -> quantile(vec(x), mcmc_quantiles[1] / 100), grid.z; dims=2)
                        qhi = mapslices(x -> quantile(vec(x), mcmc_quantiles[end] / 100), grid.z; dims=2)
                        qlo = vec(qlo); qhi = vec(qhi)
                        plot!(p, grid.y, pdf_mean; color=style.color_secondary, label="posterior mean PDF")
                        plot!(p, grid.y, qlo; color=style.color_secondary, alpha=mcmc_quantiles_alpha, linestyle=:dash, label="$(mcmc_quantiles[1])%")
                        plot!(p, grid.y, qhi; color=style.color_secondary, alpha=mcmc_quantiles_alpha, linestyle=:dash, label="$(mcmc_quantiles[end])%")
                        vline!(p, [y_obs]; color=style.color_primary, label="observed")
                        xlim = (minimum(grid.y), maximum(grid.y))
                        if y_obs isa Number
                            xlim = (min(xlim[1], float(y_obs)), max(xlim[2], float(y_obs)))
                        end
                        xlims_by_group[(i, obs_name)] = haskey(xlims_by_group, (i, obs_name)) ?
                            (min(xlims_by_group[(i, obs_name)][1], xlim[1]), max(xlims_by_group[(i, obs_name)][2], xlim[2])) :
                            xlim
                        ylims = ylims === nothing ? (minimum(qlo), maximum(qhi)) :
                                (min(ylims[1], minimum(qlo)), max(ylims[2], maximum(qhi)))
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
                    dist = if cache_obs_dists && cache.obs_dists !== nothing
                        getproperty(cache.obs_dists[i][j], obs_name)
                    else
                        vary = _varying_at_plot(dm, ind, j, row)
                        obs = sol_accessors === nothing ?
                              calculate_formulas_obs(dm.model, θ, η_ind, ind.const_cov, vary) :
                              calculate_formulas_obs(dm.model, θ, η_ind, ind.const_cov, vary, sol_accessors)
                        getproperty(obs, obs_name)
                    end

                    if dist isa DiscreteDistribution
                        if dist isa Bernoulli
                            vals = [0, 1]
                        else
                            grid = _density_grid_discrete(dist, 0.995)
                            grid === nothing && error("Unable to build PMF grid for $(obs_name).")
                            vals = grid.vals
                        end
                        probs = pdf.(Ref(dist), vals)
                        bar!(p, vals, probs; color=style.color_secondary, label="PMF")
                        ylabel!(p, "Probability Mass")
                        vline!(p, [y_obs]; color=style.color_primary, label="observed")
                        xlim = (minimum(vals), maximum(vals))
                        if y_obs isa Number
                            xlim = (min(xlim[1], float(y_obs)), max(xlim[2], float(y_obs)))
                        end
                        xlims_by_group[(i, obs_name)] = haskey(xlims_by_group, (i, obs_name)) ?
                            (min(xlims_by_group[(i, obs_name)][1], xlim[1]), max(xlims_by_group[(i, obs_name)][2], xlim[2])) :
                            xlim
                        ylims = ylims === nothing ? (minimum(probs), maximum(probs)) :
                                (min(ylims[1], minimum(probs)), max(ylims[2], maximum(probs)))
                    else
                        grid = _density_grid_continuous([dist], 0.99, 200)
                        grid === nothing && error("Unable to build PDF grid for $(obs_name).")
                        pdf_vals = vec(grid.z[:, 1])
                        plot!(p, grid.y, pdf_vals; color=style.color_secondary, label="PDF")
                        vline!(p, [y_obs]; color=style.color_primary, label="observed")
                        xlim = (minimum(grid.y), maximum(grid.y))
                        if y_obs isa Number
                            xlim = (min(xlim[1], float(y_obs)), max(xlim[2], float(y_obs)))
                        end
                        xlims_by_group[(i, obs_name)] = haskey(xlims_by_group, (i, obs_name)) ?
                            (min(xlims_by_group[(i, obs_name)][1], xlim[1]), max(xlims_by_group[(i, obs_name)][2], xlim[2])) :
                            xlim
                        ylims = ylims === nothing ? (minimum(pdf_vals), maximum(pdf_vals)) :
                                (min(ylims[1], minimum(pdf_vals)), max(ylims[2], maximum(pdf_vals)))
                    end
                end

                push!(plots, p)
                push!(plot_groups, (i, obs_name))
            end
        end
    end

    if shared_x_axis
        for (p, key) in zip(plots, plot_groups)
            lims = xlims_by_group[key]
            plot!(p; xlims=_pad_limits(lims[1], lims[2]))
        end
    end
    if shared_y_axis && ylims !== nothing
        for p in plots
            plot!(p; ylims=_pad_limits(ylims[1], ylims[2]))
        end
    end

    p = combine_plots(plots; ncols=ncols, kwargs_layout...)
    return _save_plot!(p, save_path)
end

function plot_observation_distributions(dm::DataModel; kwargs...)
    constants_re = haskey(kwargs, :constants_re) ? getfield(kwargs, :constants_re) : NamedTuple()
    cache = build_plot_cache(dm; constants_re=constants_re, cache_obs_dists=false)
    res = FitResult(MLE(), MLEResult(NamedTuple(), 0.0, 0, NamedTuple(), NamedTuple()),
                    FitSummary(0.0, true, FitParameters(ComponentArray(), ComponentArray()), NamedTuple()),
                    FitDiagnostics((;), (;), (;), (;)), dm, (), NamedTuple())
    return plot_observation_distributions(res; dm=dm, cache=cache, kwargs...)
end
