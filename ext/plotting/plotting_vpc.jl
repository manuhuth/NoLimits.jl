"""
    plot_vpc(res::FitResult; dm, n_simulations, n_sim, percentiles, show_obs_points,
             show_obs_percentiles, n_bins, seed, observables, x_axis_feature, ncols,
             kwargs_plot, save_path, obs_percentiles_mode, bandwidth,
             obs_percentiles_method, constants_re, mcmc_draws, mcmc_warmup, style)
             -> Plots.Plot

Visual Predictive Check (VPC): compares observed percentile bands to simulated
predictive percentile bands stratified by x-axis bins.

# Keyword Arguments
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
- `n_simulations::Int = 100`: number of simulated datasets for the VPC envelopes.
- `percentiles::Vector = [5, 50, 95]`: percentiles to display (in [0, 100]).
- `show_obs_points::Bool = true`: overlay observed data points.
- `show_obs_percentiles::Bool = true`: overlay observed percentile lines.
- `n_bins::Union{Nothing, Int} = nothing`: number of x-axis bins; `nothing` for auto.
- `seed::Int = 12345`: random seed for reproducible simulations.
- `observables`: outcome name(s) to plot, or `nothing` for all.
- `x_axis_feature::Union{Nothing, Symbol} = nothing`: covariate for the x-axis; defaults
  to time.
- `ncols::Int = 3`: number of subplot columns.
- `kwargs_plot`: extra keyword arguments forwarded to the plot.
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
- `obs_percentiles_mode::Symbol = :pooled`: `:pooled` or `:individual` percentile
  computation.
- `bandwidth::Union{Nothing, Float64} = nothing`: smoothing bandwidth for percentile
  curves, or `nothing` for no smoothing.
- `obs_percentiles_method::Symbol = :kernel`: `:kernel` (smooth, default) or `:quantile` (bin-based).
- `constants_re::NamedTuple = NamedTuple()`: random-effect constants.
- `mcmc_draws::Int = 1000`, `mcmc_warmup`: MCMC draw settings.
- `style::PlotStyle = PlotStyle()`: visual style configuration.
"""
function plot_vpc(res::FitResult;
        dm::Union{Nothing, DataModel} = nothing,
        n_simulations::Int = 100,
        n_sim::Union{Nothing, Int} = nothing,
        percentiles::Vector{<:Real} = [5, 50, 95],
        show_obs_points::Bool = true,
        show_obs_percentiles::Bool = true,
        n_bins::Union{Nothing, Int} = nothing,
        seed::Int = 12345,
        observables = nothing,
        x_axis_feature::Union{Nothing, Symbol} = nothing,
        ncols::Int = DEFAULT_PLOT_COLS,
        serialization = nothing,
        kwargs_plot = NamedTuple(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing,
        obs_percentiles_mode::Symbol = :pooled,
        bandwidth::Union{Nothing, Float64} = nothing,
        obs_percentiles_method::Symbol = :kernel,
        constants_re::NamedTuple = NamedTuple(),
        mcmc_draws::Int = 1000,
        mcmc_warmup::Union{Nothing, Int} = nothing,
        style::PlotStyle = PlotStyle(),
        kwargs_subplot = NamedTuple())
    dm = _get_dm(res, dm)
    save_path = _resolve_plot_path(save_path, plot_path)
    if n_sim !== nothing
        n_sim >= 1 || error("n_sim must be >= 1.")
        if n_simulations != 100 && n_simulations != n_sim
            error("Specify either n_simulations or n_sim, not conflicting values for both.")
        end
        n_simulations = n_sim
    end
    n_simulations >= 1 || error("n_simulations must be >= 1.")
    serialization === nothing ||
        throw(ArgumentError("`serialization` is not supported by `plot_vpc`."))
    constants_re_use = _res_constants_re(res, constants_re)
    model = dm.model
    if model.de.de === nothing
        x_axis_feature = _require_varying_covariate(dm, x_axis_feature)
    end

    rng = Random.MersenneTwister(seed)
    obs_names = get_formulas_meta(model.formulas.formulas).obs_names
    observables === nothing && (observables = obs_names)
    percentiles = sort(Float64.(collect(percentiles)))
    (length(percentiles) >= 2 && all(0 .<= percentiles .<= 100)) ||
        error("percentiles must be in [0,100] with length >= 2.")

    is_mcmc = _is_posterior_draw_fit(res)
    if is_mcmc
        @info "VPC uses posterior draws to simulate observations."
        res = _with_posterior_warmup(res, mcmc_warmup)
    end

    plots = Vector{Any}(undef, length(observables))
    for (oi, obs_name) in enumerate(observables)
        x_label = x_axis_feature === nothing ? "Time" : _axis_label(x_axis_feature)
        all_x = Float64[]
        all_y = Float64[]
        all_x_bins = Float64[]
        x_by_ind = Vector{Vector{Float64}}(undef, length(dm.individuals))
        y_by_ind = Vector{Vector{Float64}}(undef, length(dm.individuals))
        for (i, ind) in enumerate(dm.individuals)
            obs_rows = dm.row_groups.obs_rows[i]
            x_all_i, x_i, y_i = _collect_observed_xy(
                ind, dm, obs_rows, obs_name, x_axis_feature)
            append!(all_x_bins, x_all_i)
            append!(all_x, x_i)
            append!(all_y, y_i)
            x_by_ind[i] = x_i
            y_by_ind[i] = y_i
        end
        x_for_bins = isempty(all_x) ? all_x_bins : all_x
        if isempty(x_for_bins)
            @warn "No finite x values found for observable; returning empty VPC subplot." observable=obs_name
            _kw_vpc = merge(
                (xlabel = x_label, ylabel = _axis_label(obs_name)), kwargs_subplot)
            plots[oi] = create_styled_plot(; title = "", style = style, _kw_vpc...)
            continue
        end
        n_bins_eff = _resolve_n_bins(x_for_bins, n_bins)
        edges = _bin_edges_quantile(x_for_bins, n_bins_eff)

        sim_x_all = Float64[]
        sim_y_all = Float64[]

        dist_rep = _representative_dist(dm, obs_name, x_axis_feature)
        is_discrete = dist_rep isa DiscreteDistribution
        is_bern = dist_rep isa Bernoulli

        if is_mcmc
            θ_draws, η_draws, _ = _posterior_drawn_params(
                res, dm, constants_re_use, NamedTuple(), mcmc_draws, rng)
            n_sim = length(θ_draws)
            for s in 1:n_sim
                sim_x, sim_vals = _simulate_obs(
                    dm, θ_draws[s], η_draws[s], obs_name, rng, x_axis_feature)
                xs = reduce(vcat, sim_x)
                ys = reduce(vcat, sim_vals)
                append!(sim_x_all, xs)
                append!(sim_y_all, ys)
            end
        else
            for s in 1:n_simulations
                θ = get_params(res; scale = :untransformed)
                level_vals = _sample_random_effects_levels(dm, θ, constants_re_use, rng)
                η_vec = _eta_vec_from_levels(dm, level_vals)
                sim_x, sim_vals = _simulate_obs(dm, θ, η_vec, obs_name, rng, x_axis_feature)
                xs = reduce(vcat, sim_x)
                ys = reduce(vcat, sim_vals)
                append!(sim_x_all, xs)
                append!(sim_y_all, ys)
            end
        end

        _kw_vpc = merge((xlabel = x_label, ylabel = _axis_label(obs_name)), kwargs_subplot)
        p = create_styled_plot(; title = "", style = style, _kw_vpc...)
        if show_obs_points && !isempty(all_y)
            scatter!(p, all_x, all_y; color = style.color_primary, alpha = 0.3,
                markersize = style.marker_size, markerstrokewidth = style.marker_stroke_width,
                label = "obs")
        end

        if show_obs_percentiles && !is_discrete && !isempty(all_y)
            if obs_percentiles_method == :kernel
                obs_percentiles_mode == :pooled ||
                    error("obs_percentiles_mode=:per_individual is only supported with obs_percentiles_method=:quantile.")
                bw = bandwidth === nothing ? (maximum(all_x) - minimum(all_x)) / 10 :
                     bandwidth
                xgrid = sort(unique(all_x))
                sm = _kernel_quantiles(all_x, all_y, xgrid, bw, percentiles)
                for pctl in percentiles
                    plot!(p, xgrid, sm[pctl]; color = COLOR_ACCENT,
                        linestyle = pctl == median(percentiles) ? :solid : :dot,
                        label = "")
                end
            elseif obs_percentiles_method == :quantile
                bins = _assign_bins(all_x, edges)
                x_centers = [mean(edges[b:(b + 1)]) for b in 1:n_bins_eff]
                obs_q = Dict{Float64, Vector{Float64}}((p => Vector{Float64}(
                                                           undef, n_bins_eff))
                for p in percentiles)
                if obs_percentiles_mode == :pooled
                    for b in 1:n_bins_eff
                        vals = all_y[bins .== b]
                        for pctl in percentiles
                            obs_q[pctl][b] = isempty(vals) ? NaN :
                                             quantile(vals, pctl / 100)
                        end
                    end
                elseif obs_percentiles_mode == :per_individual
                    for b in 1:n_bins_eff
                        per_ind_vals = Dict{Float64, Vector{Float64}}((p => Float64[])
                        for p in percentiles)
                        for (x, y) in zip(x_by_ind, y_by_ind)
                            bins_ind = _assign_bins(x, edges)
                            vals = y[bins_ind .== b]
                            isempty(vals) && continue
                            for pctl in percentiles
                                push!(per_ind_vals[pctl], quantile(vals, pctl / 100))
                            end
                        end
                        for pctl in percentiles
                            obs_q[pctl][b] = isempty(per_ind_vals[pctl]) ? NaN :
                                             mean(per_ind_vals[pctl])
                        end
                    end
                else
                    error("obs_percentiles_mode must be :pooled or :per_individual.")
                end
                for pctl in percentiles
                    x_plot, y_plot = _extend_bin_series(x_centers, obs_q[pctl], edges)
                    lbl = "obs $(pctl)%"
                    plot!(p, x_plot, y_plot; color = COLOR_ACCENT,
                        linestyle = pctl == median(percentiles) ? :solid : :dot,
                        label = lbl)
                end
            else
                error("obs_percentiles_method must be :quantile or :kernel.")
            end
        end

        if !isempty(sim_x_all)
            bins_sim = _assign_bins(sim_x_all, edges)
            x_centers = [mean(edges[b:(b + 1)]) for b in 1:n_bins_eff]
            if is_discrete
                if is_bern
                    p1 = [mean(sim_y_all[bins_sim .== b]) for b in 1:n_bins_eff]
                    x_plot, y_plot = _extend_bin_series(x_centers, p1, edges)
                    scatter!(p, x_plot, y_plot; color = style.color_secondary, marker = :x,
                        markersize = style.marker_size_pmf,
                        markerstrokewidth = style.marker_stroke_width_pmf,
                        label = "sim P(Y=1)")
                else
                    added = false
                    for b in 1:n_bins_eff
                        vals = sim_y_all[bins_sim .== b]
                        isempty(vals) && continue
                        lo = floor(Int, quantile(vals, 0.005))
                        hi = ceil(Int, quantile(vals, 0.995))
                        support = collect(lo:hi)
                        probs = [mean(vals .== v) for v in support]
                        lbl = added ? "" : "sim PMF"
                        scatter!(p, fill(x_centers[b], length(support)), support;
                            marker_z = probs, color = :viridis, marker = :x,
                            markersize = style.marker_size_pmf,
                            markerstrokewidth = style.marker_stroke_width_pmf,
                            label = lbl)
                        added = true
                    end
                end
            else
                if bandwidth !== nothing
                    xgrid = collect(LinRange(minimum(sim_x_all), maximum(sim_x_all), 200))
                    sm_sim = _kernel_quantiles(
                        sim_x_all, sim_y_all, xgrid, bandwidth, percentiles)
                    for pctl in percentiles
                        lbl = "sim $(pctl)%"
                        plot!(p, xgrid, sm_sim[pctl]; color = COLOR_SECONDARY, label = lbl)
                    end
                else
                    sim_q = Dict{Float64, Vector{Float64}}((p => Vector{Float64}(
                                                               undef, n_bins_eff))
                    for p in percentiles)
                    for b in 1:n_bins_eff
                        vals = sim_y_all[bins_sim .== b]
                        for pctl in percentiles
                            sim_q[pctl][b] = isempty(vals) ? NaN :
                                             quantile(vals, pctl / 100)
                        end
                    end
                    for pctl in percentiles
                        x_plot, y_plot = _extend_bin_series(x_centers, sim_q[pctl], edges)
                        lbl = "sim $(pctl)%"
                        plot!(p, x_plot, y_plot; color = COLOR_SECONDARY, label = lbl)
                    end
                end
            end
        end

        plots[oi] = p
        xlims!(p, _pad_limits(minimum(x_for_bins), maximum(x_for_bins)))
    end

    p = combine_plots(plots; ncols = ncols, kwargs_plot...)
    return _save_plot!(p, save_path)
end
