"""
    plot_vpc(res::FitResult; dm, n_simulations, n_sim, percentiles, show_obs_points,
             show_obs_percentiles, n_bins, seed, observables, x_axis_feature, ncols,
             kwargs_subplot, kwargs_plot, save_path, obs_percentiles_mode, bandwidth,
             obs_percentiles_method, constants_re, mcmc_draws, mcmc_warmup, style)
             -> Makie.Figure

Visual Predictive Check (VPC): compares observed percentile bands to simulated
predictive percentile bands stratified by x-axis bins. Vector-valued (multivariate)
outcomes get one panel per component; observed and simulated values are binned and
summarized component-wise. Panels have their own axes; one legend is drawn below them.

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
- `kwargs_subplot`: additional Makie `Axis` attributes forwarded to each subplot.
- `kwargs_plot`: additional Makie `Figure` attributes forwarded to the combined layout.
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
- `obs_percentiles_mode::Symbol = :pooled`: `:pooled` or `:individual` percentile
  computation.
- `bandwidth::Union{Nothing, Float64} = nothing`: smoothing bandwidth for percentile
  curves, or `nothing` for no smoothing.
- `obs_percentiles_method::Symbol = :kernel`: `:kernel` (smooth, default) or `:quantile` (bin-based).
- `constants_re::NamedTuple = NamedTuple()`: random-effect constants.
- `mcmc_draws::Int = 1000`, `mcmc_warmup`: MCMC draw settings.
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `return_panel::Bool = false`: with a single observable, return the built panel (an
  AlgebraOfGraphics panel) instead of a `Figure`, for composing several diagnostics into one figure via `combine_plots`.
"""
function plot_vpc(
        res::FitResult;
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
        return_panel::Bool = false,
        obs_percentiles_mode::Symbol = :pooled,
        bandwidth::Union{Nothing, Float64} = nothing,
        obs_percentiles_method::Symbol = :kernel,
        constants_re::NamedTuple = NamedTuple(),
        mcmc_draws::Int = 1000,
        mcmc_warmup::Union{Nothing, Int} = nothing,
        style::PlotStyle = PlotStyle(),
        kwargs_subplot = NamedTuple()
    )
    dm = _get_dm(res, dm)
    save_path = _resolve_plot_path(save_path, plot_path)
    bandwidth === nothing || _check_positive(bandwidth, "bandwidth")
    _check_positive_int(ncols, "ncols")
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
    model = get_model(dm)
    if get_de(model) === nothing
        x_axis_feature = _require_varying_covariate(dm, x_axis_feature)
    end

    rng = Random.MersenneTwister(seed)
    obs_names = get_formulas_meta(get_formulas(model)).obs_names
    observables === nothing && (observables = obs_names)
    percentiles = sort(Float64.(collect(percentiles)))
    (length(percentiles) >= 2 && all(0 .<= percentiles .<= 100)) ||
        error("percentiles must be in [0,100] with length >= 2.")

    is_mcmc = _is_posterior_draw_fit(res)
    if is_mcmc
        @info "VPC uses posterior draws to simulate observations."
        res = _with_posterior_warmup(res, mcmc_warmup)
    end

    # One panel per (observable, component): vector-valued outcomes are summarized
    # component-wise instead of being dropped.
    obs_list = observables isa Symbol ? [observables] : collect(observables)
    isempty(obs_list) && error("observables selects no outcome. Pass at least one observable name, or omit the keyword to use all.")
    for o in obs_list
        Symbol(o) in obs_names ||
            error("Observable $(o) not found. Available: $(collect(obs_names)).")
    end
    panel_specs = Tuple{Symbol, Union{Nothing, Int}, String, Int}[]
    for o in obs_list
        specs, n_marg = _marginal_panels(dm, o)
        for (m, lbl) in specs
            push!(panel_specs, (o, m, lbl, n_marg))
        end
    end

    plots = Vector{Any}(undef, length(panel_specs))
    for (oi, (obs_name, marginal, obs_label, n_marginals)) in enumerate(panel_specs)
        x_label = x_axis_feature === nothing ? "Time" : _axis_label(x_axis_feature)
        all_x = Float64[]
        all_y = Float64[]
        all_x_bins = Float64[]
        x_by_ind = Vector{Vector{Float64}}(undef, length(get_individuals(dm)))
        y_by_ind = Vector{Vector{Float64}}(undef, length(get_individuals(dm)))
        for (i, ind) in enumerate(get_individuals(dm))
            obs_rows = get_obs_rows(get_row_groups(dm))[i]
            x_all_i, x_i, y_i = _collect_observed_xy(
                ind, dm, obs_rows, obs_name, x_axis_feature, marginal
            )
            append!(all_x_bins, x_all_i)
            append!(all_x, x_i)
            append!(all_y, y_i)
            x_by_ind[i] = x_i
            y_by_ind[i] = y_i
        end
        x_for_bins = isempty(all_x) ? all_x_bins : all_x
        if isempty(x_for_bins)
            @warn "No finite x values found for observable; returning empty VPC subplot." observable = obs_label
            plots[oi] = AoGPanel(
                nothing;
                axis = _aog_axis(
                    style, kwargs_subplot; xlabel = x_label, ylabel = _axis_label(obs_label)
                )
            )
            continue
        end
        n_bins_eff = _resolve_n_bins(x_for_bins, n_bins)
        edges = _bin_edges_quantile(x_for_bins, n_bins_eff)

        sim_x_all = Float64[]
        sim_y_all = Float64[]

        dist_rep = _representative_dist(dm, obs_name, x_axis_feature)
        if marginal !== nothing
            dist_rep = _obs_marginals_or_throw(
                "plot_vpc", obs_name, dist_rep, n_marginals
            )[marginal]
        end
        is_discrete = dist_rep isa DiscreteDistribution
        is_bern = dist_rep isa Bernoulli

        if is_mcmc
            θ_draws, η_draws, _ = _posterior_drawn_params(
                res, dm, constants_re_use, NamedTuple(), mcmc_draws, rng
            )
            n_sim = length(θ_draws)
            for s in 1:n_sim
                sim_x, sim_vals = _simulate_obs(
                    dm, θ_draws[s], η_draws[s], obs_name, rng, x_axis_feature, marginal
                )
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
                sim_x, sim_vals = _simulate_obs(
                    dm, θ, η_vec, obs_name, rng, x_axis_feature, marginal
                )
                xs = reduce(vcat, sim_x)
                ys = reduce(vcat, sim_vals)
                append!(sim_x_all, xs)
                append!(sim_y_all, ys)
            end
        end

        # percentile lines by line style: labeled (legend) and unlabeled kernel obs lines
        lines_df = Dict(:solid => _tidy(), :dot => _tidy())
        kern_df = Dict(:solid => _tidy(), :dot => _tidy())
        obs_df = _tidy()
        bern_df = _tidy()
        pmf = (x = Float64[], y = Float64[], p = Float64[])
        palette = Pair{String, Any}[]
        if show_obs_points && !isempty(all_y)
            _push_xy!(obs_df, "", all_x, all_y, "obs")
            push!(palette, "obs" => Makie.coloralpha(Makie.to_color(style.color_primary), 0.3))
        end
        pct_style(pctl) = pctl == median(percentiles) ? :solid : :dot

        if show_obs_percentiles && !is_discrete && !isempty(all_y)
            if obs_percentiles_method == :kernel
                obs_percentiles_mode == :pooled ||
                    error("obs_percentiles_mode=:per_individual is only supported with obs_percentiles_method=:quantile.")
                bw = bandwidth === nothing ? (maximum(all_x) - minimum(all_x)) / 10 :
                    bandwidth
                xgrid = sort(unique(all_x))
                sm = _kernel_quantiles(all_x, all_y, xgrid, bw, percentiles)
                for pctl in percentiles
                    _push_xy!(kern_df[pct_style(pctl)], "", xgrid, sm[pctl], "obs $(pctl)%")
                end
            elseif obs_percentiles_method == :quantile
                bins = _assign_bins(all_x, edges)
                x_centers = [mean(edges[b:(b + 1)]) for b in 1:n_bins_eff]
                obs_q = Dict{Float64, Vector{Float64}}(
                    (pc => Vector{Float64}(undef, n_bins_eff)) for pc in percentiles
                )
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
                        per_ind_vals = Dict{Float64, Vector{Float64}}(
                            (pc => Float64[]) for pc in percentiles
                        )
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
                    _push_xy!(lines_df[pct_style(pctl)], "", x_plot, y_plot, lbl)
                    push!(palette, lbl => COLOR_ACCENT)
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
                    _push_xy!(bern_df, "", x_plot, y_plot, "sim P(Y=1)")
                    push!(palette, "sim P(Y=1)" => style.color_secondary)
                else
                    for b in 1:n_bins_eff
                        vals = sim_y_all[bins_sim .== b]
                        isempty(vals) && continue
                        lo = floor(Int, quantile(vals, 0.005))
                        hi = ceil(Int, quantile(vals, 0.995))
                        for v in lo:hi
                            push!(pmf.x, x_centers[b])
                            push!(pmf.y, v)
                            push!(pmf.p, mean(vals .== v))
                        end
                    end
                    isempty(pmf.x) || push!(palette, "sim PMF" => :black)
                end
            else
                if bandwidth !== nothing
                    xgrid = collect(LinRange(minimum(sim_x_all), maximum(sim_x_all), 200))
                    sm_sim = _kernel_quantiles(
                        sim_x_all, sim_y_all, xgrid, bandwidth, percentiles
                    )
                    for pctl in percentiles
                        lbl = "sim $(pctl)%"
                        _push_xy!(lines_df[:solid], "", xgrid, sm_sim[pctl], lbl)
                        push!(palette, lbl => COLOR_SECONDARY)
                    end
                else
                    sim_q = Dict{Float64, Vector{Float64}}(
                        (pc => Vector{Float64}(undef, n_bins_eff)) for pc in percentiles
                    )
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
                        _push_xy!(lines_df[:solid], "", x_plot, y_plot, lbl)
                        push!(palette, lbl => COLOR_SECONDARY)
                    end
                end
            end
        end

        plots[oi] = _vpc_panel(
            style, kwargs_subplot, x_label, _axis_label(obs_label);
            obs_df, kern_df, lines_df, bern_df, pmf, palette,
            xlim = _pad_limits(minimum(x_for_bins), maximum(x_for_bins))
        )
    end

    if return_panel
        length(plots) == 1 ||
            error("return_panel requires a single observable; pass `observables = :name`.")
        return only(plots)
    end
    fig = combine_plots(plots; ncols = ncols, style = style, _nt(kwargs_plot)...)
    return _save_plot!(fig, save_path)
end

# One VPC panel: observed points, observed percentile lines (labeled for the quantile
# method, unlabeled for the kernel method as before), simulated percentile lines, the
# simulated Bernoulli probability, or the simulated PMF (markers coloured by probability).
function _vpc_panel(
        style::PlotStyle, kwargs_subplot, xlabel, ylabel;
        obs_df, kern_df, lines_df, bern_df, pmf, palette, xlim
    )
    xy = AoG.mapping(:x, :y; color = :series)
    line_layers = Any[]
    for ls in (:solid, :dot)
        push!(line_layers, _layer(lines_df[ls], xy, _line_visual(style; linestyle = ls)))
        push!(
            line_layers,
            _layer(
                kern_df[ls], AoG.mapping(:x, :y; group = :series),
                _line_visual(style; linestyle = ls, color = COLOR_ACCENT)
            )
        )
    end
    pmf_marker = (;
        marker = :xcross, markersize = style.marker_size_pmf,
        strokewidth = style.marker_stroke_width_pmf, strokecolor = :black,
    )
    # The PMF markers carry their own colour per point, so they are drawn in `post`; a
    # single zero-size AoG marker gives the "sim PMF" legend entry.
    pmf_key = isempty(pmf.x) ? nothing :
        _layer(
            _push_xy!(_tidy(), "", pmf.x[1:1], pmf.y[1:1], "sim PMF"), xy,
            AoG.visual(
                Makie.Scatter; pmf_marker..., markersize = 0,
                legend = (; markersize = style.marker_size_pmf)
            )
        )
    layers = _layers(
        _layer(obs_df, xy, _scatter_visual(style; alpha = 1.0)),
        line_layers...,
        _layer(bern_df, xy, _scatter_visual(style; pmf_marker...)),
        pmf_key,
    )
    post = Any[]
    if !isempty(pmf.x)
        push!(
            post, ax -> scatter!(
                ax, pmf.x, pmf.y; color = pmf.p, colormap = :viridis, pmf_marker...
            )
        )
    end
    push!(post, ax -> xlims!(ax, xlim...))
    return AoGPanel(
        layers;
        scales = isempty(palette) ? NamedTuple() : (; Color = _series_palette(palette)),
        axis = _aog_axis(style, kwargs_subplot; xlabel = xlabel, ylabel = ylabel),
        post = post
    )
end
