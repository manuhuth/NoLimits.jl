
function _plot_hidden_states_impl(
        dm::DataModel,
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
        individuals_idx = nothing
    )
    inds = _resolve_individuals(dm, individuals_idx; default_all = true)
    plots = Vector{Any}(undef, length(inds))
    xlims = nothing
    ylims = nothing
    for (k, i) in enumerate(inds)
        ind = get_individuals(dm)[i]
        obs_rows = get_obs_rows(get_row_groups(dm))[i]
        x_vals = _get_x_values(dm, ind, obs_rows, x_axis_feature)
        title_id = string(
            get_primary_id(dm), ": ", _individual_id(dm, i)
        )
        _kw870 = merge(
            (
                xlabel = x_axis_feature === nothing ? "Time" : _axis_label(x_axis_feature),
                ylabel = "Hidden-state probability",
            ),
            kwargs_subplot
        )
        p = create_styled_plot(; title = title_id, style = style, _kw870...)
        θ_ind = θ
        η_ind = η_vec[i]
        rowwise_re = _needs_rowwise_random_effects(dm, i; obs_only = true)
        sol_accessors = nothing
        if get_de(get_model(dm)) !== nothing
            sol = nothing
            compiled = nothing
            pre = calculate_prede(get_model(dm), θ_ind, η_ind, get_const_cov(ind))
            pc = (;
                fixed_effects = θ_ind,
                random_effects = η_ind,
                constant_covariates = get_const_cov(ind),
                varying_covariates = merge(
                    (t = get_vary(get_series(ind)).t[1],), get_dyn(get_series(ind))
                ),
                helpers = get_helper_funs(get_model(dm)),
                model_funs = get_model_funs(get_model(dm)),
                preDE = pre,
            )
            compiled = get_de_compiler(get_de(get_model(dm)))(pc)
            sol = cache = nothing
            sol = get_de(get_model(dm)) !== nothing ?
                _solve_dense_individual(dm, ind, θ_ind, η_ind)[1] : nothing
            sol_accessors = _sol_accessors_with_crossings(
                get_model(dm), sol, compiled, θ_ind, η_ind, get_const_cov(ind)
            )
        end

        times = Float64[]
        posteriors = Vector{Vector{Float64}}()
        n_states = nothing
        hmm_priors_hs = Dict{Symbol, Any}()
        for (j, row) in enumerate(obs_rows)
            vary = _varying_at(dm, ind, j, row)
            η_row = _row_random_effects_at(dm, i, j, η_ind, rowwise_re; obs_only = true)
            obs = sol_accessors === nothing ?
                calculate_formulas_obs(
                    get_model(dm), θ_ind, η_row, get_const_cov(ind), vary
                ) :
                calculate_formulas_obs(
                    get_model(dm), θ_ind, η_row, get_const_cov(ind), vary, sol_accessors
                )
            dist = getproperty(obs, obs_name)
            _is_hmm_dist(dist) ||
                error("Observable $(obs_name) must be a hidden-Markov-model outcome, got $(typeof(dist)).")
            y_val = getfield(get_obs(get_series(ind)), obs_name)[j]
            prior = get(hmm_priors_hs, obs_name, nothing)
            dist_filtered = _hmm_with_prior(dist, prior)
            if y_val === missing
                hmm_priors_hs[obs_name] = probabilities_hidden_states(dist_filtered)
                continue
            end
            post = posterior_hidden_states(dist_filtered, y_val)
            hmm_priors_hs[obs_name] = post
            if n_states === nothing
                n_states = length(post)
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
                _record!(
                    p,
                    ax -> poly!(
                        ax, rects; color = state_colors[m], strokewidth = 0, label = lbl
                    )
                )
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
function plot_hidden_states(
        res::FitResult;
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
        rng::AbstractRNG = _default_rng()
    )
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
    _is_posterior_draw_fit(res) &&
        error("plot_hidden_states does not support posterior draws yet.")

    θ = get_params(res; scale = :untransformed)
    θ = _apply_param_overrides(θ, params)
    η_vec = _default_random_effects(res, dm, constants_re_use, θ, rng, mcmc_draws)
    inds = _resolve_individuals(dm, individuals_idx; default_all = true)

    if figure_layout == :vector
        plots = Vector{Figure}(undef, length(inds))
        for (k, idx) in enumerate(inds)
            plots[k] = _plot_hidden_states_impl(
                dm,
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
                [idx]
            )
        end
        return plots
    end

    return _plot_hidden_states_impl(
        dm,
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
        inds
    )
end

function plot_hidden_states(
        dm::DataModel;
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
        rng::AbstractRNG = _default_rng()
    )
    figure_layout in (:single, :vector) ||
        error("figure_layout must be :single or :vector.")
    if figure_layout == :vector
        (save_path === nothing && plot_path === nothing) ||
            error("save_path/plot_path are not supported when returning multiple figures.")
    end
    save_path = _resolve_plot_path(save_path, plot_path)
    obs_name = _get_observable(dm, observable)

    θ = get_θ0_untransformed(get_fixed(get_model(dm)))
    θ = _apply_param_overrides(θ, params)
    η_vec = _default_random_effects_from_dm(dm, constants_re, θ)
    inds = _resolve_individuals(dm, individuals_idx; default_all = true)

    if figure_layout == :vector
        plots = Vector{Figure}(undef, length(inds))
        for (k, idx) in enumerate(inds)
            plots[k] = _plot_hidden_states_impl(
                dm,
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
                [idx]
            )
        end
        return plots
    end

    return _plot_hidden_states_impl(
        dm,
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
        inds
    )
end

function _plot_emission_for_individual(
        dm::DataModel,
        obs_name::Symbol,
        ind_idx::Int,
        θ,
        η_ind,
        row::Int,
        title_base::String,
        style::PlotStyle,
        kwargs_subplot,
        state_ncols::Int
    )
    ind = get_individuals(dm)[ind_idx]
    obs_rows = get_obs_rows(get_row_groups(dm))[ind_idx]
    row_pos = findfirst(==(row), obs_rows)
    row_pos === nothing &&
        error("Observation row $(row) not found for individual $(ind_idx).")

    sol_accessors = nothing
    if get_de(get_model(dm)) !== nothing
        compiled = nothing
        pre = calculate_prede(get_model(dm), θ, η_ind, get_const_cov(ind))
        pc = (;
            fixed_effects = θ,
            random_effects = η_ind,
            constant_covariates = get_const_cov(ind),
            varying_covariates = merge(
                (t = get_vary(get_series(ind)).t[1],), get_dyn(get_series(ind))
            ),
            helpers = get_helper_funs(get_model(dm)),
            model_funs = get_model_funs(get_model(dm)),
            preDE = pre,
        )
        compiled = get_de_compiler(get_de(get_model(dm)))(pc)
        sol = _solve_dense_individual(dm, ind, θ, η_ind)[1]
        sol_accessors = _sol_accessors_with_crossings(
            get_model(dm), sol, compiled, θ, η_ind, get_const_cov(ind)
        )
    end

    vary = _varying_at(dm, ind, row_pos, row)
    rowwise_re = _needs_rowwise_random_effects(dm, ind_idx; obs_only = true)
    η_row = _row_random_effects_at(dm, ind_idx, row_pos, η_ind, rowwise_re; obs_only = true)
    obs = sol_accessors === nothing ?
        calculate_formulas_obs(get_model(dm), θ, η_row, get_const_cov(ind), vary) :
        calculate_formulas_obs(
            get_model(dm), θ, η_row, get_const_cov(ind), vary, sol_accessors
        )
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
            title = "$title_base • State $s", style = style, _kw1209...
        )
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
                create_styled_scatter!(
                    p_state, grid.vals, grid.probs;
                    label = label,
                    color = marginal_colors[m],
                    style = style
                )
                state_xlim = _merge_limits(state_xlim, grid.vals)
                state_ylim = _merge_limits(state_ylim, grid.probs)
            else
                grid = _density_grid_continuous([dist_m], 0.995, 200)
                grid === nothing && continue
                densities = vec(grid.z[:, 1])
                create_styled_line!(
                    p_state, grid.y, densities;
                    label = label,
                    color = marginal_colors[m],
                    style = style
                )
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

function _plot_emission_impl(
        dm::DataModel,
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
        figure_layout::Symbol
    )
    # Reject non-HMM outcomes at entry instead of deep inside the per-individual path.
    dist_rep = _representative_dist(dm, obs_name, nothing)
    dist_rep isa MVDiscreteTimeDiscreteStatesHMM || throw(
        ArgumentError(
            "plot_emission_distributions requires an MVDiscreteTimeDiscreteStatesHMM " *
                "observable; :$(obs_name) is a $(typeof(dist_rep))."
        )
    )
    inds = _resolve_individuals(dm, individuals_idx; default_all = true)
    groups = Vector{MakiePanelGroup}(undef, length(inds))
    xlims = nothing
    ylims = nothing
    time_col_use = time_col === nothing ? get_time_col(dm) : time_col
    state_ncols = min(3, DEFAULT_PLOT_COLS)

    for (k, i) in enumerate(inds)
        ind = get_individuals(dm)[i]
        obs_rows = get_obs_rows(get_row_groups(dm))[i]
        row = _resolve_emission_row(dm, obs_rows, time_idx, time_point, time_col_use)
        title_id = string(get_primary_id(dm), ": ", get_df(dm)[row, get_primary_id(dm)])
        time_val = get_df(dm)[row, time_col_use]
        time_label = string(time_val)
        record = _plot_emission_for_individual(
            dm,
            obs_name,
            i,
            θ,
            η_vec[i],
            row,
            "$title_id • t = $time_label",
            style,
            kwargs_subplot,
            state_ncols
        )
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
function plot_emission_distributions(
        res::FitResult;
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
        rng::AbstractRNG = _default_rng()
    )
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

    plots = _plot_emission_impl(
        dm,
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
        figure_layout
    )

    return plots === nothing ? [] : plots
end

function plot_emission_distributions(
        dm::DataModel;
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
        rng::AbstractRNG = _default_rng()
    )
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

    θ = get_θ0_untransformed(get_fixed(get_model(dm)))
    θ = _apply_param_overrides(θ, params)
    η_vec = _default_random_effects_from_dm(dm, constants_re, θ)

    plots = _plot_emission_impl(
        dm,
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
        figure_layout
    )

    return plots === nothing ? [] : plots
end

function _plot_fits_comparison_impl(
        fits::AbstractVector{<:FitResult},
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
        plot_path::Union{Nothing, String} = nothing,
        rng::AbstractRNG = _default_rng()
    )
    length(fits) == length(labels) ||
        error("Internal error: fits and labels length mismatch.")
    isempty(fits) && error("plot_fits_comparison requires at least one fit result.")

    save_path = _resolve_plot_path(save_path, plot_path)
    dms = dm === nothing ? [_get_dm(res, nothing) for res in fits] : fill(dm, length(fits))
    dm_ref = _validate_same_data_model_for_comparison(dms)

    obs_name = _get_observable(dm_ref, observable)
    inds = _resolve_individuals(dm_ref, individuals_idx; default_all = true)
    caches = [
        build_plot_cache(fits[j]; dm = dms[j], cache_obs_dists = false, rng = rng)
            for j in eachindex(fits)
    ]
    line_colors = _comparison_line_colors(length(fits), style)

    plots = Vector{Any}(undef, length(inds))
    xlims = nothing
    ylims = nothing
    for (k, i) in enumerate(inds)
        ind = get_individuals(dm_ref)[i]
        obs_rows = get_obs_rows(get_row_groups(dm_ref))[i]
        x_obs = _get_x_values(dm_ref, ind, obs_rows, x_axis_feature)
        y_obs = getfield(get_obs(get_series(ind)), obs_name)
        x_obs_plot, y_obs_plot = _collect_scalar_series(x_obs, y_obs)
        title_id = string(
            get_primary_id(dm_ref), ": ", _individual_id(dm_ref, i)
        )
        _kw1619 = merge(
            (
                xlabel = x_axis_feature === nothing ? "Time" : _axis_label(x_axis_feature),
                ylabel = _axis_label(obs_name),
            ),
            kwargs_subplot
        )
        p = create_styled_plot(; title = title_id, style = style, _kw1619...)
        if plot_data_points
            create_styled_scatter!(
                p, x_obs_plot, y_obs_plot; label = "data",
                color = style.color_primary, style = style
            )
        end

        for j in eachindex(fits)
            curve = _fit_curve_from_cache(
                dms[j], caches[j], i, obs_name, x_axis_feature, plot_func
            )
            create_styled_line!(
                p,
                curve.x_fit,
                curve.preds;
                label = labels[j],
                color = line_colors[j],
                style = style,
                linestyle = _comparison_line_style(labels[j], style)
            )
            xlims = _merge_limits(xlims, curve.x_fit)
            ylims = _merge_limits(ylims, curve.preds)
        end

        ylims = _merge_limits(ylims, y_obs_plot)
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
    plot_fits_comparison(res::Union{FitResult, MultistartFitResult}; kwargs...)
                         -> Makie.Figure

    plot_fits_comparison(results::AbstractVector; kwargs...) -> Makie.Figure

    plot_fits_comparison(results::NamedTuple; kwargs...) -> Makie.Figure

    plot_fits_comparison(results::AbstractDict; kwargs...) -> Makie.Figure

Plot predictions from one or more fitted models side-by-side for visual comparison.

When called with a single `FitResult` or `MultistartFitResult`, behaves like
[`plot_fits`](@ref). When called with a collection, overlays predictions from each
model on the same panel, labeled by vector index, `NamedTuple` key, or `Dict` key.

All keyword arguments are forwarded to the underlying `plot_fits` implementation,
including `rng` (default: a fresh `Xoshiro(42)` per call) for the posterior-mean
parameters of MCMC/VI fits.
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
- `marginal_idx::Union{Nothing, Int} = nothing`: for vector-valued observables, plot only
  this component; `nothing` overlays every component in its own color.
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `kwargs_subplot`: additional Makie `Axis` attributes forwarded to the single panel.
- `kwargs_layout`: additional Makie `Figure` attributes forwarded to the layout call.
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
- `plot_path::Union{Nothing, String} = nothing`: alias for `save_path`.
"""
function plot_observed_profiles(
        dm::DataModel;
        observable = nothing,
        individuals_idx = nothing,
        x_axis_feature::Union{Symbol, Nothing} = nothing,
        marginal_idx::Union{Nothing, Int} = nothing,
        style::PlotStyle = PlotStyle(),
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing
    )
    save_path = _resolve_plot_path(save_path, plot_path)
    obs_name = _get_observable(dm, observable)
    inds = _resolve_individuals(dm, individuals_idx; default_all = true)
    (is_mv, n_marginals) = _obs_multivariate_info(dm, obs_name)
    marginal_idx !== nothing && !is_mv &&
        error("marginal_idx only valid when the observable is multivariate.")
    marginal_idx !== nothing && !(1 <= marginal_idx <= n_marginals) &&
        error("marginal_idx must be between 1 and $(n_marginals).")
    margins = is_mv ? (marginal_idx === nothing ? (1:n_marginals) : (marginal_idx:marginal_idx)) :
        (0:0)
    colors = _marginal_colors(max(n_marginals, 1), style)

    _kw_op = merge(
        (
            xlabel = x_axis_feature === nothing ? "Time" : _axis_label(x_axis_feature),
            ylabel = _axis_label(obs_name),
        ),
        kwargs_subplot
    )
    p = create_styled_plot(; title = "Observed profiles", style = style, _kw_op...)
    p.legend_position = :none

    for i in inds
        ind = get_individuals(dm)[i]
        obs_rows = get_obs_rows(get_row_groups(dm))[i]
        x = _get_x_values(dm, ind, obs_rows, x_axis_feature)
        y = getfield(get_obs(get_series(ind)), obs_name)
        for m in margins
            xs, ys = m == 0 ? _collect_scalar_series(x, y) :
                _collect_multivariate_series(x, y, n_marginals; marginal_idx = m)
            isempty(xs) && continue
            order = sortperm(xs)
            xs_sorted = xs[order]
            ys_sorted = ys[order]
            col = m == 0 ? style.color_primary : colors[m]
            lbl = m == 0 ? "" : _marginal_label(obs_name, m)
            create_styled_line!(
                p, xs_sorted, ys_sorted; label = lbl, color = col, style = style
            )
            create_styled_scatter!(
                p, xs_sorted, ys_sorted; label = lbl, color = col, style = style
            )
        end
    end

    fig = combine_plots([p]; ncols = 1, style = style, kwargs_layout...)
    return _save_plot!(fig, save_path)
end

function plot_observed_profiles(
        res::FitResult;
        dm::Union{Nothing, DataModel} = nothing,
        observable = nothing,
        individuals_idx = nothing,
        x_axis_feature::Union{Symbol, Nothing} = nothing,
        marginal_idx::Union{Nothing, Int} = nothing,
        style::PlotStyle = PlotStyle(),
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing
    )
    dm = _get_dm(res, dm)
    return plot_observed_profiles(
        dm;
        observable = observable,
        individuals_idx = individuals_idx,
        x_axis_feature = x_axis_feature,
        marginal_idx = marginal_idx,
        style = style,
        kwargs_subplot = kwargs_subplot,
        kwargs_layout = kwargs_layout,
        save_path = save_path,
        plot_path = plot_path
    )
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
function plot_dv_pred(
        res::FitResult;
        dm::Union{Nothing, DataModel} = nothing,
        observable = nothing,
        style::PlotStyle = PlotStyle(),
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing
    )
    dm = _get_dm(res, dm)
    save_path = _resolve_plot_path(save_path, plot_path)
    obs_name = _get_observable(dm, observable)
    θ = get_params(res; scale = :untransformed)
    η_pred = _pred_re_per_individual(dm, θ)

    panels_spec, n_marginals = _marginal_panels(dm, obs_name)
    panels = Vector{Any}(undef, length(panels_spec))
    for (k, (m, label)) in enumerate(panels_spec)
        dv, pred, _ = _collect_pred_series(
            dm, obs_name, θ, η_pred;
            marginal = m, n_marginals = n_marginals, fn = "plot_dv_pred"
        )
        isempty(dv) &&
            error("No non-missing observations found for observable :$(label).")

        lo = min(minimum(dv), minimum(pred))
        hi = max(maximum(dv), maximum(pred))
        lims = _pad_limits(lo, hi)

        _kw_dvp = merge(
            (
                xlabel = "Population prediction (PRED)",
                ylabel = "Observed (DV)",
                title = m === nothing ? "" : label,
            ),
            kwargs_subplot
        )
        p = create_styled_plot(; style = style, _kw_dvp...)
        create_styled_scatter!(
            p, pred, dv; label = "", color = style.color_primary, style = style
        )
        create_styled_line!(
            p, collect(lims), collect(lims);
            color = style.color_reference, linestyle = :dash,
            linewidth = style.line_width_secondary, label = "", style = style
        )
        _set_limits!(p; xlim = lims, ylim = lims)
        panels[k] = p
    end

    fig = combine_plots(
        panels; ncols = min(length(panels), DEFAULT_PLOT_COLS), style = style,
        kwargs_layout...
    )
    return _save_plot!(fig, save_path)
end

"""
    plot_dv_ipred(res::FitResult; dm, observable, style, rng, kwargs_subplot,
                  kwargs_layout, save_path, plot_path) -> Makie.Figure

Goodness-of-fit scatter plot of observed values (DV) against individual
predictions (IPRED). IPRED is the model prediction evaluated at each
individual's empirical Bayes estimates of the random effects. An identity
line is overlaid for reference.

# Keyword Arguments
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
- `observable`: name of the outcome variable, or `nothing` for the first.
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `rng::AbstractRNG = Xoshiro(42)`: random-number generator for the random effects of
  posterior-draw fits. A fresh generator with this fixed seed is created per call.
- `kwargs_subplot`: additional Makie `Axis` attributes forwarded to the subplot.
- `kwargs_layout`: additional Makie `Figure` attributes forwarded to the layout.
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
- `plot_path::Union{Nothing, String} = nothing`: alias for `save_path`.
- `return_panel::Bool = false`: return the built panel instead of a `Figure`, for
  composing several diagnostics into one figure via `combine_plots`.
"""
function plot_dv_ipred(
        res::FitResult;
        dm::Union{Nothing, DataModel} = nothing,
        observable = nothing,
        style::PlotStyle = PlotStyle(),
        rng::AbstractRNG = _default_rng(),
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing,
        return_panel::Bool = false
    )
    dm = _get_dm(res, dm)
    save_path = _resolve_plot_path(save_path, plot_path)
    obs_name = _get_observable(dm, observable)
    θ = get_params(res; scale = :untransformed)
    constants_re = _res_constants_re(res, NamedTuple())
    η_ebe = _default_random_effects(res, dm, constants_re, θ, rng, 1)

    panels_spec, n_marginals = _marginal_panels(dm, obs_name)
    return_panel && length(panels_spec) > 1 &&
        error("return_panel requires a scalar observable; :$(obs_name) has $(n_marginals) components.")
    panels = Vector{Any}(undef, length(panels_spec))
    for (k, (m, label)) in enumerate(panels_spec)
        dv, ipred = _collect_ipred_series(
            dm, obs_name, θ, η_ebe;
            marginal = m, n_marginals = n_marginals, fn = "plot_dv_ipred"
        )
        isempty(dv) &&
            error("No non-missing observations found for observable :$(label).")

        lo = min(minimum(dv), minimum(ipred))
        hi = max(maximum(dv), maximum(ipred))
        lims = _pad_limits(lo, hi)

        _kw_dvi = merge(
            (
                xlabel = "Individual prediction (IPRED)",
                ylabel = "Observed (DV)",
                title = m === nothing ? "" : label,
            ),
            kwargs_subplot
        )
        p = create_styled_plot(; style = style, _kw_dvi...)
        create_styled_scatter!(
            p, ipred, dv; label = "", color = style.color_primary, style = style
        )
        create_styled_line!(
            p, collect(lims), collect(lims);
            color = style.color_reference, linestyle = :dash,
            linewidth = style.line_width_secondary, label = "", style = style
        )
        _set_limits!(p; xlim = lims, ylim = lims)
        panels[k] = p
    end

    return_panel && return only(panels)
    fig = combine_plots(
        panels; ncols = min(length(panels), DEFAULT_PLOT_COLS), style = style,
        kwargs_layout...
    )
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
- `return_panel::Bool = false`: return the built panel instead of a `Figure`, for
  composing several diagnostics into one figure via `combine_plots`.
"""
function plot_wres_pred(
        res::FitResult;
        dm::Union{Nothing, DataModel} = nothing,
        observable = nothing,
        style::PlotStyle = PlotStyle(),
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing,
        return_panel::Bool = false
    )
    dm = _get_dm(res, dm)
    save_path = _resolve_plot_path(save_path, plot_path)
    obs_name = _get_observable(dm, observable)
    θ = get_params(res; scale = :untransformed)
    η_pred = _pred_re_per_individual(dm, θ)

    panels_spec, n_marginals = _marginal_panels(dm, obs_name)
    return_panel && length(panels_spec) > 1 &&
        error("return_panel requires a scalar observable; :$(obs_name) has $(n_marginals) components.")
    panels = Vector{Any}(undef, length(panels_spec))
    for (k, (m, label)) in enumerate(panels_spec)
        dv, pred, sigma = _collect_pred_series(
            dm, obs_name, θ, η_pred;
            marginal = m, n_marginals = n_marginals, fn = "plot_wres_pred"
        )
        isempty(dv) &&
            error("No non-missing observations found for observable :$(label).")

        wres = (dv .- pred) ./ sigma

        pred_lo, pred_hi = _pad_limits(minimum(pred), maximum(pred))
        wres_lo, wres_hi = _pad_limits(minimum(wres), maximum(wres))

        _kw_wres = merge(
            (
                xlabel = "Population prediction (PRED)",
                ylabel = "Weighted residual (WRES)",
                title = m === nothing ? "" : label,
            ),
            kwargs_subplot
        )
        p = create_styled_plot(; style = style, _kw_wres...)
        create_styled_scatter!(
            p, pred, wres; label = "", color = style.color_primary, style = style
        )
        add_reference_line!(
            p, 0.0; orientation = :horizontal,
            color = style.color_reference,
            linewidth = style.line_width_secondary, label = ""
        )
        _set_limits!(p; xlim = (pred_lo, pred_hi), ylim = (wres_lo, wres_hi))
        panels[k] = p
    end

    return_panel && return only(panels)
    fig = combine_plots(
        panels; ncols = min(length(panels), DEFAULT_PLOT_COLS), style = style,
        kwargs_layout...
    )
    return _save_plot!(fig, save_path)
end

"""
    plot_shrinkage(res::FitResult; dm, constants_re, threshold, style,
                   kwargs_subplot, save_path, plot_path) -> Makie.Figure

Horizontal bar chart of eta shrinkage for all scalar random effects, with a
vertical reference line at `threshold` (default 30 %).

Bars are colored by severity: below `threshold` (green), between `threshold`
and 50 % (orange), above 50 % (red). Negative shrinkage (SD of the EBEs above
the fitted RE scale, common in small samples) is drawn as a bar extending left
of 0 and colored green. Shrinkage is computed via
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
function plot_shrinkage(
        res::FitResult;
        dm::Union{Nothing, DataModel} = nothing,
        constants_re::NamedTuple = NamedTuple(),
        threshold::Real = 0.3,
        bar_color::Union{Nothing, String} = nothing,
        style::PlotStyle = PlotStyle(),
        kwargs_subplot = NamedTuple(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing
    )
    save_path = _resolve_plot_path(save_path, plot_path)
    _check_unit_interval(threshold, "threshold")
    shrink_nt = compute_shrinkage(res; dm = dm, constants_re = constants_re)
    fig = _shrinkage_figure(shrink_nt; threshold = threshold, bar_color = bar_color, style = style, kwargs_subplot = kwargs_subplot)
    return _save_plot!(fig, save_path)
end

function _shrinkage_figure(shrink_nt::NamedTuple; threshold, bar_color, style, kwargs_subplot)
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
        kwargs_subplot...
    )

    yticks_pos = collect(1:n)
    _record!(
        p,
        ax -> barplot!(
            ax, yticks_pos, values_pct;
            direction = :x, color = bar_colors, width = 0.6
        )
    )
    for i in 1:n
        add_annotation!(
            p, max(values_pct[i], 0.0) + 1.5, yticks_pos[i],
            string(round(values_pct[i]; digits = 1), "%"); fontsize = 9, halign = :left
        )
    end

    add_reference_line!(
        p, threshold * 100; orientation = :vertical,
        color = style.color_reference, linewidth = style.line_width_secondary,
        label = "$(round(Int, threshold * 100))%"
    )

    _axis_attrs!(p; yticks = (yticks_pos, re_labels))
    _set_limits!(
        p;
        xlim = (min(-2.0, minimum(values_pct) * 1.25 - 2.0), max(maximum(values_pct) * 1.25 + 5.0, threshold * 100 * 1.5)),
        ylim = (0.5, n + 0.5)
    )
    p.legend_position = :rb

    return combine_plots([p]; ncols = 1, style = style)
end
