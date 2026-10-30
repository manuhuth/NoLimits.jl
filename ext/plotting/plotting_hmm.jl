# Hidden-Markov-model outcomes on AlgebraOfGraphics: plot_hidden_states (stacked bars of the
# filtered hidden-state probabilities, one facet per individual) and
# plot_emission_distributions (per-state emission marginals on an individual x state grid).

# --- shared helpers ----------------------------------------------------------------------

# Resolve and validate the HMM observable of `fn`: an unknown name or a non-HMM outcome
# raises an ArgumentError before any per-individual work.
function _hmm_observable(dm::DataModel, observable, fn::AbstractString)
    obs = get_formulas_meta(get_formulas(get_model(dm))).obs_names
    name = if observable === nothing
        _get_observable(dm, nothing)
    elseif observable isa Symbol || observable isa AbstractString
        Symbol(observable)
    else
        throw(ArgumentError("$(fn): observable must be a Symbol; got $(repr(observable))."))
    end
    name in obs || throw(
        ArgumentError(
            "$(fn): observable :$(name) not found. Available observables: " *
                join((":" * string(o) for o in obs), ", ") * "."
        )
    )
    dist = _representative_dist(dm, name, nothing)
    _is_hmm_dist(dist) || throw(
        ArgumentError(
            "$(fn): observable :$(name) is not a hidden-Markov-model outcome " *
                "(got $(nameof(typeof(dist))))."
        )
    )
    return name
end

# Observation-formula evaluator for individual `i`: `row_dist(j, row, η_row)` returns the
# outcome distribution of `obs_name` at the j-th observation row.
function _hmm_obs_dist_fn(dm::DataModel, obs_name::Symbol, θ, η_ind, i::Integer)
    model = get_model(dm)
    ind = get_individuals(dm)[i]
    sol_accessors = nothing
    if get_de(model) !== nothing
        sol, compiled = _solve_dense_individual(dm, ind, θ, η_ind)
        sol_accessors = _sol_accessors_with_crossings(
            model, sol, compiled, θ, η_ind, get_const_cov(ind)
        )
    end
    return function (j, row, η_row)
        vary = _varying_at(dm, ind, j, row)
        obs = sol_accessors === nothing ?
            calculate_formulas_obs(model, θ, η_row, get_const_cov(ind), vary) :
            calculate_formulas_obs(model, θ, η_row, get_const_cov(ind), vary, sol_accessors)
        return getproperty(obs, obs_name)
    end
end

function _hmm_check_layout(figure_layout, save_path, plot_path)
    figure_layout in (:single, :vector) ||
        error("figure_layout must be :single or :vector.")
    if figure_layout == :vector
        (save_path === nothing && plot_path === nothing) ||
            error("save_path/plot_path are not supported when returning multiple figures.")
    end
    return nothing
end

# --- plot_hidden_states ------------------------------------------------------------------

# Filtered hidden-state posteriors of individual `i` at its non-missing observations:
# `(x, posteriors)` with one probability vector per kept observation.
function _hmm_state_posteriors(dm::DataModel, obs_name::Symbol, θ, η_ind, i::Integer, x_axis_feature)
    ind = get_individuals(dm)[i]
    obs_rows = get_obs_rows(get_row_groups(dm))[i]
    x_vals = _get_x_values(dm, ind, obs_rows, x_axis_feature)
    rowwise_re = _needs_rowwise_random_effects(dm, i; obs_only = true)
    row_dist = _hmm_obs_dist_fn(dm, obs_name, θ, η_ind, i)
    y_all = getfield(get_obs(get_series(ind)), obs_name)
    xs = Float64[]
    posteriors = Vector{Vector{Float64}}()
    prior = nothing
    for (j, row) in enumerate(obs_rows)
        η_row = _row_random_effects_at(dm, i, j, η_ind, rowwise_re; obs_only = true)
        dist = row_dist(j, row, η_row)
        _is_hmm_dist(dist) || throw(
            ArgumentError(
                "Observable :$(obs_name) must be a hidden-Markov-model outcome; " *
                    "got $(nameof(typeof(dist)))."
            )
        )
        dist_filtered = _hmm_with_prior(dist, prior)
        y_val = y_all[j]
        if y_val === missing
            prior = probabilities_hidden_states(dist_filtered)
            continue
        end
        post = posterior_hidden_states(dist_filtered, y_val)
        prior = post
        push!(xs, Float64(x_vals[j]))
        push!(posteriors, Float64.(collect(post)))
    end
    return xs, posteriors
end

# Bar width of one individual: 80 % of its smallest positive x spacing, at most 1.
function _hmm_bar_width(xs::AbstractVector{<:Real})
    spacings = filter(>(0), diff(sort(unique(xs))))
    return isempty(spacings) ? 1.0 : min(0.8 * minimum(spacings), 1.0)
end

function _plot_hidden_states_impl(
        dm::DataModel, obs_name::Symbol, θ, η_vec, inds::AbstractVector{<:Integer};
        x_axis_feature, shared_x_axis::Bool, shared_y_axis::Bool, ncols::Integer,
        style::PlotStyle, kwargs_subplot, kwargs_layout,
        save_path::Union{Nothing, String} = nothing
    )
    df = DataFrame(
        panel = String[], x = Float64[], top = Float64[], bottom = Float64[],
        width = Float64[], state = String[]
    )
    panels = String[]
    xlims = nothing
    n_states = 0
    for i in inds
        panel = _individual_panel(dm, i)
        push!(panels, panel)
        xs, posts = _hmm_state_posteriors(dm, obs_name, θ, η_vec[i], i, x_axis_feature)
        if isempty(xs)
            @warn "No non-missing observations found for individual $(panel)."
            continue
        end
        w = _hmm_bar_width(xs)
        n_states = max(n_states, length(first(posts)))
        for (x, p) in zip(xs, posts)
            bottom = 0.0
            for (m, pm) in enumerate(p)
                push!(df, (panel, x, bottom + pm, bottom, w, "State $(m)"))
                bottom += pm
            end
        end
        xlims = _merge_limits(xlims, (minimum(xs) - w / 2, maximum(xs) + w / 2))
    end
    states = ["State $(m)" for m in 1:n_states]
    colors = _marginal_colors(max(n_states, 1), style)
    layer = _layer(
        df,
        AoG.mapping(:x, :top; fillto = :bottom, width = :width, color = :state, layout = :panel),
        AoG.visual(Makie.BarPlot; gap = 0, strokewidth = 0),
    )
    fig = _draw_facets(
        layer;
        panels, ncols, style, shared_x = shared_x_axis, shared_y = shared_y_axis,
        kwargs_subplot, kwargs_layout,
        axis = (;
            xlabel = x_axis_feature === nothing ? "Time" : _axis_label(x_axis_feature),
            ylabel = "Hidden-state probability",
        ),
        series = [s => c for (s, c) in zip(states, colors)],
        xlim = _shared_lim(shared_x_axis, xlims),
        ylim = shared_y_axis ? (0.0, 1.0) : nothing,
    )
    return _save_plot!(fig, save_path)
end

function _hidden_states_figures(
        dm, obs_name, θ, η_vec, individuals_idx, figure_layout, save_path; kw...
    )
    inds = _plot_individuals(dm, individuals_idx)
    figure_layout == :vector &&
        return Figure[_plot_hidden_states_impl(dm, obs_name, θ, η_vec, [i]; kw...) for i in inds]
    return _plot_hidden_states_impl(dm, obs_name, θ, η_vec, inds; save_path, kw...)
end

"""
    plot_hidden_states(res::FitResult; kwargs...)
    plot_hidden_states(dm::DataModel; kwargs...)

Plot the filtered posterior probability of each hidden state over time for a
hidden-Markov-model outcome. Each observation time shows one stacked bar (State 1 at the
bottom); there is one panel per individual and one figure legend for the states. Bar widths
are 80 % of the smallest time spacing of each individual (at most 1). Missing observations
only propagate the filter and draw no bar.

# Keyword Arguments
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
- `observable = nothing`: HMM observable to plot; defaults to the first one. An unknown or
  non-HMM observable raises an `ArgumentError`.
- `individuals_idx`: indices or IDs of individuals to include, or `nothing` for all.
- `x_axis_feature::Union{Nothing, Symbol} = nothing`: covariate for the x-axis; defaults to time.
- `shared_x_axis::Bool = true`, `shared_y_axis::Bool = true`: link the axes of all panels.
- `ncols::Int = 3`: panels per row.
- `figure_layout::Symbol = :single`: `:vector` returns one `Figure` per individual.
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `kwargs_subplot`, `kwargs_layout`: Makie `Axis` / `Figure` attributes (user values win).
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot (`:single` only).
- `params`, `constants_re`, `mcmc_draws`, `rng`: parameter overrides and random-effect
  settings as in [`plot_fits`](@ref).
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
    _hmm_check_layout(figure_layout, save_path, plot_path)
    _check_positive_int(ncols, "ncols")
    save_path = _resolve_plot_path(save_path, plot_path)
    constants_re_use = _res_constants_re(res, constants_re)
    obs_name = _hmm_observable(dm, observable, "plot_hidden_states")
    _is_posterior_draw_fit(res) &&
        error("plot_hidden_states does not support posterior draws yet.")

    θ = get_params(res; scale = :untransformed)
    θ = _apply_param_overrides(θ, params)
    η_vec = _default_random_effects(res, dm, constants_re_use, θ, rng, mcmc_draws)
    return _hidden_states_figures(
        dm, obs_name, θ, η_vec, individuals_idx, figure_layout, save_path;
        x_axis_feature, shared_x_axis, shared_y_axis, ncols, style, kwargs_subplot,
        kwargs_layout,
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
    _hmm_check_layout(figure_layout, save_path, plot_path)
    _check_positive_int(ncols, "ncols")
    save_path = _resolve_plot_path(save_path, plot_path)
    obs_name = _hmm_observable(dm, observable, "plot_hidden_states")

    θ = get_θ0_untransformed(get_fixed(get_model(dm)))
    θ = _apply_param_overrides(θ, params)
    η_vec = _default_random_effects_from_dm(dm, constants_re, θ)
    return _hidden_states_figures(
        dm, obs_name, θ, η_vec, individuals_idx, figure_layout, save_path;
        x_axis_feature, shared_x_axis, shared_y_axis, ncols, style, kwargs_subplot,
        kwargs_layout,
    )
end

# --- plot_emission_distributions -----------------------------------------------------------

# Per-state emission marginals of individual `i` at observation `row`: a tidy table with
# columns `state, x, y, series, kind` (`kind` = "pdf" for continuous marginals drawn as
# lines, "pmf" for discrete marginals drawn as points).
function _hmm_emission_table(dm::DataModel, obs_name::Symbol, θ, η_ind, i::Integer, row::Int)
    obs_rows = get_obs_rows(get_row_groups(dm))[i]
    j = findfirst(==(row), obs_rows)
    j === nothing && error("Observation row $(row) not found for individual $(i).")
    rowwise_re = _needs_rowwise_random_effects(dm, i; obs_only = true)
    η_row = _row_random_effects_at(dm, i, j, η_ind, rowwise_re; obs_only = true)
    dist = _hmm_obs_dist_fn(dm, obs_name, θ, η_ind, i)(j, row, η_row)
    dist isa MVDiscreteTimeDiscreteStatesHMM || throw(
        ArgumentError(
            "plot_emission_distributions requires an MVDiscreteTimeDiscreteStatesHMM " *
                "observable; :$(obs_name) is a $(nameof(typeof(dist)))."
        )
    )
    df = DataFrame(state = String[], x = Float64[], y = Float64[], series = String[], kind = String[])
    for s in 1:dist.n_states
        for (m, dist_m) in enumerate(_state_emission_marginals(dist.emission_dists[s]))
            dist_m === nothing && continue
            label = _marginal_label(obs_name, m)
            if _is_discrete(dist_m)
                grid = _density_grid_discrete(dist_m, 0.995)
                grid === nothing && continue
                xs, ys, kind = grid.vals, grid.probs, "pmf"
            else
                grid = _density_grid_continuous([dist_m], 0.995, 200)
                grid === nothing && continue
                xs, ys, kind = grid.y, vec(grid.z[:, 1]), "pdf"
            end
            for (x, y) in zip(xs, ys)
                push!(df, ("State $(s)", _f64(x), _f64(y), label, kind))
            end
        end
    end
    return df, dist.n_states, dist.n_outcomes
end

# Individual x state facet grid (rows: individuals, columns: states; x axes free, y axes
# linked when shared).
function _hmm_emission_figure(
        df::DataFrame, rows::Vector{String}, n_states::Int, n_marg::Int, obs_name::Symbol;
        ylim, shared_y_axis::Bool, style::PlotStyle, kwargs_subplot, kwargs_layout
    )
    states = ["State $(s)" for s in 1:n_states]
    labels = [_marginal_label(obs_name, m) for m in 1:n_marg]
    colors = _marginal_colors(n_marg, style)
    m = AoG.mapping(:x, :y; color = :series, row = :row, col = :state)
    layers = _layers(
        _layer(df[df.kind .== "pdf", :], m, _line_visual(style)),
        _layer(df[df.kind .== "pmf", :], m, _scatter_visual(style; markersize = style.marker_size_pmf)),
    )
    layers === nothing &&
        error("Nothing to plot: no emission distribution could be evaluated.")
    kinds = unique(df.kind)
    ylabel = kinds == ["pmf"] ? "Probability" :
        (length(kinds) > 1 ? "Density / probability" : "Density")
    sc = (;
        Row = (; categories = rows),
        Col = (; categories = states),
        Color = _series_palette([l => c for (l, c) in zip(labels, colors)]),
    )
    fig_kw = _aog_figure(style, length(rows) * n_states, n_states, kwargs_layout)
    fg = AoG.draw(
        layers, AoG.scales(; sc...);
        axis = _aog_axis(style, kwargs_subplot; xlabel = "Outcome value", ylabel),
        figure = fig_kw,
        facet = (; linkxaxes = :none, linkyaxes = _facet_link(shared_y_axis)),
        legend = _aog_legend(style; nbanks = _legend_nbanks(labels, first(fig_kw.size), style)),
    )
    if ylim !== nothing
        for ae in fg.grid
            isempty(ae.entries) || ylims!(ae.axis, ylim...)
        end
    end
    return fg.figure
end

function _plot_emission_impl(
        dm::DataModel, obs_name::Symbol, θ, η_vec;
        individuals_idx, time_idx, time_point, time_col, shared_y_axis::Bool,
        style::PlotStyle, kwargs_subplot, kwargs_layout,
        save_path::Union{Nothing, String}, figure_layout::Symbol
    )
    dist_rep = _representative_dist(dm, obs_name, nothing)
    dist_rep isa MVDiscreteTimeDiscreteStatesHMM || throw(
        ArgumentError(
            "plot_emission_distributions requires an MVDiscreteTimeDiscreteStatesHMM " *
                "observable; :$(obs_name) is a $(nameof(typeof(dist_rep)))."
        )
    )
    inds = _plot_individuals(dm, individuals_idx)
    time_col_use = time_col === nothing ? get_time_col(dm) : time_col
    tables = DataFrame[]
    rows = String[]
    n_states = 0
    n_marg = 0
    for i in inds
        obs_rows = get_obs_rows(get_row_groups(dm))[i]
        row = _resolve_emission_row(dm, obs_rows, time_idx, time_point, time_col_use)
        label = string(_individual_panel(dm, i), "\nt = ", get_df(dm)[row, time_col_use])
        df, n_states, n_marg = _hmm_emission_table(dm, obs_name, θ, η_vec[i], i, row)
        df[!, :row] = fill(label, nrow(df))
        push!(tables, df)
        push!(rows, label)
    end
    ylims = nothing
    for df in tables
        ylims = _merge_limits(ylims, df.y)
    end
    ylim = _shared_lim(shared_y_axis, ylims)
    kw = (; shared_y_axis, style, kwargs_subplot, kwargs_layout)
    if figure_layout == :vector
        return Figure[
            _hmm_emission_figure(df, [r], n_states, n_marg, obs_name; ylim, kw...)
                for (df, r) in zip(tables, rows)
        ]
    end
    fig = _hmm_emission_figure(reduce(vcat, tables), rows, n_states, n_marg, obs_name; ylim, kw...)
    return _save_plot!(fig, save_path)
end

function _emission_checks(figure_layout, ncols, time_idx, time_point, save_path, plot_path)
    _hmm_check_layout(figure_layout, save_path, plot_path)
    _check_positive_int(ncols, "ncols")
    (time_idx !== nothing) && (time_point !== nothing) &&
        error("Specify only one of time_idx or time_point.")
    return nothing
end

function _emission_observable(dm, observable)
    obs_name = _hmm_observable(dm, observable, "plot_emission_distributions")
    (is_mv, _) = _obs_multivariate_info(dm, obs_name)
    is_mv || throw(
        ArgumentError("plot_emission_distributions requires a multivariate HMM observable; :$(obs_name) is univariate.")
    )
    return obs_name
end

"""
    plot_emission_distributions(res::FitResult; kwargs...)
    plot_emission_distributions(dm::DataModel; kwargs...)

Plot the per-state emission (observation) distributions of a multivariate
hidden-Markov-model outcome (`MVDiscreteTimeDiscreteStatesHMM`) at one observation time
per individual. The figure is a grid with one row per individual (row label: ID and the
time used) and one column per hidden state; each panel shows the marginal emission
density of every outcome component (lines) or its probability mass function (points for
discrete components), with one figure legend for the components. The x axes are free per
panel; the y axes are linked when `shared_y_axis = true`.

# Keyword Arguments
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
- `observable = nothing`: HMM observable to plot; defaults to the first one. An unknown,
  non-HMM or univariate observable raises an `ArgumentError`.
- `individuals_idx`: indices or IDs of individuals to include, or `nothing` for all.
- `time_idx::Union{Nothing, Int} = nothing`, `time_point = nothing`: observation at which to
  evaluate the emission distributions (by index within each individual, or the observation
  closest to a time value); the first observation by default.
- `time_col::Union{Nothing, Symbol} = nothing`: column used for `time_point` and the row
  label (defaults to the model's time column).
- `shared_y_axis::Bool = true`: common y range for all panels.
- `ncols::Int = 3`: validated for API compatibility; the grid always has one column per
  hidden state.
- `figure_layout::Symbol = :single`: `:vector` returns one `Figure` (one row of state
  panels) per individual.
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `kwargs_subplot`, `kwargs_layout`: Makie `Axis` / `Figure` attributes (user values win).
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot (`:single` only).
- `params`, `constants_re`, `mcmc_draws`, `rng`: parameter overrides and random-effect
  settings as in [`plot_fits`](@ref).
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
    _emission_checks(figure_layout, ncols, time_idx, time_point, save_path, plot_path)
    dm = _get_dm(res, dm)
    save_path = _resolve_plot_path(save_path, plot_path)
    constants_re_use = _res_constants_re(res, constants_re)
    obs_name = _emission_observable(dm, observable)
    _is_posterior_draw_fit(res) &&
        error("plot_emission_distributions does not support posterior draws.")

    θ = get_params(res; scale = :untransformed)
    θ = _apply_param_overrides(θ, params)
    η_vec = _default_random_effects(res, dm, constants_re_use, θ, rng, mcmc_draws)
    return _plot_emission_impl(
        dm, obs_name, θ, η_vec;
        individuals_idx, time_idx, time_point, time_col, shared_y_axis, style,
        kwargs_subplot, kwargs_layout, save_path, figure_layout,
    )
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
    _emission_checks(figure_layout, ncols, time_idx, time_point, save_path, plot_path)
    save_path = _resolve_plot_path(save_path, plot_path)
    obs_name = _emission_observable(dm, observable)

    θ = get_θ0_untransformed(get_fixed(get_model(dm)))
    θ = _apply_param_overrides(θ, params)
    η_vec = _default_random_effects_from_dm(dm, constants_re, θ)
    return _plot_emission_impl(
        dm, obs_name, θ, η_vec;
        individuals_idx, time_idx, time_point, time_col, shared_y_axis, style,
        kwargs_subplot, kwargs_layout, save_path, figure_layout,
    )
end
