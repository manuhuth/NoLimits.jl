function create_styled_plot(;
        title = "", xlabel = "", ylabel = "", style::PlotStyle = PlotStyle(), kwargs...)
    return plot(; title = title, xlabel = xlabel, ylabel = ylabel,
        default_plot_kwargs(style)..., kwargs...)
end

function create_styled_scatter!(p, x, y; label = "", color = COLOR_PRIMARY,
        style::PlotStyle = PlotStyle(), kwargs...)
    return scatter!(p, x, y;
        label = label,
        color = color,
        markersize = style.marker_size,
        markeralpha = style.marker_alpha,
        markerstrokewidth = style.marker_stroke_width,
        kwargs...)
end

function create_styled_line!(p, x, y; label = "", color = COLOR_SECONDARY,
        style::PlotStyle = PlotStyle(), kwargs...)
    return plot!(p, x, y;
        label = label,
        color = color,
        linewidth = style.line_width_primary,
        kwargs...)
end

function add_reference_line!(
        p, value; orientation = :horizontal, color = COLOR_REFERENCE, kwargs...)
    if orientation == :horizontal
        hline!(p, [value]; color = color, linestyle = :dash, kwargs...)
    else
        vline!(p, [value]; color = color, linestyle = :dash, kwargs...)
    end
    return p
end

function add_annotation!(p, x, y, text; fontsize = 7, halign = :right)
    annotate!(p, x, y, text, halign = halign, fontsize = fontsize)
    return p
end

function combine_plots(plots::Vector; ncols::Int = DEFAULT_PLOT_COLS,
        style::PlotStyle = PlotStyle(), kwargs...)
    nrows = ceil(Int, length(plots) / ncols)
    font_kw = (
        titlefontsize = style.font_size_title,
        guidefontsize = style.font_size_label,
        tickfontsize = style.font_size_tick,
        legendfontsize = style.font_size_legend,
        fontfamily = style.font_family
    )
    auto_kw = merge(font_kw,
        (layout = (nrows, ncols), size = calculate_plot_size(length(plots), ncols, style)))
    return plot(plots...; merge(auto_kw, NamedTuple(kwargs))...)
end

function _apply_shared_axes!(plots::AbstractVector, xlim, ylim)
    for p in plots
        xlim !== nothing && plot!(p; xlims = xlim)
        ylim !== nothing && plot!(p; ylims = ylim)
    end
    return plots
end

function _save_plot!(p, save_path::Union{Nothing, String})
    save_path = _ensure_save_path(save_path)
    save_path === nothing && return p
    _, ext = splitext(save_path)
    if ext == ".png"
        try
            savefig(p, save_path; dpi = DEFAULT_DPI)
        catch err
            if err isa MethodError
                savefig(p, save_path)
            else
                rethrow(err)
            end
        end
    else
        savefig(p, save_path)
    end
    return p
end

"""
    plot_multistart_waterfall(res::MultistartFitResult; style, kwargs_subplot, save_path)
    -> Plots.Plot

Plot the objective values of all successful multistart runs in ascending order
(waterfall plot), highlighting the best run.

# Keyword Arguments
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `kwargs_subplot`: additional keyword arguments forwarded to the subplot.
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot, or `nothing`.
"""
function plot_multistart_waterfall(res::MultistartFitResult;
        style::PlotStyle = PlotStyle(),
        kwargs_subplot = NamedTuple(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing)
    save_path = _resolve_plot_path(save_path, plot_path)

    n_ok = length(res.results_ok)
    n_ok >= 1 || error("No successful multistart runs available for plotting.")

    perm = sortperm(res.scores_ok)
    objectives = Float64[]
    ranks = Int[]
    for (rank, idx) in enumerate(perm)
        obj = get_objective(res.results_ok[idx])
        if isfinite(obj)
            push!(objectives, float(obj))
            push!(ranks, rank)
        else
            @warn "Skipping successful multistart run with non-finite objective value." rank=rank objective=obj
        end
    end

    isempty(objectives) &&
        error("No finite objective values available among successful multistart runs.")

    n_failed = length(res.errors_err)
    p = create_styled_plot(;
        title = "Multistart Objectives (best → worst; successful starts only)",
        xlabel = "Rank",
        ylabel = "Objective",
        style = style,
        kwargs_subplot...
    )
    create_styled_scatter!(
        p, ranks, objectives; label = "", color = style.color_primary, style = style)
    plot!(p; xticks = collect(1:length(objectives)))

    if n_failed > 0
        add_annotation!(
            p, maximum(ranks), maximum(objectives), "Failed starts omitted: $(n_failed)";
            fontsize = style.font_size_annotation)
    end
    return _save_plot!(p, save_path)
end

"""
    plot_multistart_fixed_effect_variability(res::MultistartFitResult; dm, k_best, mode,
                                             quantiles, scale, include_parameters,
                                             exclude_parameters, style, kwargs_subplot,
                                             save_path) -> Plots.Plot

Plot the variation of fixed-effect estimates across the `k_best` multistart runs with
the lowest objective values.

# Keyword Arguments
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
- `k_best::Int = 20`: number of best runs to include.
- `mode::Symbol = :points`: `:points` to show individual estimates; `:quantiles` to show
  quantile bands.
- `quantiles::AbstractVector = [0.1, 0.5, 0.9]`: quantile levels for `:quantiles` mode.
- `scale::Symbol = :untransformed`: `:untransformed` or `:transformed`.
- `include_parameters`, `exclude_parameters`: parameter name filters.
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `kwargs_subplot`: additional keyword arguments forwarded to each subplot.
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
"""
function plot_multistart_fixed_effect_variability(res::MultistartFitResult;
        dm::Union{Nothing, DataModel} = nothing,
        k_best::Int = 20,
        mode::Symbol = :points,
        quantiles::AbstractVector{<:Real} = [0.1, 0.5, 0.9],
        scale::Symbol = :untransformed,
        include_parameters = nothing,
        exclude_parameters = nothing,
        style::PlotStyle = PlotStyle(),
        kwargs_subplot = NamedTuple(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing)
    save_path = _resolve_plot_path(save_path, plot_path)
    mode in (:points, :quantiles) || error("mode must be :points or :quantiles.")
    scale in (:untransformed, :transformed) ||
        error("scale must be :untransformed or :transformed.")
    k_best >= 1 || error("k_best must be >= 1.")

    dm_use = _multistart_data_model(res, dm)
    fe = dm_use.model.fixed.fixed
    fe_names = get_names(fe) # declaration order
    fe_params = get_params(fe)

    include_set = _normalize_top_level_parameter_selection(
        include_parameters, "include_parameters")
    exclude_set = _normalize_top_level_parameter_selection(
        exclude_parameters, "exclude_parameters")
    known = Set(fe_names)

    if include_set !== nothing
        unknown = [n for n in include_set if !(n in known)]
        isempty(unknown) ||
            error("Unknown include_parameters names: $(unknown). Known names: $(fe_names).")
    end
    if exclude_set !== nothing
        unknown = [n for n in exclude_set if !(n in known)]
        isempty(unknown) ||
            error("Unknown exclude_parameters names: $(unknown). Known names: $(fe_names).")
    end

    # Default: only calculate_se=true blocks. include_parameters can add names explicitly.
    selected = Set{Symbol}()
    for n in fe_names
        p = getfield(fe_params, n)
        p.calculate_se && push!(selected, n)
    end
    include_set !== nothing && union!(selected, include_set)
    if exclude_set !== nothing
        for n in exclude_set
            delete!(selected, n)
        end
    end
    selected_order = [n for n in fe_names if n in selected]
    isempty(selected_order) &&
        error("No fixed-effect parameters selected for plotting after include/exclude filters.")

    n_ok = length(res.results_ok)
    n_ok >= 1 || error("No successful multistart runs available for plotting.")
    perm = sortperm(res.scores_ok)
    if k_best > n_ok
        @warn "k_best exceeds successful multistart runs; clipping to available runs." k_best=k_best available=n_ok
    end
    k_use = min(k_best, n_ok)
    keep = perm[1:k_use]

    θ_list = [get_params(res.results_ok[idx]; scale = scale) for idx in keep]

    labels = String[]
    values = Matrix{Float64}(undef, 0, k_use)

    # Build row labels and value matrix in top-level declaration order.
    row_offset = 0
    row_labels = String[]
    for pname in selected_order
        lbls_ref, vals_ref = _flatten_param_with_labels(
            pname, getproperty(θ_list[1], pname))
        nrows = length(vals_ref)
        nrows >= 1 || continue

        block_vals = Matrix{Float64}(undef, nrows, k_use)
        block_vals[:, 1] .= vals_ref
        for col in 2:k_use
            lbls_cur, vals_cur = _flatten_param_with_labels(
                pname, getproperty(θ_list[col], pname))
            lbls_cur == lbls_ref ||
                error("Parameter shape/labels changed across multistarts for $(pname).")
            length(vals_cur) == nrows ||
                error("Parameter length changed across multistarts for $(pname).")
            block_vals[:, col] .= vals_cur
        end

        append!(row_labels, lbls_ref)
        row_offset += nrows
        values = vcat(values, block_vals)
    end

    isempty(row_labels) &&
        error("No plottable fixed-effect coordinates found for selected parameters.")

    # Drop rows with non-finite values across selected starts.
    keep_rows = [i for i in 1:size(values, 1) if all(isfinite, @view values[i, :])]
    if length(keep_rows) < size(values, 1)
        @warn "Dropping non-finite parameter coordinates from variability plot." dropped=(size(
            values, 1) - length(keep_rows))
    end
    isempty(keep_rows) &&
        error("No finite fixed-effect coordinates available for variability plotting.")
    values = values[keep_rows, :]
    labels = row_labels[keep_rows]

    # Standard z-score per coordinate across the selected top-k runs.
    z = similar(values)
    for i in 1:size(values, 1)
        v = @view values[i, :]
        μ = sum(v) / length(v)
        σ = sqrt(sum((v .- μ) .^ 2) / length(v))
        if σ == 0.0 || !isfinite(σ)
            z[i, :] .= 0.0
        else
            z[i, :] .= (v .- μ) ./ σ
        end
    end

    y = collect(1:length(labels))
    plot_height = clamp(200 + 22 * length(labels), MIN_FIGURE_HEIGHT, MAX_FIGURE_HEIGHT)
    local subplot_kwargs = kwargs_subplot
    haskey(subplot_kwargs, :size) ||
        (subplot_kwargs = merge((size = (900, plot_height),), subplot_kwargs))
    haskey(subplot_kwargs, :left_margin) ||
        (subplot_kwargs = merge((left_margin = 18mm,), subplot_kwargs))
    haskey(subplot_kwargs, :legend) ||
        (subplot_kwargs = merge((legend = false,), subplot_kwargs))

    p = create_styled_plot(;
        title = "Fixed-Effect Variability Across Top-$(k_use) Multistarts",
        xlabel = "Z-score",
        ylabel = "Parameter",
        style = style,
        subplot_kwargs...
    )
    add_reference_line!(
        p, 0.0; orientation = :vertical, color = style.color_dark, alpha = 0.7, label = "")

    if mode == :points
        for i in eachindex(y)
            create_styled_scatter!(p, vec(@view z[i, :]), fill(y[i], k_use);
                label = "", color = style.color_primary, style = style)
        end
    else
        q = sort(Float64.(collect(quantiles)))
        (length(q) == 3 && all(0 .<= q .<= 1)) ||
            error("quantiles must contain three probabilities in [0, 1].")
        for i in eachindex(y)
            zi = vec(@view z[i, :])
            lo = quantile(zi, q[1])
            mid = quantile(zi, q[2])
            hi = quantile(zi, q[3])
            create_styled_line!(p, [lo, hi], [y[i], y[i]]; label = "",
                color = style.color_primary, style = style
            )
            create_styled_scatter!(
                p, [mid], [y[i]]; label = "", color = style.color_secondary, style = style)
        end
    end

    all_z = vec(z)
    zmin, zmax = extrema(all_z)
    if zmin == zmax
        zmin -= 1.0
        zmax += 1.0
    else
        pad = 0.05 * (zmax - zmin)
        zmin -= pad
        zmax += pad
    end
    plot!(p; yticks = (y, labels), ylims = (0.5, length(labels) + 0.5),
        yflip = true, xlims = (zmin, zmax))
    return _save_plot!(p, save_path)
end

"""
    plot_em_trajectories(res::FitResult; dm, scale, include_parameters, exclude_parameters,
                         ncols, style, kwargs_subplot, kwargs_layout, save_path) -> Plots.Plot

Plot the Q-function and fixed-effect parameter trajectories over EM iterations for a
[`SAEM`](@ref) or [`MCEM`](@ref) fit result.

Requires the fit to have been run with `store_diagnostics=true`. The first panel shows
the Q-function over all iterations; subsequent panels show one trace per scalar
parameter element (e.g. `β[1]`, `β[2]`, `Ω[1,1]`, ...).

# Keyword Arguments
- `dm::Union{Nothing, DataModel} = nothing`: data model for the inverse transform when
  `scale=:untransformed`. Inferred from `res` when stored.
- `scale::Symbol = :untransformed`: `:untransformed` (natural scale) or `:transformed`
  (optimizer scale). Affects the plotted values; `:untransformed` requires the DataModel.
- `include_parameters`: `Symbol`, `String`, or vector thereof. If set, only the named
  top-level parameter blocks are plotted. Defaults to `nothing` (all free parameters).
- `exclude_parameters`: `Symbol`, `String`, or vector thereof. If set, the named blocks
  are removed from the selection. Applied after `include_parameters`.
- `ncols::Int = $(DEFAULT_PLOT_COLS)`: number of subplot columns.
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `kwargs_subplot`: additional keyword arguments forwarded to each individual subplot.
- `kwargs_layout`: additional keyword arguments forwarded to the combined layout.
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
"""
function plot_em_trajectories(res::FitResult;
        dm::Union{Nothing, DataModel} = nothing,
        scale::Symbol = :untransformed,
        include_parameters = nothing,
        exclude_parameters = nothing,
        ncols::Int = DEFAULT_PLOT_COLS,
        style::PlotStyle = PlotStyle(),
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing)
    save_path = _resolve_plot_path(save_path, plot_path)
    scale in (:untransformed, :transformed) ||
        error("scale must be :untransformed or :transformed. Got: $(scale).")

    method = get_method(res)
    method isa Union{SAEM, MCEM} ||
        error("plot_em_trajectories requires a SAEM or MCEM fit result. Got: $(typeof(method)).")

    diag = get_notes(res).diagnostics
    isempty(diag.θ_hist) &&
        error("No parameter trajectory stored. Re-fit with store_diagnostics=true.")

    diagnostics_every = method isa SAEM ? method.saem.diagnostics_every :
                        method.diagnostics_every
    stored_iters = [k * diagnostics_every for k in 1:length(diag.θ_hist)]

    # Resolve DataModel and build inverse transform if needed.
    if scale == :untransformed
        dm_use = dm !== nothing ? dm : get_data_model(res)
        dm_use === nothing &&
            error("scale=:untransformed requires the DataModel. Re-fit with " *
                  "store_data_model=true (the default), or pass dm=... explicitly.")
        full_inv = get_inverse_transform(dm_use.model.fixed.fixed)
        hist_names = collect(keys(diag.θ_hist[1]))
        name_to_idx = Dict(full_inv.names[i] => i for i in eachindex(full_inv.names))
        valid_idxs = [name_to_idx[n] for n in hist_names if haskey(name_to_idx, n)]
        restricted_inv = InverseTransform(full_inv.names[valid_idxs],
            full_inv.specs[valid_idxs])
        θ_vals = [restricted_inv(θ_t) for θ_t in diag.θ_hist]

        # Determine parameter declaration order from the DataModel.
        all_fe_names = get_names(dm_use.model.fixed.fixed)
        free_fe_names = [n for n in all_fe_names if n in Set(hist_names)]
    else
        θ_vals = diag.θ_hist
        free_fe_names = collect(keys(diag.θ_hist[1]))
    end

    # Apply include / exclude filters.
    include_set = _normalize_top_level_parameter_selection(include_parameters,
        "include_parameters")
    exclude_set = _normalize_top_level_parameter_selection(exclude_parameters,
        "exclude_parameters")
    known = Set(free_fe_names)
    if include_set !== nothing
        unknown = [n for n in include_set if !(n in known)]
        isempty(unknown) ||
            error("Unknown include_parameters: $(unknown). " *
                  "Available free parameters: $(free_fe_names).")
    end
    if exclude_set !== nothing
        unknown = [n for n in exclude_set if !(n in known)]
        isempty(unknown) ||
            error("Unknown exclude_parameters: $(unknown). " *
                  "Available free parameters: $(free_fe_names).")
    end
    selected_order = if include_set !== nothing
        [n for n in free_fe_names if n in include_set]
    else
        copy(free_fe_names)
    end
    if exclude_set !== nothing
        filter!(n -> !(n in exclude_set), selected_order)
    end
    isempty(selected_order) &&
        error("No parameters selected for plotting after include/exclude filters.")

    x_label = diagnostics_every == 1 ? "EM Iteration" :
              "EM Iteration (×$(diagnostics_every))"

    plots = Plots.Plot[]

    # Q-function panel (always uses all iterations, not just stored ones).
    # Plot index 1 is always in column 1, so it gets the ylabel.
    q_plot = create_styled_plot(; title = "Q Function", xlabel = "EM Iteration",
        ylabel = "Value", style = style, kwargs_subplot...)
    create_styled_line!(q_plot, collect(1:length(diag.Q_hist)), Float64.(diag.Q_hist);
        label = "", color = style.color_primary, style = style)
    push!(plots, q_plot)

    # One panel per scalar element of each selected parameter block.
    for pname in selected_order
        first_val = getproperty(θ_vals[1], pname)
        if first_val isa Number
            traj = Float64[float(getproperty(θ, pname)) for θ in θ_vals]
            push!(plots,
                create_styled_plot(; title = string(pname), xlabel = x_label,
                    ylabel = "", style = style, kwargs_subplot...))
            create_styled_line!(plots[end], stored_iters, traj;
                label = "", color = style.color_primary, style = style)
        elseif first_val isa AbstractVector
            n_elem = length(first_val)
            for j in eachindex(first_val)
                traj = Float64[float(getproperty(θ, pname)[j]) for θ in θ_vals]
                label = n_elem == 1 ? string(pname) : string(pname, "[", j, "]")
                push!(plots,
                    create_styled_plot(; title = label, xlabel = x_label,
                        ylabel = "", style = style, kwargs_subplot...))
                create_styled_line!(plots[end], stored_iters, traj;
                    label = "", color = style.color_primary, style = style)
            end
        elseif first_val isa AbstractMatrix
            for r in axes(first_val, 1)
                for c in axes(first_val, 2)
                    traj = Float64[float(getproperty(θ, pname)[r, c]) for θ in θ_vals]
                    label = string(pname, "[", r, ",", c, "]")
                    push!(plots,
                        create_styled_plot(; title = label, xlabel = x_label,
                            ylabel = "", style = style, kwargs_subplot...))
                    create_styled_line!(plots[end], stored_iters, traj;
                        label = "", color = style.color_primary, style = style)
                end
            end
        end
    end

    # Apply "Value" ylabel only to panels in the first column.
    for i in eachindex(plots)
        if (i - 1) % ncols == 0
            plot!(plots[i]; ylabel = "Value")
        end
    end

    result = combine_plots(plots; ncols = ncols, style = style, kwargs_layout...)
    return _save_plot!(result, save_path)
end
