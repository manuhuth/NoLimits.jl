using NoLimits: _SavedFittingMethod

# --- Public plotting functions ---------------------------------------------------------

"""
    plot_multistart_waterfall(res::MultistartFitResult; style, kwargs_subplot, save_path)
    -> Makie.Figure

Plot the objective values of all successful multistart runs in ascending order
(waterfall plot), highlighting the best run.

# Keyword Arguments
- `style::PlotStyle = PlotStyle()`: visual style configuration.
- `kwargs_subplot`: additional Makie `Axis` attributes forwarded to the subplot.
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot, or `nothing`.
"""
function plot_multistart_waterfall(
        res::MultistartFitResult;
        style::PlotStyle = PlotStyle(),
        kwargs_subplot = NamedTuple(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing
    )
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
            @warn "Skipping successful multistart run with non-finite objective value." rank = rank objective = obj
        end
    end

    isempty(objectives) &&
        error("No finite objective values available among successful multistart runs.")

    # Converged starts often agree in the first 10+ digits, where Makie's tick labels all
    # read the same: then plot the offsets from the best objective in units of a power of
    # ten (short tick labels) and state the best value in the subtitle.
    ys, ylabel, best_note = _ms_waterfall_y(objectives)
    n_failed = length(res.errors_err)
    post = Any[]
    if n_failed > 0
        post = Any[
            ax -> text!(
                ax, [float(maximum(ranks))], [maximum(ys)];
                text = ["Failed starts omitted: $(n_failed)"], align = (:right, :center),
                fontsize = style.font_size_annotation
            ),
        ]
    end
    df = _tidy()
    _push_xy!(df, "", ranks, ys, "objective")
    subtitle = best_note === nothing ? "best → worst; successful starts only" :
        "best → worst; successful starts only\n$(best_note)"
    panel = AoGPanel(
        AoG.data(df) * AoG.mapping(:x, :y) * _scatter_visual(style; color = style.color_primary);
        axis = _aog_axis(
            style, kwargs_subplot;
            title = "Multistart Objectives", subtitle = subtitle,
            xlabel = "Rank", xticks = collect(1:length(objectives)), ylabel = ylabel,
        ),
        post = post, legend = false
    )
    fig = combine_plots([panel]; ncols = 1, style = style)
    return _save_plot!(fig, save_path)
end

# y values, y label and subtitle note of the waterfall. Objectives that differ only far
# behind the leading digits (spread <= 1e-6 relative) are shown as offsets from the best
# value in units of 10^k (k from the spread), so the tick labels stay short:
# ylabel "Objective − best (×10⁻¹⁴)" (Makie rich text), note "best = 30.9437636397".
# Otherwise the raw objectives, ylabel "Objective" and no note.
function _ms_waterfall_y(objectives::AbstractVector{<:Real})
    best = minimum(objectives)
    spread = maximum(objectives) - best
    (spread > 0 && spread <= 1.0e-6 * max(abs(best), 1.0)) ||
        return collect(Float64, objectives), "Objective", nothing
    k = floor(Int, log10(spread))
    ys = (objectives .- best) ./ exp10(k)
    ylabel = Makie.rich("Objective − best (×10", Makie.superscript(replace(string(k), "-" => "−")), ")")
    return ys, ylabel, "best = $(round(best; sigdigits = 12))"
end

"""
    plot_multistart_fixed_effect_variability(res::MultistartFitResult; dm, k_best, mode,
                                             quantiles, scale, include_parameters,
                                             exclude_parameters, style, kwargs_subplot,
                                             save_path) -> Makie.Figure

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
- `kwargs_subplot`: additional Makie `Axis` attributes forwarded to the subplot (a `size`
  entry, if given, sets the figure size instead).
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
"""
function plot_multistart_fixed_effect_variability(
        res::MultistartFitResult;
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
        plot_path::Union{Nothing, String} = nothing
    )
    save_path = _resolve_plot_path(save_path, plot_path)
    mode in (:points, :quantiles) || error("mode must be :points or :quantiles.")
    scale in (:untransformed, :transformed) ||
        error("scale must be :untransformed or :transformed.")
    k_best >= 1 || error("k_best must be >= 1.")

    dm_use = _multistart_data_model(res, dm)
    fe = get_fixed(get_model(dm_use))
    fe_names = get_names(fe) # declaration order
    fe_params = get_params(fe)

    include_set = _normalize_top_level_parameter_selection(
        include_parameters, "include_parameters"
    )
    exclude_set = _normalize_top_level_parameter_selection(
        exclude_parameters, "exclude_parameters"
    )
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
        @warn "k_best exceeds successful multistart runs; clipping to available runs." k_best = k_best available = n_ok
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
            pname, getproperty(θ_list[1], pname)
        )
        nrows = length(vals_ref)
        nrows >= 1 || continue

        block_vals = Matrix{Float64}(undef, nrows, k_use)
        block_vals[:, 1] .= vals_ref
        for col in 2:k_use
            lbls_cur, vals_cur = _flatten_param_with_labels(
                pname, getproperty(θ_list[col], pname)
            )
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
        @warn "Dropping non-finite parameter coordinates from variability plot." dropped = (
            size(
                values, 1
            ) - length(keep_rows)
        )
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

    y = collect(1.0:length(labels))
    plot_height = clamp(200 + 22 * length(labels), MIN_FIGURE_HEIGHT, MAX_FIGURE_HEIGHT)
    subplot_kwargs = _nt(kwargs_subplot)
    fig_size = get(subplot_kwargs, :size, (900, plot_height))
    axis_kwargs = Base.structdiff(subplot_kwargs, NamedTuple{(:size,)})

    pts = _tidy()
    seg = _tidy(; grp = String)
    series = Pair{String, Any}[]
    if mode == :points
        for i in eachindex(y)
            _push_xy!(pts, "", vec(z[i, :]), fill(y[i], k_use), "estimate")
        end
        layers = _layer(pts, AoG.mapping(:x, :y), _scatter_visual(style; color = style.color_primary))
        scales = NamedTuple()
        show_legend = false
    else
        q = sort(Float64.(collect(quantiles)))
        (length(q) == 3 && all(0 .<= q .<= 1)) ||
            error("quantiles must contain three probabilities in [0, 1].")
        range_label = "$(_ms_fmt_q(q[1]))–$(_ms_fmt_q(q[3])) quantile range"
        mid_label = "$(_ms_fmt_q(q[2])) quantile"
        for i in eachindex(y)
            zi = vec(z[i, :])
            lo, mid, hi = quantile(zi, q[1]), quantile(zi, q[2]), quantile(zi, q[3])
            for xv in (lo, hi)
                push!(seg, (panel = "", x = xv, y = y[i], series = range_label, grp = string(i)))
            end
            _push_xy!(pts, "", [mid], [y[i]], mid_label)
        end
        xy = AoG.mapping(:x, :y; color = :series)
        layers = _layers(
            _layer(seg, AoG.mapping(:x, :y; color = :series, group = :grp), _line_visual(style)),
            _layer(pts, xy, _scatter_visual(style)),
        )
        scales = (;
            Color = _series_palette(
                [range_label => style.color_primary, mid_label => style.color_secondary]
            ),
        )
        show_legend = true
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
    function zero_line!(ax)
        vl = vlines!(ax, 0.0; color = (style.color_dark, 0.7), linestyle = :dash)
        Makie.translate!(vl, 0, 0, -1)
        return ax
    end
    post = Any[zero_line!]
    haskey(axis_kwargs, :limits) ||
        push!(post, ax -> Makie.limits!(ax, (zmin, zmax), (0.5, length(labels) + 0.5)))
    panel = AoGPanel(
        layers;
        scales = scales,
        axis = _aog_axis(
            style, axis_kwargs;
            title = "Fixed-Effect Variability Across Top-$(k_use) Multistarts",
            xlabel = "Z-score", ylabel = "Parameter",
            yticks = (y, labels), yreversed = true,
        ),
        post = post, legend = show_legend
    )
    fig = combine_plots([panel]; ncols = 1, style = style, size = fig_size)
    return _save_plot!(fig, save_path)
end

# Probability as a short percentage label (0.1 -> "10%", 0.025 -> "2.5%").
_ms_fmt_q(p) = (v = round(100 * p; digits = 2); (isinteger(v) ? string(Int(v)) : string(v)) * "%")

# EM method kind of a fit (`:saem`/`:mcem`), also after `load_fit` (whose method is a
# `_SavedFittingMethod`); `nothing` for any other method.
_em_kind(::SAEM) = :saem
_em_kind(::MCEM) = :mcem
_em_kind(m::_SavedFittingMethod) = m.kind in (:saem, :mcem) ? m.kind : nothing
_em_kind(::Any) = nothing

# Stride of the stored parameter history (`θ_hist[k]` belongs to iteration
# `k * diagnostics_every`). Taken from the method, else from the notes (fits since #383
# record it), else inferred from the history lengths of a reloaded fit; `nothing` when the
# lengths admit more than one stride.
function _em_diagnostics_every(method, notes, n_q::Int, n_θ::Int)
    method isa SAEM && return method.saem.diagnostics_every
    method isa MCEM && return method.diagnostics_every
    hasproperty(notes, :diagnostics_every) && return Int(getproperty(notes, :diagnostics_every))
    n_θ == n_q && return 1
    cands = [e for e in 1:max(n_q, 1) if fld(n_q, e) == n_θ]
    return length(cands) == 1 ? only(cands) : nothing
end

"""
    plot_em_trajectories(res::FitResult; dm, scale, include_parameters, exclude_parameters,
                         ncols, style, kwargs_subplot, kwargs_layout, save_path)
    -> Makie.Figure

Plot the Q-function and fixed-effect parameter trajectories over EM iterations for a
[`SAEM`](@ref) or [`MCEM`](@ref) fit result, also after a [`save_fit`](@ref NoLimits.save_fit) /
[`load_fit`](@ref NoLimits.load_fit) round trip (the stored trajectories are part of the saved fit).

Requires the fit to have been run with `store_diagnostics=true`. The first panel shows
the Q-function over all iterations; subsequent panels show one trace per scalar
parameter element (e.g. `β[1]`, `β[2]`, `Ω[1,1]`, ...), at the iterations where it was
stored (every `diagnostics_every`-th). Panels are AlgebraOfGraphics facets with a shared
(linked) iteration axis and free y axes.

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
- `kwargs_subplot`: additional Makie `Axis` attributes forwarded to every panel.
- `kwargs_layout`: additional Makie `Figure` attributes forwarded to the figure.
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
"""
function plot_em_trajectories(
        res::FitResult;
        dm::Union{Nothing, DataModel} = nothing,
        scale::Symbol = :untransformed,
        include_parameters = nothing,
        exclude_parameters = nothing,
        ncols::Int = DEFAULT_PLOT_COLS,
        style::PlotStyle = PlotStyle(),
        kwargs_subplot = NamedTuple(),
        kwargs_layout = NamedTuple(),
        save_path::Union{Nothing, String} = nothing,
        plot_path::Union{Nothing, String} = nothing
    )
    save_path = _resolve_plot_path(save_path, plot_path)
    _check_positive_int(ncols, "ncols")
    scale in (:untransformed, :transformed) ||
        error("scale must be :untransformed or :transformed. Got: $(scale).")

    method = get_method(res)
    _em_kind(method) === nothing &&
        error(
        "plot_em_trajectories requires a SAEM or MCEM fit result. Got: " *
            (method isa _SavedFittingMethod ? "a reloaded :$(method.kind) fit." : "$(typeof(method)).")
    )

    notes = get_notes(res)
    diag = hasproperty(notes, :diagnostics) ? getproperty(notes, :diagnostics) : nothing
    (diag === nothing || !hasproperty(diag, :θ_hist) || !hasproperty(diag, :Q_hist)) &&
        error(
        "No EM trajectory stored in this fit result (its notes carry no diagnostics). " *
            "Re-fit with store_diagnostics=true."
    )
    isempty(diag.θ_hist) &&
        error("No parameter trajectory stored. Re-fit with store_diagnostics=true.")

    n_q = length(diag.Q_hist)
    every = _em_diagnostics_every(method, notes, n_q, length(diag.θ_hist))
    stored_x = every === nothing ? collect(1:length(diag.θ_hist)) :
        [k * every for k in 1:length(diag.θ_hist)]

    # Resolve DataModel and build inverse transform if needed.
    if scale == :untransformed
        dm_use = dm !== nothing ? dm : get_data_model(res)
        dm_use === nothing &&
            error(
            "scale=:untransformed requires the DataModel. Re-fit with " *
                "store_data_model=true (the default), or pass dm=... explicitly."
        )
        full_inv = get_inverse_transform(get_fixed(get_model(dm_use)))
        hist_names = collect(keys(diag.θ_hist[1]))
        name_to_idx = Dict(full_inv.names[i] => i for i in eachindex(full_inv.names))
        valid_idxs = [name_to_idx[n] for n in hist_names if haskey(name_to_idx, n)]
        restricted_inv = InverseTransform(
            full_inv.names[valid_idxs],
            full_inv.specs[valid_idxs]
        )
        θ_vals = [restricted_inv(θ_t) for θ_t in diag.θ_hist]

        # Determine parameter declaration order from the DataModel.
        all_fe_names = get_names(get_fixed(get_model(dm_use)))
        free_fe_names = [n for n in all_fe_names if n in Set(hist_names)]
    else
        θ_vals = diag.θ_hist
        free_fe_names = collect(keys(diag.θ_hist[1]))
    end

    # Apply include / exclude filters.
    include_set = _normalize_top_level_parameter_selection(
        include_parameters,
        "include_parameters"
    )
    exclude_set = _normalize_top_level_parameter_selection(
        exclude_parameters,
        "exclude_parameters"
    )
    known = Set(free_fe_names)
    if include_set !== nothing
        unknown = [n for n in include_set if !(n in known)]
        isempty(unknown) ||
            error(
            "Unknown include_parameters: $(unknown). " *
                "Available free parameters: $(free_fe_names)."
        )
    end
    if exclude_set !== nothing
        unknown = [n for n in exclude_set if !(n in known)]
        isempty(unknown) ||
            error(
            "Unknown exclude_parameters: $(unknown). " *
                "Available free parameters: $(free_fe_names)."
        )
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

    # Tidy table: the Q function over all iterations, then one panel per scalar element
    # of each selected parameter block (declaration order).
    df = _tidy()
    panels = String["Q Function"]
    _push_xy!(df, "Q Function", 1:n_q, diag.Q_hist, "trajectory")
    for pname in selected_order
        first_val = getproperty(θ_vals[1], pname)
        elems = if first_val isa Number
            [(string(pname), θ -> getproperty(θ, pname))]
        elseif first_val isa AbstractVector
            n_elem = length(first_val)
            [
                (n_elem == 1 ? string(pname) : string(pname, "[", j, "]"), θ -> getproperty(θ, pname)[j])
                    for j in eachindex(first_val)
            ]
        elseif first_val isa AbstractMatrix
            [
                (string(pname, "[", r, ",", c, "]"), θ -> getproperty(θ, pname)[r, c])
                    for r in axes(first_val, 1) for c in axes(first_val, 2)
            ]
        else
            Tuple{String, Any}[]
        end
        for (label, get_el) in elems
            push!(panels, label)
            _push_xy!(df, label, stored_x, [float(get_el(θ)) for θ in θ_vals], "trajectory")
        end
    end

    layers = _layer(
        df, AoG.mapping(:x, :y; layout = :panel),
        _line_visual(style; color = style.color_primary)
    )
    xlabel = every === nothing ?
        "EM iteration (Q function) / stored snapshot (parameters)" : "EM Iteration"
    fig = _draw_facets(
        layers; panels = panels, ncols = ncols, style = style,
        shared_x = every !== nothing, shared_y = false,
        kwargs_subplot = kwargs_subplot, kwargs_layout = kwargs_layout,
        axis = (; xlabel = xlabel, ylabel = "Value"), legend = false
    )
    return _save_plot!(fig, save_path)
end
