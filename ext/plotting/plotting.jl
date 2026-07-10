# Makie panel model + helper layer shared by every drawing file in this extension.
#
# A `MakiePanel` records the intent of a subplot (axis attributes + a list of drawing
# closures + limits/legend state) without touching Makie until `combine_plots`
# materializes the whole figure at once. Downstream drawing files build panels through
# the `create_styled_*!` / `add_*!` helpers and never call `Axis`/`Figure` directly.

mutable struct MakiePanel
    axis_kwargs::Vector{Pair{Symbol, Any}}  # title/xlabel/ylabel + user Axis kwargs (later wins)
    commands::Vector{Any}                    # callables: ax::Axis -> plot object
    xlim::Union{Nothing, Tuple{Float64, Float64}}
    ylim::Union{Nothing, Tuple{Float64, Float64}}
    legend_position::Union{Symbol, Nothing}  # nothing => auto (:rt) when has_labels; :none => suppress
    has_labels::Bool
end

# A nested grid of panels placed into a single figure cell (used by grouped layouts
# such as plot_emission_distributions).
struct MakiePanelGroup
    panels::Vector{MakiePanel}
    ncols::Int
end

# --- Panel construction / mutation -----------------------------------------------------

"""
    create_styled_plot(; title, xlabel, ylabel, style, kwargs...) -> MakiePanel

Start a new subplot. `kwargs...` are Makie `Axis` attributes (this is where a caller's
`kwargs_subplot` flows). Nothing is drawn until [`combine_plots`](@ref) materializes it.
"""
function create_styled_plot(; title = "", xlabel = "", ylabel = "",
        style::PlotStyle = PlotStyle(), kwargs...)
    axis_kwargs = Pair{Symbol, Any}[:title => title, :xlabel => xlabel, :ylabel => ylabel]
    for (k, v) in kwargs
        push!(axis_kwargs, k => v)
    end
    return MakiePanel(axis_kwargs, Any[], nothing, nothing, nothing, false)
end

# Push a drawing closure `f(ax::Axis)` onto the panel.
_record!(p::MakiePanel, f) = (push!(p.commands, f); p)

"""
    _axis_attrs!(p; kwargs...) -> MakiePanel

Append Makie `Axis` attributes mid-build (replaces the old `plot!(p; ...)` for axis
tweaks). Later entries win over earlier ones. Translate Plots names when calling:
`yflip=true` -> `yreversed=true`; `yticks`/`xticks` pass through unchanged as
`(positions, labels)` or a positions vector.
"""
function _axis_attrs!(p::MakiePanel; kwargs...)
    for (k, v) in kwargs
        push!(p.axis_kwargs, k => v)
    end
    return p
end

# Set axis limits (materialized via xlims!/ylims!). Also see `_apply_shared_axes!`.
function _set_limits!(p::MakiePanel; xlim = nothing, ylim = nothing)
    xlim !== nothing && (p.xlim = xlim)
    ylim !== nothing && (p.ylim = ylim)
    return p
end

"""
    _label(p, label) -> Union{String, Nothing}

Normalize a legend label. `""`/`nothing` -> `nothing` (no legend entry); a non-empty
String flips `p.has_labels` and is returned. ALL label handling routes through here so a
figure only gains a legend when something is actually labeled.
"""
function _label(p::MakiePanel, label)
    (label === nothing || (label isa AbstractString && isempty(label))) && return nothing
    p.has_labels = true
    return String(label)
end

# --- Styled drawing primitives ---------------------------------------------------------

"""
    create_styled_scatter!(p, x, y; label, color, style, kwargs...) -> MakiePanel

Record a styled `scatter!`. A scalar `color` is drawn with the style's marker alpha; a
vector/tuple `color` (e.g. a per-point colormap) is used verbatim. Any Makie scatter
attribute passed via `kwargs...` (`markersize`, `marker`, `colormap`, `colorrange`, a
`color` override, ...) wins over the styled defaults.
"""
function create_styled_scatter!(p::MakiePanel, x, y; label = "", color = COLOR_PRIMARY,
        style::PlotStyle = PlotStyle(), kwargs...)
    lbl = _label(p, label)
    kw = NamedTuple(kwargs)
    col = haskey(kw, :color) ? kw.color : color
    styled_col = col isa Union{AbstractVector, Tuple} ? col : (col, style.marker_alpha)
    defaults = (; color = styled_col, markersize = style.marker_size,
        strokewidth = style.marker_stroke_width, strokecolor = :black)
    attrs = merge(defaults, Base.structdiff(kw, NamedTuple{(:color,)}))
    _record!(p, ax -> scatter!(ax, x, y; label = lbl, attrs...))
    return p
end

"""
    create_styled_line!(p, x, y; label, color, style, kwargs...) -> MakiePanel

Record a styled `lines!`. `kwargs...` (`linewidth`, `linestyle`, a `color` override, ...)
win over the styled defaults.
"""
function create_styled_line!(p::MakiePanel, x, y; label = "", color = COLOR_SECONDARY,
        style::PlotStyle = PlotStyle(), kwargs...)
    lbl = _label(p, label)
    defaults = (; color = color, linewidth = style.line_width_primary)
    attrs = merge(defaults, NamedTuple(kwargs))
    _record!(p, ax -> lines!(ax, x, y; label = lbl, attrs...))
    return p
end

"""
    add_reference_line!(p, value; orientation, color, kwargs...) -> MakiePanel

Record a dashed reference line (`hlines!` for `:horizontal`, `vlines!` for `:vertical`).
A `label` kwarg is routed through [`_label`](@ref); other kwargs pass through.
"""
function add_reference_line!(p::MakiePanel, value; orientation = :horizontal,
        color = COLOR_REFERENCE, kwargs...)
    kw = NamedTuple(kwargs)
    lbl = haskey(kw, :label) ? _label(p, kw.label) : nothing
    rest = Base.structdiff(kw, NamedTuple{(:label,)})
    draw = orientation == :horizontal ? hlines! : vlines!
    _record!(p,
        ax -> draw(ax, value; color = color, linestyle = :dash, label = lbl, rest...))
    return p
end

"""
    add_annotation!(p, x, y, txt; fontsize, halign) -> MakiePanel

Record a text annotation at data coordinates `(x, y)`. `halign` (`:right`/`:left`/
`:center`) is the horizontal text alignment.
"""
function add_annotation!(p::MakiePanel, x, y, txt; fontsize = 7, halign = :right)
    _record!(p,
        ax -> text!(ax, [float(x)], [float(y)];
            text = [string(txt)], align = (halign, :center), fontsize = fontsize))
    return p
end

"""
    _hist!(p, vals; bins, normalization, color, label, style, kwargs...) -> MakiePanel

Draw a histogram as a touching `barplot!` from uniform bins (via `_histogram_xy`). This
is the mechanical replacement for the old `histogram!(p, vals; bins, normalize, fillcolor,
linecolor)` call. `normalization ∈ (:probability, :pdf, :none)`. Map old `fillcolor` to
`color` and old `linecolor` to `strokecolor` (passed via `kwargs...`).
"""
function _hist!(p::MakiePanel, vals; bins::Int = 30, normalization::Symbol = :probability,
        color = COLOR_PRIMARY, label = "", style::PlotStyle = PlotStyle(), kwargs...)
    lbl = _label(p, label)
    h = _histogram_xy(collect(float.(vals)); bins = bins, normalization = normalization)
    defaults = (; color = color, strokewidth = 0, gap = 0, width = h.width)
    attrs = merge(defaults, NamedTuple(kwargs))
    _record!(p, ax -> barplot!(ax, h.centers, h.heights; label = lbl, attrs...))
    return p
end

# --- Materialization -------------------------------------------------------------------

# Merge the style defaults with the panel's accumulated axis attributes so that later
# panel entries win over both the defaults and earlier entries (Makie does not dedup
# duplicate keyword arguments, so we dedup here).
function _merged_axis_kwargs(style::PlotStyle, panel_kwargs::Vector{Pair{Symbol, Any}})
    merged = Dict{Symbol, Any}()
    for (k, v) in Base.pairs(default_axis_kwargs(style))
        merged[k] = v
    end
    for (k, v) in panel_kwargs
        merged[k] = v
    end
    return merged
end

# Build one Axis at grid position `pos`, run its drawing commands, apply limits, and add
# a legend when the panel has labeled series and legends are not suppressed.
function _materialize!(pos, p::MakiePanel, style::PlotStyle)
    ax = Axis(pos; _merged_axis_kwargs(style, p.axis_kwargs)...)
    for cmd in p.commands
        cmd(ax)
    end
    p.xlim !== nothing && xlims!(ax, p.xlim...)
    p.ylim !== nothing && ylims!(ax, p.ylim...)
    if p.has_labels && p.legend_position !== :none
        # has_labels guards the "no labeled plots" case Makie errors on; the try/catch is
        # a defensive backstop (e.g. only reference lines were labeled).
        try
            axislegend(ax; position = something(p.legend_position, :rt),
                unique = true, labelsize = style.font_size_legend)
        catch
        end
    end
    return ax
end

# Materialize a nested panel group as a sub-GridLayout in a single figure cell.
function _materialize!(pos, g::MakiePanelGroup, style::PlotStyle)
    gl = GridLayout(pos)
    ncols_use = min(g.ncols, max(length(g.panels), 1))
    for (i, panel) in enumerate(g.panels)
        row, col = fldmod1(i, ncols_use)
        _materialize!(gl[row, col], panel, style)
    end
    return gl
end

"""
    combine_plots(panels::Vector; ncols, style, kwargs...) -> Makie.Figure

The single materialization point: lay `panels` (MakiePanel and/or MakiePanelGroup) into a
`Figure` on a `ncols`-wide grid and return it. `kwargs...` are Makie `Figure` attributes
(this is where a caller's `kwargs_layout` flows); a caller `size`/`figure_padding`/... wins
over the computed defaults.
"""
function combine_plots(panels::Vector; ncols::Int = DEFAULT_PLOT_COLS,
        style::PlotStyle = PlotStyle(), kwargs...)
    n = length(panels)
    ncols_use = min(ncols, max(n, 1))
    fig_defaults = (; size = calculate_plot_size(max(n, 1), ncols_use, style),
        fonts = (; regular = style.font_family, bold = style.font_family),
        figure_padding = style.figure_padding)
    fig = Figure(; merge(fig_defaults, NamedTuple(kwargs))...)
    for (i, panel) in enumerate(panels)
        row, col = fldmod1(i, ncols_use)
        _materialize!(fig[row, col], panel, style)
    end
    return fig
end

# Set shared x/y limits across panels (recursing into groups) before materialization.
function _apply_shared_axes!(panels::AbstractVector, xlim, ylim)
    for panel in panels
        _apply_shared_axes_one!(panel, xlim, ylim)
    end
    return panels
end

function _apply_shared_axes_one!(p::MakiePanel, xlim, ylim)
    _set_limits!(p; xlim = xlim, ylim = ylim)
end

function _apply_shared_axes_one!(g::MakiePanelGroup, xlim, ylim)
    for p in g.panels
        _set_limits!(p; xlim = xlim, ylim = ylim)
    end
    return g
end

"""
    _save_plot!(fig::Figure, save_path) -> Figure

Save `fig` to `save_path` (a `.png` uses `px_per_unit = DEFAULT_DPI/100`). Saving needs a
rasterizing Makie backend loaded (e.g. `using CairoMakie`); a clearer error is raised if
none is available.
"""
function _save_plot!(fig::Figure, save_path::Union{Nothing, String})
    save_path = _ensure_save_path(save_path)
    save_path === nothing && return fig
    _, ext = splitext(save_path)
    try
        if ext == ".png"
            Makie.save(save_path, fig; px_per_unit = DEFAULT_DPI / 100)
        else
            Makie.save(save_path, fig)
        end
    catch err
        error("Saving figures requires a rasterizing Makie backend. Load one first, " *
              "e.g. `using CairoMakie`. Underlying error: $(err)")
    end
    return fig
end

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
    _axis_attrs!(p; xticks = collect(1:length(objectives)))

    if n_failed > 0
        add_annotation!(
            p, maximum(ranks), maximum(objectives), "Failed starts omitted: $(n_failed)";
            fontsize = style.font_size_annotation)
    end
    fig = combine_plots([p]; ncols = 1, style = style)
    return _save_plot!(fig, save_path)
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
    subplot_kwargs = NamedTuple(kwargs_subplot)
    fig_size = get(subplot_kwargs, :size, (900, plot_height))
    axis_kwargs = Base.structdiff(subplot_kwargs, NamedTuple{(:size,)})

    p = create_styled_plot(;
        title = "Fixed-Effect Variability Across Top-$(k_use) Multistarts",
        xlabel = "Z-score",
        ylabel = "Parameter",
        style = style,
        axis_kwargs...
    )
    p.legend_position = :none
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
    _axis_attrs!(p; yticks = (y, labels), yreversed = true)
    _set_limits!(p; xlim = (zmin, zmax), ylim = (0.5, length(labels) + 0.5))
    fig = combine_plots([p]; ncols = 1, style = style, size = fig_size)
    return _save_plot!(fig, save_path)
end

"""
    plot_em_trajectories(res::FitResult; dm, scale, include_parameters, exclude_parameters,
                         ncols, style, kwargs_subplot, kwargs_layout, save_path)
    -> Makie.Figure

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
- `kwargs_subplot`: additional Makie `Axis` attributes forwarded to each individual subplot.
- `kwargs_layout`: additional Makie `Figure` attributes forwarded to the combined layout.
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

    panels = MakiePanel[]

    # Q-function panel (always uses all iterations, not just stored ones).
    # Plot index 1 is always in column 1, so it gets the ylabel.
    q_plot = create_styled_plot(; title = "Q Function", xlabel = "EM Iteration",
        ylabel = "Value", style = style, kwargs_subplot...)
    create_styled_line!(q_plot, collect(1:length(diag.Q_hist)), Float64.(diag.Q_hist);
        label = "", color = style.color_primary, style = style)
    push!(panels, q_plot)

    # One panel per scalar element of each selected parameter block.
    for pname in selected_order
        first_val = getproperty(θ_vals[1], pname)
        if first_val isa Number
            traj = Float64[float(getproperty(θ, pname)) for θ in θ_vals]
            push!(panels,
                create_styled_plot(; title = string(pname), xlabel = x_label,
                    ylabel = "", style = style, kwargs_subplot...))
            create_styled_line!(panels[end], stored_iters, traj;
                label = "", color = style.color_primary, style = style)
        elseif first_val isa AbstractVector
            n_elem = length(first_val)
            for j in eachindex(first_val)
                traj = Float64[float(getproperty(θ, pname)[j]) for θ in θ_vals]
                label = n_elem == 1 ? string(pname) : string(pname, "[", j, "]")
                push!(panels,
                    create_styled_plot(; title = label, xlabel = x_label,
                        ylabel = "", style = style, kwargs_subplot...))
                create_styled_line!(panels[end], stored_iters, traj;
                    label = "", color = style.color_primary, style = style)
            end
        elseif first_val isa AbstractMatrix
            for r in axes(first_val, 1)
                for c in axes(first_val, 2)
                    traj = Float64[float(getproperty(θ, pname)[r, c]) for θ in θ_vals]
                    label = string(pname, "[", r, ",", c, "]")
                    push!(panels,
                        create_styled_plot(; title = label, xlabel = x_label,
                            ylabel = "", style = style, kwargs_subplot...))
                    create_styled_line!(panels[end], stored_iters, traj;
                        label = "", color = style.color_primary, style = style)
                end
            end
        end
    end

    # Apply "Value" ylabel only to panels in the first column.
    for i in eachindex(panels)
        if (i - 1) % ncols == 0
            _axis_attrs!(panels[i]; ylabel = "Value")
        end
    end

    result = combine_plots(panels; ncols = ncols, style = style, kwargs_layout...)
    return _save_plot!(result, save_path)
end
