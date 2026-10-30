# AlgebraOfGraphics (AoG) core layer shared by every drawing file of this extension.
#
# Two figure shapes cover every NoLimits plot:
#
#  1. Faceted figures (one panel per individual / observable / parameter / ...): build tidy
#     `DataFrame`s with a `:panel` column, combine AoG layers that map `layout = :panel`,
#     and hand them to [`_draw_facets`](@ref). It applies the shared policy: panel order
#     and `ncols` wrapping, linked axes when shared (inner tick labels and repeated axis
#     labels hidden), ONE figure-level legend, `PlotStyle` + `kwargs_subplot` +
#     `kwargs_layout` mapping, explicit shared limits, and per-panel plain-Makie extras.
#
#  2. Composable single-axis panels ([`AoGPanel`](@ref)): what `return_panel = true`
#     returns; [`combine_plots`](@ref) lays any vector of panels into one figure with one
#     merged legend. A function that draws a single axis builds an `AoGPanel` and returns
#     `combine_plots([panel]; ncols = 1, style, kwargs_layout...)`.
#
# Precedence everywhere: PlotStyle defaults < values computed by the plotting function <
# user `kwargs_subplot` (Axis) / `kwargs_layout` (Figure). See
# dev/aog_battletest/PORTING.md for the full conventions.

const AoG = AlgebraOfGraphics

_nt(x::NamedTuple) = x
_nt(x::Nothing) = NamedTuple()
_nt(x) = NamedTuple(Symbol(k) => v for (k, v) in pairs(x))

# --- PlotStyle mapping -------------------------------------------------------------------

"""
    _aog_axis(style, kwargs_subplot = (;); computed...) -> NamedTuple

Axis attributes for AoG's `axis =` keyword: `default_axis_kwargs(style)` merged with the
function's `computed` attributes (xlabel, ylabel, ...) merged with the user's
`kwargs_subplot` (user wins).
"""
function _aog_axis(style::PlotStyle, kwargs_subplot = NamedTuple(); computed...)
    return merge(default_axis_kwargs(style), NamedTuple(computed), _nt(kwargs_subplot))
end

"""
    _aog_figure(style, n_panels, ncols, kwargs_layout = (;)) -> NamedTuple

Figure attributes for AoG's `figure =` keyword (size from `calculate_plot_size`, fonts,
padding) merged with the user's `kwargs_layout` (user wins).
"""
function _aog_figure(style::PlotStyle, n_panels::Integer, ncols::Integer, kwargs_layout = NamedTuple())
    n = max(Int(n_panels), 1)
    defaults = (;
        size = calculate_plot_size(n, min(Int(ncols), n), style),
        fonts = (; regular = style.font_family, bold = style.font_family),
        figure_padding = style.figure_padding,
    )
    return merge(defaults, _nt(kwargs_layout))
end

"""
    _aog_legend(style; kwargs...) -> NamedTuple

Figure-level legend attributes: one horizontal legend below the panels, no frame, no
section titles.
"""
function _aog_legend(style::PlotStyle; kwargs...)
    defaults = (;
        position = :bottom, orientation = :horizontal, framevisible = false,
        labelsize = style.font_size_legend, titlevisible = false,
    )
    return merge(defaults, NamedTuple(kwargs))
end

"""
    _legend_nbanks(labels, width, style) -> Int

Number of rows (`nbanks`) a horizontal legend with entries `labels` needs to fit into a
figure `width` units wide: the smallest row count for which the widest entry of every
column, summed over the columns, fits. Entry widths are estimated from the label length
and `style.font_size_legend` (marker patch + gaps included), so long legends wrap instead
of overflowing the figure.
"""
function _legend_nbanks(labels, width::Real, style::PlotStyle)
    n = length(labels)
    n <= 1 && return 1
    fs = style.font_size_legend
    w = [0.6 * fs * length(string(l)) + 2.5 * fs + 16 for l in labels]  # text + patch + gaps
    avail = max(Float64(width) - 2 * style.figure_padding - 20, 1.0)
    for nb in 1:n
        nc = cld(n, nb)
        sum(maximum(w[j:nc:n]) for j in 1:nc) <= avail && return nb
    end
    return n
end

# Makie's default legend patch size, scaled with the fonts of large figures.
const _LEGEND_PATCHSIZE = (20.0f0, 20.0f0)

_FONT_FIELDS = (
    :font_size_title, :font_size_label, :font_size_tick, :font_size_legend,
    :font_size_annotation,
)

"""
    _figure_style(style, size) -> PlotStyle

`style` with its font sizes multiplied by `_font_scale(size...)` (rounded), the style a
figure of `size` is drawn with. Large figures are viewed scaled down; enlarging their
fonts (and the legend patches, see `_legend_patchsize`) keeps text and legends readable.
Every figure builder ([`_draw_facets`](@ref), [`combine_plots`](@ref) and the grids that
create their own `Figure`) draws with it, before layout, so Makie lays out the final
sizes. User values in `kwargs_subplot` still win (they are merged last).
"""
function _figure_style(style::PlotStyle, size)
    f = _font_scale(size[1], size[2])
    f == 1.0 && return style
    nt = NamedTuple{fieldnames(PlotStyle)}(Tuple(getfield(style, k) for k in fieldnames(PlotStyle)))
    scaled = NamedTuple{_FONT_FIELDS}(Tuple(round(Int, getfield(style, k) * f) for k in _FONT_FIELDS))
    return PlotStyle(; merge(nt, scaled)...)
end

# Legend patch size for a figure drawn with `fstyle = _figure_style(style, size)`.
_legend_patchsize(fstyle::PlotStyle, style::PlotStyle) =
    _LEGEND_PATCHSIZE .* Float32(fstyle.font_size_legend / style.font_size_legend)

# Axis attributes of a composable panel rescaled to the figure style: font sizes that still
# equal the panel's `style` value (i.e. were not set by the user) take the `fstyle` value.
function _rescale_axis_fonts(axis::NamedTuple, style::PlotStyle, fstyle::PlotStyle)
    style === fstyle && return axis
    pairs_ = (
        :titlesize => :font_size_title, :subtitlesize => :font_size_label,
        :xlabelsize => :font_size_label, :ylabelsize => :font_size_label,
        :xticklabelsize => :font_size_tick, :yticklabelsize => :font_size_tick,
    )
    out = axis
    for (attr, field) in pairs_
        haskey(out, attr) && out[attr] == getfield(style, field) || continue
        out = merge(out, NamedTuple{(attr,)}((getfield(fstyle, field),)))
    end
    return out
end

"""
    _scatter_visual(style; kwargs...) -> AoG.Layer

`visual(Scatter)` with the style's marker size/alpha/stroke; `kwargs` win.
"""
function _scatter_visual(style::PlotStyle; kwargs...)
    defaults = (;
        markersize = style.marker_size, strokewidth = style.marker_stroke_width,
        strokecolor = :black, alpha = style.marker_alpha,
    )
    return AoG.visual(Makie.Scatter; merge(defaults, NamedTuple(kwargs))...)
end

"""
    _line_visual(style; kwargs...) -> AoG.Layer

`visual(Lines)` with the style's primary line width; `kwargs` win.
"""
function _line_visual(style::PlotStyle; kwargs...)
    return AoG.visual(Makie.Lines; merge((; linewidth = style.line_width_primary), NamedTuple(kwargs))...)
end

# --- Tidy tables ---------------------------------------------------------------------------

_f64(v) = (v === missing || v === nothing || !(v isa Real)) ? NaN : Float64(v)

"""
    _tidy(; extra...) -> DataFrame

Empty long-format table with the standard columns `panel::String, x, y::Float64,
series::String` plus `extra` columns given as `name = T` (element type).
"""
function _tidy(; extra...)
    df = DataFrame(panel = String[], x = Float64[], y = Float64[], series = String[])
    for (k, T) in extra
        df[!, k] = T[]
    end
    return df
end

"""
    _push_xy!(df, panel, xs, ys, series) -> df

Append one series to a `_tidy()` table (`missing`/non-real values become `NaN`, which
AoG/Makie draw as gaps).
"""
function _push_xy!(df::DataFrame, panel::AbstractString, xs, ys, series::AbstractString)
    for (x, y) in zip(xs, ys)
        push!(df, (String(panel), _f64(x), _f64(y), String(series)); promote = false)
    end
    return df
end

# `data(df) * mapping(...) * visual` only when the table has rows (AoG cannot infer
# scales from an empty table); returns `nothing` otherwise. Combine with `_layers`.
_layer(df::DataFrame, m, v) = nrow(df) == 0 ? nothing : AoG.data(df) * m * v

"""
    _layers(ls...) -> AoG.Layers | nothing

Sum the non-`nothing` layers (`nothing` when all are `nothing`).
"""
function _layers(ls...)
    keep = [l for l in ls if l !== nothing]
    isempty(keep) && return nothing
    return reduce(+, keep)
end

"""
    _series_palette(pairs) -> NamedTuple

`Color` scale settings mapping series labels to colors (`["data" => "#0072B2", ...]`),
keeping the given order in the legend and hiding the legend title.
"""
_series_palette(pairs) = (; palette = collect(pairs), categories = [first(p) for p in pairs], label = "")

# --- Category order ------------------------------------------------------------------------
#
# AoG sorts categorical values (facets, colours, ...) unless told otherwise, which puts
# "s10" before "s2". NoLimits shows categories in the order they first appear in the data:
# pass the levels explicitly as scale `categories` (as `_draw_facets` does for `panels`) or
# map a column with `_data_ordered(:col)`.

"""
    _ordered_levels(v) -> Vector

Distinct values of `v` in first-appearance order (never sorted). Use it for the
`categories` of an AoG scale or the `panels` of [`_draw_facets`](@ref).
"""
_ordered_levels(v) = unique(v)

"""
    _data_ordered(col::Symbol) -> Pair

AoG mapping entry `col => presorted`: categorical levels of `col` keep the row order of
the table instead of AoG's default sorting.
"""
_data_ordered(col::Symbol) = col => AoG.presorted

"""
    _individuals_in_data_order(dm, inds) -> Vector{Int}

The individual indices `inds` sorted by the first row of each individual in the data
table (`DataModel` itself sorts individuals by id, which for string ids is lexicographic:
s1, s10, s11, ..., s2).
"""
function _individuals_in_data_order(dm::DataModel, inds)
    rows = get_rows(get_row_groups(dm))
    return sort!(collect(Int, inds); by = i -> minimum(rows[i]))
end

"""
    _plot_individuals(dm, individuals_idx; default_all = true) -> Vector{Int}

Individuals to draw, in panel order. An explicit `individuals_idx` (indices or ids) is
resolved by `_resolve_individuals` and keeps the caller's order; `nothing` selects all
individuals (or the first one in data order when `default_all = false`) in the order
they first appear in the data.
"""
function _plot_individuals(dm::DataModel, individuals_idx; default_all::Bool = true)
    individuals_idx === nothing || return _resolve_individuals(dm, individuals_idx)
    order = _individuals_in_data_order(dm, eachindex(get_individuals(dm)))
    return default_all ? order : order[1:1]
end

# --- Faceted figures ---------------------------------------------------------------------

"""
    _facet_link(shared::Bool) -> Symbol

`:all` when the axis is shared across panels, `:none` otherwise (AoG then shows inner tick
labels; when shared, inner decorations are hidden and labels are spanned).
"""
_facet_link(shared::Bool) = shared ? :all : :none

"""
    _draw_facets(layers; panels, ncols, style, shared_x, shared_y, kwargs_subplot,
                 kwargs_layout, axis, series, scales, xlim, ylim, post, legend) -> Figure

Draw AoG `layers` (which must map `layout = :panel`) as a wrapped facet grid.

- `panels::Vector{String}`: panel labels in display order (the `:panel` column values).
- `ncols`: wrap width (validated `>= 1`); panel `k` sits at `fldmod1(k, min(ncols, n))`.
- `shared_x`/`shared_y`: link axes (`:all`) and hide inner decorations, or keep them free.
- `axis`: computed Axis attributes (xlabel, ylabel, ...); `kwargs_subplot` wins over them.
- `series`: `label => color` pairs for the categorical `Color` scale (one legend entry each).
- `scales`: extra AoG scale settings as a NamedTuple (e.g. `(; LineStyle = (; ...))`).
- `xlim`/`ylim`: explicit limits applied to every panel (typically the padded union when
  shared, `nothing` to let Makie autoscale).
- `post::Dict{String, Vector}`: per-panel plain-Makie extras `ax -> ...` (heatmaps,
  reference lines with panel-specific values, annotations) run after AoG drew the panel.
- `legend`: draw the single figure-level legend when a series scale exists.
"""
function _draw_facets(
        layers; panels::AbstractVector{<:AbstractString}, ncols::Integer, style::PlotStyle,
        shared_x::Bool = true, shared_y::Bool = true,
        kwargs_subplot = NamedTuple(), kwargs_layout = NamedTuple(), axis = NamedTuple(),
        series = Pair{String, Any}[], scales = NamedTuple(),
        xlim = nothing, ylim = nothing,
        post::AbstractDict = Dict{String, Vector{Any}}(), legend::Bool = true
    )
    _check_positive_int(ncols, "ncols")
    n = length(panels)
    ncols_use = min(Int(ncols), max(n, 1))
    layers === nothing &&
        error("Nothing to plot: no finite data for the selected individuals/observables.")
    sc = (; Layout = (; palette = AoG.wrapped(cols = ncols_use), categories = collect(String, panels)))
    isempty(series) || (sc = merge(sc, (; Color = _series_palette(series))))
    sc = merge(sc, _nt(scales))
    fig_kw = _aog_figure(style, n, ncols_use, kwargs_layout)
    fstyle = _figure_style(style, fig_kw.size)
    nbanks = _legend_nbanks(first.(series), first(fig_kw.size), fstyle)
    fg = AoG.draw(
        layers, AoG.scales(; sc...);
        axis = _aog_axis(fstyle, kwargs_subplot; _nt(axis)...),
        figure = fig_kw,
        facet = (; linkxaxes = _facet_link(shared_x), linkyaxes = _facet_link(shared_y)),
        legend = (legend && !isempty(series)) ?
            _aog_legend(fstyle; nbanks, patchsize = _legend_patchsize(fstyle, style)) :
            (; show = false),
    )
    # user `limits` in kwargs_subplot win over the computed shared limits
    user_limits = haskey(_nt(kwargs_subplot), :limits)
    for (k, label) in enumerate(panels)
        ax = _facet_axis(fg, k, ncols_use)
        ax === nothing && continue
        for f in get(post, label, ())
            f(ax)
        end
        user_limits && continue
        xlim === nothing || xlims!(ax, xlim...)
        ylim === nothing || ylims!(ax, ylim...)
    end
    return fg.figure
end

# Axis of panel `k` in a wrapped facet grid (`nothing` if AoG deleted it as empty).
function _facet_axis(fg, k::Integer, ncols::Integer)
    r, c = fldmod1(k, ncols)
    (r <= size(fg.grid, 1) && c <= size(fg.grid, 2)) || return nothing
    ae = fg.grid[r, c]
    return isempty(ae.entries) ? nothing : ae.axis
end

# --- Composable panels (return_panel / combine_plots) ------------------------------------

"""
    AoGPanel(layers; scales = (;), axis = (;), post = [], legend = true)

One single-axis subplot: AoG `layers` (no `layout` mapping; `nothing` for a panel drawn
only by `post`), AoG scale settings (NamedTuple, e.g. `(; Color = _series_palette(...))`),
the complete Axis attributes (build them with [`_aog_axis`](@ref)), and plain-Makie extras
`post` (`ax -> ...`) run after the layers are drawn. This is what plotting functions
return with `return_panel = true`; [`combine_plots`](@ref) materializes panels.
"""
struct AoGPanel
    layers::Any
    scales::NamedTuple
    axis::NamedTuple
    post::Vector{Any}
    legend::Bool
end

AoGPanel(layers; scales = NamedTuple(), axis = NamedTuple(), post = Any[], legend::Bool = true) =
    AoGPanel(layers, _nt(scales), _nt(axis), collect(Any, post), legend)

# Draw an AoGPanel into a grid position; returns (axis, legend-or-nothing), where the legend
# is AoG's `(elements, labels, titles)` triple.
function _materialize_panel!(pos, p::AoGPanel)
    if p.layers === nothing
        ax = Axis(pos; p.axis...)
        foreach(f -> f(ax), p.post)
        return ax, nothing
    end
    ag = AoG.draw!(pos, p.layers, AoG.scales(; p.scales...); axis = p.axis)
    ax = first(ag).axis
    foreach(f -> f(ax), p.post)
    leg = p.legend ? AoG.compute_legend(ag; order = nothing, hide_unused = true) : nothing
    return ax, leg
end

# One horizontal legend below everything, merging AoG legend triples by label (first
# occurrence wins).
function _figure_legend!(
        fig::Figure, legends, style::PlotStyle; row::Integer, ncols::Integer,
        patchsize = _LEGEND_PATCHSIZE
    )
    elements = Any[]
    labels = String[]
    for leg in legends
        leg === nothing && continue
        els, labs, _ = leg
        for (sec_els, sec_labs) in zip(els, labs), (e, l) in zip(sec_els, sec_labs)
            s = string(l)
            s in labels && continue
            push!(elements, e)
            push!(labels, s)
        end
    end
    isempty(labels) && return nothing
    nbanks = _legend_nbanks(labels, first(fig.scene.viewport[].widths), style)
    return Legend(
        fig[row, 1:ncols], elements, labels;
        orientation = :horizontal, framevisible = false, labelsize = style.font_size_legend,
        patchsize = patchsize,
        nbanks = nbanks, tellheight = true, tellwidth = false
    )
end

"""
    _link_panel_axes!(axes, ncols; link_x = false, link_y = false, keep = ()) -> axes

Link the axes of panels laid out row-wise on an `ncols`-wide grid (`axes[k]` is panel
`k`, `nothing` for a missing panel) and hide the decorations a shared axis repeats: y tick
labels and y label outside the first column, x tick labels and x label except on the
lowest panel of each column. Attributes named in `keep` (e.g. `keys(kwargs_subplot)`)
are left as the caller set them. This is the `combine_plots(...; link_x, link_y)`
policy and matches the decluttering of `_draw_facets`.
"""
function _link_panel_axes!(
        axes::AbstractVector, ncols::Integer;
        link_x::Bool = false, link_y::Bool = false, keep = ()
    )
    present = [ax for ax in axes if ax !== nothing]
    length(present) > 1 || return axes
    n = length(axes)
    ncols_use = min(Int(ncols), n)
    _hide!(ax, attr) = attr in keep || setproperty!(ax, attr, false)
    if link_y
        Makie.linkyaxes!(present...)
        for (k, ax) in enumerate(axes)
            (ax === nothing || fldmod1(k, ncols_use)[2] == 1) && continue
            _hide!(ax, :yticklabelsvisible)
            _hide!(ax, :ylabelvisible)
        end
    end
    if link_x
        Makie.linkxaxes!(present...)
        for (k, ax) in enumerate(axes)
            ax === nothing && continue
            # lowest panel of its column: no panel below it
            any(j -> axes[j] !== nothing, (k + ncols_use):ncols_use:n) || continue
            _hide!(ax, :xticklabelsvisible)
            _hide!(ax, :xlabelvisible)
        end
    end
    return axes
end

"""
    combine_plots(panels::Vector; ncols, style, link_x = false, link_y = false,
                  kwargs...) -> Makie.Figure

Lay `panels` (as returned by plotting functions with `return_panel = true`) into one
`Figure` on an `ncols`-wide grid, with a single merged legend below the panels.
`kwargs...` are Makie `Figure` attributes (this is where a caller's `kwargs_layout` flows);
a caller `size`/`figure_padding`/... wins over the computed defaults. `link_x`/`link_y`
link the panels' x/y axes and hide the repeated tick labels and axis labels (x kept on the
lowest panel of each column, y in the first column); an attribute a panel sets explicitly
in its `axis` (e.g. `yticklabelsvisible` from `kwargs_subplot`) is kept. The legend wraps
into several rows when its entries do not fit the figure width.
"""
function combine_plots(
        panels::Vector; ncols::Int = DEFAULT_PLOT_COLS,
        style::PlotStyle = PlotStyle(), link_x::Bool = false, link_y::Bool = false,
        kwargs...
    )
    _check_positive_int(ncols, "ncols")
    n = length(panels)
    ncols_use = min(ncols, max(n, 1))
    fig_kw = _aog_figure(style, n, ncols_use, NamedTuple(kwargs))
    fig = Figure(; fig_kw...)
    fstyle = _figure_style(style, fig_kw.size)
    legends = Any[]
    axes = Any[nothing for _ in 1:n]
    keep = Set{Symbol}()
    for (i, panel) in enumerate(panels)
        panel isa AoGPanel || throw(
            ArgumentError(
                "combine_plots expects panels returned with `return_panel = true` " *
                    "(AoGPanel); got $(typeof(panel))."
            )
        )
        row, col = fldmod1(i, ncols_use)
        # fonts of large figures grow before layout (sizes the user set are kept)
        fstyle === style || (
            panel = AoGPanel(
                panel.layers, panel.scales, _rescale_axis_fonts(panel.axis, style, fstyle),
                panel.post, panel.legend
            )
        )
        ax, leg = _materialize_panel!(fig[row, col], panel)
        axes[i] = ax
        push!(legends, leg)
        # decorations the panel sets explicitly (e.g. from kwargs_subplot) win
        union!(keep, keys(panel.axis))
    end
    (link_x || link_y) && _link_panel_axes!(axes, ncols_use; link_x, link_y, keep)
    nrows = cld(max(n, 1), ncols_use)
    _figure_legend!(
        fig, legends, fstyle; row = nrows + 1, ncols = ncols_use,
        patchsize = _legend_patchsize(fstyle, style)
    )
    return fig
end

"""
    _placeholder_figure(msg, style; kwargs_layout = (;)) -> Figure

Figure returned when a plotting function has nothing to draw: the message `msg` centred
on an otherwise empty single-panel figure, in the style's label font (no axis, ticks or
title).
"""
function _placeholder_figure(msg::AbstractString, style::PlotStyle; kwargs_layout = NamedTuple())
    fig = Figure(; _aog_figure(style, 1, 1, kwargs_layout)...)
    Makie.Label(
        fig[1, 1], msg; fontsize = style.font_size_label, color = style.color_dark,
        tellwidth = false, tellheight = false
    )
    return fig
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
        error(
            "Saving figures requires a rasterizing Makie backend. Load one first, " *
                "e.g. `using CairoMakie`. Underlying error: $(err)"
        )
    end
    return fig
end
