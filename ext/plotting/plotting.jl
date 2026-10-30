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
    fg = AoG.draw(
        layers, AoG.scales(; sc...);
        axis = _aog_axis(style, kwargs_subplot; _nt(axis)...),
        figure = _aog_figure(style, n, ncols_use, kwargs_layout),
        facet = (; linkxaxes = _facet_link(shared_x), linkyaxes = _facet_link(shared_y)),
        legend = (legend && !isempty(series)) ? _aog_legend(style) : (; show = false),
    )
    for (k, label) in enumerate(panels)
        ax = _facet_axis(fg, k, ncols_use)
        ax === nothing && continue
        for f in get(post, label, ())
            f(ax)
        end
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
function _figure_legend!(fig::Figure, legends, style::PlotStyle; row::Integer, ncols::Integer)
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
    return Legend(
        fig[row, 1:ncols], elements, labels;
        orientation = :horizontal, framevisible = false, labelsize = style.font_size_legend,
        tellheight = true, tellwidth = false
    )
end

"""
    combine_plots(panels::Vector; ncols, style, kwargs...) -> Makie.Figure

Lay `panels` (as returned by plotting functions with `return_panel = true`) into one
`Figure` on an `ncols`-wide grid, with a single merged legend below the panels.
`kwargs...` are Makie `Figure` attributes (this is where a caller's `kwargs_layout` flows);
a caller `size`/`figure_padding`/... wins over the computed defaults.
"""
function combine_plots(
        panels::Vector; ncols::Int = DEFAULT_PLOT_COLS,
        style::PlotStyle = PlotStyle(), kwargs...
    )
    _check_positive_int(ncols, "ncols")
    n = length(panels)
    ncols_use = min(ncols, max(n, 1))
    fig = Figure(; _aog_figure(style, n, ncols_use, NamedTuple(kwargs))...)
    legends = Any[]
    for (i, panel) in enumerate(panels)
        row, col = fldmod1(i, ncols_use)
        if panel isa AoGPanel
            _, leg = _materialize_panel!(fig[row, col], panel)
            push!(legends, leg)
        else
            # LEGACY MakiePanel / MakiePanelGroup (unported functions; legend in-axis)
            _materialize!(fig[row, col], panel, style)
        end
    end
    nrows = cld(max(n, 1), ncols_use)
    _figure_legend!(fig, legends, style; row = nrows + 1, ncols = ncols_use)
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
