# LEGACY Makie panel layer (pre-AlgebraOfGraphics). Still used by the plotting functions
# that have not been ported to AoG yet; delete this file once every function is ported
# (see dev/aog_battletest/PORTING.md). New code must use the AoG layer in plotting.jl.

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
function create_styled_plot(;
        title = "", xlabel = "", ylabel = "",
        style::PlotStyle = PlotStyle(), kwargs...
    )
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
function create_styled_scatter!(
        p::MakiePanel, x, y; label = "", color = COLOR_PRIMARY,
        style::PlotStyle = PlotStyle(), kwargs...
    )
    # Individuals with no observations (event-only rows) yield empty series; Makie's
    # dimension conversion cannot infer a type from those, so drop them.
    (x isa AbstractArray && isempty(x)) && return p
    lbl = _label(p, label)
    kw = NamedTuple(kwargs)
    col = haskey(kw, :color) ? kw.color : color
    styled_col = col isa Union{AbstractVector, Tuple} ? col : (col, style.marker_alpha)
    defaults = (;
        color = styled_col, markersize = style.marker_size,
        strokewidth = style.marker_stroke_width, strokecolor = :black,
    )
    attrs = merge(defaults, Base.structdiff(kw, NamedTuple{(:color,)}))
    _record!(p, ax -> scatter!(ax, x, y; label = lbl, attrs...))
    return p
end

"""
    create_styled_line!(p, x, y; label, color, style, kwargs...) -> MakiePanel

Record a styled `lines!`. `kwargs...` (`linewidth`, `linestyle`, a `color` override, ...)
win over the styled defaults.
"""
function create_styled_line!(
        p::MakiePanel, x, y; label = "", color = COLOR_SECONDARY,
        style::PlotStyle = PlotStyle(), kwargs...
    )
    (x isa AbstractArray && isempty(x)) && return p
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
function add_reference_line!(
        p::MakiePanel, value; orientation = :horizontal,
        color = COLOR_REFERENCE, kwargs...
    )
    kw = NamedTuple(kwargs)
    lbl = haskey(kw, :label) ? _label(p, kw.label) : nothing
    rest = Base.structdiff(kw, NamedTuple{(:label,)})
    draw = orientation == :horizontal ? hlines! : vlines!
    _record!(
        p,
        ax -> draw(ax, value; color = color, linestyle = :dash, label = lbl, rest...)
    )
    return p
end

"""
    add_annotation!(p, x, y, txt; fontsize, halign) -> MakiePanel

Record a text annotation at data coordinates `(x, y)`. `halign` (`:right`/`:left`/
`:center`) is the horizontal text alignment.
"""
function add_annotation!(p::MakiePanel, x, y, txt; fontsize = 7, halign = :right)
    _record!(
        p,
        ax -> text!(
            ax, [float(x)], [float(y)];
            text = [string(txt)], align = (halign, :center), fontsize = fontsize
        )
    )
    return p
end

"""
    _hist!(p, vals; bins, normalization, color, label, style, kwargs...) -> MakiePanel

Draw a histogram as a touching `barplot!` from uniform bins (via `_histogram_xy`). This
is the mechanical replacement for the old `histogram!(p, vals; bins, normalize, fillcolor,
linecolor)` call. `normalization ∈ (:probability, :pdf, :none)`. Map old `fillcolor` to
`color` and old `linecolor` to `strokecolor` (passed via `kwargs...`).
"""
function _hist!(
        p::MakiePanel, vals; bins::Int = 30, normalization::Symbol = :probability,
        color = COLOR_PRIMARY, label = "", style::PlotStyle = PlotStyle(), kwargs...
    )
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
            axislegend(
                ax; position = something(p.legend_position, :rt),
                unique = true, labelsize = style.font_size_legend
            )
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


# Set shared x/y limits across panels (recursing into groups) before materialization.
function _apply_shared_axes!(panels::AbstractVector, xlim, ylim)
    for panel in panels
        _apply_shared_axes_one!(panel, xlim, ylim)
    end
    return panels
end

function _apply_shared_axes_one!(p::MakiePanel, xlim, ylim)
    return _set_limits!(p; xlim = xlim, ylim = ylim)
end

function _apply_shared_axes_one!(g::MakiePanelGroup, xlim, ylim)
    for p in g.panels
        _set_limits!(p; xlim = xlim, ylim = ylim)
    end
    return g
end
