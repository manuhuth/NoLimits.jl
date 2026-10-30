# Precompile workload for the AlgebraOfGraphics core layer.
#
# The first AoG draw in a fresh process compiles AoG's scale/legend/facet machinery and
# Makie's layout for our table types. Running a small faceted figure and a composed panel
# while the extension is being precompiled stores that native code in the package image,
# so the first `plot_*` call of a session is faster. No extra dependency: the workload
# runs only under `jl_generating_output` (i.e. during precompilation) and reproduces what a
# PrecompileTools `@compile_workload` does. Rendering to a file is backend specific
# (CairoMakie) and is not covered.

function _precompile_workload()
    style = PlotStyle()
    panels = ["ID: 1", "ID: 2"]
    obs = _tidy()
    fit = _tidy()
    q = _tidy()
    for (k, p) in enumerate(panels)
        _push_xy!(obs, p, [0.0, 1.0, 2.0], [1.0, 2.0, 1.5] .+ k, "data")
        _push_xy!(fit, p, [0.0, 1.0, 2.0], [1.0, 1.5, 2.0] .+ k, "fit")
        _push_xy!(q, p, [0.0, 1.0, 2.0], [0.5, 1.0, 1.5] .+ k, "5%")
    end
    xy = AoG.mapping(:x, :y; color = :series, layout = :panel)
    layers = _layers(
        _layer(fit, xy, _line_visual(style)),
        _layer(q, xy, _line_visual(style; linestyle = :dash, alpha = 0.8)),
        _layer(obs, xy, _scatter_visual(style)),
    )
    cmap = _density_colormap(nothing, nothing)
    post = Dict{String, Vector{Any}}(
        "ID: 1" => Any[
            ax -> heatmap!(ax, [0.0, 2.0], [0.0, 4.0], [0.0 1.0; 1.0 0.0]; colormap = cmap),
        ]
    )
    series = Pair{String, Any}[
        "data" => style.color_primary, "fit" => style.color_secondary,
        "5%" => style.color_secondary,
    ]
    for shared in (true, false)
        fig = _draw_facets(
            layers; panels = panels, ncols = 2, style = style,
            shared_x = shared, shared_y = shared,
            axis = (; xlabel = "Time", ylabel = "Y"), series = series, post = post,
            xlim = shared ? (0.0, 2.0) : nothing, ylim = shared ? (0.0, 4.0) : nothing,
        )
        Makie.update_state_before_display!(fig)
    end
    panel = AoGPanel(
        AoG.data(fit) * AoG.mapping(:x, :y; color = :series) * _line_visual(style) +
            AoG.data(obs) * AoG.mapping(:x, :y; color = :series) * _scatter_visual(style);
        scales = (; Color = _series_palette(series[1:2])),
        axis = _aog_axis(style; xlabel = "x", ylabel = "y"),
        post = Any[ax -> Makie.ablines!(ax, 0, 1; color = COLOR_REFERENCE, linestyle = :dash)],
    )
    fig = combine_plots([panel, panel]; ncols = 2, style = style)
    Makie.update_state_before_display!(fig)
    # BarPlot (histograms, PMFs), Band (intervals) and VLines (reference values), used by
    # the residual, random-effect, UQ, VPC and diagnostic figures.
    bars = DataFrame(x = [0.5, 1.5, 2.5], y = [0.2, 0.5, 0.3], series = fill("histogram", 3))
    band = DataFrame(
        x = [0.0, 1.0, 2.0], lo = [0.5, 1.0, 1.5], hi = [1.5, 2.0, 2.5], series = fill("90% Interval", 3)
    )
    vl = DataFrame(x = [1.0], series = ["estimate"])
    panel2 = AoGPanel(
        AoG.data(bars) * AoG.mapping(:x, :y; color = :series) *
            AoG.visual(Makie.BarPlot; gap = 0, strokewidth = 0, width = 1.0) +
            AoG.data(band) * AoG.mapping(:x, :lo, :hi; color = :series) *
            AoG.visual(Makie.Band; alpha = 0.3) +
            AoG.data(vl) * AoG.mapping(:x; color = :series) *
            AoG.visual(Makie.VLines; linestyle = :dash);
        scales = (;
            Color = _series_palette(
                [
                    "histogram" => style.color_secondary, "90% Interval" => style.color_primary,
                    "estimate" => style.color_primary,
                ]
            ),
        ),
        axis = _aog_axis(style; xlabel = "x", ylabel = "Density"),
        post = Any[ax -> heatmap!(ax, [0.0, 2.0], [0.0, 1.0], [0.0 1.0; 1.0 0.0]; colormap = cmap)],
    )
    fig = combine_plots([panel2, panel2]; ncols = 2, style = style, link_x = true, link_y = true)
    Makie.update_state_before_display!(fig)
    return nothing
end

# On Julia >= 1.12 every method instance inferred inside the workload, including
# runtime-dispatched callees owned by AoG/Makie, is tagged for the package image (the
# mechanism PrecompileTools' `@compile_workload` uses; without the tags most of that code
# is dropped and the first draw still compiles it).
const _TAG_INFERRED = VERSION >= v"1.12.0-"

if ccall(:jl_generating_output, Cint, ()) == 1
    _TAG_INFERRED && ccall(:jl_tag_newly_inferred_enable, Cvoid, ())
    try
        _precompile_workload()
    catch err
        @debug "NoLimitsMakieExt precompile workload failed" exception = err
    finally
        _TAG_INFERRED && ccall(:jl_tag_newly_inferred_disable, Cvoid, ())
    end
end
