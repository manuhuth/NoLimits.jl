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
