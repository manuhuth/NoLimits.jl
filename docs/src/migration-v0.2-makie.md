# Migrating to v0.2 (Makie plotting)

NoLimits v0.2.0 replaces the Plots.jl drawing backend with [Makie](https://docs.makie.org/).
This guide covers everything a v0.1 user needs to change. Estimation, model building,
`DataModel`, and all numerical results are untouched, only plotting is affected.

## TL;DR

```julia
# v0.1
using NoLimits, Plots
fig = plot_fits(res)                 # Plots.Plot

# v0.2
using NoLimits, CairoMakie
fig = plot_fits(res)                 # Makie.Figure
```

All 28 `plot_*` functions keep their names, positional arguments, and (with the
exceptions below) keyword arguments. If you never passed `kwargs_subplot`,
`kwargs_layout`, or custom `PlotStyle` margins or marker sizes, swapping the
`using` line is usually the entire migration.

## Loading the plotting extension

The drawing functions live in a package extension that loads when Makie is available.
Load any Makie backend alongside NoLimits:

```julia
using NoLimits
using CairoMakie      # static publication-quality output (recommended default)
# or: using GLMakie   # interactive windows
```

Calling a plot function without a backend loaded raises a `MethodError` whose message
tells you to run `using CairoMakie`.

## Return types and display

- Every plot function returns a `Makie.Figure` (previously `Plots.Plot`).
- The vector layouts (`marginal_layout = :vector`, `figure_layout = :vector`) return
  `Vector{Makie.Figure}`.
- In notebooks and VS Code, figures display inline exactly as before.
- In a terminal REPL, CairoMakie does not open a window (GR popped one). Either save
  the figure (`save_path`) or use GLMakie for interactive display.

## Saving figures

Unchanged: every plot function accepts `save_path` (alias `plot_path`) and `.png`
output matches the previous 300 dpi resolution.

```julia
plot_fits(res; save_path = "fits.png")
```

To save a returned figure manually, use Makie's `save` instead of `savefig`:

```julia
fig = plot_fits(res)
save("fits.png", fig; px_per_unit = 3)   # px_per_unit = 3 ≈ the old dpi = 300
```

Saving requires a rasterizing backend such as CairoMakie for `.png`; `.pdf` and
`.svg` also work through CairoMakie.

## `kwargs_subplot` and `kwargs_layout` now use Makie names

These pass-through keywords are forwarded verbatim, so they now expect Makie
vocabulary: `kwargs_subplot` entries become
[`Axis` attributes](https://docs.makie.org/stable/reference/blocks/axis) (applied to
every panel) and `kwargs_layout` entries become `Figure` attributes. Plots names are
not translated. Common renames:

| Plots (v0.1)                    | Makie (v0.2)                                     | Passed via        |
| :------------------------------ | :----------------------------------------------- | :---------------- |
| `xlabel`, `ylabel`, `title`     | unchanged                                        | `kwargs_subplot`  |
| `xlims = (lo, hi)`              | `limits = ((lo, hi), nothing)`                   | `kwargs_subplot`  |
| `ylims = (lo, hi)`              | `limits = (nothing, (lo, hi))`                   | `kwargs_subplot`  |
| `xscale = :log10`               | `xscale = log10`                                 | `kwargs_subplot`  |
| `xrotation = 45`                | `xticklabelrotation = pi/4` (radians)            | `kwargs_subplot`  |
| `titlefontsize`                 | `titlesize`                                      | `kwargs_subplot`  |
| `guidefontsize`                 | `xlabelsize` and `ylabelsize`                    | `kwargs_subplot`  |
| `tickfontsize`                  | `xticklabelsize` and `yticklabelsize`            | `kwargs_subplot`  |
| `xticks = (pos, labels)`        | unchanged                                        | `kwargs_subplot`  |
| `yflip = true`                  | `yreversed = true`                               | `kwargs_subplot`  |
| `size = (w, h)`                 | unchanged (Figure size in px)                    | `kwargs_layout`   |
| `dpi = 300`                     | not needed (`save_path` handles resolution)      | -                 |
| `legend = :topright` etc.       | no direct pass-through, see below                | -                 |
| `left_margin`, `bottom_margin`  | removed, Makie sizes decorations automatically   | -                 |

Example:

```julia
# v0.1
plot_fits(res; kwargs_subplot = (guidefontsize = 10, xlims = (0.0, 24.0)))

# v0.2
plot_fits(res;
    kwargs_subplot = (xlabelsize = 10, ylabelsize = 10, limits = ((0.0, 24.0), nothing)))
```

Legends are created automatically (`axislegend`) whenever a panel has labeled series,
in the top-right corner by default; functions that previously placed legends elsewhere
(for example `plot_shrinkage`) keep their placement internally.

## `PlotStyle` changes

- `left_margin` and `bottom_margin` (Measures `mm` values) are gone. Makie sizes axis
  decorations automatically; a new `figure_padding::Int = 10` field controls the outer
  padding in px. Measures.jl is no longer a dependency, so drop any `using Measures`.
- Marker size defaults doubled (`marker_size` 5 to 10, `marker_size_small` 3 to 6,
  `marker_size_pmf` 6 to 12) because Makie marker sizes are px diameters where Plots
  used pt. If you constructed a `PlotStyle` with custom marker sizes, roughly double
  them to keep the previous visual weight.
- Font sizes, colors, and line widths keep their fields and defaults.

```julia
# v0.1
style = PlotStyle(marker_size = 4, left_margin = 12mm)

# v0.2
style = PlotStyle(marker_size = 8, figure_padding = 14)
```

## Customizing a returned figure

Plots users could mutate the returned object (`plot!(p; ...)`). Makie figures are
customized either up front through `kwargs_subplot`/`kwargs_layout`/`style`, or after
the fact through the figure's axes:

```julia
fig = plot_fits(res)
ax = first(a for a in fig.content if a isa Axis)
ax.title = "Individual fits"
ax.ylabel = "Concentration [mg/L]"
save("fits.png", fig)
```

## Name collision: `RealVector`

Makie also exports a `RealVector` type. Inside `@Model`/`@fixedEffects` blocks this is
handled automatically since v0.2.0, so model definitions need no change. Only if you
call the constructor outside the macros, qualify it:

```julia
fe = NoLimits.RealVector([0.1, 0.2]; name = :v)   # or: using NoLimits: RealVector
```

## Staying on v0.1

The registry keeps v0.1.x available. If you are not ready to migrate, pin the previous
release in your project:

```julia
using Pkg
Pkg.add(name = "NoLimits", version = "0.1")
Pkg.compat("NoLimits", "0.1")
```
