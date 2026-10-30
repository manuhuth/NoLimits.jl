# Porting NoLimits plotting to AlgebraOfGraphics (issue #383)

This directory is the battle-test harness for the port. Stage 1 (branch
`feat/383-aog-plotting`) built the AoG core layer, ported `plot_fits` and `plot_data` as
reference implementations, and rendered the full old-behaviour screenshot set. Each later
porting group works in its own worktree branched from this branch, ports a group of
functions, and verifies them with the same harness.

## 1. Ground rules

- Marvin login node: never run Julia there. Everything below is an `sbatch` job (≤ 8 CPUs,
  CPU only). Show job scripts in your report.
- Public API is frozen: same function names, same keyword arguments, same return types
  (`Makie.Figure`; `Vector{Figure}` where it exists today). `kwargs_subplot` maps to AoG
  `axis=`, `kwargs_layout` to `figure=`; user values always win.
- Format with Runic 1.9.0 (CI gate), not JuliaFormatter:
  `julia --project=/home/mhuth1_hpc/NoLimitsAutoresearch/tools/runic-env -e 'using Runic; exit(Runic.main(ARGS))' -- --inplace src test docs ext`
  (in a job; template `~/NoLimitsAutoresearch/debug/vemfix-oct28/runic_inplace.sbatch`).
- Fixture fits are test fixtures (tiny data, few iterations), never results.
- `predict` returns the observed value in a `y` column (`missing` without an observation;
  ported from `fix/383-facet-layout-predict-y`). Build tidy tables inside the extension
  (from `get_residuals`, `predict`, the plot cache, etc.).

## 2. Layout of the extension

| File | Content | Status |
|---|---|---|
| `ext/NoLimitsMakieExt.jl` | module, imports, includes; triggers on Makie **and** AlgebraOfGraphics | done |
| `ext/plotting/plotting.jl` | **AoG core** (below) + `combine_plots` + `_save_plot!` | done |
| `ext/plotting/legacy_panels.jl` | old closure-based `MakiePanel` layer | delete when all ported |
| `ext/plotting/plotting_fits.jl` | `plot_data`, `plot_fits` (reference ports) | done |
| `ext/plotting/plots.jl` | hidden states, emissions, fits_comparison, observed_profiles, dv_pred, dv_ipred, wres_pred, shrinkage | to port |
| `ext/plotting/plotting_multistart_em.jl` | multistart waterfall / FE variability, EM trajectories | to port |
| `ext/plotting/plotting_residuals.jl`, `plotting_random_effects.jl`, `plotting_vpc.jl`, `plotting_observation_distributions.jl`, `plotting_uq.jl` | the rest | to port |

The Makie-free core (PlotStyle, PlotCache, build_plot_cache, get_residuals, helpers such as
`_merge_limits`, `_pad_limits`, `_axis_label`, `_marginal_colors`) stays in `src/plotting/`.

## 3. The AoG core API (`ext/plotting/plotting.jl`)

`const AoG = AlgebraOfGraphics`; always write `AoG.data`, `AoG.mapping`, `AoG.visual`
(AoG exports names like `density`/`histogram` that clash with Makie).

Style / kwargs mapping (precedence: PlotStyle defaults < computed values < user kwargs):

- `_aog_axis(style, kwargs_subplot; computed...)` → NamedTuple for `axis=`
  (`default_axis_kwargs(style)` < `computed` (xlabel, ylabel, ...) < `kwargs_subplot`).
- `_aog_figure(style, n_panels, ncols, kwargs_layout)` → NamedTuple for `figure=` (size from
  `calculate_plot_size`, fonts, padding; `kwargs_layout` wins).
- `_aog_legend(style; kw...)` → the figure legend: bottom, horizontal, frameless, no title.
- `_scatter_visual(style; kw...)`, `_line_visual(style; kw...)` → `visual(Scatter/Lines)`
  with the style's marker size/alpha/stroke and line width.

Tidy tables:

- `_tidy(; extra...)` → `DataFrame(panel, x, y, series)` (+ extra typed columns).
- `_push_xy!(df, panel, xs, ys, series)` appends a series (`missing` → `NaN` → gap).
- `_layer(df, mapping, visual)` → layer or `nothing` for an empty table (AoG cannot build
  scales from an empty table); `_layers(l1, l2, ...)` sums the non-`nothing` ones.
- `_series_palette(["data" => c1, "fit" => c2])` → categorical `Color` scale settings with
  fixed colors and legend order. Map `color = :series` in every layer that should appear in
  the legend; AoG merges a Scatter and a Lines layer with the same label into one entry.

Faceted figures (one panel per individual / parameter / observable):

```julia
fig = _draw_facets(layers;                       # layers map layout = :panel
    panels = panel_labels,                       # display order, e.g. "ID: 3"
    ncols, style, shared_x, shared_y,            # validated ncols, link policy
    kwargs_subplot, kwargs_layout,
    axis = (; xlabel = "Time", ylabel = "Y"),    # computed axis attributes
    series = ["data" => style.color_primary, "fit" => style.color_secondary],
    scales = (;),                                # extra AoG scale settings
    xlim, ylim,                                  # explicit (shared) limits or nothing
    post = Dict(panel => [ax -> heatmap!(ax, ...)]),  # plain-Makie per-panel extras
    legend = true)
return _save_plot!(fig, save_path)
```

Policy it implements: panels wrap row-wise `ncols` wide (`AoG.wrapped(cols=...)` with the
panel labels as `Layout` categories, so panel `k` is at `fldmod1(k, ncols)`); shared axes →
`linkxaxes/linkyaxes = :all`, AoG hides inner tick labels and spans one x/y label;
unshared → `:none`; one figure legend at the bottom when `series` is non-empty; panel
titles are AoG facet labels (`Makie.Label`, **not** `ax.title`). Use `_facet_axis(fg, k,
ncols)` style indexing if you need an axis yourself.

Composable panels (`return_panel = true`):

```julia
panel = AoGPanel(layers;                          # NO layout mapping; `nothing` = post only
    scales = (; Color = _series_palette(...)),
    axis = _aog_axis(style, kwargs_subplot; title = "...", xlabel = "...", ylabel = "..."),
    post = Any[ax -> ablines!(ax, 0, 1; color = COLOR_REFERENCE, linestyle = :dash)])
return_panel && return panel
return _save_plot!(combine_plots([panel]; ncols = 1, style, kwargs_layout...), save_path)
```

`combine_plots(panels; ncols, style, kwargs...)` draws each `AoGPanel` with `AoG.draw!`
into `fig[r, c]`, runs its `post` hooks, and merges all panel legends (by label) into ONE
legend row under the grid. It still accepts legacy `MakiePanel`s during the transition.
For a single-axis function without facets, use exactly this path so it has the same
legend policy as faceted figures.

## 4. Conventions and pitfalls (learned in stage 1)

- **Density / continuous colour overlays**: an AoG continuous `Color` scale shares one
  colour range across the whole figure. `plot_fits` keeps per-panel normalisation by
  drawing densities as `post` hooks with plain `heatmap!`/`scatter!` (own colorrange per
  call) and `translate!(plot, 0, 0, -1)` so they sit behind the AoG layers. If you need an
  AoG heatmap instead, pre-normalise z per panel to [0, 1].
- **Histograms**: AoG `histogram(bins = n)` with an integer gives StatsBase "nice" edges,
  not n equal bins. Compute bins with the core helper `_histogram_xy` and draw a
  `visual(BarPlot; gap = 0)` layer (or pass explicit edges) to keep the old binning.
- **Empty panels are deleted** by AoG (`deleteemptyaxes!`). A panel with no rows in any
  layer disappears; this differs from the old layout, which kept an empty titled axis.
- **Shared limits**: linking alone gives the union of the data; the old code padded the
  union by 5 % (`_pad_limits`). Pass `xlim`/`ylim = _shared_lim(shared, lims)` to keep the
  old framing (and to keep overlays such as density grids from widening the limits).
- **Per-facet axis attributes** (different xlabel per panel, reference lines with panel
  specific values, per-panel annotations) → `post` hooks on the panel's axis, or separate
  scales. `axis=` applies to all facets.
- **Labels that collide across colours**: one categorical colour per series label. If the
  same label must appear in several colours (e.g. "5%" per component), make the label
  unique ("y[1] 5%").
- **Tests** that inspected `ax.title` or per-axis legends must change: facet titles are
  `Makie.Label`s in `fig.content`, and there is one `Makie.Legend` per figure.
- **Compile time**: the first AoG draw in a fresh process costs ~50 s of compilation
  (loading packages ~12 s); later draws take 1-7 s. Keep render jobs batched (the harness
  already loads each fixture once per chunk).
- `using AlgebraOfGraphics` alongside `using CairoMakie` is required for the extension to
  load; tests use `import AlgebraOfGraphics`.

## 5. The harness

```
dev/aog_battletest/
  envs/new        Project for this branch (NoLimits = ../../..)       [Manifest is git-ignored]
  envs/baseline   same deps, NoLimits = /home/mhuth1_hpc/NoLimits-aog-baseline (origin/main)
  fixtures/defs/<name>.jl   fixture definitions (contract in lib/common.jl)
  fixtures/data/            generated fits/extras/metadata (git-ignored, shared by all worktrees)
  lib/common.jl             package loading, generate_fixture/load_fixture
  lib/cases.jl              Case type, cross(), fixtures_for(); includes lib/cases/*.jl
  lib/cases/<group>.jl      case generators, one file per porting group
  lib/render.jl             renders a chunk of cases, writes status TOML + PNG
  lib/compare.jl            out/summary.csv, side-by-side images, contact sheets
  jobs/*.sbatch             precompile, gen_fixtures, render, compare, tests, smoke
  out/<mode>/png, out/<mode>/status, out/compare, out/sheets, out/summary.csv
```

### Setting up a porting worktree

```bash
cd /home/mhuth1_hpc/NoLimits && git worktree add -b feat/383-aog-<group> /home/mhuth1_hpc/NoLimits-aog-<group> feat/383-aog-plotting
cd /home/mhuth1_hpc/NoLimits-aog-<group>/dev/aog_battletest
cp /home/mhuth1_hpc/NoLimits-aog/dev/aog_battletest/envs/new/Manifest.toml envs/new/
cp /home/mhuth1_hpc/NoLimits-aog/dev/aog_battletest/envs/baseline/Manifest.toml envs/baseline/
ln -s /home/mhuth1_hpc/NoLimits-aog/dev/aog_battletest/fixtures/data fixtures/data
mkdir -p logs out && ln -s /home/mhuth1_hpc/NoLimits-aog/dev/aog_battletest/out/baseline out/baseline
sed -i "s#/home/mhuth1_hpc/NoLimits-aog/dev/aog_battletest#$PWD#g" jobs/*.sbatch   # job paths
MODES=new sbatch jobs/precompile.sbatch      # compiles this worktree's NoLimits once
```

The baseline screenshots (`out/baseline`) were rendered once from origin/main and are
shared read-only; do not re-render them unless you add cases (then render only the new
ids: `MODE=baseline FILTER='^plot_xyz__' ...`).

### Running

```bash
# render new cases of your group (chunks of ~15 cases keep each task < 10 min)
MODE=new NCHUNKS=8 FILTER='^plot_residual' sbatch --array=1-8 jobs/render.sbatch
# re-render after a fix (status files are skipped unless FORCE=1)
FORCE=1 MODE=new NCHUNKS=8 FILTER='^plot_residual' sbatch --array=1-8 jobs/render.sbatch
# summary CSV + side-by-side + contact sheets
FILTER='^plot_residual' sbatch jobs/compare.sbatch
grep -h 'status = ' out/new/status/plot_residual*.toml | sort | uniq -c
```

Statuses: `ok`, `ok_expected_error` (validation case threw as intended), `error`,
`fail_unexpected_success`, `fail_missing_file` (save_path case did not write the file).
Each status TOML also records timing, return type and figure introspection (axes, legends,
colorbars, labels, plots per axis), which `out/summary.csv` tabulates for both modes.

### Reviewing

Look at the images yourself (Read tool on PNGs): `out/compare/<case_id>.png` shows
baseline | new. Check every option set at least once: same data points, same curves,
facets/titles/labels/legend correct, no clipped or empty panels. Old-implementation bugs
found while reviewing are listed in the stage-1 report; do not reproduce them.

### Adding cases

Add generators to `lib/cases/<group>.jl` (function `cases_<group>()`), e.g.

```julia
function cases_mygroup()
    cs = Case[]
    append!(cs, cross(:plot_shrinkage, fixtures_for(:plot_shrinkage),
        ["default" => "", "thr" => "threshold = 0.1", "ncols0!err" => "ncols = 0"]))
    push!(cs, Case("plot_x__fixture__save", :plot_x, "fixture",
        "plot_x(res; save_path = joinpath(OUT, \"x.png\"))"; files = ["x.png"]))
    return cs
end
```

Call strings see `res`, `dm`, `fx` (fixture extras, e.g. `fx.uq_wald`, `fx.fit_focei`) and
`OUT`. Option names ending in `!err` are expected-error cases.

### Fixture notes

- 19 fixtures; `feeds` were verified against origin/main (`fixtures/check_feeds.jl`,
  results in `out/check_feeds_baseline/`).
- Reloaded fits carry `_SavedFittingMethod`: use `fx.res_em` (method restored) for
  `plot_em_trajectories` on `saem_diag` / `mcem_diag`.
- `fx.uq_profile` cannot be plotted by design (no draws); use it for error cases only.
- `vi_nore` uses natural-scale parameters because a reloaded VI fit returns
  linked-scale posterior draws (NoLimits bug, see stage-1 report).
- Old-implementation bugs found by the baseline render (do not reproduce): residual and
  random-effect functions break on any `kwargs_subplot` (splatted positionally into
  `create_styled_plot`); `plot_random_effects_pdf` accepts invalid `levels`;
  `plot_random_effect_pairplot` does not validate `ncols`; HMM functions raise a raw
  `FieldError` for an unknown observable; emission-distribution panel titles overlap;
  `plot_fits(...; plot_func = median)` on vector observables raised a BoundsError (fixed in
  the port); `:vector` marginal figures framed y on all components (fixed in the port).

### Adding fixtures

Write `fixtures/defs/<name>.jl` following the contract in `lib/common.jl` (INFO with
`feeds`/`tags`/optional `x_feature`, `observables`, `re_names`, `covariates`, `params`;
`build_dm`, `build_fit`, optional `build_extras` / `extras_at_load`), then
`FIXTURES=<name> sbatch jobs/gen_fixtures.sbatch`. `fixtures_for(:fn)` picks it up from the
metadata TOML automatically.
