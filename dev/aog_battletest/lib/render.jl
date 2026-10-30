# Render a chunk of the case matrix to PNG and record per-case status.
#
# Usage (inside a SLURM job, see jobs/render.sbatch):
#   julia --project=envs/<mode> lib/render.jl <mode> <chunk> <nchunks> [id_regex]
#
# Cases are sorted by (fixture, id) and split into `nchunks` contiguous chunks, so each
# job loads only a few fixtures. Output:
#   out/<mode>/png/<case_id>.png            (Vector{Figure}: <case_id>__k.png, k = 1, 2, ...)
#   out/<mode>/status/<case_id>.toml        status, timing, figure introspection
# Existing status files are skipped unless FORCE=1 (re-runs only redo missing cases).

const MODE = ARGS[1]
const CHUNK = parse(Int, ARGS[2])
const NCHUNKS = parse(Int, ARGS[3])
const IDRE = length(ARGS) >= 4 && !isempty(ARGS[4]) ? Regex(ARGS[4]) : nothing

t_load = @elapsed include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "cases.jl"))
println("packages + harness loaded in ", round(t_load; digits = 1), " s; NoLimits at ", pkgdir(NoLimits))
println("NoLimitsMakieExt loaded: ", Base.get_extension(NoLimits, :NoLimitsMakieExt) !== nothing)

# `BT_OUT` renders into another output root (default out/).
const OUTDIR = joinpath(abspath(get(ENV, "BT_OUT", joinpath(BT_ROOT, "out"))), MODE)
mkpath(joinpath(OUTDIR, "png")); mkpath(joinpath(OUTDIR, "status")); mkpath(joinpath(OUTDIR, "scratch"))
const PX_PER_UNIT = parse(Float64, get(ENV, "PX_PER_UNIT", "1.5"))
const FORCE = get(ENV, "FORCE", "0") == "1"

cases = all_cases()
IDRE === nothing || (cases = filter(c -> occursin(IDRE, c.id), cases))
n = length(cases)
lo = div((CHUNK - 1) * n, NCHUNKS) + 1
hi = div(CHUNK * n, NCHUNKS)
mine = cases[lo:hi]
println("mode=$(MODE) chunk $(CHUNK)/$(NCHUNKS): cases $(lo):$(hi) of $(n)")

_figures(x::Makie.Figure) = [x]
_figures(x::AbstractVector) = all(f -> f isa Makie.Figure, x) && !isempty(x) ? collect(x) : nothing
_figures(x) = nothing

# Light-weight introspection used by compare/summarize (data-level parity checks).
function _introspect(fig::Makie.Figure)
    axes = [c for c in fig.content if c isa Makie.Axis]
    legends = count(c -> c isa Makie.Legend, fig.content)
    cbars = count(c -> c isa Makie.Colorbar, fig.content)
    labels = count(c -> c isa Makie.Label, fig.content)
    nplots = [length(ax.scene.plots) for ax in axes]
    titles = [string(Makie.to_value(ax.title)) for ax in axes]
    return Dict(
        "size" => collect(Int.(fig.scene.viewport[].widths)),
        "n_axes" => length(axes), "n_legends" => legends, "n_colorbars" => cbars,
        "n_labels" => labels, "plots_per_axis" => nplots,
        "axis_titles" => titles,
    )
end

function _short(err)
    io = IOBuffer()
    showerror(io, err)
    s = String(take!(io))
    return length(s) > 1500 ? first(s, 1500) * " …" : s
end

fixtures_cache = Dict{String, Any}()
function get_fixture(name)
    return get!(fixtures_cache, name) do
        t = @elapsed fxt = load_fixture(name)
        println("  loaded fixture $(name) in $(round(t; digits = 1)) s")
        fxt
    end
end

n_ok = 0; n_fail = 0
for c in mine
    statusfile = joinpath(OUTDIR, "status", c.id * ".toml")
    (!FORCE && isfile(statusfile)) && continue
    rec = Dict{String, Any}(
        "id" => c.id, "fn" => string(c.fn), "fixture" => c.fixture, "mode" => MODE,
        "call" => c.call, "tags" => c.tags, "expect_error" => c.expect_error,
    )
    scratch = joinpath(OUTDIR, "scratch", c.id)
    rm(scratch; force = true, recursive = true); mkpath(scratch)
    for f in readdir(joinpath(OUTDIR, "png"))
        (startswith(f, c.id * ".") || startswith(f, c.id * "__")) && rm(joinpath(OUTDIR, "png", f))
    end
    try
        fxt = get_fixture(c.fixture)
        sb = Module(Symbol("Case_", hash(c.id)))
        Core.eval(sb, :(using NoLimits, CairoMakie, Statistics, Distributions, DataFrames, Random))
        Core.eval(sb, :(import AlgebraOfGraphics))
        Core.eval(sb, :(const res = $(fxt.res)))
        Core.eval(sb, :(const dm = $(fxt.dm)))
        Core.eval(sb, :(const fx = $(fxt.fx)))
        Core.eval(sb, :(const OUT = $(scratch)))
        expr = Meta.parse(c.call)
        local val
        t = @elapsed (val = Base.invokelatest(Core.eval, sb, expr))
        rec["seconds"] = t
        rec["return_type"] = string(typeof(val))
        figs = _figures(val)
        kind = figs === nothing ? "panel" : (val isa AbstractVector ? "vector" : "figure")
        if figs === nothing
            # return_panel = true: wrap the panel to render it
            figs = [Base.invokelatest(combine_plots, [val])]
        end
        rec["kind"] = kind
        rec["n_figures"] = length(figs)
        intro = Dict{String, Any}()
        for (k, fig) in enumerate(figs)
            png = length(figs) == 1 && kind != "vector" ? "$(c.id).png" : "$(c.id)__$(k).png"
            Makie.save(joinpath(OUTDIR, "png", png), fig; px_per_unit = PX_PER_UNIT)
            intro[string(k)] = _introspect(fig)
        end
        rec["figures"] = intro
        missing_files = [f for f in c.files if !isfile(joinpath(scratch, f))]
        if !isempty(missing_files)
            rec["status"] = "fail_missing_file"
            rec["error"] = "expected files not created: $(missing_files)"
        else
            rec["status"] = c.expect_error ? "fail_unexpected_success" : "ok"
        end
    catch err
        rec["error"] = _short(err)
        rec["error_type"] = string(typeof(err))
        rec["status"] = c.expect_error ? "ok_expected_error" : "error"
    end
    ok = startswith(rec["status"], "ok")
    ok ? (global n_ok += 1) : (global n_fail += 1)
    println(rpad(rec["status"], 24), " ", rpad(string(round(get(rec, "seconds", NaN); digits = 2)), 8), " ", c.id)
    ok || println("    ", first(split(get(rec, "error", ""), '\n')))
    flush(stdout)
    open(statusfile, "w") do io
        TOML.print(io, rec)
    end
end
println("done: ok=$(n_ok) fail=$(n_fail)")
