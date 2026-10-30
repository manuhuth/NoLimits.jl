# Shared harness code: environment loading + fixture build/save/load.
#
# Included by gen_fixture.jl and render.jl. Runs under either envs/new (this branch) or
# envs/baseline (origin/main worktree); both resolve to identical dependency versions.
#
# A fixture is a file fixtures/defs/<name>.jl evaluated inside a fresh module that already
# has `using NoLimits, DataFrames, Distributions, Random, Statistics, LinearAlgebra` and
# `import Turing`. It must define:
#
#   const INFO = (; description = "...", feeds = [:plot_fits, ...], tags = [...])
#   build_dm()           -> DataModel          (cheap, deterministic; called at load time)
#   build_fit(dm)        -> FitResult | MultistartFitResult | nothing
#   build_extras(res, dm) -> NamedTuple        (optional; e.g. (; uq_wald = compute_uq(...)))
#
# Fits are saved with `save_fit(...; include_data = true)` (JLD2) and reloaded with
# `load_fit(path; dm = build_dm())`. Extras are saved with `Serialization`; a fixture may
# instead define `extras_at_load(res, dm) -> NamedTuple` for objects that are cheap to
# recompute but awkward to serialize.
#
# FIXTURE FITS ARE TEST FIXTURES, NOT RESULTS: tiny data, few iterations.

using NoLimits
using NoLimits: FitResult, MultistartFitResult
using DataFrames, Distributions, Random, Statistics, LinearAlgebra
using JLD2, Serialization, TOML, Printf
import Turing
using CairoMakie
import AlgebraOfGraphics

const BT_ROOT = normpath(joinpath(@__DIR__, ".."))
const FIX_DEFS = joinpath(BT_ROOT, "fixtures", "defs")
# `BT_FIXTURE_DATA` points generation/loading at another fixture directory (e.g. a frozen
# set, or a baseline-generated copy for fits origin/main cannot load).
const FIX_DATA = abspath(get(ENV, "BT_FIXTURE_DATA", joinpath(BT_ROOT, "fixtures", "data")))

fixture_names() = sort([splitext(f)[1] for f in readdir(FIX_DEFS) if endswith(f, ".jl")])

function _fixture_module(name::AbstractString)
    m = Module(Symbol("Fixture_", name))
    # `Module(...)` (unlike the `module` keyword) defines no `eval`/`include`; @Model needs
    # `eval` (RuntimeGeneratedFunctions init), so add both as the `module` keyword would.
    Core.eval(m, :(eval(x) = Core.eval($m, x)))
    Core.eval(m, :(include(p) = Base.include($m, p)))
    Core.eval(
        m, quote
            using NoLimits
            using DataFrames, Distributions, Random, Statistics, LinearAlgebra
            import Turing
        end
    )
    Base.include(m, joinpath(FIX_DEFS, name * ".jl"))
    return m
end

_has(m::Module, f::Symbol) = isdefined(m, f)

"""
    generate_fixture(name) -> NamedTuple

Build the fixture's DataModel, fit it, compute extras, and write
`data/<name>.jld2`, `data/<name>_extras.jls`, `data/<name>.toml` (metadata/timing).
"""
function generate_fixture(name::AbstractString)
    mkpath(FIX_DATA)
    m = _fixture_module(name)
    t_dm = @elapsed dm = Base.invokelatest(m.build_dm)
    t_fit = @elapsed res = Base.invokelatest(m.build_fit, dm)
    fitpath = joinpath(FIX_DATA, name * ".jld2")
    if res !== nothing
        save_fit(fitpath, res; include_data = true)
    else
        isfile(fitpath) && rm(fitpath)
    end
    t_ex = @elapsed extras = _has(m, :build_extras) ?
        Base.invokelatest(m.build_extras, res, dm) : NamedTuple()
    # Extra fits (e.g. a second method for plot_fits_comparison) go through save_fit; the
    # rest is serialized. Extra fits must share the fixture's DataModel.
    fit_keys = [k for k in keys(extras) if getfield(extras, k) isa Union{FitResult, MultistartFitResult}]
    for k in fit_keys
        save_fit(joinpath(FIX_DATA, string(name, "__", k, ".jld2")), getfield(extras, k); include_data = true)
    end
    plain = NamedTuple(k => getfield(extras, k) for k in keys(extras) if !(k in fit_keys))
    serialize(joinpath(FIX_DATA, name * "_extras.jls"), plain)
    info = m.INFO
    meta = Dict(
        "name" => name,
        "description" => string(info.description),
        "feeds" => string.(collect(info.feeds)),
        "tags" => string.(collect(get(info, :tags, String[]))),
        "fit_type" => string(typeof(res)),
        "method" => res === nothing ? "none" : string(nameof(typeof(get_method(res)))),
        "extras" => string.(collect(keys(extras))),
        "extra_fits" => string.(fit_keys),
        "n_individuals" => length(get_individuals(dm)),
        "time_build_dm" => t_dm, "time_fit" => t_fit, "time_extras" => t_ex,
        "nolimits_path" => pkgdir(NoLimits),
        "generated" => string(Base.Libc.strftime(time())),
        "note" => "TEST FIXTURE ONLY (tiny data / few iterations); not a production result",
    )
    # any further INFO fields (e.g. x_feature = :wt, observables = [:y1, :y2]) are kept
    # as strings / string vectors for case generators
    for k in keys(info)
        k in (:description, :feeds, :tags) && continue
        v = getfield(info, k)
        meta[string(k)] = v isa Union{AbstractVector, Tuple} ? string.(collect(v)) : string(v)
    end
    open(joinpath(FIX_DATA, name * ".toml"), "w") do io
        TOML.print(io, meta)
    end
    return (; dm, res, extras, meta)
end

"""
    load_fixture(name) -> NamedTuple (name, dm, res, fx, meta, mod)

Rebuild the DataModel, reload the saved fit against it, and deserialize the extras.
`fx` holds the extras NamedTuple (merged with `extras_at_load` when defined).
"""
function load_fixture(name::AbstractString)
    m = _fixture_module(name)
    dm = Base.invokelatest(m.build_dm)
    fitpath = joinpath(FIX_DATA, name * ".jld2")
    res = isfile(fitpath) ? load_fit(fitpath; dm = dm) : nothing
    expath = joinpath(FIX_DATA, name * "_extras.jls")
    fx = isfile(expath) ? deserialize(expath) : NamedTuple()
    metapath0 = joinpath(FIX_DATA, name * ".toml")
    if isfile(metapath0)
        for k in get(TOML.parsefile(metapath0), "extra_fits", String[])
            p = joinpath(FIX_DATA, string(name, "__", k, ".jld2"))
            fx = merge(fx, NamedTuple{(Symbol(k),)}((load_fit(p; dm = dm),)))
        end
    end
    if _has(m, :extras_at_load)
        fx = merge(fx, Base.invokelatest(m.extras_at_load, res, dm))
    end
    metapath = joinpath(FIX_DATA, name * ".toml")
    meta = isfile(metapath) ? TOML.parsefile(metapath) : Dict{String, Any}()
    return (; name, dm, res, fx, meta, mod = m)
end
