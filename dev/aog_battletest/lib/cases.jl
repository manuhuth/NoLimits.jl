# Case matrix: every plotting function x applicable fixtures x option sets.
#
# A case is one call to one public plotting function. `call` is a Julia expression (as a
# String) evaluated in a fresh sandbox module in which these names are bound:
#
#   res   the fixture's fit (FitResult / MultistartFitResult), or `nothing`
#   dm    the fixture's DataModel
#   fx    NamedTuple of fixture extras (e.g. fx.uq_wald, fx.fit_focei)
#   OUT   a per-case scratch directory (use it for save_path/plot_path cases)
#
# plus `using NoLimits, CairoMakie, Statistics, Distributions, DataFrames, Random`.
# The value of the expression is what gets rendered (Figure, Vector{Figure}, or a panel,
# which is wrapped with `combine_plots([panel])`).
#
# Cases live in lib/cases/*.jl; each file defines a function `cases_<file>()` returning
# Vector{Case}. Add a new file per porting group to avoid merge conflicts.

using TOML

struct Case
    id::String               # unique + filesystem-safe; convention "<fn>__<fixture>__<optset>"
    fn::Symbol               # public plotting function under test
    fixture::String          # fixture name (fixtures/defs/<fixture>.jl)
    call::String             # Julia expression, see above
    tags::Vector{String}     # free-form (e.g. "default", "style", "error")
    expect_error::Bool       # true => the call must throw (validation cases)
    files::Vector{String}    # files (relative to OUT) the call must create (save_path cases)
end

Case(id, fn, fixture, call; tags = String[], expect_error = false, files = String[]) =
    Case(id, Symbol(fn), String(fixture), String(call), String.(tags), expect_error, String.(files))

const _BT_DATA = normpath(joinpath(@__DIR__, "..", "fixtures", "data"))

"""
    fixture_meta() -> Dict{String, Dict}

Metadata of all generated fixtures (`fixtures/data/<name>.toml`): `feeds`, `tags`,
`extras`, `n_individuals`, `method`, ...
"""
function fixture_meta()
    out = Dict{String, Dict{String, Any}}()
    isdir(_BT_DATA) || return out
    for f in readdir(_BT_DATA)
        endswith(f, ".toml") || continue
        # tolerate a metadata file that is being rewritten by a fixture job
        d = try
            TOML.parsefile(joinpath(_BT_DATA, f))
        catch err
            @warn "skipping unreadable fixture metadata $(f)" err
            continue
        end
        out[d["name"]] = d
    end
    return out
end

"""
    fixtures_for(fn; tags_any = nothing, tags_all = nothing, extras = nothing) -> Vector{String}

Generated fixtures whose `feeds` include `fn` (optionally filtered by fixture tags or by
required extras keys), sorted by name.
"""
function fixtures_for(fn::Symbol; tags_any = nothing, tags_all = nothing, extras = nothing)
    meta = fixture_meta()
    names = String[]
    for (name, d) in meta
        string(fn) in get(d, "feeds", String[]) || continue
        tags = get(d, "tags", String[])
        tags_any === nothing || any(t -> t in tags, string.(tags_any)) || continue
        tags_all === nothing || all(t -> t in tags, string.(tags_all)) || continue
        extras === nothing || all(e -> e in get(d, "extras", String[]), string.(extras)) || continue
        push!(names, name)
    end
    return sort(names)
end

has_tag(fixture::AbstractString, tag) = string(tag) in get(get(fixture_meta(), fixture, Dict()), "tags", String[])

"""
    cross(fn, fixtures, optsets; input = "res", kwargs...) -> Vector{Case}

Build cases `fn(<input>; <opts>)` for every fixture and every `name => opts` pair in
`optsets`. `opts` is the keyword-argument source text, e.g. `"ncols = 2"`; use `""` for
the defaults. An optset name ending in `!err` marks an expect_error case.
"""
function cross(fn::Symbol, fixtures, optsets; input = "res", tags = String[], files = String[])
    out = Case[]
    for fixture in fixtures, (optname, opts) in optsets
        err = endswith(optname, "!err")
        oname = replace(optname, "!err" => "")
        call = isempty(opts) ? "$(fn)($(input))" : "$(fn)($(input); $(opts))"
        push!(
            out, Case(
                "$(fn)__$(fixture)__$(oname)", fn, fixture, call;
                tags = vcat(tags, [oname]), expect_error = err, files = files
            )
        )
    end
    return out
end

for f in sort(readdir(joinpath(@__DIR__, "cases")))
    endswith(f, ".jl") && include(joinpath(@__DIR__, "cases", f))
end

"""
    all_cases() -> Vector{Case}

Every case from every `cases_<file>()` generator in lib/cases/, sorted by (fixture, id).
Duplicate ids are an error.
"""
function all_cases()
    cs = Case[]
    for f in sort(readdir(joinpath(@__DIR__, "cases")))
        endswith(f, ".jl") || continue
        gen = Symbol("cases_", splitext(f)[1])
        append!(cs, getfield(@__MODULE__, gen)())
    end
    ids = [c.id for c in cs]
    dup = [i for i in unique(ids) if count(==(i), ids) > 1]
    isempty(dup) || error("duplicate case ids: $(dup)")
    return sort(cs; by = c -> (c.fixture, c.id))
end
