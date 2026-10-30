# Summarize statuses of both modes and build review images.
#
# Usage: julia --project=envs/new lib/compare.jl [id_regex] [chunk nchunks]
# Writes
#   out/summary.csv                        one row per case id (both modes)
#   out/compare/<case_id>.png              baseline (left) | new (right), with header
#   out/sheets/<fn>__<mode>__<k>.png       contact sheets: up to 12 thumbnails per sheet
# Only cases that have at least one PNG get images. The CSV is always rebuilt in full
# (chunk 1 only) so it can be read on the login node.

using CairoMakie, TOML, Printf
const BT = normpath(joinpath(@__DIR__, ".."))
const OUT = abspath(get(ENV, "BT_OUT", joinpath(BT, "out")))
const IDRE = length(ARGS) >= 1 && !isempty(ARGS[1]) ? Regex(ARGS[1]) : r""
const CHUNK = length(ARGS) >= 3 ? parse(Int, ARGS[2]) : 1
const NCHUNKS = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1
const MODES = ("baseline", "new")

function statuses(mode)
    d = Dict{String, Dict{String, Any}}()
    dir = joinpath(OUT, mode, "status")
    isdir(dir) || return d
    for f in readdir(dir)
        endswith(f, ".toml") || continue
        r = TOML.parsefile(joinpath(dir, f))
        d[r["id"]] = r
    end
    return d
end

pngs(mode, id) = begin
    dir = joinpath(OUT, mode, "png")
    isdir(dir) || return String[]
    fs = filter(f -> f == id * ".png" || (startswith(f, id * "__") && endswith(f, ".png") && all(isdigit, f[(length(id) + 3):(end - 4)])), readdir(dir))
    joinpath.(dir, sort(fs))
end

st = Dict(m => statuses(m) for m in MODES)
ids = sort(collect(union(keys(st["baseline"]), keys(st["new"]))))
ids = filter(i -> occursin(IDRE, i), ids)

csvesc(s) = "\"" * replace(string(s), "\"" => "\"\"", "\n" => " ") * "\""
if CHUNK == 1
    open(joinpath(OUT, "summary.csv"), "w") do io
        println(io, "id,fn,fixture,expect_error,baseline_status,new_status,baseline_s,new_s,baseline_axes,new_axes,baseline_legends,new_legends,baseline_error,new_error")
        for id in ids
            b = get(st["baseline"], id, Dict{String, Any}())
            n = get(st["new"], id, Dict{String, Any}())
            r = isempty(n) ? b : n
            ax(x) = haskey(x, "figures") ? join([string(v["n_axes"]) for (_, v) in sort(collect(x["figures"]); by = first)], "+") : ""
            lg(x) = haskey(x, "figures") ? join([string(v["n_legends"]) for (_, v) in sort(collect(x["figures"]); by = first)], "+") : ""
            println(
                io, join(
                    [
                        id, get(r, "fn", ""), get(r, "fixture", ""), get(r, "expect_error", ""),
                        get(b, "status", "missing"), get(n, "status", "missing"),
                        @sprintf("%.2f", get(b, "seconds", NaN)), @sprintf("%.2f", get(n, "seconds", NaN)),
                        ax(b), ax(n), lg(b), lg(n),
                        csvesc(first(split(get(b, "error", ""), '\n'))),
                        csvesc(first(split(get(n, "error", ""), '\n'))),
                    ], ","
                )
            )
        end
    end
    println("wrote ", joinpath(OUT, "summary.csv"), " (", length(ids), " ids)")
end

loadimg(p) = rotr90(CairoMakie.FileIO.load(p))

function show_img!(pos, path; title = "")
    ax = Axis(pos; aspect = DataAspect(), title = title, titlesize = 11)
    hidedecorations!(ax); hidespines!(ax)
    if path === nothing
        text!(ax, 0.5, 0.5; text = "(no image)", align = (:center, :center), space = :relative)
    else
        image!(ax, loadimg(path); interpolate = true)
    end
    return ax
end

mine = ids[(div((CHUNK - 1) * length(ids), NCHUNKS) + 1):div(CHUNK * length(ids), NCHUNKS)]
mkpath(joinpath(OUT, "compare"))
for id in mine
    pb = pngs("baseline", id); pn = pngs("new", id)
    (isempty(pb) && isempty(pn)) && continue
    k = max(length(pb), length(pn))
    sb = get(get(st["baseline"], id, Dict()), "status", "missing")
    sn = get(get(st["new"], id, Dict()), "status", "missing")
    fig = Figure(size = (1600, 560 * k))
    Label(fig[0, 1:2], "$(id)    baseline: $(sb)    new: $(sn)"; fontsize = 14, tellwidth = false)
    for j in 1:k
        show_img!(fig[j, 1], j <= length(pb) ? pb[j] : nothing; title = "baseline")
        show_img!(fig[j, 2], j <= length(pn) ? pn[j] : nothing; title = "new")
    end
    save(joinpath(OUT, "compare", id * ".png"), fig)
end

# contact sheets per function and mode (chunk 1 only, they are cheap)
if CHUNK == 1
    mkpath(joinpath(OUT, "sheets"))
    for mode in MODES
        byfn = Dict{String, Vector{String}}()
        for id in ids
            fn = first(split(id, "__"))
            for p in pngs(mode, id)
                push!(get!(byfn, fn, String[]), p)
            end
        end
        for (fn, ps) in byfn
            for (s, part) in enumerate(Iterators.partition(ps, 12))
                fig = Figure(size = (1800, 1500))
                for (i, p) in enumerate(part)
                    r, c = fldmod1(i, 4)
                    show_img!(fig[r, c], p; title = replace(basename(p), ".png" => ""))
                end
                save(joinpath(OUT, "sheets", "$(fn)__$(mode)__$(s).png"), fig)
            end
        end
    end
end
println("compare done: ", length(mine), " ids in this chunk")
