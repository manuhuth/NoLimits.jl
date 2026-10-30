# Execute the ```@example blocks of documentation pages in order, one module per
# (page, example name), like Documenter does; reports pass/fail per block.
# Usage: julia --project=<docs env> lib/run_doc_examples.jl page1.md [page2.md ...]
using Test
blocks_failed = 0
for page in ARGS
    txt = read(page, String)
    mods = Dict{String, Module}()
    for (k, m) in enumerate(eachmatch(r"```@example\s*([^\n]*)\n(.*?)\n```"s, txt))
        name = strip(m.captures[1])
        name = isempty(name) ? "anon$(k)" : name
        mod = get!(mods, name) do
            Module(Symbol("Ex_", replace(name, r"\W" => "_")))
        end
        Core.eval(mod, :(include(p) = Base.include($mod, p)))
        Core.eval(mod, :(eval(x) = Core.eval($mod, x)))
        t = time()
        try
            v = include_string(mod, m.captures[2], "$(basename(page))#$(k)")
            # display-like materialization of returned figures
            if v !== nothing && parentmodule(typeof(v)) !== Core && isdefined(Main, :CairoMakie)
                try
                    io = IOBuffer(); show(io, MIME"image/png"(), v)
                catch
                end
            end
            println("OK   ", basename(page), " block ", k, " [", name, "] ", round(time() - t; digits = 1), " s")
        catch err
            global blocks_failed += 1
            println("FAIL ", basename(page), " block ", k, " [", name, "]")
            showerror(stdout, err); println()
        end
        flush(stdout)
    end
end
println("failed blocks: ", blocks_failed)
exit(blocks_failed == 0 ? 0 : 1)
