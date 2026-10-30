# Usage: julia --project=envs/<mode> lib/gen_fixture.jl <fixture_name> [more names...]
# Fixtures are generated ONCE (with envs/new; core NoLimits code is identical on both
# branches) and loaded by both baseline and new render jobs.
include(joinpath(@__DIR__, "common.jl"))
status = 0
for name in ARGS
    println("=== fixture ", name); flush(stdout)
    try
        t = @elapsed out = generate_fixture(name)
        println("OK ", name, " in ", round(t; digits = 1), " s; meta = ", out.meta)
        # round-trip check: the saved fit must reload against a freshly built DataModel
        t2 = @elapsed lf = load_fixture(name)
        println("reload OK in ", round(t2; digits = 1), " s; res type = ", typeof(lf.res), "; extras = ", keys(lf.fx))
    catch err
        global status = 1
        println("FAILED ", name)
        showerror(stdout, err, catch_backtrace())
        println()
    end
    flush(stdout)
end
exit(status)
