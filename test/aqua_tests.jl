using Test
using NoLimits
using Aqua

# Aqua.jl static quality assurance (https://juliatesting.github.io/Aqua.jl):
#   * unbound type parameters, undefined exports
#   * project hygiene: stale deps, missing [compat] entries, test-project extras
#   * type piracy (all Distributions/Base extensions dispatch on owned types)
#   * persistent tasks blocking precompilation
# All checks run with defaults and no ignore lists; keep it that way.
# The method-ambiguity scan lives in aqua_ambiguities_tests.jl: it takes several
# CI minutes on its own, so it runs as a separate shard.
@testset "Aqua quality assurance" begin
    # `persistent_tasks` is run separately below, with retries.
    Aqua.test_all(NoLimits; persistent_tasks = false, ambiguities = false)
end

# `Aqua.has_persistent_tasks` builds a wrapper package that loads NoLimits and signals
# completion by writing `done.log`, then runs `Pkg.precompile(; io = devnull)` on it in a
# subprocess. It reports a failure when that file never appears — which happens both when a
# task really does outlive loading AND when the subprocess simply dies (a fresh resolve of
# ~270 packages inside an already-loaded test process is memory-hungry, and Pkg's output is
# sent to devnull, so the reason is invisible). The two are indistinguishable from the
# outside, and the second is sporadic: it has failed once on CI and once on the cluster while
# passing repeatedly on the same commits everywhere else.
#
# A genuine persistent task is deterministic, so retrying separates the cases without
# weakening the check: a real one fails every attempt, a dead subprocess does not.
@testset "Aqua persistent tasks (retried)" begin
    attempts = 3
    ok = false
    for i in 1:attempts
        # Cap the wrapper's parallel precompile workers: at the default (one per
        # core) the subprocess OOMs next to the test process and dies without
        # writing done.log — deterministically on attempt 1, warm on attempt 2 —
        # so every CI run paid the check twice (confirmed from the CI log's
        # "done.log was not created, but precompilation exited").
        ok = withenv("JULIA_NUM_PRECOMPILE_TASKS" => "2") do
            !Aqua.has_persistent_tasks(Base.PkgId(NoLimits))
        end
        ok && break
        i < attempts &&
            @info "Aqua persistent-tasks check failed on attempt $i/$attempts; retrying (a dead precompile subprocess is reported identically to a real persistent task)."
    end
    @test ok
end
