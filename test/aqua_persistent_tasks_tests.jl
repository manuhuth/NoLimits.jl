using Test
using NoLimits
using Aqua

# Split out of aqua_tests.jl for CI shard balance: the wrapper-package
# precompile below reliably dies on attempt 1 on CI runners and passes warm on
# attempt 2, so the check costs ~4 min and deserves its own lane instead of
# serializing behind the rest of Aqua.
#
# `Aqua.has_persistent_tasks` builds a wrapper package that loads NoLimits and signals
# completion by writing `done.log`, then runs `Pkg.precompile(; io = devnull)` on it in a
# subprocess. It reports a failure when that file never appears — which happens both when a
# task really does outlive loading AND when the subprocess simply dies (a fresh resolve of
# ~270 packages inside an already-loaded test process is memory-hungry, and Pkg's output is
# sent to devnull, so the reason is invisible). The two are indistinguishable from the
# outside; on CI the dead-subprocess mode is deterministic on attempt 1 (confirmed from the
# CI log's "done.log was not created, but precompilation exited") and warm on attempt 2.
#
# A genuine persistent task is deterministic across attempts, so retrying separates the
# cases without weakening the check: a real one fails every attempt.
@testset "Aqua persistent tasks (retried)" begin
    attempts = 3
    ok = false
    for i in 1:attempts
        # Cap the wrapper's parallel precompile workers to bound the subprocess
        # memory spike next to the test process.
        ok = withenv("JULIA_NUM_PRECOMPILE_TASKS" => "2") do
            !Aqua.has_persistent_tasks(Base.PkgId(NoLimits))
        end
        ok && break
        i < attempts &&
            @info "Aqua persistent-tasks check failed on attempt $i/$attempts; retrying (a dead precompile subprocess is reported identically to a real persistent task)."
    end
    @test ok
end
