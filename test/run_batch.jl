# Batch runner: executed as a fresh `julia` subprocess by runtests.jl, one per
# batch of test files (passed as ARGS). Running each batch in its own process is
# what keeps memory bounded: every distinct `@Model` in the suite generates
# type-specialized native code that Julia never frees within a process, so a
# single process running all ~97 files accumulates enough compiled code to
# exhaust RAM (→ swap → stall). Exiting between batches releases all of it.
#
# Fixtures are included fresh here (lazy/memoized per process), so each batch
# builds only the canonical models its own files actually touch.
using Test
using NoLimits

include(joinpath(@__DIR__, "fixtures.jl"))

# A failing @test inside an inner @testset is recorded (not thrown); the
# outermost @testset throws a TestSetException at the end if anything failed,
# which makes this subprocess exit non-zero so the orchestrator sees it.
# verbose: print per-file times in the summary (shard balancing depends on them)
@testset "batch" verbose = true begin
    for f in ARGS
        @testset "$f" begin
            include(joinpath(@__DIR__, f))
        end
    end
end
