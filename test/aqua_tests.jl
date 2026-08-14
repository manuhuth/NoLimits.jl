using Test
using NoLimits
using Aqua

# Aqua.jl static quality assurance (https://juliatesting.github.io/Aqua.jl):
#   * unbound type parameters, undefined exports
#   * project hygiene: stale deps, missing [compat] entries, test-project extras
#   * type piracy (all Distributions/Base extensions dispatch on owned types)
# All checks run with defaults and no ignore lists; keep it that way.
# Two checks run in other shards purely for CI wall-clock, at full strength:
# the method-ambiguity scan (aqua_ambiguities_tests.jl) and the retried
# persistent-tasks check (aqua_persistent_tasks_tests.jl).
@testset "Aqua quality assurance" begin
    Aqua.test_all(NoLimits; persistent_tasks = false, ambiguities = false)
end
