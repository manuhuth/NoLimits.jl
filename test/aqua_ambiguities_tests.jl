using Test
using NoLimits
using Aqua

# Split out of aqua_tests.jl purely for CI wall-clock: the ambiguity scan is the
# most expensive Aqua check by far. Same defaults, no ignore lists — method
# ambiguities are kept at ZERO (see the disambiguation blocks in
# src/distributions/outcomes/*ObservedStatesMarkov*.jl).
@testset "Aqua method ambiguities" begin
    Aqua.test_ambiguities(NoLimits)
end
