using Test
using NoLimits
using LinearAlgebra

@testset "Helpers macro returns functions" begin
    helpers = @helpers begin
        clamp01(u) = max(0.0, min(1.0, u))
        softplus(u) = log1p(exp(u))
        dotp(a, b) = dot(a, b)
    end

    @test helpers isa NamedTuple
    @test haskey(helpers, :clamp01)
    @test haskey(helpers, :softplus)
    @test haskey(helpers, :dotp)
    @test helpers.clamp01(-1.0) == 0.0
    @test helpers.clamp01(0.5) == 0.5
    @test helpers.clamp01(2.0) == 1.0
    @test isapprox(helpers.softplus(0.0), log1p(exp(0.0)); rtol=1e-6, atol=1e-8)
    @test helpers.dotp([1.0, 2.0], [3.0, 4.0]) == 11.0
end

@testset "Helpers edge cases" begin
    empty_helpers = @helpers begin
    end
    @test empty_helpers == NamedTuple()

    @test_throws LoadError @eval @helpers begin
        a = 1.0
    end

    @test_throws LoadError @eval @helpers begin
        dup(x) = x
        dup(x) = x + 1
    end

    helpers_typed = @helpers begin
        typed(x::Float64) = x + 1.0
    end
    @test helpers_typed.typed(1.0) == 2.0
end
