using Test
using NoLimits
using LinearAlgebra
using ForwardDiff

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
    @test isapprox(helpers.softplus(0.0), log1p(exp(0.0)); rtol = 1e-6, atol = 1e-8)
    @test helpers.dotp([1.0, 2.0], [3.0, 4.0]) == 11.0
end

@testset "Helpers edge cases" begin
    empty_helpers = @helpers begin end
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

@testset "rowsoftmax" begin
    L = [1.0 2.0 0.5;
         0.0 -1.0 3.0;
         2.0 2.0 2.0]
    P = rowsoftmax(L)

    # every row is a probability distribution
    @test all(isapprox.(sum(P; dims = 2), 1.0; atol = 1e-12))
    @test all(0.0 .< P .< 1.0)

    # matches the explicit exponentiate-and-normalize construction, row by row
    for i in 1:3
        e = exp.(L[i, :])
        @test isapprox(P[i, :], e ./ sum(e); atol = 1e-12)
    end

    # equal logits within a row give a uniform row
    @test isapprox(P[3, :], fill(1 / 3, 3); atol = 1e-12)

    # numerically stable for large logits (no overflow to NaN/Inf)
    Lbig = [1000.0 1001.0 999.0; -1000.0 -1000.0 -999.0; 0.0 500.0 -500.0]
    Pbig = rowsoftmax(Lbig)
    @test all(isfinite, Pbig)
    @test all(isapprox.(sum(Pbig; dims = 2), 1.0; atol = 1e-12))

    # shift-invariance: adding a per-row constant leaves the result unchanged
    @test isapprox(rowsoftmax(L .+ [10.0, -3.0, 7.0]), P; atol = 1e-12)

    # works for non-square (general m-by-n) logit matrices
    R = rowsoftmax([1.0 2.0 3.0 4.0; 0.0 0.0 0.0 0.0])
    @test size(R) == (2, 4)
    @test all(isapprox.(sum(R; dims = 2), 1.0; atol = 1e-12))
    @test isapprox(R[2, :], fill(0.25, 4); atol = 1e-12)

    # automatic-differentiation safe (ForwardDiff through rowsoftmax is finite)
    J = ForwardDiff.jacobian(v -> vec(rowsoftmax(reshape(v, 3, 3))), vec(L))
    @test all(isfinite, J)
end

@testset "Helpers mutation warnings" begin
    # Mutating helpers should emit warnings (reverse-mode AD compatibility).
    @test_logs (:warn,) @eval @helpers begin
        bump!(x) = (y = copy(x); push!(y, 1.0); y)
    end
end
