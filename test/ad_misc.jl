using Test
using NoLimits
using DifferentiationInterface
using ForwardDiff
using ReverseDiff
using Zygote

@testset "Helpers mutation warnings" begin
    # Mutating helpers should emit warnings for Zygote compatibility.
    @test_logs (:warn,) @eval @helpers begin
        bump!(x) = (y = copy(x); push!(y, 1.0); y)
    end
end

@testset "Spline AD" begin
    # AD through spline coefficients (params).
    knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
    coeffs = [0.1, 0.2, 0.3, 0.4]
    x = 0.25

    f(v) = bspline_eval(x, v, knots, 2)

    val_fwd, grad_fwd = value_and_gradient(f, AutoForwardDiff(), coeffs)
    @test length(grad_fwd) == length(coeffs)

    val_rev, grad_rev = value_and_gradient(f, AutoReverseDiff(), coeffs)
    @test isapprox(val_rev, val_fwd; rtol=1e-6, atol=1e-8)
    @test isapprox(grad_rev, grad_fwd; rtol=1e-6, atol=1e-8)

    grad_zyg = Zygote.gradient(f, coeffs)[1]
    @test isapprox(grad_zyg, grad_fwd; rtol=1e-6, atol=1e-8)
end
